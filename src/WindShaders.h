#pragma once

// Compute shaders for the shared wind + fog field (PLAN.md 16.6, REWRITE-PLAN.md). One 2D grid around the
// player: velocity (game units / s), fog amount, pressure, divergence, walls.
// Velocity: Stam "Stable Fluids" (1999) as laid out in GPU Gems 1 ch. 38 (Harris 2004), pressure solved from a
// zero guess every frame as there; walls handled in advection, divergence, Jacobi and gradient as in GPU Gems 3
// ch. 30 (Crane et al.).
// Fog: one scalar, the fog amount rho (1 = the undisturbed layer), carried by the air like Stam's smoke
// density and diluted by the continuity equation, D(rho)/Dt = -rho div(u), with div(u) the divergence source
// the projection is given. It is stored as its departure from the equilibrium pattern rho_eq (EqFog), see CSAdvectFog.
// Impacts and moving bodies are divergence sources shaped as a Ricker function (the Laplacian of a Gaussian), whose
// integral over the plane is zero; fire also removes fog (Evaporation).
// Per frame: shift (when the grid moves) -> walls -> curl -> forces -> advect velocity -> divergence ->
// pressure (Jacobi) -> subtract gradient -> advect fog. Projection is last, so fog and every reader use the
// same projected velocity.

namespace WindShaders
{
	inline constexpr int   kMaxDrags = 32;
	inline constexpr int   kMaxBlasts = 12;
	inline constexpr int   kGroup = 8;
	// Upper bound of the fog amount (numerical safety against pile-up at inflow rings). The renderer's largest
	// optical thickness is derived from the same number (FogRenderer.cpp wMax). Must match kMaxFog in the shader.
	inline constexpr float kMaxFogCpp = 2.0f;
	// Column height response (CSColumn, LookP.z = SurfaceResponse; judge's plan r11 step 1a, design A r07): the low
	// layer's column height is baseH x s, s = (rho / rho_eq)^SurfaceResponse held to these bounds; W keeps its formula.
	// Must match kSurfaceMin / kSurfaceMax in the shader.
	inline constexpr float kSurfaceMinCpp = 0.15f;
	inline constexpr float kSurfaceMaxCpp = 1.6f;
	// Highest the low layer's top stands over the ground (units): Thickness, x kSurfaceMaxCpp while the height response
	// is on. FogRenderer's march box top and RaymarchCheck's crest check use this one expression.
	// Banks (ini [Render] BankAmount, 0 = off): the layer thicker and thinner over thousands of units, after roxorvic's
	// Community Shaders fog (skyrim-community-shaders fluid-field/fog-v3, GroundFogCS.hlsl Banks: 8000-unit noise, 0.45 to
	// 1.3 x; GPL-3.0, used with permission). kBankHigh bounds the march box.
	inline constexpr float kBankLowCpp = 0.45f;
	inline constexpr float kBankHighCpp = 1.3f;
	inline float LowLayerTop(float a_thickness, float a_surfaceResponse, float a_bankAmount = 0.0f)
	{
		const float bank = a_bankAmount > 0.0f ? 1.0f + a_bankAmount * (kBankHighCpp - 1.0f) : 1.0f;
		return a_thickness * (a_surfaceResponse > 0.0f ? kSurfaceMaxCpp : 1.0f) * bank;
	}

	// Must match cbuffer SimCB.
	struct alignas(16) SimCB
	{
		float grid[4];       // x, y = world position of texel (0,0) corner, z = cell size, w = resolution
		float time[4];       // x = dt, y = field time (s, stops while paused; the same clock as CSColumn's Iter.y), z, w = base wind travel (world units)
		float wind[4];       // xy = base wind (units/s), z = relax toward it (1/s), w = vorticity
		float gust[4];       // x = gust strength (units/s), y = gust wavelength (units), z = fog refill toward equilibrium (1/s), w = velocity dissipation (1/s)
		float shift[4];      // xy = texel shift (new = old[x + sx, y + sy]), z = the frame's field time (s, time.x x steps; WakeDose), w = height grid size
		float counts[4];     // x = drags, y = blasts, z = wall rise threshold (units per unit), w = wisp amount of the equilibrium pattern (0 = uniform)
		float heightGrid[4]; // x, y = height grid origin, z = height cell, w = 1 if heights are bound
		float drags[kMaxDrags][4];     // xy = position, zw = velocity
		float dragSize[kMaxDrags][4];  // x = 1 / radius^2 (the body's own radius), y = push per unit of speed (1/unit, 0 = off), z = distance the body really moved this frame (units, WakeDose), w = length of its straight run so far, this frame included (units)
		float dragParams[4];           // x = share of the push a walker already gets (0..1), y = follow rate (1/s, at the body's centre), z = WakeEvaporate / steps this frame (0 = no wake), w = speed where the running push starts (units/s)
		float blasts[kMaxBlasts][4];   // xy = position, z = 1 / radius^2, w = divergence source rate now (1/s)
		float blastKicks[kMaxBlasts][4]; // xy = acceleration along the hit direction now (units/s^2), z = evaporation rate now (1/s, a fire hit), w unused
	};
	// The same rows in the same order as the shader's cbuffer SimCB below: 1536 bytes.
	static_assert(sizeof(SimCB) == 16 * (7 + 2 * kMaxDrags + 1 + 2 * kMaxBlasts));

	inline constexpr const char* kSource = R"(
#define MAX_DRAGS 32
#define MAX_BLASTS 12

static const float kMaxFog = 2.0;   // WindShaders::kMaxFogCpp
// Column height response (CSColumn): s = q^LookP.z held to kSurfaceMin..kSurfaceMax (WindShaders::kSurfaceMinCpp,
// kSurfaceMaxCpp), q = rho / rho_eq where rho_eq is over kSurfaceEqMin, faded out from kSurfaceFadeStart to
// kSurfaceFadeEnd units from the field centre (past 1500 a march step at a level ray is wider than a lane: design A r07).
static const float kSurfaceMin = 0.15;   // = kShapeFloor
static const float kSurfaceMax = 1.6;    // a runner's banks 1.12, an explosion's rim 1.63 (WindRules.h)
static const float kSurfaceEqMin = 0.05;
static const float kSurfaceFadeStart = 1500.0;
static const float kSurfaceFadeEnd = 2500.0;

cbuffer SimCB : register(b0)
{
	float4 Grid;
	float4 Time;
	float4 Wind;
	float4 Gust;
	float4 Shift;
	float4 Counts;
	float4 HeightGrid;
	float4 Drags[MAX_DRAGS];
	float4 DragSize[MAX_DRAGS];
	float4 DragParams;
	float4 Blasts[MAX_BLASTS];
	float4 BlastKicks[MAX_BLASTS];
};

cbuffer PlaceCB : register(b2)
{
	float4 PlaceP;   // x = radiation-fog multiplier (FogConditions global), y = steam cold (0..1), z = wind factor, w = 1 if InPlace is valid
	float4 SteamP;   // x = evaporation (subtracted from the radiation multiplier), y = steam column height per unit of steam
	                 // (units; 0 = steam only thickens the layer, as before), z = steam density (W per metre of steam column),
	                 // w = pool density (W per metre of pooled fog; PoolDensity x radiation strength, 0 = no pools)
	float4 PlaceG;   // the placement texture's own grid: origin x, y, cell, size (cell 0 = the height grid's); it may lag
	                 // the height grid by a few frames while a newer build runs on the worker thread
	float4 LookP;    // x = wisp amount (0 = none): fine structure in the low layer's W; y = region value added to the
	                 // placement multiplier (RegionRules: the player's weather region); z = column height response
	                 // (SurfaceResponse, 0 = off: the height from the shape alone); w = BankAmount (0 = off)
	float4 CoarseG;  // coarse pool texture (InCoarse) grid: origin x, y, cell, size (cell 0 = none)
	float4 CoarseP;  // x = ground tolerance (units): coarse pools fade where the fine ground leaves the coarse one by this
};

cbuffer IterCB : register(b1)
{
	float4 Iter;   // CSShift: fill value (w = 1 copies the edge); CSColumn: x = base thickness, y = time, zw = wind travel
};

SamplerState Linear : register(s0);

Texture2D<float2> InVel      : register(t0);
Texture2D<float2> InFog      : register(t1);   // x = fog amount rho, y = delta = rho - rho_eq (CSAdvectFog)
Texture2D<float>  InPressure : register(t2);
Texture2D<float>  InDiv      : register(t3);   // right-hand side of the pressure equation: divergence - source
Texture2D<float>  InWall     : register(t4);
Texture2D<float>  InCurl     : register(t5);
Texture2D<float>  InHeight   : register(t6);
Texture2D<float4> InAny      : register(t7);   // shift source (any format)
Texture2D<float2> InCoarse   : register(t9);   // coarse pools: ground z, pool depth (units) per coarse cell
Texture2D<float4> InPlace    : register(t8);   // per height cell: x radiation multiplier, y hot-spring steam, z open-water steam (PlacementRules)

RWTexture2D<float2> OutVel   : register(u0);
RWTexture2D<float2> OutFog   : register(u1);
RWTexture2D<float>  OutScalar: register(u2);
RWTexture2D<float>  OutDiv   : register(u3);
RWTexture2D<float4> OutAny   : register(u4);
RWTexture2D<float2> OutUpper : register(u5);   // CSColumn: the tall layer (steam and pooled fog), H x known and W x known
RWTexture2D<float>  OutRelief: register(u6);   // CSColumn: the height factor's departure s - 1, x known (relief shading)

static const float kKnownLimit = -1.0e29;

uint Res() { return (uint)Grid.w; }
float2 ToUv(float2 texel) { return (texel + 0.5) / Grid.w; }
float2 ToWorld(float2 texel) { return Grid.xy + (texel + 0.5) * Grid.z; }

bool Inside(int2 p) { return all(p >= 0) && all(p < (int)Res()); }
float WallAt(int2 p) { return Inside(p) ? InWall[p] : 0.0; }

uint Hash(uint2 p)
{
	uint h = p.x * 0x8da6b343u ^ p.y * 0xd8163841u;
	h ^= h >> 13; h *= 0x2c1b3c6du; h ^= h >> 12;
	return h;
}

float ValueNoise(float2 p)
{
	const float2 i = floor(p);
	const float2 f = p - i;
	const float2 u = f * f * (3.0 - 2.0 * f);
	const uint2 c = uint2(int2(i));
	const float a = Hash(c) / 4294967295.0;
	const float b = Hash(c + uint2(1, 0)) / 4294967295.0;
	const float d = Hash(c + uint2(0, 1)) / 4294967295.0;
	const float e = Hash(c + uint2(1, 1)) / 4294967295.0;
	return lerp(lerp(a, b, u.x), lerp(d, e, u.x), u.y);
}

// Gust stream function: noise that travels with the wind. Its curl is divergence-free, so the pressure
// projection does not erase the gusts (Codex review 2026-11-06, finding 6).
float GustPsi(float2 world)
{
	const float2 p = (world - Time.zw) / max(Gust.y, 1.0);
	// two octaves, each half the amplitude of the one before (fBm gain 0.5): weights 2/3 and 1/3
	return (2.0 * ValueNoise(p) + ValueNoise(p * 2.1 + 17.0)) / 3.0 * Gust.x * Gust.y * 0.25;
}

// Equilibrium fog amount rho_eq: 1 on average, with banks and gaps inside the layer (photos 2026-11-08: not one
// smooth sheet). Two octaves of value noise, 6 m and 2 m at 70 units per metre (about three layer heights and one: banks wider than the layer is tall, with gaps a person fills), weights 2/3 and 1/3
// (fBm gain 0.5), riding the base wind (travel) and changing slowly in time. amount 0 = uniform.
float EqFog(float2 world, float2 travel, float time, float amount)
{
	const float2 q = world - travel;
	const float  n = (2.0 * ValueNoise(q / 420.0 + time * float2(0.003, 0.001)) + ValueNoise(q / 140.0 + time * float2(-0.004, 0.006))) / 3.0;
	return max(1.0 + amount * (2.0 * n - 1.0), 0.0);
}

// Height of the low layer relative to Thickness at q = world xy - base wind travel: the layer's top is not flat,
// it rises into mounds and sinks into hollows about 13 m across. Two octaves of value noise (fBm, lacunarity 2,
// gain 0.5: weights 2/3, 1/3), each drifting slowly its own way; finer structure is the fog amount's (EqFog). The noise sits around 0.5 (measured: mean
// 0.50, standard deviation 0.16); kShapeContrast stretches it about kShapeLevel and kShapeFloor is the lowest the top
// gets, so the layer never opens to the ground by its shape alone (holes come from the fog amount). With these values
// the mean height is 0.41 x Thickness (numpy, 2026-11-10): 51 units at Thickness 126, the mean the layer had before. RaymarchShaders.h FarLayerH is the same function: keep them
// identical, or the far tier will not continue the near layer.
static const float kShapeFloor = 0.15;
static const float kShapeContrast = 1.8;
static const float kShapeLevel = 0.62;   // noise value that maps to mid-height: most of the layer is low, with mounds

// Banks (LookP.w = BankAmount, 0 = off): one octave of value noise 8000 units across (much wider than the shape's
// 900), drifting slowly, scaling the layer between kBankLow and kBankHigh x, blended in by the amount. RaymarchShaders.h
// FarBank is the same function: keep them identical (the far tier continues the near layer).
static const float kBankSpan = 8000.0;
static const float kBankLow = 0.45;    // WindShaders::kBankLowCpp
static const float kBankHigh = 1.3;    // WindShaders::kBankHighCpp

float BankFactor(float2 q, float time, float amount)
{
	if (!(amount > 0.0)) return 1.0;
	const float n = ValueNoise(q / kBankSpan + time * float2(0.0006, 0.0004));
	return 1.0 + amount * (lerp(kBankLow, kBankHigh, n) - 1.0);
}

float LayerShape(float2 q, float time)
{
	const float2 p = q / 900.0;
	const float  n = (2.0 * ValueNoise(p + time * float2(0.004, 0.0015)) + ValueNoise(p * 2.0 + 19.7 + time * float2(-0.005, 0.003))) / 3.0;
	return lerp(kShapeFloor, 1.0, saturate((n - kShapeLevel) * kShapeContrast + 0.5));
}

// Ricker function (Laplacian of a Gaussian, Marr & Hildreth 1980) of d with q = |d|^2 / R^2: positive inside
// r < R, negative outside. Over the plane, int (1 - q) e^-q 2 pi r dr = pi R^2 - pi R^2 = 0, so a source of this
// shape pushes air from the centre to its rim without adding or removing any. Along a straight line through the
// centre, int (1 - x^2/R^2) e^(-x^2/R^2) dx = sqrt(pi) R - sqrt(pi) R / 2 = sqrt(pi) R / 2.
float Ricker(float2 d, float invR2)
{
	const float q = dot(d, d) * invR2;
	return (1.0 - q) * exp(-q);
}

// ---- shift: new(x, y) = old(x + sx, y + sy); exposed texels get the fill value in OutAny's slot.
// Velocity and fog are both two-channel textures seen through these float4 views: the store keeps only the
// channels the format has, so the fill value's x, y land and its z, w are ignored (w only picks the mode).
[numthreads(8, 8, 1)]
void CSShift(uint3 id : SV_DispatchThreadID)
{
	if (any(id.xy >= Res())) return;
	const int2 src = int2(id.xy) + int2(Shift.xy);
	if (Iter.w > 0.5) {
		// velocity: copy the nearest edge value so new strips do not start with a jump (reviewer finding 2)
		OutAny[id.xy] = InAny[clamp(src, 0, (int)Res() - 1)];
		return;
	}
	OutAny[id.xy] = Inside(src) ? InAny[src] : Iter;   // Iter carries the fill value for this texture
}

// ---- walls from the height grid: a texel is a wall where the ground rises faster than the threshold
[numthreads(8, 8, 1)]
void CSWall(uint3 id : SV_DispatchThreadID)
{
	if (any(id.xy >= Res())) return;
	float wall = 0.0;
	if (HeightGrid.w > 0.5) {
		const float2 world = ToWorld(float2(id.xy));
		const float  size = Shift.w;
		const float2 g = (world - HeightGrid.xy) / HeightGrid.z;
		const int2   c = clamp(int2(floor(g)), 0, (int)size - 2);
		const float h00 = InHeight[c];
		const float h10 = InHeight[c + int2(1, 0)];
		const float h01 = InHeight[c + int2(0, 1)];
		const float h11 = InHeight[c + int2(1, 1)];
		if (h00 > kKnownLimit && h10 > kKnownLimit && h01 > kKnownLimit && h11 > kKnownLimit) {
			const float gx = 0.5 * ((h10 - h00) + (h11 - h01)) / HeightGrid.z;
			const float gy = 0.5 * ((h01 - h00) + (h11 - h10)) / HeightGrid.z;
			wall = smoothstep(Counts.z, Counts.z * 1.5, length(float2(gx, gy)));
		}
	}
	OutScalar[id.xy] = wall > 0.5 ? 1.0 : 0.0;
}

// ---- curl (scalar vorticity) of the velocity
[numthreads(8, 8, 1)]
void CSCurl(uint3 id : SV_DispatchThreadID)
{
	if (any(id.xy >= Res())) return;
	const int2 p = int2(id.xy);
	const int  n = (int)Res() - 1;
	const float l = InVel[int2(max(p.x - 1, 0), p.y)].y;
	const float r = InVel[int2(min(p.x + 1, n), p.y)].y;
	const float b = InVel[int2(p.x, max(p.y - 1, 0))].x;
	const float t = InVel[int2(p.x, min(p.y + 1, n))].x;
	OutScalar[p] = ((r - l) - (t - b)) / (2.0 * Grid.z);
}

// ---- forces: relax toward base wind + gusts, vorticity confinement, actor drag, blast push
[numthreads(8, 8, 1)]
void CSForce(uint3 id : SV_DispatchThreadID)
{
	if (any(id.xy >= Res())) return;
	const int2  p = int2(id.xy);
	const float dt = Time.x;
	if (InWall[p] > 0.5) { OutVel[p] = 0.0; return; }

	float2 v = InVel[p];
	const float2 world = ToWorld(float2(p));

	// Base wind plus divergence-free gusts: curl of the stream function, by central differences.
	const float e = Grid.z;
	const float2 gustV = float2(GustPsi(world + float2(0, e)) - GustPsi(world - float2(0, e)),
	                            -(GustPsi(world + float2(e, 0)) - GustPsi(world - float2(e, 0)))) / (2.0 * e);
	const float2 target = Wind.xy + gustV;
	v += (target - v) * saturate(Wind.z * dt);

	// Vorticity confinement keeps small swirls alive after numerical smoothing.
	const int n = (int)Res() - 1;
	const float cl = abs(InCurl[int2(max(p.x - 1, 0), p.y)]);
	const float cr = abs(InCurl[int2(min(p.x + 1, n), p.y)]);
	const float cb = abs(InCurl[int2(p.x, max(p.y - 1, 0))]);
	const float ct = abs(InCurl[int2(p.x, min(p.y + 1, n))]);
	float2 grad = float2(cr - cl, ct - cb);
	grad /= length(grad) + 1e-5;
	v += Wind.w * InCurl[p] * float2(grad.y, -grad.x) * Grid.z * dt;

	// Actors pull the air along with them: the air at a body takes the body's speed at DragParams.y per second.
	// 1 - exp(-rate dt) is that pull over a step of any length (rate x dt, as it was, pulled 6 % harder at 30
	// frames a second than at 60 and 21 % harder at 15, and the wake differed with the frame rate).
	for (int i = 0; i < (int)Counts.x; ++i) {
		const float2 d = world - Drags[i].xy;
		const float  w = exp(-dot(d, d) * DragSize[i].x);
		v = lerp(v, Drags[i].zw, 1.0 - exp(-w * DragParams.y * dt));
	}

	// Impacts carry momentum: an acceleration along the hit direction, spread as a Gaussian at the impact
	// (the force splat of GPU Gems 1 ch. 38). The radial push is the divergence source (Source below).
	for (int k = 0; k < (int)Counts.y; ++k) {
		const float2 d = world - Blasts[k].xy;
		v += BlastKicks[k].xy * exp(-dot(d, d) * Blasts[k].z) * dt;
	}

	OutVel[p] = v;
}

// Semi-Lagrangian back-trace that stops at the first wall on the way (Codex finding 1):
// walk the path in steps of half a cell and keep the last free point.
float2 TraceBack(float2 texel, float2 vel)
{
	float2 path = vel * Time.x / Grid.z;                // in texels
	float  len = length(path);
	if (len > 16.0) {                                   // faster than 16 texels per step: shorten the trace
		path *= 16.0 / len;                             // so the wall checks stay half a texel apart
		len = 16.0;
	}
	const int steps = clamp((int)ceil(len * 2.0), 1, 32);
	float2 pos = texel;
	for (int i = 1; i <= steps; ++i) {
		const float2 next = texel - path * (i / (float)steps);
		if (WallAt(int2(round(next))) > 0.5) break;
		pos = next;
	}
	return pos;
}

// Bilinear sample of the fog field that leaves wall texels out and renormalises (plan 16.6: no wall values in interpolation).
float2 SampleFogNoWall(float2 texel, float2 fallback)
{
	const float2 b = floor(texel);
	const float2 f = texel - b;
	float2 sum = 0.0;
	float  wsum = 0.0;
	[unroll] for (int k = 0; k < 4; ++k) {
		const int2  c = int2(b) + int2(k & 1, k >> 1);
		const int2  cc = clamp(c, 0, (int)Res() - 1);
		const float w = ((k & 1) ? f.x : 1.0 - f.x) * ((k >> 1) ? f.y : 1.0 - f.y);
		if (InWall[cc] < 0.5) {
			sum += InFog[cc] * w;
			wsum += w;
		}
	}
	return wsum > 1e-4 ? sum / wsum : fallback;
}

float2 SampleVelNoWall(float2 texel, float2 fallback)
{
	const float2 b = floor(texel);
	const float2 f = texel - b;
	float2 sum = 0.0;
	float  wsum = 0.0;
	[unroll] for (int k = 0; k < 4; ++k) {
		const int2  cc = clamp(int2(b) + int2(k & 1, k >> 1), 0, (int)Res() - 1);
		const float w = ((k & 1) ? f.x : 1.0 - f.x) * ((k >> 1) ? f.y : 1.0 - f.y);
		if (InWall[cc] < 0.5) {
			sum += InVel[cc] * w;
			wsum += w;
		}
	}
	return wsum > 1e-4 ? sum / wsum : fallback;
}

// ---- advect velocity by itself, with dt-independent dissipation v / (1 + k dt)
[numthreads(8, 8, 1)]
void CSAdvectVel(uint3 id : SV_DispatchThreadID)
{
	if (any(id.xy >= Res())) return;
	const int2 p = int2(id.xy);
	if (InWall[p] > 0.5) { OutVel[p] = 0.0; return; }
	const float2 from = TraceBack(float2(p), InVel[p]);
	const float2 v = SampleVelNoWall(from, InVel[p]);
	// Dissipate only the departure from the base wind, so steady wind reaches its full speed (reviewer).
	OutVel[p] = Wind.xy + (v - Wind.xy) / (1.0 + Gust.w * Time.x);
}

)" R"(
// Divergence source (1/s): where it is positive the air spreads out and the fog thins, where it is negative
// the air converges and the fog piles up.
//  - Impacts: a Ricker of the impact's radius at the rate the CPU gives for this frame (a pulse in time).
//  - Moving actors: a Ricker of the body's radius R, rate = z x speed. The fog amount lost on a straight path
//    through the centre is then z x speed x (sqrt(pi) R / 2) / speed = z sqrt(pi) R / 2, whatever the speed;
//    the CPU sets z per body (DragSize.y) so that this is the ActorPush setting (more for a wide body,
//    WindRules::SetDrag). The push comes in two parts: the share DragParams.x fades in from 0.18 to 0.36 x
//    DragParams.w (a walker's trace; about 40 to 79 at 220), the rest from DragParams.w to 1.5 x it (running), so a
//    walker never gets more than that share and nobody more than the whole.
float Source(float2 world)
{
	float s = 0.0;
	for (int k = 0; k < (int)Counts.y; ++k) {
		s += Blasts[k].w * Ricker(world - Blasts[k].xy, Blasts[k].z);
	}
	const float run = max(DragParams.w, 1.0);
	for (int i = 0; i < (int)Counts.x; ++i) {
		const float speed = length(Drags[i].zw);
		const float gate = DragParams.x * smoothstep(0.18 * run, 0.36 * run, speed) + (1.0 - DragParams.x) * smoothstep(run, 1.5 * run, speed);
		s += DragSize[i].y * speed * gate * Ricker(world - Drags[i].xy, DragSize[i].x);
	}
	return s;
}

// Fog a fire boils off (1/s): a Gaussian on the impact's footprint, at the rate the CPU gives for this frame. It is
// a sink of the fog amount only (CSAdvectFog); the air is not asked to move for it, so nothing piles up on a rim.
float Evaporation(float2 world)
{
	float s = 0.0;
	for (int k = 0; k < (int)Counts.y; ++k) {
		const float2 d = world - Blasts[k].xy;
		s += BlastKicks[k].z * exp(-dot(d, d) * Blasts[k].z);
	}
	return s;
}

// Wake evaporation (ini [Wind] WakeEvaporate; DEBUG-LEDGER.md 2026-11-14 15:xx): the turbulent wake behind a moving body
// mixes the drier air over the layer down into it and droplets evaporate. Not a rate: each frame a body gives the fog
// behind it the dose of the path it moved through the air there, so a parcel of air on its path gets e x gate in all (the
// centre line; a Gaussian of the body's radius across), whatever the frame rate or the clamped field time, and nothing
// once the body is gone; air carried along with the body collects no more for it (DEBUG-LEDGER.md 2026-11-14 15:xx (2)).
//   dose = gate x exp(-c^2 / R^2) x (S(b) - S(b - L_rel)) for b <= run, S(b) = smoothstep(0.5 R, 2 R, b),
//   L_rel = max(0, L - (u_air . heading) x Shift.z),
// b = distance behind the body along its heading, c = across it, L = DragSize.z (the distance it moved this frame, real
// time), u_air = a_vel (this step's air velocity here), Shift.z = the frame's field time, run = DragSize.w (its straight
// run so far, over the ground; nothing behind where that run began). A parcel's b grows by L_rel a frame, so its doses add
// up to S(b now) - S(b at the start) <= 1 (WindRules::WakeDose is the same on the CPU).
// CSAdvectFog thins rho by exp(-DragParams.z x WakeDose) every step, DragParams.z = WakeEvaporate / steps, but the wake
// alone never takes it under kWakeFloor x rho_eq (paths crossed again and again; the push has no such floor).
static const float kWakeFloor = 0.15;   // WindRules::kWakeFloor
float WakeDose(float2 world, float2 a_vel)
{
	float dose = 0.0;
	const float run = max(DragParams.w, 1.0);
	for (int i = 0; i < (int)Counts.x; ++i) {
		const float step = DragSize[i].z;
		const float speed = length(Drags[i].zw);
		if (!(step > 0.0) || speed < 1.0) continue;
		const float2 d = world - Drags[i].xy;
		const float  b = -dot(d, Drags[i].zw / speed);
		if (b > DragSize[i].w) continue;
		const float R = rsqrt(DragSize[i].x);
		const float gate = DragParams.x * smoothstep(0.18 * run, 0.36 * run, speed) + (1.0 - DragParams.x) * smoothstep(run, 1.5 * run, speed);
		const float c2 = max(dot(d, d) - b * b, 0.0);
		const float rel = max(step - dot(a_vel, Drags[i].zw / speed) * Shift.z, 0.0);
		dose += gate * exp(-c2 * DragSize[i].x) * (smoothstep(0.5 * R, 2.0 * R, b) - smoothstep(0.5 * R, 2.0 * R, b - rel));
	}
	return dose;
}

// Central-difference divergence of InVel at p; a wall neighbour contributes zero velocity.
float DivergenceAt(int2 p)
{
	const int  n = (int)Res() - 1;
	const int2 pl = int2(max(p.x - 1, 0), p.y), pr = int2(min(p.x + 1, n), p.y);
	const int2 pb = int2(p.x, max(p.y - 1, 0)), pt = int2(p.x, min(p.y + 1, n));
	const float l = InWall[pl] > 0.5 ? 0.0 : InVel[pl].x;
	const float r = InWall[pr] > 0.5 ? 0.0 : InVel[pr].x;
	const float b = InWall[pb] > 0.5 ? 0.0 : InVel[pb].y;
	const float t = InWall[pt] > 0.5 ? 0.0 : InVel[pt].y;
	return ((r - l) + (t - b)) / (2.0 * Grid.z);
}

// ---- right-hand side of the pressure equation: the divergence the projection has to remove, i.e. all of it
// except the wanted source (the projected velocity then has divergence = source, up to solver error)
[numthreads(8, 8, 1)]
void CSDivergence(uint3 id : SV_DispatchThreadID)
{
	if (any(id.xy >= Res())) return;
	const int2 p = int2(id.xy);
	const float src = InWall[p] > 0.5 ? 0.0 : Source(ToWorld(float2(p)));
	OutDiv[p] = DivergenceAt(p) - src;
}

// ---- one Jacobi iteration; a wall neighbour takes the centre pressure (no flow through the wall)
[numthreads(8, 8, 1)]
void CSPressure(uint3 id : SV_DispatchThreadID)
{
	if (any(id.xy >= Res())) return;
	const int2 p = int2(id.xy);
	const int  n = (int)Res() - 1;
	const float pc = InPressure[p];
	const int2 pl = int2(max(p.x - 1, 0), p.y), pr = int2(min(p.x + 1, n), p.y);
	const int2 pb = int2(p.x, max(p.y - 1, 0)), pt = int2(p.x, min(p.y + 1, n));
	const float l = InWall[pl] > 0.5 ? pc : InPressure[pl];
	const float r = InWall[pr] > 0.5 ? pc : InPressure[pr];
	const float b = InWall[pb] > 0.5 ? pc : InPressure[pb];
	const float t = InWall[pt] > 0.5 ? pc : InPressure[pt];
	OutScalar[p] = (l + r + b + t - InDiv[p] * Grid.z * Grid.z) * 0.25;
}

// ---- subtract the pressure gradient; walls get zero velocity
[numthreads(8, 8, 1)]
void CSSubtract(uint3 id : SV_DispatchThreadID)
{
	if (any(id.xy >= Res())) return;
	const int2 p = int2(id.xy);
	if (InWall[p] > 0.5) { OutVel[p] = 0.0; return; }
	const int  n = (int)Res() - 1;
	const float pc = InPressure[p];
	const int2 pl = int2(max(p.x - 1, 0), p.y), pr = int2(min(p.x + 1, n), p.y);
	const int2 pb = int2(p.x, max(p.y - 1, 0)), pt = int2(p.x, min(p.y + 1, n));
	const float l = InWall[pl] > 0.5 ? pc : InPressure[pl];
	const float r = InWall[pr] > 0.5 ? pc : InPressure[pr];
	const float b = InWall[pb] > 0.5 ? pc : InPressure[pb];
	const float t = InWall[pt] > 0.5 ? pc : InPressure[pt];
	OutVel[p] = InVel[p] - float2(r - l, t - b) / (2.0 * Grid.z);
}

// ---- carry the fog amount rho with the projected velocity u (InVel).
// rho obeys the continuity equation D(rho)/Dt = -rho div(u). It is stored as delta = rho - rho_eq, where
// rho_eq (EqFog) is a fixed pattern riding the base wind U: d(rho_eq)/dt + U . grad(rho_eq) = 0 (its slow change
// in time is left to the refill). Subtracting the two,
//     D(delta)/Dt = -(u - U) . grad(rho_eq) - rho div(u),
// so air moving with the base wind changes nothing (the pattern stays sharp: only delta goes through the
// semi-Lagrangian resampling), and air that departs from it drags the pattern along.
// div(u) is the prescribed source (Source): the projection is asked for exactly that divergence. Measuring it
// on the projected velocity instead was tried and dropped (offline checks 2026-11-10): what 20 Jacobi sweeps
// leave undone then shows up as fog, e.g. a walker's drag opened a 25 % hole and a cliff face emptied the fog
// in front of it. A fire's evaporation (Evaporation) is a second sink, with no air movement asked for it.
// The refill pulls rho back to rho_eq (implicit step, stable for any dt).
// Output: x = rho (what CSColumn reads), y = delta (the carried state).
//
// [Wind] FogSampling (WindSim::Create puts "#define FOG_SAMPLING 1" in front of this source; 0 or absent = bilinear,
// the code as it was). The bilinear resampling mixes in the neighbours across the flow whenever the flow is not along
// a grid axis, so a runner's trail, carried along itself by the air the runner drags, filled in on diagonal headings
// (mean q on the trail 0.35 along an axis, 0.66 at 45 degrees; DEBUG-LEDGER.md 2026-11-15). 1 reads delta with a
// Catmull-Rom 4 x 4 instead and holds the result to the range of the 4 nearest texels, so no new high or low appears.
// Only the thin side is sharpened: where the result is above the pattern (delta > 0: the banks beside a trail, the
// crests of gusts) it is not let above max(the bilinear value, 0), because the sharper read left the banks higher too (s 1.36
// against 1.19 beside a runner; ledger 2026-11-15). Any wall texel among the 16 -> the bilinear no-wall sample, as
// before. And only in air a body drags: the sharper value is blended in by how far the air's velocity is from the
// base wind, from kSharpFrom to kSharpFull units/s (the air behind a runner moves at 120 to 190). With the default
// gust settings gusts alone stay under kSharpFrom, so fog that only the wind and its gusts move is read exactly as
// before (measured at wind 57, GustMinimum 40: identical over 30 s); a large GustAmount with a strong wind, or the
// seconds after the wind changes, can pass it, and the sharper read then takes part there too. Limits: not conservative
// (neither is the bilinear one), and capping one side only loses a little fog on a trail (the offline tool compares
// the grid mean); 16 reads in place of 4 where the air is dragged.
#ifndef FOG_SAMPLING
#define FOG_SAMPLING 0
#endif
#if FOG_SAMPLING == 1
static const float kSharpFrom = 60.0;
static const float kSharpFull = 110.0;

float4 CatmullRom(float t)
{
	const float t2 = t * t, t3 = t2 * t;
	return float4(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}

float SampleDeltaCubic(float2 texel, float fallback, float2 vel)
{
	const float sharp = smoothstep(kSharpFrom, kSharpFull, length(vel - Wind.xy));
	if (sharp <= 0.0) return SampleFogNoWall(texel, float2(0.0, fallback)).y;
	const float2 b = floor(texel);
	const float2 f = texel - b;
	const float4 wx = CatmullRom(f.x), wy = CatmullRom(f.y);
	float sum = 0.0, bil = 0.0;
	float lo = 1.0e30, hi = -1.0e30;
	bool  wall = false;
	[unroll] for (int j = 0; j < 4; ++j) {
		[unroll] for (int i = 0; i < 4; ++i) {
			const int2 cc = clamp(int2(b) + int2(i - 1, j - 1), 0, (int)Res() - 1);
			if (InWall[cc] > 0.5) wall = true;
			const float d = InFog[cc].y;
			sum += d * wx[i] * wy[j];
			if ((i == 1 || i == 2) && (j == 1 || j == 2)) {   // the 4 texels the bilinear sample reads
				lo = min(lo, d);
				hi = max(hi, d);
				bil += d * (i == 1 ? 1.0 - f.x : f.x) * (j == 1 ? 1.0 - f.y : f.y);
			}
		}
	}
	if (wall) return SampleFogNoWall(texel, float2(0.0, fallback)).y;
	float v = clamp(sum, lo, hi);
	if (v > 0.0) v = min(v, max(bil, 0.0));
	return lerp(bil, v, sharp);
}
#endif

[numthreads(8, 8, 1)]
void CSAdvectFog(uint3 id : SV_DispatchThreadID)
{
	if (any(id.xy >= Res())) return;
	const int2  p = int2(id.xy);
	const float dt = Time.x;
	if (InWall[p] > 0.5) { OutFog[p] = float2(1, 0); return; }
	const float2 vel = InVel[p];
	const float2 from = TraceBack(float2(p), vel);
#if FOG_SAMPLING == 1
	float delta = SampleDeltaCubic(from, InFog[p].y, vel);
#else
	float delta = SampleFogNoWall(from, InFog[p]).y;
#endif

	const float2 world = ToWorld(float2(p));
	const float  eq = EqFog(world, Time.zw, Time.y, Counts.w);
	if (Counts.w > 0.0) {
		const float  e = Grid.z;
		const float2 grad = float2(EqFog(world + float2(e, 0), Time.zw, Time.y, Counts.w) - EqFog(world - float2(e, 0), Time.zw, Time.y, Counts.w),
		                           EqFog(world + float2(0, e), Time.zw, Time.y, Counts.w) - EqFog(world - float2(0, e), Time.zw, Time.y, Counts.w)) / (2.0 * e);
		delta -= dot(vel - Wind.xy, grad) * dt;
	}
	const float rho = clamp((eq + delta) * exp(-(Source(world) + Evaporation(world)) * dt), 0.0, kMaxFog);
	float rhoAfter = rho;
	if (DragParams.z > 0.0) {   // wake evaporation (WakeDose), held at the floor; off: rho as it was
		const float wakeFloor = kWakeFloor * eq;
		if (rho > wakeFloor) rhoAfter = max(rho * exp(-DragParams.z * WakeDose(world, vel)), wakeFloor);
	}
	delta = (rhoAfter - eq) / (1.0 + Gust.z * dt);
	OutFog[p] = float2(eq + delta, delta);
}

)" R"(
// ---- fog columns for the raymarch (yomotsu ground-fog 2 idea: a 2D field inflated in height).
// Per texel: x = column height H (units above ground), y = optical thickness W (metres of fog at unit
// density, seen from straight above), z = ground height (world z), w = 1 where the ground is known.
// Iter: x = base thickness (units), y = time (s), z, w = base wind travel (units).
[numthreads(8, 8, 1)]
void CSColumn(uint3 id : SV_DispatchThreadID)
{
	if (any(id.xy >= Res())) return;
	const int2   p = int2(id.xy);
	const float2 world = ToWorld(float2(p));
	float ground = 0.0;
	float known = 0.0;
	if (HeightGrid.w > 0.5) {
		const float  size = Shift.w;
		const float2 g = (world - HeightGrid.xy) / HeightGrid.z;
		const int2   c = clamp(int2(floor(g)), 0, (int)size - 2);
		const float2 f = saturate(g - float2(c));
		const float h00 = InHeight[c];
		const float h10 = InHeight[c + int2(1, 0)];
		const float h01 = InHeight[c + int2(0, 1)];
		const float h11 = InHeight[c + int2(1, 1)];
		if (h00 > kKnownLimit && h10 > kKnownLimit && h01 > kKnownLimit && h11 > kKnownLimit) {
			ground = lerp(lerp(h00, h10, f.x), lerp(h01, h11, f.x), f.y);
			known = 1.0;
		}
	}
	// Fog amount from the wind field (CSAdvectFog): 1 = the undisturbed layer, banks and gaps included.
	const float2 fogS = InFog[p];   // x = rho, y = rho - rho_eq
	const float  rho = fogS.x;
	const float  shape = LayerShape(world - Iter.zw, Iter.y);
	// Round fade around the field centre (the player): full fog to 55% of the half size, none from 95%. A square
	// 8% edge showed as a straight wall about 29 m ahead in the Blender look render (2026-11-07).
	const float  r = length(ToUv(float2(p)) - 0.5) * 2.0;
	const float  edge = 1.0 - smoothstep(0.55, 0.95, r);
	const float  baseH = Iter.x * shape * BankFactor(world - Iter.zw, Iter.y, LookP.w);
	// Height response (LookP.z = SurfaceResponse k; judge's plan r11 step 1a, design A r07): where the air has thinned or
	// piled up the fog relative to its pattern, q = rho / rho_eq (rho_eq = x - y exactly, CSAdvectFog), the column is
	// that much lower or higher, s = q^k: a runner leaves a trough with banks, a hit a bowl. Radiation fog is a thin cold
	// layer whose depth, not its density, changes when it is pushed aside (the continuity equation of CSAdvectFog is the
	// shallow-water depth equation). Gusts move q too, so in wind the top undulates with nobody near (measured std(q)
	// 0.05 on flat ground, 0.15 beside a cliff; accepted by the user 2026-11-14). It fades with the round edge (the far
	// tier continues the undisturbed layer) and from 1500 to 2500 units from the field centre. k = 0: s = 1, as before.
	const float  eqv = fogS.x - fogS.y;
	const float  q = eqv > kSurfaceEqMin ? fogS.x / eqv : 1.0;
	const float  reach = 1.0 - smoothstep(kSurfaceFadeStart, kSurfaceFadeEnd, length(world - (Grid.xy + 0.5 * Grid.w * Grid.z)));
	const float  s = LookP.z > 0.0 ? lerp(1.0, clamp(pow(max(q, 1e-4), LookP.z), kSurfaceMin, kSurfaceMax), edge * reach) : 1.0;
	// The column's height comes from the shape (and the height response); the fog amount scales its optical
	// thickness (W is the column's mass, W = integral of density over height, so W is proportional to rho). W keeps
	// its formula with baseH whatever s: the march's profile is normalised (RaymarchShaders ProfileSpan), so a lower
	// top holds the same mass, denser. The march box is sized from the shape's maximum x kSurfaceMax while the
	// response is on (WindShaders::LowLayerTop). The round edge fades the density only, keeping the height, so
	// the far tier (RaymarchShaders FarLayer) can take over the same layer with weight 1 - edge at the same
	// place (GPT review 2026-11-07).
	// H is held to at least one unit (baseH where that is lower): the march's LayerTau skips a column under one unit,
	// and the lowest column with the response on is 0.15 x 0.15 x Thickness, under one unit at Thickness < 44.4, so
	// such columns and their mass would leave the screen (review r5 finding 5). With s = 1 (response off) this is
	// baseH exactly.
	const float  H = max(baseH * s, min(baseH, 1.001));   // a hair over LayerTau's H < 1 cut, so filtered reads stay above it
	// Where and when fog forms (PlacementRules x FogConditions): radiation fog x conditions, plus steam over hot
	// springs, plus steam over open water when the air is cold; steam thins with the wind like the rest.
	// Radiation fog evaporates from the weakest places first by day (SteamP.x subtracted; photos 2026-11-08: midday
	// fog stays only over water and marsh). Steam (hot springs, cold air over water) rises as its own taller column
	// broken into wisps by a finer noise drifting upward, instead of only thickening the low layer (Blender
	// placement render: a hot spring under a 0.17 m patch of the layer showed nothing).
	// (water and terrain + region) x conditions: the region value (LookP.y) is added to the placement multiplier,
	// 1 without a placement grid, and the sum is held at 0.
	float radiation = max(PlaceP.x * max(1.0 + LookP.y, 0.0) - SteamP.x, 0.0);
	float steam = 0.0;
	float poolDepth = 0.0;
	if (PlaceP.w > 0.5 && HeightGrid.w > 0.5) {
		const bool   own = PlaceG.z > 0.0;
		const float  size = own ? PlaceG.w : Shift.w;
		const float2 g = (world - (own ? PlaceG.xy : HeightGrid.xy)) / (own ? PlaceG.z : HeightGrid.z);
		const int2   c = clamp(int2(floor(g)), 0, (int)size - 2);
		const float2 f = saturate(g - float2(c));
		float4 p = lerp(lerp(InPlace[c], InPlace[c + int2(1, 0)], f.x), lerp(InPlace[c + int2(0, 1)], InPlace[c + int2(1, 1)], f.x), f.y);
		// Outside the texture's own footprint (it lags the field while a newer build runs) the clamped edge would copy
		// a hot spring or a pool onto new ground: keep only the radiation multiplier there (GPT review 2026-11-08).
		if (any(g < 0.0) || any(g > size - 1.0)) p.yzw = 0.0;
		radiation = max(PlaceP.x * max(p.x + LookP.y, 0.0) - SteamP.x, 0.0);
		steam = max((p.y + p.z * PlaceP.y) * PlaceP.z, 0.0);
		poolDepth = max(p.w, 0.0);
	}
	// The fine structure (banks and gaps) is part of rho (EqFog), so it moves with the air. It scales W, not the
	// profile, so the march keeps its exact profile integral (GPT design review 2026-11-08).
	const float  W = (baseH / 70.0) * rho * edge * (radiation + (SteamP.y > 0.0 ? 0.0 : steam));
	// The tall layer is its own column (OutUpper), marched next to the low one: one H per column either stretched
	// the low layer or, scaled back to its ground density, invented optical depth (GPT design review 2026-11-08).
	// Its parts add their W and take the taller H. Pooled fog: PlacementRules bakes the depth to a level lid into
	// p.w (closing of the terrain, lowered toward the basin floor as the fog weakens).
	float upperH = 0.0;
	float upperW = 0.0;
	if (SteamP.y > 0.0 && steam > 0.0) {
		const float wisp = smoothstep(0.3, 0.8, ValueNoise((world - Iter.zw) / 160.0 + Iter.y * float2(0.011, -0.027)));
		const float steamH = SteamP.y * min(steam, 3.0) * lerp(0.5, 1.0, wisp);
		upperH = steamH;
		upperW = (steamH / 70.0) * SteamP.z * lerp(0.2, 1.0, wisp) * rho * edge;
	}
	// Wide valleys: the coarse grid's pools (same closing, larger window) reach where the fine grid cannot see the
	// rims. Its lid (coarse ground + depth) over the fine ground, faded where the two grounds disagree (cliffs between
	// coarse samples; GPT design review 2026-11-08), and held to the fine rule's 2 x PoolMaxDepth (CoarseP.y).
	if (CoarseG.z > 0.0 && known > 0.5) {
		const float2 cg = (world - CoarseG.xy) / CoarseG.z;
		if (all(cg >= 0.0) && all(cg <= CoarseG.w - 1.0)) {
			const int2   cc = clamp(int2(floor(cg)), 0, (int)CoarseG.w - 2);
			const float2 cf = saturate(cg - float2(cc));
			const float2 c00 = InCoarse[cc], c10 = InCoarse[cc + int2(1, 0)], c01 = InCoarse[cc + int2(0, 1)], c11 = InCoarse[cc + int2(1, 1)];
			// Unknown coarse cells carry -1e30 as their ground: mixing one in would drag the lid (reviewer 2026-11-08).
			const bool cKnown = min(min(c00.x, c10.x), min(c01.x, c11.x)) > kKnownLimit;
			const float  cDepth = lerp(lerp(c00.y, c10.y, cf.x), lerp(c01.y, c11.y, cf.x), cf.y);
			const float  cGround = lerp(lerp(c00.x, c10.x, cf.x), lerp(c01.x, c11.x, cf.x), cf.y);
			const float  cLid = lerp(lerp(c00.x + c00.y, c10.x + c10.y, cf.x), lerp(c01.x + c01.y, c11.x + c11.y, cf.x), cf.y);
			if (cKnown && cDepth >= 1.0) {
				const float agree = 1.0 - smoothstep(0.5 * CoarseP.x, CoarseP.x, abs(ground - cGround));
				poolDepth = max(poolDepth, clamp(cLid - ground, 0.0, CoarseP.y) * agree);
			}
		}
	}
	if (SteamP.w > 0.0 && poolDepth >= 1.0) {
		// A deep pool shows less of the surface structure than the thin layer: 60 % of rho's departure from 1.
		// rho is held to the equilibrium pattern's peak, 1 + wisp amount (Counts.w): an impact or a runner can thin
		// a pool but not pile it up, so the pool's density never passes 1 + 0.6 x WispAmount times its setting,
		// the peak the visibility floor is computed for (FogRenderer.cpp PoolDensityNow; Codex review 2026-11-10).
		upperW += (poolDepth / 70.0) * SteamP.w * edge * lerp(1.0, min(rho, 1.0 + Counts.w), 0.6);
		// The height response lowers a pool's lid where the fog is thinned (s < 1), by 0.6 of at most one low layer's
		// height: a runner does not sink a 4 m lid by 4 m. Never raised, so the box keeps its top; upperW stays.
		upperH = max(upperH, poolDepth - min(poolDepth, Iter.x) * 0.6 * (1.0 - min(s, 1.0)));
	}
	// z = ground * known, so linear filtering next to unknown cells fades the fog instead of dragging the ground
	// toward 0 (reviewer 2026-11-07); the march divides by w.
	const float  k = known * (InWall[p] > 0.5 ? 0.0 : 1.0);
	OutAny[p] = float4(H, W, ground * k, k);
	// H and W both x known, like the ground: filtering next to a wall texel then averages only the known heights
	// (the march divides H by the filtered known), so a wall's height cannot lift a pool's top (GPT review 2026-11-08).
	OutUpper[p] = float2(upperH * k, upperW * k);
	// For the composite's relief shading (RaymarchShaders CompositeOut, ReliefGain): x known and 0 at walls like the
	// rest, so filtering next to them fades it. 0 everywhere while the response is off.
	OutRelief[p] = (s - 1.0) * k;
}
)";
}
