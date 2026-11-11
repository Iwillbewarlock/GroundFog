#pragma once

// Screen-space raymarch of the ground fog (idea from yomotsu ground-fog 2: a thin 2D field inflated in height,
// marched per pixel against the scene depth; code written fresh). Two pixel passes on a fullscreen triangle:
//   march     (reduced resolution): per pixel, walk the camera ray to the scene depth through the box the fog can
//             occupy, accumulate density from the column texture (H, W, ground), Beer-Lambert.
//   composite (full resolution): upsample with depth-aware weights and blend the fog onto the scene.
// Conventions (same as the game and DTD): Z up, game units, row vectors, row_major matrices, positions relative
// to PosAdjust, clip = mul(float4(worldPos - PosAdjust, 1), ViewProj).

namespace RaymarchShaders
{
	// Must match cbuffer RayCB.
	struct alignas(16) RayCB
	{
		float invViewProj[16];  // inverse of the (jittered) viewProj, row vectors
		float field[4];         // x, y = column texture origin, z = extent, w = 1 if columns valid
		float box[4];           // x = lowest ground z, y = highest fog top z, z = max march distance, w = steps
		float fog[4];           // x = opacity per metre, y = profile steepness a (low layer), z = frame mod 64, w = debug view
		float depth[4];         // x = proj _33, y = proj _43, z = full width, w = full height
		float lowRes[4];        // x = low width, y = low height, z = 1 / scale (full / low), w = 0
		float colorLow[4];
		float colorHigh[4];
		float distFog[4];       // near, far, power, max
		float distColorNear[4];
		float distColorFar[4];
		float posAdjust[4];     // camera position (world)
		float heightFog[4];     // x = density per unit at the base (0 = off), y = 1 / falloff (units), z = base z, w = max distance
		float farFog[4];        // x = strength (0 = off; the radiation-fog conditions), y = flat length (units), z = base thickness, w = noise amount
		float farNoise[4];      // x, y = wind travel (units) as CSColumn Iter.zw, z = time (s) as Iter.y
		float sunDir[4];        // xyz = unit direction toward the sun (world), w = glow strength (0 = off; day and sun height applied)
		float sunColor[4];      // rgb = sunlight colour (working space), w = anisotropy g
		float poolGrid[4];      // coarse grid (far pools): origin x, y (world), cell, size; t5
		float poolFar[4];       // x = pool density per metre x strength (0 = off), y = depth cap, z = highest lid z (world)
	};
	static_assert(sizeof(RayCB) == 16 * 22);
	inline constexpr float kUpperSteepnessCpp = 1.0f;   // the HLSL kUpperSteepness (the tall layer's profile)

	inline constexpr const char* kCommon = R"(
cbuffer RayCB : register(b0)
{
	row_major float4x4 InvViewProj;
	float4 Field;
	float4 Box;
	float4 FogP;
	float4 DepthP;
	float4 LowRes;
	float4 ColorLow;
	float4 ColorHigh;
	float4 DistFog;
	float4 DistColorNear;
	float4 DistColorFar;
	float4 PosAdjust;
	float4 HeightP;
	float4 FarP;
	float4 FarN;
	float4 SunDir;     // xyz toward the sun, w = glow strength
	float4 SunColor;   // rgb sunlight, w = anisotropy g
	float4 PoolG;      // far pools: coarse grid origin x, y, cell, size
	float4 PoolF;      // far pools: x = density per metre (0 = off), y = depth cap, z = highest lid z
};

static const float kUnitsPerMetre = 70.0;

struct VSOut
{
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

float LinearDepth(float d)
{
	const float denom = d - DepthP.x;
	return abs(denom) > 1e-7 ? DepthP.y / denom : 1e9;
}
)";

	inline constexpr const char* kVS = R"(
VSOut main(uint id : SV_VertexID)
{
	VSOut o;
	o.uv = float2((id << 1) & 2, id & 2);
	o.pos = float4(o.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
	return o;
}
)";

	// Shared by both passes: march one full-resolution pixel's ray to its scene depth.
#define GROUNDFOG_MARCH_CORE R"(
Texture2D<float>  SceneDepth : register(t0);
Texture2D<float4> Columns    : register(t1);   // H, W, ground z * known, known
Texture2D<float2> UpperCols  : register(t4);   // the tall layer (steam, pooled fog): H x known, W x known; unbound = none
SamplerState      LinearClamp : register(s0);
static const float kUpperSteepness = 1.0;      // gentler than the low layer: pooled fog and steam keep more density up high

// Interleaved gradient noise (Jimenez 2014): per-pixel start offset hides the banding of few steps. FogP.z is the
// frame number mod 64, so the pattern stays exact however long the game runs.
float Ign(float2 p) { return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715)))); }

// Camera position relative to PosAdjust: the eye is the point the projection sends to w = 0, i.e. clip (0, 0, 1, 0).
float3 EyeRel()
{
	const float4 e = mul(float4(0.0, 0.0, 1.0, 0.0), InvViewProj);
	return abs(e.w) > 1e-12 ? e.xyz / e.w : float3(0.0, 0.0, 0.0);
}

// Ray from the eye through pixel px: direction and distance to the scene surface (1e9 for an infinite far plane).
void PixelRay(int2 px, float3 eye, out float3 dir, out float sceneDist)
{
	const float  d = SceneDepth.Load(int3(px, 0));
	const float2 ndc = float2((px.x + 0.5) / DepthP.z * 2.0 - 1.0, 1.0 - (px.y + 0.5) / DepthP.w * 2.0);
	const float4 farH = mul(float4(ndc, d, 1.0), InvViewProj);
	if (farH.w > 1e-6) {
		const float3 seg = farH.xyz / farH.w - eye;
		sceneDist = length(seg);
		dir = seg / max(sceneDist, 1e-3);
	} else {   // depth at an infinite far plane: take the direction from a mid-depth point
		const float4 midH = mul(float4(ndc, 0.5, 1.0), InvViewProj);
		dir = normalize(midH.xyz / midH.w - eye);
		sceneDist = 1e9;
	}
}

float SceneDistance(int2 px)
{
	float3 dir;
	float  dist;
	PixelRay(px, EyeRel(), dir, dist);
	return dist;
}

// Column profile (exponential decay with height, as an isothermal scale-height atmosphere): density
//   rho(h) = rho0 exp(-a h / H)  for 0 <= h <= H, zero outside,  rho0 = W a / (H (1 - exp(-a))),
// so the vertical integral is exactly W whatever the steepness a. Derivation:
//   integral of exp(-a h/H) dh from ha to hb = (H/a) (exp(-a ha/H) - exp(-a hb/H)),
// and over the whole column (0..H) that is (H/a)(1 - exp(-a)), hence rho0. The span between ha and hb per unit W is
//   (exp(-a ha/H) - exp(-a hb/H)) / (1 - exp(-a)),  ha, hb clamped to [0, H].
// a is held to >= kMinSteepness (a -> 0 is the even slab: span -> (hb - ha) / H). Both differences of exponentials are
// written as exp(-a lo) (1 - exp(-a d)) with a series for small a d, so a near 0 keeps its precision in float.
static const float kMinSteepness = 1e-3;

float OneMinusExp(float x)   // 1 - exp(-x), x >= 0
{
	return x < 1e-2 ? x * (1.0 - x * (0.5 - x / 6.0)) : 1.0 - exp(-x);
}

float ProfileSpan(float ha, float hb, float H, float a)
{
	a = max(a, kMinSteepness);
	const float xa = saturate(ha / H);
	const float xb = saturate(hb / H);
	return exp(-a * min(xa, xb)) * OneMinusExp(a * abs(xb - xa)) / OneMinusExp(a);
}

// rho(h) H / W: the profile's density at height h relative to the mean W / H (the ground factor a / (1 - exp(-a)) at
// h <= 0, zero from H up).
float ProfileDensity(float h, float H, float a)
{
	a = max(a, kMinSteepness);
	const float x = max(h / H, 0.0);
	return x < 1.0 ? a * exp(-a * x) / OneMinusExp(a) : 0.0;
}

// Optical depth of one column layer (height H, optical thickness W, profile steepness a) over a step whose heights
// above the ground go from a_ha to a_hb; a_h is the step point's height (for nearly level steps). a_k scales W.
float LayerTau(float a_ha, float a_hb, float a_h, float a_H, float a_W, float a_k, float a_a, float a_stepLen, float a_dz)
{
	if (a_H < 1.0 || a_W <= 0.0 || max(a_ha, a_hb) <= 0.0 || min(a_ha, a_hb) >= a_H) return 0.0;
	if (abs(a_dz) > 1e-3 * a_H) {
		// metres along the ray per metre of height = stepLen / |dz|
		return FogP.x * a_W * a_k * ProfileSpan(a_ha, a_hb, a_H, a_a) * (a_stepLen / abs(a_dz));
	}
	if (a_h < 0.0 || a_h >= a_H) return 0.0;
	const float rho = a_W / (a_H / kUnitsPerMetre) * ProfileDensity(a_h, a_H, a_a);
	return FogP.x * a_k * rho * (a_stepLen / kUnitsPerMetre);
}

// Returns fog, brightness, mean fog distance, scene distance.
float4 MarchPixel(int2 px, float2 noisePos, int steps)
{
	const float3 eye = EyeRel();
	float3 dir;
	float  sceneDist;
	PixelRay(px, eye, dir, sceneDist);
	const float maxDist = min(sceneDist, Box.z);

	// Clip the ray to the fog box: horizontal = column texture square, vertical = [lowest ground, highest top].
	const float3 boxMin = float3(Field.xy, Box.x) - PosAdjust.xyz - eye;
	const float3 boxMax = float3(Field.xy + Field.z, Box.y) - PosAdjust.xyz - eye;
	const float3 inv = 1.0 / (abs(dir) > 1e-6 ? dir : 1e-6);
	const float3 t0 = boxMin * inv;
	const float3 t1 = boxMax * inv;
	const float  tEnter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
	const float  tExit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), maxDist));
	if (Field.w < 0.5 || tExit <= tEnter) {
		return float4(0.0, 0.0, 0.0, sceneDist);
	}

	// Each step reads the column once (at a jittered point) and integrates the height profile exactly over the
	// step's height span, so a thin layer is caught even when one step is longer than the layer is tall (the box
	// spans the whole terrain height range). Nearly level steps fall back to the midpoint rule.
	const float  stepLen = (tExit - tEnter) / steps;
	const float  jitter = Ign(noisePos + 5.588238 * FogP.z);
	const float  a = FogP.y;
	const float3 origin = PosAdjust.xyz + eye;
	const float  dz = dir.z * stepLen;
	float transmit = 1.0;
	float sumWeight = 0.0;
	float sumBright = 0.0;
	float sumDist = 0.0;
	[loop] for (int s = 0; s < steps; ++s) {
		const float  ta = tEnter + s * stepLen;
		const float  t = ta + jitter * stepLen;
		const float3 world = origin + dir * t;
		const float2 cuv = (world.xy - Field.xy) / Field.z;
		const float4 col = Columns.SampleLevel(LinearClamp, cuv, 0);
		if (col.w < 0.05) continue;
		const float2 up = UpperCols.SampleLevel(LinearClamp, cuv, 0);
		const float ground = col.z / col.w;
		const float ha = origin.z + dir.z * ta - ground;
		const float hb = ha + dz;
		const float h = world.z - ground;
		// Low layer and tall layer marched as two columns over the same ground, their optical depths added.
		const float tauLow = LayerTau(ha, hb, h, col.x, col.y, col.w, a, stepLen, dz);
		const float upH = up.x / col.w;   // H was stored x known, as the ground
		const float tauUp = LayerTau(ha, hb, h, upH, up.y, 1.0, kUpperSteepness, stepLen, dz);
		const float tau = tauLow + tauUp;
		if (tau <= 0.0) continue;
		// Brightness = height fraction h / H in each layer (0 at the ground, 1 at its top); ColorLow and ColorHigh set the range.
		const float brightLow = clamp(0.5 * (ha + hb), 0.0, max(col.x, 1.0)) / max(col.x, 1.0);
		const float brightUp = clamp(0.5 * (ha + hb), 0.0, max(upH, 1.0)) / max(upH, 1.0);
		const float absorbed = transmit * (1.0 - exp(-tau));
		sumWeight += absorbed;
		sumBright += absorbed * (tauLow * brightLow + tauUp * brightUp) / tau;   // brighter toward each layer's top
		sumDist += absorbed * t;
		transmit *= exp(-tau);
	}
	const float fog = 1.0 - transmit;
	const float bright = sumWeight > 1e-5 ? sumBright / sumWeight : 0.0;
	const float meanDist = sumWeight > 1e-5 ? sumDist / sumWeight : sceneDist;
	return float4(fog, bright, meanDist, sceneDist);
}
)"

	inline constexpr const char* kMarchPS = GROUNDFOG_MARCH_CORE R"(
float4 main(VSOut i) : SV_TARGET
{
	// The low-res pixel stands for the full-res pixel at its block's lower right (2x2 at scale 0.5).
	const int2 px = int2(min(i.pos.xy * LowRes.z, DepthP.zw - 1.0));
	return MarchPixel(px, i.pos.xy, (int)Box.w);
}
)";

	// Shared by the composite pass and the probe: upsample the low-res march for one full-res pixel, and its colour.
#define GROUNDFOG_COMPOSITE_CORE R"(
Texture2D<float4> MarchTex   : register(t2);   // fog, brightness, mean distance, scene distance

// Returns fog, brightness, mean fog distance, scene distance for full-res pixel `full`.
float4 CompositeM(int2 full, float2 noisePos)
{
	const float myDist = SceneDistance(full);

	// 3x3 low-res neighbours whose scene distance matches this pixel's (within 15%). Where none matches (a thin
	// foreground object the low-res samples missed, or ground near the horizon where distance changes fast per
	// pixel) this pixel marches its own ray at full resolution, so it ends at its own depth (Codex reviews
	// 2026-11-07). Brightness and distance are averaged weighted by fog amount, so clear neighbours do not pull the
	// colour; the !(<=) test also skips NaN.
	const float2 lowPos = (float2(full) + 0.5) / LowRes.z - 0.5;
	const int2   base = int2(floor(lowPos));
	float  sumFog = 0.0;
	float2 sumYZ = 0.0;
	float  wsum = 0.0;
	float  wfog = 0.0;
	[unroll] for (int y = -1; y <= 1; ++y) {
		[unroll] for (int x = -1; x <= 1; ++x) {
			const int2   c = clamp(base + int2(x, y), int2(0, 0), int2(LowRes.xy) - 1);
			const float4 mm = MarchTex.Load(int3(c, 0));
			const float  rel = abs(mm.w - myDist) / max(myDist, 1.0);
			if (!(rel <= 0.15)) continue;
			const float2 off = float2(c) - lowPos;
			const float  w = (1.0 - rel / 0.15) / (1.0 + dot(off, off));
			sumFog += mm.x * w;
			sumYZ += mm.yz * (w * mm.x);
			wsum += w;
			wfog += w * mm.x;
		}
	}
	float4 m;
	if (wsum > 1e-5) {
		m.x = sumFog / wsum;
		m.yz = wfog > 1e-6 ? sumYZ / wfog : float2(0.0, myDist);
		m.w = myDist;
	} else {
		m = MarchPixel(full, noisePos, (int)Box.w);
	}
	return m;
}

float3 FogColour(float4 m)
{
	float3 colour = lerp(ColorLow.rgb, ColorHigh.rgb, saturate(m.y));
	const float range = max(DistFog.y - DistFog.x, 1.0);
	const float f = min(DistFog.w, pow(saturate((m.z - DistFog.x) / range), DistFog.z));
	return lerp(colour, lerp(DistColorNear.rgb, DistColorFar.rgb, f), f);
}

// ---- far tier: the ground layer beyond the wind field, on the local ground plane at each pixel ----
uint FarHash(uint2 p)
{
	uint h = p.x * 0x8da6b343u ^ p.y * 0xd8163841u;
	h ^= h >> 13; h *= 0x2c1b3c6du; h ^= h >> 12;
	return h;
}

// Same value noise as WindShaders.h ValueNoise, so the far patches continue the near ones.
float FarNoise(float2 p)
{
	const float2 i = floor(p);
	const float2 f = p - i;
	const float2 u = f * f * (3.0 - 2.0 * f);
	const uint2 c = uint2(int2(i));
	const float a = FarHash(c) / 4294967295.0;
	const float b = FarHash(c + uint2(1, 0)) / 4294967295.0;
	const float d = FarHash(c + uint2(0, 1)) / 4294967295.0;
	const float e = FarHash(c + uint2(1, 1)) / 4294967295.0;
	return lerp(lerp(a, b, u.x), lerp(d, e, u.x), u.y);
}

// Layer thickness H (units) at world xy: the same shape as CSColumn's (WindShaders.h LayerShape; the constants and
// the two octaves must stay identical there and here, so the far tier continues the near layer).
static const float kShapeFloor = 0.15;
static const float kShapeContrast = 1.8;
static const float kShapeLevel = 0.62;   // noise value that maps to mid-height: most of the layer is low, with mounds

float FarLayerH(float2 worldXY)
{
	const float2 p = (worldXY - FarN.xy) / 900.0;
	const float  n = (2.0 * FarNoise(p + FarN.z * float2(0.004, 0.0015)) + FarNoise(p * 2.0 + 19.7 + FarN.z * float2(-0.005, 0.003))) / 3.0;
	const float  shape = lerp(kShapeFloor, 1.0, saturate((n - kShapeLevel) * kShapeContrast + 0.5));
	return FarP.z * lerp(1.0, shape, FarP.w);
}

// Share of the layer the far tier carries at world xy: 1 - the field's round fade (CSColumn), 1 with no field.
float FarWeight(float2 worldXY)
{
	if (Field.w < 0.5 || Field.z < 1.0) return 1.0;
	const float r = length((worldXY - Field.xy) / Field.z - 0.5) * 2.0;
	return smoothstep(0.55, 0.95, r);
}

// Scene point of render-area pixel px relative to PosAdjust; false for sky or off the scene area.
bool FarScenePoint(int2 px, float3 eye, out float3 pos, out float dist)
{
	pos = float3(0.0, 0.0, 0.0);
	dist = 1e9;
	if (any(px < 0) || any(px >= int2(DepthP.zw))) return false;
	float3 dir;
	PixelRay(px, eye, dir, dist);
	pos = eye + dir * dist;
	return dist < 1e8;
}

// How far this pixel is off the surface seen on one side: 1/distance at 1 and 2 px extrapolated to this pixel, relative
// (on a plane 1/view depth is linear across the screen; the 1/cos difference over 2 px is negligible). 1e30 when
// either point is missing. pos2 / dist2 = the 2 px point.
float FarSideError(int2 full, int2 dir1, float3 eye, float dist, out float3 pos2, out float dist2)
{
	float3 p1;
	float  d1;
	const bool ok2 = FarScenePoint(full + 2 * dir1, eye, pos2, dist2);
	if (!ok2 || !FarScenePoint(full + dir1, eye, p1, d1)) return 1e30;
	return abs((2.0 / d1 - 1.0 / dist2) * dist - 1.0);
}

// Neighbour 2 px away for the ground plane, never across a depth jump (35% of the distance): the side that continues
// this pixel's surface best (FarSideError); where that one is across a jump, the other side, only if it continues the
// surface too. Taking the first side that passed the jump test let a pixel just above a far silhouette pair with the
// nearer ridge below it: the plane through both lay along the ray, which then crossed the layer for the whole window
// (white lines along far ridges and roofs, Nexus report 2026-11-10). Comparing plain distance differences does not
// separate them: on far ground seen at a low angle the distance also grows ~20% per 2 px. The second rule keeps flat
// ground 4-7 px under the horizon, where both sides continue the plane (errors < 1e-3, their order is noise) and the
// far side is past the jump test (Codex and reviewer 2026-11-10); a ridge nearer by 19% is off by 0.04-0.05 there, so
// it stays out and that pixel carries no far layer.
static const float kFarSideTol = 0.02;

bool FarNeighbour(int2 full, int2 step, float3 eye, float dist, out float3 pos)
{
	const int2 dir1 = step / 2;
	float3 pa, pb;
	float  da, db;
	const float ea = FarSideError(full, dir1, eye, dist, pa, da);
	const float eb = FarSideError(full, -dir1, eye, dist, pb, db);
	const bool  firstA = ea <= eb;
	const float e1 = firstA ? ea : eb, e2 = firstA ? eb : ea;
	const float d1 = firstA ? da : db, d2 = firstA ? db : da;
	const bool  ok1 = e1 < 1e29 && abs(d1 - dist) < 0.35 * dist;
	const bool  ok2 = e2 < kFarSideTol && abs(d2 - dist) < 0.35 * dist;
	pos = (ok1 == firstA) ? pa : pb;   // the first side if it passed, else the second
	return ok1 || ok2;
}

// Share of the far layer a surface carries by its slope: all of it up to about 20 degrees (normal.z 0.94), none from
// about 42 degrees (0.74), smooth between (0.93 at 25 degrees, 0.69 at 30, 0.05 at 40). A hard cut at 0.7 put the far
// layer on 40-degree roofs at full strength and made surfaces near 45 degrees switch between fog and none from pixel
// to pixel: the plane's normal comes from the depth buffer, and one step of a 24-bit depth tilts it by a fraction of
// a degree (hatching and a frosted look on far roofs, arca.live report 2026-11-11; tools/offline/HatchRepro.cpp).
// The band is wide on purpose: the same tilt noise moves the share inside the band, and a narrower band (0.74..0.88)
// doubled the pixel noise on 35-degree roofs. Known limit: the normal still comes from neighbours 2 px away, so some
// noise stays inside the band (about 2 of 255 steps at 12000 units on a 1080p pixel). Taking the neighbours 4 px away
// halves it but loses the fog rows next to far silhouettes (the two silhouette checks in RaymarchCheck fail); the
// upgrade path is a wider baseline only where both sides continue the surface.
static const float kFarSlopeNone = 0.74;
static const float kFarSlopeFull = 0.94;

// Optical depth of the far layer for pixel `full` along its ray `dir`. The ground under the last FarP.y units
// before the scene point P is taken as the plane through P with the normal from two neighbours; walls, cliffs,
// trunks and steep roofs carry none and slopes fade out (kFarSlopeNone, kFarSlopeFull). Height above that plane falls linearly along the ray, so each of four
// sub-segments integrates the column profile exactly (ProfileSpan) with the H and share at its middle.
// Returns tau; a_bright = absorption-weighted height fraction h/H for the colour, as the march.
float FarLayerTau(int2 full, float3 eye, float3 dir, float dist, out float a_bright, out float a_nz)
{
	a_bright = 0.7;
	a_nz = -1.0;
	if (!(FarP.x > 0.0) || !(dist < 1e8)) return 0.0;
	const float3 P = eye + dir * dist;
	float3 A, B;
	if (!FarNeighbour(full, int2(2, 0), eye, dist, A) || !FarNeighbour(full, int2(0, 2), eye, dist, B)) return 0.0;
	float3 nrm = cross(A - P, B - P);
	const float nl = length(nrm);
	if (!(nl > 1e-6)) return 0.0;
	nrm /= nl;
	if (nrm.z < 0.0) nrm = -nrm;
	a_nz = nrm.z;
	const float slopeShare = smoothstep(kFarSlopeNone, kFarSlopeFull, nrm.z);
	if (!(slopeShare > 0.0)) return 0.0;
	const float rate = dot(dir, nrm) / nrm.z;   // vertical height above the plane per unit along the ray
	if (!(rate < 0.0)) return 0.0;              // ray not coming down onto the plane
	// Only the last H_max / |rate| units can be inside the layer (H_max = the thickest patch, FarP.z): split that
	// part, so the share and H are read where the fog is (GPT review 2026-11-07: whole-window midpoints missed it).
	const float t0 = max(max(0.0, dist - FarP.y), dist - FarP.z / max(-rate, 1e-6));
	const float2 base = PosAdjust.xy + eye.xy;
	if (FarWeight(base + P.xy - eye.xy) <= 0.0 && FarWeight(base + dir.xy * t0) <= 0.0) return 0.0;
	const float a = FogP.y;
	float tau = 0.0;
	float sumW = 0.0;
	float sumB = 0.0;
	[unroll] for (int k = 0; k < 4; ++k) {
		const float ta = lerp(t0, dist, k / 4.0);
		const float tb = lerp(t0, dist, (k + 1) / 4.0);
		const float tm = 0.5 * (ta + tb);
		const float2 xy = base + dir.xy * tm;
		const float  share = FarWeight(xy);
		const float  H = FarLayerH(xy);
		const float  ha = rate * (ta - dist);
		const float  hb = rate * (tb - dist);
		if (share <= 0.0 || H < 1.0 || min(ha, hb) >= H) continue;
		const float W = H / kUnitsPerMetre;
		const float len = tb - ta;
		float t;
		if (abs(ha - hb) > 1e-3 * H) {
			t = FogP.x * W * ProfileSpan(ha, hb, H, a) * (len / abs(ha - hb));
		} else {
			const float h = 0.5 * (ha + hb);
			t = FogP.x * (W / (H / kUnitsPerMetre)) * ProfileDensity(h, H, a) * (len / kUnitsPerMetre);
		}
		t *= share * FarP.x * slopeShare;   // strength inside the loop, so the brightness weights see the same density as alpha
		const float hMid = clamp(0.5 * (ha + hb), 0.0, H);
		const float absorbed = exp(-tau) * (1.0 - exp(-t));
		sumW += absorbed;
		sumB += absorbed * (hMid / H);
		tau += t;
	}
	a_bright = sumW > 1e-6 ? sumB / sumW : 0.7;
	return tau;
}
)" R"(
// ---- far pools: the coarse grid's pooled fog beyond the wind field (CoarsePools) ----
// Its own medium in absolute heights: the coarse ground and pool depth sampled along the ray. It takes no ground plane
// from the depth buffer (GPT design review 2026-11-08: an absolute lid does not fit the far layer's fixed-thickness
// integral over that plane on slopes): the ray up to its scene point is open air, so every part of it under the lid
// is fog. Hands over from the field's pools with FarWeight, as the far layer.
Texture2D<float2> CoarseR : register(t5);   // ground z, pool depth; unknown ground -1e30; unbound = no far pools
static const int   kFarPoolMinSteps = 10;   // short windows: the field's fade still needs several shares
static const int   kFarPoolMaxSteps = 24;
static const float kFarPoolMinPiece = 512.0;   // one coarse cell

// Coarse ground (bilinear), the pool's lid and a fade over the grid's last two cells at world xy; false off the grid,
// next to an unknown cell or without a pool (bilinear depth < 1, as CSColumn's coarse block). The lid mixes only the
// pooled corners' lids (ground + depth, weights renormalised): the plain bilinear lid also mixed in the ground of a
// corner without a pool, so next to a cliff it rose toward the cliff top (reviewer 2026-11-08: dense fog 100 units
// above a 200-unit level lid). In a basin every pooled corner has the same lid, so it is unchanged there.
bool CoarsePoolAt(float2 xy, out float a_ground, out float a_lid, out float a_fade)
{
	a_ground = 0.0;
	a_lid = 0.0;
	a_fade = 0.0;
	const float2 g = (xy - PoolG.xy) / PoolG.z;
	const float  last = PoolG.w - 1.0;
	if (any(g < 0.0) || any(g > last)) return false;
	const int2   c = clamp(int2(floor(g)), 0, (int)PoolG.w - 2);
	const float2 f = saturate(g - float2(c));
	const float2 c00 = CoarseR.Load(int3(c, 0));
	const float2 c10 = CoarseR.Load(int3(c + int2(1, 0), 0));
	const float2 c01 = CoarseR.Load(int3(c + int2(0, 1), 0));
	const float2 c11 = CoarseR.Load(int3(c + int2(1, 1), 0));
	if (!(min(min(c00.x, c10.x), min(c01.x, c11.x)) > -1e29)) return false;
	const float4 w = float4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
	const float4 d = float4(c00.y, c10.y, c01.y, c11.y);
	if (!(dot(w, d) >= 1.0)) return false;
	const float4 wp = w * saturate(d);   // pooled corners only (depth >= 1 -> full weight)
	const float  sw = wp.x + wp.y + wp.z + wp.w;
	if (!(sw > 1e-6)) return false;
	a_ground = dot(w, float4(c00.x, c10.x, c01.x, c11.x));
	a_lid = dot(wp, float4(c00.x, c10.x, c01.x, c11.x) + d) / sw;
	a_fade = saturate(min(min(g.x, g.y), min(last - g.x, last - g.y)) / 2.0);
	return a_fade > 0.0;
}

// Optical depth of the far pools along the ray (eye relative to PosAdjust, unit dir) up to `dist`; a_bright as
// FarLayerTau, a_meanT the absorption-weighted distance of the fog along the ray (for the distance colour). The window
// runs from where the ray drops under the highest lid (PoolF.z), leaves the field's inner circle (FarWeight 0: the
// field's own pools) and enters the grid, to the scene point, the grid's edge or (rising) the highest lid; it is cut
// into pieces of about one coarse cell (kFarPoolMinSteps to kFarPoolMaxSteps). Per piece the ground, lid and share at its middle.
// Fog fills the air under the lid: above the coarse ground with the tall layer's profile (kUpperSteepness, height
// H = lid - ground held to PoolF.y), and below the coarse ground (a bilinear ground over a hollow floor sits above the
// real one; next to a cliff the ground rises above the lid) at the density at the ground, up to the lid. Heights are
// linear along a piece, so both parts integrate exactly.
float FarPoolTau(float3 eye, float3 dir, float dist, out float a_bright, out float a_meanT)
{
	a_bright = 0.7;
	a_meanT = dist;
	if (!(PoolF.x > 0.0) || !(PoolF.y >= 1.0) || !(PoolG.z > 0.0) || !(PoolG.w >= 2.0)) return 0.0;
	const float3 e = PosAdjust.xyz + eye;
	const float2 inv = 1.0 / (abs(dir.xy) > 1e-6 ? dir.xy : 1e-6);
	const float2 ta = (PoolG.xy - e.xy) * inv;
	const float2 tb = (PoolG.xy + PoolG.z * (PoolG.w - 1.0) - e.xy) * inv;
	float t0 = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), 0.0);
	float t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), dist);
	if (e.z > PoolF.z) {
		if (!(dir.z < -1e-6)) return 0.0;
		t0 = max(t0, (PoolF.z - e.z) / dir.z);
	} else if (dir.z > 1e-6) {
		// Rising out of a pool: the window ends at the highest lid, so the pieces sit where the fog is (GPT review
		// 2026-11-08: a sky ray from inside a valley pool spread 10 pieces to the grid edge, 3.8x the optical depth).
		t1 = min(t1, (PoolF.z - e.z) / dir.z);
	}
	if (Field.w > 0.5 && Field.z >= 1.0) {
		const float2 oc = e.xy - (Field.xy + 0.5 * Field.z);
		const float  R = 0.275 * Field.z;   // r = 0.55 of the half size
		const float  c = dot(oc, oc) - R * R;
		if (c < 0.0) {
			const float a = dot(dir.xy, dir.xy);
			if (!(a > 1e-8)) return 0.0;   // straight down inside the field
			const float b = dot(oc, dir.xy);
			t0 = max(t0, (-b + sqrt(max(b * b - a * c, 0.0))) / a);
		}
	}
	if (!(t1 > t0)) return 0.0;
	// Pieces of about one coarse cell (reviewer 2026-11-08: 10 pieces over ~20000 units were 3-4 cells long, so narrow
	// pools came and went as the grid recentred and the hand-over shares no longer added up).
	const int   steps = clamp((int)ceil((t1 - t0) / kFarPoolMinPiece), kFarPoolMinSteps, kFarPoolMaxSteps);
	const float aUp = kUpperSteepness;
	const float rho0 = PoolF.x * ProfileDensity(0.0, 1.0, aUp);   // per metre at the ground (W / (H / 70) = PoolF.x)
	float tau = 0.0;
	float sumW = 0.0;
	float sumB = 0.0;
	float sumT = 0.0;
	[loop] for (int k = 0; k < steps; ++k) {
		const float tA = lerp(t0, t1, k / (float)steps);
		const float tB = lerp(t0, t1, (k + 1) / (float)steps);
		const float2 xy = e.xy + dir.xy * (0.5 * (tA + tB));
		float g, lid, fade;
		if (!CoarsePoolAt(xy, g, lid, fade)) continue;
		// Where the fine ground disagrees with the coarse one (a cliff between coarse samples) the field fades its own
		// pool (CSColumn agree) and this one has no fine ground to test: across the field's fade the look blends from
		// the field's shape to this one. Taking the field's factor here too was tried and left a step at the field's
		// square edge (offline check 2026-11-08), so the shares stay 1 - edge and edge.
		const float share = FarWeight(xy) * fade;
		const float H = min(lid - g, PoolF.y);       // profile part (when >= 1)
		const float floorTop = min(g, lid);           // top of the part below the coarse ground
		const float top = H >= 1.0 ? g + H : floorTop;
		const float zA = e.z + dir.z * tA;
		const float zB = e.z + dir.z * tB;
		if (share <= 0.0 || min(zA, zB) >= top) continue;
		const float len = tB - tA;
		const float dz = abs(zA - zB);
		float t = 0.0;
		float below;
		if (dz > 1e-3) {
			below = saturate((floorTop - min(zA, zB)) / dz);
			if (H >= 1.0) t = FogP.x * (H / kUnitsPerMetre) * PoolF.x * ProfileSpan(zA - g, zB - g, H, aUp) * (len / dz);
		} else {
			const float z = 0.5 * (zA + zB);
			below = z < floorTop ? 1.0 : 0.0;
			if (H >= 1.0 && z >= g && z < g + H) t = FogP.x * PoolF.x * ProfileDensity(z - g, H, aUp) * (len / kUnitsPerMetre);
		}
		t = (t + FogP.x * rho0 * below * (len / kUnitsPerMetre)) * share;
		const float zMid = 0.5 * (zA + zB);
		const float absorbed = exp(-tau) * (1.0 - exp(-t));
		sumW += absorbed;
		sumB += absorbed * saturate((zMid - floorTop) / max(top - floorTop, 1.0));
		sumT += absorbed * 0.5 * (tA + tB);
		tau += t;
	}
	if (sumW > 1e-6) {
		a_bright = sumB / sumW;
		a_meanT = sumT / sumW;
	}
	return tau;
}

// Optical depth of the height fog along the ray from the eye to distance `dist`: density HeightP.x * exp(-h * k)
// with h the height above HeightP.z and k = HeightP.y. Closed form (h(t) = h0 + dir.z t):
//   tau = D * (exp(-h0 k) - exp(-hL k)) / (dir.z k),  or  D * L * exp(-hmid k) for a nearly level ray.
// Each exponent is clamped on its own, so a ray far below the base cannot overflow a product.
float HeightFogTau(float3 eye, float3 dir, float dist)
{
	if (!(HeightP.x > 0.0)) return 0.0;
	const float L = min(dist, HeightP.w);
	const float k = HeightP.y;
	const float h0 = PosAdjust.z + eye.z - HeightP.z;
	const float hL = h0 + dir.z * L;
	// tau = D * L * exp(-k min(h0, hL)) * (1 - exp(-x)) / x with x = k |hL - h0|: one exponent, never a difference
	// of two clamped ones (GPT review 2026-11-07: deep below the base both clamped to the same value and gave 0).
	const float x = k * abs(hL - h0);
	const float phi = x < 1e-3 ? 1.0 - 0.5 * x + x * x / 6.0 : (1.0 - exp(-x)) / x;
	const float tau = L * exp(clamp(-k * min(h0, hL), -80.0, 80.0)) * phi;
	return min(HeightP.x * tau, 1e30);   // finite; alpha saturates in CompositeOut, the colour uses the real value
}

// Colour of the height fog: the game's distance fog colour at the mean distance the light comes from (taken as
// for an even fog of the same optical depth over L).
float3 HeightFogColour(float tau, float L)
{
	float mean = 0.5 * L;
	if (tau > 1e-3) {
		mean = L * saturate(1.0 / tau - exp(-tau) / max(1.0 - exp(-tau), 1e-6));
	}
	const float range = max(DistFog.y - DistFog.x, 1.0);
	return lerp(DistColorNear.rgb, DistColorFar.rgb, saturate((mean - DistFog.x) / range));
}

// What the composite writes for m at pixel `full`; alpha 0 where it discards (the probe records the same value).
// Layered front to back: wind field fog, far ground layer, height fog: alpha = 1 - (1 - aF)(1 - aG)(1 - aH).
float4 CompositeOut(float4 m, int2 full)
{
	const float aF = saturate(m.x);
	const float3 eye = EyeRel();
	float3 dir;
	float  dist;
	PixelRay(full, eye, dir, dist);
	float brightG;
	float nzG;
	const float tauL = FarLayerTau(full, eye, dir, dist, brightG, nzG);
	// Far pools share the far layer's place in the stack (both lie on the ground beyond the field): optical depths
	// add, brightness weighted by them.
	float brightP, meanP;
	const float tauP = FarPoolTau(eye, dir, dist, brightP, meanP);
	const float tauG = tauL + tauP;
	float meanG = dist;   // the far layer lies at the scene point; the pools along the ray (reviewer 2026-11-08)
	if (tauP > 0.0) {
		brightG = (tauL * brightG + tauP * brightP) / tauG;
		meanG = (tauL * dist + tauP * meanP) / tauG;
	}
	const float aG = 1.0 - exp(-tauG);
	const float tauH = HeightFogTau(eye, dir, dist);
	const float aH = 1.0 - exp(-min(tauH, 50.0));
	const float a = 1.0 - (1.0 - aF) * (1.0 - aG) * (1.0 - aH);
	if (!(a >= 1e-3)) return float4(0.0, 0.0, 0.0, 0.0);
	if (FogP.w > 0.5) return float4(1.0, 0.0, 1.0, 0.5 * step(0.01, aF));   // debug: magenta where the field fog is
	const float3 cF = FogColour(m);
	const float3 cG = FogColour(float4(aG, brightG, meanG, dist));
	const float3 cH = HeightFogColour(tauH, min(dist, HeightP.w));
	float3 colour = (aF * cF + (1.0 - aF) * aG * cG + (1.0 - aF) * (1.0 - aG) * aH * cH) / a;
	// Backlit glow (photos 2026-11-08: fog toward a low sun glows far brighter than the land): Henyey-Greenstein
	// forward peak over the isotropic level, capped at 1.5 (Blender Cycles with g 0.6: multiple scattering flattens
	// the peak to about 1.6x the plain fog near the sun, 1.3x at 40 degrees), times the sunlight colour. Added once to the mixed colour of
	// all three media (the same as adding it to each, they share the light); alpha unchanged (GPT design review).
	if (SunDir.w > 0.0) {
		const float g = SunColor.w;
		const float c = dot(dir, SunDir.xyz);
		const float hg4pi = (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
		colour += SunColor.rgb * SunDir.w * min(max(hg4pi - 1.0, 0.0), 1.5);
	}
	return float4(colour, a);
}
)"

	inline constexpr const char* kCompositePS = GROUNDFOG_MARCH_CORE GROUNDFOG_COMPOSITE_CORE R"(
float4 main(VSOut i) : SV_TARGET
{
	const float4 o = CompositeOut(CompositeM(int2(i.pos.xy), i.pos.xy), int2(i.pos.xy));
	if (!(o.a > 0.0)) discard;
	return o;
}
)";

	// Probe (ini Probe=1): one thread group of kProbeCols x kProbeRows threads, one screen point each. Phase 0 runs
	// after the march and before the composite and records the ray, the scene point, the column there, a full-res
	// march, the composite result and the scene colour; phase 1 runs after the composite and records the scene
	// colour again, so the log can check the blend. Read back on the CPU a few frames later.
	inline constexpr int kProbeCols = 9;
	inline constexpr int kProbeRows = 7;
	inline constexpr int kProbeSlots = 9;   // float4 records per point
	inline constexpr int kProbeExtra = 2;   // float4 records after the points (eye, box)
	inline constexpr const char* kProbeCS = GROUNDFOG_MARCH_CORE GROUNDFOG_COMPOSITE_CORE R"(
Texture2D<float4>          SceneTex : register(t3);
RWStructuredBuffer<float4> ProbeOut : register(u0);
cbuffer ProbeCB : register(b1)
{
	float4 ProbeP;   // x = phase
};
static const uint kCols = 9;
static const uint kRows = 7;
static const uint kSlots = 9;

[numthreads(9, 7, 1)]
void main(uint3 t : SV_GroupThreadID)
{
	const uint i = t.y * kCols + t.x;
	const uint b = i * kSlots;
	const int2 px = int2((float2(t.xy) + 0.5) / float2(kCols, kRows) * DepthP.zw);
	if (ProbeP.x > 0.5) {
		ProbeOut[b + 7] = SceneTex.Load(int3(px, 0));
		return;
	}
	const float3 eye = EyeRel();
	float3 dir;
	float  dist;
	PixelRay(px, eye, dir, dist);
	const float  d = SceneDepth.Load(int3(px, 0));
	const float3 world = PosAdjust.xyz + eye + dir * min(dist, 1e6);
	const float2 cuv = (world.xy - Field.xy) / Field.z;
	const bool   inField = all(cuv >= 0.0) && all(cuv <= 1.0) && dist < 1e6;
	const float4 col = inField ? Columns.SampleLevel(LinearClamp, cuv, 0) : float4(0.0, 0.0, 0.0, 0.0);
	const float4 c = CompositeM(px, float2(px) + 0.5);
	ProbeOut[b + 0] = float4(float2(px), d, dist);
	float brightP, meanP;
	ProbeOut[b + 1] = float4(dir, FarPoolTau(eye, dir, dist, brightP, meanP));   // w = far pools' optical depth
	ProbeOut[b + 2] = float4(world, col.w >= 0.05 ? col.z / col.w : -1e9);
	ProbeOut[b + 3] = col;
	ProbeOut[b + 4] = MarchPixel(px, float2(px) + 0.5, (int)Box.w);
	ProbeOut[b + 5] = CompositeOut(c, px);
	ProbeOut[b + 6] = SceneTex.Load(int3(px, 0));
	ProbeOut[b + 7] = float4(0.0, 0.0, 0.0, 0.0);
	{
		float brightG, nzG;
		const float tauG = FarLayerTau(px, eye, dir, dist, brightG, nzG);
		ProbeOut[b + 8] = float4(nzG, tauG, FarWeight(PosAdjust.xy + world.xy - PosAdjust.xy), HeightFogTau(eye, dir, dist));
	}
	if (i == 0) {
		ProbeOut[kCols * kRows * kSlots + 0] = float4(eye, Box.x);
		ProbeOut[kCols * kRows * kSlots + 1] = float4(Box.y, Box.z, Box.w, FogP.x);
	}
}
)";
#undef GROUNDFOG_COMPOSITE_CORE
#undef GROUNDFOG_MARCH_CORE
}
