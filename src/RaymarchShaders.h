#pragma once

#include <algorithm>
#include <cmath>

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
		float lowRes[4];        // x = low width, y = low height, z = 1 / scale (full / low), w = ThinFogStart (units from the eye, 0 = off; CompositeM)
		float colorLow[4];      // rgb = the colour at the layer's ground, w = GrassCover (units from the eye, 0 = off; kCoverPS, CompositeM)
		float colorHigh[4];     // rgb = the colour at the layer's top, w = relief gain (CompositeOut; ini [Render] ReliefGain, 0 = off)
		float distFog[4];       // near, far, power, max
		float distColorNear[4];
		float distColorFar[4];
		float posAdjust[4];     // camera position (world)
		float heightFog[4];     // x = density per unit at the base (0 = off), y = 1 / falloff (units), z = base z, w = max distance
		float farFog[4];        // x = strength (0 = off; the radiation-fog conditions), y = ground z used without a land table (the ground under the player), z = base thickness, w = noise amount
		float farNoise[4];      // x, y = wind travel (units) as CSColumn Iter.zw, z = time (s) as Iter.y, w = a full-resolution pixel row's angle (radians; 0 = kFarPixel)
		float sunDir[4];        // xyz = unit direction toward the sun (world), w = glow strength (0 = off; day and sun height applied)
		float sunColor[4];      // rgb = sunlight colour (working space), w = anisotropy g
		float poolGrid[4];      // coarse grid (far pools): origin x, y (world), cell, size; t5
		float poolFar[4];       // x = pool density per metre x strength (0 = off), y = depth cap, z = highest lid z (world)
		float landGrid[4];      // land table (far layer): origin x, y (world), cell, size; t6
		float tune[4];          // tests (2026-11-16, all 0 = as before): x = OwnRayTaps, y = UpsampleKernel, z = FogBlurRadius, w = FogBlurTolerance
		float prevViewProj[16]; // kTemporalPS: last drawn frame's viewProj (the same matrix choice as invViewProj inverts)
		float prevPos[4];       // kTemporalPS: xyz = this PosAdjust - last frame's (to turn this frame's relative positions into last frame's), w = 1 if the history is usable
		float temporal[4];      // kTemporalPS (ini [Render] TemporalBlend etc.): x = weight of this frame (0 = off), y = distance tolerance, z = 1 clamp history to the neighbours
		float prevEye[4];       // kTemporalPS: xyz = last frame's eye relative to last frame's PosAdjust (from the inverse of prevViewProj)
		float tune2[4];         // tests (2026-11-16, all 0 = as before): x = NoisePattern (1 = golden-ratio phase), y = that phase (0 - 1), z = DepthPick (1 = checkerboard min/max)
		float sunShadow[4];     // SunShadow (2026-11-17): x = strength (0 = off: the brightness is the height fraction, as before), y = blend toward
		                        // the sunlit look (0 - 1: the sun's reach on the layer x day), z = lowest sun slant (sin of the elevation), w = 1 if SunCols (t12) is this frame's
		float shadowProj0[16];  // the game's sun shadow, cascade 0: absolute world position (row vector) -> shadow map uv and depth (raw lightTransform)
		float shadowProj1[16];  // cascade 1
		float shadowSplit[4];   // x, y = cascade 0 and 1 end distances (units from the eye), z, w = their slices in the shadow map array (t13)
		float shadowUse[4];     // x = 1 if the sun shadow map is valid this frame, y = depth bias, z = ShadowMatrixMode (diagnosis), w unused
		float lightPos[16][4];  // LocalLights: xyz = the light relative to PosAdjust, w = radius (units)
		float lightCol[16][4];  // rgb = colour x fade x LOD dimmer (working space), w unused
		float lightInfo[4];     // x = lights in use (0 = off), y = strength x day dimming (ini [Render] LocalLightStrength), z = LocalLightReach
		float misc[4];          // x = BankAmount (0 = off; the far tier's banks, as CSColumn's), y, z, w unused
	};
	static_assert(sizeof(RayCB) == 16 * 77);
	inline constexpr int kMaxLocalLights = 16;

	// LocalLights: how much fog a light stands in, against the fog at the ground: the layer's profile (ProfileDensity's
	// shape, exp(-a h / H)) at the light's height h over the ground, 1 at or under the ground, 0 from the layer's top H
	// up. A light scatters only where there is fog round it (H-ll4, in game 2026-11-17: the torch in the player's hand,
	// always next to the camera, tinted the fog in every direction, as the halo is integrated along the whole ray).
	inline float LightFogWeight(float a_h, float a_H, float a_a)
	{
		if (!(a_H > 1.0f)) return 0.0f;
		if (a_h <= 0.0f) return 1.0f;
		return a_h < a_H ? std::exp(-std::max(a_a, 1e-3f) * a_h / a_H) : 0.0f;
	}
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
	float4 LandG;      // far layer: land table origin x, y, cell, size (cell 0 = none)
	float4 Tune;       // tests: x = OwnRayTaps (0 = off), y = UpsampleKernel (0 = the 3x3 window from floor, 1 = tent), z = FogBlurRadius (kBlurPS), w = FogBlurTolerance
	row_major float4x4 PrevViewProj;   // kTemporalPS: last drawn frame's viewProj
	float4 PrevPos;    // kTemporalPS: xyz = PosAdjust now - PosAdjust then, w = 1 if the history is usable
	float4 TemporalP;  // kTemporalPS: x = this frame's weight (0 = off), y = distance tolerance, z = 1 clamp history
	float4 PrevEye;    // kTemporalPS: xyz = last frame's eye relative to last frame's PosAdjust
	float4 Tune2;      // tests: x = NoisePattern (1 = IGN plus golden-ratio phase .y), z = DepthPick (1 = checkerboard min/max)
	float4 SunShadowP; // SunShadow: x = strength (0 = off), y = blend toward the sunlit look, z = lowest sun slant, w = 1 if SunCols is valid
	row_major float4x4 ShadowProj0;   // the game's sun shadow cascades: float4(absolute world, 1) x M = (uv, depth)
	row_major float4x4 ShadowProj1;
	float4 ShadowSplit; // x, y = cascade end distances, z, w = their slices in SunShadowMap
	float4 ShadowUse;   // x = 1 valid, y = depth bias, z = ShadowMatrixMode (bit 0: M x v instead of v x M; bit 1: relative to PosAdjust)
	float4 LightPos[16];   // LocalLights: xyz relative to PosAdjust, w = radius
	float4 LightCol[16];   // rgb colour x fade x dimmer
	float4 LightInfo;      // x = count (0 = off), y = strength x day dimming, z = reach (radius multiplier)
	float4 Misc;           // x = BankAmount (0 = off)
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
Texture2D<float4> SunCols    : register(t12);  // SunShadow: optical depth toward the sun at 0, 1/3, 2/3, 1 of the cell's top (kSunColsCS)
Texture2DArray<float> SunShadowMap : register(t13);   // the game's sun shadow map array (depthStencils[kSHADOWMAPS]); read only while ShadowUse.x = 1
SamplerState      LinearClamp : register(s0);
SamplerComparisonState ShadowCmp : register(s1);      // less-equal comparison, linear
static const float kUpperSteepness = 1.0;      // gentler than the low layer: pooled fog and steam keep more density up high

// Interleaved gradient noise (Jimenez 2014): per-pixel start offset hides the banding of few steps. FogP.z is the
// frame number mod 64, so the pattern stays exact however long the game runs.
float Ign(float2 p) { return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715)))); }

// The march's start offset (0 - 1) for low-res position p. NoisePattern 0: IGN moved by 5.588238 per frame (frame mod 64),
// as before. NoisePattern 1 (2026-11-16 test): the same IGN pattern, its value shifted by a golden-ratio phase that the CPU
// steps by 0.618034 a frame (Tune2.y, kept exact over long sessions): each texel's offsets over consecutive frames spread
// evenly over 0 - 1 (a low-discrepancy sequence), which the temporal accumulation then averages.
float MarchJitter(float2 p)
{
	return Tune2.x > 0.5 ? frac(Ign(p) + Tune2.y) : Ign(p + 5.588238 * FogP.z);
}

// The full-res pixel whose ray the low-res texel at centre a_low marches. DepthPick 0: the pixel at its block's lower
// right (2 x 2 at scale 0.5), as before. DepthPick 1 (2026-11-16 test, Pesce 2016 "downsampled effects with depth-aware
// upsampling"): from the block's pixels [floor(t z), floor((t + 1) z)) the one with the smallest depth value where t.x + t.y
// is even and the largest where odd (nearest and farthest, or the other way round with a reversed depth buffer: either
// works), so every 2 x 2 texel group tends to hold both a leaf and what is behind it; the
// composite already picks taps by distance. The block can be 5 pixels wide where z is not a whole number. Ties (the sky,
// any block at one depth) keep the lower-right pixel, which the temporal pass's NearestMarchTexel assumes (Codex review).
int2 MarchSourcePixel(float2 a_low)
{
	const int2 fixedPx = int2(min(a_low * LowRes.z, DepthP.zw - 1.0));
	if (!(Tune2.z > 0.5)) return fixedPx;
	const int2 t = int2(a_low);
	const int2 lo = min(int2(floor(float2(t) * LowRes.z)), int2(DepthP.zw) - 1);
	const int2 hi = max(min(int2(floor(float2(t + 1) * LowRes.z)), int2(DepthP.zw)) - 1, lo);
	const bool wantMin = ((t.x + t.y) & 1) == 0;
	int2  best = clamp(fixedPx, lo, hi);
	float bestD = SceneDepth.Load(int3(best, 0));
	[loop] for (int y = lo.y; y <= hi.y; ++y) {
		[loop] for (int x = lo.x; x <= hi.x; ++x) {
			const float d = SceneDepth.Load(int3(x, y, 0));
			if (wantMin ? d < bestD : d > bestD) {
				bestD = d;
				best = int2(x, y);
			}
		}
	}
	return best;
}

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

)" R"(
// SunShadow (ini [Render] SunShadow = SunShadowP.x; 0 = off and every caller keeps its old expression, bit for bit):
// the direct sun reaching a fog point is exp(-strength x the optical depth between it and the sun). The march's
// brightness then carries that sunlit share instead of the height fraction (blended by SunShadowP.y, 0 at night and
// with the sun under the horizon line the layer sees, so those keep the old look), and the CPU sets ColorLow to the
// sky-lit (shaded) colour, ColorHigh to the sunlit one. Self-shadowing as in roxorvic's Community Shaders fog branch
// (skyrim-community-shaders fluid-field/fog-v3, GroundFogCS.hlsl: exp(-1.1 x fog above / max(sun z, 0.35)), GPL-3.0,
// used with permission); here the depth toward the sun is marched across the column grid once per texel (kSunColsCS,
// 2D slice model 2026-11-17: the local column alone was 0.11 off the true value at a 5 degree sun, 3 taps toward the sun 0.04).
// Representative height of a step for the light: the middle of its part inside [0, top] (a step's own middle can lie
// above a thin layer it crosses whole; Codex design review 2026-11-17).
float SunRepHeight(float ha, float hb, float top)
{
	return 0.5 * (clamp(min(ha, hb), 0.0, top) + clamp(max(ha, hb), 0.0, top));
}

// Optical depth toward the sun at height h over the field texel under cuv, from SunCols (0, 1/3, 2/3, 1 of the cell's
// top `top`, linear between; above the top it fades out over another third). 0 without SunCols.
float SunTauField(float2 cuv, float h, float top)
{
	if (!(SunShadowP.w > 0.5) || !(top >= 1.0)) return 0.0;
	const float4 s = SunCols.SampleLevel(LinearClamp, cuv, 0);
	const float  f = max(h, 0.0) / top * 3.0;
	if (f >= 3.0) return s.w * saturate(4.0 - f);
	const float u = frac(f);
	if (f < 1.0) return lerp(s.x, s.y, u);
	if (f < 2.0) return lerp(s.y, s.z, u);
	return lerp(s.z, s.w, u);
}

// The brightness a step adds: its height fraction blended toward its sunlit share.
float SunBright(float heightFrac, float tauSun)
{
	return lerp(heightFrac, exp(-SunShadowP.x * max(tauSun, 0.0)), SunShadowP.y);
}

float SunFieldBright(float heightFrac, float2 cuv, float ha, float hb, float top)
{
	return SunBright(heightFrac, SunTauField(cuv, SunRepHeight(ha, hb, top), top));
}

// The far tiers have no column grid: the fog straight above, along the sun's slant (the local column only).
float SunTauVertical(float tauAbove)
{
	return tauAbove / max(SunDir.z, SunShadowP.z);
}

)" R"(
// View depth (along the camera's forward axis) of a point `rel` (relative to PosAdjust): the game splits its shadow
// cascades by it (CS's height fog does the same; Codex design review 2026-11-17: by distance the edges of the screen
// switched cascade earlier than the game). The forward axis is the ray through the screen's centre.
float ViewDepth(float3 rel)
{
	const float3 eye = EyeRel();
	float3 fwd;
	float  d;
	PixelRay(int2(DepthP.zw * 0.5), eye, fwd, d);
	return dot(rel - eye, fwd);
}

// The game's sun shadow at a point `rel` (relative to PosAdjust) at view depth `depth`: 1 lit, 0 in shadow; 1 when
// the map is not valid this frame, past the second cascade or outside the map. Five comparison taps (centre x 4
// and the four neighbours), as roxorvic's SceneSunVisibility (skyrim-community-shaders fluid-field/fog-v3,
// GroundFogCS.hlsl, GPL-3.0, used with permission) and Community Shaders' height fog, which read the same map.
float SceneSunVisibility(float3 rel, float depth)
{
	if (!(ShadowUse.x > 0.5) || !(depth < ShadowSplit.y)) return 1.0;
	const bool   far = depth > ShadowSplit.x;
	// ShadowMatrixMode (diagnosis, in game 2026-11-17 the shadows came out shifted): bit 0 multiplies the other way round,
	// bit 1 leaves the position relative to PosAdjust; 0 is Community Shaders' convention
	const int    mode = (int)ShadowUse.z;
	const float4 wp = float4(rel + ((mode & 2) ? float3(0.0, 0.0, 0.0) : PosAdjust.xyz), 1.0);
	const float3 ls = (mode & 1) ? mul(far ? ShadowProj1 : ShadowProj0, wp).xyz : mul(wp, far ? ShadowProj1 : ShadowProj0).xyz;
	if (any(ls.xy < 0.0) || any(ls.xy > 1.0)) return 1.0;
	const float slice = far ? ShadowSplit.w : ShadowSplit.z;
	uint w, h, n;
	SunShadowMap.GetDimensions(w, h, n);
	const float2 texel = 1.0 / float2(max(w, 1u), max(h, 1u));
	const float  cmp = ls.z - ShadowUse.y;
	float s = 4.0 * SunShadowMap.SampleCmpLevelZero(ShadowCmp, float3(ls.xy, slice), cmp);
	s += SunShadowMap.SampleCmpLevelZero(ShadowCmp, float3(ls.xy + float2(texel.x, 0.0), slice), cmp);
	s += SunShadowMap.SampleCmpLevelZero(ShadowCmp, float3(ls.xy - float2(texel.x, 0.0), slice), cmp);
	s += SunShadowMap.SampleCmpLevelZero(ShadowCmp, float3(ls.xy + float2(0.0, texel.y), slice), cmp);
	s += SunShadowMap.SampleCmpLevelZero(ShadowCmp, float3(ls.xy - float2(0.0, texel.y), slice), cmp);
	return s / 8.0;
}

// LocalLights (ini [Render] LocalLights = LightInfo.x, 0 = off): light the game's nearest point lights (torches, fires)
// scatter toward the eye from the fog along the pixel's ray. A street lamp in fog (the user's photo, 2026-11-17) shows a
// small bright core and a round halo fading smoothly outward, centred on the lamp; read at one point along the ray (the
// fog's mean distance, after roxorvic's LocalLight, skyrim-community-shaders fluid-field/fog-v3, GPL-3.0) it came out as
// a band shaped like the fog layer. So here each light's falloff f(d) = 1 / (1 + k d^2 / R^2) (R = radius x LightInfo.z)
// is integrated along the ray from the eye to the scene point, in closed form: with s along the ray, tc the ray's nearest
// point to the light and h its distance there, f = A / (a^2 + (s - tc)^2), A = R^2 / k, a^2 = A + h^2, so
//   integral = A / a (atan((s1 - tc) / a) - atan((s0 - tc) / a)),
// times sqrt(k) / (sqrt 8 R) (at most pi / sqrt 8 = 1.11 for a ray through the light, falling as 1 / h past it), and
// faded out for rays passing farther than 3 R. k = 64 (kHaloCore; 8 before 2026-11-17): the user asked for a steeper
// inside, a small bright core as in the photo; a ray passing R / 4 off keeps 45 % of the core's light (82 % with 8),
// one passing R off 12 % (33 %); the peak stays 1.11. The fog's own amount (the pixel's alpha) still decides how much of it shows: it is added to
// the field fog's colour, which the blend weights by that alpha. LightInfo.y = strength x the day dimming (CPU).
float LocalHalo(float3 eye, float3 dir, float dist, float3 light, float R)
{
	const float3 d = light - eye;
	const float  tc = dot(d, dir);
	const float  h2 = max(dot(d, d) - tc * tc, 0.0);
	static const float kHaloCore = 64.0;
	const float  A = R * R / kHaloCore;
	const float  a = sqrt(A + h2);
	const float  s1 = min(dist, 1e6);
	const float  through = A / a * (atan((s1 - tc) / a) - atan(-tc / a)) * sqrt(kHaloCore / 8.0) / R;
	return through * (1.0 - smoothstep(R, 3.0 * R, sqrt(h2)));
}

float3 LocalGlow(float3 eye, float3 dir, float dist)
{
	float3    sum = float3(0.0, 0.0, 0.0);
	const int n = min((int)LightInfo.x, 16);
	[loop] for (int i = 0; i < n; ++i) {
		sum += LightCol[i].rgb * LocalHalo(eye, dir, dist, LightPos[i].xyz, max(LightPos[i].w * LightInfo.z, 1.0));
	}
	return sum * LightInfo.y;
}

// Round 5 (grasscover/gates.md sections 9, 9a, 9b; HatchRepro "sink" only, the plugin never defines these): with
// GROUNDFOG_SINK 1 (row S, "sink") or 2 (row V, "drop") the march below is renamed MarchPixelPlain and MarchPixel becomes
// the wrapper after it. Row V moves the march's end point: its PixelRay call reads the lowered point while gDropOn is set.
#ifndef GROUNDFOG_SINK
#define GROUNDFOG_SINK 0
#endif
#ifndef GROUNDFOG_SINK_DIAG
#define GROUNDFOG_SINK_DIAG 0
#endif
#ifndef GROUNDFOG_SINK_W0
#define GROUNDFOG_SINK_W0 0
#endif
#ifndef GROUNDFOG_SINK_GSHIFT
#define GROUNDFOG_SINK_GSHIFT 0.0
#endif
#if GROUNDFOG_SINK
#define MarchPixel MarchPixelPlain
#endif
#if GROUNDFOG_SINK == 2
static bool   gDropOn = false;
static float3 gDropSeg = float3(0.0, 0.0, 0.0);   // the lowered end point from the eye
void SinkPixelRay(int2 px, float3 eye, out float3 dir, out float sceneDist)
{
	PixelRay(px, eye, dir, sceneDist);
	if (gDropOn) {
		sceneDist = length(gDropSeg);
		dir = gDropSeg / max(sceneDist, 1e-3);
	}
}
#define PixelRay SinkPixelRay
#endif

// Returns fog, brightness, mean fog distance, scene distance.
float4 MarchPixel(int2 px, float2 noisePos, int steps)
{
	const float3 eye = EyeRel();
	float3 dir;
	float  sceneDist;
	PixelRay(px, eye, dir, sceneDist);
	const float maxDist = min(sceneDist, Box.z);

	// Clip the ray to where the fog can be. First the box: horizontal = column texture square, vertical = [lowest
	// ground, highest top].
	const float3 boxMin = float3(Field.xy, Box.x) - PosAdjust.xyz - eye;
	const float3 boxMax = float3(Field.xy + Field.z, Box.y) - PosAdjust.xyz - eye;
	const float3 inv = 1.0 / (abs(dir) > 1e-6 ? dir : 1e-6);
	const float3 t0 = boxMin * inv;
	const float3 t1 = boxMax * inv;
	float tEnter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
	float tExit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), maxDist));
	// And to the round fade's outer circle (CSColumn: no fog from r = 0.95 of the half side = 0.475 x Field.z from the
	// field's centre), so the steps are spent where the fog is: from the centre the longest level stretch is 0.475 x
	// Field.z whatever the direction, where the square gave 0.5 to 0.707 x. A ray that misses the circle is not marched.
	// (The sampler's linear filter leaves a trace of the last texels inside the circle up to one texel beyond it: under
	// 0.1 % of the layer at 384 texels, cut here.) GROUNDFOG_NO_FADE_CLIP: the offline check's comparison only.
#ifndef GROUNDFOG_NO_FADE_CLIP
	{
		const float2 oc = PosAdjust.xy + eye.xy - (Field.xy + 0.5 * Field.z);   // the eye from the field's centre
		const float  R = 0.475 * Field.z;
		const float  qa = dot(dir.xy, dir.xy);
		const float  qc = dot(oc, oc) - R * R;
		if (qa > 1e-8) {
			const float qb = dot(oc, dir.xy);
			const float disc = qb * qb - qa * qc;
			if (!(disc > 0.0)) return float4(0.0, 0.0, 0.0, sceneDist);
			const float root = sqrt(disc);
			tEnter = max(tEnter, (-qb - root) / qa);
			tExit = min(tExit, (-qb + root) / qa);
		} else if (!(qc < 0.0)) {   // straight up or down outside the circle
			return float4(0.0, 0.0, 0.0, sceneDist);
		}
	}
#endif
	if (Field.w < 0.5 || tExit <= tEnter) {
		return float4(0.0, 0.0, 0.0, sceneDist);
	}

	// Each step reads the column once (at a jittered point) and integrates the height profile exactly over the
	// step's height span, so a thin layer is caught even when one step is longer than the layer is tall (the box
	// spans the whole terrain height range). Nearly level steps fall back to the midpoint rule.
	const float  stepLen = (tExit - tEnter) / steps;
	const float  jitter = MarchJitter(noisePos);
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
		[branch] if (SunShadowP.x > 0.0) sumBright += absorbed * SunFieldBright((tauLow * brightLow + tauUp * brightUp) / tau, cuv, ha, hb, max(col.x, upH));
		else sumBright += absorbed * (tauLow * brightLow + tauUp * brightUp) / tau;   // brighter toward each layer's top
		sumDist += absorbed * t;
		transmit *= exp(-tau);
	}
	const float fog = 1.0 - transmit;
	const float bright = sumWeight > 1e-5 ? sumBright / sumWeight : 0.0;
	const float meanDist = sumWeight > 1e-5 ? sumDist / sumWeight : sceneDist;
	return float4(fog, bright, meanDist, sceneDist);
}
)" R"(
#if GROUNDFOG_SINK
#undef MarchPixel
#if GROUNDFOG_SINK == 2
#undef PixelRay
#endif
// Round 5 rows S and V (gates.md 9, 9a, 9b). Branches by the scene point's height h over the column ground: sky, unknown
// ground, h < 0, 0 <= h < kSinkEps (ground-like: the ray already ends on the ground) and h >= kSinkH1 keep the march as it
// is; for kSinkEps <= h < kSinkH1 the share is w = 1 - smoothstep(kSinkH0, kSinkH1, h).
// S (GROUNDFOG_SINK 1): the ray is continued from the scene point to the ground behind it (GrassCover's continuation, kCoverPS
// CoverMarch, copied here with a hit flag: the step where the height over the ground reaches 0, with the column ground known
// at every step end, inside the field); with a hit the march is combined once with the continuation as CoverTap does and
// blended by w (fog linear, brightness and fog distance alpha-weighted); without a hit the march stays.
// V (GROUNDFOG_SINK 2): the march's end point is the scene point lowered vertically by w h, the ray eye -> lowered point is
// marched once. Both keep the pixel's own scene distance (w) for the 15 % match.
// The helpers repeat CoverGround / CoverDescent under their own names (GROUNDFOG_COVER_CONST is not in every pass).
// GROUNDFOG_SINK_GSHIFT (rows S+e, S-e) shifts the column ground the sink reads; GROUNDFOG_SINK_W0 forces w to 0 (mutation).
static const float kSinkH0 = 56.0;          // units over the column ground: all of the share up to here (0.8 m) ...
static const float kSinkH1 = 84.0;          // ... none from here (1.2 m)
static const float kSinkEps = 4.0;          // ground-like below this (kCoverEps)
static const int   kSinkSteps = 8;          // as kCoverSteps
static const float kSinkMaxExtra = 4096.0;  // as kCoverMaxExtra
static const float kSinkGrad = 32.0;        // as kCoverGrad
static float4 gSinkInfo = float4(0.0, 0.0, 0.0, 0.0);   // diagnostics: x = 1 hit, 2 no hit, 3 dropped, 0 unchanged; y = w; z = h

// The column ground at world xy as the sink reads it; false outside the field square or where it is unknown.
bool SinkGround(float2 xy, out float a_ground)
{
	const float2 uv = (xy - Field.xy) / Field.z;
	a_ground = 0.0;
	if (any(uv < 0.0) || any(uv > 1.0)) return false;
	const float4 col = Columns.SampleLevel(LinearClamp, uv, 0);
	if (col.w < 0.05) return false;
	a_ground = col.z / col.w + GROUNDFOG_SINK_GSHIFT;
	return true;
}

float4 MarchPixel(int2 px, float2 noisePos, int steps)
{
	gSinkInfo = float4(0.0, 0.0, 0.0, 0.0);
	const float3 eye = EyeRel();
	float3 dir;
	float  dist;
	PixelRay(px, eye, dir, dist);
	const float3 origin = PosAdjust.xyz + eye;
	const float3 p = origin + dir * dist;
	float ground = 0.0;
	float h = 0.0;
	float w = 0.0;
	if (dist < 1e8 && Field.w >= 0.5 && SinkGround(p.xy, ground)) {
		h = p.z - ground;
		gSinkInfo.z = h;
		if (h >= kSinkEps && h < kSinkH1) w = 1.0 - smoothstep(kSinkH0, kSinkH1, h);
	}
#if GROUNDFOG_SINK == 2
#if GROUNDFOG_SINK_W0
	w = 0.0;   // mutation check: nothing is lowered
#endif
	const bool drop = w > 0.0;
	gDropOn = drop;
	gDropSeg = dir * dist - float3(0.0, 0.0, w * h);
	float4 m = MarchPixelPlain(px, noisePos, steps);
	gDropOn = false;
	if (drop) {
		m.w = dist;
		gSinkInfo.xy = float2(3.0, w);
	}
	return m;
#else
	const float4 m = MarchPixelPlain(px, noisePos, steps);
	if (!(w > 0.0)) return m;
	gSinkInfo.y = w;
	gSinkInfo.x = 2.0;
	float gx, gy;
	if (!SinkGround(p.xy + float2(kSinkGrad, 0.0), gx)) gx = ground;
	if (!SinkGround(p.xy + float2(0.0, kSinkGrad), gy)) gy = ground;
	const float descent = dot(float2(gx - ground, gy - ground) / kSinkGrad, dir.xy) - dir.z;
	float len = min(kSinkMaxExtra, Box.z - dist);
	if (descent > 1e-4) len = min(len, 1.5 * h / descent + 64.0);
	if (!(len > 0.0)) return m;
	const float stepLen = len / kSinkSteps;
	const float jitter = MarchJitter(noisePos);
	const float a = FogP.y;
	float transmit = 1.0;
	float sumBright = 0.0;
	float sumDist = 0.0;
	bool  hit = false;
	bool  known = true;
	float ha = h;
	float gPrev = ground;
	[loop] for (int s = 0; s < kSinkSteps; ++s) {
		const float  ta = dist + s * stepLen;
		const float3 wb = origin + dir * (ta + stepLen);
		float        gb;
		if (SinkGround(wb.xy, gb)) {
			gPrev = gb;
		} else {
			known = false;
		}
		const float  hb = wb.z - gPrev;
		const float  t = ta + jitter * stepLen;
		const float2 cuv = (origin.xy + dir.xy * t - Field.xy) / Field.z;
		const float4 col = Columns.SampleLevel(LinearClamp, cuv, 0);
		if (col.w >= 0.05) {
			const float2 up = UpperCols.SampleLevel(LinearClamp, cuv, 0);
			const float  hm = lerp(ha, hb, jitter);
			const float  dzg = hb - ha;
			const float  tauLow = LayerTau(ha, hb, hm, col.x, col.y, col.w, a, stepLen, dzg);
			const float  upH = up.x / col.w;
			const float  tauUp = LayerTau(ha, hb, hm, upH, up.y, 1.0, kUpperSteepness, stepLen, dzg);
			const float  tau = tauLow + tauUp;
			if (tau > 0.0) {
				const float brightLow = clamp(0.5 * (ha + hb), 0.0, max(col.x, 1.0)) / max(col.x, 1.0);
				const float brightUp = clamp(0.5 * (ha + hb), 0.0, max(upH, 1.0)) / max(upH, 1.0);
				const float absorbed = transmit * (1.0 - exp(-tau));
				[branch] if (SunShadowP.x > 0.0) sumBright += absorbed * SunFieldBright((tauLow * brightLow + tauUp * brightUp) / tau, cuv, ha, hb, max(col.x, upH));
				else sumBright += absorbed * (tauLow * brightLow + tauUp * brightUp) / tau;
				sumDist += absorbed * t;
				transmit *= exp(-tau);
			}
		}
		if (hb <= 0.0) {
			hit = known;
			break;
		}
		ha = hb;
	}
	if (!hit) return m;
	gSinkInfo.x = 1.0;
#if GROUNDFOG_SINK_W0
	w = 0.0;   // mutation check: the continuation runs, its share is 0
#endif
	// CoverTap's combination, once, with the original march; then the share w: out.x = lerp(m.x, g.x, w), out.yz =
	// lerp(m.x m.yz, g.x g.yz, w) / out.x (m.yz where out.x is 0), out.w = the own scene distance.
	const float  tm = 1.0 - m.x;
	const float  gxx = 1.0 - tm * transmit;
	const float2 gyz = gxx > 1e-6 ? (m.x * m.yz + tm * float2(sumBright, sumDist)) / gxx : m.yz;
	float4 o = m;
	o.x = m.x + w * (gxx - m.x);
	o.yz = o.x > 0.0 ? lerp(m.x * m.yz, gxx * gyz, w) / o.x : m.yz;
	return o;
#endif
}
#endif
)"

	inline constexpr const char* kMarchPS = GROUNDFOG_MARCH_CORE R"(
float4 main(VSOut i) : SV_TARGET
{
	// The low-res pixel stands for one full-res pixel of its block (MarchSourcePixel: the lower right unless DepthPick).
	const int2 px = MarchSourcePixel(i.pos.xy);
#if GROUNDFOG_SINK_DIAG
	const float4 m = MarchPixel(px, i.pos.xy, (int)Box.w);
	return float4(gSinkInfo.xyz, m.w);
#else
	return MarchPixel(px, i.pos.xy, (int)Box.w);
#endif
}
)";

	// GrassCover (ini [Render] GrassCover = ColorLow.w, units from the eye; 0 = off: the two passes below do not run and
	// CompositeM gives the old result bit for bit). Prototype 2026-11-15 (GroundFog-handoff/grasscover/gates.md). Thin low
	// plants standing in the layer take the fog along their own ray continued to the ground behind them (the amount the
	// ground there has, with no plant in front), by a share k that comes from how much of their neighbourhood is ground:
	// thin plants over visible ground are covered, the inside of a solid thing (rock, trunk, wall, body) has no ground
	// pixels and keeps its own fog except for a rim of kCoverBoxR pixels. Ground pixels beside plants take the same
	// to-ground amount, so the whole band tends to the plant-free fog. ThinFogStart (ThinShare) is separate.
#define GROUNDFOG_COVER_CONST R"(
// Prototype variants (HatchRepro "coverdef NAME=VALUE" compiles with these defined; the plugin never defines them).
#ifndef GROUNDFOG_COVER_BOX_R
#define GROUNDFOG_COVER_BOX_R 3
#endif
#ifndef GROUNDFOG_COVER_FRAC_FULL
#define GROUNDFOG_COVER_FRAC_FULL 0.25
#endif
#ifndef GROUNDFOG_COVER_FRAC_DEAD
#define GROUNDFOG_COVER_FRAC_DEAD 0.0
#endif
#ifndef GROUNDFOG_COVER_TWOSIDED
#define GROUNDFOG_COVER_TWOSIDED 0
#endif
#ifndef GROUNDFOG_COVER_NOISE_SCALE
#define GROUNDFOG_COVER_NOISE_SCALE 0
#endif
#ifndef GROUNDFOG_COVER_NOISE_AMP
#define GROUNDFOG_COVER_NOISE_AMP 0.0
#endif
#ifndef GROUNDFOG_COVER_MASK
#define GROUNDFOG_COVER_MASK 0
#endif
static const float kCoverEps = 4.0;          // a point less than this over the column ground is ground-like (99 % of the land under 20 degrees, eps_terrain.py)
static const float kCoverH0 = 1.25;          // the height gate in layer heights, as kThinH0 / kThinH1 / kThinHFloor
static const float kCoverH1 = 2.0;
static const float kCoverHFloor = 48.0;
static const int   kCoverBoxR = GROUNDFOG_COVER_BOX_R;          // ground fraction over (2 R + 1)^2 full-res pixels: a straight outline keeps all but R px
static const float kCoverFracFull = GROUNDFOG_COVER_FRAC_FULL;  // k of the fraction: saturate((frac - dead) / this)
static const float kCoverFracDead = GROUNDFOG_COVER_FRAC_DEAD;
static const int   kCoverSteps = 8;          // the continuation's steps
static const float kCoverMaxExtra = 4096.0;  // and its longest reach past the scene point (units)
static const float kCoverGrad = 32.0;        // baseline of the local ground plane (units; one column texel at the game's sizes)

// Ground z at world xy from the columns, a_fallback where the ground is unknown.
float CoverGround(float2 xy, float a_fallback)
{
	const float4 col = Columns.SampleLevel(LinearClamp, (xy - Field.xy) / Field.z, 0);
	return col.w >= 0.05 ? col.z / col.w : a_fallback;
}

// The column at world xy: ground z and the height gate's layer height (the low and the tall layer, as MarchPixel reads
// them, at least kCoverHFloor); false where the ground is unknown.
bool CoverColumn(float2 xy, out float a_ground, out float a_Hc)
{
	const float2 cuv = (xy - Field.xy) / Field.z;
	const float4 col = Columns.SampleLevel(LinearClamp, cuv, 0);
	a_ground = 0.0;
	a_Hc = kCoverHFloor;
	if (Field.w < 0.5 || col.w < 0.05) return false;
	const float2 up = UpperCols.SampleLevel(LinearClamp, cuv, 0);
	a_ground = col.z / col.w;
	a_Hc = max(max(col.x, up.x / col.w), kCoverHFloor);
	return true;
}

#if GROUNDFOG_COVER_NOISE_SCALE > 0
// Variant NOISE (round 2, R2-N): a static two-octave value noise of world xy (lattice GROUNDFOG_COVER_NOISE_SCALE units),
// 0 to 1, anchored to the ground point; no time term.
float CoverHash(int2 p)
{
	uint h = uint(p.x) * 0x8da6b343u ^ uint(p.y) * 0xd8163841u;
	h ^= h >> 13;
	h *= 0x2c1b3c6du;
	h ^= h >> 12;
	return h / 4294967295.0;
}
float CoverValueNoise(float2 p)
{
	const float2 i = floor(p);
	const float2 f = p - i;
	const float2 u = f * f * (3.0 - 2.0 * f);
	const int2   c = int2(i);
	return lerp(lerp(CoverHash(c), CoverHash(c + int2(1, 0)), u.x), lerp(CoverHash(c + int2(0, 1)), CoverHash(c + int2(1, 1)), u.x), u.y);
}
float CoverNoise(float2 xy)
{
	const float2 p = xy / GROUNDFOG_COVER_NOISE_SCALE;
	return (2.0 * CoverValueNoise(p) + CoverValueNoise(p * 2.0 + 19.7)) / 3.0;
}
#endif

// How fast a ray along dir comes down toward the local ground plane at xy: height over the ground lost per unit of ray.
float CoverDescent(float2 xy, float a_ground, float3 dir)
{
	const float gx = CoverGround(xy + float2(kCoverGrad, 0.0), a_ground);
	const float gy = CoverGround(xy + float2(0.0, kCoverGrad), a_ground);
	return dot(float2(gx - a_ground, gy - a_ground) / kCoverGrad, dir.xy) - dir.z;
}
)"

	// Low resolution, after the march (same pixel per texel): x = transmittance of the ray's continuation from its scene
	// point to the ground behind it (1 = none), y, z = the continuation's absorbed-weighted brightness and distance sums
	// (absorption counted from the scene point as if the march's transmittance there were 1; CompositeM scales them),
	// w = where the continuation ends (the scene distance when there is none). Continued: points from GrassCover out that
	// stand in the height gate above the ground-like band, inside the round fade.
	inline constexpr const char* kCoverPS = GROUNDFOG_MARCH_CORE GROUNDFOG_COVER_CONST R"(
float4 CoverMarch(int2 px, float2 noisePos)
{
	const float3 eye = EyeRel();
	float3 dir;
	float  dist;
	PixelRay(px, eye, dir, dist);
	const float4 none = float4(1.0, 0.0, 0.0, dist);
	if (!(dist < 1e8) || dist < ColorLow.w) return none;
	const float3 origin = PosAdjust.xyz + eye;
	const float3 p = origin + dir * dist;
	float ground, Hc;
	if (!CoverColumn(p.xy, ground, Hc)) return none;
	const float h = p.z - ground;
	if (h < kCoverEps || h >= kCoverH1 * Hc) return none;                        // ground-like, or above the gate
	if (length((p.xy - Field.xy) / Field.z - 0.5) * 2.0 >= 0.95) return none;   // beyond the round fade: k is 0
	// Reach: until the ray is at or under the ground. On the local ground plane it gets there after h / descent; half as
	// far again and 64 units for curved ground, at most kCoverMaxExtra and inside the march's distance limit.
	const float descent = CoverDescent(p.xy, ground, dir);
	float len = min(kCoverMaxExtra, Box.z - dist);
	if (descent > 1e-4) len = min(len, 1.5 * h / descent + 64.0);
	if (!(len > 0.0)) return none;
	// Each step reads the ground at both ends, so the height over the ground is linear inside it (exact on a plane) and
	// LayerTau integrates the profile over that height span; the step that reaches the ground counts only its part above.
	const float stepLen = len / kCoverSteps;
	const float jitter = MarchJitter(noisePos);
	const float a = FogP.y;
	float transmit = 1.0;
	float sumBright = 0.0;
	float sumDist = 0.0;
	float tEnd = dist + len;
	float ha = h;
	float gPrev = ground;
	[loop] for (int s = 0; s < kCoverSteps; ++s) {
		const float  ta = dist + s * stepLen;
		const float3 wb = origin + dir * (ta + stepLen);
		gPrev = CoverGround(wb.xy, gPrev);
		const float  hb = wb.z - gPrev;
		const float  t = ta + jitter * stepLen;
		const float2 cuv = (origin.xy + dir.xy * t - Field.xy) / Field.z;
		const float4 col = Columns.SampleLevel(LinearClamp, cuv, 0);
		if (col.w >= 0.05) {
			const float2 up = UpperCols.SampleLevel(LinearClamp, cuv, 0);
			const float  hm = lerp(ha, hb, jitter);
			const float  dzg = hb - ha;
			const float  tauLow = LayerTau(ha, hb, hm, col.x, col.y, col.w, a, stepLen, dzg);
			const float  upH = up.x / col.w;
			const float  tauUp = LayerTau(ha, hb, hm, upH, up.y, 1.0, kUpperSteepness, stepLen, dzg);
			const float  tau = tauLow + tauUp;
			if (tau > 0.0) {
				const float brightLow = clamp(0.5 * (ha + hb), 0.0, max(col.x, 1.0)) / max(col.x, 1.0);
				const float brightUp = clamp(0.5 * (ha + hb), 0.0, max(upH, 1.0)) / max(upH, 1.0);
				const float absorbed = transmit * (1.0 - exp(-tau));
				[branch] if (SunShadowP.x > 0.0) sumBright += absorbed * SunFieldBright((tauLow * brightLow + tauUp * brightUp) / tau, cuv, ha, hb, max(col.x, upH));
				else sumBright += absorbed * (tauLow * brightLow + tauUp * brightUp) / tau;
				sumDist += absorbed * t;
				transmit *= exp(-tau);
			}
		}
		if (hb <= 0.0) {
			tEnd = ta + stepLen * saturate(ha / max(ha - hb, 1e-6));
			break;
		}
		ha = hb;
	}
	return float4(transmit, sumBright, sumDist, tEnd);
}

float4 main(VSOut i) : SV_TARGET
{
	const int2 px = MarchSourcePixel(i.pos.xy);   // the pixel kMarchPS marched for this texel
	return CoverMarch(px, i.pos.xy);
}
)";

	// Full resolution, before the composite: 1 where the pixel's own point is ground-like (under kCoverEps over the
	// column ground), else 0 (sky, unknown ground, anything standing).
	inline constexpr const char* kCoverFlagPS = GROUNDFOG_MARCH_CORE GROUNDFOG_COVER_CONST R"(
float main(VSOut i) : SV_TARGET
{
	const int2 px = int2(i.pos.xy);
	const float3 eye = EyeRel();
	float3 dir;
	float  dist;
	PixelRay(px, eye, dir, dist);
	if (!(dist < 1e8)) return 0.0;
	const float3 p = PosAdjust.xyz + eye + dir * dist;
	float ground, Hc;
	const bool known = CoverColumn(p.xy, ground, Hc);
	return known && p.z - ground < kCoverEps ? 1.0 : 0.0;
}
)";

	// Shared by the composite pass and the probe: upsample the low-res march for one full-res pixel, and its colour.
#define GROUNDFOG_COMPOSITE_CORE GROUNDFOG_COVER_CONST R"(
Texture2D<float4> MarchTex   : register(t2);   // fog, brightness, mean distance, scene distance

// Thin low things at a distance lie inside the fog (user 2026-11-14: the fog should cover the far grass's movement;
// plan r12 step 2, the cover part of design B r09). From LowRes.w units out (ini [Render] ThinFogStart; 0 = off, and
// CompositeM then gives the old result bit for bit) a pixel whose opposite low-res taps both lie a little farther
// than itself (a grass blade, a twig, a bush top in front of the ground) takes a share k of the fog of the taps at the
// surface right behind it. Across the straight outline of a big thing (a trunk, a body) one tap of every pair lies on
// the thing, so it keeps its own fog; a convex corner passes one diagonal at most (k <= 0.5, low parts only). The
// march pass and the 15 % match stay as they were (plan gate 0: at opacity 0.43 the upsampling adds no flicker).
static const float kThinRel0 = 0.025;    // a pair counts as behind from 2.5 % farther than the pixel ...
static const float kThinRel1 = 0.06;     // ... and fully from 6 %
static const float kThinRelMax = 0.6;    // up to 60 % farther (fading out from 45 %): the sky, a far valley do not count
static const float kThinAgree = 0.15;    // the pair's two taps agree within this (relative), fully within half of it
static const float kThinHFloor = 48.0;   // the height gate takes the layer as at least this high (units)
static const float kThinH0 = 1.25;       // the pixel's point up to this many layer heights over the column's ground: all
static const float kThinH1 = 2.0;        // none from here

// How far a pair of taps at distances a_dA, a_dB counts as lying behind a pixel at a_dist (0 to 1); a_shift is taken off
// their relative distance first (the level pair's row correction); a_dP = the nearer tap.
float ThinPair(float a_dA, float a_dB, float a_dist, float a_shift, out float a_dP)
{
	a_dP = min(a_dA, a_dB);
	const float rel = a_dP / max(a_dist, 1.0) - 1.0 - a_shift;
	const float agree = abs(a_dA - a_dB) / max(a_dP, 1.0);
	return smoothstep(kThinRel0, kThinRel1, rel) * (1.0 - smoothstep(0.75 * kThinRelMax, kThinRelMax, rel)) *
	       (1.0 - smoothstep(0.5 * kThinAgree, kThinAgree, agree));
}

// k of full-res pixel `full` at scene distance myDist from the 3x3 taps around low-res texel `base`; a_dBehind = the
// distance of the surface behind it. Pairs: the level pair, the vertical pair and the two diagonals; two passing pairs
// give all of it (an isolated blade passes three). k rises from ThinFogStart to twice that, fades out with the field's
// round fade (r 0.85 to 0.95), and is only for points low over the column's ground (kThinH0, kThinH1 layer heights).
float ThinShare(int2 full, float myDist, int2 base, out float a_dBehind)
{
	a_dBehind = myDist;
	float k = 0.0;
	const float kD = smoothstep(LowRes.w, 2.0 * LowRes.w, myDist);
	if (kD > 0.0) {
		float d[9];   // the taps' scene distances, j = (y + 1) * 3 + (x + 1)
		[unroll] for (int j = 0; j < 9; ++j) d[j] = MarchTex.Load(int3(clamp(base + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), int2(LowRes.xy) - 1), 0)).w;
		// The march samples texel c at full-res pixel floor((c + 0.5) z) (kMarchPS), so a tap row or column need not pass
		// through the pixel: the pixel lies between the middle taps' sample row (column) and the next one's, and at scale
		// 0.5 an even pixel row (column) lies 1 px from both. The level pair is taken in the nearer tap row, the vertical
		// pair in the nearer tap column, and when both are as near the pair must pass in both: else across a straight
		// outline one of them can lie wholly off the thing (HatchRepro grass, 2026-11-14: the body's left edge took
		// k 0.47 from a vertical pair in the column 1 px outside it). The diagonals straddle the pixel either way. The
		// level pair is corrected by the distance's rise over the rows between (gRow, relative per low-res row: on level
		// ground a higher row is farther); with both rows taken, a blade read into gRow cannot make either pass (it
		// did with one row: ground beside blades took up to 0.04).
		const float  gRow = (d[1] - d[7]) / (2.0 * max(myDist, 1.0));
		const float2 sM = floor(min((clamp(float2(base), 0.0, LowRes.xy - 1.0) + 0.5) * LowRes.z, DepthP.zw - 1.0));
		const float2 sN = floor(min((clamp(float2(base) + 1.0, 0.0, LowRes.xy - 1.0) + 0.5) * LowRes.z, DepthP.zw - 1.0));
		const float2 toM = float2(full) - sM;   // pixels to the middle taps' sample column, row
		const float2 toN = sN - float2(full);   // and to the next ones'
		float dLM, dLN, dVM, dVN, dD1, dD2;
		const float hLM = ThinPair(d[3], d[5], myDist, toM.y / LowRes.z * gRow, dLM);    // level, middle row (above)
		const float hLN = ThinPair(d[6], d[8], myDist, -toN.y / LowRes.z * gRow, dLN);   // level, next row (below)
		const float hVM = ThinPair(d[1], d[7], myDist, 0.0, dVM);                        // vertical, middle column
		const float hVN = ThinPair(d[2], d[8], myDist, 0.0, dVN);                        // vertical, next column (right)
		const float hD1 = ThinPair(d[0], d[8], myDist, 0.0, dD1);                        // the two diagonals
		const float hD2 = ThinPair(d[2], d[6], myDist, 0.0, dD2);
		const float hL = toM.y < toN.y ? hLM : (toN.y < toM.y ? hLN : min(hLM, hLN));
		const float dL = toM.y < toN.y ? dLM : (toN.y < toM.y ? dLN : min(dLM, dLN));
		const float hV = toM.x < toN.x ? hVM : (toN.x < toM.x ? hVN : min(hVM, hVN));
		const float dV = toM.x < toN.x ? dVM : (toN.x < toM.x ? dVN : min(dVM, dVN));
		// A diagonal counts only beside a level or vertical pair: alone it passes along the 45-degree outline of a big low
		// thing (a box, a rock, a roof edge), whose pixels took the ground's fog (review 2026-11-14; thinprobe).
		const float gD = saturate(hL + hV);
		const float sp = hL + hV + (hD1 + hD2) * gD;
		const float sd = hL * dL + hV * dV + (hD1 * dD1 + hD2 * dD2) * gD;
		k = saturate(0.5 * sp) * kD;
		if (k > 0.0) {
			a_dBehind = sd / sp;
			const float3 eye = EyeRel();
			float3 dir;
			float  dist;
			PixelRay(full, eye, dir, dist);
			const float3 world = PosAdjust.xyz + eye + dir * myDist;
			const float2 cuv = (world.xy - Field.xy) / Field.z;
			const float4 col = Columns.SampleLevel(LinearClamp, cuv, 0);
			if (Field.w > 0.5 && col.w >= 0.05) {
				const float2 up = UpperCols.SampleLevel(LinearClamp, cuv, 0);
				const float  Hc = max(max(col.x, up.x / col.w), kThinHFloor);   // the low and the tall layer, as MarchPixel reads them
				const float  hMe = world.z - col.z / col.w;
				const float  r = length(cuv - 0.5) * 2.0;   // CSColumn's r (WindShaders.h)
				k *= (1.0 - smoothstep(kThinH0 * Hc, kThinH1 * Hc, hMe)) * (1.0 - smoothstep(0.85, 0.95, r));
			} else {
				k = 0.0;
			}
		}
	}
	return k;
}
)" R"(
// GrassCover (see kCoverPS). Both read 0 while unbound, which counts as "no continuation" and "not ground".
Texture2D<float4> CoverTex   : register(t8);   // per low-res texel: continuation transmittance, brightness sum, distance sum, end distance
Texture2D<float>  GroundFlag : register(t9);   // per full-res pixel: 1 = ground-like

// The to-ground amount of a low-res texel from its march value mm and its continuation cv (fog, brightness, mean fog
// distance); a_end = where it ends. A texel without a continuation gives its march value and its scene distance.
float4 CoverTap(float4 mm, float4 cv, out float a_end)
{
	if (!(cv.w > 0.0) || !(cv.x < 1.0)) {
		a_end = mm.w;
		return mm;
	}
	a_end = cv.w;
	const float tm = 1.0 - mm.x;
	float4 g;
	g.x = 1.0 - tm * cv.x;
	g.yz = g.x > 1e-6 ? (mm.x * mm.yz + tm * cv.yz) / g.x : mm.yz;
	g.w = cv.w;
	return g;
}

// m for full-res pixel `full` with GrassCover on: m moves toward the to-ground amount G of the 3 x 3 taps by k. A
// ground-like pixel takes k = distance ramp x round fade (its own amount is already the plant-free one; this replaces the
// plant taps the plain upsampling mixes in). Any other pixel also needs the height gate and the ground fraction of its
// (2 kCoverBoxR + 1)^2 neighbourhood. The taps match within 15 % on where their continuation ends, against this pixel's
// own distance (ground-like) or where its own ray meets the local ground plane (others), weighted as CompositeM's loop.
float4 GrassCoverM(int2 full, float myDist, float2 lowPos, int2 base, float4 m)
{
	const float kD = smoothstep(ColorLow.w, 2.0 * ColorLow.w, myDist);
	if (!(kD > 0.0)) return m;
	const float3 eye = EyeRel();
	float3 dir;
	float  dist;
	PixelRay(full, eye, dir, dist);
	const float3 p = PosAdjust.xyz + eye + dir * myDist;
	float ground, Hc;
	if (!CoverColumn(p.xy, ground, Hc)) return m;
	const float hMe = p.z - ground;
	float k = kD * (1.0 - smoothstep(0.85, 0.95, length((p.xy - Field.xy) / Field.z - 0.5) * 2.0));
	float tRef = myDist;
	const bool groundLike = hMe < kCoverEps;
	if (!groundLike) {
		k *= 1.0 - smoothstep(kCoverH0 * Hc, kCoverH1 * Hc, hMe);
		if (!(k > 0.0)) return m;
		const float descent = CoverDescent(p.xy, ground, dir);
		if (!(descent > 1e-4)) return m;   // its ray does not come down to the ground here
		tRef = myDist + hMe / descent;
		// Prototype: the box is summed here tap by tap; the plugin would blur the flags in two separable passes.
		float nG = 0.0;
		float nAll = 0.0;
		float4 side = 0.0;   // variant TWOSIDED: ground pixels left, right, above, below in the pixel's own row and column
		[loop] for (int by = -kCoverBoxR; by <= kCoverBoxR; ++by) {
			[loop] for (int bx = -kCoverBoxR; bx <= kCoverBoxR; ++bx) {
				const int2 q = full + int2(bx, by);
				if (any(q < 0) || any(q >= int2(DepthP.zw))) continue;
				const float fl = GroundFlag.Load(int3(q, 0));
				nG += fl;
				nAll += 1.0;
				if (by == 0) side.xy += fl * float2(bx < 0 ? 1.0 : 0.0, bx > 0 ? 1.0 : 0.0);
				if (bx == 0) side.zw += fl * float2(by < 0 ? 1.0 : 0.0, by > 0 ? 1.0 : 0.0);
			}
		}
#if GROUNDFOG_COVER_NOISE_SCALE > 0
		const float pattern = GROUNDFOG_COVER_NOISE_AMP * (CoverNoise(p.xy + dir.xy * (hMe / descent)) - 0.5);   // at its ground hit
#else
		const float pattern = 0.0;
#endif
		k *= saturate((nG / max(nAll, 1.0) - kCoverFracDead + pattern) / kCoverFracFull);
#if GROUNDFOG_COVER_TWOSIDED
		// a thin thing has ground on both sides along its row or its column; the outline of a solid thing on one side only
		k *= saturate(max(min(side.x, side.y), min(side.z, side.w)));
#endif
		if (!(k > 0.0)) return m;
	}
	float  sumFog = 0.0;
	float2 sumYZ = 0.0;
	float  wsum = 0.0;
	float  wfog = 0.0;
	bool   cont = false;   // a continued tap took part (else a ground-like pixel's G is m itself)
	[unroll] for (int j = 0; j < 9; ++j) {
		const int2   c = clamp(base + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), int2(LowRes.xy) - 1);
		float        end;
		const float4 g = CoverTap(MarchTex.Load(int3(c, 0)), CoverTex.Load(int3(c, 0)), end);
		const float  rel = abs(end - tRef) / max(tRef, 1.0);
		if (!(rel <= 0.15)) continue;
		const float2 off = float2(c) - lowPos;
		const float  w = (1.0 - rel / 0.15) / (1.0 + dot(off, off));
		sumFog += g.x * w;
		sumYZ += g.yz * (w * g.x);
		wsum += w;
		wfog += w * g.x;
		cont = cont || end != MarchTex.Load(int3(c, 0)).w;
	}
	if (!(wsum > 1e-5) || (groundLike && !cont)) return m;
	const float  gx = sumFog / wsum;
	const float2 gyz = wfog > 1e-6 ? sumYZ / wfog : float2(0.0, tRef);
	const float  wo = (1.0 - k) * m.x;
	const float  wb = k * gx;
	float4 o = m;
	o.yz = wo + wb > 1e-6 ? (m.yz * wo + gyz * wb) / (wo + wb) : m.yz;
	o.x = m.x + k * (gx - m.x);
	o.w = m.w + k * (tRef - m.w);   // CompositeOut runs the far layer, far pools and height fog to here
	return o;
}

#if GROUNDFOG_COVER_MASK
// Round 3 (gates.md section 7; HatchRepro "mask" only, the plugin never defines it): k = the grass mask the harness
// uploads into GroundFlag (1 = grass) and nothing else: no distance ramp, round fade, height gate, ground fraction or
// ground-like class. A mask pixel's reference is its own ground hit on the local ground plane. MASK 1: the 3 x 3
// half-res taps as GrassCoverM; MASK 2: its own full-res to-ground amount (MarchFull, CoverFull: the march and the
// continuation run at full resolution), the single own tap under the same 15 % match.
Texture2D<float4> MarchFull : register(t10);
Texture2D<float4> CoverFull : register(t11);
float4 GrassCoverMask(int2 full, float myDist, float2 lowPos, int2 base, float4 m)
{
	if (!(GroundFlag.Load(int3(full, 0)) > 0.5)) return m;
	const float3 eye = EyeRel();
	float3 dir;
	float  dist;
	PixelRay(full, eye, dir, dist);
	const float3 p = PosAdjust.xyz + eye + dir * myDist;
	float ground, Hc;
	if (!CoverColumn(p.xy, ground, Hc)) return m;
	const float descent = CoverDescent(p.xy, ground, dir);
	if (!(descent > 1e-4)) return m;
	const float tRef = myDist + (p.z - ground) / descent;
	float  sumFog = 0.0;
	float2 sumYZ = 0.0;
	float  wsum = 0.0;
	float  wfog = 0.0;
#if GROUNDFOG_COVER_MASK == 2
	{
		float        end;
		const float4 g = CoverTap(MarchFull.Load(int3(full, 0)), CoverFull.Load(int3(full, 0)), end);
		const float  rel = abs(end - tRef) / max(tRef, 1.0);
		if (rel <= 0.15) {
			const float w = 1.0 - rel / 0.15;
			sumFog = g.x * w;
			sumYZ = g.yz * (w * g.x);
			wsum = w;
			wfog = w * g.x;
		}
	}
#else
	[unroll] for (int j = 0; j < 9; ++j) {
		const int2   c = clamp(base + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), int2(LowRes.xy) - 1);
		float        end;
		const float4 g = CoverTap(MarchTex.Load(int3(c, 0)), CoverTex.Load(int3(c, 0)), end);
		const float  rel = abs(end - tRef) / max(tRef, 1.0);
		if (!(rel <= 0.15)) continue;
		const float2 off = float2(c) - lowPos;
		const float  w = (1.0 - rel / 0.15) / (1.0 + dot(off, off));
		sumFog += g.x * w;
		sumYZ += g.yz * (w * g.x);
		wsum += w;
		wfog += w * g.x;
	}
#endif
	if (!(wsum > 1e-5)) return m;
	const float  gx = sumFog / wsum;
	const float2 gyz = wfog > 1e-6 ? sumYZ / wfog : float2(0.0, tRef);
	float4 o = m;   // GrassCoverM's blend at k = 1
	o.yz = gx > 1e-6 ? gyz : m.yz;
	o.x = m.x + (gx - m.x);
	o.w = m.w + (tRef - m.w);
	return o;
}
#endif
)" R"(
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
	// Thin low things (ThinShare): a share k takes the fog of the taps at the surface behind, weighted as the loop
	// below weights the taps at this pixel's own distance. Skipped whole while ThinFogStart is 0.
	float  k = 0.0;
	float  dBehind = myDist;
	float  sumB = 0.0;
	float2 yzB = 0.0;
	float  wB = 0.0;
	float  wfB = 0.0;
	if (LowRes.w > 0.0 && myDist < 1e8) {
		k = ThinShare(full, myDist, base, dBehind);
		if (k > 0.0) {
			[unroll] for (int jb = 0; jb < 9; ++jb) {
				const int2   c = clamp(base + int2(jb % 3 - 1, jb / 3 - 1), int2(0, 0), int2(LowRes.xy) - 1);
				const float4 mb = MarchTex.Load(int3(c, 0));
				const float  relB = abs(mb.w - dBehind) / max(dBehind, 1.0);
				if (!(relB <= 0.15)) continue;
				const float2 off = float2(c) - lowPos;
				const float  w = (1.0 - relB / 0.15) / (1.0 + dot(off, off));
				sumB += mb.x * w;
				yzB += mb.yz * (w * mb.x);
				wB += w;
				wfB += w * mb.x;
			}
			if (!(wB > 1e-5)) k = 0.0;   // nothing was marched at that distance: as before
		}
	}
	float  sumFog = 0.0;
	float2 sumYZ = 0.0;
	float  wsum = 0.0;
	float  wfog = 0.0;
	float  wsq = 0.0;
	// UpsampleKernel (Tune.y; 0 = as before): 1 centres the 3x3 window on the nearest texel and weighs the taps by a tent
	// that reaches 0 at 1.5 texels, so a tap enters and leaves the window at weight 0 as the pixel crosses a texel
	// border. The old window (from floor, weights 1 / (1 + d^2), never 0) changes by a 0.3-weight tap at every border:
	// a 4-pixel grid, flickering with the texels' frame-to-frame noise (ledger 2026-11-16, H-kernel).
	const bool tent = Tune.y > 0.5;
	const int2 wbase = tent ? int2(floor(lowPos + 0.5)) : base;
	[unroll] for (int y = -1; y <= 1; ++y) {
		[unroll] for (int x = -1; x <= 1; ++x) {
			const int2   c = clamp(wbase + int2(x, y), int2(0, 0), int2(LowRes.xy) - 1);
			const float4 mm = MarchTex.Load(int3(c, 0));
			const float  rel = abs(mm.w - myDist) / max(myDist, 1.0);
			if (!(rel <= 0.15)) continue;
			const float2 off = float2(c) - lowPos;
			const float2 tw = saturate(1.0 - abs(float2(wbase + int2(x, y)) - lowPos) / 1.5);
			const float  w = tent ? (1.0 - rel / 0.15) * (tw.x * tw.y) : (1.0 - rel / 0.15) / (1.0 + dot(off, off));
			sumFog += mm.x * w;
			sumYZ += mm.yz * (w * mm.x);
			wsum += w;
			wfog += w * mm.x;
			wsq += w * w;
		}
	}
	float4 m;
	if (wsum > 1e-5) {
		m.x = sumFog / wsum;
		m.yz = wfog > 1e-6 ? sumYZ / wfog : float2(0.0, myDist);
		m.w = myDist;
		// Few-tap pixels (ini [Render] OwnRayTaps = Tune.x, 0 = off, skipped whole; 2026-11-16 bush flicker): where the
		// taps that match this pixel's distance amount to fewer than about Tune.x equal ones (N_eff = (sum w)^2 / sum w^2),
		// one low-res texel's value shows as a 4x4 block and jumps with it from frame to frame. Such pixels take their own
		// full-resolution ray, faded in over N_eff from Tune.x down to Tune.x - 1 so the switch itself does not flicker.
		if (Tune.x > 0.0 && k < 0.999) {
			const float nEff = wsum * wsum / max(wsq, 1e-12);
			const float own = 1.0 - smoothstep(Tune.x - 1.0, Tune.x, nEff);
			if (own > 0.0) {
				const float4 o = MarchPixel(full, noisePos, (int)Box.w);
				const float  wu = (1.0 - own) * m.x;   // brightness and fog distance weighted by fog amount, as in the loop
				const float  wo = own * o.x;
				m.yz = wu + wo > 1e-6 ? (m.yz * wu + o.yz * wo) / (wu + wo) : m.yz;
				m.x = lerp(m.x, o.x, own);
			}
		}
	} else if (k < 0.999) {
		m = MarchPixel(full, noisePos, (int)Box.w);
	} else {
		m = float4(0.0, 0.0, myDist, myDist);   // all of it comes from behind: its own march would not show
	}
	if (k > 0.0) {
		const float4 b = float4(sumB / wB, wfB > 1e-6 ? yzB / wfB : float2(0.0, dBehind), dBehind);
		const float  wo = (1.0 - k) * m.x;   // brightness and fog distance weighted by fog amount, as in the loop
		const float  wb = k * b.x;
		m.yz = wo + wb > 1e-6 ? (m.yz * wo + b.yz * wb) / (wo + wb) : m.yz;
		m.x = lerp(m.x, b.x, k);
		m.w = lerp(myDist, dBehind, k);   // CompositeOut runs the far layer, far pools and height fog to here
	}
	// GrassCover (ColorLow.w; 0 = off, skipped whole)
#if GROUNDFOG_COVER_MASK
	if (ColorLow.w > 0.0 && myDist < 1e8) m = GrassCoverMask(full, myDist, lowPos, base, m);
#else
	if (ColorLow.w > 0.0 && myDist < 1e8) m = GrassCoverM(full, myDist, lowPos, base, m);
#endif
	// ini DebugView (diagnosis only): every pixel marches its own ray at full resolution, no upsampling; 2 = 64 steps,
	// 3 = RaymarchSteps (Box.w). DebugView 0 minus 3 is the upsampling's share of a difference, 3 minus 2 the step
	// count's. Taken here at the end: as an early return at the top it drew fxc warning X4000 (review 2026-11-14).
	if (FogP.w > 1.5 && FogP.w < 3.5) m = MarchPixel(full, noisePos, FogP.w > 2.5 ? (int)Box.w : 64);
	return m;
}

float3 FogColour(float4 m)
{
	float3 colour = lerp(ColorLow.rgb, ColorHigh.rgb, saturate(m.y));
	const float range = max(DistFog.y - DistFog.x, 1.0);
	const float f = min(DistFog.w, pow(saturate((m.z - DistFog.x) / range), DistFog.z));
	return lerp(colour, lerp(DistColorNear.rgb, DistColorFar.rgb, f), f);
}

// ---- far tier: the ground layer beyond the wind field; its patches and the share it takes over from the field ----
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

// Banks over the far tier: WindShaders.h BankFactor with the far tier's travel and time (FarN.xy, FarN.z = CSColumn's
// Iter.zw, Iter.y); keep the two identical.
float FarBank(float2 worldXY)
{
	if (!(Misc.x > 0.0)) return 1.0;
	const float n = FarNoise((worldXY - FarN.xy) / 8000.0 + FarN.z * float2(0.0006, 0.0004));
	return 1.0 + Misc.x * (lerp(0.45, 1.3, n) - 1.0);
}

// Highest the far layer can stand over its ground: FarP.z, x the banks' top factor while they are on (Codex review
// 2026-11-17: the pieces above FarP.z were cut although the banks raised the layer there).
float FarTopMax()
{
	return Misc.x > 0.0 ? FarP.z * (1.0 + Misc.x * 0.3) : FarP.z;
}

float FarLayerH(float2 worldXY)
{
	if (!(FarP.w > 0.0)) return FarP.z;   // even layer: no noise to evaluate
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

// ---- far layer: the ground layer beyond the wind field, on ground heights read from the land table ----
// It used to lie on a ground plane estimated per pixel from depth neighbours. Its fog then depended on the surface's
// slope, so far meshes showed their facets, depth steps showed as grain and silhouettes as lines (reports 2026-11-10,
// -11, -13; DEBUG-LEDGER.md). Now nothing is guessed from the depth buffer: the ray is walked from the field's inner
// circle to its scene point and the layer is integrated over the ground under it, as the field's march does.
//   - While the land table knows the land (the cells the game has loaded) the ground is the table's, bilinear. The
//     table is as fine as the land itself (LandG.z = 128 units): the coarse pools' table (512) left the layer, which
//     is 60 to 130 units thick, lying off the real ground on rough land (TerrainFogRepro 2026-11-13: white patches up
//     mountain sides, a worse result than the old layer when looking down from a height).
//   - The table's second channel is the share of the layer the land carries there by its slope (1 on gentle land, 0
//     on steep), worked out on the CPU over a wide baseline: fog does not lie on a mountain side. It comes from the
//     table, not from the pixel's depth, so it does not show the mesh.
//   - Beyond the known land the ground runs from the last known height toward the scene point, as far as a gentle
//     slope reaches (kFarRise): land that goes on about level carries the layer on. A scene point the gentle slope
//     does not reach (a mountain side, a far valley, anything standing high) loses it with the gap (kFarOff0,
//     kFarOff1), the sky has none (CompositeOut).
//   - All of it thins out with distance along the ray (FarThinStart, kFarThin1).
//   - Without a table (not filled yet) the ground is the level FarP.y.
// What a pixel gets depends only on where its ray runs. Tried and dropped: a level sheet (lost the fog 80 m out on a
// 5-degree slope, tools/offline/far_haze_section.py); continuing with one plane fitted to the known land (left the
// land beyond 117 m clear on the Whiterun plain); a straight line to the scene point with no limit (fogged the sky
// and every far mountain).
// GPT review 2026-11-13, six findings: the continuation's start is closed in on by bisection so it does not jump by a
// whole piece between neighbouring pixels at the edge of the known land; a ray under the estimated ground counts as
// at the ground only kFarUnder deep, and a piece is split at the ground; the continued part has its own pieces across
// the field's fade ring; the fog's distance for the colour is where the fog is; the brightness takes only the part of
// a piece inside the layer. Still an approximation: the ground between two samples is taken as linear.
Texture2D<float2> LandR : register(t6);   // ground z, slope share; unknown ground -1e30; LandG.z = 0 when unbound
static const float kFarOff0 = 150.0;      // scene point this far off the continued ground: still all of it
static const float kFarOff1 = 900.0;      // and from here none
static const float kFarRise = 0.0875;     // the continued ground rises or falls by at most this (5 degrees)
static const float kFarRiseLayers = 2.0;  // and rises by at most this many layer thicknesses in all
static const float kFarThin0 = 5000.0;    // fog this far along the ray (71 m): still all of it; see FarThinStart
static const float kFarThin1 = 30000.0;   // and from here none (430 m)
static const float kFarPixel = 0.001;     // a pixel's angle when FarN.w does not give it (1080 rows over 59 degrees: 0.00095)
static const float kFarPatch = 900.0;     // the patches' size (FarLayerH)
static const float kFarPatchRows0 = 30.0; // a patch this many pixel rows tall along the ray: still all of its shape
static const float kFarPatchRows1 = 8.0;  // and from here none (a patch is about 60 pixels wide there: it reads as a stripe)
static const float kFarPatchMean = 0.41;  // mean of FarLayerH's shape (two-octave value noise through kShape*, 2M samples)
static const float kFarStep = 256.0;      // piece length over the known land (two land vertices)
static const int   kFarKnownSteps = 64;   // such pieces at most (16384 units: the table's diagonal past the field's inner circle)
static const int   kFarEdgeSearch = 6;    // bisections toward the edge of the known land (4 units)
static const int   kFarRingSteps = 3;     // pieces of the continued ground across the field's fade ring
static const int   kFarBeyondSteps = 3;   // and past it (the patches change along it)

// Where the thinning with distance starts: kFarThin0, or the end of the field's fade ring (0.475 x Field.z) when that
// is further. The field's own fog does not thin with distance, so the far layer and far pools must take over from it
// at full strength at the ring's end whatever the field's size (5000 lay past the ring of an 8192 field only: 3891;
// 12288 ends at 5837, 16384 at 7782). The distance is measured along the ray from the eye, the ring from the field's
// centre: for an eye off the centre, or looking down from above, the fog at the ring's end is that much further
// along the ray and has begun to thin (0.5 % 1000 units past the start, 2 % at 2000: the smoothstep starts flat).
float FarThinStart()
{
	return Field.w > 0.5 ? max(kFarThin0, 0.475 * Field.z) : kFarThin0;
}

// Distance along the ray (world origin e, unit dir) at which it leaves the circle of radius R around the field's
// centre; 0 when it starts outside. a_none = the ray never leaves it (straight up or down inside).
float FarCircleExit(float3 e, float3 dir, float R, out bool a_none)
{
	a_none = false;
	const float2 oc = e.xy - (Field.xy + 0.5 * Field.z);
	const float  c = dot(oc, oc) - R * R;
	if (!(c < 0.0)) return 0.0;
	const float a = dot(dir.xy, dir.xy);
	if (!(a > 1e-8)) {
		a_none = true;
		return 0.0;
	}
	const float b = dot(oc, dir.xy);
	return (-b + sqrt(max(b * b - a * c, 0.0))) / a;
}

// Ground and slope share of the land table at world xy (bilinear); false off the table or next to an unknown vertex.
bool FarGround(float2 xy, out float a_ground, out float a_share)
{
	a_ground = 0.0;
	a_share = 1.0;
	if (!(LandG.z > 0.0) || !(LandG.w >= 2.0)) return false;
	const float2 g = (xy - LandG.xy) / LandG.z;
	const float  last = LandG.w - 1.0;
	if (any(g < 0.0) || any(g > last)) return false;
	const int2   c = clamp(int2(floor(g)), 0, (int)LandG.w - 2);
	const float2 f = saturate(g - float2(c));
	const float2 v00 = LandR.Load(int3(c, 0));
	const float2 v10 = LandR.Load(int3(c + int2(1, 0), 0));
	const float2 v01 = LandR.Load(int3(c + int2(0, 1), 0));
	const float2 v11 = LandR.Load(int3(c + int2(1, 1), 0));
	if (!(min(min(v00.x, v10.x), min(v01.x, v11.x)) > -1e29)) return false;
	const float2 v = lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
	a_ground = v.x;
	a_share = saturate(v.y);
	return true;
}

)" R"(
// One piece of the far layer: the ray from ta to tb over ground ga and gb under its ends (linear between), thickness
// and share at its middle. Heights are signed: the part above the ground integrates the layer's profile, the part
// under it down to the thickest patch's depth (the estimate smooths over a hollow there) counts at the ground's
// density, deeper is not fog (the estimate is wrong there). `under` is that depth: the layer's thickness over the land
// table, 0 over the continued ground (see FarLayerTau).
// Centre of what an evenly dense piece of optical depth t absorbs, as a fraction of its length: 1/t - e^-t / (1 - e^-t),
// 1/2 for a thin piece, toward its front for an opaque one. The series below 0.05 avoids the loss of digits in
// 1 - e^-t (reviewer 2026-11-13).
float FarCentre(float t)
{
	return t < 0.05 ? 0.5 - t / 12.0 : 1.0 / t - exp(-t) / (1.0 - exp(-t));
}

void FarPiece(float3 e, float3 dir, float ta, float tb, float ga, float gb, float keep, float under, inout float tau, inout float sumW, inout float sumB, inout float sumT)
{
	if (!(tb > ta) || !(keep > 0.0)) return;
	const float ha = e.z + dir.z * ta - ga;
	const float hb = e.z + dir.z * tb - gb;
	if (min(ha, hb) >= FarTopMax() || max(ha, hb) <= -under) return;   // most pieces of a ray that looks down end here
	const float2 xy = e.xy + dir.xy * (0.5 * (ta + tb));
	const float len = tb - ta;
	const float dh = abs(ha - hb);
	// The patches seen at a grazing angle far away are squeezed into rows a few pixels tall and many wide: they came
	// out as white stripes across far hills where the even layer is a smooth haze (TerrainFogRepro plains_n
	// 2026-11-13). So the patches are filtered by the ground one pixel row covers along the ray here: the fewer rows
	// a patch is tall, the nearer the layer is to its mean thickness.
	const float pixel = FarN.w > 0.0 ? FarN.w : kFarPixel;   // the CPU's value from the projection and the render height
	const float covered = 0.5 * (ta + tb) * pixel * len / max(dh, 1e-4 * len);
	const float even = smoothstep(kFarPatch / kFarPatchRows0, kFarPatch / kFarPatchRows1, covered);
	const float H = lerp(FarLayerH(xy), FarP.z * lerp(1.0, kFarPatchMean, saturate(FarP.w)), even) * FarBank(xy);
	if (H < 1.0) return;
	const float a = FogP.y;
	const float ca = clamp(ha, 0.0, H);
	const float cb = clamp(hb, 0.0, H);
	float t;
	float u1 = 0.0;   // the part of the piece that holds fog, as fractions of its length
	float u2 = 1.0;
	// Under the ground the density is the ground's, fading to none at `under` deep: a ramp, so that a piece lying
	// about level near that depth does not switch between all and nothing (GPT review 2026-11-13).
	const float rho0 = ProfileDensity(0.0, H, a);
	if (dh > 1e-3 * H) {
		const float inLayer = H * ProfileSpan(ca, cb, H, a);                                        // units of height x relative density
		float below = 0.0;
		if (under > 0.0) {
			const float ua = clamp(ha, -under, 0.0);
			const float ub = clamp(hb, -under, 0.0);
			below = abs((ua - ub) + (ua * ua - ub * ub) / (2.0 * under)) * rho0;
		}
		t = FogP.x * (inLayer + below) * (len / dh) / kUnitsPerMetre;
		u1 = saturate((ha - clamp(ha, -under, H)) / (ha - hb));
		u2 = saturate((ha - clamp(hb, -under, H)) / (ha - hb));
	} else {
		const float h = 0.5 * (ha + hb);
		// max(): the continued ground's calls pass a literal 0, which the compiler flagged here (X4008) though never divided by
		const float rho = h >= 0.0 ? ProfileDensity(h, H, a) : (under > 0.0 ? rho0 * saturate(1.0 + h / max(under, 1e-3)) : 0.0);
		t = FogP.x * rho * (len / kUnitsPerMetre);
	}
	t *= keep * FarP.x;
	// What this piece shows: what it absorbed, by how far along the ray its fog lies (see FarLayerTau: thinner with
	// distance). Brightness and mean distance take the same weights, so a piece thinned to nothing does not colour
	// the rest (GPT review 2026-11-13).
	// Where the piece's fog is seen: the centre of what it absorbs (density taken as even over its fog-bearing part).
	// An opaque piece absorbs at its front; read at its middle, long pieces lost the layer too early and the result
	// changed with how the ray was cut into pieces (reviewer 2026-11-13).
	// The field's share (FarWeight) is read there too, not at the piece's middle: across the field's fade ring the
	// share changes within one piece, and read at the middle the far layer came out up to half again too strong just
	// inside the ring (reviewer 2026-11-13: 0.050 against 0.033). Twice, since the share changes how opaque the piece
	// is and with that where its centre lies.
	float tFog = lerp(ta, tb, lerp(u1, u2, FarCentre(t)));
	float share = FarWeight(e.xy + dir.xy * tFog);
	tFog = lerp(ta, tb, lerp(u1, u2, FarCentre(t * share)));
	share = FarWeight(e.xy + dir.xy * tFog);
	t *= share;
	if (!(t > 0.0)) return;
	const float shown = exp(-tau) * (1.0 - exp(-t)) * (1.0 - smoothstep(FarThinStart(), kFarThin1, tFog));
	sumW += shown;
	[branch] if (SunShadowP.x > 0.0) {
		// the whole layer above (not x the field's share: the field carries the rest of the same fog)
		const float above = FogP.x * H * ProfileSpan(0.5 * (ca + cb), H, H, a) / kUnitsPerMetre * keep * FarP.x;
		sumB += shown * SunBright(0.5 * (ca + cb) / H, SunTauVertical(above));
	} else {
		sumB += shown * (0.5 * (ca + cb) / H);
	}
	sumT += shown * tFog;
	tau += t;
}

// Optical depth of the far layer along the ray (eye relative to PosAdjust, unit dir) up to the scene point at `dist`.
// Returns tau; a_bright = absorption-weighted height fraction h/H for the colour, as the march; a_meanT = the
// absorption-weighted distance of the fog along the ray (for the distance colour, as FarPoolTau).
float FarLayerTau(float3 eye, float3 dir, float dist, out float a_bright, out float a_meanT)
{
	a_bright = 0.7;
	a_meanT = dist;
	if (!(FarP.x > 0.0) || !(FogP.x > 0.0) || !(dist < 1e8) || !(FarP.z >= 1.0)) return 0.0;
	const float3 e = PosAdjust.xyz + eye;
	float t0 = 0.0;
	float tRing = 0.0;
	if (Field.w > 0.5 && Field.z >= 1.0) {
		bool none;
		t0 = FarCircleExit(e, dir, 0.275 * Field.z, none);      // r = 0.55 of the half size: FarWeight is 0 inside
		if (none) return 0.0;
		tRing = FarCircleExit(e, dir, 0.475 * Field.z, none);   // r = 0.95: FarWeight is 1 from here
		if (none) tRing = dist;
	}
	if (!(dist > t0)) return 0.0;
	float tau = 0.0;
	float sumW = 0.0;
	float sumB = 0.0;
	float sumT = 0.0;
	// Known land.
	float tK = t0;
	float gK = FarP.y;
	float sK = 1.0;
	if (FarGround(e.xy + dir.xy * t0, gK, sK)) {
		[loop] for (int k = 0; k < kFarKnownSteps && tK < dist; ++k) {
			const float tN = min(tK + kFarStep, dist);
			float gN, sN;
			if (!FarGround(e.xy + dir.xy * tN, gN, sN)) {
				// The edge of the known land lies inside this piece: close in on it, so that the continuation starts
				// at (nearly) the same place for neighbouring pixels.
				float lo = tK;
				float hi = tN;
				float gLo = gK;
				float sLo = sK;
				for (int b = 0; b < kFarEdgeSearch; ++b) {
					const float mid = 0.5 * (lo + hi);
					float gM, sM;
					if (FarGround(e.xy + dir.xy * mid, gM, sM)) {
						lo = mid;
						gLo = gM;
						sLo = sM;
					} else {
						hi = mid;
					}
				}
				FarPiece(e, dir, tK, lo, gK, gLo, 0.5 * (sK + sLo), FarP.z, tau, sumW, sumB, sumT);
				tK = lo;
				gK = gLo;
				sK = sLo;
				break;
			}
			FarPiece(e, dir, tK, tN, gK, gN, 0.5 * (sK + sN), FarP.z, tau, sumW, sumB, sumT);
			tK = tN;
			gK = gN;
			sK = sN;
		}
	} else {
		gK = FarP.y;
		sK = 1.0;
	}
	// Beyond the known land.
	if (tK < dist) {
		const float len = dist - tK;
		// Upward the continued ground is also held to kFarRiseLayers thicknesses in all: a scene point within the
		// 5-degree reach may be the top of a far tree or wall, not land, and that reach grows with distance (500 units
		// at 230 m), so far things were fogged to their tops (reviewer 2026-11-13). Fog lies in low ground: land that
		// rises further stands out of it.
		const float slope = clamp((e.z + dir.z * dist - gK) / len, -kFarRise, min(kFarRise, kFarRiseLayers * FarP.z / len));
		const float off = abs(e.z + dir.z * dist - (gK + slope * len));
		const float keep = 1.0 - smoothstep(kFarOff0, kFarOff1, off);
		if (keep > 0.0) {
			// The continued ground starts at the last known height. A scene point lower than the gentle slope reaches
			// (a far basin, a steady downhill) has the ground laid through itself instead, the one height known out
			// there, so the ray stays above it: from the last known height the ray ran under the ground for a long way,
			// which counted as fog filled basins with flat white and counted as nothing lost the layer on a downhill
			// (TerrainFogRepro plains_high, HatchRepro -10 degrees, 2026-11-13). A scene point higher than the slope
			// reaches (a wall, a mountain side) keeps the ground from the last known height and stands clear above it.
			const float g0 = min(gK, e.z + dir.z * dist - slope * len);
			// Height above the continued ground is linear in t: only the stretch within the layer's reach matters.
			const float hK = e.z + dir.z * tK - g0;
			const float rate = dir.z - slope;
			float ta = tK;
			float tb = dist;
			if (abs(rate) > 1e-6) {
				const float tTop = tK + (FarTopMax() - hK) / rate;
				const float tBot = tK - hK / rate;
				ta = max(ta, min(tTop, tBot));
				tb = min(tb, max(tTop, tBot));
			} else if (hK >= FarTopMax() || hK < 0.0) {
				ta = tb;
			}
			const float tr = clamp(tRing, ta, tb);
			int k;
			for (k = 0; k < kFarRingSteps; ++k) {
				const float pa = lerp(ta, tr, k / (float)kFarRingSteps);
				const float pb = lerp(ta, tr, (k + 1) / (float)kFarRingSteps);
				FarPiece(e, dir, pa, pb, g0 + slope * (pa - tK), g0 + slope * (pb - tK), keep, 0.0, tau, sumW, sumB, sumT);
			}
			for (k = 0; k < kFarBeyondSteps; ++k) {
				const float pa = lerp(tr, tb, k / (float)kFarBeyondSteps);
				const float pb = lerp(tr, tb, (k + 1) / (float)kFarBeyondSteps);
				FarPiece(e, dir, pa, pb, g0 + slope * (pa - tK), g0 + slope * (pb - tK), keep, 0.0, tau, sumW, sumB, sumT);
			}
		}
	}
	if (sumW > 1e-6) {
		a_bright = sumB / sumW;
		a_meanT = sumT / sumW;
	}
	// The far layer thins out with distance (user 2026-11-13: beside the game's own distance fog a ground layer that
	// stays opaque to the horizon looks foreign): each bit of fog shows by how far along the ray it lies, from all of
	// it at FarThinStart to none at kFarThin1. Near fog still hides far fog in full (the weights apply to what each
	// piece absorbed, not to its density: at a grazing angle a thinned density would still be opaque).
	return -log(max(1.0 - sumW, 1e-6));
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
// linear along a piece, so both parts integrate exactly. Like the far layer it thins out with distance (FarThinStart,
// kFarThin1; weights on what each piece absorbed): the returned optical depth is -log(1 - the weighted sum).
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
		// Thinner with distance, as the far layer (FarPiece, FarLayerTau): what the piece absorbed shows by how far along
		// the ray its fog lies, the centre of what it absorbs over its part under the top (density taken as even there).
		float u1 = 0.0;
		float u2 = 1.0;
		if (dz > 1e-3) {
			const float uTop = saturate((top - zA) / (zB - zA));
			if (zB > zA) u2 = uTop;
			else u1 = uTop;
		}
		const float centre = FarCentre(t);
		const float tFog = lerp(tA, tB, lerp(u1, u2, centre));
		const float shown = exp(-tau) * (1.0 - exp(-t)) * (1.0 - smoothstep(FarThinStart(), kFarThin1, tFog));
		sumW += shown;
		[branch] if (SunShadowP.x > 0.0) {
			// profile part above zMid, plus the even part under the coarse ground up to its top (not x share, as the far layer)
			float above = H >= 1.0 ? FogP.x * (H / kUnitsPerMetre) * PoolF.x * ProfileSpan(clamp(zMid - g, 0.0, H), H, H, aUp) : 0.0;
			above += FogP.x * rho0 * max(floorTop - zMid, 0.0) / kUnitsPerMetre;
			sumB += shown * SunBright(saturate((zMid - floorTop) / max(top - floorTop, 1.0)), SunTauVertical(above));
		} else {
			sumB += shown * saturate((zMid - floorTop) / max(top - floorTop, 1.0));
		}
		sumT += shown * tFog;
		tau += t;
	}
	if (sumW > 1e-6) {
		a_bright = sumB / sumW;
		a_meanT = sumT / sumW;
	}
	return -log(max(1.0 - sumW, 1e-6));   // within FarThinStart the same as tau
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

// The sky (depth at the far plane) is not a scene point: the far layer's continued ground must not run up to it.
// DepthP.y < 0: depth grows with distance (far plane at 1); otherwise it is reversed (far plane at 0).
bool SkyPixel(int2 full)
{
	const float depthRaw = SceneDepth.Load(int3(full, 0));
	return DepthP.y < 0.0 ? depthRaw >= 1.0 - 1e-7 : depthRaw <= 1e-7;
}

// Relief (ini [Render] ReliefGain = ColorHigh.w, 0 = off): CSColumn's height factor's departure s - 1 on the column grid
// (WindSim relief, x known). The march's brightness is the height fraction h / H, so a lowered top keeps the brightness of an
// undisturbed one; measured against the undisturbed top a trough is darker by about G (1 - s), G = 0.06 to 0.09 (design A
// r07; light from above). Unbound reads 0: no shading.
Texture2D<float> ReliefCols : register(t7);

// What the composite writes for m at pixel `full`; alpha 0 where it discards (the probe records the same value).
// Layered front to back: wind field fog, far ground layer, height fog: alpha = 1 - (1 - aF)(1 - aG)(1 - aH).
float4 CompositeOut(float4 m, int2 full)
{
	const float aF = saturate(m.x);
	const float3 eye = EyeRel();
	float3 dir;
	float  dist;
	PixelRay(full, eye, dir, dist);
	// ThinFogStart: a thin low thing takes the media behind it along its own ray, so the far layer, far pools and height
	// fog run to CompositeM's m.w (the surface behind it; dist for every other pixel, and always while the setting is 0).
	const float distB = (LowRes.w > 0.0 || ColorLow.w > 0.0) && m.w > dist && m.w < 1e8 ? m.w : dist;   // GrassCover likewise
	float brightG = 0.7;
	float meanL = dist;
	float tauL = 0.0;
	if (!SkyPixel(full)) tauL = FarLayerTau(eye, dir, distB, brightG, meanL);
	// Far pools share the far layer's place in the stack (both lie on the ground beyond the field): optical depths
	// add, brightness weighted by them.
	float brightP, meanP;
	const float tauP = FarPoolTau(eye, dir, distB, brightP, meanP);
	const float tauG = tauL + tauP;
	float meanG = meanL;   // where each one's fog lies along the ray (reviewer 2026-11-08, GPT 2026-11-13)
	if (tauP > 0.0) {
		brightG = (tauL * brightG + tauP * brightP) / tauG;
		meanG = (tauL * meanL + tauP * meanP) / tauG;
	}
	const float aG = 1.0 - exp(-tauG);
	const float tauH = HeightFogTau(eye, dir, distB);
	const float aH = 1.0 - exp(-min(tauH, 50.0));
	const float a = 1.0 - (1.0 - aF) * (1.0 - aG) * (1.0 - aH);
	if (!(a >= 1e-3)) return float4(0.0, 0.0, 0.0, 0.0);
	if (FogP.w > 0.5 && FogP.w < 1.5) return float4(1.0, 0.0, 1.0, 0.5 * step(0.01, aF));   // debug: magenta where the field fog is
	float3 cF = FogColour(m);
	if (ColorHigh.w > 0.0 && Field.w > 0.5) {
		// read where this pixel's field fog lies along its ray (m.z, the absorption-weighted distance)
		const float3 pm = PosAdjust.xyz + eye + dir * m.z;
		cF *= clamp(1.0 + ColorHigh.w * ReliefCols.SampleLevel(LinearClamp, (pm.xy - Field.xy) / Field.z, 0), 0.7, 1.2);
	}
	[branch] if (LightInfo.x > 0.5 && aF > 1e-3) cF += LocalGlow(eye, dir, dist);   // LocalLights: halos along the pixel's ray
	const float3 cG = FogColour(float4(aG, brightG, meanG, dist));
	const float3 cH = HeightFogColour(tauH, min(distB, HeightP.w));
	float3 colour = (aF * cF + (1.0 - aF) * aG * cG + (1.0 - aF) * (1.0 - aG) * aH * cH) / a;
	// Backlit glow (photos 2026-11-08: fog toward a low sun glows far brighter than the land): Henyey-Greenstein
	// forward peak over the isotropic level, capped at 1.5 (Blender Cycles with g 0.6: multiple scattering flattens
	// the peak to about 1.6x the plain fog near the sun, 1.3x at 40 degrees), times the sunlight colour. Added once to the mixed colour of
	// all three media (the same as adding it to each, they share the light); alpha unchanged (GPT design review).
	if (SunDir.w > 0.0) {
		const float g = SunColor.w;
		const float c = dot(dir, SunDir.xyz);
		const float hg4pi = (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
		[branch] if (SunShadowP.x > 0.0) {
			// SunShadow: only the sunlit share glows (the media's brightness, mixed as their colours; height fog taken as lit)
			const float lit = (aF * saturate(m.y) + (1.0 - aF) * aG * saturate(brightG) + (1.0 - aF) * (1.0 - aG) * aH) / a;
			colour += SunColor.rgb * SunDir.w * min(max(hg4pi - 1.0, 0.0), 1.5) * lerp(1.0, lit, SunShadowP.y);
		} else {
			colour += SunColor.rgb * SunDir.w * min(max(hg4pi - 1.0, 0.0), 1.5);
		}
	}
	return float4(colour, a);
}

// DebugView 4: the game's sun shadow at every scene surface as grey (white lit, black shadowed; sky white), magenta
// everywhere when the shadow map is not valid this frame (reviewer 2026-11-17: else it read as "all lit"). Its own
// function called before the fog: an early return inside CompositeOut drew fxc warning X4000 (2026-11-17).
float4 ShadowDebugOut(int2 full)
{
	if (!(ShadowUse.x > 0.5)) return float4(1.0, 0.0, 1.0, 1.0);
	const float3 eye = EyeRel();
	float3 dir;
	float  dist;
	PixelRay(full, eye, dir, dist);
	const float v = dist < 1e8 ? SceneSunVisibility(eye + dir * dist, ViewDepth(eye + dir * dist)) : 1.0;
	return float4(v, v, v, 1.0);
}
)"

	// Blur of the low-res march (ini [Render] FogBlurRadius = Tune.z, FogBlurTolerance = Tune.w; 2026-11-16 bush flicker,
	// drawn only while the radius is above 0): each texel's fog amount is averaged with its neighbours (tent weights over
	// (2r+1)^2 texels), brightness and fog distance weighted by fog amount as CompositeM does. The march jitter differs
	// from texel to texel, so the average cancels most of it, and with it the frame-to-frame jumps of single texels.
	// Tolerance > 0 leaves out neighbours whose scene distance differs by more than that share (fading over it); 0 takes
	// them all (fog from behind thin plants may then spread onto them). .w keeps the texel's own scene distance, which
	// CompositeM's depth test reads.
	inline constexpr const char* kBlurPS = R"(
Texture2D<float4> MarchIn : register(t2);
float4 main(VSOut i) : SV_TARGET
{
	const int2   p = int2(i.pos.xy);
	const float4 c = MarchIn.Load(int3(p, 0));
	const int    r = (int)Tune.z;
	const float  tol = Tune.w;
	float  sumX = 0.0;
	float2 sumYZ = 0.0;
	float  wsum = 0.0;
	float  wf = 0.0;
	[unroll] for (int y = -2; y <= 2; ++y) {
		[unroll] for (int x = -2; x <= 2; ++x) {
			if (abs(x) > r || abs(y) > r) continue;
			const int2   q = clamp(p + int2(x, y), int2(0, 0), int2(LowRes.xy) - 1);
			const float4 n = MarchIn.Load(int3(q, 0));
			float w = (1.0 - abs(x) / (r + 1.0)) * (1.0 - abs(y) / (r + 1.0));
			if (tol > 0.0) w *= saturate(1.0 - abs(n.w - c.w) / max(c.w, 1.0) / tol);
			sumX += n.x * w;
			sumYZ += n.yz * (w * n.x);
			wsum += w;
			wf += w * n.x;
		}
	}
	float4 o = c;
	if (wsum > 1e-6) {
		o.x = sumX / wsum;
		o.yz = wf > 1e-6 ? sumYZ / wf : c.yz;
	}
	return o;
}
)";

	// Temporal accumulation of the low-res march (ini [Render] TemporalBlend = TemporalP.x, TemporalTolerance = .y,
	// TemporalClamp = .z; 2026-11-16 bush flicker, drawn only while the weight is above 0). Each texel's march ends at its
	// scene point; that point is projected with last frame's camera to find the texel that marched it then, and the
	// result is lerp(history, this frame, TemporalP.x): the march jitter and the field's frame-to-frame changes average
	// out over about 1 / TemporalP.x frames. The history is not used where the point fell off screen, where the history
	// texel's scene distance differs by more than TemporalP.y (relative: something moved in front, or a thin leaf gave way
	// to the background) or while PrevPos.w is 0 (first frame, a teleport, a size change). With TemporalP.z the history's
	// fog amount is first clamped to the range of this frame's 3x3 neighbours at a similar distance, so fog that really
	// changed (a trail, a blast) is not held back. .w stays this frame's scene distance (CompositeM's depth test).
	inline constexpr const char* kTemporalPS = GROUNDFOG_MARCH_CORE R"(
Texture2D<float4> MarchCur  : register(t2);
Texture2D<float4> MarchHist : register(t3);

// The low-res texel whose march pixel lies nearest to full-res position a_f along one axis: texel t marches the pixel
// floor((t + 0.5) z) (kMarchPS), centred at that + 0.5, for any RaymarchScale (Codex review 2026-11-16).
int NearestMarchTexel(float a_f, float a_z)
{
	const int t0 = (int)floor((a_f - 0.5) / a_z - 0.5);
	int   best = t0;
	float bestD = 1e30;
	[unroll] for (int k = -1; k <= 2; ++k) {
		const float d = abs(floor((t0 + k + 0.5) * a_z) + 0.5 - a_f);
		if (d < bestD) {
			bestD = d;
			best = t0 + k;
		}
	}
	return best;
}

float4 main(VSOut i) : SV_TARGET
{
	const int2   t = int2(i.pos.xy);
	const float4 c = MarchCur.Load(int3(t, 0));
	if (!(PrevPos.w > 0.5) || !(TemporalP.x > 0.0)) return c;
	const int2 px = MarchSourcePixel(i.pos.xy);   // the pixel kMarchPS marched for this texel
	float3       dir;
	float        dist;
	const float3 eye = EyeRel();
	PixelRay(px, eye, dir, dist);
	const float3 pPrev = eye + dir * min(dist, 1.0e6) + PrevPos.xyz;   // the scene point, relative to last frame's PosAdjust
	const float4 h = mul(float4(pPrev, 1.0), PrevViewProj);
	if (!(h.w > 1e-6)) return c;
	const float2 ndc = h.xy / h.w;
	const float2 f = float2((ndc.x * 0.5 + 0.5) * DepthP.z, (0.5 - ndc.y * 0.5) * DepthP.w);   // full-res position (pixel centres at k + 0.5)
	int2 q = int2(NearestMarchTexel(f.x, LowRes.z), NearestMarchTexel(f.y, LowRes.z));
	if (Tune2.z > 0.5) {
		// DepthPick: off screen means no history. The search starts at the texel whose block holds the point (last frame's
		// texels marched any pixel of their blocks, so the nearest fixed march pixel can belong to a neighbour; reviewer
		// 2026-11-16); at the screen edge that is the edge texel.
		if (any(f < 0.0) || any(f >= DepthP.zw)) return c;
		const float2 pf = floor(f);
		int2 tb = int2(floor(pf / LowRes.z));
		tb += int2(floor(float2(tb + 1) * LowRes.z) <= pf);
		tb -= int2(floor(float2(tb) * LowRes.z) > pf);
		q = clamp(tb, int2(0, 0), int2(LowRes.xy) - 1);
	} else if (any(q < 0) || any(q >= int2(LowRes.xy))) {
		return c;
	}
	// The history texel's scene distance was measured from last frame's eye: compare it with this point's distance from
	// there (the sky, 1e9 in both, matches the sky only) (Codex review 2026-11-16).
	const bool  sky = dist >= 1.0e8;
	const float expect = length(pPrev - PrevEye.xyz);
	float4 hs = MarchHist.Load(int3(q, 0));
	if (Tune2.z > 0.5) {
		// DepthPick: the block's texel may have marched the other surface of its block (nearest or farthest). Keep it when
		// its distance is within the tolerance; only otherwise take, of its 8 neighbours, the one whose distance matches
		// best (so smooth ground, where all nine pass, does not slide the history sideways; reviewer 2026-11-16). The
		// tolerance below still decides whether it is used; a NaN distance counts as no match.
		const int2 q0 = q;
		float bestErr = sky ? (hs.w >= 1.0e8 ? 0.0 : 1e30) : abs(hs.w - expect) / max(expect, 1.0);
		if (!(bestErr <= 1e30)) bestErr = 1e30;
		if (!(bestErr <= TemporalP.y)) [unroll] for (int y = -1; y <= 1; ++y) {
			[unroll] for (int x = -1; x <= 1; ++x) {
				if (x == 0 && y == 0) continue;
				const int2 qn = q0 + int2(x, y);
				if (any(qn < 0) || any(qn >= int2(LowRes.xy))) continue;
				const float4 hn = MarchHist.Load(int3(qn, 0));
				const float  err = sky ? (hn.w >= 1.0e8 ? 0.0 : 1e30) : abs(hn.w - expect) / max(expect, 1.0);
				if (err < bestErr) {
					bestErr = err;
					q = qn;
					hs = hn;
				}
			}
		}
	}
	if (sky ? !(hs.w >= 1.0e8) : !(abs(hs.w - expect) / max(expect, 1.0) <= TemporalP.y)) return c;
	float hx = hs.x;
	if (TemporalP.z > 0.5) {
		float lo = c.x;
		float hi = c.x;
		int   others = 0;
		[unroll] for (int y = -1; y <= 1; ++y) {
			[unroll] for (int x = -1; x <= 1; ++x) {
				if (x == 0 && y == 0) continue;
				const float4 n = MarchCur.Load(int3(clamp(t + int2(x, y), int2(0, 0), int2(LowRes.xy) - 1), 0));
				if (abs(n.w - c.w) / max(c.w, 1.0) <= TemporalP.y) {
					lo = min(lo, n.x);
					hi = max(hi, n.x);
					++others;
				}
			}
		}
		// With fewer than two neighbours at this distance (a thin leaf among background texels) the range would be this
		// frame's value alone and the clamp would throw the history away exactly where the flicker is: skip it there
		// (API review 2026-11-16).
		if (others >= 2) hx = clamp(hx, lo, hi);
	}
	const float a = TemporalP.x;
	float4      o;
	o.x = lerp(hx, c.x, a);
	// brightness and fog distance weighted by fog amount, as everywhere else; the history's by the fog it really held
	// (not the clamped amount), so an empty history adds no brightness or distance of its own (Codex review 2026-11-16)
	const float wh = (1.0 - a) * min(hx, hs.x);
	const float wc = a * c.x;
	o.yz = wh + wc > 1e-6 ? (hs.yz * wh + c.yz * wc) / (wh + wc) : c.yz;
	o.w = c.w;
	return o;
}
)";

	inline constexpr const char* kCompositePS = GROUNDFOG_MARCH_CORE GROUNDFOG_COMPOSITE_CORE R"(
float4 main(VSOut i) : SV_TARGET
{
	if (FogP.w > 3.5 && FogP.w < 4.5) return ShadowDebugOut(int2(i.pos.xy));
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
	ProbeOut[b + 5] = FogP.w > 3.5 && FogP.w < 4.5 ? ShadowDebugOut(px) : CompositeOut(c, px);   // as the composite draws it (Codex review 2026-11-17)
	ProbeOut[b + 6] = SceneTex.Load(int3(px, 0));
	ProbeOut[b + 7] = float4(0.0, 0.0, 0.0, 0.0);
	{
		// x = 1 on the sky (the composite gives it no far layer), y = the far layer's optical depth as the composite has it.
		const bool sky = SkyPixel(px);
		float brightG, meanL;
		const float tauG = sky ? 0.0 : FarLayerTau(eye, dir, dist, brightG, meanL);
		ProbeOut[b + 8] = float4(sky ? 1.0 : 0.0, tauG, FarWeight(world.xy), HeightFogTau(eye, dir, dist));
	}
	if (i == 0) {
		ProbeOut[kCols * kRows * kSlots + 0] = float4(eye, Box.x);
		ProbeOut[kCols * kRows * kSlots + 1] = float4(Box.y, Box.z, Box.w, FogP.x);
	}
}
)";

	// SunShadow (ini [Render] SunShadow > 0): per field texel, the optical depth from its column toward the sun at four
	// heights (0, 1/3, 2/3, 1 of the texel's top, the taller of its two layers), marched across the column grid in
	// kSunTaps taps (denser near the texel) up to the highest fog top (SunColsCB Top.x, the march box's top) or Top.y units, whichever comes
	// first. Both layers of every texel the ray passes count, at their own ground. A sun under Sun.w (sin of the
	// elevation) is taken as at Sun.w. The density matches LayerTau's (per world unit FogP.x x W x k x profile / H);
	// ProfileDensity and OneMinusExp are copies of the march's (keep them identical). Output R16G16B16A16F, read by
	// SunTauField. Bound at b1 (kCommon, which the compile prepends, declares RayCB at b0). Cost: texels x 4 x kSunTaps x 2
	// loads; at half the column grid's resolution (Origin.z = 2: SunShadow cost +0.17-0.21 ms in game at full resolution, gate
	// +0.15; 2026-11-17) 192^2 output texels, about 3.5 M.
	struct alignas(16) SunColsCB
	{
		float grid[4];   // x = texels per side, y = texel size (world units), z = profile steepness a (low layer), w = opacity per metre
		float sun[4];    // xyz = unit direction toward the sun, w = lowest sun slant
		float top[4];    // x = highest fog top (world z), y = longest path (units), z = 1: undo the field's round fade (while the far layer carries the rest)
		float origin[4]; // x, y = the grid's origin (world), as the march's Field.xy (unused by the kernel); z = column texels per output texel (0 = 1)
	};
	static_assert(sizeof(SunColsCB) == 64);

	inline constexpr const char* kSunColsCS = R"(
cbuffer SunColsCB : register(b1)   // b1: the plugin prepends kCommon, whose RayCB holds b0
{
	float4 Grid;   // x = texels per side, y = texel size, z = steepness a, w = opacity per metre
	float4 Sun;    // xyz toward the sun, w = lowest slant
	float4 Top;    // x = highest fog top (world z), y = longest path, z = 1 undo the round fade
	float4 Origin;   // z = column texels per output texel (0 = 1)
};
Texture2D<float4>   Columns   : register(t0);   // H, W, ground z x known, known
Texture2D<float2>   UpperCols : register(t1);   // the tall layer: H x known, W x known; unbound = none
RWTexture2D<float4> OutSun    : register(u0);
static const float kUpperSteepness = 1.0;   // RaymarchShaders' kUpperSteepness
static const float kMinSteepness = 1e-3;    // RaymarchShaders' kMinSteepness
static const int   kSunTaps = 12;
static const int   kSunNearTaps = 8;        // of kSunTaps: over the near part of the path, the rest evenly over the far part
                                            // (8 taps with 2 far missed a quarter of the depth over rising ground; 2026-11-17)
static const float kSunRise = 3.0;          // the near part: until the path has risen this many times the texel's top (plus kSunRiseAdd units)
static const float kSunRiseAdd = 32.0;

float OneMinusExp(float x)
{
	return x < 1e-2 ? x * (1.0 - x * (0.5 - x / 6.0)) : 1.0 - exp(-x);
}

float ProfileDensity(float h, float H, float a)
{
	a = max(a, kMinSteepness);
	const float x = max(h / H, 0.0);
	return x < 1.0 ? a * exp(-a * x) / OneMinusExp(a) : 0.0;
}

// Optical depth per world unit at world height z over texel q (0 off the grid, where the column is unknown and under
// its ground: LayerTau has no fog there either; reviewer 2026-11-17). With Top.z = 1 the field's round fade (CSColumn:
// W x edge, edge = 1 - smoothstep(0.55, 0.95, r)) is undone, so the depth is the whole layer's, as the far tiers take it
// (the far layer carries the faded share; reviewer 2026-11-17: else a brighter ring at the fade).
float SunDensityAt(int2 q, float z)
{
	const int n = (int)Grid.x;
	if (any(q < 0) || any(q >= n)) return 0.0;
	const float4 col = Columns.Load(int3(q, 0));
	if (col.w < 0.05) return 0.0;
	const float h = z - col.z / col.w;
	if (h < 0.0) return 0.0;
	float fade = 1.0;
	if (Top.z > 0.5) {
		fade = 1.0 - smoothstep(0.55, 0.95, length((float2(q) + 0.5) / Grid.x - 0.5) * 2.0);
		if (!(fade > 0.0)) return 0.0;
	}
	float d = 0.0;
	if (col.x >= 1.0 && col.y > 0.0) d += col.y * col.w * ProfileDensity(h, col.x, Grid.z) / col.x;
	const float2 up = UpperCols.Load(int3(q, 0));
	const float  upH = up.x / col.w;
	if (upH >= 1.0 && up.y > 0.0) d += up.y * ProfileDensity(h, upH, kUpperSteepness) / upH;
	return Grid.w * d / max(fade, 0.1);
}

// Ground height and top (the taller layer's H) at continuous column-texel coordinate pc (texel centres at i + 0.5),
// bilinear over the known texels around it; false where none of them is known. The half-resolution output texels sit
// between column centres, and the nearest column's ground started the ray at the wrong height on slopes (Codex review
// 2026-11-17).
bool SunColumnAt(float2 pc, out float a_ground, out float a_top)
{
	const int    n = (int)Grid.x;
	const float2 q = pc - 0.5;
	const int2   i0 = int2(floor(q));
	const float2 f = q - float2(i0);
	float        wSum = 0.0, gSum = 0.0, tSum = 0.0;
	[unroll] for (int k = 0; k < 4; ++k) {
		const int2   o = int2(k & 1, k >> 1);
		const int2   c = clamp(i0 + o, 0, n - 1);
		const float4 col = Columns.Load(int3(c, 0));
		const float  w = (o.x ? f.x : 1.0 - f.x) * (o.y ? f.y : 1.0 - f.y) * col.w;   // col.w = known
		if (col.w >= 0.05 && w > 0.0) {
			wSum += w;
			gSum += w * col.z / col.w;
			tSum += w * max(col.x, UpperCols.Load(int3(c, 0)).x / col.w);
		}
	}
	a_ground = wSum > 0.0 ? gSum / wSum : 0.0;
	a_top = wSum > 0.0 ? max(tSum / wSum, 1.0) : 1.0;
	return wSum > 1e-4;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	const int   n = (int)Grid.x;
	const float scale = max(Origin.z, 1.0);
	const int   nOut = (int)ceil(n / scale);
	if ((int)id.x >= nOut || (int)id.y >= nOut) return;
	// the output texel's centre in column texels, by the real ratio n / nOut, so its uv (id + 0.5) / nOut is the same
	// point the march's SunTauField reads (an odd n made a fixed x 2 drift by up to a texel; Codex review 2026-11-17)
	const float2 pc = (float2(id.xy) + 0.5) * ((float)n / (float)nOut);
	float ground, top;
	if (!SunColumnAt(pc, ground, top) || !(Grid.y > 0.0)) {
		OutSun[id.xy] = float4(0.0, 0.0, 0.0, 0.0);
		return;
	}
	float3 v = Sun.xyz;
	v.z = max(v.z, Sun.w);
	v = normalize(v);
	float4 o = float4(0.0, 0.0, 0.0, 0.0);
	[unroll] for (int k = 0; k < 4; ++k) {
		const float z0 = ground + top * (k / 3.0);
		// The path runs to the highest fog top (or Top.y). Its near part, until it has risen kSunRise x this texel's top,
		// gets kSunNearTaps taps at d = near u^2 (u at the midpoints of even pieces, weight 2 near u / n): dense near the
		// start, where the ray is low in the layer and a bank next to the texel shades it (even taps over a long low-sun
		// path stepped over a bank three texels wide), and a thin layer under a box top far above (a valley) is still
		// covered (reviewer 2026-11-17). The rest of the path, which matters where the ground rises toward the sun and
		// the ray stays in the fog (Codex review 2026-11-17), gets the other taps evenly.
		const float len = min(max(Top.x - z0, 0.0) / v.z, Top.y);
		const float nearLen = min(len, (kSunRise * top + kSunRiseAdd) / v.z);
		const int   nearTaps = nearLen < len ? kSunNearTaps : kSunTaps;
		float tau = 0.0;
		[loop] for (int s = 0; s < kSunTaps; ++s) {
			float d, w;
			if (s < nearTaps) {
				const float u = (s + 0.5) / nearTaps;
				d = nearLen * u * u;
				w = 2.0 * nearLen * u / nearTaps;
			} else {
				const float seg = (len - nearLen) / (kSunTaps - kSunNearTaps);
				d = nearLen + (s - nearTaps + 0.5) * seg;
				w = seg;
			}
			const float2 q = pc + v.xy * (d / Grid.y);
			tau += SunDensityAt(int2(floor(q)), z0 + v.z * d) * w;
		}
		o[k] = tau;
	}
	OutSun[id.xy] = o;
}
)";

	// The march and composite code without an entry point, for the offline check's own kernels (RaymarchCheck.cpp).
	inline constexpr const char* kCompositeCore = GROUNDFOG_MARCH_CORE GROUNDFOG_COMPOSITE_CORE;
#undef GROUNDFOG_COMPOSITE_CORE
#undef GROUNDFOG_MARCH_CORE
}
