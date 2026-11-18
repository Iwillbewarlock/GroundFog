// Reproduction attempt for the "hatching / frost on far walls and roofs" report (arca.live b/tullius 188808534,
// 2026-11-11; DEBUG-LEDGER.md). Runs the real march and composite shaders on WARP over a street-level view of a
// wall with a sloped roof on flat ground and measures, per surface, how much the fog differs from pixel to pixel.
// Grass mode: HatchRepro <output folder> grass [quick] [dump] [dumpall] [thin <units>] [oracle <units>] [only <text>] [boxes] [cover <units>] [bigobj] [r2] [turn] [mask]
// (see namespace Grass; build-grass.cmd).
//   Blade-free depth (harness only): [free <units>] [freesrc ref|close] [freepx <n>] [freemut lpx1|noveto|nogate] [order rev] [eye120] [tilt] [post].
// Usage: HatchRepro <output folder> [near plane, default 15] [vertical field of view in degrees, default 60] [roof slope in degrees, default 45] [hill] [hill roughness, default 1]
// With "hill" as the fifth argument the wall and roof are replaced by a faceted hillside (a grid of triangles whose
// slopes scatter about 8 degrees around the given slope, as a terrain or rock mesh does): Nexus reports 2026-11-13,
// "sketched" far slopes on 0.2.0. Extra line per case: how much the fog differs from facet to facet, along the facet
// edges, and from pixel to pixel inside a facet. Dumps fog_<case>.bin for the far-only cases and tris_<dist>.bin
// (uint16 facet number + 1 per pixel, 0 = not the hill).
// The pixel's angular size sets how far apart neighbouring scene points lie: 768 px over 41 degrees is the pixel size of
// 1080p at a 90-degree horizontal field of view, 31 degrees that of 1440p.
// Prints one line per case and surface; writes fog_<case>.bin (kSize x kSize float, fog alpha) and labels_<dist>.bin
// (0 sky, 1 ground, 2 wall, 3 roof) for the cases named in kDump.
//
// Film mode: HatchRepro <output folder> video <old|off|new> <frames>. One scene (the faceted 30-degree hill 12000 units
// away, 60-degree view, 24-bit depth, field and far layer together, patches on) with the fog drifting; writes
// video_<variant>.u8 (frames x kSize x kSize bytes, fog alpha x 255). old = FarP.y is the old shader's FlatLength
// (build against the 0.2.0 header), off = far layer off (0.2.1), new = this header's far layer with the coarse table.
//
// What varies:
//   distance  wall at 1500 (inside the wind field), 4000, 12000, 40000 units; wall and roof sizes scale with it
//   fog       "field" = the marched wind-field fog only, "far" = the far layer only, "both"
//   depth     "z24" = standard depth rounded to 24 bits (the game's D24 buffer), "rev" = reversed depth in float32
//             (as good as exact; float32 standard depth near 1.0 has the same 2^-24 step as D24)
//   frames    kFrames consecutive frame numbers (the march jitter changes per frame): noise of one frame and of
//             their average (what a temporal anti-aliasing pass would leave)

#include "RaymarchShaders.h"
#include "WindSim.h"

#include <DirectXMath.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace
{
	template <class T>
	void SafeRelease(T*& a_p)
	{
		if (a_p) {
			a_p->Release();
			a_p = nullptr;
		}
	}

	constexpr int   kRes = 64;             // wind field cells per side
	constexpr float kExtent = 8192.0f;     // wind field side, units, centred on the camera (the ini default WorldSize)
	constexpr float kOrigin = -4096.0f;
	constexpr int   kHeightSize = 17;
	constexpr int   kCoarseSize = 65;       // the plugin's coarse terrain table (Terrain.h)
	constexpr float kCoarseCell = 512.0f;
	constexpr float kKnownHalf = 10240.0f;  // land the game has loaded around the player (5 x 5 cells)
	float           kGroundTan = 0.0f;      // eighth optional argument: the whole ground rises along +y by this slope (degrees given)
	int             kVideoFrames = 0;       // film mode: frames to write (0 = the normal measurement run)
	int             kVideoVariant = 2;      // 0 old shader (FlatLength), 1 far layer off, 2 new
	constexpr float kVideoDist = 12000.0f;
	constexpr float kVideoFps = 24.0f;
	constexpr float kVideoWind = 90.0f;     // units a second the patches travel (a light breeze)
	bool            kTable = true;          // ninth optional argument 0: no coarse table (the far layer falls back to a level sheet)
	constexpr UINT  kSize = 768;           // 60 degrees over 768 px: 0.078 degrees per pixel (1080p at 90 degrees: 0.055 vertical)
	float           kFovYDeg = 60.0f;      // second optional argument
	float           kNear = 15.0f;         // first optional argument (the game's fNearDistance default is 15)
	constexpr float kFar = 100000.0f;
	float           kRoofDeg = 45.0f;      // third optional argument: the roof's slope (the far layer fades out between 20 and 42 degrees: kFarSlopeFull, kFarSlopeNone)
	bool            kHill = false;         // fifth optional argument "hill"
	float           kHillRough = 1.0f;     // sixth optional argument: multiplies the vertex scatter (a craggy rock: 3 to 5)
	constexpr int   kHillNx = 10;          // quads across
	constexpr int   kHillNy = 8;           // quads up the slope
	constexpr float kEyeZ = 120.0f;
	constexpr float kThickness = 126.0f;   // ini Thickness
	constexpr float kOpacity = 0.18f;      // ini Opacity (the user's value)
	constexpr float kSteepness = 3.0f;     // ini ProfileSteepness
	constexpr int   kSteps = 24;           // ini RaymarchSteps
	constexpr float kFarBase = 0.0f;       // world z of the far haze's base: the ground of every scene here
	float           kFarNoise = 0.0f;      // seventh optional argument: the far haze's patches (ini NoiseAmount; 0 = even sheet)
	constexpr int   kFrames = 8;
	constexpr float kDistances[] = { 1500.0f, 4000.0f, 12000.0f, 40000.0f };
	constexpr const char* kDump[] = { "4000_both_z24", "4000_both_rev", "4000_far_z24", "1500_field_z24" };

	std::vector<float> ReadFloats(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11Texture2D* a_tex, UINT a_w, UINT a_h, int a_ch)
	{
		D3D11_TEXTURE2D_DESC d{};
		a_tex->GetDesc(&d);
		d.Usage = D3D11_USAGE_STAGING;
		d.BindFlags = 0;
		d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ID3D11Texture2D*   st = nullptr;
		std::vector<float> out(static_cast<std::size_t>(a_w) * a_h * a_ch, 0.0f);
		if (FAILED(a_dev->CreateTexture2D(&d, nullptr, &st))) return out;
		a_ctx->CopyResource(st, a_tex);
		D3D11_MAPPED_SUBRESOURCE m{};
		if (SUCCEEDED(a_ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
			for (UINT y = 0; y < a_h; ++y) {
				std::memcpy(&out[static_cast<std::size_t>(y) * a_w * a_ch], static_cast<const std::uint8_t*>(m.pData) + y * m.RowPitch, sizeof(float) * a_w * a_ch);
			}
			a_ctx->Unmap(st, 0);
		}
		SafeRelease(st);
		return out;
	}

	// RayCB.landGrid exists from the far layer's land table on; the 0.2.0 header (the "old" build) has none.
	template <class T>
	void SetLandGrid(T& a_rc, float a_ox, float a_oy, float a_cell, float a_size)
	{
		if constexpr (requires { a_rc.landGrid[0]; }) {
			a_rc.landGrid[0] = a_ox;
			a_rc.landGrid[1] = a_oy;
			a_rc.landGrid[2] = a_cell;
			a_rc.landGrid[3] = a_size;
		}
	}

	bool WriteBin(const std::string& a_path, const void* a_data, std::size_t a_bytes)
	{
		std::ofstream f(a_path, std::ios::binary);
		f.write(static_cast<const char*>(a_data), static_cast<std::streamsize>(a_bytes));
		return static_cast<bool>(f);
	}

	// Root mean square of (pixel - mean of its 3x3 block) over the pixels of one surface whose whole block is on it.
	double HighPassRms(const std::vector<float>& a_img, const std::vector<std::uint8_t>& a_label, std::uint8_t a_which, int& a_n)
	{
		double sum = 0.0;
		a_n = 0;
		for (int y = 1; y + 1 < static_cast<int>(kSize); ++y) {
			for (int x = 1; x + 1 < static_cast<int>(kSize); ++x) {
				double mean = 0.0;
				bool   inside = true;
				for (int dy = -1; dy <= 1 && inside; ++dy) {
					for (int dx = -1; dx <= 1; ++dx) {
						const std::size_t i = static_cast<std::size_t>(y + dy) * kSize + (x + dx);
						if (a_label[i] != a_which) {
							inside = false;
							break;
						}
						mean += a_img[i];
					}
				}
				if (!inside) continue;
				const double d = a_img[static_cast<std::size_t>(y) * kSize + x] - mean / 9.0;
				sum += d * d;
				++a_n;
			}
		}
		return a_n ? std::sqrt(sum / a_n) : 0.0;
	}

	struct Tri
	{
		double a[3], b[3], c[3];
	};

	// Deterministic scatter in [-1, 1] per grid vertex (no random generator: the same hill every run).
	double Scatter(int a_i, int a_j)
	{
		std::uint32_t h = static_cast<std::uint32_t>(a_i) * 73856093u ^ static_cast<std::uint32_t>(a_j) * 19349663u;
		h ^= h >> 13;
		h *= 0x5bd1e995u;
		h ^= h >> 15;
		return (h & 0xFFFFu) / 32767.5 - 1.0;
	}

	// Hillside starting at y = a_dist on the ground: half width 0.25 dist, 0.3 dist deep, mean slope a_tan, vertex heights
	// scattered by 0.0035 dist (rows 0.0375 dist apart: facet slopes about 8 degrees either side of the mean at 30 degrees).
	std::vector<Tri> BuildHill(double a_dist, double a_tan)
	{
		std::vector<double> z(static_cast<std::size_t>(kHillNx + 1) * (kHillNy + 1));
		const double        lx = 0.5 * a_dist / kHillNx, ly = 0.3 * a_dist / kHillNy;
		for (int j = 0; j <= kHillNy; ++j) {
			for (int i = 0; i <= kHillNx; ++i) {
				z[static_cast<std::size_t>(j) * (kHillNx + 1) + i] = a_tan * j * ly + (j == 0 ? 0.0 : Scatter(i, j) * 0.0035 * kHillRough * a_dist);
			}
		}
		auto vertex = [&](int a_i, int a_j, double* a_out) {
			a_out[0] = -0.25 * a_dist + a_i * lx;
			a_out[1] = a_dist + a_j * ly;
			a_out[2] = z[static_cast<std::size_t>(a_j) * (kHillNx + 1) + a_i];
		};
		std::vector<Tri> tris;
		for (int j = 0; j < kHillNy; ++j) {
			for (int i = 0; i < kHillNx; ++i) {
				Tri t1{}, t2{};
				vertex(i, j, t1.a);
				vertex(i + 1, j, t1.b);
				vertex(i + 1, j + 1, t1.c);
				vertex(i, j, t2.a);
				vertex(i + 1, j + 1, t2.b);
				vertex(i, j + 1, t2.c);
				tris.push_back(t1);
				tris.push_back(t2);
			}
		}
		return tris;
	}

	// Ray o + d t against a triangle (Moeller-Trumbore, both faces); t, or -1 for a miss.
	double HitTri(const double* o, const double* d, const Tri& a_t)
	{
		const double e1[3]{ a_t.b[0] - a_t.a[0], a_t.b[1] - a_t.a[1], a_t.b[2] - a_t.a[2] };
		const double e2[3]{ a_t.c[0] - a_t.a[0], a_t.c[1] - a_t.a[1], a_t.c[2] - a_t.a[2] };
		const double p[3]{ d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0] };
		const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
		if (std::fabs(det) < 1e-18) return -1.0;
		const double s[3]{ o[0] - a_t.a[0], o[1] - a_t.a[1], o[2] - a_t.a[2] };
		const double u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) / det;
		if (u < 0.0 || u > 1.0) return -1.0;
		const double q[3]{ s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0] };
		const double v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
		if (v < 0.0 || u + v > 1.0) return -1.0;
		const double t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
		return t > 0.0 ? t : -1.0;
	}

	struct HillStats
	{
		int    facets = 0;          // facets with at least 50 interior pixels
		double facetSpread = 0.0;   // standard deviation of the facets' mean fog
		double facetMin = 0.0, facetMax = 0.0;
		double edgeStep = 0.0;      // mean |pixel - its facet's interior mean| over pixels within 2 px of another facet
		double innerStep = 0.0;     // the same over interior pixels (what a smooth gradient inside a facet gives)
		double innerNoise = 0.0;    // root mean square of (pixel - 3x3 mean) over interior pixels
	};

	// a_tri: facet number + 1 per pixel (0 = not the hill). Interior = the whole 5x5 block is the same facet.
	HillStats MeasureHill(const std::vector<float>& a_img, const std::vector<std::uint16_t>& a_tri, int a_facets)
	{
		const int                 n = static_cast<int>(kSize);
		std::vector<std::uint8_t> kind(a_tri.size(), 0);   // 1 interior, 2 edge band (another facet within 2 px)
		std::vector<double>       sum(static_cast<std::size_t>(a_facets) + 1, 0.0);
		std::vector<int>          cnt(static_cast<std::size_t>(a_facets) + 1, 0);
		for (int y = 2; y + 2 < n; ++y) {
			for (int x = 2; x + 2 < n; ++x) {
				const std::size_t   i = static_cast<std::size_t>(y) * n + x;
				const std::uint16_t id = a_tri[i];
				if (!id) continue;
				bool same = true, hillOnly = true;
				for (int dy = -2; dy <= 2; ++dy) {
					for (int dx = -2; dx <= 2; ++dx) {
						const std::uint16_t o = a_tri[static_cast<std::size_t>(y + dy) * n + (x + dx)];
						same = same && o == id;
						hillOnly = hillOnly && o != 0;
					}
				}
				if (same) {
					kind[i] = 1;
					sum[id] += a_img[i];
					++cnt[id];
				} else if (hillOnly) {
					kind[i] = 2;
				}
			}
		}
		HillStats           out;
		std::vector<double> mean(static_cast<std::size_t>(a_facets) + 1, 0.0);
		double              m1 = 0.0, m2 = 0.0;
		out.facetMin = 1e30;
		out.facetMax = -1e30;
		for (int f = 1; f <= a_facets; ++f) {
			if (cnt[f] < 50) continue;
			mean[f] = sum[f] / cnt[f];
			m1 += mean[f];
			m2 += mean[f] * mean[f];
			out.facetMin = std::min(out.facetMin, mean[f]);
			out.facetMax = std::max(out.facetMax, mean[f]);
			++out.facets;
		}
		if (!out.facets) {
			out.facetMin = out.facetMax = 0.0;
			return out;
		}
		m1 /= out.facets;
		out.facetSpread = std::sqrt(std::max(m2 / out.facets - m1 * m1, 0.0));
		double edge = 0.0, inner = 0.0, noise = 0.0;
		int    nEdge = 0, nInner = 0;
		for (int y = 2; y + 2 < n; ++y) {
			for (int x = 2; x + 2 < n; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * n + x;
				const int         f = a_tri[i];
				if (!kind[i] || cnt[f] < 50) continue;
				const double d = std::fabs(a_img[i] - mean[f]);
				if (kind[i] == 2) {
					edge += d;
					++nEdge;
					continue;
				}
				inner += d;
				++nInner;
				double block = 0.0;
				for (int dy = -1; dy <= 1; ++dy) {
					for (int dx = -1; dx <= 1; ++dx) block += a_img[static_cast<std::size_t>(y + dy) * n + (x + dx)];
				}
				const double hp = a_img[i] - block / 9.0;
				noise += hp * hp;
			}
		}
		out.edgeStep = nEdge ? edge / nEdge : 0.0;
		out.innerStep = nInner ? inner / nInner : 0.0;
		out.innerNoise = nInner ? std::sqrt(noise / nInner) : 0.0;
		return out;
	}
}

// Grass mode (plan r12 step 0a, 2026-11-14): far grass under the fog, with the game's per-frame projection jitter and
// swaying blades, so the half-resolution upsampling's flicker can be told from the scene's own. Every frame gets its
// own depth: Halton(2, 3) jitter in 8 phases (+-0.5 px, the same jittered matrix for the depth and rc.invViewProj, as
// MatrixMode 1 in game), 16 frames, rc.fog[2] = frame. Scenes: flat ground and a 10-degree slope rising away from the
// camera; eye 250 and 400 units over the ground under the camera; blades (seed 1234, vertical rectangles facing the
// camera, width 2 or 5, height 20..100, tops swaying +-1 unit sideways) in the 1500..3000 band, three layouts plus a
// bare control; a trunk (30 wide, 400 tall, 2000 out), a person (capsule 40 wide, 128 tall, an 8-wide arm at 90 sticking
// out 60 from the body's axis, 1800 out) and the player (capsule 40 x 128, 250 out). The camera's pitch keeps the
// player's legs 30 units up 3 degrees inside the bottom edge (the band stays in view in every scene). Fog: the wind field's columns at rest (game defaults: field 12288,
// 384 texels, height cells 64; the round fade starts at 3379 units, beyond the band) or an even layer (H 51.7,
// W 0.7386, known 1), Opacity 0.43 and 0.18, 24 steps. Variants: now (DebugView 0), exact (DebugView 2: every pixel
// marches its own ray at full resolution, 64 steps), exact24 (DebugView 3: the same at 24 steps; now against exact24 is
// the upsampling's share, exact24 against exact the step count's), no fog.
// With "thin <units>" (plan r12 step 2) every frame also renders the half-resolution path with that ThinFogStart
// (rc.lowRes[3]) and each case gets a second line, indented, comparing it with ThinFogStart 0 pixel by pixel; the first
// line is then exactly what the run without it prints. "only <text>": just the scenes whose name holds the text.
// "oracle <units>" (harness only, 2026-11-14): the cover's upper bound. Every blade pixel at least that far out (ray) gets
// exactly the fog of the true ground behind it (k = 1: the same frame rendered without blades, objects kept), all other
// pixels keep now's; its own indented line per case. The lines above stay as they are.
// "boxes" (2026-11-14, ledger "ground side"): four more scenes "<ground>_E<eye>_lowbox", no blades, with two low boxes beside the
// usual objects: a box 120 x 40 x 40 (x, y, z) at (60, 2200) and the same 40 x 40 section turned 45 degrees about the y axis
// (its front face a diamond on screen, 45-degree outlines; top 56.6 up) at (-80, 2400), both under 1.25 x 48 units. With "thin"
// every case also gets a "thin+" line: the unrounded ratios, dBody and the pixel-frames taking more fog (a_thin - a_now > 0.005)
// per object, ground changes away from (beyond 4 px of) and near objects, and a CPU replica's own-ray shares.
// "oraclethin" (diagnostic, with thin and oracle): the oracle's non-blade pixels keep the thin path's alpha instead of now's, so the
// oracle line shows a perfect cover on blade pixels combined with whatever the thin path does to the ground (its "off" column is
// then the thin path's change, not 0).
// "cover <units>" (GrassCover prototype 2026-11-15, grasscover/gates.md): every frame also renders the half-resolution path with that GrassCover
// (rc.colorLow[3], its two extra passes) and each case gets cover / cover+ / coverobj lines against now and the blade-free render. "bigobj": four
// more scenes per ground and eye, "<ground>_E<eye>_bigobj<D>", bare ground with two boxes (70 and 140 tall), a trunk and a body at D = 1000..4000.
// Usage: HatchRepro <output folder> grass [quick] [dump] [dumpall] [thin <units>] [oracle <units>] [only <text>] [boxes] [oraclethin]   (quick: one case and
// the bare control; dump: frame 0 images; dumpall: with thin, every frame's now and thin images and labels, arm, ray)
namespace Grass
{
	using namespace DirectX;

	constexpr float kExtent = 12288.0f;
	constexpr float kOrigin = -0.5f * kExtent;
	constexpr int   kRes = 384;
	constexpr int   kHeightSize = 193;
	constexpr float kBand0 = 1500.0f;
	constexpr float kBand1 = 3000.0f;
	constexpr int   kFrames = 16;
	constexpr int   kPhases = 8;
	constexpr float kEvenH = 51.7f;
	constexpr float kEvenW = 51.7f / 70.0f;
	constexpr float kLeafC = 92.0f;     // composite brightness L = C (1 - a) + kFogC a, sRGB levels (diagnosis 2)
	constexpr float kGroundC = 43.0f;
	constexpr float kFogC = 215.0f;
	constexpr float kSway = 1.0f;
	constexpr std::uint64_t kSeed = 1234;
	constexpr float kBladeH0 = 20.0f;
	constexpr float kBladeH1 = 100.0f;
	constexpr float kBladeSpread = 0.62f;   // blades out to |x| <= this x y (the view's half width is tan 30 = 0.577 y)
	constexpr int   kClump = 30;            // blades per bush
	constexpr float kClumpR = 20.0f;        // bush radius (40 across)
	constexpr float kTrunkX = 250.0f, kTrunkY = 2000.0f, kTrunkR = 15.0f, kTrunkH = 400.0f;
	constexpr float kPersonX = -200.0f, kPersonY = 1800.0f;
	constexpr float kBodyR = 20.0f, kBodyH = 128.0f, kArmR = 4.0f, kArmZ = 90.0f, kArmLen = 60.0f;
	constexpr float kPlayerY = 250.0f;
	constexpr float kAimZ = 30.0f;          // the camera pitch keeps this point of the player (units up) ...
	constexpr float kAimMarginDeg = 3.0f;    // ... this far inside the bottom edge, so its legs stand in the fog
	constexpr int   kOutline = 4;           // S: pixels within this many of an outline
	constexpr int   kWindow = 2;            // band pixels: their 5 x 5 window holds both a blade and the ground
	constexpr int   kNearSearch = 3;        // K: the ground pixel beside a blade, at most this many pixels along the row
	constexpr int   kCalibrations = 5;      // rounds fitting the blade count to the coverage
	enum : std::uint8_t { kSky = 0, kGround = 1, kBlade = 5, kTrunk = 6, kPerson = 7, kPlayer = 8, kBox = 9, kDiamond = 10, kPost = 11,
		kBigBox1 = 12, kBigBox2 = 13, kBigTrunk = 14, kBigBody = 15 };
	enum Layout { kBare, kSparse, kDense, kClumps, kLowBox, kPostScene, kBig1000, kBig2000, kBig3000, kBig4000 };
	constexpr const char* kLayoutName[] = { "bare", "sparse", "dense", "clumps", "lowbox", "post", "bigobj1000", "bigobj2000", "bigobj3000", "bigobj4000" };
	constexpr double      kCoverage[] = { 0.0, 0.15, 0.60, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };   // share of band pixels on blades at frame 0 (clumps: chosen here)
	// GrassCover gate G4 (CLI bigobj, grasscover/gates.md): bare ground and four big objects at one distance D per scene, at
	// x = D times these fractions: a box 100 x 60 x 70 (x, y, z), the same 140 tall, a trunk (capsule r 15, 400 tall) and a
	// body (capsule r 20, 128 tall). The trunk (x 250, y 2000) and person (x -200, y 1800) of every scene stay clear of them.
	constexpr double kBigFrac[4]{ -0.45, -0.25, 0.30, 0.48 };
	constexpr double kBigBoxHalf1[3]{ 50.0, 30.0, 35.0 }, kBigBoxHalf2[3]{ 50.0, 30.0, 70.0 };
	constexpr double kBigTrunkR = 15.0, kBigTrunkH = 400.0, kBigBodyR = 20.0, kBigBodyH = 128.0;
	double BigDistance(int a_layout) { return a_layout >= kBig1000 && a_layout <= kBig4000 ? 1000.0 * (a_layout - kBig1000 + 1) : 0.0; }
	// GrassCover CPU replica (diagnostics only; the shader's constants in RaymarchShaders.h GROUNDFOG_COVER_CONST)
	constexpr float kCoverEps = 4.0f, kCoverH0 = 1.25f, kCoverH1 = 2.0f, kCoverHFloor = 48.0f;
	// The variant knobs (CLI coverdef, the shader's GROUNDFOG_COVER_* defines); the defaults are the plain A-D row.
	int   gCoverBoxR = 3;
	float gCoverFracFull = 0.25f, gCoverFracDead = 0.0f;
	bool  gCoverTwoSided = false;
	bool  gOpacity043Only = false;   // CLI op43: opacity 0.43 only (variant runs)
	constexpr double      kBoxX = 60.0, kBoxY = 2200.0, kBoxHalf[3]{ 60.0, 20.0, 20.0 };   // the low box (lowbox scenes)
	constexpr double      kDiaX = -80.0, kDiaY = 2400.0, kDiaHalf[3]{ 20.0, 20.0, 20.0 };   // the 45-degree box: this section turned about y
	constexpr double      kGainEps = 0.005;   // thin+: an object pixel takes more fog when a_thin - a_now exceeds this
	// Blade-free depth (ledger 2026-11-14 "grass mosaic", scratchpad mosaic/design.md section 2, steps S1 and S2; harness
	// only, CLI free): the depth the fog reads is replaced, on the CPU, by an effective depth in which far thin blades give
	// way to the surface behind them. The march and composite shaders are untouched.
	constexpr float  kFreeWidth = 27.0f;        // source CLOSE: near runs narrower than this many units are filled ...
	constexpr int    kFreeMinPx = 3;            // ... Lpx = clamp(round(kFreeWidth / (z0 pixelAngle)), kFreeMinPx, maxPx)
	// round here = floor(x + 0.5) in float, and the shader of S3 must write it the same way: HLSL's round() takes a half to
	// the even neighbour, which would give another Lpx at a half.
	constexpr int    kFreeMaxPx = 9;            // maxPx (CLI freepx, kFreeMinPx to kFreePxLimit)
	constexpr int    kFreePxLimit = 31;         // the largest freepx taken
	constexpr float  kFreeRel0 = 0.025f;        // gate on rel = zc / z0 - 1: none up to here (kThinRel0) ...
	constexpr float  kFreeRel1 = 0.06f;         // ... all from here (kThinRel1)
	constexpr float  kFreeRelMax0 = 0.45f;      // fading out from here (0.75 kThinRelMax) ...
	constexpr float  kFreeRelMax1 = 0.60f;      // ... to none (kThinRelMax)
	constexpr float  kFreeH0 = 110.0f;          // the pixel's point up to this many units over the column's ground: all ...
	constexpr float  kFreeH1 = 150.0f;          // ... none from here
	constexpr float  kFreeKnown = 0.05f;        // column unknown (col.w below this): none
	constexpr float  kFreeR0 = 0.85f;           // the field's round fade, CSColumn's r ...
	constexpr float  kFreeR1 = 0.95f;
	constexpr float  kFreeVetoAgree = 0.15f;    // tip veto: the pixel m rows below counts when its z is within this of z0 (relative) ...
	// ... and it is "not filled". Reading fixed before any run (the design says only "not filled"): filled = the closing
	// puts it more than kFreeRel0 farther, i.e. its own rel gate would open. A strict "zc > z" would call every pixel of a
	// round object filled (the closing lifts a convex row by a hair), and the veto would never fire on a capsule's top.
	constexpr float  kFreeVetoFilled = kFreeRel0;
	constexpr double kFreeClosedK = 0.99;       // "closed": a blade pixel with k at least this
	constexpr float  kTiltDeg = 10.0f;          // CLI tilt: the ground also rises along +x by this
	constexpr double kPostX = 450.0, kPostY = 2300.0, kPostR = 3.0, kPostH = 150.0;   // CLI post: a 6-unit-wide, 150-high post
	enum FreeSrc { kFreeRef, kFreeClose };
	enum FreeMut { kMutNone, kMutLpx1, kMutNoVeto, kMutNoGate };
	constexpr const char* kFreeSrcName[] = { "ref", "close" };
	constexpr const char* kFreeMutName[] = { "", "-lpx1", "-noveto", "-nogate" };

	struct Rng   // splitmix64: the same blades on every compiler
	{
		std::uint64_t s;
		std::uint64_t Next()
		{
			std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
			z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
			z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
			return z ^ (z >> 31);
		}
		double Uniform() { return static_cast<double>(Next() >> 11) * (1.0 / 9007199254740992.0); }
	};

	double Halton(int a_index, int a_base)
	{
		double f = 1.0, r = 0.0;
		for (int i = a_index; i > 0; i /= a_base) {
			f /= a_base;
			r += f * (i % a_base);
		}
		return r;
	}

	struct Blade
	{
		double x, y, h, phase;
	};

	struct Capsule
	{
		double a[3], b[3], r;
		std::uint8_t label;
	};

	// Blades in the band, uniform over the ground's area; clumps: bush centres so, kClump blades in each.
	std::vector<Blade> MakeBlades(Layout a_layout, int a_count)
	{
		Rng                r{ kSeed };
		std::vector<Blade> out;
		auto               place = [&](double& a_x, double& a_y) {
            a_y = std::sqrt(kBand0 * kBand0 + r.Uniform() * (kBand1 * kBand1 - kBand0 * kBand0));
            a_x = (2.0 * r.Uniform() - 1.0) * kBladeSpread * a_y;
		};
		for (int i = 0; i < a_count; ++i) {
			double x, y;
			place(x, y);
			if (a_layout != kClumps) {
				const double h = kBladeH0 + r.Uniform() * (kBladeH1 - kBladeH0);
				out.push_back({ x, y, h, r.Uniform() });
				continue;
			}
			for (int k = 0; k < kClump; ++k) {
				const double rr = kClumpR * std::sqrt(r.Uniform()), th = 6.283185307179586 * r.Uniform();
				const double h = kBladeH0 + r.Uniform() * (kBladeH1 - kBladeH0);
				out.push_back({ x + rr * std::cos(th), y + rr * std::sin(th), h, r.Uniform() });
			}
		}
		return out;
	}

	// Ray o + d s (|d| = 1) against a capsule; s, or -1 (Quilez's capsule intersection).
	double HitCapsule(const double* o, const double* d, const Capsule& c)
	{
		const double ba[3]{ c.b[0] - c.a[0], c.b[1] - c.a[1], c.b[2] - c.a[2] };
		const double oa[3]{ o[0] - c.a[0], o[1] - c.a[1], o[2] - c.a[2] };
		auto         dot = [](const double* p, const double* q) { return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]; };
		const double baba = dot(ba, ba), bard = dot(ba, d), baoa = dot(ba, oa), rdoa = dot(d, oa), oaoa = dot(oa, oa);
		const double A = baba - bard * bard;
		double       B = baba * rdoa - baoa * bard;
		double       C = baba * oaoa - baoa * baoa - c.r * c.r * baba;
		double       h = B * B - A * C;
		if (h < 0.0) return -1.0;   // misses the infinite cylinder, so the end spheres too
		double y = bard > 0.0 ? -1.0 : baba + 1.0;   // a ray along the axis meets the nearer end
		if (A > 1e-12) {
			const double s = (-B - std::sqrt(h)) / A;
			y = baoa + s * bard;
			if (y > 0.0 && y < baba) return s > 0.0 ? s : -1.0;
		}
		const double* e = y <= 0.0 ? c.a : c.b;
		const double  oc[3]{ o[0] - e[0], o[1] - e[1], o[2] - e[2] };
		B = dot(d, oc);
		C = dot(oc, oc) - c.r * c.r;
		h = B * B - C;
		if (h <= 0.0) return -1.0;
		const double s = -B - std::sqrt(h);
		return s > 0.0 ? s : -1.0;
	}

	// Ray o + d s against a box with centre a_c, half sizes a_h, turned a_ang (radians) about the y axis; s, or -1 (slabs).
	double HitBox(const double* o, const double* d, const double* a_c, const double* a_h, double a_ang)
	{
		const double cs = std::cos(a_ang), sn = std::sin(a_ang);
		const double r[3]{ o[0] - a_c[0], o[1] - a_c[1], o[2] - a_c[2] };
		const double lo[3]{ cs * r[0] + sn * r[2], r[1], -sn * r[0] + cs * r[2] };
		const double ld[3]{ cs * d[0] + sn * d[2], d[1], -sn * d[0] + cs * d[2] };
		double       t0 = -1e30, t1 = 1e30;
		for (int a = 0; a < 3; ++a) {
			if (std::fabs(ld[a]) < 1e-12) {
				if (std::fabs(lo[a]) > a_h[a]) return -1.0;
				continue;
			}
			double ta = (-a_h[a] - lo[a]) / ld[a], tb = (a_h[a] - lo[a]) / ld[a];
			if (ta > tb) std::swap(ta, tb);
			t0 = std::max(t0, ta);
			t1 = std::min(t1, tb);
		}
		if (t0 > t1 || t1 <= 0.0) return -1.0;
		return t0 > 0.0 ? t0 : -1.0;
	}

	struct View
	{
		bool     slope;
		float    eye;    // over the ground under the camera (the ground there is at z 0)
		double   tan;    // ground z = tan y + tanX x
		double   tanX;   // 0 except on the laterally tilted ground (CLI tilt)
		XMMATRIX view;
		XMMATRIX proj;   // unjittered
	};

	struct FrameData
	{
		std::vector<float>        depth;    // D24-rounded standard depth (1 = sky)
		std::vector<std::uint8_t> label;
		std::vector<std::uint8_t> arm;      // 1 where the person's arm is the visible surface (it carries kPerson)
		std::vector<float>        horiz;    // horizontal distance of the hit point (-1 sky)
		std::vector<float>        height;   // hit point over the ground under it
		std::vector<float>        ray;      // distance from the eye along the ray
		XMFLOAT4X4                inv;      // inverse of the jittered viewProj (rc.invViewProj)
		XMFLOAT4X4                vp;
		std::vector<std::uint8_t> grass;      // round 3 (CLI mask): 1 where a blade lies in front of the ground (blades tested against the ground only)
		std::vector<float>        grassDepth; // and that blade's D24 depth (the depth the grass draw writes over a terrain-only buffer)
	};

	bool gMask = false;   // CLI mask (round 3, gates.md section 7): rows M0-M3, k = a grass mask
	bool gSink = false;   // CLI sink (round 5, gates.md section 9): rows S (GROUNDFOG_SINK), C (march phase 0), SC
	bool gSinkW0 = false; // CLI sinkw0: the mutation check, S compiled with GROUNDFOG_SINK_W0 (its share forced to 0)

	View MakeView(bool a_slope, float a_eye, bool a_tilt = false, double a_yawDeg = 0.0)
	{
		View v{};
		v.slope = a_slope;
		v.eye = a_eye;
		v.tan = a_slope ? std::tan(XMConvertToRadians(10.0f)) : 0.0;
		v.tanX = a_tilt ? std::tan(XMConvertToRadians(kTiltDeg)) : 0.0;
		const double aimZ = v.tan * kPlayerY + kAimZ;
		const double down = std::atan2(a_eye - aimZ, static_cast<double>(kPlayerY)) + XMConvertToRadians(kAimMarginDeg) - 0.5 * XMConvertToRadians(60.0f);
		const double   yaw = a_yawDeg * 3.141592653589793 / 180.0;   // round 2 G12 (CLI turn): about the z axis, + turns toward +x
		const XMVECTOR fwd = a_yawDeg == 0.0 ? XMVectorSet(0.0f, static_cast<float>(std::cos(down)), static_cast<float>(-std::sin(down)), 0.0f)
		                                     : XMVectorSet(static_cast<float>(std::sin(yaw) * std::cos(down)), static_cast<float>(std::cos(yaw) * std::cos(down)),
		                                           static_cast<float>(-std::sin(down)), 0.0f);
		v.view = XMMatrixLookToLH(XMVectorZero(), fwd, XMVectorSet(0, 0, 1, 0));
		v.proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(60.0f), 1.0f, kNear, kFar);
		return v;
	}

	// Depth, labels and hit points of one frame: ground, objects, blades (rasterised by their screen boxes).
	void BuildFrame(const View& a_v, int a_frame, const std::vector<Blade>& a_blades, double a_width, FrameData& a_out, bool a_boxes = false, bool a_post = false,
		double a_big = 0.0)
	{
		const std::size_t n = static_cast<std::size_t>(kSize) * kSize;
		XMFLOAT4X4        pf;
		XMStoreFloat4x4(&pf, a_v.proj);
		const int phase = a_frame % kPhases + 1;
		pf._31 += static_cast<float>((Halton(phase, 2) - 0.5) * 2.0 / kSize);
		pf._32 += static_cast<float>((Halton(phase, 3) - 0.5) * 2.0 / kSize);
		const XMMATRIX vp = XMMatrixMultiply(a_v.view, XMLoadFloat4x4(&pf));
		const XMMATRIX invVp = XMMatrixInverse(nullptr, vp);
		XMStoreFloat4x4(&a_out.inv, invVp);
		XMStoreFloat4x4(&a_out.vp, vp);
		const double cam[3]{ 0.0, 0.0, a_v.eye };
		std::vector<double> dirs(n * 3);
		std::vector<double> best(n, -1.0);
		a_out.label.assign(n, kSky);
		a_out.arm.assign(n, 0);
		for (UINT y = 0; y < kSize; ++y) {
			for (UINT x = 0; x < kSize; ++x) {
				XMFLOAT4 q;
				XMStoreFloat4(&q, XMVector4Transform(XMVectorSet((x + 0.5f) / kSize * 2 - 1, 1 - (y + 0.5f) / kSize * 2, 0.5f, 1), invVp));
				double       d[3]{ q.x / q.w, q.y / q.w, q.z / q.w };
				const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
				const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
				for (int c = 0; c < 3; ++c) dirs[i * 3 + c] = d[c] / len;
				const double den = dirs[i * 3 + 2] - a_v.tan * dirs[i * 3 + 1] - a_v.tanX * dirs[i * 3];
				if (den < -1e-12) {
					best[i] = -cam[2] / den;
					a_out.label[i] = kGround;
				}
			}
		}
		auto project = [&](double a_x, double a_y, double a_z, float& a_sx, float& a_sy) {
			XMFLOAT4 b;
			XMStoreFloat4(&b, XMVector4Transform(XMVectorSet(static_cast<float>(a_x - cam[0]), static_cast<float>(a_y - cam[1]), static_cast<float>(a_z - cam[2]), 1.0f), vp));
			a_sx = (b.x / b.w + 1.0f) * 0.5f * kSize;
			a_sy = (1.0f - b.y / b.w) * 0.5f * kSize;
		};
		auto screenBox = [&](const double (*a_pts)[3], int a_count, int& x0, int& x1, int& y0, int& y1) {
			float minX = 1e30f, maxX = -1e30f, minY = 1e30f, maxY = -1e30f;
			for (int k = 0; k < a_count; ++k) {
				float sx, sy;
				project(a_pts[k][0], a_pts[k][1], a_pts[k][2], sx, sy);
				minX = std::min(minX, sx);
				maxX = std::max(maxX, sx);
				minY = std::min(minY, sy);
				maxY = std::max(maxY, sy);
			}
			x0 = std::max(static_cast<int>(std::floor(minX)) - 1, 0);
			x1 = std::min(static_cast<int>(std::ceil(maxX)) + 1, static_cast<int>(kSize) - 1);
			y0 = std::max(static_cast<int>(std::floor(minY)) - 1, 0);
			y1 = std::min(static_cast<int>(std::ceil(maxY)) + 1, static_cast<int>(kSize) - 1);
		};
		const std::vector<double> groundS = gMask ? best : std::vector<double>{};   // mask: the ground alone (terrain-only depth)
		std::vector<double>       rawS(gMask ? n : 0, -1.0);                        // mask: the nearest blade in front of the ground
		// Objects: the trunk's axis starts under the ground so its foot is a plain cylinder.
		const double tz = a_v.tan * kTrunkY + a_v.tanX * kTrunkX, pz = a_v.tan * kPersonY + a_v.tanX * kPersonX, gz = a_v.tan * kPlayerY;
		const double qz = a_v.tan * kPostY + a_v.tanX * kPostX;
		const Capsule caps[] = {
			{ { kTrunkX, kTrunkY, tz - kTrunkR }, { kTrunkX, kTrunkY, tz + kTrunkH - kTrunkR }, kTrunkR, kTrunk },
			{ { kPersonX, kPersonY, pz + kBodyR }, { kPersonX, kPersonY, pz + kBodyH - kBodyR }, kBodyR, kPerson },
			{ { kPersonX, kPersonY, pz + kArmZ }, { kPersonX - kArmLen, kPersonY, pz + kArmZ }, kArmR, kPerson },
			{ { 0.0, kPlayerY, gz + kBodyR }, { 0.0, kPlayerY, gz + kBodyH - kBodyR }, kBodyR, kPlayer },
			{ { kPostX, kPostY, qz - kPostR }, { kPostX, kPostY, qz + kPostH - kPostR }, kPostR, kPost },   // only with a_post; its top is kPostH up
		};
		for (const Capsule& c : caps) {
			if (c.label == kPost && !a_post) continue;
			double pts[8][3];
			for (int k = 0; k < 8; ++k) {
				for (int a = 0; a < 3; ++a) pts[k][a] = ((k >> a) & 1) ? std::max(c.a[a], c.b[a]) + c.r : std::min(c.a[a], c.b[a]) - c.r;
			}
			int x0, x1, y0, y1;
			screenBox(pts, 8, x0, x1, y0, y1);
			for (int y = y0; y <= y1; ++y) {
				for (int x = x0; x <= x1; ++x) {
					const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
					const double      s = HitCapsule(cam, &dirs[i * 3], c);
					if (s > 0.0 && (best[i] < 0.0 || s < best[i])) {
						best[i] = s;
						a_out.label[i] = c.label;
						a_out.arm[i] = &c == &caps[2] ? 1 : 0;
					}
				}
			}
		}
		// Low boxes (lowbox scenes): the front bottom edge 1 unit into the ground under it (the slope's ground rises behind it).
		if (a_boxes) {
			const double sq2 = std::sqrt(2.0);
			const double bc[3]{ kBoxX, kBoxY, a_v.tan * (kBoxY - kBoxHalf[1]) - 1.0 + kBoxHalf[2] };
			const double dc[3]{ kDiaX, kDiaY, a_v.tan * (kDiaY - kDiaHalf[1]) - 1.0 + kDiaHalf[0] * sq2 };
			struct B
			{
				const double* c;
				const double* h;
				double        ang, ext;
				std::uint8_t  label;
			};
			const B boxes[]{ { bc, kBoxHalf, 0.0, kBoxHalf[0], kBox }, { dc, kDiaHalf, 0.25 * 3.141592653589793, kDiaHalf[0] * sq2, kDiamond } };
			for (const B& b : boxes) {
				double pts[8][3];
				for (int k = 0; k < 8; ++k) {
					pts[k][0] = b.c[0] + ((k & 1) ? b.ext : -b.ext);
					pts[k][1] = b.c[1] + ((k & 2) ? b.h[1] : -b.h[1]);
					pts[k][2] = b.c[2] + ((k & 4) ? b.ext : -b.ext);
				}
				int x0, x1, y0, y1;
				screenBox(pts, 8, x0, x1, y0, y1);
				for (int y = y0; y <= y1; ++y) {
					for (int x = x0; x <= x1; ++x) {
						const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
						const double      s = HitBox(cam, &dirs[i * 3], b.c, b.h, b.ang);
						if (s > 0.0 && (best[i] < 0.0 || s < best[i])) {
							best[i] = s;
							a_out.label[i] = b.label;
							a_out.arm[i] = 0;
						}
					}
				}
			}
		}
		// Big objects (bigobj scenes, GrassCover gate G4): two boxes and two capsules at distance a_big, each standing on the
		// ground under it (a box's front bottom edge 1 unit into the ground, as the low boxes; a capsule's axis starts under it).
		if (a_big > 0.0) {
			struct Obj
			{
				int           kind;   // 0 box, 1 capsule
				double        x;
				const double* h;
				double        r, len;
				std::uint8_t  label;
			};
			const Obj objs[]{ { 0, kBigFrac[0] * a_big, kBigBoxHalf1, 0.0, 0.0, kBigBox1 }, { 0, kBigFrac[1] * a_big, kBigBoxHalf2, 0.0, 0.0, kBigBox2 },
				{ 1, kBigFrac[2] * a_big, nullptr, kBigTrunkR, kBigTrunkH, kBigTrunk }, { 1, kBigFrac[3] * a_big, nullptr, kBigBodyR, kBigBodyH, kBigBody } };
			for (const Obj& o : objs) {
				double pts[8][3];
				double bc[3]{};
				Capsule cap{};
				if (o.kind == 0) {
					bc[0] = o.x;
					bc[1] = a_big;
					bc[2] = a_v.tan * (a_big - o.h[1]) + a_v.tanX * o.x - 1.0 + o.h[2];
					for (int k = 0; k < 8; ++k) {
						for (int a = 0; a < 3; ++a) pts[k][a] = bc[a] + (((k >> a) & 1) ? o.h[a] : -o.h[a]);
					}
				} else {
					const double gz = a_v.tan * a_big + a_v.tanX * o.x;
					cap = { { o.x, a_big, gz - o.r }, { o.x, a_big, gz + o.len - o.r }, o.r, o.label };
					for (int k = 0; k < 8; ++k) {
						for (int a = 0; a < 3; ++a) pts[k][a] = ((k >> a) & 1) ? std::max(cap.a[a], cap.b[a]) + cap.r : std::min(cap.a[a], cap.b[a]) - cap.r;
					}
				}
				int x0, x1, y0, y1;
				screenBox(pts, 8, x0, x1, y0, y1);
				for (int y = y0; y <= y1; ++y) {
					for (int x = x0; x <= x1; ++x) {
						const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
						const double      s = o.kind == 0 ? HitBox(cam, &dirs[i * 3], bc, o.h, 0.0) : HitCapsule(cam, &dirs[i * 3], cap);
						if (s > 0.0 && (best[i] < 0.0 || s < best[i])) {
							best[i] = s;
							a_out.label[i] = o.label;
							a_out.arm[i] = 0;
						}
					}
				}
			}
		}
		// Blades: vertical rectangles facing the camera, the top shifted sideways by the sway (sheared).
		const double half = 0.5 * a_width;
		for (const Blade& b : a_blades) {
			const double bz = a_v.tan * b.y + a_v.tanX * b.x;
			double       nx = cam[0] - b.x, ny = cam[1] - b.y;
			const double nl = std::sqrt(nx * nx + ny * ny);
			nx /= nl;
			ny /= nl;
			const double px = ny, py = -nx;   // sideways
			const double sway = kSway * std::sin(6.283185307179586 * (a_frame / 8.0 + b.phase));
			const double pts[4][3]{ { b.x - half * px, b.y - half * py, bz }, { b.x + half * px, b.y + half * py, bz },
				{ b.x + (sway - half) * px, b.y + (sway - half) * py, bz + b.h }, { b.x + (sway + half) * px, b.y + (sway + half) * py, bz + b.h } };
			int x0, x1, y0, y1;
			screenBox(pts, 4, x0, x1, y0, y1);
			const double off = (b.x - cam[0]) * nx + (b.y - cam[1]) * ny;
			for (int y = y0; y <= y1; ++y) {
				for (int x = x0; x <= x1; ++x) {
					const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
					const double*     d = &dirs[i * 3];
					const double      den = d[0] * nx + d[1] * ny;
					if (std::fabs(den) < 1e-12) continue;
					const double s = off / den;
					const bool   front = s > 0.0 && !(best[i] > 0.0 && s >= best[i]);
					const bool   rawFront = gMask && s > 0.0 && !(groundS[i] > 0.0 && s >= groundS[i]) && !(rawS[i] > 0.0 && s >= rawS[i]);
					if (!front && !rawFront) continue;
					const double hx = cam[0] + d[0] * s - b.x, hy = cam[1] + d[1] * s - b.y, hh = cam[2] + d[2] * s - bz;
					if (hh < 0.0 || hh > b.h) continue;
					if (std::fabs(hx * px + hy * py - sway * hh / b.h) > half) continue;
					if (rawFront) rawS[i] = s;
					if (!front) continue;
					best[i] = s;
					a_out.label[i] = kBlade;
					a_out.arm[i] = 0;
				}
			}
		}
		a_out.depth.assign(n, 1.0f);
		a_out.horiz.assign(n, -1.0f);
		a_out.height.assign(n, 0.0f);
		a_out.ray.assign(n, 1.0e9f);
		for (std::size_t i = 0; i < n; ++i) {
			if (!(best[i] > 0.0)) continue;
			const double* d = &dirs[i * 3];
			const double  s = best[i];
			const double  wx = cam[0] + d[0] * s, wy = cam[1] + d[1] * s, wz = cam[2] + d[2] * s;
			XMFLOAT4      b;
			XMStoreFloat4(&b, XMVector4Transform(XMVectorSet(static_cast<float>(d[0] * s), static_cast<float>(d[1] * s), static_cast<float>(d[2] * s), 1.0f), vp));
			a_out.depth[i] = static_cast<float>(std::round(static_cast<double>(b.z) / b.w * 16777215.0) / 16777215.0);   // D24_UNORM
			a_out.horiz[i] = static_cast<float>(std::sqrt(wx * wx + wy * wy));
			a_out.height[i] = static_cast<float>(wz - a_v.tan * wy - a_v.tanX * wx);
			a_out.ray[i] = static_cast<float>(s);
		}
		if (gMask) {   // the same expression as the depth above, so a visible blade's two depths are equal bit for bit
			a_out.grass.assign(n, 0);
			a_out.grassDepth.assign(n, 1.0f);
			for (std::size_t i = 0; i < n; ++i) {
				if (!(rawS[i] > 0.0)) continue;
				const double* d = &dirs[i * 3];
				const double  s = rawS[i];
				XMFLOAT4      b;
				XMStoreFloat4(&b, XMVector4Transform(XMVectorSet(static_cast<float>(d[0] * s), static_cast<float>(d[1] * s), static_cast<float>(d[2] * s), 1.0f), vp));
				a_out.grass[i] = 1;
				a_out.grassDepth[i] = static_cast<float>(std::round(static_cast<double>(b.z) / b.w * 16777215.0) / 16777215.0);
			}
		}
	}

	bool InBand(const FrameData& a_f, std::size_t a_i)
	{
		const std::uint8_t l = a_f.label[a_i];
		return (l == kGround || l == kBlade) && a_f.horiz[a_i] >= kBand0 && a_f.horiz[a_i] <= kBand1;
	}

	double Coverage(const FrameData& a_f, int* a_bandPx = nullptr)
	{
		int band = 0, blade = 0;
		for (std::size_t i = 0; i < a_f.label.size(); ++i) {
			if (!InBand(a_f, i)) continue;
			++band;
			blade += a_f.label[i] == kBlade ? 1 : 0;
		}
		if (a_bandPx) *a_bandPx = band;
		return band ? static_cast<double>(blade) / band : 0.0;
	}

	// Pixels within kOutline of an outline of the objects labelled a_l0 or a_l1 (an object pixel with a 4-neighbour
	// labelled otherwise).
	std::vector<std::uint8_t> OutlineBand(const std::vector<std::uint8_t>& a_label, std::uint8_t a_l0, std::uint8_t a_l1)
	{
		const int                 n = static_cast<int>(kSize);
		std::vector<std::uint8_t> edge(a_label.size(), 0), out(a_label.size(), 0);
		auto                      is = [&](int x, int y) {
            const std::uint8_t l = a_label[static_cast<std::size_t>(y) * n + x];
            return l == a_l0 || l == a_l1;
		};
		for (int y = 1; y + 1 < n; ++y) {
			for (int x = 1; x + 1 < n; ++x) {
				if (is(x, y) && (!is(x - 1, y) || !is(x + 1, y) || !is(x, y - 1) || !is(x, y + 1))) edge[static_cast<std::size_t>(y) * n + x] = 1;
			}
		}
		for (int y = 0; y < n; ++y) {
			for (int x = 0; x < n; ++x) {
				if (!edge[static_cast<std::size_t>(y) * n + x]) continue;
				for (int dy = std::max(y - kOutline, 0); dy <= std::min(y + kOutline, n - 1); ++dy) {
					for (int dx = std::max(x - kOutline, 0); dx <= std::min(x + kOutline, n - 1); ++dx) out[static_cast<std::size_t>(dy) * n + dx] = 1;
				}
			}
		}
		return out;
	}

	// The fog's alpha under ProfileSpan in closed form at a point a_h over flat ground a_ray from an eye a_eye up (an
	// even layer H, W; valid while the eye is above the layer): the exact march must give this.
	double ClosedAlpha(double a_h, double a_ray, double a_eye, double a_opacity)
	{
		const double a = kSteepness, x = std::clamp(a_h / kEvenH, 0.0, 1.0);
		const double span = std::exp(-a * x) * (1.0 - std::exp(-a * (1.0 - x))) / (1.0 - std::exp(-a));
		return 1.0 - std::exp(-a_opacity * kEvenW * span * a_ray / (a_eye - a_h));
	}

	struct Gpu
	{
		ID3D11Device*             dev;
		ID3D11DeviceContext*      ctx;
		ID3D11VertexShader*       vs;
		ID3D11PixelShader*        march;
		ID3D11PixelShader*        comp;
		ID3D11Texture2D*          lowTex;
		ID3D11RenderTargetView*   lowRtv;
		ID3D11ShaderResourceView* lowSrv;
		ID3D11Texture2D*          sceneTex;
		ID3D11RenderTargetView*   sceneRtv;
		ID3D11Texture2D*          depthTex;
		ID3D11ShaderResourceView* depthSrv;
		ID3D11Buffer*             cb;
		ID3D11SamplerState*       linear;
		ID3D11BlendState*         blend;
		ID3D11PixelShader*        cover;      // GrassCover (RaymarchShaders.h kCoverPS, kCoverFlagPS) and their targets
		ID3D11PixelShader*        coverFlag;
		ID3D11Texture2D*          coverTex;
		ID3D11RenderTargetView*   coverRtv;
		ID3D11ShaderResourceView* coverSrv;
		ID3D11Texture2D*          flagTex;
		ID3D11RenderTargetView*   flagRtv;
		ID3D11ShaderResourceView* flagSrv;
		ID3D11PixelShader*        compMask[3];   // CLI mask: composites with GROUNDFOG_COVER_MASK = 1 (M3), 2 (M0-M2); [0] unused
		ID3D11Texture2D*          marchFullTex;  // CLI mask, M0-M2: the march and the continuation at full resolution
		ID3D11RenderTargetView*   marchFullRtv;
		ID3D11ShaderResourceView* marchFullSrv;
		ID3D11Texture2D*          coverFullTex;
		ID3D11RenderTargetView*   coverFullRtv;
		ID3D11ShaderResourceView* coverFullSrv;
		ID3D11PixelShader*        marchSink[4];   // CLI sink: the march, [0] S (GROUNDFOG_SINK 1), [1] V (2), [2] S+e, [3] S-e (GSHIFT +-13.4); W0 with sinkw0
		ID3D11PixelShader*        compSink[4];    // and the composite likewise
		ID3D11PixelShader*        marchSinkDiag;  // and the march with GROUNDFOG_SINK_DIAG 1 (x = 1 hit, 2 no hit, 0 not continued; y = w; z = h)
		ID3D11Texture2D*          sinkFullTex;    // full-resolution target of the diagnostic pass
		ID3D11RenderTargetView*   sinkFullRtv;
	};

	// ---- Blade-free depth: the CPU effective-depth builder (design section 2; float arithmetic, as the shader of S3) ----

	// A full-resolution pixel row's angle (rc.posAdjust[3]; design S2): 2 tan(30 degrees) / 768.
	float FreePixelAngle()
	{
		return 2.0f * std::tan(XMConvertToRadians(30.0f)) / static_cast<float>(kSize);
	}

	// LinearDepth of kCommon (RaymarchShaders.h): view z from the device depth; a_p33, a_p43 = DepthP.xy.
	float FreeLinear(float a_d, float a_p33, float a_p43)
	{
		const float denom = a_d - a_p33;
		return std::fabs(denom) > 1e-7f ? a_p43 / denom : 1e9f;
	}

	float FreeStep(float a_e0, float a_e1, float a_x)   // HLSL smoothstep
	{
		const float t = std::clamp((a_x - a_e0) / (a_e1 - a_e0), 0.0f, 1.0f);
		return t * t * (3.0f - 2.0f * t);
	}

	// Columns.SampleLevel(LinearClamp, uv, 0) on the CPU copy of the column texture (kRes x kRes, 4 floats a texel).
	// Exact float weights here; a GPU sampler takes the weights in 1/256 steps, so the shader of S3 cannot match this bit
	// for bit through SampleLevel (to settle at S3: four Loads and the same lerp there, or the 1/256 steps here).
	void FreeSampleCols(const std::vector<float>& a_cols, float a_u, float a_v, float a_out[4])
	{
		const float fx = a_u * kRes - 0.5f, fy = a_v * kRes - 0.5f;
		const float bx = std::floor(fx), by = std::floor(fy);
		const float tx = fx - bx, ty = fy - by;
		const int   x0 = static_cast<int>(std::clamp(bx, 0.0f, static_cast<float>(kRes - 1)));
		const int   x1 = static_cast<int>(std::clamp(bx + 1.0f, 0.0f, static_cast<float>(kRes - 1)));
		const int   y0 = static_cast<int>(std::clamp(by, 0.0f, static_cast<float>(kRes - 1)));
		const int   y1 = static_cast<int>(std::clamp(by + 1.0f, 0.0f, static_cast<float>(kRes - 1)));
		for (int c = 0; c < 4; ++c) {
			const float a = a_cols[(static_cast<std::size_t>(y0) * kRes + x0) * 4 + c], b = a_cols[(static_cast<std::size_t>(y0) * kRes + x1) * 4 + c];
			const float d = a_cols[(static_cast<std::size_t>(y1) * kRes + x0) * 4 + c], e = a_cols[(static_cast<std::size_t>(y1) * kRes + x1) * 4 + c];
			const float top = a + (b - a) * tx, bottom = d + (e - d) * tx;
			a_out[c] = top + (bottom - top) * ty;
		}
	}

	// Horizontal 1-D closing of the linear depth a_z at (a_x, a_y) with segments of a_len pixels: the smallest, over the
	// a_len segments that hold the pixel, of the segment's largest z (2 a_len - 1 taps, clamped to the row). A near run
	// shorter than a_len is filled; a monotone row is returned unchanged. a_tap = the column that z was read at.
	// Tie rule (the shader of S3 must copy it): going outward from the pixel, a tap replaces the running maximum only when
	// it is larger; in a segment the left part's maximum wins a tie with the right part's; over the segments, from the
	// one ending at the pixel to the one starting at it, a later one wins only when its maximum is smaller.
	float FreeClose(const std::vector<float>& a_z, int a_x, int a_y, int a_len, int& a_tap)
	{
		const int    n = static_cast<int>(kSize);
		const float* row = &a_z[static_cast<std::size_t>(a_y) * n];
		float        lMax[kFreePxLimit], rMax[kFreePxLimit];
		int          lTap[kFreePxLimit], rTap[kFreePxLimit];
		lMax[0] = rMax[0] = row[a_x];
		lTap[0] = rTap[0] = a_x;
		for (int i = 1; i < a_len; ++i) {
			const int xl = std::max(a_x - i, 0), xr = std::min(a_x + i, n - 1);
			lMax[i] = lMax[i - 1];
			lTap[i] = lTap[i - 1];
			if (row[xl] > lMax[i]) {
				lMax[i] = row[xl];
				lTap[i] = xl;
			}
			rMax[i] = rMax[i - 1];
			rTap[i] = rTap[i - 1];
			if (row[xr] > rMax[i]) {
				rMax[i] = row[xr];
				rTap[i] = xr;
			}
		}
		float best = 0.0f;
		a_tap = a_x;
		for (int j = a_len - 1; j >= 0; --j) {   // the segment a_x - j .. a_x - j + a_len - 1
			const bool  left = lMax[j] >= rMax[a_len - 1 - j];
			const float m = left ? lMax[j] : rMax[a_len - 1 - j];
			if (j == a_len - 1 || m < best) {
				best = m;
				a_tap = left ? lTap[j] : rTap[a_len - 1 - j];
			}
		}
		return best;
	}

	struct FreeParams
	{
		float   start = 0.0f;         // BladeFreeStart, units from the eye (0 = off)
		FreeSrc src = kFreeRef;
		int     maxPx = kFreeMaxPx;
		FreeMut mut = kMutNone;
	};

	struct FreeOut
	{
		std::vector<float>        depth;   // the effective depth (device depth, as the frame's)
		std::vector<float>        k;       // the share of the candidate depth taken, 0..1
		std::vector<std::uint8_t> veto;    // 1 where only the tip veto took k to 0
		std::vector<std::uint8_t> lpx;     // the closing length used (0 where no closing ran)
	};

	// The effective depth of one frame (design section 2, steps 1 to 4). a_ref = the blade-free depth of the same pixels
	// (source REF) or null (source CLOSE). a_cols = the CPU copy of the column texture the passes get at t1.
	//   1. d0, z0 = LinearDepth(d0), dist0 = PixelRay's distance, kD = smoothstep(start, 2 start, dist0); no surface
	//      (PixelRay's far branch) or kD = 0: d0.
	//   2. the candidate (dc, zc): REF the reference depth; CLOSE the closing over Lpx pixels, dc = the depth loaded at the
	//      tap that gave zc, and the tip veto: the same closing m rows below (m = (Lpx - 1) / 2 rounded up); that pixel not filled and its z
	//      within 15 % of z0 -> k = 0.
	//   3. rel = zc / z0 - 1, g = smoothstep(0.025, 0.06, rel) (1 - smoothstep(0.45, 0.60, rel)); the pixel's point over
	//      the column's ground fades out from 110 to 150 units; column unknown -> 0; the field's round fade 0.85 to 0.95.
	//   4. k = kD x gates; d = d0 at k = 0, dc at k = 1, d0 + (dc - d0) k between.
	// m: the design writes (Lpx - 1) / 2 and does not say which way a half goes. Under a 45-degree tip the row r below is
	// 2 r + 1 pixels wide, and the veto needs a row the closing does not fill, 2 m + 1 >= Lpx: m = Lpx / 2 in integers.
	// Rounded down, an even Lpx looks at a row of Lpx - 1 pixels, which the closing fills, and the veto cannot fire there
	// (odd Lpx: the same row either way).
	// Mutations: kMutLpx1 Lpx = 1 (the closing returns the pixel: nothing changes), kMutNoVeto no tip veto, kMutNoGate
	// k = 1 wherever the candidate is farther (self-check a: with REF the result is the blade-free frame's depth).
	FreeOut BuildFreeDepth(const FrameData& a_f, const View& a_v, const std::vector<float>& a_cols, const std::vector<float>* a_ref, const FreeParams& a_p)
	{
		const int         n = static_cast<int>(kSize);
		const std::size_t count = a_f.depth.size();
		FreeOut           out;
		out.depth = a_f.depth;
		out.k.assign(count, 0.0f);
		out.veto.assign(count, 0);
		out.lpx.assign(count, 0);
		XMFLOAT4X4 pf;
		XMStoreFloat4x4(&pf, a_v.proj);
		const float        p33 = pf._33, p43 = pf._43;   // rc.depth[0], rc.depth[1]
		std::vector<float> z(count);
		for (std::size_t i = 0; i < count; ++i) z[i] = FreeLinear(a_f.depth[i], p33, p43);
		const XMMATRIX inv = XMLoadFloat4x4(&a_f.inv);
		XMFLOAT4       e;
		XMStoreFloat4(&e, XMVector4Transform(XMVectorSet(0, 0, 1, 0), inv));
		const float eye[3]{ std::fabs(e.w) > 1e-12f ? e.x / e.w : 0.0f, std::fabs(e.w) > 1e-12f ? e.y / e.w : 0.0f, std::fabs(e.w) > 1e-12f ? e.z / e.w : 0.0f };
		const float posAdjust[3]{ 0.0f, 0.0f, a_v.eye };   // rc.posAdjust
		const float angle = FreePixelAngle();
		const int   maxPx = std::clamp(a_p.maxPx, kFreeMinPx, kFreePxLimit);
		for (int y = 0; y < n; ++y) {
			for (int x = 0; x < n; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * n + x;
				const float       d0 = a_f.depth[i], z0 = z[i];
				XMFLOAT4          h;
				XMStoreFloat4(&h, XMVector4Transform(XMVectorSet((x + 0.5f) / kSize * 2 - 1, 1 - (y + 0.5f) / kSize * 2, d0, 1), inv));
				if (!(h.w > 1e-6f)) continue;   // no surface
				const float seg[3]{ h.x / h.w - eye[0], h.y / h.w - eye[1], h.z / h.w - eye[2] };
				const float dist0 = std::sqrt(seg[0] * seg[0] + seg[1] * seg[1] + seg[2] * seg[2]);
				const float kD = a_p.mut == kMutNoGate ? 1.0f : FreeStep(a_p.start, 2.0f * a_p.start, dist0);
				if (!(kD > 0.0f)) continue;
				float dc = d0, zc = z0;
				int   len = 0;
				if (a_ref) {
					dc = (*a_ref)[i];
					zc = FreeLinear(dc, p33, p43);
				} else {
					len = a_p.mut == kMutLpx1 ? 1 : std::clamp(static_cast<int>(std::floor(kFreeWidth / (z0 * angle) + 0.5f)), kFreeMinPx, maxPx);
					int tap = x;
					zc = FreeClose(z, x, y, len, tap);
					dc = a_f.depth[static_cast<std::size_t>(y) * n + tap];
					out.lpx[i] = static_cast<std::uint8_t>(len);
				}
				const float rel = zc / z0 - 1.0f;
				float       k = 0.0f;
				if (a_p.mut == kMutNoGate) {
					k = zc > z0 ? 1.0f : 0.0f;
				} else {
					k = kD * FreeStep(kFreeRel0, kFreeRel1, rel) * (1.0f - FreeStep(kFreeRelMax0, kFreeRelMax1, rel));
					if (k > 0.0f) {
						// the pixel's own point, as ThinShare takes it: PosAdjust + eye + dir * dist
						const float invLen = 1.0f / std::max(dist0, 1e-3f);
						const float world[3]{ posAdjust[0] + eye[0] + seg[0] * invLen * dist0, posAdjust[1] + eye[1] + seg[1] * invLen * dist0,
							posAdjust[2] + eye[2] + seg[2] * invLen * dist0 };
						const float u = (world[0] - kOrigin) / kExtent, v = (world[1] - kOrigin) / kExtent;
						float       col[4];
						FreeSampleCols(a_cols, u, v, col);
						if (col[3] >= kFreeKnown) {
							const float hMe = world[2] - col[2] / col[3];
							const float r = std::sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f)) * 2.0f;
							k *= (1.0f - FreeStep(kFreeH0, kFreeH1, hMe)) * (1.0f - FreeStep(kFreeR0, kFreeR1, r));
						} else {
							k = 0.0f;
						}
					}
				}
				if (k > 0.0f && !a_ref && a_p.mut != kMutNoVeto && a_p.mut != kMutNoGate) {
					const int   yq = std::min(y + len / 2, n - 1);
					const float zq = z[static_cast<std::size_t>(yq) * n + x];
					int         tapQ = x;
					const float zcq = FreeClose(z, x, yq, len, tapQ);
					const bool  filled = zcq / zq - 1.0f > kFreeVetoFilled;
					if (!filled && std::fabs(zq - z0) <= kFreeVetoAgree * z0) {
						k = 0.0f;
						out.veto[i] = 1;
					}
				}
				out.k[i] = k;
				if (k >= 1.0f) {
					out.depth[i] = dc;
				} else if (k > 0.0f) {
					out.depth[i] = d0 + (dc - d0) * k;
				}
			}
		}
		return out;
	}

	// The narrowest near-run of the solid objects (trunk, the person's body, player, boxes; not the arm, not the post) among
	// the runs the closing ran on: a run = one object's pixels side by side along a row, with a farther pixel next to it on
	// both sides (a blade in front or the picture's edge beside it: not counted). a_len = the largest closing length over
	// that run's pixels; a_short counts the runs narrower than theirs (the closing alone would fill them).
	void FreeMinRun(const FrameData& a_f, const std::vector<std::uint8_t>& a_lpx, int& a_min, int& a_len, long long& a_short)
	{
		const int n = static_cast<int>(kSize);
		auto      solid = [&](std::size_t i) -> int {
			const std::uint8_t l = a_f.label[i];
			if (l == kPerson) return a_f.arm[i] ? 0 : static_cast<int>(kPerson);
			return l == kTrunk || l == kPlayer || l == kBox || l == kDiamond ? static_cast<int>(l) : 0;
		};
		for (int y = 0; y < n; ++y) {
			const std::size_t row = static_cast<std::size_t>(y) * n;
			for (int x = 0; x < n;) {
				const int c = solid(row + x);
				int       x1 = x;
				while (x1 + 1 < n && solid(row + x1 + 1) == c) ++x1;
				if (c != 0 && x > 0 && x1 + 1 < n && a_f.depth[row + x - 1] > a_f.depth[row + x] && a_f.depth[row + x1 + 1] > a_f.depth[row + x1]) {
					int len = 0;
					for (int k = x; k <= x1; ++k) len = std::max(len, static_cast<int>(a_lpx[row + k]));
					const int run = x1 - x + 1;
					if (len > 0) {
						if (a_min < 0 || run < a_min || (run == a_min && len > a_len)) {
							a_min = run;
							a_len = len;
						}
						if (run < len) ++a_short;
					}
				}
				x = x1 + 1;
			}
		}
	}

	// One frame of fog: alpha per full-res pixel; a_low (optional) gets the march texture's scene distances. a_debugView as
	// the ini's: 0 = now (the half-resolution march and the upsampling), 2 / 3 = every pixel marches its own ray at 64 /
	// kSteps steps (no march pass: CompositeM still upsamples whatever the low texture holds, then replaces the result).
	// a_thinStart = the ini's ThinFogStart (rc.lowRes[3]; 0 = off). a_freeDepth (free): this effective depth is uploaded as
	// the depth texture in place of the frame's, for both passes; a_freeStart, a_freePx fill the constant slots the design
	// gives the shader of step S3 (no shader reads them yet).
	// a_coverStart = the ini's GrassCover (rc.colorLow[3]; 0 = off: no extra pass); a_coverLow (optional) gets the cover
	// pass's continuation per low-res texel (x = transmittance, w = end distance; 2 floats a texel).
	std::vector<float> Render(const Gpu& g, const FrameData& a_f, const View& a_v, ID3D11ShaderResourceView* a_cols, float a_opacity, int a_frame, int a_debugView,
		std::vector<float>* a_low, float a_thinStart = 0.0f, const std::vector<float>* a_freeDepth = nullptr, float a_freeStart = 0.0f, int a_freePx = 0,
		float a_coverStart = 0.0f, std::vector<float>* a_coverLow = nullptr, const std::vector<float>* a_mask = nullptr, int a_maskMode = 0,
		std::vector<std::uint8_t>* a_fullCont = nullptr, int a_sink = 0, bool a_phase0 = false)
	{
		// a_mask (CLI mask, with a_coverStart): the grass mask goes into the flag target in place of the flag pass and the
		// composite with GROUNDFOG_COVER_MASK = a_maskMode runs; mode 2 first marches and continues at full resolution
		// (a_fullCont gets 1 per pixel where that full-res continuation ran).
		// a_sink (CLI sink, without GrassCover): the march and the composite compiled for row 1 S, 2 V, 3 S+e, 4 S-e (0 = today's);
		// a_phase0: the march phase FogP.z held at 0 (row C; with a_sink: SC, VC).
		const bool cover = a_coverStart > 0.0f && a_debugView == 0;
		const UINT lowW = kSize / 2;
		g.ctx->UpdateSubresource(g.depthTex, 0, nullptr, a_freeDepth ? a_freeDepth->data() : a_f.depth.data(), kSize * sizeof(float), 0);
		XMFLOAT4X4 pf;
		XMStoreFloat4x4(&pf, a_v.proj);
		RaymarchShaders::RayCB rc{};
		std::memcpy(rc.invViewProj, &a_f.inv, sizeof(rc.invViewProj));
		rc.field[0] = kOrigin;
		rc.field[1] = kOrigin;
		rc.field[2] = kExtent;
		rc.field[3] = 1.0f;
		rc.box[0] = -1.0f - static_cast<float>(std::fabs(a_v.tan * kOrigin) + std::fabs(a_v.tanX * kOrigin));   // the field's lowest ground
		rc.box[1] = kThickness + 2.0f + static_cast<float>(std::fabs(a_v.tan * kOrigin) + std::fabs(a_v.tanX * kOrigin));   // FogRenderer: Thickness + 2 over the highest ground
		rc.box[2] = kExtent;
		rc.box[3] = static_cast<float>(kSteps);
		rc.fog[0] = a_opacity;
		rc.fog[1] = kSteepness;
		rc.fog[2] = a_phase0 ? 0.0f : static_cast<float>(a_frame);
		rc.fog[3] = static_cast<float>(a_debugView);   // ini DebugView
		rc.depth[0] = pf._33;
		rc.depth[1] = pf._43;
		rc.depth[2] = static_cast<float>(kSize);
		rc.depth[3] = static_cast<float>(kSize);
		rc.lowRes[0] = static_cast<float>(lowW);
		rc.lowRes[1] = static_cast<float>(lowW);
		rc.lowRes[2] = static_cast<float>(kSize) / lowW;
		rc.lowRes[3] = a_thinStart;
		for (int c = 0; c < 3; ++c) {
			rc.colorLow[c] = 1.0f;
			rc.colorHigh[c] = 1.0f;
		}
		rc.distFog[0] = 1.0e6f;
		rc.distFog[1] = 2.0e6f;
		rc.distFog[2] = 1.0f;
		rc.posAdjust[2] = a_v.eye;
		if (a_freeDepth) {
			rc.distColorNear[3] = a_freeStart;
			rc.distColorFar[3] = static_cast<float>(a_freePx);
			rc.posAdjust[3] = FreePixelAngle();
		}
		if (cover) rc.colorLow[3] = a_coverStart;
		g.ctx->UpdateSubresource(g.cb, 0, nullptr, &rc, 0, 0);
		const float               black[4]{};
		ID3D11ShaderResourceView* none[3]{};
		g.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		g.ctx->VSSetShader(g.vs, nullptr, 0);
		g.ctx->PSSetConstantBuffers(0, 1, &g.cb);
		g.ctx->PSSetSamplers(0, 1, &g.linear);
		if (a_debugView == 0) {
			D3D11_VIEWPORT lowVp{ 0, 0, static_cast<float>(lowW), static_cast<float>(lowW), 0, 1 };
			g.ctx->ClearRenderTargetView(g.lowRtv, black);
			g.ctx->RSSetViewports(1, &lowVp);
			g.ctx->OMSetRenderTargets(1, &g.lowRtv, nullptr);
			g.ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);
			g.ctx->PSSetShader(a_sink ? g.marchSink[a_sink - 1] : g.march, nullptr, 0);
			ID3D11ShaderResourceView* in[2]{ g.depthSrv, a_cols };
			g.ctx->PSSetShaderResources(0, 2, in);
			g.ctx->Draw(3, 0);
			g.ctx->PSSetShaderResources(0, 3, none);
		}
		D3D11_VIEWPORT fullVp{ 0, 0, static_cast<float>(kSize), static_cast<float>(kSize), 0, 1 };
		if (cover) {   // GrassCover: the continuation at the march's resolution, then the ground-like flags at full resolution
			D3D11_VIEWPORT lowVp{ 0, 0, static_cast<float>(lowW), static_cast<float>(lowW), 0, 1 };
			ID3D11ShaderResourceView* in[2]{ g.depthSrv, a_cols };
			g.ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);
			g.ctx->ClearRenderTargetView(g.coverRtv, black);
			g.ctx->RSSetViewports(1, &lowVp);
			g.ctx->OMSetRenderTargets(1, &g.coverRtv, nullptr);
			g.ctx->PSSetShader(g.cover, nullptr, 0);
			g.ctx->PSSetShaderResources(0, 2, in);
			g.ctx->Draw(3, 0);
			if (a_mask) {
				g.ctx->OMSetRenderTargets(0, nullptr, nullptr);
				g.ctx->UpdateSubresource(g.flagTex, 0, nullptr, a_mask->data(), kSize * sizeof(float), 0);
			} else {
				g.ctx->ClearRenderTargetView(g.flagRtv, black);
				g.ctx->RSSetViewports(1, &fullVp);
				g.ctx->OMSetRenderTargets(1, &g.flagRtv, nullptr);
				g.ctx->PSSetShader(g.coverFlag, nullptr, 0);
				g.ctx->Draw(3, 0);
			}
			if (a_mask && a_maskMode == 2) {   // the same two passes at full resolution (LowRes = the full size, scale 1)
				RaymarchShaders::RayCB rf = rc;
				rf.lowRes[0] = static_cast<float>(kSize);
				rf.lowRes[1] = static_cast<float>(kSize);
				rf.lowRes[2] = 1.0f;
				g.ctx->UpdateSubresource(g.cb, 0, nullptr, &rf, 0, 0);
				g.ctx->RSSetViewports(1, &fullVp);
				g.ctx->ClearRenderTargetView(g.marchFullRtv, black);
				g.ctx->OMSetRenderTargets(1, &g.marchFullRtv, nullptr);
				g.ctx->PSSetShader(g.march, nullptr, 0);
				g.ctx->Draw(3, 0);
				g.ctx->ClearRenderTargetView(g.coverFullRtv, black);
				g.ctx->OMSetRenderTargets(1, &g.coverFullRtv, nullptr);
				g.ctx->PSSetShader(g.cover, nullptr, 0);
				g.ctx->Draw(3, 0);
				g.ctx->UpdateSubresource(g.cb, 0, nullptr, &rc, 0, 0);
			}
			g.ctx->PSSetShaderResources(0, 3, none);
			g.ctx->OMSetRenderTargets(0, nullptr, nullptr);
		}
		g.ctx->ClearRenderTargetView(g.sceneRtv, black);
		g.ctx->RSSetViewports(1, &fullVp);
		g.ctx->OMSetRenderTargets(1, &g.sceneRtv, nullptr);
		g.ctx->OMSetBlendState(g.blend, black, 0xFFFFFFFF);
		g.ctx->PSSetShader(cover && a_mask ? g.compMask[a_maskMode] : a_sink ? g.compSink[a_sink - 1] : g.comp, nullptr, 0);
		ID3D11ShaderResourceView* compIn[3]{ g.depthSrv, a_cols, g.lowSrv };
		g.ctx->PSSetShaderResources(0, 3, compIn);
		if (cover) {
			ID3D11ShaderResourceView* coverIn[2]{ g.coverSrv, g.flagSrv };
			g.ctx->PSSetShaderResources(8, 2, coverIn);
		}
		if (cover && a_mask && a_maskMode == 2) {
			ID3D11ShaderResourceView* fullIn[2]{ g.marchFullSrv, g.coverFullSrv };
			g.ctx->PSSetShaderResources(10, 2, fullIn);
		}
		g.ctx->Draw(3, 0);
		g.ctx->PSSetShaderResources(0, 3, none);
		if (cover) g.ctx->PSSetShaderResources(8, 2, none);
		if (cover && a_mask && a_maskMode == 2) g.ctx->PSSetShaderResources(10, 2, none);
		g.ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);
		const auto         img = ReadFloats(g.dev, g.ctx, g.sceneTex, kSize, kSize, 4);
		std::vector<float> fog(static_cast<std::size_t>(kSize) * kSize);
		for (std::size_t i = 0; i < fog.size(); ++i) fog[i] = img[i * 4];
		if (a_low) {
			const auto low = ReadFloats(g.dev, g.ctx, g.lowTex, lowW, lowW, 4);
			a_low->resize(static_cast<std::size_t>(lowW) * lowW);
			for (std::size_t i = 0; i < a_low->size(); ++i) (*a_low)[i] = low[i * 4 + 3];
		}
		if (cover && a_coverLow) {
			const auto cv = ReadFloats(g.dev, g.ctx, g.coverTex, lowW, lowW, 4);
			a_coverLow->resize(static_cast<std::size_t>(lowW) * lowW * 2);
			for (std::size_t i = 0; i < a_coverLow->size() / 2; ++i) {
				(*a_coverLow)[i * 2] = cv[i * 4];
				(*a_coverLow)[i * 2 + 1] = cv[i * 4 + 3];
			}
		}
		if (cover && a_mask && a_maskMode == 2 && a_fullCont) {   // CoverTap's test: a continuation ran where w > 0 and x < 1
			const auto cv = ReadFloats(g.dev, g.ctx, g.coverFullTex, kSize, kSize, 4);
			a_fullCont->assign(static_cast<std::size_t>(kSize) * kSize, 0);
			for (std::size_t i = 0; i < a_fullCont->size(); ++i) (*a_fullCont)[i] = cv[i * 4 + 3] > 0.0f && cv[i * 4] < 1.0f ? 1 : 0;
		}
		return fog;
	}

	// Round 5 (CLI sink): the sink's decision for every full-res pixel's own march (the march with GROUNDFOG_SINK_DIAG at full
	// resolution): 4 floats a pixel, x = 1 continued and hit, 2 continued without a hit, 0 not continued; y = w; z = h. The half-res
	// texel c evaluates the pixel min(floor((c + 0.5) 2), size - 1) and the own-ray fallback the pixel itself, so this one pass
	// gives both (neither w nor the hit depends on the jitter, the phase or the opacity).
	std::vector<float> SinkDiag(const Gpu& g, const FrameData& a_f, const View& a_v, ID3D11ShaderResourceView* a_cols, float a_opacity)
	{
		g.ctx->UpdateSubresource(g.depthTex, 0, nullptr, a_f.depth.data(), kSize * sizeof(float), 0);
		XMFLOAT4X4 pf;
		XMStoreFloat4x4(&pf, a_v.proj);
		RaymarchShaders::RayCB rc{};
		std::memcpy(rc.invViewProj, &a_f.inv, sizeof(rc.invViewProj));
		rc.field[0] = kOrigin;
		rc.field[1] = kOrigin;
		rc.field[2] = kExtent;
		rc.field[3] = 1.0f;
		rc.box[0] = -1.0f - static_cast<float>(std::fabs(a_v.tan * kOrigin) + std::fabs(a_v.tanX * kOrigin));
		rc.box[1] = kThickness + 2.0f + static_cast<float>(std::fabs(a_v.tan * kOrigin) + std::fabs(a_v.tanX * kOrigin));
		rc.box[2] = kExtent;
		rc.box[3] = static_cast<float>(kSteps);
		rc.fog[0] = a_opacity;
		rc.fog[1] = kSteepness;
		rc.depth[0] = pf._33;
		rc.depth[1] = pf._43;
		rc.depth[2] = static_cast<float>(kSize);
		rc.depth[3] = static_cast<float>(kSize);
		rc.lowRes[0] = static_cast<float>(kSize);
		rc.lowRes[1] = static_cast<float>(kSize);
		rc.lowRes[2] = 1.0f;
		rc.distFog[0] = 1.0e6f;
		rc.distFog[1] = 2.0e6f;
		rc.distFog[2] = 1.0f;
		rc.posAdjust[2] = a_v.eye;
		g.ctx->UpdateSubresource(g.cb, 0, nullptr, &rc, 0, 0);
		const float               black[4]{};
		ID3D11ShaderResourceView* none[2]{};
		ID3D11ShaderResourceView* in[2]{ g.depthSrv, a_cols };
		D3D11_VIEWPORT            fullVp{ 0, 0, static_cast<float>(kSize), static_cast<float>(kSize), 0, 1 };
		g.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		g.ctx->VSSetShader(g.vs, nullptr, 0);
		g.ctx->PSSetConstantBuffers(0, 1, &g.cb);
		g.ctx->PSSetSamplers(0, 1, &g.linear);
		g.ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);
		g.ctx->ClearRenderTargetView(g.sinkFullRtv, black);
		g.ctx->RSSetViewports(1, &fullVp);
		g.ctx->OMSetRenderTargets(1, &g.sinkFullRtv, nullptr);
		g.ctx->PSSetShader(g.marchSinkDiag, nullptr, 0);
		g.ctx->PSSetShaderResources(0, 2, in);
		g.ctx->Draw(3, 0);
		g.ctx->PSSetShaderResources(0, 2, none);
		g.ctx->OMSetRenderTargets(0, nullptr, nullptr);
		return ReadFloats(g.dev, g.ctx, g.sinkFullTex, kSize, kSize, 4);
	}

	// Replica of CompositeM's match test (RaymarchShaders.h, the 3 x 3 low-res neighbours within 15 %) on the CPU: 1
	// where no neighbour matches and the pixel marches its own ray. The scene distance is PixelRay's from the depth
	// (a_depth, free: the effective depth the passes read in place of the frame's).
	std::vector<std::uint8_t> OwnRay(const FrameData& a_f, const std::vector<float>& a_low, const std::vector<float>* a_depth = nullptr)
	{
		const std::vector<float>& depth = a_depth ? *a_depth : a_f.depth;
		const int      lowW = static_cast<int>(kSize / 2);
		const float    scale = static_cast<float>(kSize) / lowW;
		const XMMATRIX inv = XMLoadFloat4x4(&a_f.inv);
		XMFLOAT4       e;
		XMStoreFloat4(&e, XMVector4Transform(XMVectorSet(0, 0, 1, 0), inv));
		const float eye[3]{ std::fabs(e.w) > 1e-12f ? e.x / e.w : 0.0f, std::fabs(e.w) > 1e-12f ? e.y / e.w : 0.0f, std::fabs(e.w) > 1e-12f ? e.z / e.w : 0.0f };
		std::vector<std::uint8_t> own(depth.size(), 0);
		for (int y = 0; y < static_cast<int>(kSize); ++y) {
			for (int x = 0; x < static_cast<int>(kSize); ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
				XMFLOAT4          h;
				XMStoreFloat4(&h, XMVector4Transform(XMVectorSet((x + 0.5f) / kSize * 2 - 1, 1 - (y + 0.5f) / kSize * 2, depth[i], 1), inv));
				float myDist = 1e9f;
				if (h.w > 1e-6f) {
					const float sx = h.x / h.w - eye[0], sy = h.y / h.w - eye[1], sz = h.z / h.w - eye[2];
					myDist = std::sqrt(sx * sx + sy * sy + sz * sz);
				}
				const float lx = (x + 0.5f) / scale - 0.5f, ly = (y + 0.5f) / scale - 0.5f;
				const int   bx = static_cast<int>(std::floor(lx)), by = static_cast<int>(std::floor(ly));
				float       wsum = 0.0f;
				for (int dy = -1; dy <= 1; ++dy) {
					for (int dx = -1; dx <= 1; ++dx) {
						const int   cx = std::clamp(bx + dx, 0, lowW - 1), cy = std::clamp(by + dy, 0, lowW - 1);
						const float rel = std::fabs(a_low[static_cast<std::size_t>(cy) * lowW + cx] - myDist) / std::max(myDist, 1.0f);
						if (!(rel <= 0.15f)) continue;
						const float ox = cx - lx, oy = cy - ly;
						wsum += (1.0f - rel / 0.15f) / (1.0f + ox * ox + oy * oy);
					}
				}
				own[i] = static_cast<std::uint8_t>(wsum > 1e-5f ? 0 : 1);
			}
		}
		return own;
	}

	// CPU replica of the ground-side tap rule (ledger "ground side" V2) for the GPU cost estimate: 1 where it leaves no tap
	// (wsum <= 1e-5) and the pixel would march its own ray (the cover's k >= 0.999 skip is not modelled: an upper bound).
	std::vector<std::uint8_t> OwnRayGround(const FrameData& a_f, const std::vector<float>& a_low, float a_thinStart)
	{
		const int      lowW = static_cast<int>(kSize / 2);
		const float    scale = static_cast<float>(kSize) / lowW;
		const XMMATRIX inv = XMLoadFloat4x4(&a_f.inv);
		XMFLOAT4       e;
		XMStoreFloat4(&e, XMVector4Transform(XMVectorSet(0, 0, 1, 0), inv));
		const float eye[3]{ std::fabs(e.w) > 1e-12f ? e.x / e.w : 0.0f, std::fabs(e.w) > 1e-12f ? e.y / e.w : 0.0f, std::fabs(e.w) > 1e-12f ? e.z / e.w : 0.0f };
		std::vector<float> dist(a_f.depth.size(), 1e9f);
		for (int y = 0; y < static_cast<int>(kSize); ++y) {
			for (int x = 0; x < static_cast<int>(kSize); ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
				XMFLOAT4          h;
				XMStoreFloat4(&h, XMVector4Transform(XMVectorSet((x + 0.5f) / kSize * 2 - 1, 1 - (y + 0.5f) / kSize * 2, a_f.depth[i], 1), inv));
				if (h.w > 1e-6f) {
					const float sx = h.x / h.w - eye[0], sy = h.y / h.w - eye[1], sz = h.z / h.w - eye[2];
					dist[i] = std::sqrt(sx * sx + sy * sy + sz * sz);
				}
			}
		}
		auto ss = [](float a, float b, float x) {
			const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
			return t * t * (3.0f - 2.0f * t);
		};
		std::vector<std::uint8_t> own(a_f.depth.size(), 0);
		for (int y = 0; y < static_cast<int>(kSize); ++y) {
			for (int x = 0; x < static_cast<int>(kSize); ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
				const float       myDist = dist[i];
				const float       lx = (x + 0.5f) / scale - 0.5f, ly = (y + 0.5f) / scale - 0.5f;
				const int         bx = static_cast<int>(std::floor(lx)), by = static_cast<int>(std::floor(ly));
				const float       kD = myDist < 1e8f ? ss(a_thinStart, 2.0f * a_thinStart, myDist) : 0.0f;
				float             wsum = 0.0f;
				for (int dy = -1; dy <= 1; ++dy) {
					const int   cy = std::clamp(by + dy, 0, lowW - 1);
					const int   sy = std::min(static_cast<int>((cy + 0.5f) * scale), static_cast<int>(kSize) - 1);
					const float ref = std::min(myDist, dist[static_cast<std::size_t>(sy) * kSize + x]);
					for (int dx = -1; dx <= 1; ++dx) {
						const int   cx = std::clamp(bx + dx, 0, lowW - 1);
						const float dt = a_low[static_cast<std::size_t>(cy) * lowW + cx];
						const float rel = std::fabs(dt - myDist) / std::max(myDist, 1.0f);
						if (!(rel <= 0.15f)) continue;
						const float ox = cx - lx, oy = cy - ly;
						float       w = (1.0f - rel / 0.15f) / (1.0f + ox * ox + oy * oy);
						if (kD > 0.0f && ref < 1e8f) w *= 1.0f - kD * ss(0.025f, 0.06f, ref / std::max(dt, 1.0f) - 1.0f);
						wsum += w;
					}
				}
				own[i] = static_cast<std::uint8_t>(wsum > 1e-5f ? 0 : 1);
			}
		}
		return own;
	}

	// Pixels within kOutline of any object (trunk, person, player, boxes).
	std::vector<std::uint8_t> NearObjects(const std::vector<std::uint8_t>& a_label)
	{
		const int                 n = static_cast<int>(kSize);
		std::vector<std::uint8_t> out(a_label.size(), 0);
		for (int y = 0; y < n; ++y) {
			for (int x = 0; x < n; ++x) {
				const std::uint8_t l = a_label[static_cast<std::size_t>(y) * n + x];
				if (l == kSky || l == kGround || l == kBlade) continue;
				for (int dy = std::max(y - kOutline, 0); dy <= std::min(y + kOutline, n - 1); ++dy) {
					for (int dx = std::max(x - kOutline, 0); dx <= std::min(x + kOutline, n - 1); ++dx) out[static_cast<std::size_t>(dy) * n + dx] = 1;
				}
			}
		}
		return out;
	}

	struct FreeRun   // the CLI words of the blade-free depth
	{
		FreeParams p;
		bool       orderRev = false;   // order rev: the free render once more over the frames in reverse order (gate STATE)
		bool       eye120 = false;     // extra scenes: eye 120 over flat ground (sparse, clumps)
		bool       tilt = false;       // extra scenes: ground rising 10 degrees along +x (bare, sparse)
		bool       post = false;       // extra scenes: bare ground with the post
	};

	// GrassCover CPU replica of k (diagnostics only): per pixel of one frame, from the frame's exact hit heights and the CPU
	// column texture (H at the hit point; the harness binds no tall layer). a_kH, a_kF = the height gate and the fraction
	// term of non-ground pixels (1 for ground-like ones).
	void CoverReplica(const FrameData& a_f, const std::vector<float>& a_cols, float a_start, std::vector<float>& a_k, std::vector<float>& a_kH, std::vector<float>& a_kF)
	{
		const int         n = static_cast<int>(kSize);
		const std::size_t np = a_f.label.size();
		std::vector<int>  integ(static_cast<std::size_t>(n + 1) * (n + 1), 0);   // summed-area table of the ground-like flags
		for (int y = 0; y < n; ++y) {
			int row = 0;
			for (int x = 0; x < n; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * n + x;
				row += (a_f.label[i] != kSky && a_f.height[i] < kCoverEps) ? 1 : 0;
				integ[static_cast<std::size_t>(y + 1) * (n + 1) + x + 1] = integ[static_cast<std::size_t>(y) * (n + 1) + x + 1] + row;
			}
		}
		const XMMATRIX inv = XMLoadFloat4x4(&a_f.inv);
		a_k.assign(np, 0.0f);
		a_kH.assign(np, 1.0f);
		a_kF.assign(np, 1.0f);
		for (int y = 0; y < n; ++y) {
			for (int x = 0; x < n; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * n + x;
				if (a_f.label[i] == kSky) continue;
				const float kD = FreeStep(a_start, 2.0f * a_start, a_f.ray[i]);
				if (!(kD > 0.0f)) continue;
				XMFLOAT4 h;
				XMStoreFloat4(&h, XMVector4Transform(XMVectorSet((x + 0.5f) / n * 2 - 1, 1 - (y + 0.5f) / n * 2, a_f.depth[i], 1), inv));
				const float wx = h.x / h.w, wy = h.y / h.w;
				float       c[4];
				FreeSampleCols(a_cols, (wx - kOrigin) / kExtent, (wy - kOrigin) / kExtent, c);
				const float r = 2.0f * std::sqrt(((wx - kOrigin) / kExtent - 0.5f) * ((wx - kOrigin) / kExtent - 0.5f) + ((wy - kOrigin) / kExtent - 0.5f) * ((wy - kOrigin) / kExtent - 0.5f));
				const float kR = 1.0f - FreeStep(0.85f, 0.95f, r);
				if (a_f.height[i] < kCoverEps) {
					a_k[i] = kD * kR;
					continue;
				}
				const float Hc = std::max(c[0], kCoverHFloor);
				a_kH[i] = 1.0f - FreeStep(kCoverH0 * Hc, kCoverH1 * Hc, a_f.height[i]);
				const int x0 = std::max(x - gCoverBoxR, 0), x1 = std::min(x + gCoverBoxR, n - 1), y0 = std::max(y - gCoverBoxR, 0), y1 = std::min(y + gCoverBoxR, n - 1);
				const int cnt = (x1 - x0 + 1) * (y1 - y0 + 1);
				const int gsum = integ[static_cast<std::size_t>(y1 + 1) * (n + 1) + x1 + 1] - integ[static_cast<std::size_t>(y0) * (n + 1) + x1 + 1] -
				                 integ[static_cast<std::size_t>(y1 + 1) * (n + 1) + x0] + integ[static_cast<std::size_t>(y0) * (n + 1) + x0];
				a_kF[i] = std::clamp((static_cast<float>(gsum) / cnt - gCoverFracDead) / gCoverFracFull, 0.0f, 1.0f);
				if (gCoverTwoSided) {
					auto flag = [&](int a_x, int a_y) { const std::size_t j = static_cast<std::size_t>(a_y) * n + a_x; return a_f.label[j] != kSky && a_f.height[j] < kCoverEps; };
					int l = 0, r = 0, u = 0, d = 0;
					for (int t = 1; t <= gCoverBoxR; ++t) {
						l += x - t >= 0 && flag(x - t, y);
						r += x + t < n && flag(x + t, y);
						u += y - t >= 0 && flag(x, y - t);
						d += y + t < n && flag(x, y + t);
					}
					a_kF[i] *= std::clamp(static_cast<float>(std::max(std::min(l, r), std::min(u, d))), 0.0f, 1.0f);
				}
				a_k[i] = kD * kR * a_kH[i] * a_kF[i];
			}
		}
	}

	// Chebyshev depth of each object pixel inside its object (0 = beside a pixel of another label, ..., capped at a_cap).
	std::vector<std::uint8_t> ObjectDepth(const std::vector<std::uint8_t>& a_label, int a_cap)
	{
		const int                 n = static_cast<int>(kSize);
		std::vector<std::uint8_t> d(a_label.size(), 0);
		for (int y = 0; y < n; ++y) {
			for (int x = 0; x < n; ++x) {
				const std::uint8_t l = a_label[static_cast<std::size_t>(y) * n + x];
				if (l == kSky || l == kGround || l == kBlade) continue;
				int depth = a_cap;
				for (int r = 1; r <= a_cap && depth == a_cap; ++r) {
					for (int dy = -r; dy <= r && depth == a_cap; ++dy) {
						for (int dx = -r; dx <= r; ++dx) {
							if (std::max(std::abs(dx), std::abs(dy)) != r) continue;
							const int xx = std::clamp(x + dx, 0, n - 1), yy = std::clamp(y + dy, 0, n - 1);
							if (a_label[static_cast<std::size_t>(yy) * n + xx] != l) {
								depth = r - 1;
								break;
							}
						}
					}
				}
				d[static_cast<std::size_t>(y) * n + x] = static_cast<std::uint8_t>(depth);
			}
		}
		return d;
	}

	// ---- Round 2 (grasscover/gates.md section 6): CPU models of a temporal history and a distance blur of the fog amount,
	// and one accumulator per candidate row (the same columns as the cover line, plus bfi on the per-frame band). ----
	bool gR2 = false;      // CLI r2: the round-2 rows
	bool gTurn = false;    // CLI turn: the camera yaws kTurnDeg a frame (G12)
	constexpr double kTurnDeg = 2.0;
	constexpr float  kHistNew = 1.0f / 8.0f;   // R2-T: weight of the new frame
	constexpr float  kHistReset = 0.15f;       // R2-T: reset when the distance differs by more than this (relative)
	constexpr float  kBlurStart = 600.0f;      // R2-B: no blur nearer; full tolerance from twice this
	constexpr float  kBlurSigma = 1.5f;        // R2-B: Gaussian over 7 x 7 full-res pixels
	constexpr int    kBlurR = 3;

	struct History
	{
		std::vector<float> a, ray;
		XMFLOAT4X4         vp{};
		bool               valid = false;
	};

	// R2-T: exponential history with reprojection: the pixel's camera-relative point (from its depth and this frame's
	// inverse) is projected with the previous frame's matrix and the history read bilinearly there; reset (new value)
	// off screen, on the sky, or when the nearest history pixel's distance differs by more than kHistReset.
	std::vector<float> Temporal(History& a_h, const FrameData& a_f, const std::vector<float>& a_a)
	{
		const int n = static_cast<int>(kSize);
		if (!a_h.valid) {
			a_h.a = a_a;
			a_h.ray = a_f.ray;
			a_h.vp = a_f.vp;
			a_h.valid = true;
			return a_a;
		}
		std::vector<float> out(a_a);
		const XMMATRIX     inv = XMLoadFloat4x4(&a_f.inv), vpPrev = XMLoadFloat4x4(&a_h.vp);
		for (int y = 0; y < n; ++y) {
			for (int x = 0; x < n; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * n + x;
				if (!(a_f.ray[i] < 1e8f)) continue;
				XMFLOAT4 w;
				XMStoreFloat4(&w, XMVector4Transform(XMVectorSet((x + 0.5f) / n * 2 - 1, 1 - (y + 0.5f) / n * 2, a_f.depth[i], 1), inv));
				XMFLOAT4 c;
				XMStoreFloat4(&c, XMVector4Transform(XMVectorSet(w.x / w.w, w.y / w.w, w.z / w.w, 1.0f), vpPrev));
				if (!(c.w > 1e-6f)) continue;
				const float px = (c.x / c.w + 1.0f) * 0.5f * n - 0.5f, py = (1.0f - c.y / c.w) * 0.5f * n - 0.5f;
				if (!(px >= 0.0f && py >= 0.0f && px <= n - 1.0f && py <= n - 1.0f)) continue;
				const int   nx = static_cast<int>(std::lround(px)), ny = static_cast<int>(std::lround(py));
				const float hr = a_h.ray[static_cast<std::size_t>(ny) * n + nx];
				if (!(hr < 1e8f) || std::fabs(hr - a_f.ray[i]) > kHistReset * a_f.ray[i]) continue;
				const int   x0 = std::min(static_cast<int>(px), n - 2), y0 = std::min(static_cast<int>(py), n - 2);
				const float tx = px - x0, ty = py - y0;
				auto        h = [&](int a_x, int a_y) { return a_h.a[static_cast<std::size_t>(a_y) * n + a_x]; };
				const float hist = (h(x0, y0) * (1 - tx) + h(x0 + 1, y0) * tx) * (1 - ty) + (h(x0, y0 + 1) * (1 - tx) + h(x0 + 1, y0 + 1) * tx) * ty;
				out[i] = hist + kHistNew * (a_a[i] - hist);
			}
		}
		a_h.a = out;
		a_h.ray = a_f.ray;
		a_h.vp = a_f.vp;
		return out;
	}

	// Post-hoc combined row (TaG, TbG): the history only inside the band gate g = smoothstep(600, 1200, d) kH, so nothing
	// nearer than 600 or above the layer band changes (the plain R2-T changes every pixel, G5 and G10b).
	std::vector<float> TemporalGated(History& a_h, const FrameData& a_f, const std::vector<float>& a_a, const std::vector<float>& a_kh);

	// R2-B: the height gate per pixel (1 - smoothstep(1.25 Hc, 2 Hc, h), Hc = max(H at the hit point, 48); 0 on the sky).
	std::vector<float> HeightGate(const FrameData& a_f, const std::vector<float>& a_cols)
	{
		const int          n = static_cast<int>(kSize);
		std::vector<float> kh(a_f.label.size(), 0.0f);
		const XMMATRIX     inv = XMLoadFloat4x4(&a_f.inv);
		for (int y = 0; y < n; ++y) {
			for (int x = 0; x < n; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * n + x;
				if (a_f.label[i] == kSky) continue;
				XMFLOAT4 h;
				XMStoreFloat4(&h, XMVector4Transform(XMVectorSet((x + 0.5f) / n * 2 - 1, 1 - (y + 0.5f) / n * 2, a_f.depth[i], 1), inv));
				float c[4];
				FreeSampleCols(a_cols, (h.x / h.w - kOrigin) / kExtent, (h.y / h.w - kOrigin) / kExtent, c);
				const float Hc = std::max(c[0], kCoverHFloor);
				kh[i] = 1.0f - FreeStep(kCoverH0 * Hc, kCoverH1 * Hc, a_f.height[i]);
			}
		}
		return kh;
	}

	// R2-B: distance-proportional blur of the fog amount (see gates.md section 6).
	std::vector<float> BlurDist(const FrameData& a_f, const std::vector<float>& a_a, const std::vector<float>& a_kh, float a_tol)
	{
		const int          n = static_cast<int>(kSize);
		std::vector<float> out(a_a);
		float              g[2 * kBlurR + 1];
		for (int t = -kBlurR; t <= kBlurR; ++t) g[t + kBlurR] = std::exp(-0.5f * t * t / (kBlurSigma * kBlurSigma));
		for (int y = 0; y < n; ++y) {
			for (int x = 0; x < n; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * n + x;
				const float       d = a_f.ray[i];
				if (!(d >= kBlurStart && d < 1e8f) || !(a_kh[i] > 0.0f)) continue;
				const float tolE = a_tol * FreeStep(kBlurStart, 2.0f * kBlurStart, d);
				if (!(tolE > 0.0f)) continue;
				float sum = 0.0f, ws = 0.0f;
				for (int dy = -kBlurR; dy <= kBlurR; ++dy) {
					const int yy = y + dy;
					if (yy < 0 || yy >= n) continue;
					for (int dx = -kBlurR; dx <= kBlurR; ++dx) {
						const int xx = x + dx;
						if (xx < 0 || xx >= n) continue;
						const std::size_t j = static_cast<std::size_t>(yy) * n + xx;
						const float       rel = std::fabs(a_f.ray[j] - d) / d;
						if (!(rel <= tolE)) continue;
						const float w = g[dx + kBlurR] * g[dy + kBlurR] * (1.0f - rel / tolE);
						sum += w * a_a[j];
						ws += w;
					}
				}
				if (ws > 0.0f) out[i] = a_a[i] + a_kh[i] * (sum / ws - a_a[i]);
			}
		}
		return out;
	}

	std::vector<float> TemporalGated(History& a_h, const FrameData& a_f, const std::vector<float>& a_a, const std::vector<float>& a_kh)
	{
		History           h = a_h;
		std::vector<float> t = Temporal(h, a_f, a_a);
		std::vector<float> out(a_a);
		for (std::size_t i = 0; i < out.size(); ++i) {
			const float g = a_f.ray[i] < 1e8f ? FreeStep(kBlurStart, 2.0f * kBlurStart, a_f.ray[i]) * a_kh[i] : 0.0f;
			out[i] = a_a[i] + g * (t[i] - a_a[i]);
		}
		a_h.a = out;   // the history holds what was shown
		a_h.ray = a_f.ray;
		a_h.vp = a_f.vp;
		a_h.valid = true;
		return out;
	}

	struct RowAcc
	{
		std::vector<double> s, s2;
		std::vector<float>  prev, prevNow;
		double    tmp[2]{}, bf[6]{}, bfi[2]{}, K = 0.0, sb[2]{}, sp = 0.0, pl = 0.0, dP = 0.0, dG[3]{}, out = 0.0, E = 0.0, obj[8][6]{};
		long long tmpN = 0, bfN[3]{}, bfiN = 0, KN = 0, plN = 0, gain[2]{}, objN[8][6]{}, eN = 0;
		void      Init(std::size_t a_nb)
		{
			s.assign(a_nb, 0.0);
			s2.assign(a_nb, 0.0);
			prev.assign(a_nb, 0.0f);
			prevNow.assign(a_nb, 0.0f);
		}
	};

	struct RowCtx
	{
		const FrameData*                 fr;
		const std::vector<float>*        aNow;
		const std::vector<float>*        aEx;
		const std::vector<float>*        aBf;
		const std::vector<std::size_t>*  band;
		const std::vector<std::uint8_t>* outline;
		const std::vector<std::uint8_t>* ring;
		const std::vector<std::uint8_t>* nearObj;
		const std::vector<std::uint8_t>* objDepth;
		float                            outH, start;
		int                              f;
	};

	void AccRow(RowAcc& a_r, const RowCtx& a_c, const std::vector<float>& a_a)
	{
		const FrameData& fr = *a_c.fr;
		const auto&      aNow = *a_c.aNow;
		const auto&      aEx = *a_c.aEx;
		const auto&      aBf = *a_c.aBf;
		auto             lum = [&](std::size_t i, double a) { return (fr.label[i] == kBlade ? kLeafC : kGroundC) * (1.0 - a) + kFogC * a; };
		const auto&      band = *a_c.band;
		for (std::size_t b = 0; b < band.size(); ++b) {
			const std::size_t i = band[b];
			const double      l = lum(i, a_a[i]);
			a_r.s[b] += l;
			a_r.s2[b] += l * l;
			a_r.E += std::fabs(a_a[i] - aEx[i]);
			++a_r.eN;
			if (a_c.f > 0) {
				a_r.tmp[0] += std::fabs(a_a[i] - a_r.prev[b]);
				a_r.tmp[1] += std::fabs(aNow[i] - a_r.prevNow[b]);
				++a_r.tmpN;
			}
			a_r.prev[b] = a_a[i];
			a_r.prevNow[b] = aNow[i];
			const int cls = fr.label[i] == kBlade ? 1 : 2;
			a_r.bf[0] += std::fabs(a_a[i] - aBf[i]);
			a_r.bf[3] += std::fabs(aNow[i] - aBf[i]);
			a_r.bf[cls] += std::fabs(a_a[i] - aBf[i]);
			a_r.bf[cls + 3] += std::fabs(aNow[i] - aBf[i]);
			++a_r.bfN[0];
			++a_r.bfN[cls];
			if (fr.label[i] != kBlade) continue;
			const int x = static_cast<int>(i % kSize), y = static_cast<int>(i / kSize);
			for (int dx = 1; dx <= kNearSearch; ++dx) {
				double gs = 0.0, nn = 0.0;
				for (const int side : { -dx, dx }) {
					const int xx = x + side;
					if (xx < 0 || xx >= static_cast<int>(kSize)) continue;
					const std::size_t j = static_cast<std::size_t>(y) * kSize + xx;
					if (fr.label[j] != kGround) continue;
					gs += lum(j, a_a[j]);
					nn += 1.0;
				}
				if (nn == 0.0) continue;
				a_r.K += std::fabs(lum(i, a_a[i]) - gs / nn);
				++a_r.KN;
				break;
			}
		}
		for (std::size_t i = 0; i < a_a.size(); ++i) {
			const std::uint8_t l = fr.label[i];
			const double       dn = std::fabs(a_a[i] - aNow[i]);
			if (InBand(fr, i)) {
				a_r.bfi[0] += std::fabs(a_a[i] - aBf[i]);
				a_r.bfi[1] += std::fabs(aNow[i] - aBf[i]);
				++a_r.bfiN;
			}
			if ((*a_c.outline)[i] && !fr.arm[i] && l != kBlade) {
				a_r.sb[0] = std::max(a_r.sb[0], static_cast<double>(std::fabs(aNow[i] - aEx[i])));
				a_r.sb[1] = std::max(a_r.sb[1], static_cast<double>(std::fabs(a_a[i] - aEx[i])));
			}
			if ((*a_c.ring)[i]) {
				a_r.sp = std::max(a_r.sp, static_cast<double>(std::fabs(a_a[i] - aEx[i])));
				a_r.dP = std::max(a_r.dP, dn);
			}
			if (l == kPlayer) {
				a_r.pl += a_a[i];
				++a_r.plN;
				a_r.dP = std::max(a_r.dP, dn);
			}
			if (l == kGround) {
				a_r.dG[0] = std::max(a_r.dG[0], dn);
				a_r.dG[(*a_c.nearObj)[i] ? 2 : 1] = std::max(a_r.dG[(*a_c.nearObj)[i] ? 2 : 1], dn);
			}
			if (l != kSky && (fr.ray[i] < a_c.start || fr.height[i] > a_c.outH)) a_r.out = std::max(a_r.out, dn);
			if (l == kBox) a_r.gain[0] += a_a[i] - aNow[i] > kGainEps ? 1 : 0;
			if (l == kDiamond) a_r.gain[1] += a_a[i] - aNow[i] > kGainEps ? 1 : 0;
			const int o = l == kTrunk ? 0 : (l == kPerson && !fr.arm[i]) ? 1 : l == kBox ? 2 : l == kDiamond ? 3 : l == kBigBox1 ? 4 : l == kBigBox2 ? 5 : l == kBigTrunk ? 6 : l == kBigBody ? 7 : -1;
			if (o >= 0) {
				const int dep = (*a_c.objDepth)[i];
				a_r.obj[o][dep] += dn;
				++a_r.objN[o][dep];
				a_r.obj[o][5] += dn;
				++a_r.objN[o][5];
			}
		}
	}

	// One row printed as the cover line (same keys, so gate_cover.py reads it with the word "r2 <name>"), a dummy "+" line
	// (summary.py expects one) and the objects line.
	void PrintRow(const char* a_name, const RowAcc& a_r, const double* a_t, std::size_t a_nb)
	{
		double tR = 0.0;
		for (std::size_t b = 0; b < a_nb; ++b) {
			const double m = a_r.s[b] / kFrames;
			tR += std::sqrt(std::max(a_r.s2[b] / kFrames - m * m, 0.0));
		}
		tR = a_nb ? tR / a_nb : 0.0;
		auto ratio = [](double a, double b) { return b > 1e-9 ? a / b : 0.0; };
		auto mean = [](double a_s, long long a_n) { return a_n ? a_s / static_cast<double>(a_n) : 0.0; };
		char kText[16] = "-";
		if (a_r.KN) std::snprintf(kText, sizeof(kText), "%.4f", a_r.K / a_r.KN);
		std::printf("  r2 %s: T %7.4f cov/no %.6f cov/ex %.6f cov/now %.6f K %s Sb %.6f %.6f Sp %.4f player %.6f dP %.9g dG %.6f far %.6f near %.6f out %.9g"
		            " gain box %lld diamond %lld tmp now %.6f cov %.6f cov/now %.6f thin 0 oracle 0 bf cov %.6f %.6f %.6f now %.6f %.6f %.6f E %.4f bfi %.6f %.6f\n",
			a_name, tR, ratio(tR, a_t[2]), ratio(tR, a_t[1]), ratio(tR, a_t[0]), kText, a_r.sb[0], a_r.sb[1], a_r.sp, mean(a_r.pl, a_r.plN), a_r.dP, a_r.dG[0], a_r.dG[1],
			a_r.dG[2], a_r.out, a_r.gain[0], a_r.gain[1], mean(a_r.tmp[1], a_r.tmpN), mean(a_r.tmp[0], a_r.tmpN), ratio(mean(a_r.tmp[0], a_r.tmpN), mean(a_r.tmp[1], a_r.tmpN)),
			mean(a_r.bf[0], a_r.bfN[0]), mean(a_r.bf[1], a_r.bfN[1]), mean(a_r.bf[2], a_r.bfN[2]), mean(a_r.bf[3], a_r.bfN[0]), mean(a_r.bf[4], a_r.bfN[1]),
			mean(a_r.bf[5], a_r.bfN[2]), mean(a_r.E, a_r.eN), mean(a_r.bfi[0], a_r.bfiN), mean(a_r.bfi[1], a_r.bfiN));
		std::printf("  r2 %s+ 600: k blade full 0 part 0 zero 0 lim height 0 frac 0 hpart 0 blade px 0 dk 0 cont 0 len 0\n", a_name);
		static const char* const kObjName[8]{ "trunk", "body", "box", "diamond", "box70", "box140", "bigtrunk", "bigbody" };
		std::string objText;
		for (int o = 0; o < 8; ++o) {
			if (!a_r.objN[o][5]) continue;
			char buf[256];
			std::snprintf(buf, sizeof(buf), " %s n %lld %lld %lld %lld %lld d %.5f %.5f %.5f %.5f %.5f in %lld %.6f all %.6f |", kObjName[o], a_r.objN[o][0], a_r.objN[o][1],
				a_r.objN[o][2], a_r.objN[o][3], a_r.objN[o][4], mean(a_r.obj[o][0], a_r.objN[o][0]), mean(a_r.obj[o][1], a_r.objN[o][1]), mean(a_r.obj[o][2], a_r.objN[o][2]),
				mean(a_r.obj[o][3], a_r.objN[o][3]), mean(a_r.obj[o][4], a_r.objN[o][4]), a_r.objN[o][3] + a_r.objN[o][4],
				mean(a_r.obj[o][3] + a_r.obj[o][4], a_r.objN[o][3] + a_r.objN[o][4]), mean(a_r.obj[o][5], a_r.objN[o][5]));
			objText += buf;
		}
		std::printf("  r2 %sobj 600:%s\n", a_name, objText.c_str());
	}

	// ---- Round 5 (CLI sink, grasscover/gates.md sections 9, 9a, 9b): what the rows S, V, C, SC, VC (and S+e, S-e) need beyond RowAcc ----
	constexpr float        kSinkFbH = 40.0f;   // smoke check: own-ray band blade pixels 8 to 40 units up (well inside the sink) must take more fog in S
	constexpr int          kSinkObjs = 9;
	constexpr std::uint8_t kSinkObjLabel[kSinkObjs]{ kTrunk, kPerson, kPlayer, kBox, kDiamond, kBigBox1, kBigBox2, kBigTrunk, kBigBody };
	constexpr const char*  kSinkObjName[kSinkObjs]{ "trunk", "person", "player", "box", "diamond", "box70", "box140", "bigtrunk", "bigbody" };
	constexpr int          kSinkHBins = 5;     // object pixels by height over the ground: 0-28, 28-56, 56-84, 84-98, 98+ (G18 the first three,
	constexpr float        kSinkHEdge[kSinkHBins + 1]{ -1.0e30f, 28.0f, 56.0f, 84.0f, 98.0f, 1.0e30f };   // 9a 0-56 = the first two, G4s the last)
	constexpr int          kSinkBins = 5;      // distance bands by the frame-0 ray: 0-10, 10-20, 20-30, 30-50, 50+ m (70 units a metre)
	constexpr float        kSinkBinEdge[kSinkBins + 1]{ 0.0f, 700.0f, 1400.0f, 2100.0f, 3500.0f, 1.0e30f };

	struct SinkAcc
	{
		long long g16n = 0, g16lo = 0, g16bad = 0, g16loNow = 0;   // pixel-frames; a_row < a_ref - 1e-4 (ref: now, or C for SC, VC);
		double    g16min = 1e30, g16max = -1e30, g16minNow = 1e30;  // not finite or outside [0, 1]; a_row < a_now - 1e-4 (info)
		double    sky = 0.0, skyRef = 0.0;                          // G5s: max |a_row - a_now| over sky pixels; and against the ref
		long long skyN = 0, neq = 0;                                 // neq: pixel-frames whose alpha is not bit-identical to today's
		std::vector<float> da[kSinkObjs][kSinkHBins];               // a_row - a_now per object pixel-frame by height bin
		double    aRow[kSinkObjs][kSinkHBins]{}, aNow[kSinkObjs][kSinkHBins]{};
		long long fbN = 0, fbUp = 0;                                // own-ray band blade pixels under kSinkFbH: all, a_row > a_now
		long long lowN = 0, lowDiff = 0, ownN = 0, nonSky = 0;      // texel distances differing from today's; own-ray pixels (row's low)
		std::vector<double> s, s2;                                  // distance-band T: luminance sums over the case's pixel set
		void      Init(std::size_t a_n)
		{
			s.assign(a_n, 0.0);
			s2.assign(a_n, 0.0);
		}
	};

	// The distance-band pixel set of a case, from frame 0: ground or blade pixels whose 5 x 5 window holds only ground and
	// blades (no sky, no object), any distance, binned by the frame-0 ray.
	void SinkBandSet(const FrameData& a_f, std::vector<std::size_t>& a_px, std::vector<int>& a_bin)
	{
		a_px.clear();
		a_bin.clear();
		for (int y = kWindow; y + kWindow < static_cast<int>(kSize); ++y) {
			for (int x = kWindow; x + kWindow < static_cast<int>(kSize); ++x) {
				bool ok = true;
				for (int dy = -kWindow; dy <= kWindow && ok; ++dy) {
					for (int dx = -kWindow; dx <= kWindow; ++dx) {
						const std::uint8_t l = a_f.label[static_cast<std::size_t>(y + dy) * kSize + (x + dx)];
						if (l != kGround && l != kBlade) {
							ok = false;
							break;
						}
					}
				}
				if (!ok) continue;
				const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
				int               b = 0;
				while (b + 1 < kSinkBins && a_f.ray[i] >= kSinkBinEdge[b + 1]) ++b;
				a_px.push_back(i);
				a_bin.push_back(b);
			}
		}
	}

	// One row's frame: G16, G5s, the objects by height, the smoke check, the distance-band sums, the texel distances and
	// the own-ray share (a_ref: C for SC and VC, else now; a_low, a_lowNow: the row's and today's march texture distances).
	void SinkFrameAcc(SinkAcc& a_s, const FrameData& a_f, const std::vector<float>& a_a, const std::vector<float>& a_now, const std::vector<float>& a_ref,
		const std::vector<std::uint8_t>& a_own, const std::vector<float>& a_low, const std::vector<float>& a_lowNow, const std::vector<std::size_t>& a_px)
	{
		for (std::size_t t = 0; t < a_low.size(); ++t) {
			++a_s.lowN;
			a_s.lowDiff += std::memcmp(&a_low[t], &a_lowNow[t], sizeof(float)) != 0 ? 1 : 0;
		}
		for (std::size_t i = 0; i < a_a.size(); ++i) {
			const double d = static_cast<double>(a_a[i]) - a_now[i], dr = static_cast<double>(a_a[i]) - a_ref[i];
			++a_s.g16n;
			a_s.neq += std::memcmp(&a_a[i], &a_now[i], sizeof(float)) != 0 ? 1 : 0;
			a_s.g16lo += dr < -1e-4 ? 1 : 0;
			a_s.g16loNow += d < -1e-4 ? 1 : 0;
			a_s.g16bad += !std::isfinite(a_a[i]) || a_a[i] < 0.0f || a_a[i] > 1.0f ? 1 : 0;
			a_s.g16min = std::min(a_s.g16min, dr);
			a_s.g16minNow = std::min(a_s.g16minNow, d);
			a_s.g16max = std::max(a_s.g16max, static_cast<double>(a_a[i]));
			const std::uint8_t l = a_f.label[i];
			if (l == kSky) {
				++a_s.skyN;
				a_s.sky = std::max(a_s.sky, std::fabs(d));
				a_s.skyRef = std::max(a_s.skyRef, std::fabs(dr));
				continue;
			}
			++a_s.nonSky;
			a_s.ownN += a_own[i];
			if (l == kBlade && a_own[i] && a_f.height[i] >= 2.0f * kCoverEps && a_f.height[i] < kSinkFbH && InBand(a_f, i)) {
				++a_s.fbN;
				a_s.fbUp += d > 0.0 ? 1 : 0;
			}
			for (int o = 0; o < kSinkObjs; ++o) {
				if (l != kSinkObjLabel[o]) continue;
				int b = 0;
				while (b + 1 < kSinkHBins && a_f.height[i] >= kSinkHEdge[b + 1]) ++b;
				a_s.da[o][b].push_back(static_cast<float>(d));
				a_s.aRow[o][b] += a_a[i];
				a_s.aNow[o][b] += a_now[i];
				break;
			}
		}
		for (std::size_t k = 0; k < a_px.size(); ++k) {
			const std::size_t i = a_px[k];
			const double      lm = (a_f.label[i] == kBlade ? kLeafC : kGroundC) * (1.0 - a_a[i]) + kFogC * a_a[i];
			a_s.s[k] += lm;
			a_s.s2[k] += lm * lm;
		}
	}

	// Distance-band T (mean over the band's pixels of the temporal std of L) per bin; a_n = pixels per bin.
	void SinkBandT(const SinkAcc& a_s, const std::vector<int>& a_bin, double* a_t, long long* a_n)
	{
		for (int b = 0; b < kSinkBins; ++b) {
			a_t[b] = 0.0;
			a_n[b] = 0;
		}
		for (std::size_t k = 0; k < a_bin.size(); ++k) {
			const double m = a_s.s[k] / kFrames;
			a_t[a_bin[k]] += std::sqrt(std::max(a_s.s2[k] / kFrames - m * m, 0.0));
			++a_n[a_bin[k]];
		}
		for (int b = 0; b < kSinkBins; ++b) a_t[b] = a_n[b] ? a_t[b] / static_cast<double>(a_n[b]) : 0.0;
	}

	// The 99th percentile (nearest rank) of a copy of the values.
	double SinkP99(std::vector<float> a_v)
	{
		if (a_v.empty()) return 0.0;
		const std::size_t k = static_cast<std::size_t>(std::ceil(0.99 * static_cast<double>(a_v.size()))) - 1;
		std::nth_element(a_v.begin(), a_v.begin() + static_cast<std::ptrdiff_t>(k), a_v.end());
		return a_v[k];
	}

	// Lines per row: "sink <row>:" (G16, G5s, smoke check, texel distances, own-ray share), "sink <row> obj:" (per object and
	// height bin: n, mean / p99 / max of a_row - a_now, mean and p99 of |a_row - a_now|, mean alpha row and now), "sink <row> T:".
	void PrintSink(const char* a_name, const char* a_ref, const SinkAcc& a_s, const SinkAcc& a_now, const std::vector<int>& a_bin)
	{
		auto sh = [](long long a_n, long long a_d) { return a_d ? static_cast<double>(a_n) / static_cast<double>(a_d) : 0.0; };
		std::printf("  sink %s: ref %s g16 n %lld lo %lld min %.9g bad %lld max %.9g now lo %lld min %.9g sky n %lld max %.9g ref %.9g fb n %lld up %lld"
		            " low diff %lld of %lld own %.9f neq %lld\n",
			a_name, a_ref, a_s.g16n, a_s.g16lo, a_s.g16min, a_s.g16bad, a_s.g16max, a_s.g16loNow, a_s.g16minNow, a_s.skyN, a_s.sky, a_s.skyRef, a_s.fbN, a_s.fbUp,
			a_s.lowDiff, a_s.lowN, sh(a_s.ownN, a_s.nonSky), a_s.neq);
		std::string ob;
		for (int o = 0; o < kSinkObjs; ++o) {
			bool any = false;
			for (int b = 0; b < kSinkHBins; ++b) any = any || !a_s.da[o][b].empty();
			if (!any) continue;
			ob += std::string(" ") + kSinkObjName[o];
			for (int b = 0; b < kSinkHBins; ++b) {
				const auto&        v = a_s.da[o][b];
				std::vector<float> av(v.size());
				double             sum = 0.0, asum = 0.0, mx = -1e30;
				for (std::size_t k = 0; k < v.size(); ++k) {
					sum += v[k];
					asum += std::fabs(v[k]);
					mx = std::max(mx, static_cast<double>(v[k]));
					av[k] = std::fabs(v[k]);
				}
				const double n = static_cast<double>(v.size());
				char         buf[256];
				if (v.empty()) {
					std::snprintf(buf, sizeof(buf), " b%d n 0", b);
				} else {
					std::snprintf(buf, sizeof(buf), " b%d n %zu d %.9g %.9g %.9g ad %.9g %.9g a %.9g %.9g", b, v.size(), sum / n, SinkP99(v), mx, asum / n, SinkP99(av),
						a_s.aRow[o][b] / n, a_s.aNow[o][b] / n);
				}
				ob += buf;
			}
			ob += " |";
		}
		std::printf("  sink %s obj:%s\n", a_name, ob.c_str());
		double    tr[kSinkBins], tn[kSinkBins];
		long long nr[kSinkBins], nn[kSinkBins];
		SinkBandT(a_s, a_bin, tr, nr);
		SinkBandT(a_now, a_bin, tn, nn);
		std::string bt;
		for (int b = 0; b < kSinkBins; ++b) {
			char buf[128];
			std::snprintf(buf, sizeof(buf), " %d n %lld now %.6f row %.6f ratio %.6f |", b, nn[b], tn[b], tr[b], tn[b] > 1e-12 ? tr[b] / tn[b] : 0.0);
			bt += buf;
		}
		std::printf("  sink %s T:%s\n", a_name, bt.c_str());
	}

	int Run(const Gpu& g, const std::string& a_outDir, bool a_quick, bool a_dump, bool a_dumpAll, float a_thinStart, const std::string& a_only, float a_oracleStart,
		bool a_boxes = false, bool a_oracleOnThin = false, const FreeRun& a_free = FreeRun{}, float a_coverStart = 0.0f, bool a_bigObj = false)
	{
		const bool thin = a_thinStart > 0.0f;
		const bool oracle = a_oracleStart > 0.0f;
		const bool freeOn = a_free.p.start > 0.0f;
		const bool cover = a_coverStart > 0.0f;
		const int  grounds = a_free.tilt ? 3 : 2;   // 0 flat, 1 rising 10 degrees along +y, 2 (tilt) rising 10 degrees along +x
		// Columns per ground: the wind field at rest (as the main run, at the game's sizes) and the even layer.
		WindSim             sims[3];
		ID3D11Texture2D*    evenTex[3]{};
		ID3D11ShaderResourceView* evenSrv[3]{};
		ID3D11Texture2D*    hTex[3]{};
		ID3D11ShaderResourceView* hSrv[3]{};
		std::vector<float>  colsCpu[3][2];   // free, cover: CPU copies of the column textures (wind field, even layer)
		float               hMaxAll[3][2]{};
		for (int slope = 0; slope < grounds; ++slope) {
			const double tanG = slope == 1 ? std::tan(XMConvertToRadians(10.0f)) : 0.0;
			const double tanGX = slope == 2 ? std::tan(XMConvertToRadians(kTiltDeg)) : 0.0;
			std::string  error;
			if (!sims[slope].Create(g.dev, kRes, error)) {
				std::fprintf(stderr, "WindSim: %s\n", error.c_str());
				return 1;
			}
			sims[slope].Reset(g.ctx);
			std::vector<float> heights(static_cast<std::size_t>(kHeightSize) * kHeightSize);
			for (int hy = 0; hy < kHeightSize; ++hy) {
				for (int hx = 0; hx < kHeightSize; ++hx) heights[static_cast<std::size_t>(hy) * kHeightSize + hx] =
						static_cast<float>(tanG * (kOrigin + hy * kExtent / (kHeightSize - 1)) + tanGX * (kOrigin + hx * kExtent / (kHeightSize - 1)));
			}
			D3D11_TEXTURE2D_DESC hd{};
			hd.Width = hd.Height = kHeightSize;
			hd.MipLevels = hd.ArraySize = 1;
			hd.Format = DXGI_FORMAT_R32_FLOAT;
			hd.SampleDesc.Count = 1;
			hd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			D3D11_SUBRESOURCE_DATA hinit{ heights.data(), kHeightSize * sizeof(float), 0 };
			g.dev->CreateTexture2D(&hd, &hinit, &hTex[slope]);
			g.dev->CreateShaderResourceView(hTex[slope], nullptr, &hSrv[slope]);
			WindShaders::SimCB scb{};
			scb.grid[0] = kOrigin;
			scb.grid[1] = kOrigin;
			scb.grid[2] = kExtent / kRes;
			scb.grid[3] = static_cast<float>(kRes);
			scb.time[0] = 1.0f / 30.0f;
			scb.shift[3] = static_cast<float>(kHeightSize);
			scb.counts[2] = 1.2f;
			scb.heightGrid[0] = kOrigin;
			scb.heightGrid[1] = kOrigin;
			scb.heightGrid[2] = kExtent / (kHeightSize - 1);
			scb.heightGrid[3] = 1.0f;
			sims[slope].Step(g.ctx, scb, hSrv[slope], 5);
			sims[slope].BuildColumns(g.ctx, hSrv[slope], kThickness, 0.0f, 0.0f, 0.0f);
			// Even layer: z = ground at the texel centre (as CSColumn's ToWorld), so the linear filter gives the plane exactly.
			std::vector<float> even(static_cast<std::size_t>(kRes) * kRes * 4);
			for (int cy = 0; cy < kRes; ++cy) {
				for (int cx = 0; cx < kRes; ++cx) {
					float* t = &even[(static_cast<std::size_t>(cy) * kRes + cx) * 4];
					t[0] = kEvenH;
					t[1] = kEvenW;
					t[2] = static_cast<float>(tanG * (kOrigin + (cy + 0.5) * kExtent / kRes) + tanGX * (kOrigin + (cx + 0.5) * kExtent / kRes));
					t[3] = 1.0f;
				}
			}
			D3D11_TEXTURE2D_DESC ed = hd;
			ed.Width = ed.Height = kRes;
			ed.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			D3D11_SUBRESOURCE_DATA einit{ even.data(), kRes * sizeof(float) * 4, 0 };
			g.dev->CreateTexture2D(&ed, &einit, &evenTex[slope]);
			g.dev->CreateShaderResourceView(evenTex[slope], nullptr, &evenSrv[slope]);
			const auto cols = ReadFloats(g.dev, g.ctx, sims[slope].ColumnTexture(), kRes, kRes, 4);
			if (freeOn || cover) {
				colsCpu[slope][0] = cols;
				colsCpu[slope][1] = even;
			}
			hMaxAll[slope][0] = kCoverHFloor;   // GrassCover gate G5: the largest column H anywhere (at least the gate's floor)
			hMaxAll[slope][1] = std::max(kEvenH, kCoverHFloor);
			for (int t = 0; t < kRes * kRes; ++t) hMaxAll[slope][0] = std::max(hMaxAll[slope][0], cols[static_cast<std::size_t>(t) * 4]);
			double     sumH = 0.0, sumW = 0.0, maxH = 0.0;
			int        cnt = 0;
			for (int cy = 0; cy < kRes; ++cy) {
				for (int cx = 0; cx < kRes; ++cx) {
					const double wx = kOrigin + (cx + 0.5) * kExtent / kRes, wy = kOrigin + (cy + 0.5) * kExtent / kRes;
					const double r = std::sqrt(wx * wx + wy * wy);
					if (wy < 0.0 || r < kBand0 || r > kBand1) continue;
					const float* c = &cols[(static_cast<std::size_t>(cy) * kRes + cx) * 4];
					sumH += c[0];
					sumW += c[1];
					maxH = std::max(maxH, static_cast<double>(c[0]));
					++cnt;
				}
			}
			std::printf("grass: %s ground, wind field columns over the band ahead: H mean %.2f (max %.2f), W mean %.4f over %d texels; even layer H %.1f W %.4f\n",
				slope == 2 ? "tilted (10 degrees along x)" : slope ? "10-degree" : "flat", cnt ? sumH / cnt : 0.0, maxH, cnt ? sumW / cnt : 0.0, cnt, kEvenH, kEvenW);
		}

		std::printf("grass: %u x %u px over 60 degrees, field %.0f (%d texels), %d steps (exact: 64 at full resolution, exact24: %d at full resolution), %d frames, jitter Halton(2,3) %d phases\n",
			kSize, kSize, kExtent, kRes, kSteps, kSteps, kFrames, kPhases);
		std::printf("columns: T = mean over band pixels of the temporal std of L (now, exact, no fog) and the ratios; E = mean |a_now - a_exact|; K = mean |L blade - L ground beside it| (now, exact);\n"
		            "         S = max |a_now - a_exact| within %d px of the trunk and person outlines, Sp the same for the player, player = mean alpha on the player (now, exact);\n"
		            "         F = share of pixels marching their own ray (band, all non-sky); cf = max |a_exact - closed form| in the band (even layer on flat ground only);\n"
		            "         x24 T = T of exact24, now/x24 = the upsampling's share, x24/ex = the step count's (ratios of T); Eu = mean |a_now - a_x24|, Es = mean |a_x24 - a_exact|\n", kOutline);
		if (thin) {
			std::printf("thin %.0f (second line per case, the half-resolution path with ThinFogStart %.0f): T, its ratios to no fog, exact and now; E, K, S as above;\n"
			            "         Sb = S over the outline band's trunk, body and ground pixels only, no blades, no arm (now, thin); Sp, player as above;\n"
			            "         dG = max |a_thin - a_now| over ground pixels, dBand over band pixels,\n"
			            "         dP over the player and its %d px ring, dBody over trunk and body (not the arm), dArm max and mean over the arm; chg = share of band pixels that changed\n",
				a_thinStart, a_thinStart, kOutline);
		}
		if (oracle) {
			std::printf("oracle %.0f (own line per case; harness only, the cover's upper bound): every blade pixel at least %.0f units out (ray) takes the alpha the\n"
			            "         same frame without blades gets there (DebugView 0; same jitter and noise, objects kept), all other pixels keep now's:\n"
			            "         T and its ratios to no fog, exact and now (unrounded); E, K, Sb (now, oracle), Sp, player as above; off = max |a_oracle - a_now| over\n"
			            "         non-blade pixels (must be 0); sub = blade pixels replaced, near = blade pixels nearer than the start, bad = replaced pixels whose\n"
			            "         blade-free hit is not ground or not farther; cf (even layer, flat ground): max |a_oracle - closed form at height 0 and the ground's ray|,\n"
			            "         max |a_oracle - closed form at the blade's own point|, max |exact without blades - closed form at the ground|;\n"
			            "         Tfog = T of the blade-free render as ground, Tlab = T with each pixel's alpha frozen at its mean over the frames (blade/ground swaps\n"
			            "         only), flip = share of band pixels whose label changes over the frames, T st/fl = T of the oracle (now) over the steady / flipping\n"
			            "         pixels, gnd = mean |a_now - a_blade-free| over band ground pixels; after |: bf T = T when every band pixel (blade and ground)\n"
			            "         takes the blade-free alpha, its ratio to now's T and its T over the flipping pixels; mean alpha over band blade pixels (now,\n"
			            "         blade-free) and band ground pixels (now, blade-free), all frames\n",
				a_oracleStart, a_oracleStart);
		}
		if (freeOn) {
			std::printf("free %.0f %s%s (own line per case; harness only, ledger \"grass mosaic\" S1 / S2): the fog reads an effective depth built on the CPU and the\n"
			            "         frame is rendered once more with it (DebugView 0, ThinFogStart 0). ref = the candidate depth is the same frame without blades (the\n"
			            "         ceiling), close = a horizontal closing of the linear depth over Lpx = clamp(round(%.0f / (z x pixel angle)), %d, maxpx) pixels with the\n"
			            "         tip veto (Lpx / 2 rows below); gates: start to 2 x start by distance, %.3f-%.2f and %.2f-%.2f farther, %.0f-%.0f units over the column's ground, the field's\n"
			            "         round fade. -lpx1 = Lpx forced to 1, -noveto = no tip veto, -nogate = k 1 wherever the candidate is farther (mutations, self-check a).\n"
			            "         In the scenes without blades the blade-free frame is the frame itself.\n"
			            "         T and its ratios to no fog, exact, now and bf (bf = every band pixel on the blade-free alpha); K; Sb (now, free), Sp, player as above;\n"
			            "         dG = max |a_free - a_now| over ground pixels; gain = box / diamond pixel-frames with a_free - a_now > %.3f, px = their pixel-frames;\n"
			            "         closed / part / veto = share of the band's blade pixel-frames with k >= %.2f / 0 < k < %.2f / k taken to 0 by the tip veto alone;\n"
			            "         gap = mean |z_eff - z_blade-free| / z_blade-free over the closed ones; Eg = mean |a - a_blade-free| over band ground pixels (now, free);\n"
			            "         flipd = over the band pixels whose label flips, mean |mean alpha as blade - mean alpha as ground| (now, free);\n"
			            "         obj = trunk, body, player, box and diamond pixel-frames whose depth changed, of all; minrun = the narrowest near-run of those objects\n"
			            "         among the runs the closing ran on (px), L = the closing length there, short = such runs narrower than theirs; F as above, on the\n"
			            "         effective depth; after |: bf/now; E; dP = max |a_free - a_now| over the player and its %d px ring; dBody over trunk and body; arm = arm\n"
			            "         pixel-frames whose depth changed, of all, dArm max and mean |a_free - a_now|; post likewise and its mean alpha (now, free); dpx =\n"
			            "         pixel-frames whose depth changed (all; not blade, arm or post); dbf = pixel-frames whose alpha differs from the blade-free render's;\n"
			            "         Fnow = F of now; flip = share of band pixels whose label flips; blade = band blade pixel-frames; maxpx; objs = changed pixel-frames of\n"
			            "         trunk, body, player, box, diamond\n",
				a_free.p.start, kFreeSrcName[a_free.p.src], kFreeMutName[a_free.p.mut], kFreeWidth, kFreeMinPx, kFreeRel0, kFreeRel1, kFreeRelMax0, kFreeRelMax1, kFreeH0, kFreeH1,
				kGainEps, kFreeClosedK, kFreeClosedK, kOutline);
			if (a_free.orderRev) std::printf("         order rev (second line): the free render repeated over the frames in reverse order; alpha pixels that differ from the first pass\n");
		}
		if (cover) {
			std::printf("cover %.0f (GrassCover prototype, own lines per case; grasscover/gates.md): the half-resolution path with GrassCover %.0f (ThinFogStart 0):\n"
			            "         T and its ratios to no fog, exact and now; K; Sb (now, cover); Sp; player; dP = max |a_cover - a_now| over the player and its ring;\n"
			            "         dG = max |a_cover - a_now| over ground pixels, far / near = beyond / within %d px of objects; out = the same over pixels nearer than\n"
			            "         the start or more than 2 max(Hmax, 48) over the ground (G5); gain box / diamond = pixel-frames with a_cover - a_now > %.3f;\n"
			            "         tmp = mean over band pixels and frame pairs of |a(f+1) - a(f)| (now, cover, ratio; thin and oracle when on); bf = mean |a - a_blade-free|\n"
			            "         over band pixel-frames (cover: all, blade, ground; now: all, blade, ground); E = mean |a_cover - a_exact|.\n"
			            "  cover+: CPU replica of k over band blade pixel-frames (full >= 0.99, partial, 0; of the k < 0.99 ones the share limited by the height\n"
			            "         gate / the ground fraction; hpart = share with the height gate under 0.99), dk = mean |k(f+1) - k(f)| over band pixels;\n"
			            "         cont = share of non-sky low-res texels that ran the continuation, len = their mean continuation length (units)\n"
			            "  coverobj: per object with pixels: pixel-frames and mean |a_cover - a_now| by depth inside the outline d = 0, 1, 2, 3, 4+ px,\n"
			            "         interior (d >= 3: the 7 x 7 window all that object) pixel-frames and mean, whole-object mean (G4)\n",
				a_coverStart, a_coverStart, kOutline, kGainEps);
		}
		if (gMask) {
			std::printf("mask: round 3 rows (grasscover/gates.md section 7) as r2 rows M0-M3 (k = the mask, 1 or 0) and a \"mask\" line each: px = mask pixel-frames,\n"
			            "mask:   hidden = M0 pixel-frames whose final depth is not the blade's (dropped in M2), fan = fan pixels (none in this harness: M1 = M0),\n"
			            "mask:   kDkRmin = min distance ramp x round fade over M0's pixel-frames, khlow = share of them with the plain height gate < 0.99,\n"
			            "mask:   nocont = share of the row's mask pixel-frames without a full-res continuation (M0-M2), unchanged = share with a_row == a_now,\n"
			            "mask:   tmp = mean |a(f) - a(f-1)| over band pixels by their label in frame f (blade, ground): now, row, ratio\n");
		}
		if (gSink) {
			std::printf("sink: round 5 rows (grasscover/gates.md 9, 9a, 9b) as r2 rows S (sink), V (drop), C (march phase 0), SC, VC, on the slope clumps scenes\n"
			            "sink:   also Spe / Sme (S with the sink's column ground shifted by +13.4 / -13.4 units); GrassCover and ThinFogStart 0 in all%s;\n"
			            "sink:   r2 lines as the other rows (out = the old G5, GrassCover 600 as the near limit); \"sink <row>:\" ref = G16's reference (now; C for\n"
			            "sink:   SC, VC), g16 = pixel-frames, a_row < a_ref - 1e-4, min (a_row - a_ref), not finite or outside [0, 1], max a_row, the same against now;\n"
			            "sink:   sky = max |a_row - a_now| (and against ref) over sky pixel-frames (G5s); fb = own-ray band blade pixel-frames 8-40 units up, of\n"
			            "sink:   them a_row > a_now (S must raise them); low diff = march texels whose distance differs from today's; own = share of non-sky\n"
			            "sink:   pixel-frames on the own-ray path, neq = not bit-identical to today; \"obj\": per object and height bin b0-b4 (0-28, 28-56, 56-84, 84-98, 98+ units over the\n"
			            "sink:   ground): n, a_row - a_now mean p99 max, |a_row - a_now| mean p99, mean alpha row and now; \"T\": distance bands 0-4 (0-10, 10-20,\n"
			            "sink:   20-30, 30-50, 50+ m by the frame-0 ray) over the frame-0 ground and blade pixels whose 5 x 5 window holds only those;\n"
			            "sinkeval: the sink's decision (S) over its march evaluations: half-res texels (non-sky) and own-ray pixels\n", gSinkW0 ? " (MUTATION sinkw0: w forced to 0)" : "");
		}
		if (!a_only.empty()) std::printf("only the scenes whose name holds \"%s\"\n", a_only.c_str());
		const auto start = std::chrono::steady_clock::now();
		const float opacities[] = { 0.43f, 0.18f };
		for (int slope = 0; slope < grounds; ++slope) {
			for (const float eye : { 250.0f, 400.0f, 120.0f }) {
				if (eye < 200.0f && !(a_free.eye120 && slope == 0)) continue;   // eye120: flat ground only
				const View v = MakeView(slope == 1, eye, slope == 2);
				XMFLOAT4X4 pfv;
				XMStoreFloat4x4(&pfv, v.proj);
				for (int layout = kBare; layout <= kBig4000; ++layout) {
					if ((layout == kLowBox && !a_boxes) || (layout == kPostScene && !a_free.post)) continue;
					if (layout >= kBig1000 && !a_bigObj) continue;
					if (eye < 200.0f && layout != kSparse && layout != kClumps) continue;   // eye120: sparse and clumps
					if (slope == 2 && layout != kBare && layout != kSparse) continue;       // tilt: bare and sparse
					const bool noBlades = layout == kBare || layout == kLowBox || layout == kPostScene || layout >= kBig1000;
					for (const double width : { 2.0, 5.0 }) {
						if (noBlades && width > 2.5) continue;   // no blades: the width does not matter
						if (a_quick && (slope || eye > 300.0f || width > 2.5 || layout > kSparse)) continue;
						const std::string scene = std::string(slope == 2 ? "tilt" : slope ? "slope" : "flat") + "_E" + std::to_string(static_cast<int>(eye)) +
						                          (noBlades ? std::string() : "_w" + std::to_string(static_cast<int>(width))) + "_" + kLayoutName[layout];
						if (!a_only.empty()) {   // several substrings separated by commas (round 2): any of them
							bool hit = false;
							for (std::size_t p0 = 0; p0 <= a_only.size();) {
								const std::size_t p1 = std::min(a_only.find(',', p0), a_only.size());
								hit = hit || scene.find(a_only.substr(p0, p1 - p0)) != std::string::npos;
								p0 = p1 + 1;
							}
							if (!hit) continue;
						}
						// Blade count fitted to the coverage at frame 0 (overlaps: Poisson, n ~ -ln(1 - c)).
						int                count = 0;
						std::vector<Blade> blades;
						FrameData          f0;
						double             cov = 0.0;
						int                bandPx = 0;
						if (!noBlades) {
							count = layout == kClumps ? 60 : 1000;
							for (int round = 0; round < kCalibrations; ++round) {
								blades = MakeBlades(static_cast<Layout>(layout), count);
								BuildFrame(v, 0, blades, width, f0);
								cov = Coverage(f0, &bandPx);
								if (round + 1 == kCalibrations) break;
								const double fit = std::log(1.0 - kCoverage[layout]) / std::log(1.0 - std::clamp(cov, 1e-4, 0.999));
								count = std::max(1, static_cast<int>(std::lround(count * fit)));
							}
						}
						std::vector<FrameData> frames(kFrames);
						std::vector<View> views(kFrames, v);   // round 2 turn: the camera yaws kTurnDeg a frame (Render reads only v's projection, eye and ground)
						if (gTurn) {
							for (int f = 0; f < kFrames; ++f) views[f] = MakeView(slope == 1, eye, slope == 2, kTurnDeg * f);
						}
						for (int f = 0; f < kFrames; ++f) BuildFrame(views[f], f, blades, width, frames[f], layout == kLowBox, layout == kPostScene, BigDistance(layout));
						// oracle: the same frames without blades (same jitter; trunk, person and player kept) = the true surface behind each blade
						// (free: also built for the scenes with blades; in the scenes without, free takes the frame itself, so the boxes and
						// the post stay in its reference, and the oracle's frames are as they were)
						const std::vector<Blade> bladeless;
						std::vector<FrameData>   framesG(oracle || ((freeOn || cover) && !noBlades) ? kFrames : 0);
						for (int f = 0; f < static_cast<int>(framesG.size()); ++f) BuildFrame(views[f], f, bladeless, width, framesG[f]);
						cov = Coverage(frames[0], &bandPx);
						// Band pixels, fixed per scene from frame 0: ground or blade 1500..3000 out; with blades their 5 x 5 window
						// holds both and nothing else (no sky or object a jitter step away).
						const FrameData&         z = frames[0];
						std::vector<std::size_t> band;
						int                      rowMin = static_cast<int>(kSize), rowMax = -1;
						for (int y = kWindow; y + kWindow < static_cast<int>(kSize); ++y) {
							for (int x = kWindow; x + kWindow < static_cast<int>(kSize); ++x) {
								const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
								if (!InBand(z, i)) continue;
								bool hasBlade = false, hasGround = false, other = false;
								for (int dy = -kWindow; dy <= kWindow; ++dy) {
									for (int dx = -kWindow; dx <= kWindow; ++dx) {
										const std::uint8_t l = z.label[static_cast<std::size_t>(y + dy) * kSize + (x + dx)];
										hasBlade = hasBlade || l == kBlade;
										hasGround = hasGround || l == kGround;
										other = other || (l != kBlade && l != kGround);
									}
								}
								if (other || (!noBlades && !(hasBlade && hasGround))) continue;
								band.push_back(i);
								rowMin = std::min(rowMin, y);
								rowMax = std::max(rowMax, y);
							}
						}
						int objPx[3]{};
						for (const std::uint8_t l : z.label) {
							objPx[0] += l == kTrunk;
							objPx[1] += l == kPerson;
							objPx[2] += l == kPlayer;
						}
						std::printf("scene %-24s blades %6zu (%d %s), coverage %.3f (aim %.2f) of %d band px, measured px %zu (rows %d..%d), trunk %d person %d player %d px\n", scene.c_str(),
							blades.size(), count, layout == kClumps ? "bushes" : "placed", cov, kCoverage[layout], bandPx, band.size(), rowMin, rowMax, objPx[0], objPx[1], objPx[2]);
						if (a_dump) {
							WriteBin(a_outDir + "/grass_labels_" + scene + ".bin", z.label.data(), z.label.size());
							WriteBin(a_outDir + "/grass_arm_" + scene + ".bin", z.arm.data(), z.arm.size());
							WriteBin(a_outDir + "/grass_ray_" + scene + ".bin", z.ray.data(), z.ray.size() * sizeof(float));
							WriteBin(a_outDir + "/grass_height_" + scene + ".bin", z.height.data(), z.height.size() * sizeof(float));
						}
						std::vector<std::vector<std::uint8_t>> outlines(kFrames), playerRing(kFrames), nearObj(thin || cover ? kFrames : 0), objDepth(cover ? kFrames : 0);
						for (int f = 0; f < kFrames; ++f) {
							outlines[f] = OutlineBand(frames[f].label, kTrunk, kPerson);
							playerRing[f] = OutlineBand(frames[f].label, kPlayer, kPlayer);
							if (thin || cover) nearObj[f] = NearObjects(frames[f].label);
							if (cover) objDepth[f] = ObjectDepth(frames[f].label, 4);
						}
						std::vector<std::size_t> sinkPx;   // round 5 (CLI sink): the distance-band pixel set, from frame 0
						std::vector<int>         sinkBin;
						if (gSink) SinkBandSet(frames[0], sinkPx, sinkBin);

						for (int colsKind = 0; colsKind < 2; ++colsKind) {
							for (const float opacity : opacities) {
								if ((a_quick || gOpacity043Only) && opacity < 0.3f) continue;
								ID3D11ShaderResourceView* cols = colsKind == 0 ? sims[slope].ColumnSrv() : evenSrv[slope];
								const std::size_t         nb = band.size();
								std::vector<double>       sL[5], sL2[5];   // now, exact, no fog, exact24, thin
								for (int k = 0; k < 5; ++k) {
									sL[k].assign(nb, 0.0);
									sL2[k].assign(nb, 0.0);
								}
								double eSum = 0.0, kSum[2]{}, sMax = 0.0, spMax = 0.0, plSum[2]{}, cfMax = 0.0, euSum = 0.0, esSum = 0.0;
								long long eN = 0, kN = 0, plN = 0, ownBand = 0, ownAll = 0, allN = 0;
								// thin: E, K, S, Sb (now, thin), Sp, player sum; the changes against ThinFogStart 0 (max over all frames)
								double tE = 0.0, tK = 0.0, tS = 0.0, tSb[2]{}, tSp = 0.0, tPl = 0.0, dG = 0.0, dBand = 0.0, dP = 0.0, dBody = 0.0, dArm = 0.0, dArmSum = 0.0;
								long long armN = 0, chg = 0;
								// thin+: per object (trunk, person body, box, diamond) max |a_thin - a_now|, pixel-frames, pixel-frames
								// taking more fog; ground beyond / within 4 px of objects; the replica's own-ray counts (band, non-sky)
								double    pdMax[4]{}, gFar = 0.0, gNear = 0.0;
								long long pN[4]{}, pGain[4]{}, ownG[2]{};
								const bool closed = colsKind == 1 && !slope;
								// oracle: band luminance sums as sL (so T is bit for bit now's where nothing changed), per-frame band alpha and
								// labels for the diagnostic T's, E, K, Sb (now, oracle), Sp, player sum and the proof columns
								std::vector<double>       sO(oracle ? nb : 0, 0.0), sO2(oracle ? nb : 0, 0.0);
								std::vector<float>        orA(oracle ? nb * kFrames : 0), gA(oracle ? nb * kFrames : 0), nwA(oracle ? nb * kFrames : 0);
								std::vector<std::uint8_t> orBl(oracle ? nb * kFrames : 0);
								double oE = 0.0, oK = 0.0, oSb[2]{}, oSp = 0.0, oPl = 0.0, oOff = 0.0, oCf[3]{}, oGnd = 0.0;
								long long oSub = 0, oNear = 0, oBad = 0, oGndN = 0;
								// free: band luminance sums of the render on the effective depth (sF) and of bf, every band pixel on the
								// blade-free alpha (sB); per-frame band alpha (free, now) and labels for flipd; the row's sums and counts
								std::vector<double>             sF(freeOn ? nb : 0, 0.0), sF2(freeOn ? nb : 0, 0.0), sB(freeOn ? nb : 0, 0.0), sB2(freeOn ? nb : 0, 0.0);
								std::vector<float>              fFr(freeOn ? nb * kFrames : 0), fNw(freeOn ? nb * kFrames : 0);
								std::vector<std::uint8_t>       fBl(freeOn ? nb * kFrames : 0);
								std::vector<std::vector<float>> fwd(freeOn && a_free.orderRev ? kFrames : 0);
								double    fE = 0.0, fK = 0.0, fSb[2]{}, fSp = 0.0, fPl = 0.0, fdG = 0.0, fdP = 0.0, fdBody = 0.0, fdArm = 0.0, fdArmSum = 0.0, fGap = 0.0, fEg[2]{}, fPost[2]{};
								long long fBlade = 0, fClosed = 0, fPart = 0, fVeto = 0, fGapN = 0, fEgN = 0, fObj[5]{}, fObjN[5]{}, fGain[2]{}, fArm = 0, fArmN = 0, fPostMoved = 0,
								          fPostN = 0, fDpx[2]{}, fDbf = 0, fOwn[2]{}, fShort = 0, fRev = 0;
								int       fMinRun = -1, fMinLen = 0;
								// cover: band luminance sums, previous band alpha (now, cover, thin, oracle) for tmp, the replica's previous
								// k, and the row's sums
								std::vector<double> sC(cover ? nb : 0, 0.0), sC2(cover ? nb : 0, 0.0);
								std::vector<float>  pvA[4], pvK(cover ? nb : 0, 0.0f);
								for (auto& p : pvA) p.assign(cover ? nb : 0, 0.0f);
								double    cK = 0.0, cSb[2]{}, cSp = 0.0, cPl = 0.0, cdP = 0.0, cdG[3]{}, cOut = 0.0, cE = 0.0, cTmp[4]{}, cBf[6]{}, cDk = 0.0, cLen = 0.0;
								long long cGain[2]{}, cTmpN = 0, cBfN[3]{}, cDkN = 0, cKb[6]{}, cCont = 0, cLowN = 0;   // cKb: blade, full, part, zero, lim height, hpart
								long long cLimF = 0;
								double    cObj[8][6]{};    // per object: sum |a_cover - a_now| by depth 0..4+, whole
								long long cObjN[8][6]{};
								const float outH = 2.0f * hMaxAll[slope][colsKind];
								// round 2 rows (CLI r2, needs cover): candidates as CPU post-processes of now's and cover's alpha
								static const char* const kR2Name[]{ "now", "cover", "Ta", "Tb", "B10", "B20", "B35", "B10T", "B20T", "B35T", "CB20", "CB20T", "TaG", "TbG" };   // TaG, TbG: post-hoc, history only inside the band gate
								constexpr int            kR2 = 14;
								RowAcc                   r2acc[kR2];
								History                  r2hist[kR2];
								if (gR2 && cover) {
									for (auto& r : r2acc) r.Init(nb);
								}
								// round 3 rows (CLI mask): M0 (every blade in front of the ground), M1 (= M0: no fans here), M2 (M1 and the final
								// depth equals the blade's), M3 (M2's mask, half-res march); diagnostics per row on the "mask" lines
								static const char* const kMName[4]{ "M0", "M1", "M2", "M3" };
								RowAcc                   mAcc[4];
								double    mKdr = 1.0, mTmp[4][2][2]{};   // min kD kR over mask pixel-frames; tmp sums [row][blade, ground][row, now]
								long long mPx[4]{}, mHid = 0, mKhLow = 0, mNoCont[4]{}, mSame[4]{}, mTmpN[2]{};
								std::vector<float> mPv[4];
								std::vector<float> mPvNow;
								if (gMask) {
									for (int q = 0; q < 4; ++q) {
										mAcc[q].Init(nb);
										mPv[q].assign(nb, 0.0f);
									}
									mPvNow.assign(nb, 0.0f);
								}
								// round 5 rows (CLI sink, gates.md 9, 9a, 9b): S (sink), V (drop), C (march phase 0), SC, VC, and on the slope clumps
								// scenes S+e / S-e (the sink's column ground shifted by +-13.4 units; printed as Spe, Sme); GrassCover and ThinFogStart
								// 0 in all; the sink diagnostics per row on the "sink" lines and the shares of the sink's march evaluations
								// (half-res texels and own-ray pixels) on the "sinkeval" line
								static const char* const kSName[7]{ "S", "V", "C", "SC", "VC", "Spe", "Sme" };
								static const int         kSMode[7]{ 1, 2, 0, 1, 2, 3, 4 };   // Render's a_sink
								static const bool        kSPh0[7]{ false, false, true, true, true, false, false };
								const int                sRows = gSink ? (slope == 1 && layout == kClumps ? 7 : 5) : 0;
								RowAcc                   sRow[7];
								SinkAcc                  sAcc[7], sNowAcc;
								long long                sEv[5]{};   // evaluations (non-sky), w > 0, hit, no hit, w > 0 on the own-ray path
								long long                sEvT = 0, sEvF = 0;
								for (int q = 0; q < sRows; ++q) {
									sRow[q].Init(nb);
									sAcc[q].Init(sinkPx.size());
								}
								if (gSink) sNowAcc.Init(sinkPx.size());
								for (int f = 0; f < kFrames; ++f) {
									const FrameData&   fr = frames[f];
									std::vector<float> low;
									const auto         aNow = Render(g, fr, v, cols, opacity, f, 0, &low);
									// The march pass is the same with ThinFogStart (only the composite reads it), so the exact renders
									// below still find this frame's march in the low texture.
									const auto         aThin = thin ? Render(g, fr, v, cols, opacity, f, 0, nullptr, a_thinStart) : std::vector<float>{};
									const auto         aEx = Render(g, fr, v, cols, opacity, f, 2, nullptr);
									const auto         aX24 = Render(g, fr, v, cols, opacity, f, 3, nullptr);
									const auto         own = OwnRay(fr, low);
									const auto         ownGr = thin ? OwnRayGround(fr, low, a_thinStart) : std::vector<std::uint8_t>{};
									// oracle (after the renders above, so they are untouched): blade pixels from the start out take the
									// blade-free frame's alpha
									std::vector<float> aG, aOr;
									if (oracle) {
										const FrameData& frG = framesG[f];
										aG = Render(g, frG, v, cols, opacity, f, 0, nullptr);
										const auto aGx = closed ? Render(g, frG, v, cols, opacity, f, 2, nullptr) : std::vector<float>{};
										aOr = a_oracleOnThin && thin ? aThin : aNow;   // oraclethin: the other pixels keep the thin path's alpha
										for (std::size_t i = 0; i < aOr.size(); ++i) {
											if (fr.label[i] != kBlade) continue;
											if (!(fr.ray[i] >= a_oracleStart)) {
												++oNear;
												continue;
											}
											aOr[i] = aG[i];
											++oSub;
											if (frG.label[i] != kGround || !(frG.ray[i] > fr.ray[i])) ++oBad;
											if (closed && frG.label[i] == kGround) {
												oCf[0] = std::max(oCf[0], std::fabs(aG[i] - ClosedAlpha(0.0, frG.ray[i], eye, opacity)));
												oCf[1] = std::max(oCf[1], std::fabs(aG[i] - ClosedAlpha(fr.height[i], fr.ray[i], eye, opacity)));
												oCf[2] = std::max(oCf[2], std::fabs(aGx[i] - ClosedAlpha(0.0, frG.ray[i], eye, opacity)));
											}
										}
									}
									const std::string  name = scene + (colsKind ? "_even_" : "_sim_") + (opacity > 0.3f ? "0.43" : "0.18");
									if (a_dump && f == 0) {
										WriteBin(a_outDir + "/grass_now_" + name + ".bin", aNow.data(), aNow.size() * sizeof(float));
										WriteBin(a_outDir + "/grass_exact_" + name + ".bin", aEx.data(), aEx.size() * sizeof(float));
										WriteBin(a_outDir + "/grass_exact24_" + name + ".bin", aX24.data(), aX24.size() * sizeof(float));
										if (thin) WriteBin(a_outDir + "/grass_thin_" + name + ".bin", aThin.data(), aThin.size() * sizeof(float));
									}
									if (a_dumpAll && thin) {   // every frame: now and thin, and the frame's labels, arm and ray once per scene
										const std::string fs = "_f" + std::to_string(f);
										WriteBin(a_outDir + "/fr_now_" + name + fs + ".bin", aNow.data(), aNow.size() * sizeof(float));
										WriteBin(a_outDir + "/fr_thin_" + name + fs + ".bin", aThin.data(), aThin.size() * sizeof(float));
										if (colsKind == 0 && opacity > 0.3f) {
											WriteBin(a_outDir + "/fr_labels_" + scene + fs + ".bin", fr.label.data(), fr.label.size());
											WriteBin(a_outDir + "/fr_arm_" + scene + fs + ".bin", fr.arm.data(), fr.arm.size());
											WriteBin(a_outDir + "/fr_ray_" + scene + fs + ".bin", fr.ray.data(), fr.ray.size() * sizeof(float));
										}
									}
									// free (after every render above, which it leaves untouched): the effective depth, its render and the own-ray
									// replica on it. The reference (REF's candidate; the blade-free alpha of bf, Eg, gap) is the same frame without
									// blades; in the scenes without blades that is the frame itself.
									std::vector<float>        aFree;
									std::vector<std::uint8_t> ownFree;
									FreeOut                   fo;
									const FrameData&          frR = freeOn && !noBlades ? framesG[f] : fr;
									if (freeOn) {
										if (!oracle && !noBlades) aG = Render(g, frR, v, cols, opacity, f, 0, nullptr);
										fo = BuildFreeDepth(fr, v, colsCpu[slope][colsKind], a_free.p.src == kFreeRef ? &frR.depth : nullptr, a_free.p);
										std::vector<float> lowFree;
										aFree = Render(g, fr, v, cols, opacity, f, 0, &lowFree, 0.0f, &fo.depth, a_free.p.start, a_free.p.maxPx);
										ownFree = OwnRay(fr, lowFree, &fo.depth);
										if (a_free.orderRev) fwd[f] = aFree;
										FreeMinRun(fr, fo.lpx, fMinRun, fMinLen, fShort);
										if (a_dump && f == 0) {
											WriteBin(a_outDir + "/grass_free_" + name + ".bin", aFree.data(), aFree.size() * sizeof(float));
											WriteBin(a_outDir + "/grass_freedepth_" + name + ".bin", fo.depth.data(), fo.depth.size() * sizeof(float));
											WriteBin(a_outDir + "/grass_freek_" + name + ".bin", fo.k.data(), fo.k.size() * sizeof(float));
										}
									}
									// cover (after every render above, which it leaves untouched): GrassCover's render, the blade-free render
									// it is judged against (the same frame without blades; the frame itself without blades), the replica's k
									std::vector<float> aCov, covLow, rk, rkH, rkF;
									if (cover) {
										aCov = Render(g, fr, v, cols, opacity, f, 0, nullptr, 0.0f, nullptr, 0.0f, 0, a_coverStart, &covLow);
										if (!noBlades && aG.empty()) aG = Render(g, framesG[f], v, cols, opacity, f, 0, nullptr);
										CoverReplica(fr, colsCpu[slope][colsKind], a_coverStart, rk, rkH, rkF);
										for (std::size_t t = 0; t < low.size(); ++t) {
											if (!(low[t] < 1e8f)) continue;
											++cLowN;
											if (covLow[t * 2 + 1] != low[t]) {
												++cCont;
												cLen += covLow[t * 2 + 1] - low[t];
											}
										}
										if (a_dump && f == 0) {
											const std::string nm = scene + (colsKind ? "_even_" : "_sim_") + (opacity > 0.3f ? "0.43" : "0.18");
											WriteBin(a_outDir + "/grass_cover_" + nm + ".bin", aCov.data(), aCov.size() * sizeof(float));
											const std::vector<float>& bfd = noBlades ? aNow : aG;
											WriteBin(a_outDir + "/grass_bf_" + nm + ".bin", bfd.data(), bfd.size() * sizeof(float));
											if (oracle) WriteBin(a_outDir + "/grass_oracle_" + nm + ".bin", aOr.data(), aOr.size() * sizeof(float));
											WriteBin(a_outDir + "/grass_coverk_" + nm + ".bin", rk.data(), rk.size() * sizeof(float));
										}
									}
									if (gR2 && cover) {
										const std::vector<float>& bfr = noBlades ? aNow : aG;
										const auto                kh = HeightGate(fr, colsCpu[slope][colsKind]);
										const auto                b10 = BlurDist(fr, aNow, kh, 0.10f), b20 = BlurDist(fr, aNow, kh, 0.20f), b35 = BlurDist(fr, aNow, kh, 0.35f);
										const auto                cb20 = BlurDist(fr, aCov, kh, 0.20f);
										const std::vector<float>  imgs[kR2]{ aNow, aCov, Temporal(r2hist[2], fr, aNow), Temporal(r2hist[3], fr, aCov), b10, b20, b35,
                                            Temporal(r2hist[7], fr, b10), Temporal(r2hist[8], fr, b20), Temporal(r2hist[9], fr, b35), cb20, Temporal(r2hist[11], fr, cb20),
                                            TemporalGated(r2hist[12], fr, aNow, kh), TemporalGated(r2hist[13], fr, aCov, kh) };
										const RowCtx ctx{ &fr, &aNow, &aEx, &bfr, &band, &outlines[f], &playerRing[f], &nearObj[f], &objDepth[f], outH, a_coverStart, f };
										for (int q = 0; q < kR2; ++q) AccRow(r2acc[q], ctx, imgs[q]);
										if (a_dump && f < 2) {
											const std::string nm = scene + (colsKind ? "_even_" : "_sim_") + (opacity > 0.3f ? "0.43" : "0.18") + "_f" + std::to_string(f);
											for (int q = 0; q < kR2; ++q) WriteBin(a_outDir + "/r2_" + kR2Name[q] + "_" + nm + ".bin", imgs[q].data(), imgs[q].size() * sizeof(float));
											WriteBin(a_outDir + "/r2_bf_" + nm + ".bin", bfr.data(), bfr.size() * sizeof(float));
											WriteBin(a_outDir + "/r2_labels_" + nm + ".bin", fr.label.data(), fr.label.size());
										}
									}
									if (gMask && gR2 && cover) {
										const std::size_t  np = fr.label.size();
										std::vector<float> mk0(np, 0.0f), mk2(np, 0.0f);
										long long          hidden = 0;
										for (std::size_t i = 0; i < np; ++i) {
											if (!fr.grass[i]) continue;
											mk0[i] = 1.0f;
											if (fr.depth[i] == fr.grassDepth[i]) {
												mk2[i] = 1.0f;
											} else {
												++hidden;
											}
										}
										mHid += hidden;
										std::vector<std::uint8_t> cont0, cont2;
										const auto aM0 = Render(g, fr, v, cols, opacity, f, 0, nullptr, 0.0f, nullptr, 0.0f, 0, a_coverStart, nullptr, &mk0, 2, &cont0);
										const auto aM2 = hidden ? Render(g, fr, v, cols, opacity, f, 0, nullptr, 0.0f, nullptr, 0.0f, 0, a_coverStart, nullptr, &mk2, 2, &cont2) : aM0;
										if (!hidden) cont2 = cont0;
										const auto aM3 = Render(g, fr, v, cols, opacity, f, 0, nullptr, 0.0f, nullptr, 0.0f, 0, a_coverStart, nullptr, &mk2, 1);
										const std::vector<float>* imgs[4]{ &aM0, &aM0, &aM2, &aM3 };
										const std::vector<float>* masks[4]{ &mk0, &mk0, &mk2, &mk2 };
										const std::vector<std::uint8_t>* conts[4]{ &cont0, &cont0, &cont2, &cont2 };
										const std::vector<float>& bfr = noBlades ? aNow : aG;
										const RowCtx              ctx{ &fr, &aNow, &aEx, &bfr, &band, &outlines[f], &playerRing[f], &nearObj[f], &objDepth[f], outH, a_coverStart, f };
										const XMMATRIX            inv = XMLoadFloat4x4(&fr.inv);
										for (std::size_t i = 0; i < np; ++i) {   // kD kR and the plain row's height gate over M0's pixels (M0 holds M2's)
											if (!(mk0[i] > 0.5f)) continue;
											XMFLOAT4 h;
											XMStoreFloat4(&h, XMVector4Transform(XMVectorSet((i % kSize + 0.5f) / kSize * 2 - 1, 1 - (i / kSize + 0.5f) / kSize * 2, fr.depth[i], 1), inv));
											const float ux = (h.x / h.w - kOrigin) / kExtent - 0.5f, uy = (h.y / h.w - kOrigin) / kExtent - 0.5f;
											const float kdr = FreeStep(a_coverStart, 2.0f * a_coverStart, fr.ray[i]) * (1.0f - FreeStep(0.85f, 0.95f, 2.0f * std::sqrt(ux * ux + uy * uy)));
											mKdr = std::min(mKdr, static_cast<double>(kdr));
											mKhLow += rkH[i] < 0.99f ? 1 : 0;
										}
										for (int q = 0; q < 4; ++q) {
											AccRow(mAcc[q], ctx, *imgs[q]);
											for (std::size_t i = 0; i < np; ++i) {
												if (!((*masks[q])[i] > 0.5f)) continue;
												++mPx[q];
												mSame[q] += (*imgs[q])[i] == aNow[i] ? 1 : 0;
												mNoCont[q] += q < 3 && !(*conts[q])[i] ? 1 : 0;
											}
										}
										for (std::size_t b = 0; b < nb; ++b) {   // tmp split by the pixel's label in this frame (blade / ground)
											const std::size_t i = band[b];
											const int         cls = fr.label[i] == kBlade ? 0 : 1;
											if (f > 0) {
												for (int q = 0; q < 4; ++q) {
													mTmp[q][cls][0] += std::fabs((*imgs[q])[i] - mPv[q][b]);
													mTmp[q][cls][1] += std::fabs(aNow[i] - mPvNow[b]);
												}
												++mTmpN[cls];
											}
											for (int q = 0; q < 4; ++q) mPv[q][b] = (*imgs[q])[i];
											mPvNow[b] = aNow[i];
										}
										if (a_dump && f < 2) {
											const std::string nm = scene + (colsKind ? "_even_" : "_sim_") + (opacity > 0.3f ? "0.43" : "0.18") + "_f" + std::to_string(f);
											for (const int q : { 0, 2, 3 }) WriteBin(a_outDir + "/r2_" + kMName[q] + "_" + nm + ".bin", imgs[q]->data(), imgs[q]->size() * sizeof(float));
											WriteBin(a_outDir + "/r2_mask0_" + nm + ".bin", mk0.data(), mk0.size() * sizeof(float));
											WriteBin(a_outDir + "/r2_mask2_" + nm + ".bin", mk2.data(), mk2.size() * sizeof(float));
										}
									}
									if (gSink && gR2 && cover) {   // after every render above (these overwrite the low texture)
										std::vector<float> rows[7], lows[7];
										for (int q = 0; q < sRows; ++q) {
											rows[q] = Render(g, fr, v, cols, opacity, f, 0, &lows[q], 0.0f, nullptr, 0.0f, 0, 0.0f, nullptr, nullptr, 0, nullptr, kSMode[q], kSPh0[q]);
										}
										const std::vector<float>& bfr = noBlades ? aNow : aG;
										// PrintRow's G5 ("out") keeps the old form (GrassCover 600 as the near limit); G5s is on the sink line
										const RowCtx ctx{ &fr, &aNow, &aEx, &bfr, &band, &outlines[f], &playerRing[f], &nearObj[f], &objDepth[f], outH, a_coverStart, f };
										for (int q = 0; q < sRows; ++q) {
											AccRow(sRow[q], ctx, rows[q]);
											const auto ownQ = std::memcmp(lows[q].data(), low.data(), low.size() * sizeof(float)) == 0 ? own : OwnRay(fr, lows[q]);
											SinkFrameAcc(sAcc[q], fr, rows[q], aNow, q == 3 || q == 4 ? rows[2] : aNow, ownQ, lows[q], low, sinkPx);
										}
										for (std::size_t k = 0; k < sinkPx.size(); ++k) {
											const std::size_t i = sinkPx[k];
											const double      lm = (fr.label[i] == kBlade ? kLeafC : kGroundC) * (1.0 - aNow[i]) + kFogC * aNow[i];
											sNowAcc.s[k] += lm;
											sNowAcc.s2[k] += lm * lm;
										}
										const auto dg = SinkDiag(g, fr, v, cols, opacity);
										auto       ev = [&](std::size_t a_p, bool a_own) {
                                            const float cls = dg[a_p * 4];
                                            ++sEv[0];
                                            if (!(cls > 0.5f)) return;
                                            ++sEv[1];
                                            ++sEv[cls < 1.5f ? 2 : 3];
                                            sEv[4] += a_own ? 1 : 0;
										};
										const int lowW = static_cast<int>(kSize / 2);
										for (int t = 0; t < lowW * lowW; ++t) {
											if (!(low[static_cast<std::size_t>(t)] < 1e8f)) continue;
											const int px = std::min((t % lowW) * 2 + 1, static_cast<int>(kSize) - 1), py = std::min((t / lowW) * 2 + 1, static_cast<int>(kSize) - 1);
											ev(static_cast<std::size_t>(py) * kSize + px, false);
											++sEvT;
										}
										for (std::size_t i = 0; i < own.size(); ++i) {
											if (!own[i] || !(fr.ray[i] < 1e8f)) continue;
											ev(i, true);
											++sEvF;
										}
										if (a_dump && f < 2) {
											const std::string nm = scene + (colsKind ? "_even_" : "_sim_") + (opacity > 0.3f ? "0.43" : "0.18") + "_f" + std::to_string(f);
											for (int q = 0; q < sRows; ++q) WriteBin(a_outDir + "/r2_" + kSName[q] + "_" + nm + ".bin", rows[q].data(), rows[q].size() * sizeof(float));
										}
									}
									const std::vector<float>& aBf = noBlades ? aNow : aG;   // free: the blade-free alpha
									auto lum = [&](std::size_t i, double a) {
										const double c = fr.label[i] == kBlade ? kLeafC : kGroundC;
										return c * (1.0 - a) + kFogC * a;
									};
									for (std::size_t b = 0; b < nb; ++b) {
										const std::size_t i = band[b];
										const double      l[5]{ lum(i, aNow[i]), lum(i, aEx[i]), lum(i, 0.0), lum(i, aX24[i]), thin ? lum(i, aThin[i]) : 0.0 };
										for (int k = 0; k < 5; ++k) {
											sL[k][b] += l[k];
											sL2[k][b] += l[k] * l[k];
										}
										if (oracle) {
											const double lo = lum(i, aOr[i]);
											sO[b] += lo;
											sO2[b] += lo * lo;
											orA[static_cast<std::size_t>(f) * nb + b] = aOr[i];
											gA[static_cast<std::size_t>(f) * nb + b] = aG[i];
											nwA[static_cast<std::size_t>(f) * nb + b] = aNow[i];
											orBl[static_cast<std::size_t>(f) * nb + b] = fr.label[i] == kBlade ? 1 : 0;
											oE += std::fabs(aOr[i] - aEx[i]);
											if (fr.label[i] == kGround) {
												oGnd += std::fabs(aNow[i] - aG[i]);
												++oGndN;
											}
										}
										if (freeOn) {
											const std::size_t j = static_cast<std::size_t>(f) * nb + b;
											const double      lf = lum(i, aFree[i]), lbf = lum(i, aBf[i]);
											sF[b] += lf;
											sF2[b] += lf * lf;
											sB[b] += lbf;
											sB2[b] += lbf * lbf;
											fFr[j] = aFree[i];
											fNw[j] = aNow[i];
											fBl[j] = fr.label[i] == kBlade ? 1 : 0;
											fE += std::fabs(aFree[i] - aEx[i]);
											fOwn[0] += ownFree[i];
											if (fr.label[i] == kGround) {
												fEg[0] += std::fabs(aNow[i] - aBf[i]);
												fEg[1] += std::fabs(aFree[i] - aBf[i]);
												++fEgN;
											} else if (fr.label[i] == kBlade) {
												++fBlade;
												fClosed += fo.k[i] >= kFreeClosedK ? 1 : 0;
												fPart += fo.k[i] > 0.0f && fo.k[i] < kFreeClosedK ? 1 : 0;
												fVeto += fo.veto[i];
												if (fo.k[i] >= kFreeClosedK) {
													const double zE = FreeLinear(fo.depth[i], pfv._33, pfv._43), zR = FreeLinear(frR.depth[i], pfv._33, pfv._43);
													fGap += std::fabs(zE - zR) / zR;
													++fGapN;
												}
											}
										}
										if (cover) {
											const double lc = lum(i, aCov[i]);
											sC[b] += lc;
											sC2[b] += lc * lc;
											cE += std::fabs(aCov[i] - aEx[i]);
											const float cur[4]{ aNow[i], aCov[i], thin ? aThin[i] : 0.0f, oracle ? aOr[i] : 0.0f };
											if (f > 0) {
												for (int q = 0; q < 4; ++q) cTmp[q] += std::fabs(cur[q] - pvA[q][b]);
												++cTmpN;
												cDk += std::fabs(rk[i] - pvK[b]);
												++cDkN;
											}
											for (int q = 0; q < 4; ++q) pvA[q][b] = cur[q];
											pvK[b] = rk[i];
											const int cls = fr.label[i] == kBlade ? 1 : 2;
											cBf[0] += std::fabs(aCov[i] - aBf[i]);
											cBf[3] += std::fabs(aNow[i] - aBf[i]);
											cBf[cls] += std::fabs(aCov[i] - aBf[i]);
											cBf[cls + 3] += std::fabs(aNow[i] - aBf[i]);
											++cBfN[0];
											++cBfN[cls];
											if (fr.label[i] == kBlade) {
												++cKb[0];
												cKb[1] += rk[i] >= 0.99f ? 1 : 0;
												cKb[2] += rk[i] > 0.0f && rk[i] < 0.99f ? 1 : 0;
												cKb[3] += rk[i] > 0.0f ? 0 : 1;
												if (rk[i] < 0.99f) {
													if (rkH[i] <= rkF[i]) {
														++cKb[4];
													} else {
														++cLimF;
													}
												}
												cKb[5] += rkH[i] < 0.99f ? 1 : 0;
											}
										}
										eSum += std::fabs(aNow[i] - aEx[i]);
										euSum += std::fabs(aNow[i] - aX24[i]);
										esSum += std::fabs(aX24[i] - aEx[i]);
										++eN;
										ownBand += own[i];
										if (thin) ownG[0] += ownGr[i];
										if (thin) {
											tE += std::fabs(aThin[i] - aEx[i]);
											dBand = std::max(dBand, static_cast<double>(std::fabs(aThin[i] - aNow[i])));
											chg += aThin[i] != aNow[i] ? 1 : 0;
										}
										if (closed && (fr.label[i] == kGround || fr.label[i] == kBlade)) {
											cfMax = std::max(cfMax, std::fabs(aEx[i] - ClosedAlpha(fr.height[i], fr.ray[i], eye, opacity)));
										}
										if (fr.label[i] != kBlade) continue;
										// The nearest ground pixel along the row (both sides averaged when equally near).
										const int x = static_cast<int>(i % kSize), y = static_cast<int>(i / kSize);
										for (int dx = 1; dx <= kNearSearch; ++dx) {
											double gs[6]{}, n = 0.0;
											for (const int side : { -dx, dx }) {
												const int xx = x + side;
												if (xx < 0 || xx >= static_cast<int>(kSize)) continue;
												const std::size_t j = static_cast<std::size_t>(y) * kSize + xx;
												if (fr.label[j] != kGround) continue;
												gs[0] += lum(j, aNow[j]);
												gs[1] += lum(j, aEx[j]);
												if (thin) gs[2] += lum(j, aThin[j]);
												if (oracle) gs[3] += lum(j, aOr[j]);
												if (freeOn) gs[4] += lum(j, aFree[j]);
												if (cover) gs[5] += lum(j, aCov[j]);
												n += 1.0;
											}
											if (n == 0.0) continue;
											kSum[0] += std::fabs(lum(i, aNow[i]) - gs[0] / n);
											kSum[1] += std::fabs(lum(i, aEx[i]) - gs[1] / n);
											if (thin) tK += std::fabs(lum(i, aThin[i]) - gs[2] / n);
											if (oracle) oK += std::fabs(lum(i, aOr[i]) - gs[3] / n);
											if (freeOn) fK += std::fabs(lum(i, aFree[i]) - gs[4] / n);
											if (cover) cK += std::fabs(lum(i, aCov[i]) - gs[5] / n);
											++kN;
											break;
										}
									}
									for (std::size_t i = 0; i < aNow.size(); ++i) {
										const double d = std::fabs(aNow[i] - aEx[i]);
										if (outlines[f][i]) sMax = std::max(sMax, d);
										if (playerRing[f][i]) spMax = std::max(spMax, d);
										if (fr.label[i] == kPlayer) {
											plSum[0] += aNow[i];
											plSum[1] += aEx[i];
											++plN;
										}
										if (fr.label[i] != kSky) {
											ownAll += own[i];
											++allN;
											if (thin) ownG[1] += ownGr[i];
										}
										if (oracle) {
											const double dto = std::fabs(aOr[i] - aEx[i]);
											if (fr.label[i] != kBlade) oOff = std::max(oOff, static_cast<double>(std::fabs(aOr[i] - aNow[i])));
											if (outlines[f][i] && !fr.arm[i] && fr.label[i] != kBlade) {
												oSb[0] = std::max(oSb[0], d);
												oSb[1] = std::max(oSb[1], dto);
											}
											if (playerRing[f][i]) oSp = std::max(oSp, dto);
											if (fr.label[i] == kPlayer) oPl += aOr[i];
										}
										if (freeOn) {
											const double       df = std::fabs(aFree[i] - aEx[i]), dn = std::fabs(aFree[i] - aNow[i]);
											const bool         moved = fo.depth[i] != fr.depth[i];
											const std::uint8_t l = fr.label[i];
											if (outlines[f][i] && !fr.arm[i] && l != kBlade) {
												fSb[0] = std::max(fSb[0], d);
												fSb[1] = std::max(fSb[1], df);
											}
											if (playerRing[f][i]) {
												fSp = std::max(fSp, df);
												fdP = std::max(fdP, dn);
											}
											if (l == kPlayer) {
												fPl += aFree[i];
												fdP = std::max(fdP, dn);
											}
											if (l == kGround) fdG = std::max(fdG, dn);
											const int o = l == kTrunk ? 0 : (l == kPerson && !fr.arm[i]) ? 1 : l == kPlayer ? 2 : l == kBox ? 3 : l == kDiamond ? 4 : -1;
											if (o >= 0) {
												++fObjN[o];
												fObj[o] += moved ? 1 : 0;
												if (o < 2) fdBody = std::max(fdBody, dn);
												if (o >= 3) fGain[o - 3] += aFree[i] - aNow[i] > kGainEps ? 1 : 0;
											}
											if (fr.arm[i]) {
												++fArmN;
												fArm += moved ? 1 : 0;
												fdArm = std::max(fdArm, dn);
												fdArmSum += dn;
											}
											if (l == kPost) {
												++fPostN;
												fPostMoved += moved ? 1 : 0;
												fPost[0] += aNow[i];
												fPost[1] += aFree[i];
											}
											fDpx[0] += moved ? 1 : 0;
											fDpx[1] += moved && l != kBlade && l != kPost && !fr.arm[i] ? 1 : 0;
											fDbf += aFree[i] != aBf[i] ? 1 : 0;
											if (l != kSky) fOwn[1] += ownFree[i];
										}
										if (cover) {
											const double       dc = std::fabs(aCov[i] - aEx[i]), dn = std::fabs(aCov[i] - aNow[i]);
											const std::uint8_t l = fr.label[i];
											if (outlines[f][i] && !fr.arm[i] && l != kBlade) {
												cSb[0] = std::max(cSb[0], d);
												cSb[1] = std::max(cSb[1], dc);
											}
											if (playerRing[f][i]) {
												cSp = std::max(cSp, dc);
												cdP = std::max(cdP, dn);
											}
											if (l == kPlayer) {
												cPl += aCov[i];
												cdP = std::max(cdP, dn);
											}
											if (l == kGround) {
												cdG[0] = std::max(cdG[0], dn);
												cdG[nearObj[f][i] ? 2 : 1] = std::max(cdG[nearObj[f][i] ? 2 : 1], dn);
											}
											if (l != kSky && (fr.ray[i] < a_coverStart || fr.height[i] > outH)) cOut = std::max(cOut, dn);
											if (l == kBox) cGain[0] += aCov[i] - aNow[i] > kGainEps ? 1 : 0;
											if (l == kDiamond) cGain[1] += aCov[i] - aNow[i] > kGainEps ? 1 : 0;
											const int o = l == kTrunk ? 0 : (l == kPerson && !fr.arm[i]) ? 1 : l == kBox ? 2 : l == kDiamond ? 3 : l == kBigBox1 ? 4 : l == kBigBox2 ? 5 :
											              l == kBigTrunk ? 6 : l == kBigBody ? 7 : -1;
											if (o >= 0) {
												const int dep = objDepth[f][i];
												cObj[o][dep] += dn;
												++cObjN[o][dep];
												cObj[o][5] += dn;
												++cObjN[o][5];
											}
										}
										if (!thin) continue;
										const double dt = std::fabs(aThin[i] - aEx[i]);
										const double dd = std::fabs(aThin[i] - aNow[i]);
										if (outlines[f][i]) {
											tS = std::max(tS, dt);
											// Sb: the trunk's and body's own outline, their pixels and the ground's (blades near them take
											// cover by design, the arm is reported apart)
											if (!fr.arm[i] && fr.label[i] != kBlade) {
												tSb[0] = std::max(tSb[0], d);
												tSb[1] = std::max(tSb[1], dt);
											}
										}
										if (playerRing[f][i]) {
											tSp = std::max(tSp, dt);
											dP = std::max(dP, dd);
										}
										if (fr.label[i] == kPlayer) {
											tPl += aThin[i];
											dP = std::max(dP, dd);
										}
										if (fr.label[i] == kGround) dG = std::max(dG, dd);
										if (fr.label[i] == kGround && nearObj[f][i]) gNear = std::max(gNear, dd);
										if (fr.label[i] == kGround && !nearObj[f][i]) gFar = std::max(gFar, dd);
										{
											const int o = fr.label[i] == kTrunk ? 0 : (fr.label[i] == kPerson && !fr.arm[i]) ? 1 : fr.label[i] == kBox ? 2 : fr.label[i] == kDiamond ? 3 : -1;
											if (o >= 0) {
												pdMax[o] = std::max(pdMax[o], dd);
												++pN[o];
												pGain[o] += aThin[i] - aNow[i] > kGainEps ? 1 : 0;
											}
										}
										if ((fr.label[i] == kTrunk || fr.label[i] == kPerson) && !fr.arm[i]) dBody = std::max(dBody, dd);
										if (fr.arm[i]) {
											dArm = std::max(dArm, dd);
											dArmSum += dd;
											++armN;
										}
									}
								}
								if (freeOn && a_free.orderRev) {   // gate STATE: the pass has no history, so the reverse order must give the same alpha
									for (int f = kFrames - 1; f >= 0; --f) {
										const FrameData& fr = frames[f];
										const FrameData& frR = noBlades ? fr : framesG[f];
										const FreeOut    fo = BuildFreeDepth(fr, v, colsCpu[slope][colsKind], a_free.p.src == kFreeRef ? &frR.depth : nullptr, a_free.p);
										const auto       a = Render(g, fr, v, cols, opacity, f, 0, nullptr, 0.0f, &fo.depth, a_free.p.start, a_free.p.maxPx);
										for (std::size_t i = 0; i < a.size(); ++i) fRev += a[i] != fwd[f][i] ? 1 : 0;
									}
								}
								double t[5]{};
								for (int k = 0; k < 5; ++k) {
									for (std::size_t b = 0; b < nb; ++b) {
										const double m = sL[k][b] / kFrames;
										t[k] += std::sqrt(std::max(sL2[k][b] / kFrames - m * m, 0.0));
									}
									t[k] = nb ? t[k] / nb : 0.0;
								}
								auto ratio = [](double a, double b) { return b > 1e-9 ? a / b : 0.0; };
								const std::string name = scene + (colsKind ? "_even_" : "_sim_") + (opacity > 0.3f ? "0.43" : "0.18");
								char kText[32] = "    -     -";
								if (kN) std::snprintf(kText, sizeof(kText), "%5.1f %5.1f", kSum[0] / kN, kSum[1] / kN);
								char cfText[16] = "-";
								if (closed) std::snprintf(cfText, sizeof(cfText), "%.5f", cfMax);
								// exact24's numbers go after cf, so the earlier columns keep their places (scripts read them by position).
								std::printf("%-37s T %6.3f %6.3f %6.3f now/no %5.2f ex/no %5.2f now/ex %5.2f E %.4f K %s S %.4f Sp %.4f player %.6f %.6f F %.4f %.4f cf %s"
								            " x24 T %6.3f now/x24 %5.2f x24/ex %5.2f Eu %.4f Es %.4f\n", name.c_str(), t[0], t[1], t[2],
									ratio(t[0], t[2]), ratio(t[1], t[2]), ratio(t[0], t[1]), eN ? eSum / eN : 0.0, kText, sMax, spMax, plN ? plSum[0] / plN : 0.0, plN ? plSum[1] / plN : 0.0,
									eN ? static_cast<double>(ownBand) / eN : 0.0, allN ? static_cast<double>(ownAll) / allN : 0.0, cfText, t[3], ratio(t[0], t[3]), ratio(t[3], t[1]),
									eN ? euSum / eN : 0.0, eN ? esSum / eN : 0.0);
								if (thin) {
									char tkText[16] = "    -";
									if (kN) std::snprintf(tkText, sizeof(tkText), "%5.1f", tK / kN);
									std::printf("  thin %4.0f: T %6.3f thin/no %5.2f thin/ex %5.2f thin/now %5.2f E %.4f K %s S %.4f Sb %.4f %.4f Sp %.4f player %.6f"
									            " dG %.3g dBand %.3g dP %.3g dBody %.3g dArm %.3g %.4f chg %.4f\n", a_thinStart, t[4], ratio(t[4], t[2]), ratio(t[4], t[1]),
										ratio(t[4], t[0]), eN ? tE / eN : 0.0, tkText, tS, tSb[0], tSb[1], tSp, plN ? tPl / plN : 0.0, dG, dBand, dP, dBody, dArm,
										armN ? dArmSum / armN : 0.0, eN ? static_cast<double>(chg) / eN : 0.0);
									std::printf("  thin+ %4.0f: thin/now %.4f thin/ex %.4f thin/no %.4f dBody trunk %.3g person %.3g box %.3g diamond %.3g gain trunk %lld person %lld box %lld"
									            " diamond %lld px trunk %lld person %lld box %lld diamond %lld dG far %.3g near %.3g Fold %.4f %.4f Frule %.4f %.4f\n", a_thinStart,
										ratio(t[4], t[0]), ratio(t[4], t[1]), ratio(t[4], t[2]), pdMax[0], pdMax[1], pdMax[2], pdMax[3], pGain[0], pGain[1], pGain[2], pGain[3],
										pN[0], pN[1], pN[2], pN[3], gFar, gNear, eN ? static_cast<double>(ownBand) / eN : 0.0, allN ? static_cast<double>(ownAll) / allN : 0.0,
										eN ? static_cast<double>(ownG[0]) / eN : 0.0, allN ? static_cast<double>(ownG[1]) / allN : 0.0);
								}
								if (oracle) {
									// T as t[] (same sums), then the diagnostic T's from the stored band values
									double tO = 0.0, tFog = 0.0, tLab = 0.0, tSt[2]{}, tFl[2]{}, tBf = 0.0, tBfFl = 0.0, mA[4]{};   // mA: blade now, blade free, ground now, ground free
									long long mN[2]{};
									long long nSt = 0, nFl = 0;
									auto      sd = [](const double* a_v, int a_n) {
                                        double m = 0.0, m2 = 0.0;
                                        for (int k = 0; k < a_n; ++k) {
                                            m += a_v[k];
                                            m2 += a_v[k] * a_v[k];
                                        }
                                        m /= a_n;
                                        return std::sqrt(std::max(m2 / a_n - m * m, 0.0));
									};
									for (std::size_t b = 0; b < nb; ++b) {
										const double m = sO[b] / kFrames;
										const double tb = std::sqrt(std::max(sO2[b] / kFrames - m * m, 0.0));
										tO += tb;
										const double mn = sL[0][b] / kFrames;
										const double tn = std::sqrt(std::max(sL2[0][b] / kFrames - mn * mn, 0.0));
										double lg[kFrames], ll[kFrames], lb[kFrames], aMean = 0.0;
										bool   flip = false;
										for (int f = 0; f < kFrames; ++f) {
											const std::size_t j = static_cast<std::size_t>(f) * nb + b;
											lg[f] = kGroundC * (1.0 - gA[j]) + kFogC * gA[j];
											lb[f] = (orBl[j] ? kLeafC : kGroundC) * (1.0 - gA[j]) + kFogC * gA[j];   // every band pixel takes the blade-free alpha
											mA[orBl[j] ? 0 : 2] += nwA[j];
											mA[orBl[j] ? 1 : 3] += gA[j];
											++mN[orBl[j] ? 0 : 1];
											aMean += orA[j];
											flip = flip || orBl[j] != orBl[b];
										}
										aMean /= kFrames;
										for (int f = 0; f < kFrames; ++f) {
											const double c = orBl[static_cast<std::size_t>(f) * nb + b] ? kLeafC : kGroundC;
											ll[f] = c * (1.0 - aMean) + kFogC * aMean;
										}
										tFog += sd(lg, kFrames);
										tLab += sd(ll, kFrames);
										const double tbf = sd(lb, kFrames);
										tBf += tbf;
										if (flip) tBfFl += tbf;
										double* acc = flip ? tFl : tSt;
										acc[0] += tb;
										acc[1] += tn;
										++(flip ? nFl : nSt);
									}
									tO = nb ? tO / nb : 0.0;
									char okText[16] = "    -";
									if (kN) std::snprintf(okText, sizeof(okText), "%5.1f", oK / kN);
									char ocText[48] = "- - -";
									if (closed) std::snprintf(ocText, sizeof(ocText), "%.5f %.5f %.5f", oCf[0], oCf[1], oCf[2]);
									std::printf("  oracle %4.0f: T %7.4f or/no %6.4f or/ex %6.4f or/now %6.4f E %.4f K %s Sb %.4f %.4f Sp %.4f player %.6f off %.3g sub %lld near %lld bad %lld cf %s"
									            " Tfog %.4f Tlab %.4f flip %.4f T st/fl %.4f %.4f now st/fl %.4f %.4f gnd %.4f"
									            " | bf T %.4f bf/now %.4f bf fl %.4f mean a blade now %.4f free %.4f ground now %.4f free %.4f\n", a_oracleStart, tO, ratio(tO, t[2]), ratio(tO, t[1]),
										ratio(tO, t[0]), eN ? oE / eN : 0.0, okText, oSb[0], oSb[1], oSp, plN ? oPl / plN : 0.0, oOff, oSub, oNear, oBad, ocText,
										nb ? tFog / nb : 0.0, nb ? tLab / nb : 0.0, nb ? static_cast<double>(nFl) / nb : 0.0, nSt ? tSt[0] / nSt : 0.0, nFl ? tFl[0] / nFl : 0.0,
										nSt ? tSt[1] / nSt : 0.0, nFl ? tFl[1] / nFl : 0.0, oGndN ? oGnd / oGndN : 0.0,
										nb ? tBf / nb : 0.0, ratio(nb ? tBf / nb : 0.0, t[0]), nFl ? tBfFl / nFl : 0.0, mN[0] ? mA[0] / mN[0] : 0.0, mN[0] ? mA[1] / mN[0] : 0.0,
										mN[1] ? mA[2] / mN[1] : 0.0, mN[1] ? mA[3] / mN[1] : 0.0);
								}
								if (freeOn) {
									// T as t[] (same form of sums); flipd from the stored band values. The values the gates read against a fixed
									// number are printed with more digits than the older rows carry, so that a value just over a limit does not
									// round onto it (free_gate.py reads the row's text).
									double    tF = 0.0, tB = 0.0, flipd[2]{};
									long long nFlip = 0;
									for (std::size_t b = 0; b < nb; ++b) {
										const double mf = sF[b] / kFrames, mb = sB[b] / kFrames;
										tF += std::sqrt(std::max(sF2[b] / kFrames - mf * mf, 0.0));
										tB += std::sqrt(std::max(sB2[b] / kFrames - mb * mb, 0.0));
										double sa[2][2]{};   // [ground, blade][now, free]
										int    cnt[2]{};
										for (int f = 0; f < kFrames; ++f) {
											const std::size_t j = static_cast<std::size_t>(f) * nb + b;
											sa[fBl[j]][0] += fNw[j];
											sa[fBl[j]][1] += fFr[j];
											++cnt[fBl[j]];
										}
										if (cnt[0] && cnt[1]) {
											++nFlip;
											flipd[0] += std::fabs(sa[1][0] / cnt[1] - sa[0][0] / cnt[0]);
											flipd[1] += std::fabs(sa[1][1] / cnt[1] - sa[0][1] / cnt[0]);
										}
									}
									tF = nb ? tF / nb : 0.0;
									tB = nb ? tB / nb : 0.0;
									char fkText[16] = "    -";
									if (kN) std::snprintf(fkText, sizeof(fkText), "%.4f", fK / kN);
									char gapText[24] = "-";
									if (fGapN) std::snprintf(gapText, sizeof(gapText), "%.7f", fGap / fGapN);
									char runText[32] = "- L -";
									if (fMinRun >= 0) std::snprintf(runText, sizeof(runText), "%d L %d", fMinRun, fMinLen);
									const long long allFrames = static_cast<long long>(kFrames) * static_cast<long long>(kSize) * kSize;
									std::printf("  free %4.0f %s%s: T %7.4f free/no %.6f free/ex %.6f free/now %.6f free/bf %.6f K %s Sb %.6f %.6f Sp %.4f player %.6f dG %.9g"
									            " gain box %lld diamond %lld px box %lld diamond %lld closed %.6f part %.6f veto %.6f gap %s Eg %.6f %.6f flipd %.6f %.6f"
									            " obj %lld of %lld minrun %s short %lld F %.9f %.9f"
									            " | bf/now %.6f E %.4f dP %.9g dBody %.9g arm %lld of %lld dArm %.3g %.4f post %lld of %lld a %.4f %.4f dpx %lld %lld dbf %lld of %lld"
									            " Fnow %.9f %.9f flip %.4f blade %lld maxpx %d objs %lld %lld %lld %lld %lld\n",
										a_free.p.start, kFreeSrcName[a_free.p.src], kFreeMutName[a_free.p.mut], tF, ratio(tF, t[2]), ratio(tF, t[1]), ratio(tF, t[0]), ratio(tF, tB), fkText,
										fSb[0], fSb[1], fSp, plN ? fPl / plN : 0.0, fdG, fGain[0], fGain[1], fObjN[3], fObjN[4],
										fBlade ? static_cast<double>(fClosed) / fBlade : 0.0, fBlade ? static_cast<double>(fPart) / fBlade : 0.0,
										fBlade ? static_cast<double>(fVeto) / fBlade : 0.0, gapText, fEgN ? fEg[0] / fEgN : 0.0, fEgN ? fEg[1] / fEgN : 0.0,
										nFlip ? flipd[0] / nFlip : 0.0, nFlip ? flipd[1] / nFlip : 0.0, fObj[0] + fObj[1] + fObj[2] + fObj[3] + fObj[4],
										fObjN[0] + fObjN[1] + fObjN[2] + fObjN[3] + fObjN[4], runText, fShort, eN ? static_cast<double>(fOwn[0]) / eN : 0.0,
										allN ? static_cast<double>(fOwn[1]) / allN : 0.0,
										ratio(tB, t[0]), eN ? fE / eN : 0.0, fdP, fdBody, fArm, fArmN, fdArm, fArmN ? fdArmSum / fArmN : 0.0, fPostMoved, fPostN,
										fPostN ? fPost[0] / fPostN : 0.0, fPostN ? fPost[1] / fPostN : 0.0, fDpx[0], fDpx[1], fDbf, allFrames,
										eN ? static_cast<double>(ownBand) / eN : 0.0, allN ? static_cast<double>(ownAll) / allN : 0.0, nb ? static_cast<double>(nFlip) / nb : 0.0,
										fBlade, a_free.p.maxPx, fObj[0], fObj[1], fObj[2], fObj[3], fObj[4]);
									if (a_free.orderRev) std::printf("  free order rev: %lld differing alpha px over %d frames\n", fRev, kFrames);
								}
								if (cover) {
									double tC = 0.0;
									for (std::size_t b = 0; b < nb; ++b) {
										const double m = sC[b] / kFrames;
										tC += std::sqrt(std::max(sC2[b] / kFrames - m * m, 0.0));
									}
									tC = nb ? tC / nb : 0.0;
									auto mean = [](double a_s, long long a_n) { return a_n ? a_s / static_cast<double>(a_n) : 0.0; };
									char ckText[16] = "-";
									if (kN) std::snprintf(ckText, sizeof(ckText), "%.4f", cK / kN);
									std::printf("  cover %4.0f: T %7.4f cov/no %.6f cov/ex %.6f cov/now %.6f K %s Sb %.6f %.6f Sp %.4f player %.6f dP %.9g dG %.6f far %.6f near %.6f out %.9g"
									            " gain box %lld diamond %lld tmp now %.6f cov %.6f cov/now %.6f thin %.6f oracle %.6f bf cov %.6f %.6f %.6f now %.6f %.6f %.6f E %.4f\n",
										a_coverStart, tC, ratio(tC, t[2]), ratio(tC, t[1]), ratio(tC, t[0]), ckText, cSb[0], cSb[1], cSp, plN ? cPl / plN : 0.0, cdP, cdG[0], cdG[1], cdG[2],
										cOut, cGain[0], cGain[1], mean(cTmp[0], cTmpN), mean(cTmp[1], cTmpN), ratio(mean(cTmp[1], cTmpN), mean(cTmp[0], cTmpN)), mean(cTmp[2], cTmpN),
										mean(cTmp[3], cTmpN), mean(cBf[0], cBfN[0]), mean(cBf[1], cBfN[1]), mean(cBf[2], cBfN[2]), mean(cBf[3], cBfN[0]), mean(cBf[4], cBfN[1]),
										mean(cBf[5], cBfN[2]), mean(cE, eN));
									std::printf("  cover+ %4.0f: k blade full %.4f part %.4f zero %.4f lim height %.4f frac %.4f hpart %.4f blade px %lld dk %.6f cont %.6f len %.1f\n", a_coverStart,
										mean(static_cast<double>(cKb[1]), cKb[0]), mean(static_cast<double>(cKb[2]), cKb[0]), mean(static_cast<double>(cKb[3]), cKb[0]),
										mean(static_cast<double>(cKb[4]), cKb[0]), mean(static_cast<double>(cLimF), cKb[0]), mean(static_cast<double>(cKb[5]), cKb[0]), cKb[0],
										mean(cDk, cDkN), mean(static_cast<double>(cCont), cLowN), mean(cLen, cCont));
									static const char* const kObjName[8]{ "trunk", "body", "box", "diamond", "box70", "box140", "bigtrunk", "bigbody" };
									std::string objText;
									for (int o = 0; o < 8; ++o) {
										if (!cObjN[o][5]) continue;
										char buf[256];
										std::snprintf(buf, sizeof(buf), " %s n %lld %lld %lld %lld %lld d %.5f %.5f %.5f %.5f %.5f in %lld %.6f all %.6f |", kObjName[o], cObjN[o][0], cObjN[o][1],
											cObjN[o][2], cObjN[o][3], cObjN[o][4], mean(cObj[o][0], cObjN[o][0]), mean(cObj[o][1], cObjN[o][1]), mean(cObj[o][2], cObjN[o][2]),
											mean(cObj[o][3], cObjN[o][3]), mean(cObj[o][4], cObjN[o][4]), cObjN[o][3] + cObjN[o][4], mean(cObj[o][3] + cObj[o][4], cObjN[o][3] + cObjN[o][4]),
											mean(cObj[o][5], cObjN[o][5]));
										objText += buf;
									}
									std::printf("  coverobj %4.0f:%s\n", a_coverStart, objText.c_str());
									if (gR2) {
										for (int q = 0; q < kR2; ++q) PrintRow(kR2Name[q], r2acc[q], t, nb);
									}
									if (gMask && gR2) {
										for (int q = 0; q < 4; ++q) {
											PrintRow(kMName[q], mAcc[q], t, nb);
											auto sh = [](long long a_n, long long a_d) { return a_d ? static_cast<double>(a_n) / static_cast<double>(a_d) : 0.0; };
											auto rt = [](double a, double b) { return b > 1e-12 ? a / b : 0.0; };
											char nc[24] = "-";   // M3 has no full-res continuation
											if (q < 3) std::snprintf(nc, sizeof(nc), "%.6f", sh(mNoCont[q], mPx[q]));
											std::printf("  mask %s: px %lld hidden %lld fan 0 kDkRmin %.6f khlow %.6f nocont %s unchanged %.6f tmp blade now %.6f row %.6f ratio %.6f"
											            " ground now %.6f row %.6f ratio %.6f\n",
												kMName[q], mPx[q], q < 2 ? 0LL : mHid, mPx[0] ? mKdr : 1.0, sh(mKhLow, mPx[0]), nc, sh(mSame[q], mPx[q]),
												mTmpN[0] ? mTmp[q][0][1] / mTmpN[0] : 0.0,
												mTmpN[0] ? mTmp[q][0][0] / mTmpN[0] : 0.0, rt(mTmp[q][0][0], mTmp[q][0][1]), mTmpN[1] ? mTmp[q][1][1] / mTmpN[1] : 0.0,
												mTmpN[1] ? mTmp[q][1][0] / mTmpN[1] : 0.0, rt(mTmp[q][1][0], mTmp[q][1][1]));
										}
									}
									if (gSink && gR2) {
										for (int q = 0; q < sRows; ++q) {
											PrintRow(kSName[q], sRow[q], t, nb);
											PrintSink(kSName[q], q == 3 || q == 4 ? "C" : "now", sAcc[q], sNowAcc, sinkBin);
										}
										auto sh = [](long long a_n, long long a_d) { return a_d ? static_cast<double>(a_n) / static_cast<double>(a_d) : 0.0; };
										std::printf("  sinkeval: evaluations %lld (texels %lld own-ray %lld) w>0 %.9f hit %.9f nohit %.9f w>0 on own-ray %lld\n", sEv[0], sEvT, sEvF,
											sh(sEv[1], sEv[0]), sh(sEv[2], sEv[0]), sh(sEv[3], sEv[0]), sEv[4]);
									}
								}
								std::fflush(stdout);
							}
						}
					}
				}
			}
		}
		std::printf("grass: %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
		for (int slope = 0; slope < grounds; ++slope) {
			SafeRelease(evenSrv[slope]);
			SafeRelease(evenTex[slope]);
			SafeRelease(hSrv[slope]);
			SafeRelease(hTex[slope]);
			sims[slope].Release();
		}
		return 0;
	}
}

int main(int argc, char** argv)
{
	using namespace DirectX;
	if (argc < 2) {
		std::fprintf(stderr, "usage: HatchRepro <output folder>\n");
		return 2;
	}
	const std::string outDir = argv[1];
	bool              grass = false, grassQuick = false, grassDump = false, grassDumpAll = false, grassBoxes = false, grassOracleThin = false;
	float             grassThin = 0.0f, grassOracle = 0.0f, grassCover = 0.0f;
	bool              grassBigObj = false;
	std::vector<std::string> coverDefs;   // coverdef NAME=VALUE: the shader's GROUNDFOG_COVER_* variant defines (composite and cover passes)
	std::string       grassOnly;
	Grass::FreeRun    grassFree;
	if (argc > 2 && std::strcmp(argv[2], "grass") == 0) {
		for (int a = 3; a < argc; ++a) {
			grassQuick = grassQuick || std::strcmp(argv[a], "quick") == 0;
			grassDump = grassDump || std::strcmp(argv[a], "dump") == 0;
			grassDumpAll = grassDumpAll || std::strcmp(argv[a], "dumpall") == 0;
			grassBoxes = grassBoxes || std::strcmp(argv[a], "boxes") == 0;
			grassOracleThin = grassOracleThin || std::strcmp(argv[a], "oraclethin") == 0;
			grassFree.eye120 = grassFree.eye120 || std::strcmp(argv[a], "eye120") == 0;
			grassFree.tilt = grassFree.tilt || std::strcmp(argv[a], "tilt") == 0;
			grassFree.post = grassFree.post || std::strcmp(argv[a], "post") == 0;
			grassBigObj = grassBigObj || std::strcmp(argv[a], "bigobj") == 0;
			Grass::gOpacity043Only = Grass::gOpacity043Only || std::strcmp(argv[a], "op43") == 0;
			Grass::gR2 = Grass::gR2 || std::strcmp(argv[a], "r2") == 0;
			Grass::gTurn = Grass::gTurn || std::strcmp(argv[a], "turn") == 0;
			Grass::gMask = Grass::gMask || std::strcmp(argv[a], "mask") == 0;
			Grass::gSink = Grass::gSink || std::strcmp(argv[a], "sink") == 0;
			Grass::gSinkW0 = Grass::gSinkW0 || std::strcmp(argv[a], "sinkw0") == 0;
			if (std::strcmp(argv[a], "thin") == 0 && a + 1 < argc) {
				grassThin = std::strtof(argv[++a], nullptr);
			} else if (std::strcmp(argv[a], "cover") == 0 && a + 1 < argc) {
				grassCover = std::strtof(argv[++a], nullptr);
			} else if (std::strcmp(argv[a], "coverdef") == 0 && a + 1 < argc) {
				const std::string d = argv[++a];
				const auto        eq = d.find('=');
				if (eq == std::string::npos || d.rfind("GROUNDFOG_COVER_", 0) != 0) {
					std::fprintf(stderr, "coverdef takes GROUNDFOG_COVER_<NAME>=<value> (got %s)\n", d.c_str());
					return 2;
				}
				coverDefs.push_back(d);
				const std::string name = d.substr(0, eq), value = d.substr(eq + 1);
				if (name == "GROUNDFOG_COVER_BOX_R") Grass::gCoverBoxR = static_cast<int>(std::strtol(value.c_str(), nullptr, 10));
				if (name == "GROUNDFOG_COVER_FRAC_FULL") Grass::gCoverFracFull = std::strtof(value.c_str(), nullptr);
				if (name == "GROUNDFOG_COVER_FRAC_DEAD") Grass::gCoverFracDead = std::strtof(value.c_str(), nullptr);
				if (name == "GROUNDFOG_COVER_TWOSIDED") Grass::gCoverTwoSided = std::strtol(value.c_str(), nullptr, 10) != 0;
			} else if (std::strcmp(argv[a], "oracle") == 0 && a + 1 < argc) {
				grassOracle = std::strtof(argv[++a], nullptr);
			} else if (std::strcmp(argv[a], "only") == 0 && a + 1 < argc) {
				grassOnly = argv[++a];
			} else if (std::strcmp(argv[a], "free") == 0 && a + 1 < argc) {
				grassFree.p.start = std::strtof(argv[++a], nullptr);
			} else if (std::strcmp(argv[a], "freesrc") == 0 && a + 1 < argc) {
				const char* w = argv[++a];
				if (std::strcmp(w, "ref") == 0) {
					grassFree.p.src = Grass::kFreeRef;
				} else if (std::strcmp(w, "close") == 0) {
					grassFree.p.src = Grass::kFreeClose;
				} else {
					std::fprintf(stderr, "freesrc must be ref or close (got %s; the shader source comes with step S3)\n", w);
					return 2;
				}
			} else if (std::strcmp(argv[a], "freepx") == 0 && a + 1 < argc) {
				grassFree.p.maxPx = static_cast<int>(std::strtol(argv[++a], nullptr, 10));
			} else if (std::strcmp(argv[a], "freemut") == 0 && a + 1 < argc) {
				const char* w = argv[++a];
				if (std::strcmp(w, "lpx1") == 0) {
					grassFree.p.mut = Grass::kMutLpx1;
				} else if (std::strcmp(w, "noveto") == 0) {
					grassFree.p.mut = Grass::kMutNoVeto;
				} else if (std::strcmp(w, "nogate") == 0) {
					grassFree.p.mut = Grass::kMutNoGate;
				} else {
					std::fprintf(stderr, "freemut must be lpx1, noveto or nogate (got %s)\n", w);
					return 2;
				}
			} else if (std::strcmp(argv[a], "order") == 0 && a + 1 < argc) {
				const char* w = argv[++a];
				if (std::strcmp(w, "rev") != 0) {
					std::fprintf(stderr, "order can only be rev (got %s)\n", w);
					return 2;
				}
				grassFree.orderRev = true;
			}
		}
		if (!(grassFree.p.start >= 0.0f && grassFree.p.start <= 3000.0f)) {
			std::fprintf(stderr, "free must be 0..3000 units, the ini's BladeFreeStart range (got %g)\n", grassFree.p.start);
			return 2;
		}
		if (grassFree.p.maxPx < Grass::kFreeMinPx || grassFree.p.maxPx > Grass::kFreePxLimit) {
			std::fprintf(stderr, "freepx must be %d..%d pixels (got %d)\n", Grass::kFreeMinPx, Grass::kFreePxLimit, grassFree.p.maxPx);
			return 2;
		}
		if (!(grassThin >= 0.0f && grassThin <= 3000.0f)) {
			std::fprintf(stderr, "thin must be 0..3000 units, the ini's ThinFogStart range (got %g)\n", grassThin);
			return 2;
		}
		if (!(grassCover >= 0.0f && grassCover <= 3000.0f)) {
			std::fprintf(stderr, "cover must be 0..3000 units, the ini's GrassCover range (got %g)\n", grassCover);
			return 2;
		}
		if (!(grassOracle >= 0.0f && grassOracle <= 3000.0f)) {
			std::fprintf(stderr, "oracle must be 0..3000 units, a ThinFogStart (got %g)\n", grassOracle);
			return 2;
		}
		if (Grass::gMask && !(grassCover > 0.0f && Grass::gR2)) {
			std::fprintf(stderr, "mask needs cover <units> and r2 (its rows are r2 rows)\n");
			return 2;
		}
		if ((Grass::gSink || Grass::gSinkW0) && !(Grass::gSink && grassCover > 0.0f && Grass::gR2)) {
			std::fprintf(stderr, "sink needs cover <units> and r2 (its rows are r2 rows); sinkw0 needs sink\n");
			return 2;
		}
		grass = true;
		argc = 2;   // the positional arguments below do not apply
	}
	if (argc > 2 && std::strcmp(argv[2], "video") == 0) {
		if (argc < 5) {
			std::fprintf(stderr, "usage: HatchRepro <output folder> video <old|off|new> <frames>\n");
			return 2;
		}
		kVideoVariant = std::strcmp(argv[3], "old") == 0 ? 0 : std::strcmp(argv[3], "off") == 0 ? 1 : 2;
		kVideoFrames = std::clamp(static_cast<int>(std::strtol(argv[4], nullptr, 10)), 1, 2000);
		kFovYDeg = 60.0f;
		kRoofDeg = 30.0f;
		kFarNoise = 1.0f;
		argc = 2;   // the positional arguments below do not apply
	}
	if (argc > 2) kNear = std::strtof(argv[2], nullptr);
	if (argc > 3) kFovYDeg = std::strtof(argv[3], nullptr);
	if (argc > 4) kRoofDeg = std::strtof(argv[4], nullptr);
	if (!(kRoofDeg >= 5.0f && kRoofDeg <= 85.0f)) {
		std::fprintf(stderr, "roof slope must be 5..85 degrees (got %s)\n", argv[4]);
		return 2;
	}
	kHill = kVideoFrames > 0 || (argc > 5 && std::strcmp(argv[5], "hill") == 0);
	if (argc > 5 && !kHill && std::strcmp(argv[5], "-") != 0) {   // "-" = wall and roof, to reach the later arguments
		std::fprintf(stderr, "the fifth argument can only be \"hill\" (got %s)\n", argv[5]);
		return 2;
	}
	if (argc > 6) kHillRough = std::strtof(argv[6], nullptr);
	if (argc > 7) kFarNoise = std::clamp(std::strtof(argv[7], nullptr), 0.0f, 1.0f);
	if (argc > 8) {
		const float deg = std::strtof(argv[8], nullptr);
		if (!(deg >= -30.0f && deg <= 30.0f)) {
			std::fprintf(stderr, "the ground slope must be -30..30 degrees (got %s)\n", argv[8]);
			return 2;
		}
		kGroundTan = std::tan(XMConvertToRadians(deg));
	}
	if (argc > 9) kTable = std::strtol(argv[9], nullptr, 10) != 0;
	const bool sloped = kGroundTan != 0.0f;   // a sloped ground has no wall, roof or hill on it
	if (!(kHillRough >= 0.0f && kHillRough <= 20.0f)) {
		std::fprintf(stderr, "the hill roughness must be 0..20 (got %s)\n", argv[6]);
		return 2;
	}
	const float roofTan = std::tan(XMConvertToRadians(kRoofDeg));
	if (!(kNear >= 0.5f && kNear <= 100.0f) || !(kFovYDeg >= 5.0f && kFovYDeg <= 120.0f)) {
		std::fprintf(stderr, "near plane must be 0.5..100 and the field of view 5..120 degrees (got %s %s)\n", argc > 2 ? argv[2] : "-", argc > 3 ? argv[3] : "-");
		return 2;
	}
	ID3D11Device*        dev = nullptr;
	ID3D11DeviceContext* ctx = nullptr;
	if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
		std::fprintf(stderr, "WARP device failed\n");
		return 1;
	}

	// Wind field at rest on flat ground, as RaymarchExport.
	WindSim     sim;
	std::string error;
	if (!sim.Create(dev, kRes, error)) {
		std::fprintf(stderr, "WindSim: %s\n", error.c_str());
		return 1;
	}
	sim.Reset(ctx);
	std::vector<float>   heights(kHeightSize * kHeightSize, 0.0f);
	for (int hy = 0; hy < kHeightSize; ++hy) {
		for (int hx = 0; hx < kHeightSize; ++hx) heights[static_cast<std::size_t>(hy) * kHeightSize + hx] = kGroundTan * (kOrigin + hy * kExtent / (kHeightSize - 1));
	}
	// Coarse table for the far layer: ground z and no pool; unknown outside the loaded land.
	const float        coarseOrigin = -0.5f * kCoarseCell * (kCoarseSize - 1);
	std::vector<float> coarseRg(static_cast<std::size_t>(kCoarseSize) * kCoarseSize * 2, 0.0f);
	for (int cy = 0; cy < kCoarseSize; ++cy) {
		for (int cx = 0; cx < kCoarseSize; ++cx) {
			const float wx = coarseOrigin + cx * kCoarseCell, wy = coarseOrigin + cy * kCoarseCell;
			const bool  known = std::fabs(wx) <= kKnownHalf && std::fabs(wy) <= kKnownHalf;
			coarseRg[(static_cast<std::size_t>(cy) * kCoarseSize + cx) * 2] = known ? kGroundTan * wy : -1.0e30f;
			coarseRg[(static_cast<std::size_t>(cy) * kCoarseSize + cx) * 2 + 1] = 1.0f;   // slope share: these scenes' ground carries all of the layer
		}
	}
	D3D11_TEXTURE2D_DESC cd{};
	cd.Width = cd.Height = kCoarseSize;
	cd.MipLevels = cd.ArraySize = 1;
	cd.Format = DXGI_FORMAT_R32G32_FLOAT;
	cd.SampleDesc.Count = 1;
	cd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	D3D11_SUBRESOURCE_DATA    cinit{ coarseRg.data(), kCoarseSize * sizeof(float) * 2, 0 };
	ID3D11Texture2D*          cTex = nullptr;
	ID3D11ShaderResourceView* cSrv = nullptr;
	dev->CreateTexture2D(&cd, &cinit, &cTex);
	dev->CreateShaderResourceView(cTex, nullptr, &cSrv);
	D3D11_TEXTURE2D_DESC hd{};
	hd.Width = hd.Height = kHeightSize;
	hd.MipLevels = hd.ArraySize = 1;
	hd.Format = DXGI_FORMAT_R32_FLOAT;
	hd.SampleDesc.Count = 1;
	hd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	D3D11_SUBRESOURCE_DATA    hinit{ heights.data(), kHeightSize * sizeof(float), 0 };
	ID3D11Texture2D*          hTex = nullptr;
	ID3D11ShaderResourceView* hSrv = nullptr;
	dev->CreateTexture2D(&hd, &hinit, &hTex);
	dev->CreateShaderResourceView(hTex, nullptr, &hSrv);
	WindShaders::SimCB scb{};
	scb.grid[0] = kOrigin;
	scb.grid[1] = kOrigin;
	scb.grid[2] = kExtent / kRes;
	scb.grid[3] = kRes;
	scb.time[0] = 1.0f / 30.0f;
	scb.shift[3] = kHeightSize;
	scb.counts[2] = 1.2f;
	scb.heightGrid[0] = kOrigin;
	scb.heightGrid[1] = kOrigin;
	scb.heightGrid[2] = kExtent / (kHeightSize - 1);
	scb.heightGrid[3] = 1.0f;
	sim.Step(ctx, scb, hSrv, 5);
	sim.BuildColumns(ctx, hSrv, kThickness, 0.0f, 0.0f, 0.0f);
	const auto cols = ReadFloats(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
	float      maxH = 0.0f;
	for (int i = 0; i < kRes * kRes; ++i) maxH = std::max(maxH, cols[i * 4]);

	auto compile = [&](const char* a_body, const char* a_target, bool a_coverDefs = false, const char* a_maskValue = nullptr,
	                   const std::vector<std::pair<std::string, std::string>>& a_extra = {}) -> ID3DBlob* {
		std::string src = RaymarchShaders::kCommon;
		src += a_body;
		ID3DBlob* code = nullptr;
		ID3DBlob* errs = nullptr;
		std::vector<std::string>      names, values;
		std::vector<D3D_SHADER_MACRO> macros;
		if (a_coverDefs) {
			for (const auto& d : coverDefs) {
				names.push_back(d.substr(0, d.find('=')));
				values.push_back(d.substr(d.find('=') + 1));
			}
			for (std::size_t m = 0; m < names.size(); ++m) macros.push_back({ names[m].c_str(), values[m].c_str() });
			if (a_maskValue) macros.push_back({ "GROUNDFOG_COVER_MASK", a_maskValue });
		}
		for (const auto& e : a_extra) macros.push_back({ e.first.c_str(), e.second.c_str() });
		if (a_coverDefs || !a_extra.empty()) macros.push_back({ nullptr, nullptr });
		if (FAILED(D3DCompile(src.data(), src.size(), "Ray", macros.empty() ? nullptr : macros.data(), nullptr, "main", a_target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errs))) {
			std::fprintf(stderr, "%s\n", errs ? static_cast<const char*>(errs->GetBufferPointer()) : "compile failed");
			std::exit(1);
		}
		SafeRelease(errs);
		return code;
	};
	ID3DBlob*           vsb = compile(RaymarchShaders::kVS, "vs_5_0");
	ID3DBlob*           psb = compile(RaymarchShaders::kMarchPS, "ps_5_0");
	ID3DBlob*           cpb = compile(RaymarchShaders::kCompositePS, "ps_5_0", true);
	ID3DBlob*           cvb = compile(RaymarchShaders::kCoverPS, "ps_5_0", true);
	ID3DBlob*           cfb = compile(RaymarchShaders::kCoverFlagPS, "ps_5_0", true);
	if (!coverDefs.empty()) {
		std::printf("coverdef (prototype variant):");
		for (const auto& d : coverDefs) std::printf(" %s", d.c_str());
		std::printf("\n");
	}
	ID3D11PixelShader*  coverPs = nullptr;
	ID3D11PixelShader*  coverFlagPs = nullptr;
	dev->CreatePixelShader(cvb->GetBufferPointer(), cvb->GetBufferSize(), nullptr, &coverPs);
	dev->CreatePixelShader(cfb->GetBufferPointer(), cfb->GetBufferSize(), nullptr, &coverFlagPs);
	SafeRelease(cvb);
	SafeRelease(cfb);
	ID3D11VertexShader* vs = nullptr;
	ID3D11PixelShader*  ps = nullptr;
	ID3D11PixelShader*  comp = nullptr;
	dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
	dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps);
	dev->CreatePixelShader(cpb->GetBufferPointer(), cpb->GetBufferSize(), nullptr, &comp);
	SafeRelease(vsb);
	SafeRelease(psb);
	SafeRelease(cpb);

	const UINT           lowW = kSize / 2;   // the game marches at half resolution
	D3D11_TEXTURE2D_DESC td{};
	td.Width = td.Height = lowW;
	td.MipLevels = td.ArraySize = 1;
	td.SampleDesc.Count = 1;
	td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	ID3D11Texture2D*          lowTex = nullptr;
	ID3D11RenderTargetView*   lowRtv = nullptr;
	ID3D11ShaderResourceView* lowSrv = nullptr;
	dev->CreateTexture2D(&td, nullptr, &lowTex);
	dev->CreateRenderTargetView(lowTex, nullptr, &lowRtv);
	dev->CreateShaderResourceView(lowTex, nullptr, &lowSrv);
	ID3D11Texture2D*          coverTex = nullptr;   // GrassCover's continuation (same format and size as the march target)
	ID3D11RenderTargetView*   coverRtv = nullptr;
	ID3D11ShaderResourceView* coverSrv = nullptr;
	dev->CreateTexture2D(&td, nullptr, &coverTex);
	dev->CreateRenderTargetView(coverTex, nullptr, &coverRtv);
	dev->CreateShaderResourceView(coverTex, nullptr, &coverSrv);
	td.Width = td.Height = kSize;
	td.BindFlags = D3D11_BIND_RENDER_TARGET;
	ID3D11Texture2D*        sceneTex = nullptr;   // black scene, white fog: red = fog alpha after the blend
	ID3D11RenderTargetView* sceneRtv = nullptr;
	dev->CreateTexture2D(&td, nullptr, &sceneTex);
	dev->CreateRenderTargetView(sceneTex, nullptr, &sceneRtv);
	td.Format = DXGI_FORMAT_R32_FLOAT;
	td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	std::vector<float>        depthData(static_cast<std::size_t>(kSize) * kSize, 1.0f);
	D3D11_SUBRESOURCE_DATA    dinit{ depthData.data(), kSize * sizeof(float), 0 };
	ID3D11Texture2D*          depthTex = nullptr;
	ID3D11ShaderResourceView* depthSrv = nullptr;
	dev->CreateTexture2D(&td, &dinit, &depthTex);
	dev->CreateShaderResourceView(depthTex, nullptr, &depthSrv);
	td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	ID3D11Texture2D*          flagTex = nullptr;   // GrassCover's ground-like flags (full resolution; R8 in the plugin)
	ID3D11RenderTargetView*   flagRtv = nullptr;
	ID3D11ShaderResourceView* flagSrv = nullptr;
	dev->CreateTexture2D(&td, nullptr, &flagTex);
	dev->CreateRenderTargetView(flagTex, nullptr, &flagRtv);
	dev->CreateShaderResourceView(flagTex, nullptr, &flagSrv);
	ID3D11PixelShader*        compMask[3]{};   // CLI mask: GROUNDFOG_COVER_MASK 1 and 2, and the full-res march and continuation targets
	ID3D11Texture2D*          marchFullTex = nullptr;
	ID3D11RenderTargetView*   marchFullRtv = nullptr;
	ID3D11ShaderResourceView* marchFullSrv = nullptr;
	ID3D11Texture2D*          coverFullTex = nullptr;
	ID3D11RenderTargetView*   coverFullRtv = nullptr;
	ID3D11ShaderResourceView* coverFullSrv = nullptr;
	if (Grass::gMask) {
		for (int m = 1; m <= 2; ++m) {
			ID3DBlob* b = compile(RaymarchShaders::kCompositePS, "ps_5_0", true, m == 1 ? "1" : "2");
			dev->CreatePixelShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &compMask[m]);
			SafeRelease(b);
		}
		D3D11_TEXTURE2D_DESC fd = td;
		fd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		dev->CreateTexture2D(&fd, nullptr, &marchFullTex);
		dev->CreateRenderTargetView(marchFullTex, nullptr, &marchFullRtv);
		dev->CreateShaderResourceView(marchFullTex, nullptr, &marchFullSrv);
		dev->CreateTexture2D(&fd, nullptr, &coverFullTex);
		dev->CreateRenderTargetView(coverFullTex, nullptr, &coverFullRtv);
		dev->CreateShaderResourceView(coverFullTex, nullptr, &coverFullSrv);
	}
	ID3D11PixelShader*      marchSink[4]{};   // CLI sink: S, V, S+e, S-e (march and composite), the diagnostic march, its full-res target
	ID3D11PixelShader*      compSink[4]{};
	ID3D11PixelShader*      marchSinkDiag = nullptr;
	ID3D11Texture2D*        sinkFullTex = nullptr;
	ID3D11RenderTargetView* sinkFullRtv = nullptr;
	if (Grass::gSink) {
		static const char* const kMode[4]{ "1", "2", "1", "1" };
		static const char* const kShift[4]{ "0.0", "0.0", "13.4", "-13.4" };
		for (int s = 0; s < 4; ++s) {
			std::vector<std::pair<std::string, std::string>> ex{ { "GROUNDFOG_SINK", kMode[s] }, { "GROUNDFOG_SINK_GSHIFT", kShift[s] } };
			if (Grass::gSinkW0) ex.push_back({ "GROUNDFOG_SINK_W0", "1" });
			ID3DBlob* mb = compile(RaymarchShaders::kMarchPS, "ps_5_0", false, nullptr, ex);
			ID3DBlob* sb = compile(RaymarchShaders::kCompositePS, "ps_5_0", true, nullptr, ex);
			dev->CreatePixelShader(mb->GetBufferPointer(), mb->GetBufferSize(), nullptr, &marchSink[s]);
			dev->CreatePixelShader(sb->GetBufferPointer(), sb->GetBufferSize(), nullptr, &compSink[s]);
			SafeRelease(mb);
			SafeRelease(sb);
		}
		ID3DBlob* db = compile(RaymarchShaders::kMarchPS, "ps_5_0", false, nullptr, { { "GROUNDFOG_SINK", "1" }, { "GROUNDFOG_SINK_DIAG", "1" } });
		dev->CreatePixelShader(db->GetBufferPointer(), db->GetBufferSize(), nullptr, &marchSinkDiag);
		SafeRelease(db);
		D3D11_TEXTURE2D_DESC sd2 = td;
		sd2.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		sd2.BindFlags = D3D11_BIND_RENDER_TARGET;
		dev->CreateTexture2D(&sd2, nullptr, &sinkFullTex);
		dev->CreateRenderTargetView(sinkFullTex, nullptr, &sinkFullRtv);
	}
	D3D11_BUFFER_DESC cbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
	ID3D11Buffer*     cb = nullptr;
	dev->CreateBuffer(&cbd, nullptr, &cb);
	D3D11_SAMPLER_DESC sd{};
	sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	sd.MaxLOD = D3D11_FLOAT32_MAX;
	ID3D11SamplerState* linear = nullptr;
	dev->CreateSamplerState(&sd, &linear);
	D3D11_BLEND_DESC bd{};
	bd.RenderTarget[0].BlendEnable = TRUE;
	bd.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
	bd.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
	bd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
	bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO;
	bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
	bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
	bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
	ID3D11BlendState* blend = nullptr;
	dev->CreateBlendState(&bd, &blend);

	const float               black[4]{};
	ID3D11ShaderResourceView* none[3]{};
	int                       exitCode = 0;
	if (grass) {
		const Grass::Gpu gpu{ dev, ctx, vs, ps, comp, lowTex, lowRtv, lowSrv, sceneTex, sceneRtv, depthTex, depthSrv, cb, linear, blend, coverPs, coverFlagPs,
			coverTex, coverRtv, coverSrv, flagTex, flagRtv, flagSrv, { nullptr, compMask[1], compMask[2] }, marchFullTex, marchFullRtv, marchFullSrv,
			coverFullTex, coverFullRtv, coverFullSrv, { marchSink[0], marchSink[1], marchSink[2], marchSink[3] }, { compSink[0], compSink[1], compSink[2], compSink[3] },
			marchSinkDiag, sinkFullTex, sinkFullRtv };
		exitCode = Grass::Run(gpu, outDir, grassQuick, grassDump, grassDumpAll, grassThin, grassOnly, grassOracle, grassBoxes, grassOracleThin, grassFree, grassCover,
			grassBigObj);
	} else {
		std::printf("field %.0f units, layer H up to %.1f, %u x %u px over %.0f degrees, near plane %.1f, roof %.0f degrees, march at half resolution, %d steps, %d frames\n", kExtent, maxH, kSize, kSize, kFovYDeg, kNear, kRoofDeg, kSteps, kFrames);
		std::printf("%-22s %-7s %8s %9s %12s %12s %10s\n", "case", "surface", "pixels", "mean fog", "noise 1 frm", "noise avg", "fogged %");
	}

	std::ofstream videoOut;
	if (kVideoFrames) videoOut.open(outDir + "/video_" + (kVideoVariant == 0 ? "old" : kVideoVariant == 1 ? "off" : "new") + ".u8", std::ios::binary);
	for (const float dist : kDistances) {
		if (grass || (kVideoFrames && dist != kVideoDist)) continue;
		const float    wallH = kHill ? roofTan * 0.3f * dist : 0.22f * dist, roofLen = 0.18f * dist;
		const std::vector<Tri> hill = kHill ? BuildHill(dist, roofTan) : std::vector<Tri>{};
		const XMFLOAT3 cam{ 0.0f, 0.0f, kEyeZ };
		const XMVECTOR fwd = XMVector3Normalize(XMVectorSet(0.0f, dist, (sloped ? kGroundTan * dist : wallH * 0.55f) - kEyeZ, 0.0f));
		const XMMATRIX view = XMMatrixLookToLH(XMVectorZero(), fwd, XMVectorSet(0, 0, 1, 0));

		for (int rev = 0; rev < (kVideoFrames ? 1 : 2); ++rev) {
			// Reversed depth: near and far swapped in the projection (1 at the near plane, 0 at the far plane).
			const XMMATRIX proj = rev ? XMMatrixPerspectiveFovLH(XMConvertToRadians(kFovYDeg), 1.0f, kFar, kNear)
									  : XMMatrixPerspectiveFovLH(XMConvertToRadians(kFovYDeg), 1.0f, kNear, kFar);
			XMFLOAT4X4 pf;
			XMStoreFloat4x4(&pf, proj);
			const XMMATRIX vp = XMMatrixMultiply(view, proj);
			const XMMATRIX invVp = XMMatrixInverse(nullptr, vp);
			XMFLOAT4X4     inv;
			XMStoreFloat4x4(&inv, invVp);

			// Depth and surface label per pixel: ground z = 0, wall at y = dist up to wallH, roof rising behind it.
			std::vector<std::uint8_t> label(static_cast<std::size_t>(kSize) * kSize, 0);
			std::vector<std::uint16_t> triId(static_cast<std::size_t>(kSize) * kSize, 0);
			std::vector<float>         groundR(static_cast<std::size_t>(kSize) * kSize, -1.0f);   // horizontal distance of ground pixels
			for (UINT y = 0; y < kSize; ++y) {
				for (UINT x = 0; x < kSize; ++x) {
					XMFLOAT4 q;
					XMStoreFloat4(&q, XMVector4Transform(XMVectorSet((x + 0.5f) / kSize * 2 - 1, 1 - (y + 0.5f) / kSize * 2, 0.5f, 1), invVp));
					const XMFLOAT3 d{ q.x / q.w, q.y / q.w, q.z / q.w };   // a point on the pixel's ray, relative to the camera
					float          t = -1.0f;
					std::uint8_t   what = 0;
					auto           take = [&](float a_t, std::uint8_t a_what) {
						if (a_t > 0.0f && (t <= 0.0f || a_t < t)) {
							t = a_t;
							what = a_what;
						}
					};
					if (d.z - kGroundTan * d.y < -1e-9f) take(-(cam.z - kGroundTan * cam.y) / (d.z - kGroundTan * d.y), 1);   // ground z = kGroundTan y
					std::uint16_t hitTri = 0;
					if (sloped) {
					} else if (kHill) {
						const double o[3]{ cam.x, cam.y, cam.z }, dd[3]{ d.x, d.y, d.z };
						double       best = -1.0;
						for (std::size_t k = 0; k < hill.size(); ++k) {
							const double th = HitTri(o, dd, hill[k]);
							if (th > 0.0 && (best < 0.0 || th < best)) {
								best = th;
								hitTri = static_cast<std::uint16_t>(k + 1);
							}
						}
						if (best > 0.0) take(static_cast<float>(best), 4);
					} else {
						if (d.y > 1e-9f) {
							const float tw = (dist - cam.y) / d.y;
							const float hz = cam.z + d.z * tw;
							if (hz >= 0.0f && hz <= wallH) take(tw, 2);
						}
						if (std::fabs(d.z - roofTan * d.y) > 1e-9f) {   // roof plane z - roofTan y = wallH - roofTan dist
							const float tr = (wallH - roofTan * dist - (cam.z - roofTan * cam.y)) / (d.z - roofTan * d.y);
							const float hy = cam.y + d.y * tr;
							if (hy >= dist && hy <= dist + roofLen) take(tr, 3);
						}
					}
					const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
					triId[i] = what == 4 ? hitTri : std::uint16_t{ 0 };
					if (what == 1) groundR[i] = t * std::sqrt(d.x * d.x + d.y * d.y);
					label[i] = what;
					if (t <= 0.0f) {
						depthData[i] = rev ? 0.0f : 1.0f;
						continue;
					}
					XMFLOAT4 b;
					XMStoreFloat4(&b, XMVector4Transform(XMVectorSet(d.x * t, d.y * t, d.z * t, 1), vp));
					double z = static_cast<double>(b.z) / b.w;
					if (!rev) z = std::round(z * 16777215.0) / 16777215.0;   // D24_UNORM
					depthData[i] = static_cast<float>(z);
				}
			}
			ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);
			if (!rev) WriteBin(outDir + "/labels_" + std::to_string(static_cast<int>(dist)) + ".bin", label.data(), label.size());
			if (!rev && kHill) WriteBin(outDir + "/tris_" + std::to_string(static_cast<int>(dist)) + ".bin", triId.data(), triId.size() * sizeof(std::uint16_t));

			for (int mode = kVideoFrames ? 2 : 0; mode < 3; ++mode) {   // 0 field, 1 far, 2 both
				const char*        modeName = mode == 0 ? "field" : mode == 1 ? "far" : "both";
				std::vector<float> first, mean(static_cast<std::size_t>(kSize) * kSize, 0.0f);
				for (int frame = 0; frame < (kVideoFrames ? kVideoFrames : kFrames); ++frame) {
					const float filmTime = kVideoFrames ? frame / kVideoFps : 0.0f;
					const float travelX = kVideoWind * filmTime, travelY = 0.35f * kVideoWind * filmTime;
					if (kVideoFrames) sim.BuildColumns(ctx, hSrv, kThickness, filmTime, travelX, travelY);
					RaymarchShaders::RayCB rc{};
					std::memcpy(rc.invViewProj, &inv, sizeof(rc.invViewProj));
					rc.field[0] = kOrigin;
					rc.field[1] = kOrigin;
					rc.field[2] = kExtent;
					rc.field[3] = mode == 1 ? 0.0f : 1.0f;
					rc.box[0] = -1.0f - std::fabs(kGroundTan * kOrigin);   // the field's lowest ground
					rc.box[1] = kThickness + 2.0f + std::fabs(kGroundTan * kOrigin);
					rc.box[2] = kExtent;   // FogRenderer: never march further than the field size
					rc.box[3] = static_cast<float>(kSteps);
					rc.fog[0] = kOpacity;
					rc.fog[1] = kSteepness;
					rc.fog[2] = static_cast<float>(frame);
					rc.depth[0] = pf._33;
					rc.depth[1] = pf._43;
					rc.depth[2] = static_cast<float>(kSize);
					rc.depth[3] = static_cast<float>(kSize);
					rc.lowRes[0] = static_cast<float>(lowW);
					rc.lowRes[1] = static_cast<float>(lowW);
					rc.lowRes[2] = static_cast<float>(kSize) / lowW;
					for (int c = 0; c < 3; ++c) {
						rc.colorLow[c] = 1.0f;
						rc.colorHigh[c] = 1.0f;
					}
					rc.distFog[0] = 1.0e6f;
					rc.distFog[1] = 2.0e6f;
					rc.distFog[2] = 1.0f;
					rc.posAdjust[0] = cam.x;
					rc.posAdjust[1] = cam.y;
					rc.posAdjust[2] = cam.z;
					rc.farNoise[0] = travelX;
					rc.farNoise[1] = travelY;
					rc.farNoise[2] = filmTime;
					if (mode != 0 && kTable) SetLandGrid(rc, coarseOrigin, coarseOrigin, kCoarseCell, static_cast<float>(kCoarseSize));   // the far layer's land table
					if (mode != 0) {
						rc.farFog[0] = 1.0f;
						rc.farFog[1] = kFarBase;
						rc.farFog[2] = kThickness;
						rc.farFog[3] = kFarNoise;
						if (kVideoFrames && kVideoVariant == 0) rc.farFog[1] = 4000.0f;   // the old shader's FlatLength
						if (kVideoFrames && kVideoVariant == 1) rc.farFog[0] = 0.0f;      // [FarFog] Enable = 0
					}
					ctx->UpdateSubresource(cb, 0, nullptr, &rc, 0, 0);

					D3D11_VIEWPORT lowVp{ 0, 0, static_cast<float>(lowW), static_cast<float>(lowW), 0, 1 };
					ctx->ClearRenderTargetView(lowRtv, black);
					ctx->RSSetViewports(1, &lowVp);
					ctx->OMSetRenderTargets(1, &lowRtv, nullptr);
					ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);
					ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
					ctx->VSSetShader(vs, nullptr, 0);
					ctx->PSSetShader(ps, nullptr, 0);
					ctx->PSSetConstantBuffers(0, 1, &cb);
					ID3D11ShaderResourceView* in[2]{ depthSrv, sim.ColumnSrv() };
					ctx->PSSetShaderResources(0, 2, in);
					ctx->PSSetSamplers(0, 1, &linear);
					ctx->Draw(3, 0);
					ctx->PSSetShaderResources(0, 3, none);

					D3D11_VIEWPORT fullVp{ 0, 0, static_cast<float>(kSize), static_cast<float>(kSize), 0, 1 };
					ctx->ClearRenderTargetView(sceneRtv, black);
					ctx->RSSetViewports(1, &fullVp);
					ctx->OMSetRenderTargets(1, &sceneRtv, nullptr);
					ctx->OMSetBlendState(blend, black, 0xFFFFFFFF);
					ctx->PSSetShader(comp, nullptr, 0);
					ID3D11ShaderResourceView* compIn[3]{ depthSrv, sim.ColumnSrv(), lowSrv };
					ctx->PSSetShaderResources(0, 3, compIn);
					ctx->PSSetShaderResources(6, 1, &cSrv);
					ctx->Draw(3, 0);
					ctx->PSSetShaderResources(0, 3, none);
					ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);

					const auto         img = ReadFloats(dev, ctx, sceneTex, kSize, kSize, 4);
					std::vector<float> fog(static_cast<std::size_t>(kSize) * kSize);
					for (std::size_t i = 0; i < fog.size(); ++i) {
						fog[i] = img[i * 4];
						mean[i] += fog[i] / kFrames;
					}
					if (frame == 0) first = fog;
					if (kVideoFrames) {
						std::vector<std::uint8_t> bytes(fog.size());
						for (std::size_t i = 0; i < fog.size(); ++i) bytes[i] = static_cast<std::uint8_t>(std::clamp(fog[i], 0.0f, 1.0f) * 255.0f + 0.5f);
						videoOut.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
					}
				}
				const std::string name = std::to_string(static_cast<int>(dist)) + "_" + modeName + "_" + (rev ? "rev" : "z24");
				for (const char* dump : kDump) {
					if (name == dump) {
						WriteBin(outDir + "/fog_" + name + ".bin", first.data(), first.size() * sizeof(float));
						WriteBin(outDir + "/fogavg_" + name + ".bin", mean.data(), mean.size() * sizeof(float));
					}
				}
				if (kHill && mode == 1) WriteBin(outDir + "/fog_" + name + ".bin", first.data(), first.size() * sizeof(float));
				const char* surfaces[] = { "sky", "ground", "wall", "roof", "hill" };
				for (std::uint8_t s = 1; s <= 4; ++s) {
					int    n1 = 0, n2 = 0, count = 0, fogged = 0;
					double sum = 0.0;
					for (std::size_t i = 0; i < first.size(); ++i) {
						if (label[i] != s) continue;
						++count;
						sum += first[i];
						fogged += first[i] > 0.004f ? 1 : 0;   // one step of an 8-bit colour channel
					}
					if (!count) continue;   // the scene has no such surface (wall and roof, or the hill)
					const double r1 = HighPassRms(first, label, s, n1);
					const double ra = HighPassRms(mean, label, s, n2);
					std::printf("%-22s %-7s %8d %9.4f %12.5f %12.5f %9.1f%%\n", name.c_str(), surfaces[s], count, count ? sum / count : 0.0, r1, ra,
						count ? 100.0 * fogged / count : 0.0);
				}
				if (!rev && dist >= 39000.0f) {
					// Fog over the open ground by distance: the middle 40 columns, mean alpha of the pixels within 6% of each distance.
					const float marks[] = { 700.0f, 1400.0f, 2100.0f, 2800.0f, 3500.0f, 4200.0f, 5600.0f, 7000.0f, 8400.0f, 10500.0f, 14000.0f, 28000.0f };
					std::printf("%-22s ground by distance:", name.c_str());
					for (const float mark : marks) {
						double sumA = 0.0;
						int    nA = 0;
						for (UINT y = 0; y < kSize; ++y) {
							for (UINT x = kSize / 2 - 20; x < kSize / 2 + 20; ++x) {
								const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
								if (groundR[i] > 0.94f * mark && groundR[i] < 1.06f * mark) {
									sumA += first[i];
									++nA;
								}
							}
						}
						if (nA) std::printf(" %.0fm %.3f", mark / 70.0f, sumA / nA);
					}
					std::printf("\n");
				}
				if (kHill) {
					const HillStats hs = MeasureHill(first, triId, static_cast<int>(hill.size()));
					std::printf("%-22s facets %3d: mean fog %.4f..%.4f, spread %.4f; edge step %.4f, inner step %.4f, inner noise %.5f\n", name.c_str(), hs.facets,
						hs.facetMin, hs.facetMax, hs.facetSpread, hs.edgeStep, hs.innerStep, hs.innerNoise);
				}
			}
		}
	}

	SafeRelease(blend);
	SafeRelease(linear);
	SafeRelease(cb);
	SafeRelease(depthSrv);
	SafeRelease(depthTex);
	SafeRelease(sceneRtv);
	SafeRelease(sceneTex);
	SafeRelease(lowSrv);
	SafeRelease(lowRtv);
	SafeRelease(lowTex);
	SafeRelease(comp);
	SafeRelease(ps);
	SafeRelease(vs);
	SafeRelease(cSrv);
	SafeRelease(cTex);
	SafeRelease(hSrv);
	SafeRelease(hTex);
	sim.Release();
	SafeRelease(ctx);
	SafeRelease(dev);
	return exitCode;
}
