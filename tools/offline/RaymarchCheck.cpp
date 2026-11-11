// Offline check for the raymarched fog on WARP.
//   For several cameras looking at the same ground point (different pitch, distance and yaw) the GPU fog at the
//   screen centre must match a fine CPU integral of the same column texture along the same ray. That is the
//   "fog stays on the ground whatever the camera does" property, measured. Plus: a near scene surface cuts the fog.

#include "RaymarchShaders.h"
#include "WeatherRules.h"
#include "WorldspaceRules.h"
#include "RegionRules.h"
#include "FogConditions.h"
#include "PlacementRules.h"
#include "WindSim.h"

#include <DirectXMath.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// windows.h defines near / far as empty macros (see include/PCH.h); free the names for locals.
#undef near
#undef far

namespace
{
	int g_fails = 0;

	float HalfToFloat(std::uint16_t a_h)
	{
		const std::uint32_t sign = (a_h >> 15) & 1, ex = (a_h >> 10) & 0x1F, man = a_h & 0x3FF;
		float v = ex == 0 ? std::ldexp(static_cast<float>(man), -24) : std::ldexp(static_cast<float>(man | 0x400), static_cast<int>(ex) - 25);
		if (ex == 31) {
			v = man ? NAN : INFINITY;
		}
		return sign ? -v : v;
	}

	void Check(bool a_ok, const char* a_name, const std::string& a_detail)
	{
		std::printf("%s %s : %s\n", a_ok ? "PASS" : "FAIL", a_name, a_detail.c_str());
		if (!a_ok) {
			++g_fails;
		}
	}

	template <class T>
	void SafeRelease(T*& a_p)
	{
		if (a_p) {
			a_p->Release();
			a_p = nullptr;
		}
	}

	constexpr int   kRes = 64;
	constexpr float kExtent = 4096.0f;
	constexpr int   kHeightSize = 17;
	constexpr UINT  kSize = 96;
	constexpr int   kSteps = 24;   // the default RaymarchSteps
	constexpr float kOpacity = 1.4f;
	constexpr float kSteepness = 3.0f;   // the default ProfileSteepness (low layer)
	constexpr float kOrigin = -2048.0f;

	// Column profile for the CPU references, from the point density alone: per unit W, a column of height H holds
	// shape(h / H) / H at height h, shape(x) = exp(-a x) / N(a) for 0 <= x < 1 (0 outside), N(a) = integral_0^1 exp(-a s) ds
	// summed numerically here (fine midpoint sum). The references integrate this along their rays themselves, so they
	// never reuse the shader's closed-form span or ground factor.
	struct Profile
	{
		double a;
		double norm;   // 1 / N(a)
		explicit Profile(double a_a) :
			a(a_a), norm(0.0)
		{
			constexpr int kN = 200000;
			double        sum = 0.0;
			for (int i = 0; i < kN; ++i) sum += std::exp(-a * (i + 0.5) / kN);
			norm = kN / sum;
		}
		double Shape(double a_x) const { return a_x >= 0.0 && a_x < 1.0 ? std::exp(-a * a_x) * norm : 0.0; }
	};
	const Profile g_lowProfile{ kSteepness };
	const Profile g_upperProfile{ RaymarchShaders::kUpperSteepnessCpp };

	ID3DBlob* Compile(const char* a_body, const char* a_target)
	{
		std::string source = RaymarchShaders::kCommon;
		source += a_body;
		ID3DBlob* code = nullptr;
		ID3DBlob* errors = nullptr;
		const HRESULT hr = D3DCompile(source.data(), source.size(), "GroundFogRay", nullptr, nullptr, "main", a_target,
			D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
		Check(SUCCEEDED(hr), a_target, errors ? static_cast<const char*>(errors->GetBufferPointer()) : "compiled");
		SafeRelease(errors);
		return SUCCEEDED(hr) ? code : nullptr;
	}

	std::vector<float> ReadF32(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11Texture2D* a_tex, int a_w, int a_h, int a_channels)
	{
		D3D11_TEXTURE2D_DESC desc{};
		a_tex->GetDesc(&desc);
		desc.Usage = D3D11_USAGE_STAGING;
		desc.BindFlags = 0;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ID3D11Texture2D*   staging = nullptr;
		std::vector<float> out(static_cast<std::size_t>(a_w) * a_h * a_channels, NAN);
		if (FAILED(a_dev->CreateTexture2D(&desc, nullptr, &staging))) {
			return out;
		}
		a_ctx->CopyResource(staging, a_tex);
		D3D11_MAPPED_SUBRESOURCE m{};
		if (SUCCEEDED(a_ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m))) {
			for (int y = 0; y < a_h; ++y) {
				std::memcpy(&out[static_cast<std::size_t>(y) * a_w * a_channels], static_cast<const std::uint8_t*>(m.pData) + y * m.RowPitch,
					sizeof(float) * a_w * a_channels);
			}
			a_ctx->Unmap(staging, 0);
		}
		SafeRelease(staging);
		return out;
	}

	// Bilinear sample of the column texture at world xy, like the sampler (texel centres at (i + 0.5) * cell).
	void SampleColumn(const std::vector<float>& a_cols, float a_x, float a_y, float (&a_out)[4])
	{
		const float u = (a_x - kOrigin) / kExtent * kRes - 0.5f;
		const float v = (a_y - kOrigin) / kExtent * kRes - 0.5f;
		const int   x0 = std::clamp(static_cast<int>(std::floor(u)), 0, kRes - 1);
		const int   y0 = std::clamp(static_cast<int>(std::floor(v)), 0, kRes - 1);
		const int   x1 = std::min(x0 + 1, kRes - 1);
		const int   y1 = std::min(y0 + 1, kRes - 1);
		const float fx = std::clamp(u - std::floor(u), 0.0f, 1.0f);
		const float fy = std::clamp(v - std::floor(v), 0.0f, 1.0f);
		for (int c = 0; c < 4; ++c) {
			auto at = [&](int a_xx, int a_yy) { return a_cols[(static_cast<std::size_t>(a_yy) * kRes + a_xx) * 4 + c]; };
			a_out[c] = (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) + (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
		}
	}

	// Fine CPU integral of the same density along a ray (camera at a_cam, direction a_dir, up to a_maxDist).
	float CpuFog(const std::vector<float>& a_cols, DirectX::XMFLOAT3 a_cam, DirectX::XMFLOAT3 a_dir, float a_maxDist)
	{
		constexpr int n = 4000;
		const float   step = a_maxDist / n;
		double        tau = 0.0;
		for (int i = 0; i < n; ++i) {
			const float t = (i + 0.5f) * step;
			const float x = a_cam.x + a_dir.x * t, y = a_cam.y + a_dir.y * t, z = a_cam.z + a_dir.z * t;
			if (x < kOrigin || y < kOrigin || x > kOrigin + kExtent || y > kOrigin + kExtent) continue;
			float c[4];
			SampleColumn(a_cols, x, y, c);
			if (c[3] < 0.05f || c[0] < 1.0f) continue;
			const float h = z - c[2] / c[3];
			if (h < 0.0f || h >= c[0]) continue;
			const float rho = c[3] * c[1] / (c[0] / 70.0f) * static_cast<float>(g_lowProfile.Shape(h / c[0]));
			tau += kOpacity * rho * (step / 70.0f);
		}
		return 1.0f - static_cast<float>(std::exp(-tau));
	}
}

int RunRaymarchChecks()
{
	using namespace DirectX;
	ID3D11Device*        dev = nullptr;
	ID3D11DeviceContext* ctx = nullptr;
	if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
		Check(false, "raymarch WARP device", "failed");
		return g_fails;
	}

	// Columns from a real WindSim: flat ground at z = 0, fog at rest, base thickness 98.
	WindSim     sim;
	std::string error;
	if (!sim.Create(dev, kRes, error)) {
		Check(false, "raymarch columns", error);
		return g_fails;
	}
	sim.Reset(ctx);
	std::vector<float> heights(kHeightSize * kHeightSize, 0.0f);
	D3D11_TEXTURE2D_DESC hd{};
	hd.Width = hd.Height = kHeightSize;
	hd.MipLevels = hd.ArraySize = 1;
	hd.Format = DXGI_FORMAT_R32_FLOAT;
	hd.SampleDesc.Count = 1;
	hd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	D3D11_SUBRESOURCE_DATA hinit{ heights.data(), kHeightSize * sizeof(float), 0 };
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
	sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f);

	// Placement on the GPU: a grid with radiation x2 on the left half and hot-spring steam 0.5 on the right half;
	// with conditions (global 1, cold 0, wind 1) the column W must be 2x the plain build on the left and 1.5x on the
	// right (1 + 0.5 steam), away from the middle where the two halves blend.
	{
		std::vector<float> place(static_cast<std::size_t>(kHeightSize) * kHeightSize * 4, 0.0f);
		for (int y = 0; y < kHeightSize; ++y) {
			for (int x = 0; x < kHeightSize; ++x) {
				float* c = &place[(static_cast<std::size_t>(y) * kHeightSize + x) * 4];
				const bool left = x < kHeightSize / 2;
				c[0] = left ? 2.0f : 1.0f;
				c[1] = left ? 0.0f : 0.5f;
			}
		}
		D3D11_TEXTURE2D_DESC pd = hd;
		pd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		D3D11_SUBRESOURCE_DATA pinit{ place.data(), kHeightSize * sizeof(float) * 4, 0 };
		ID3D11Texture2D*          pTex = nullptr;
		ID3D11ShaderResourceView* pSrv = nullptr;
		dev->CreateTexture2D(&pd, &pinit, &pTex);
		dev->CreateShaderResourceView(pTex, nullptr, &pSrv);
		sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f);
		const auto plain = ReadF32(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
		sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, pSrv, { 1.0f, 0.0f, 1.0f, 0.0f });
		const auto placed = ReadF32(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
		auto ratioAt = [&](int a_x, int a_y) {
			const std::size_t i = (static_cast<std::size_t>(a_y) * kRes + a_x) * 4 + 1;
			return plain[i] > 1e-6f ? placed[i] / plain[i] : -1.0f;
		};
		const float l = ratioAt(kRes / 4, kRes / 2), r = ratioAt(kRes * 3 / 4, kRes / 2);
		Check(std::fabs(l - 2.0f) < 0.01f && std::fabs(r - 1.5f) < 0.01f, "placement grid applied in the columns",
			"W ratio left " + std::to_string(l) + " (expected 2), right " + std::to_string(r) + " (expected 1.5)");

		// Region value (LookP.y, RegionRules): added to the placement multiplier before the conditions. +0.5: left 2 + 0.5 =
		// 2.5x, right (1 + 0.5) + 0.5 steam = 2x; without a grid 1 + 0.5 = 1.5x; -3 holds the sum at 0: left 0, right 0.5 steam.
		{
			auto build = [&](ID3D11ShaderResourceView* a_place, float a_region) {
				sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, a_place, { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f },
					{ 0.0f, a_region, 0.0f, 0.0f });
				return ReadF32(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
			};
			auto ratio = [&](const std::vector<float>& a_v, int a_x, int a_y) {
				const std::size_t i = (static_cast<std::size_t>(a_y) * kRes + a_x) * 4 + 1;
				return plain[i] > 1e-6f ? a_v[i] / plain[i] : -1.0f;
			};
			const auto up = build(pSrv, 0.5f), upNoGrid = build(nullptr, 0.5f), down = build(pSrv, -3.0f), zero = build(pSrv, 0.0f);
			const float gl = ratio(up, kRes / 4, kRes / 2), gr = ratio(up, kRes * 3 / 4, kRes / 2), gn = ratio(upNoGrid, kRes / 2, kRes / 3);
			const float dl = ratio(down, kRes / 4, kRes / 2), dr = ratio(down, kRes * 3 / 4, kRes / 2);
			const float zl = ratio(zero, kRes / 4, kRes / 2), zr = ratio(zero, kRes * 3 / 4, kRes / 2);
			// Conditions G = 0.5, region +0.5, evaporation 0.25: left 0.5 x (2 + 0.5) - 0.25 = 1.0, right 0.5 x (1 + 0.5) - 0.25 + 0.5 x
			// 0.5 steam = 0.75 (steam thins with the wind factor PlaceP.z = 0.5). "G x placement + region" would give 1.25 on the left.
			// Without a grid and a region of -3: max(1 - 3, 0) = 0.
			sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, pSrv, { 0.5f, 0.0f, 0.5f, 0.0f }, { 0.25f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f },
				{ 0.0f, 0.5f, 0.0f, 0.0f });
			const auto mixed = ReadF32(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
			const auto downNoGrid = build(nullptr, -3.0f);
			const float ml = ratio(mixed, kRes / 4, kRes / 2), mr = ratio(mixed, kRes * 3 / 4, kRes / 2), dn = ratio(downNoGrid, kRes / 2, kRes / 3);
			Check(std::fabs(gl - 2.5f) < 0.01f && std::fabs(gr - 2.0f) < 0.01f && std::fabs(gn - 1.5f) < 0.01f && std::fabs(dl) < 0.01f &&
					  std::fabs(dr - 0.5f) < 0.01f && std::fabs(zl - 2.0f) < 0.01f && std::fabs(zr - 1.5f) < 0.01f && std::fabs(ml - 1.0f) < 0.01f &&
					  std::fabs(mr - 0.75f) < 0.01f && std::fabs(dn) < 0.01f,
				"region value is added to the placement multiplier",
				"+0.5: left " + std::to_string(gl) + " (2.5), right " + std::to_string(gr) + " (2), no grid " + std::to_string(gn) + " (1.5); -3: left " +
					std::to_string(dl) + " (0), right " + std::to_string(dr) + " (0.5); 0: " + std::to_string(zl) + ", " + std::to_string(zr) + " (2, 1.5); G 0.5 evap 0.25: " +
					std::to_string(ml) + ", " + std::to_string(mr) + " (1, 0.75); no grid -3: " + std::to_string(dn) + " (0)");
		}

		// Evaporation (SteamP.x 0.5, old steam path): left max(2 - 0.5, 0) = 1.5x, right (1 - 0.5) + 0.5 steam = 1x.
		sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, pSrv, { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.5f, 0.0f, 0.0f, 0.0f });
		const auto evap = ReadF32(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
		auto ratioOf = [&](const std::vector<float>& a_v, int a_x, int a_y) {
			const std::size_t i = (static_cast<std::size_t>(a_y) * kRes + a_x) * 4 + 1;
			return plain[i] > 1e-6f ? a_v[i] / plain[i] : -1.0f;
		};
		const float el = ratioOf(evap, kRes / 4, kRes / 2), er = ratioOf(evap, kRes * 3 / 4, kRes / 2);
		Check(std::fabs(el - 1.5f) < 0.01f && std::fabs(er - 1.0f) < 0.01f, "evaporation thins the weakest fog first",
			"W ratio left " + std::to_string(el) + " (expected 1.5), right " + std::to_string(er) + " (expected 1)");
		// Clamp at 0 (SteamP.x 1.5): left max(2 - 1.5, 0) = 0.5x, right max(1 - 1.5, 0) + 0.5 steam = 0.5x; and the path
		// without a placement grid (SteamP.x 0.4): max(1 - 0.4, 0) = 0.6x everywhere.
		sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, pSrv, { 1.0f, 0.0f, 1.0f, 0.0f }, { 1.5f, 0.0f, 0.0f, 0.0f });
		const auto evapHi = ReadF32(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
		sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, nullptr, { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.4f, 0.0f, 0.0f, 0.0f });
		const auto evapNoGrid = ReadF32(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
		const float cl = ratioOf(evapHi, kRes / 4, kRes / 2), cr = ratioOf(evapHi, kRes * 3 / 4, kRes / 2), cn = ratioOf(evapNoGrid, kRes / 2, kRes / 3);
		Check(std::fabs(cl - 0.5f) < 0.01f && std::fabs(cr - 0.5f) < 0.01f && std::fabs(cn - 0.6f) < 0.01f, "evaporation clamps at zero, also without a grid",
			"W ratio left " + std::to_string(cl) + ", right " + std::to_string(cr) + " (expected 0.5 both), no grid " + std::to_string(cn) + " (expected 0.6)");

		// Two layers. Steam (SteamP.y 400 units per unit of steam): the right half's steam 0.5 builds a tall layer
		// 100..200 units high; the low layer stays exactly the plain build everywhere (no stretching), and the tall
		// layer's W is steamH / 70 x 0.6 x lerp(0.2, 1, wisp) x densityShape x edge. At rest density 1, plain H = baseH
		// and plain W = baseH / 70 x densityShape x edge, so densityShape x edge = plain W / (plain H / 70); steamH gives
		// wisp. Pools: a second grid carries w = 150 (pool depth) on the left half; with SteamP.w 0.5 the tall layer
		// there is H 150, W 150 / 70 x 0.5 x densityShape x edge, and nothing on the right (no steam path, SteamP.y 0).
		sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, pSrv, { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 400.0f, 0.6f, 0.0f });
		const auto steamLow = ReadF32(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
		const auto steamUp = ReadF32(dev, ctx, sim.UpperTexture(), kRes, kRes, 2);
		float rightMinH = 1e9f, leftMaxUp = 0.0f, lowDiff = 0.0f, worstRel = 0.0f;
		int   exactCount = 0;
		const bool steamFinite = std::all_of(steamUp.begin(), steamUp.end(), [](float v) { return std::isfinite(v); }) &&
		                         std::all_of(steamLow.begin(), steamLow.end(), [](float v) { return std::isfinite(v); });
		for (int y = kRes / 4; y < kRes * 3 / 4; ++y) {
			for (int x = 4; x < kRes * 3 / 4; ++x) {
				const std::size_t i = (static_cast<std::size_t>(y) * kRes + x) * 4;
				const std::size_t u = (static_cast<std::size_t>(y) * kRes + x) * 2;
				const bool right = x > kRes / 2 + 4, left = x < kRes / 2 - 4;
				const float plainRadiation = left ? 2.0f : 1.0f;   // low layer: radiation only (steam went up)
				const float expectLow = plain[i + 1] * plainRadiation;
				if (left || right) lowDiff = std::max(lowDiff, std::fabs(steamLow[i] - plain[i]) + std::fabs(steamLow[i + 1] - expectLow));
				if (left) leftMaxUp = std::max(leftMaxUp, steamUp[u] + steamUp[u + 1]);
				if (!right) continue;
				rightMinH = std::min(rightMinH, steamUp[u]);
				const float hPlain = plain[i], wPlain = plain[i + 1], hUp = steamUp[u];
				if (hPlain > 1.0f && wPlain > 1e-4f) {
					const float wisp = std::clamp((hUp / 200.0f - 0.5f) / 0.5f, 0.0f, 1.0f);
					const float shapeEdge = wPlain / (hPlain / 70.0f);
					const float expect = hUp / 70.0f * 0.6f * (0.2f + 0.8f * wisp) * shapeEdge;
					worstRel = std::max(worstRel, std::fabs(steamUp[u + 1] - expect) / expect);
					++exactCount;
				}
			}
		}
		Check(steamFinite && rightMinH >= 99.0f && leftMaxUp < 1e-4f && lowDiff < 1e-3f && worstRel < 0.02f && exactCount > 100, "steam rises as its own layer",
			"right half lowest tall-layer H " + std::to_string(rightMinH) + " (expected > 98), left half tall layer " + std::to_string(leftMaxUp) +
				", low layer off the plain build by " + std::to_string(lowDiff) + ", worst relative W error " + std::to_string(worstRel) + " over " +
				std::to_string(exactCount) + " texels");

		std::vector<float> poolPlace = place;
		for (int y = 0; y < kHeightSize; ++y) {
			for (int x = 0; x < kHeightSize / 2; ++x) poolPlace[(static_cast<std::size_t>(y) * kHeightSize + x) * 4 + 3] = 150.0f;
		}
		D3D11_SUBRESOURCE_DATA    poolInit{ poolPlace.data(), kHeightSize * sizeof(float) * 4, 0 };
		ID3D11Texture2D*          poolTex = nullptr;
		ID3D11ShaderResourceView* poolSrv = nullptr;
		dev->CreateTexture2D(&pd, &poolInit, &poolTex);
		dev->CreateShaderResourceView(poolTex, nullptr, &poolSrv);
		sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, poolSrv, { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.5f });
		const auto poolUp = ReadF32(dev, ctx, sim.UpperTexture(), kRes, kRes, 2);
		float poolWorst = 0.0f, poolRightMax = 0.0f, poolHWorst = 0.0f;
		const bool poolFinite = std::all_of(poolUp.begin(), poolUp.end(), [](float v) { return std::isfinite(v); });
		for (int y = kRes / 4; y < kRes * 3 / 4; ++y) {
			for (int x = 4; x < kRes * 3 / 4; ++x) {
				const std::size_t i = (static_cast<std::size_t>(y) * kRes + x) * 4;
				const std::size_t u = (static_cast<std::size_t>(y) * kRes + x) * 2;
				if (x > kRes / 2 + 4) poolRightMax = std::max(poolRightMax, poolUp[u] + poolUp[u + 1]);
				if (x >= kRes / 2 - 4 || plain[i] <= 1.0f || plain[i + 1] <= 1e-4f) continue;
				const float shapeEdge = plain[i + 1] / (plain[i] / 70.0f);
				const float expect = 150.0f / 70.0f * 0.5f * shapeEdge;
				poolHWorst = std::max(poolHWorst, std::fabs(poolUp[u] - 150.0f));
				poolWorst = std::max(poolWorst, std::fabs(poolUp[u + 1] - expect) / expect);
			}
		}
		Check(poolFinite && poolHWorst < 0.01f && poolWorst < 0.01f && poolRightMax < 1e-4f, "pooled fog fills to its depth in the tall layer",
			"left half H off 150 by " + std::to_string(poolHWorst) + ", worst relative W error " + std::to_string(poolWorst) + ", right half " +
				std::to_string(poolRightMax));
		// Steam and pool in the same cells (left half: pool 150; right half: steam 0.5 at 400 units/steam, no pool): with
		// a grid carrying both everywhere, W must be steam W + pool W and H the larger of the two (GPT review 2026-11-08).
		{
			std::vector<float> both = poolPlace;
			for (int y = 0; y < kHeightSize; ++y)
				for (int x = 0; x < kHeightSize; ++x) {
					float* c = &both[(static_cast<std::size_t>(y) * kHeightSize + x) * 4];
					c[1] = 0.5f;    // hot-spring steam everywhere
					c[3] = 150.0f;  // pool everywhere
				}
			std::vector<float> steamOnly = both, poolOnly = both;
			for (std::size_t i = 0; i < steamOnly.size(); i += 4) {
				steamOnly[i + 3] = 0.0f;
				poolOnly[i + 1] = 0.0f;
			}
			auto build = [&](const std::vector<float>& a_grid, std::array<float, 4> a_steam) {
				D3D11_SUBRESOURCE_DATA    init{ a_grid.data(), kHeightSize * sizeof(float) * 4, 0 };
				ID3D11Texture2D*          tex = nullptr;
				ID3D11ShaderResourceView* srv = nullptr;
				dev->CreateTexture2D(&pd, &init, &tex);
				dev->CreateShaderResourceView(tex, nullptr, &srv);
				sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, srv, { 1.0f, 0.0f, 1.0f, 0.0f }, a_steam);
				auto r = ReadF32(dev, ctx, sim.UpperTexture(), kRes, kRes, 2);
				SafeRelease(srv);
				SafeRelease(tex);
				return r;
			};
			const auto rs = build(steamOnly, { 0.0f, 400.0f, 0.6f, 0.0f });
			const auto rp = build(poolOnly, { 0.0f, 400.0f, 0.6f, 0.5f });
			const auto rb = build(both, { 0.0f, 400.0f, 0.6f, 0.5f });
			float worstW = 0.0f, worstH = 0.0f;
			bool  finite = true;
			for (int y = kRes / 4; y < kRes * 3 / 4; ++y)
				for (int x = kRes / 4; x < kRes * 3 / 4; ++x) {
					const std::size_t u = (static_cast<std::size_t>(y) * kRes + x) * 2;
					finite = finite && std::isfinite(rb[u]) && std::isfinite(rb[u + 1]) && std::isfinite(rs[u]) && std::isfinite(rs[u + 1]) &&
					         std::isfinite(rp[u]) && std::isfinite(rp[u + 1]);
					worstW = std::max(worstW, std::fabs(rb[u + 1] - (rs[u + 1] + rp[u + 1])));
					worstH = std::max(worstH, std::fabs(rb[u] - std::max(rs[u], rp[u])));
				}
			Check(finite && worstW < 1e-4f && worstH < 1e-3f, "steam and pool add in the tall layer",
				"worst |W both - (W steam + W pool)| " + std::to_string(worstW) + ", worst |H both - max| " + std::to_string(worstH));
		}
		// The placement texture's own grid (PlaceG): with its origin half a field to the east, the world's right half
		// reads the texture's left half (radiation 2x); past the texture's footprint steam and pools are dropped.
		{
			const float shift = kExtent / 2.0f;
			sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, pSrv, { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f },
				{ kOrigin + shift, kOrigin, kExtent / (kHeightSize - 1), static_cast<float>(kHeightSize) });
			const auto own = ReadF32(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
			const float rightRatio = ratioOf(own, kRes * 3 / 4 - 2, kRes / 2);   // world 3/4 -> texture 1/4 (left half: 2x)
			// Footprint: the pool grid shifted the same way; world x < shift is outside it (g < 0), so no pool there.
			sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, poolSrv, { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.5f },
				{ kOrigin + shift, kOrigin, kExtent / (kHeightSize - 1), static_cast<float>(kHeightSize) });
			const auto ownUp = ReadF32(dev, ctx, sim.UpperTexture(), kRes, kRes, 2);
			const float outside = ownUp[(static_cast<std::size_t>(kRes / 2) * kRes + kRes / 4) * 2 + 1];
			const float inside = ownUp[(static_cast<std::size_t>(kRes / 2) * kRes + kRes * 3 / 4 - 2) * 2 + 1];
			Check(std::fabs(rightRatio - 2.0f) < 0.01f && outside == 0.0f && inside > 0.0f, "placement texture keeps its own grid",
				"right half W ratio " + std::to_string(rightRatio) + " (expected 2), pool W outside the footprint " + std::to_string(outside) +
					", inside " + std::to_string(inside));
		}
		// Wisps (the field's equilibrium pattern, amount 0.5): W varies around the plain build (mean within 10%,
		// spread > 5%), H unchanged; 0 = plain. The pattern is part of the wind field's fog amount, so step the field.
		{
			auto wispCb = scb;
			wispCb.counts[3] = 0.5f;
			sim.Step(ctx, wispCb, hSrv, 5);
			sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, nullptr, { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f });
			const auto wisp = ReadF32(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
			sim.Reset(ctx);
			sim.Step(ctx, scb, hSrv, 5);   // back to the uniform field for the checks below
			double sum = 0.0, sum2 = 0.0;
			int    cnt = 0;
			float  hDiff = 0.0f, lo = 1e9f, hi = -1e9f;
			bool   finite = true;
			for (int y = kRes / 4; y < kRes * 3 / 4; ++y)
				for (int x = kRes / 4; x < kRes * 3 / 4; ++x) {
					const std::size_t i = (static_cast<std::size_t>(y) * kRes + x) * 4;
					finite = finite && std::isfinite(wisp[i]) && std::isfinite(wisp[i + 1]);
					hDiff = std::max(hDiff, std::fabs(wisp[i] - plain[i]));
					if (plain[i + 1] <= 1e-4f) continue;
					const double r = wisp[i + 1] / plain[i + 1];
					sum += r;
					sum2 += r * r;
					lo = std::min(lo, static_cast<float>(r));
					hi = std::max(hi, static_cast<float>(r));
					++cnt;
				}
			const double mean = cnt ? sum / cnt : 0.0, sd = cnt ? std::sqrt(std::max(0.0, sum2 / cnt - mean * mean)) : 0.0;
			Check(finite && cnt > 100 && std::fabs(mean - 1.0) < 0.1 && sd > 0.05 && lo >= 0.5f - 1e-3f && hi <= 1.5f + 1e-3f && hDiff < 1e-3f,
				"wisps vary the low layer around its mean", "W ratio mean " + std::to_string(mean) + ", spread " + std::to_string(sd) + ", range " +
					std::to_string(lo) + ".." + std::to_string(hi) + ", H change " + std::to_string(hDiff));
		}
		// Coarse pools (InCoarse): a coarse grid with ground 0 and depth 150 everywhere over the flat fine ground (0) gives a
		// 150-unit pool in the tall layer; with the coarse ground 400 units off (tolerance 250) it fades to nothing.
		{
			const int cs = 9;
			auto coarseRun = [&](float a_ground, bool a_unknownCentre = false) {
				std::vector<float> rg(static_cast<std::size_t>(cs) * cs * 2);
				for (std::size_t i = 0; i < rg.size(); i += 2) {
					rg[i] = a_ground;
					rg[i + 1] = 150.0f;
				}
				if (a_unknownCentre) rg[(static_cast<std::size_t>(cs / 2) * cs + cs / 2) * 2] = -1.0e30f;   // one unknown coarse cell
				D3D11_TEXTURE2D_DESC cd{};
				cd.Width = cd.Height = cs;
				cd.MipLevels = cd.ArraySize = 1;
				cd.Format = DXGI_FORMAT_R32G32_FLOAT;
				cd.SampleDesc.Count = 1;
				cd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
				D3D11_SUBRESOURCE_DATA    ci{ rg.data(), cs * sizeof(float) * 2, 0 };
				ID3D11Texture2D*          ct = nullptr;
				ID3D11ShaderResourceView* cv = nullptr;
				dev->CreateTexture2D(&cd, &ci, &ct);
				dev->CreateShaderResourceView(ct, nullptr, &cv);
				const float cell = kExtent / (cs - 1);
				std::vector<float> noPool = place;   // w = 0: the fine grid has no pool
				D3D11_SUBRESOURCE_DATA    ni{ noPool.data(), kHeightSize * sizeof(float) * 4, 0 };
				ID3D11Texture2D*          nt = nullptr;
				ID3D11ShaderResourceView* nv = nullptr;
				dev->CreateTexture2D(&pd, &ni, &nt);
				dev->CreateShaderResourceView(nt, nullptr, &nv);
				sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f, nv, { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.5f },
					{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, cv, { kOrigin, kOrigin, cell, static_cast<float>(cs) }, { 250.0f, 560.0f, 0.0f, 0.0f });
				auto r = ReadF32(dev, ctx, sim.UpperTexture(), kRes, kRes, 2);
				SafeRelease(nv);
				SafeRelease(nt);
				SafeRelease(cv);
				SafeRelease(ct);
				return r;
			};
			const auto onFlat = coarseRun(0.0f);
			const auto offCliff = coarseRun(400.0f);
			const auto holed = coarseRun(0.0f, true);
			// The unknown cell sits at the field centre (coarse cell 4 of 0..8): the texels around it mix it in and get no
			// pool; a quarter of the way in from the corner they keep 150.
			const float atHole = holed[(static_cast<std::size_t>(kRes / 2) * kRes + kRes / 2) * 2];
			const float awayHole = holed[(static_cast<std::size_t>(kRes / 8) * kRes + kRes / 8 + 2) * 2];
			float hWorst = 0.0f, offMax = 0.0f;
			bool  finite = true;
			for (int y = kRes / 4; y < kRes * 3 / 4; ++y)
				for (int x = kRes / 4; x < kRes * 3 / 4; ++x) {
					const std::size_t u = (static_cast<std::size_t>(y) * kRes + x) * 2;
					finite = finite && std::isfinite(onFlat[u]) && std::isfinite(onFlat[u + 1]) && std::isfinite(offCliff[u]) && std::isfinite(offCliff[u + 1]);
					hWorst = std::max(hWorst, std::fabs(onFlat[u] - 150.0f));
					offMax = std::max(offMax, offCliff[u] + offCliff[u + 1]);
				}
			const bool holeOk = std::isfinite(atHole) && std::isfinite(awayHole) && atHole == 0.0f && std::fabs(awayHole - 150.0f) < 0.01f;
			Check(finite && hWorst < 0.01f && offMax < 1e-4f && holeOk, "coarse pools fill wide valleys, not across cliffs or unknown cells",
				"pool H off 150 by " + std::to_string(hWorst) + " over flat ground; with the coarse ground 400 off: " + std::to_string(offMax) +
					"; next to an unknown coarse cell " + std::to_string(atHole) + ", away from it " + std::to_string(awayHole));
		}
		SafeRelease(poolSrv);
		SafeRelease(poolTex);
		// Back to the plain build for the checks that follow.
		sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f);
		SafeRelease(pSrv);
		SafeRelease(pTex);
	}
	const auto cols = ReadF32(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
	float      maxH = 0.0f;
	for (int i = 0; i < kRes * kRes; ++i) maxH = std::max(maxH, cols[i * 4]);
	Check(maxH > 20.0f && maxH < 200.0f, "columns built", "highest column " + std::to_string(maxH) + " units");
	// Round edge fade (on the density W since 2026-11-07): corner and side middles (r >= 0.95) carry none, r = 0.4 keeps it.
	{
		auto hAt = [&](int a_x, int a_y) { return cols[(static_cast<std::size_t>(a_y) * kRes + a_x) * 4 + 1]; };
		const float corner = hAt(0, 0), side = hAt(kRes / 2, 0);
		// Outside the circle but inside a square fade's full zone: tells round from square (Codex 2026-11-07).
		const float diagonal = hAt(kRes * 54 / 64, kRes * 54 / 64);
		float       inner = 1e9f;
		for (int a = 0; a < 16; ++a) {
			const float ang = a * 0.3927f;
			const int   x = static_cast<int>(kRes / 2 + std::cos(ang) * 0.4f * kRes / 2), y = static_cast<int>(kRes / 2 + std::sin(ang) * 0.4f * kRes / 2);
			inner = std::min(inner, hAt(x, y));
		}
		Check(corner == 0.0f && side < 0.007f && diagonal < 0.007f && inner > 0.02f, "round edge fade",
			"W corner " + std::to_string(corner) + ", side middle " + std::to_string(side) + ", diagonal " + std::to_string(diagonal) + ", lowest at r 0.4 " + std::to_string(inner));
	}


	// Weather rules (WeatherRules.h, the same defaults as WeatherFog.cpp): real EditorIDs from Azurite Weathers III
	// (vanilla names) and NAT.ENB III (its own names, read from NAT-ENB.esp 3.1.1C on 2026-11-08).
	{
		int  bad = 0;
		auto rules = WeatherRules::ParseRules(WeatherRules::kDefaultRules, &bad);
		WeatherRules::Params p;
		p.rules = rules;
		p.overrides = { { "skyrimfogre", 3.0f }, { "nat-enb.esp|53b6", 0.25f } };
		struct Case
		{
			const char* id;
			float       expect;
			const char* how;
		};
		const Case cases[] = {
			{ "SkyrimFogFF", 2.5f, "word fog" }, { "FogRain", 2.5f, "word fog" }, { "SClearSkyTU", 0.45f, "word clear" },
			{ "OvercastMA", 0.9f, "word overcast" }, { "Snowstorm_Alt", 1.4f, "word snowstorm" }, { "SnowClear", 0.9f, "word snow" },
			{ "RainCloudy_DLC2_ALT", 1.2f, "word rain" }, { "SunnyPF", 0.45f, "word sunny" }, { "SkyrimStormRainTU", 1.3f, "word storm" },
			{ "SkyrimFogRE", 3.0f, "override" },
			// Obsidian Weathers 1.07a (Nexus 12125, read 2026-11-08)
			{ "DLC02VolcanicAshTundra01_A", 1.5f, "word volcanicash" },
		};
		int wrong = 0;
		std::string firstWrong;
		for (const auto& c : cases) {
			WeatherRules::Weather w;
			w.editorID = c.id;
			w.plugin = "Skyrim.esm";
			const auto r = WeatherRules::FactorFor(w, p);
			if (std::fabs(r.factor - c.expect) > 1e-5f || r.how != c.how) {
				if (wrong++ == 0) firstWrong = std::string(c.id) + " -> " + std::to_string(r.factor) + " (" + r.how + ")";
			}
		}
		WeatherRules::Weather fallback;   // no word: cloudy flag, fog ending at the reference distance
		fallback.editorID = "BlackreachWeather";
		fallback.cloudy = true;
		fallback.dayFar = 100000.0f;
		const auto rf = WeatherRules::FactorFor(fallback, p);
		WeatherRules::Weather noId;   // no EditorID: pleasant flag, fog at a quarter of the reference -> distance term 2
		noId.pleasant = true;
		noId.dayFar = 25000.0f;
		noId.plugin = "NAT-ENB.esp";
		noId.localFormID = 0x1234;
		const auto rn = WeatherRules::FactorFor(noId, p);
		WeatherRules::Weather byForm;   // override by plugin|formid
		byForm.plugin = "NAT-ENB.esp";
		byForm.localFormID = 0x050053B6;   // the load-order byte (top 8 bits) is ignored
		byForm.editorID = "OvercastPF";
		const auto rb = WeatherRules::FactorFor(byForm, p);
		const bool keysOk = WeatherRules::NormalizeOverrideKey("NAT-ENB.esp|0053B6") == WeatherRules::FormKey("NAT-ENB.esp", 0x53B6) &&
		                    WeatherRules::NormalizeOverrideKey(" SkyrimFogRE ") == "skyrimfogre" &&
		                    WeatherRules::NormalizeOverrideKey("x.esp|0x00ABCD") == WeatherRules::FormKey("x.esp", 0xABCD) &&
		                    WeatherRules::NormalizeOverrideKey("x.esp|zz") != WeatherRules::FormKey("x.esp", 0);
		int badCount = 0;
		const auto badRules = WeatherRules::ParseRules("fog:2, bad, x:y, :3", &badCount);
		Check(bad == 0 && wrong == 0 && std::fabs(rf.factor - 0.9f) < 1e-5f && rf.how == "flags" && std::fabs(rn.factor - 1.0f) < 1e-5f &&
				  rn.how == "flags (no EditorID)" && std::fabs(rb.factor - 0.25f) < 1e-5f && rb.how == "override" && badRules.size() == 1 && badCount == 3 && keysOk,
			"weather rules (Azurite and NAT names)",
			std::to_string(std::size(cases)) + " names, " + std::to_string(wrong) + " wrong" + (wrong ? " (first: " + firstWrong + ")" : std::string()) +
				", fallback " + std::to_string(rf.factor) + ", no EditorID " + std::to_string(rn.factor) + ", form override " + std::to_string(rb.factor) +
				", bad entries " + std::to_string(badCount) + ", override keys " + (keysOk ? "ok" : "WRONG"));
	}


	// Worldspace rules (WorldspaceRules.h): the value forms, the keys, the parent chain and the built-in table.
	{
		namespace WR = WorldspaceRules;
		auto parses = [](const char* a_text, WR::Kind a_kind, float a_value) {
			WR::Rule r;
			return WR::ParseRule(a_text, r) && r.kind == a_kind && std::fabs(r.value - a_value) < 1e-6f;
		};
		auto rejects = [](const char* a_text) {
			WR::Rule r{ WR::Kind::kScale, 7.0f };
			return !WR::ParseRule(a_text, r) && r.kind == WR::Kind::kScale && r.value == 7.0f;   // untouched on failure
		};
		// normal
		const bool normal = parses("off", WR::Kind::kOff, 0.0f) && parses("0.5", WR::Kind::kScale, 0.5f) && parses("always 1.5", WR::Kind::kAlways, 1.5f) &&
		                    parses("default", WR::Kind::kDefault, 1.0f) && parses("  ALWAYS	2 ", WR::Kind::kAlways, 2.0f) && parses("Off", WR::Kind::kOff, 0.0f);
		// boundary: zero, the ceiling, just over it
		const bool boundary = parses("0", WR::Kind::kScale, 0.0f) && parses("always 0", WR::Kind::kAlways, 0.0f) && parses("4", WR::Kind::kScale, 4.0f) &&
		                      parses("9", WR::Kind::kScale, WR::kMaxValue) && parses("always 100", WR::Kind::kAlways, WR::kMaxValue) &&
		                      parses(".5", WR::Kind::kScale, 0.5f) && parses("1e1", WR::Kind::kScale, WR::kMaxValue) && parses("always  2", WR::Kind::kAlways, 2.0f);
		// failure
		const bool failure = rejects("") && rejects("   ") && rejects("always") && rejects("always1") && rejects("-1") && rejects("always -0.5") &&
		                     rejects("on") && rejects("1.0x") && rejects("nan") && rejects("inf") && rejects("always off") && rejects("+1") &&
		                     rejects("0x10") && rejects("always nan") && rejects("1e39");

		const std::vector<WR::Entry> ini{
			{ WeatherRules::NormalizeOverrideKey("Tamriel"), { WR::Kind::kScale, 0.5f } },
			{ WeatherRules::NormalizeOverrideKey("MyMod.esp|000800"), { WR::Kind::kOff, 0.0f } },
			{ WeatherRules::NormalizeOverrideKey("DLC01SoulCairn"), { WR::Kind::kDefault, 1.0f } },
		};
		const auto builtIn = WR::BuiltIn();
		const WR::Space tamriel{ "Tamriel", "Skyrim.esm", 0x3C };
		const WR::Space whiterun{ "WhiterunWorld", "Skyrim.esm", 0x1A26F };
		const WR::Space blackreach{ "Blackreach", "Skyrim.esm", 0x1EE62 };
		const WR::Space soulCairn{ "DLC01SoulCairn", "Dawnguard.esm", 0x02001408 };   // load-order byte ignored
		const WR::Space byForm{ "", "mymod.ESP", 0x800 };
		const WR::Space other{ "SomeCaveWorld", "Skyrim.esm", 0x1E49D };
		std::string how;
		auto resolve = [&](std::initializer_list<WR::Space> a_chain, const std::vector<WR::Entry>& a_ini) {
			const std::vector<WR::Space> chain(a_chain);
			return WR::Resolve(chain, a_ini, builtIn, &how);
		};
		bool chainOk = true;
		chainOk &= resolve({ tamriel }, ini) == WR::Rule{ WR::Kind::kScale, 0.5f } && how == "ini (Tamriel)";
		chainOk &= resolve({ whiterun, tamriel }, ini) == WR::Rule{ WR::Kind::kScale, 0.5f } && how == "ini (parent Tamriel)";   // the parent's entry
		chainOk &= resolve({ blackreach }, ini) == WR::Rule{ WR::Kind::kAlways, 1.0f } && how == "built-in (Blackreach)";
		chainOk &= resolve({ soulCairn }, {}) == WR::Rule{ WR::Kind::kAlways, 1.5f } && how == "built-in (DLC01SoulCairn)";
		chainOk &= resolve({ soulCairn }, ini) == WR::Rule{} && how == "ini (DLC01SoulCairn)";   // ini "default" cancels the built-in
		chainOk &= resolve({ byForm }, ini) == WR::Rule{ WR::Kind::kOff, 0.0f } && how == "ini (mymod.esp|800)";   // plugin|formid, case-insensitive
		chainOk &= resolve({ other }, ini) == WR::Rule{} && how == "none";
		chainOk &= resolve({ blackreach, tamriel }, ini) == WR::Rule{ WR::Kind::kAlways, 1.0f };   // the nearest worldspace decides, before a parent's ini entry
		chainOk &= resolve({}, ini) == WR::Rule{} && how == "none";
		chainOk &= how.empty() == false && resolve({ blackreach, tamriel }, ini) == WR::Rule{ WR::Kind::kAlways, 1.0f } && how == "built-in (Blackreach)";
		// plugin|formid with no EditorID; the load-order byte of the runtime id is ignored
		const std::vector<WR::Entry> byKey{ { WeatherRules::NormalizeOverrideKey("Dawnguard.esm|001408"), { WR::Kind::kScale, 2.0f } } };
		chainOk &= resolve({ WR::Space{ "", "Dawnguard.esm", 0x02001408 } }, byKey) == WR::Rule{ WR::Kind::kScale, 2.0f } && how == "ini (dawnguard.esm|1408)";
		// a child's ini "default" decides before its parent's "off"
		const std::vector<WR::Entry> childDefault{ { "whiterunworld", { WR::Kind::kDefault, 1.0f } }, { "tamriel", { WR::Kind::kOff, 0.0f } } };
		chainOk &= resolve({ whiterun, tamriel }, childDefault) == WR::Rule{} && how == "ini (WhiterunWorld)";
		chainOk &= resolve({ tamriel }, childDefault) == WR::Rule{ WR::Kind::kOff, 0.0f };
		Check(normal && boundary && failure && chainOk, "worldspace rules (values, keys, parents, built-in)",
			std::string("values normal ") + (normal ? "ok" : "WRONG") + ", boundary " + (boundary ? "ok" : "WRONG") + ", rejected " + (failure ? "ok" : "WRONG") +
				", chain " + (chainOk ? "ok" : "WRONG") + ", built-in entries " + std::to_string(builtIn.size()));
	}


	// Region rules (RegionRules.h): the value forms, the lookup order and the built-in table.
	{
		namespace RR = RegionRules;
		auto parses = [](const char* a_text, float a_value) {
			float v = 99.0f;
			return RR::ParseAdd(a_text, v) && std::fabs(v - a_value) < 1e-6f;
		};
		auto rejects = [](const char* a_text) {
			float v = 99.0f;
			return !RR::ParseAdd(a_text, v) && v == 99.0f;   // untouched on failure
		};
		const bool normal = parses("0.5", 0.5f) && parses("+0.3", 0.3f) && parses("-0.2", -0.2f) && parses(" default ", 0.0f) && parses("0", 0.0f);
		float cv = 0.0f;
		bool  cl = false, cl2 = true;
		const bool boundary = parses("1", 1.0f) && parses("-1", -1.0f) && parses("5", RR::kMaxAdd) && parses("-7", RR::kMinAdd) && parses("+.5", 0.5f) &&
		                      RR::ParseAdd("1.5", cv, &cl) && cl && RR::ParseAdd("1", cv, &cl2) && !cl2 && RR::ParseAdd("-0", cv) && !std::signbit(cv);
		const bool failure = rejects("") && rejects("+") && rejects("++1") && rejects("+-1") && rejects("always 1") && rejects("off") && rejects("nan") &&
		                     rejects("0.5x") && rejects("1e39");
		const auto builtIn = RR::BuiltIn();
		const std::vector<RR::Entry> ini{
			RR::IniEntry("WeatherPineForest", 0.9f),   // as FogRenderer builds them from [Regions]
			RR::IniEntry("Skyrim.esm|0C5858", -0.3f),
			RR::IniEntry("WeatherReach", 0.0f),
		};
		std::string how;
		bool lookup = true;
		// built-in by form id, with no EditorID at runtime (no EditorID mod); the load-order byte is ignored
		lookup &= std::fabs(RR::AddFor({ "", "Skyrim.esm", 0x0002A72D }, {}, builtIn, &how) - 0.25f) < 1e-6f && how == "built-in";
		lookup &= std::fabs(RR::AddFor({ "", "skyrim.ESM", 0x000C5856 }, {}, builtIn, &how) - 0.3f) < 1e-6f && how == "built-in";
		// the marsh is left to the water-and-terrain rules near the player and keeps its value for the far fog
		float farValue = 99.0f;
		lookup &= RR::AddFor({ "", "Skyrim.esm", 0x0C5855 }, {}, builtIn, &how, &farValue) == 0.0f && how == "built-in" && std::fabs(farValue - 0.5f) < 1e-6f;
		lookup &= std::fabs(RR::AddFor({ "", "Skyrim.esm", 0x0C5856 }, {}, builtIn, &how, &farValue) - 0.3f) < 1e-6f && std::fabs(farValue - 0.4f) < 1e-6f;
		// an ini line sets both; a region in neither table gives 0 for both
		lookup &= std::fabs(RR::AddFor({ "WeatherPineForest", "Skyrim.esm", 0x02A72D }, ini, builtIn, &how, &farValue) - 0.9f) < 1e-6f && std::fabs(farValue - 0.9f) < 1e-6f;
		farValue = 99.0f;
		lookup &= RR::AddFor({ "WeatherTundra", "Skyrim.esm", 0x0C5858 }, {}, builtIn, &how, &farValue) == 0.0f && farValue == 0.0f;
		// the ini by EditorID beats the built-in entry; by form id it reaches a region the table does not name
		lookup &= std::fabs(RR::AddFor({ "WeatherPineForest", "Skyrim.esm", 0x02A72D }, ini, builtIn, &how) - 0.9f) < 1e-6f && how == "ini";
		lookup &= std::fabs(RR::AddFor({ "", "Skyrim.esm", 0x0C5858 }, ini, builtIn, &how) + 0.3f) < 1e-6f && how == "ini";
		// an ini EditorID line does nothing when the game has no EditorID for the region: the built-in value stays
		lookup &= std::fabs(RR::AddFor({ "", "Skyrim.esm", 0x02A72D }, ini, builtIn, &how) - 0.25f) < 1e-6f && how == "built-in";
		// ini 0 cancels a built-in entry; a region in neither table is 0
		lookup &= RR::AddFor({ "WeatherReach", "Skyrim.esm", 0x0C5856 }, ini, builtIn, &how) == 0.0f && how == "ini";
		lookup &= RR::AddFor({ "WeatherTundra", "Skyrim.esm", 0x0C5858 }, {}, builtIn, &how) == 0.0f && how == "none";
		lookup &= RR::AddFor({ "WeatherSnow", "Other.esp", 0x0C5856 }, {}, builtIn, &how) == 0.0f;   // same id in another plugin
		// every entry as LORE-FOG.md has it (form id, near the player, far fog): a slip in one number fails here
		struct Expected { std::uint32_t id; float add; float farAdd; };
		const Expected expected[] = {
			{ 0x0C5855, 0.0f, 0.5f },
			{ 0x1046C8, 0.0f, 0.5f },
			{ 0x0C5856, 0.3f, 0.4f },
			{ 0x07042F, 0.3f, 0.4f },
			{ 0x02A72D, 0.25f, 0.3f },
			{ 0x10474D, 0.25f, 0.3f },
			{ 0x0C5852, 0.0f, 0.3f },
			{ 0x10FE79, 0.0f, 0.3f },
			{ 0x0C5854, 0.0f, 0.2f },
			{ 0x1046C7, 0.0f, 0.2f },
			{ 0x0C5857, 0.15f, 0.2f },
			{ 0x1046CC, 0.15f, 0.2f },
			{ 0x07042E, 0.15f, 0.2f },
			{ 0x10FDC5, 0.15f, 0.2f },
		};
		bool table = builtIn.size() == std::size(expected);
		for (const auto& x : expected) {
			float xf = 99.0f;
			table &= std::fabs(RR::AddFor({ "", "Skyrim.esm", x.id }, {}, builtIn, &how, &xf) - x.add) < 1e-6f && std::fabs(xf - x.farAdd) < 1e-6f && how == "built-in";
		}
		for (std::size_t i = 0; i < builtIn.size(); ++i) {
			const auto& e = builtIn[i];
			// near the player never more than far away (the overlap is only taken out), and far within the evidence
			table &= e.add >= 0.0f && e.add <= e.farAdd && e.farAdd > 0.0f && e.farAdd <= 0.5f && e.key.rfind("skyrim.esm|", 0) == 0;
			for (std::size_t j = i + 1; j < builtIn.size(); ++j) {
				table &= builtIn[j].key != e.key;   // no region listed twice
			}
		}
		const bool combined = std::fabs(RR::Combined(0.7f, 0.5f) - 1.2f) < 1e-6f && RR::Combined(0.7f, -1.0f) == 0.0f && RR::Combined(2.2f, 0.0f) == 2.2f;
		Check(normal && boundary && failure && lookup && table && combined, "region rules (values, lookup, built-in)",
			std::string("values normal ") + (normal ? "ok" : "WRONG") + ", boundary " + (boundary ? "ok" : "WRONG") + ", rejected " + (failure ? "ok" : "WRONG") +
				", lookup " + (lookup ? "ok" : "WRONG") + ", table " + (table ? "ok" : "WRONG") + " (" + std::to_string(builtIn.size()) + " entries), sum " +
				(combined ? "ok" : "WRONG"));
	}


	// Realistic conditions (FogConditions.h) with the default parameters.
	{
		FogConditions::Params p;
		auto run = [&](float a_hour, int a_month, float a_wind, bool a_clear, bool a_cloudy, bool a_snow, float a_survival) {
			FogConditions::Sky sk;
			sk.hour = a_hour;
			sk.month = a_month;
			sk.windSpeed = a_wind;
			sk.pleasant = a_clear;
			sk.cloudy = a_cloudy;
			sk.snow = a_snow;
			sk.survivalCold = a_survival;
			return FogConditions::Compute(sk, p);
		};
		auto same = [](float a, float b) { return std::fabs(a - b) < 1e-4f; };
		const auto dawnClear = run(6.5f, 0, 0.0f, true, false, false, -1.0f);    // Morning Star, calm, clear dawn
		const auto noonClear = run(13.0f, 6, 0.0f, true, false, false, -1.0f);   // Sun's Height, noon
		const auto dawnCloudy = run(6.5f, 3, 0.0f, false, true, false, -1.0f);   // clouds keep the heat in
		const auto dawnWindy = run(6.5f, 3, 1.0f, true, false, false, -1.0f);    // strong wind
		const auto survival = run(13.0f, 6, 0.0f, true, false, false, 0.2f);    // air 0.2 over summer water 0.05
		const auto snowNoon = run(13.0f, 6, 0.0f, false, false, true, -1.0f);
		const auto springCold = run(13.0f, 3, 0.0f, true, false, false, 0.4f); // air 0.4 over spring water 0.45: no steam
		const auto before7 = run(6.99f, 6, 0.0f, true, false, false, -1.0f);
		const auto after7 = run(7.01f, 6, 0.0f, true, false, false, -1.0f);
		const auto before20 = run(19.99f, 6, 0.0f, true, false, false, -1.0f);
		const auto after20 = run(20.01f, 6, 0.0f, true, false, false, -1.0f);
		const bool smooth = std::fabs(before7.cold - after7.cold) < 0.02f && std::fabs(before20.cold - after20.cold) < 0.02f && before7.cold > 0.1f;
		const bool ok = same(dawnClear.time, 1.6f) && same(dawnClear.global, 1.6f * 1.25f) && same(noonClear.time, 0.35f) &&
						same(noonClear.global, 0.35f * 0.75f) && same(dawnCloudy.time, 1.3f) && same(dawnWindy.wind, 0.3f) &&
						same(dawnWindy.global, 1.6f * 0.3f) && same(survival.air, 0.2f) && same(survival.cold, 0.375f) && same(snowNoon.cold, 1.0f) &&
						same(noonClear.cold, 0.0f) && same(dawnClear.air, 0.6f) && same(dawnClear.cold, 0.625f) && same(springCold.cold, 0.0f) && smooth &&
						same(noonClear.evaporation, 0.2f * (1.0f - 0.35f * 0.75f)) && same(dawnClear.evaporation, 0.0f);
		Check(ok, "fog conditions (time, sky, wind, season, cold)",
			"dawn clear " + std::to_string(dawnClear.global) + ", noon " + std::to_string(noonClear.global) + ", dawn cloudy time " +
				std::to_string(dawnCloudy.time) + ", windy " + std::to_string(dawnWindy.global) + ", survival cold " + std::to_string(survival.cold) +
				", snow cold " + std::to_string(snowNoon.cold) + ", spring steam " + std::to_string(springCold.cold) + ", steam 06:59/07:01 " +
				std::to_string(before7.cold) + "/" + std::to_string(after7.cold) + ", 19:59/20:01 " + std::to_string(before20.cold) + "/" + std::to_string(after20.cold));
	}

	// Placement (PlacementRules.h): water kinds from real water type EditorIDs, shore band, valley, steam channels.
	{
		using PlacementRules::Kind;
		const float sea = -14000.0f;
		const bool kindsOk = PlacementRules::WaterKind("DefaultMarshWater", 0, sea) == Kind::kMarshWater &&
							 PlacementRules::WaterKind("MorthalWater", 0, sea) == Kind::kMarshWater &&
							 PlacementRules::WaterKind("DefaultVolcanicWater", 0, sea) == Kind::kHotSpring &&
							 PlacementRules::WaterKind("RiverWaterFlowNE", 0, sea) == Kind::kRiver &&
							 PlacementRules::WaterKind("HonrichWater", 0, sea) == Kind::kLake &&
							 PlacementRules::WaterKind("DefaultWater", -14000.0f, sea) == Kind::kSea &&
							 PlacementRules::WaterKind("DefaultWater", -2000.0f, sea) == Kind::kLake &&
							 PlacementRules::WaterKind("", -13950.0f, sea) == Kind::kSea && PlacementRules::IsMarshLand("LFrozenMarshGrass01") &&
							 PlacementRules::IsMarshLand("RP_LMarshMud01") && !PlacementRules::IsMarshLand("LTundra01");
		PlacementRules::Params pp;
		pp.land = 1.0f;              // the shore and valley expectations below are relative to land = 1
		pp.shoreDistance = 128.0f;   // two cells at 64 units
		const int n = 9;
		std::vector<float> g(n * n, 0.0f);
		std::vector<Kind>  k(n * n, Kind::kLand);
		for (int y = 3; y <= 5; ++y)
			for (int x = 3; x <= 5; ++x) k[y * n + x] = Kind::kLake;
		auto out = PlacementRules::Build(g, k, n, 64.0f, pp);
		auto at = [&](int x, int y) { return out[static_cast<std::size_t>(y * n + x)]; };
		const bool shoreOk = std::fabs(at(4, 4)[0] - pp.lake) < 1e-4f && std::fabs(at(4, 4)[2] - pp.freshSteam) < 1e-4f &&
							 std::fabs(at(2, 4)[0] - pp.shoreFactor) < 1e-4f && std::fabs(at(1, 4)[0] - 1.25f) < 1e-3f && std::fabs(at(0, 4)[0] - 1.0f) < 1e-4f;
		std::vector<float> gv(n * n, 0.0f);
		std::vector<Kind>  kv(n * n, Kind::kLand);
		gv[2 * n + 2] = -600.0f;   // a hollow
		gv[6 * n + 6] = 600.0f;    // a knoll
		kv[0] = Kind::kHotSpring;
		kv[n * n - 1] = Kind::kSea;
		PlacementRules::Params pv;
		pv.shoreFactor = 1.0f;   // the water cells here only test the steam channels, not the shore
		pv.land = 1.0f;
		pv.valleyRadius = 128.0f;   // two cells: each test spot's neighbourhood lies inside the 9x9 grid
		auto ov = PlacementRules::Build(gv, kv, n, 64.0f, pv);
		const bool valleyOk = std::fabs(ov[2 * n + 2][0] - 1.8f) < 1e-4f && std::fabs(ov[6 * n + 6][0] - 0.5f) < 1e-4f &&
							  std::fabs(ov[0][1] - PlacementRules::Params{}.hotSpring) < 1e-4f && std::fabs(ov[n * n - 1][2] - PlacementRules::Params{}.seaSteam) < 1e-4f;
		Check(kindsOk && shoreOk && valleyOk, "placement (water kinds, shore band, valley, steam)",
			std::string("kinds ") + (kindsOk ? "ok" : "WRONG") + ", lake " + std::to_string(at(4, 4)[0]) + ", shore 1/2/3 cells " + std::to_string(at(2, 4)[0]) + "/" +
				std::to_string(at(1, 4)[0]) + "/" + std::to_string(at(0, 4)[0]) + ", hollow " + std::to_string(ov[2 * n + 2][0]) + ", knoll " +
				std::to_string(ov[6 * n + 6][0]) + ", hot spring steam " + std::to_string(ov[0][1]) + ", sea smoke " + std::to_string(ov[n * n - 1][2]));
	}


	// Placement fixes (GPT review 2026-11-08): kind multipliers kept as set, ShoreDistance 0 = no shore, and a valley
	// value that stays with the world spot when the grid moves over a uniform slope.
	{
		using PlacementRules::Kind;
		const int n = 9;
		std::vector<float> g(n * n, 0.0f);
		std::vector<Kind>  k(n * n, Kind::kLand);
		k[4 * n + 4] = Kind::kLake;
		k[4 * n + 6] = Kind::kMarshLand;
		PlacementRules::Params p0;
		p0.lake = 0.0f;
		p0.marshLand = 5.0f;
		p0.shoreDistance = 0.0f;
		p0.land = 1.0f;
		const auto o0 = PlacementRules::Build(g, k, n, 64.0f, p0);
		const bool keepOk = o0[4 * n + 4][0] == 0.0f && std::fabs(o0[4 * n + 6][0] - 5.0f) < 1e-4f;
		const bool noShore = std::fabs(o0[4 * n + 3][0] - 1.0f) < 1e-4f && std::fabs(o0[3 * n + 4][0] - 1.0f) < 1e-4f;

		// 129 x 129 cells of 64 units on a 50% slope along y; the window then moves 16 cells up the slope.
		const int    m = 129;
		const float  cell = 64.0f;
		auto slope = [&](int a_shiftY) {
			std::vector<float> h(static_cast<std::size_t>(m) * m);
			for (int y = 0; y < m; ++y)
				for (int x = 0; x < m; ++x) h[static_cast<std::size_t>(y) * m + x] = 0.5f * cell * static_cast<float>(y + a_shiftY);
			return PlacementRules::Build(h, std::vector<Kind>(h.size(), Kind::kLand), m, cell, PlacementRules::Params{});
		};
		const auto s0 = slope(0);
		const auto s1 = slope(16);
		const float before = s0[static_cast<std::size_t>(80) * m + 64][0];   // world row 80
		const float after = s1[static_cast<std::size_t>(64) * m + 64][0];    // the same world row after the move
		const float land = PlacementRules::Params{}.land;   // dry ground's own multiplier
		const bool  slopeOk = std::fabs(before - land) < 1e-3f && std::fabs(after - land) < 1e-3f;
		Check(keepOk && noShore && slopeOk, "placement fixes (kind multipliers kept, no shore at 0, valley fixed to the world)",
			"lake 0 -> " + std::to_string(o0[4 * n + 4][0]) + ", marsh land 5 -> " + std::to_string(o0[4 * n + 6][0]) + ", next to water " +
				std::to_string(o0[4 * n + 3][0]) + ", slope spot before/after the move " + std::to_string(before) + "/" + std::to_string(after));
	}

	// Pooled fog depth (PlacementRules, closing + one floor and lid per closed pool): a bowl fills to its rim with a
	// level top; a cap or a lower fill gives a lower level top, measured from the pool's 10th-percentile floor; an
	// asymmetric trench stays level; one deep cell does not empty the pool; a uniform slope and a cliff step stay empty
	// (also at the grid edge); a ditch fills to its banks; an open descending valley keeps its depth on the world spot
	// when the grid moves (reviewer 2026-11-08).
	{
		using PlacementRules::Kind;
		const int   n = 33;
		const float cell = 64.0f;
		auto grid = [&](auto a_height, int a_size, int a_shiftY) {
			std::vector<float> g(static_cast<std::size_t>(a_size) * a_size);
			for (int y = 0; y < a_size; ++y)
				for (int x = 0; x < a_size; ++x) g[static_cast<std::size_t>(y) * a_size + x] = a_height(x, y + a_shiftY);
			return g;
		};
		auto build = [&](const std::vector<float>& a_g, int a_size, float a_cap, float a_fill) {
			PlacementRules::Params pp;
			pp.poolRadius = 8.0f * cell;
			pp.poolMaxDepth = a_cap;
			pp.poolFill = a_fill;
			return PlacementRules::Build(a_g, std::vector<Kind>(a_g.size(), Kind::kLand), a_size, cell, pp);
		};
		// Top range over the filled cells (ground + depth), skipping cells deeper than a_skipBelow (a pit).
		auto tops = [&](const std::vector<float>& a_g, const std::vector<std::array<float, 4>>& a_out, float a_skipBelow) {
			float lo = 1e9f, hi = -1e9f;
			int   filled = 0;
			for (std::size_t i = 0; i < a_g.size(); ++i) {
				if (a_out[i][3] <= 0.0f || a_g[i] < a_skipBelow) continue;
				lo = std::min(lo, a_g[i] + a_out[i][3]);
				hi = std::max(hi, a_g[i] + a_out[i][3]);
				++filled;
			}
			return std::array<float, 3>{ lo, hi, static_cast<float>(filled) };
		};
		// The 10th-percentile ground of the cells below the rim (the pool's floor as Build takes it).
		auto floorOf = [&](std::vector<float> a_g, float a_rim) {
			std::vector<float> in;
			for (float v : a_g)
				if (v < a_rim - 0.5f) in.push_back(v);
			const std::size_t k = in.size() / 10;
			std::nth_element(in.begin(), in.begin() + static_cast<std::ptrdiff_t>(k), in.end());
			return in[k];
		};
		auto bowl = [](int x, int y) {
			const float d2 = static_cast<float>((x - 16) * (x - 16) + (y - 16) * (y - 16));
			return d2 < 25.0f ? 100.0f - 80.0f * (1.0f - d2 / 25.0f) : 100.0f;
		};
		auto trench = [](int x, int y) { return (y < 15 || y > 17 || x < 4 || x > 28) ? 100.0f : 1.25f * static_cast<float>(x - 4); };
		auto pit = [](int x, int y) {
			if (x == 9 && y == 16) return -300.0f;
			return (x >= 8 && x <= 23 && y >= 8 && y <= 23) ? 20.0f : 100.0f;
		};
		const auto gb = grid(bowl, n, 0), gt = grid(trench, n, 0), gp = grid(pit, n, 0);
		const float fb = floorOf(gb, 100.0f), ft = floorOf(gt, 100.0f);
		const auto bowlFull = tops(gb, build(gb, n, 1000.0f, 1.0f), -1e9f);
		const auto bowlCap = tops(gb, build(gb, n, 50.0f, 1.0f), -1e9f);
		const auto bowlHalf = tops(gb, build(gb, n, 1000.0f, 0.5f), -1e9f);
		const auto trenchHalf = tops(gt, build(gt, n, 1000.0f, 0.5f), -1e9f);
		const auto trenchCap = tops(gt, build(gt, n, 50.0f, 1.0f), -1e9f);
		const auto pitOut = build(gp, n, 280.0f, 1.0f);
		const auto pitTops = tops(gp, pitOut, -100.0f);
		const float pitDepth = pitOut[static_cast<std::size_t>(16) * n + 9][3];
		const auto slope = tops(grid([](int x, int) { return 32.0f * static_cast<float>(x); }, n, 0),
			build(grid([](int x, int) { return 32.0f * static_cast<float>(x); }, n, 0), n, 1000.0f, 1.0f), -1e9f);
		const auto cliffG = grid([](int x, int) { return x < 16 ? 0.0f : 1000.0f; }, n, 0);
		const auto cliff = tops(cliffG, build(cliffG, n, 1000.0f, 1.0f), -1e9f);
		const auto ditchG = grid([](int x, int) { return (x >= 15 && x <= 17) ? 0.0f : 100.0f; }, n, 0);
		const auto ditch = tops(ditchG, build(ditchG, n, 1000.0f, 1.0f), -1e9f);
		auto eq = [](float a, float b) { return std::fabs(a - b) < 1e-3f; };
		const bool closedOk = eq(bowlFull[0], 100.0f) && eq(bowlFull[1], 100.0f) && eq(bowlCap[0], fb + 50.0f) && eq(bowlCap[1], fb + 50.0f) &&
		                      eq(bowlHalf[0], fb + 0.5f * (100.0f - fb)) && eq(bowlHalf[1], bowlHalf[0]) &&
		                      eq(trenchHalf[0], ft + 0.5f * (100.0f - ft)) && eq(trenchHalf[1], trenchHalf[0]) && eq(trenchCap[0], ft + 50.0f) &&
		                      eq(trenchCap[1], trenchCap[0]) && eq(pitTops[0], 100.0f) && eq(pitTops[1], 100.0f) && pitTops[2] > 200.0f &&
		                      eq(pitDepth, 400.0f);   // top 100 over a -300 cell, under the 2 x 280 hold
		const bool emptyOk = slope[2] == 0.0f && cliff[2] == 0.0f && eq(ditch[0], 100.0f) && eq(ditch[1], 100.0f);
		// Open descending valley (V across x, falling 3 units a cell along y, 150 deep, 25 cells wide): grid of 65,
		// shifted 8 rows downhill; the depth on the axis at the same world rows must not change.
		const int  m = 65;
		auto       valley = [](int x, int y) { return -3.0f * static_cast<float>(y) + 150.0f * std::min(std::fabs(static_cast<float>(x - 32)) / 12.5f, 1.0f); };
		const auto g0 = grid(valley, m, 0), g1 = grid(valley, m, 8);
		const auto o0 = build(g0, m, 280.0f, 1.0f), o1 = build(g1, m, 280.0f, 1.0f);
		float worstShift = 0.0f, axisDepth = 0.0f;
		for (int wy = 24; wy <= 40; ++wy) {
			const float d0 = o0[static_cast<std::size_t>(wy) * m + 32][3], d1 = o1[static_cast<std::size_t>(wy - 8) * m + 32][3];
			worstShift = std::max(worstShift, std::fabs(d0 - d1));
			axisDepth = std::max(axisDepth, d0);
		}
		// Open steep valley (V across x, sides 0.6, the same along y so it leaves the grid): the closing lid is the
		// side's height one window out (307), far above a 50-unit cap. The top must be level at floor + cap and the sides above
		// it empty; capping the closing depth per cell laid 50 units of fog up both sides to 307.
		auto       steep = [](int x, int) { return 38.4f * std::fabs(static_cast<float>(x - 32)); };
		const auto gs = grid(steep, m, 0);
		const auto os = build(gs, m, 50.0f, 1.0f);
		const auto steepTops = tops(gs, os, -1e9f);
		const float sideDepth = os[static_cast<std::size_t>(32) * m + 37][3];   // ground 192, above the new top (floor + cap)
		const auto steepHalf = tops(gs, build(gs, m, 50.0f, 0.5f), -1e9f);   // half fill: level at floor + 25
		const bool steepOk = eq(steepTops[0], 50.0f) && eq(steepTops[1], 50.0f) && steepTops[2] > 0.0f && sideDepth == 0.0f &&
		                     eq(steepHalf[0], 25.0f) && eq(steepHalf[1], 25.0f) && steepHalf[2] > 0.0f;
		// The three cases of the GPT review 2026-11-12 on the first form of that rule.
		// (a) Low ground across a ridge is not the pool's floor: rows 0, 1000, 500, 1000, 1000 (and so on) along x.
		auto       hanging = [](int x, int) { return x == 0 ? 0.0f : x == 2 ? 500.0f : 1000.0f; };
		const auto gh = grid(hanging, n, 0);
		const float hangDepth = build(gh, n, 280.0f, 1.0f)[static_cast<std::size_t>(16) * n + 2][3];
		// (b) A closed basin (rim 1000, floor 500, one cell at 0) keeps its fog when one rim cell turns unknown.
		auto       basin = [](int x, int y) { return (x < 10 || x > 22 || y < 10 || y > 22) ? 1000.0f : (x == 16 && y == 16) ? 0.0f : 500.0f; };
		auto       gc = grid(basin, n, 0);
		const float closedDepth = build(gc, n, 280.0f, 1.0f)[static_cast<std::size_t>(14) * n + 14][3];
		for (int y = 0; y < n; ++y) gc[static_cast<std::size_t>(y) * n + 23] = -1.0e30f;   // the column next to the basin: the pool is open now
		const float openDepth = build(gc, n, 280.0f, 1.0f)[static_cast<std::size_t>(14) * n + 14][3];
		// (c) A valley falling 10 units a cell: 16 cells from the grid's downhill edge the depth stays on its world
		// spot when the grid moves 8 rows (the floor's reach is 4 x cap = 1120 units = 18 cells here, clipped only
		// within that of the edge).
		auto       falling = [](int x, int y) { return -10.0f * static_cast<float>(y) + 50.0f * std::fabs(static_cast<float>(x - 32)); };
		const auto f0 = build(grid(falling, m, 0), m, 280.0f, 1.0f), f1 = build(grid(falling, m, 8), m, 280.0f, 1.0f);
		float       fallShift = 0.0f, fallDepth = 0.0f;
		for (int wy = 28; wy <= 44; ++wy) {
			const float d0 = f0[static_cast<std::size_t>(wy) * m + 32][3], d1 = f1[static_cast<std::size_t>(wy - 8) * m + 32][3];
			fallShift = std::max(fallShift, std::fabs(d0 - d1));
			fallDepth = std::max(fallDepth, d0);
		}
		const bool reviewOk = eq(hangDepth, 280.0f) && eq(closedDepth, 280.0f) && eq(openDepth, 280.0f) && fallShift < 1e-3f && fallDepth > 50.0f;
		const bool valleyOk = worstShift < 1e-3f && axisDepth > 50.0f && steepOk && reviewOk;
		// Easing (EasePools): after a 2-cell move a depth stays on its world spot and moves halfway to a new target;
		// cells new to the grid take their target; a second ease with alpha 1 lands on it.
		std::vector<float> prev(static_cast<std::size_t>(5) * 5, 0.0f);
		prev[static_cast<std::size_t>(2) * 5 + 3] = 100.0f;   // world spot at grid (3, 2)
		std::vector<std::array<float, 4>> tgt(25, { 1.0f, 0.0f, 0.0f, 40.0f });
		const float leftHalf = PlacementRules::EasePools(prev, 5, 2, 0, tgt, 0.5f);   // grid moved 2 cells +x: spot now at (1, 2)
		const bool  easeSpot = eq(tgt[static_cast<std::size_t>(2) * 5 + 1][3], 70.0f) && eq(tgt[static_cast<std::size_t>(2) * 5 + 4][3], 40.0f) &&
		                      eq(tgt[0][3], 20.0f) && eq(leftHalf, 30.0f);
		std::vector<std::array<float, 4>> tgt2(25, { 1.0f, 0.0f, 0.0f, 40.0f });
		const float leftDone = PlacementRules::EasePools(prev, 5, 0, 0, tgt2, 1.0f);
		const bool  easeOk = easeSpot && eq(tgt2[static_cast<std::size_t>(2) * 5 + 1][3], 40.0f) && eq(leftDone, 0.0f);
		Check(closedOk && emptyOk && valleyOk && easeOk, "pooled fog depth (level tops, cap, fill, pit, slope, cliff, ditch, open valley, easing)",
			"bowl top " + std::to_string(bowlFull[0]) + ".." + std::to_string(bowlFull[1]) + ", capped " + std::to_string(bowlCap[0]) + ".." + std::to_string(bowlCap[1]) + " (floor " +
				std::to_string(fb) + " + 50), half " + std::to_string(bowlHalf[0]) + ".." + std::to_string(bowlHalf[1]) + "; trench half " +
				std::to_string(trenchHalf[0]) + ".." + std::to_string(trenchHalf[1]) + ", capped " + std::to_string(trenchCap[0]) + ".." +
				std::to_string(trenchCap[1]) + "; pit pool top " + std::to_string(pitTops[0]) + ".." + std::to_string(pitTops[1]) + " over " +
				std::to_string(pitTops[2]) + " cells, pit depth " + std::to_string(pitDepth) + "; slope " + std::to_string(slope[2]) + ", cliff " +
				std::to_string(cliff[2]) + ", ditch " + std::to_string(ditch[0]) + ".." + std::to_string(ditch[1]) + "; open valley depth change on a 8-row move " +
				std::to_string(worstShift) + " (axis depth " + std::to_string(axisDepth) + "), steep open valley top " + std::to_string(steepTops[0]) + ".." +
					std::to_string(steepTops[1]) + " side depth " + std::to_string(sideDepth) + ", hanging valley " + std::to_string(hangDepth) + ", basin closed " +
					std::to_string(closedDepth) + " open " + std::to_string(openDepth) + ", falling valley change " + std::to_string(fallShift) + " (depth " +
					std::to_string(fallDepth) + "); easing " + (easeOk ? "ok" : "WRONG"));
	}

	ID3DBlob* vsb = Compile(RaymarchShaders::kVS, "vs_5_0");
	ID3DBlob* psb = Compile(RaymarchShaders::kMarchPS, "ps_5_0");
	// ShaderCache: the background job's result is what Compile hands back; a source that does not compile is left out,
	// so Compile fails there with the compiler's message (the caller logs it), and an unknown source still compiles.
	{
		const std::string good = "float4 main() : SV_TARGET { return float4(1.0, 0.0, 0.0, 1.0); }";
		const std::string bad = "float4 main() : SV_TARGET { return undefinedName; }";
		const std::string other = "float4 main() : SV_TARGET { return float4(0.0, 1.0, 0.0, 1.0); }";
		const bool started = ShaderCache::StartPrecompile({ { good, "main", "ps_5_0", "check" }, { bad, "main", "ps_5_0", "check" } });
		ID3DBlob* a = nullptr; ID3DBlob* b = nullptr; ID3DBlob* e = nullptr; ID3DBlob* badCode = nullptr; ID3DBlob* badErr = nullptr; ID3DBlob* o2 = nullptr;
		const HRESULT h1 = ShaderCache::Compile(good, "main", "ps_5_0", "check", &a, &e);
		const HRESULT h2 = ShaderCache::Compile(good, "main", "ps_5_0", "check", &b, nullptr);
		const HRESULT h3 = ShaderCache::Compile(bad, "main", "ps_5_0", "check", &badCode, &badErr);
		const HRESULT h4 = ShaderCache::Compile(other, "main", "ps_5_0", "check", &o2, nullptr);
		const auto [cached, asked] = ShaderCache::PrecompileCount();
		Check(started && SUCCEEDED(h1) && SUCCEEDED(h2) && a && a == b && !e && FAILED(h3) && !badCode && badErr && SUCCEEDED(h4) && o2 && o2 != a && cached == 1 && asked == 2,
			"shader cache serves the background compile", "same blob " + std::to_string(a == b) + ", failed source reports errors " + std::to_string(badErr != nullptr) +
				", cached " + std::to_string(cached) + " of " + std::to_string(asked));
		SafeRelease(a); SafeRelease(b); SafeRelease(badErr); SafeRelease(o2);
	}
	// Pool visibility floor: untouched when thin enough, 3 / extinction = the floor when not, off with 0. The ground
	// density factor (1.582 for the tall layer's a = 1) is taken from the numeric profile, not the closed form.
	{
		constexpr float a = RaymarchShaders::kUpperSteepnessCpp;
		const float ground = static_cast<float>(g_upperProfile.Shape(0.0));
		const float thin = PlacementRules::LimitPoolDensity(0.1f, 0.3f, a, 50.0f);          // 63 m already
		const float thick = PlacementRules::LimitPoolDensity(0.2f, 0.75f, a, 50.0f);        // would be 12.6 m
		const float vis = 3.0f / (0.75f * thick * ground);
		const float off = PlacementRules::LimitPoolDensity(0.2f, 0.75f, a, 0.0f);
		const float noFog = PlacementRules::LimitPoolDensity(0.2f, 0.0f, a, 50.0f);
		// a = 0 (even slab): ground factor 1, so the floor is 3 / 50 / 0.75 = 0.08, finite.
		const float even = PlacementRules::LimitPoolDensity(0.2f, 0.75f, 0.0f, 50.0f);
		Check(std::fabs(thin - 0.1f) < 1e-6f && std::fabs(vis - 50.0f) < 1e-3f && off == 0.2f && noFog == 0.2f && std::fabs(even - 0.08f) < 1e-4f,
			"pool visibility floor",
			"thin " + std::to_string(thin) + ", thick clamps to visibility " + std::to_string(vis) + " m (ground factor " + std::to_string(ground) +
				"), floor 0 keeps " + std::to_string(off) + ", even slab " + std::to_string(even));
	}

	// Column profile on the GPU (RaymarchShaders LayerTau, one step per case) against a fine midpoint sum of the point
	// density along the same step (Profile above), and against W for full columns and the even slab for a -> 0.
	{
		constexpr float kH = 100.0f, kW = 0.8f;
		struct Ray
		{
			float ha, hb, h, a, len, dz;
		};
		// Straight down through the whole column (also below the ground and above the lid), in one step and in 7.
		std::vector<Ray> rays;
		rays.push_back({ kH + 20.0f, -20.0f, 50.0f, 3.0f, kH + 40.0f, -(kH + 40.0f) });   // 0: a = 3
		rays.push_back({ kH + 20.0f, -20.0f, 50.0f, 1.0f, kH + 40.0f, -(kH + 40.0f) });   // 1: a = 1
		constexpr int kDownSteps = 7;
		constexpr float kDownLen = (kH + 40.0f) / kDownSteps;
		for (int s = 0; s < kDownSteps; ++s) {   // 2..8: a = 3 in 7 steps (the spans must add up)
			const float ha = kH + 30.0f - s * kDownLen;
			rays.push_back({ ha, ha - kDownLen, ha - 0.5f * kDownLen, 3.0f, kDownLen, -kDownLen });
		}
		const std::size_t levelAt = rays.size();
		rays.push_back({ 0.3f * kH - 0.025f, 0.3f * kH + 0.025f, 0.3f * kH, 3.0f, 500.0f, 0.05f });   // under 1e-3 H: midpoint rule
		rays.push_back({ 0.7f * kH, 0.7f * kH, 0.7f * kH, 1.0f, 300.0f, 0.0f });
		const std::size_t lidAt = rays.size();
		rays.push_back({ 0.6f * kH, 1.5f * kH, 1.05f * kH, 3.0f, 180.0f, 0.9f * kH });    // rising 30 degrees out through the lid
		rays.push_back({ 1.5f * kH, 0.2f * kH, 0.85f * kH, 1.0f, 260.0f, -1.3f * kH });   // coming down through the lid
		const std::size_t slabAt = rays.size();
		rays.push_back({ 0.2f * kH, 0.7f * kH, 0.45f * kH, 0.001f, 250.0f, 0.5f * kH });
		rays.push_back({ 0.4f * kH, 0.4f * kH, 0.4f * kH, 0.001f, 300.0f, 0.0f });
		rays.push_back({ 0.2f * kH, 0.7f * kH, 0.45f * kH, 0.0f, 250.0f, 0.5f * kH });   // a = 0 from the ini: held to 1e-3
		rays.push_back({ 0.4f * kH, 0.4f * kH, 0.4f * kH, 0.0f, 300.0f, 0.0f });

		// The march pixel shader's source (its march core holds LayerTau) with a compute entry point of its own.
		const std::string body = std::string(RaymarchShaders::kCommon) + RaymarchShaders::kMarchPS + R"(
StructuredBuffer<float4>   ProfileCases : register(t8);
RWStructuredBuffer<float4> ProfileOut   : register(u0);
[numthreads(1, 1, 1)]
void ProfileMain(uint3 id : SV_DispatchThreadID)
{
	const float4 p = ProfileCases[id.x * 2];       // ha, hb, h, H
	const float4 q = ProfileCases[id.x * 2 + 1];   // W, a, step length, dz
	ProfileOut[id.x] = float4(LayerTau(p.x, p.y, p.z, p.w, q.x, 1.0, q.y, q.z, q.w), 0.0, 0.0, 0.0);
}
)";
		std::vector<float> in;
		for (const Ray& r : rays) {
			in.insert(in.end(), { r.ha, r.hb, r.h, kH, kW, r.a, r.len, r.dz });
		}
		const UINT count = static_cast<UINT>(rays.size());
		std::vector<float> gpu(count, NAN);
		ID3D11ComputeShader*       cs = nullptr;
		ID3D11Buffer*              inBuf = nullptr;
		ID3D11Buffer*              outBuf = nullptr;
		ID3D11Buffer*              stBuf = nullptr;
		ID3D11Buffer*              pcb = nullptr;
		ID3D11ShaderResourceView*  inSrv = nullptr;
		ID3D11UnorderedAccessView* outUav = nullptr;
		ID3DBlob*                  csb = nullptr;
		ID3DBlob*                  csErr = nullptr;
		const HRESULT              csHr = D3DCompile(body.data(), body.size(), "GroundFogProfile", nullptr, nullptr, "ProfileMain", "cs_5_0",
							 D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &csb, &csErr);
		Check(SUCCEEDED(csHr), "profile check compute shader", csErr ? static_cast<const char*>(csErr->GetBufferPointer()) : "compiled");
		SafeRelease(csErr);
		bool                       ok = csb && SUCCEEDED(dev->CreateComputeShader(csb->GetBufferPointer(), csb->GetBufferSize(), nullptr, &cs));
		SafeRelease(csb);
		D3D11_BUFFER_DESC      ind{ static_cast<UINT>(in.size() * sizeof(float)), D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 16 };
		D3D11_SUBRESOURCE_DATA inInit{ in.data(), 0, 0 };
		ok = ok && SUCCEEDED(dev->CreateBuffer(&ind, &inInit, &inBuf)) && SUCCEEDED(dev->CreateShaderResourceView(inBuf, nullptr, &inSrv));
		D3D11_BUFFER_DESC outd{ count * 16u, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 16 };
		ok = ok && SUCCEEDED(dev->CreateBuffer(&outd, nullptr, &outBuf)) && SUCCEEDED(dev->CreateUnorderedAccessView(outBuf, nullptr, &outUav));
		D3D11_BUFFER_DESC stDesc{ count * 16u, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 16 };
		ok = ok && SUCCEEDED(dev->CreateBuffer(&stDesc, nullptr, &stBuf));
		RaymarchShaders::RayCB prc{};
		prc.fog[0] = kOpacity;
		D3D11_BUFFER_DESC      pcbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
		D3D11_SUBRESOURCE_DATA pcbInit{ &prc, 0, 0 };
		ok = ok && SUCCEEDED(dev->CreateBuffer(&pcbd, &pcbInit, &pcb));
		if (ok) {
			ctx->CSSetShader(cs, nullptr, 0);
			ctx->CSSetConstantBuffers(0, 1, &pcb);
			ctx->CSSetShaderResources(8, 1, &inSrv);
			ctx->CSSetUnorderedAccessViews(0, 1, &outUav, nullptr);
			ctx->Dispatch(count, 1, 1);
			ID3D11ShaderResourceView*  noSrv = nullptr;
			ID3D11UnorderedAccessView* noUav = nullptr;
			ctx->CSSetShaderResources(8, 1, &noSrv);
			ctx->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
			ctx->CSSetShader(nullptr, nullptr, 0);
			ctx->CopyResource(stBuf, outBuf);
			D3D11_MAPPED_SUBRESOURCE m{};
			if (SUCCEEDED(ctx->Map(stBuf, 0, D3D11_MAP_READ, 0, &m))) {
				for (UINT i = 0; i < count; ++i) gpu[i] = static_cast<const float*>(m.pData)[i * 4];
				ctx->Unmap(stBuf, 0);
			}
		}
		SafeRelease(cs); SafeRelease(inSrv); SafeRelease(inBuf); SafeRelease(outUav); SafeRelease(outBuf); SafeRelease(stBuf); SafeRelease(pcb);

		// Midpoint sum of opacity x rho along the step: rho per metre = W / (H / 70) x shape(h / H), length per piece in metres.
		auto numeric = [&](const Ray& a_r) {
			const Profile p(a_r.a);
			constexpr int kN = 200000;
			double        tau = 0.0;
			for (int i = 0; i < kN; ++i) {
				const double s = (i + 0.5) / kN;
				const double h = a_r.ha + (static_cast<double>(a_r.hb) - a_r.ha) * s;
				tau += kOpacity * kW / (kH / 70.0) * p.Shape(h / kH) * (a_r.len / kN) / 70.0;
			}
			return tau;
		};
		auto rel = [](double a_x, double a_ref) { return std::fabs(a_x - a_ref) / std::max(std::fabs(a_ref), 1e-9); };
		bool finite = ok;
		for (float g : gpu) finite = finite && std::isfinite(g);

		// (a) Vertical integral = W (tau = opacity x W) for a = 3 and a = 1, GPU and the numeric reference both.
		{
			const double full = kOpacity * kW;
			double       sum7 = 0.0;
			for (int s = 0; s < kDownSteps; ++s) sum7 += gpu[2 + s];
			const double e3 = rel(gpu[0], full), e1 = rel(gpu[1], full), e7 = rel(sum7, full);
			const double n3 = rel(numeric(rays[0]), full), n1 = rel(numeric(rays[1]), full);
			Check(finite && e3 < 1e-3 && e1 < 1e-3 && e7 < 1e-3 && n3 < 1e-3 && n1 < 1e-3, "profile column integral is W",
				"opacity x W " + std::to_string(full) + ": GPU a=3 " + std::to_string(gpu[0]) + ", a=1 " + std::to_string(gpu[1]) + ", a=3 in 7 steps " +
					std::to_string(sum7) + "; numeric relative error a=3 " + std::to_string(n3) + ", a=1 " + std::to_string(n1));
		}
		// (b) Level ray inside the layer (midpoint fallback) and (c) a ray through the lid, against the numeric sum.
		auto against = [&](std::size_t a_from, const char* a_name) {
			double      worst = 0.0;
			std::string detail;
			for (std::size_t i = a_from; i < a_from + 2; ++i) {
				const double ref = numeric(rays[i]);
				worst = std::max(worst, rel(gpu[i], ref));
				detail += "a=" + std::to_string(rays[i].a).substr(0, 4) + " GPU " + std::to_string(gpu[i]) + " vs numeric " + std::to_string(ref) + "; ";
			}
			Check(finite && worst < 1e-3, a_name, detail + "worst relative " + std::to_string(worst));
		};
		against(levelAt, "profile level ray (midpoint rule)");
		against(lidAt, "profile ray through the lid");
		// (d) a -> 0 (0.001, and 0 held to it) is the even slab: density W / H everywhere in the column.
		{
			double      worst = 0.0;
			std::string detail;
			for (std::size_t i = slabAt; i < slabAt + 4; ++i) {
				const Ray&   r = rays[i];
				const double slab = r.dz != 0.0f ? kOpacity * kW * (r.hb - r.ha) / kH * (r.len / std::fabs(r.dz)) : kOpacity * kW / kH * r.len;
				worst = std::max({ worst, rel(gpu[i], slab), rel(numeric(r), slab) });
				detail += "GPU " + std::to_string(gpu[i]) + " vs slab " + std::to_string(slab) + "; ";
			}
			Check(finite && worst < 1e-2, "profile a -> 0 is an even slab", detail + "worst relative " + std::to_string(worst));
		}
	}
	ID3DBlob* cpb = Compile(RaymarchShaders::kCompositePS, "ps_5_0");
	if (!vsb || !psb || !cpb) {
		return g_fails;
	}
	ID3D11VertexShader* vs = nullptr;
	ID3D11PixelShader*  ps = nullptr;
	ID3D11PixelShader*  comp = nullptr;
	dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
	dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps);
	dev->CreatePixelShader(cpb->GetBufferPointer(), cpb->GetBufferSize(), nullptr, &comp);
	SafeRelease(vsb);
	SafeRelease(psb);
	SafeRelease(cpb);

	// Targets: march output (RGBA32F, full size) and a depth texture we fill by hand.
	D3D11_TEXTURE2D_DESC td{};
	td.Width = td.Height = kSize;
	td.MipLevels = td.ArraySize = 1;
	td.SampleDesc.Count = 1;
	td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	td.BindFlags = D3D11_BIND_RENDER_TARGET;
	ID3D11Texture2D*        out = nullptr;
	ID3D11RenderTargetView* rtv = nullptr;
	dev->CreateTexture2D(&td, nullptr, &out);
	dev->CreateRenderTargetView(out, nullptr, &rtv);
	td.Format = DXGI_FORMAT_R32_FLOAT;
	td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	td.Usage = D3D11_USAGE_DEFAULT;
	ID3D11Texture2D*          depthTex = nullptr;
	ID3D11ShaderResourceView* depthSrv = nullptr;
	std::vector<float>        depthData(kSize * kSize, 1.0f);
	D3D11_SUBRESOURCE_DATA    dinit{ depthData.data(), kSize * sizeof(float), 0 };
	dev->CreateTexture2D(&td, &dinit, &depthTex);
	dev->CreateShaderResourceView(depthTex, nullptr, &depthSrv);

	D3D11_BUFFER_DESC cbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
	ID3D11Buffer* cb = nullptr;
	dev->CreateBuffer(&cbd, nullptr, &cb);
	D3D11_SAMPLER_DESC sd{};
	sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	sd.MaxLOD = D3D11_FLOAT32_MAX;
	ID3D11SamplerState* linear = nullptr;
	dev->CreateSamplerState(&sd, &linear);

	const XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(70.0f), 1.0f, 15.0f, 100000.0f);
	XMFLOAT4X4     pf;
	XMStoreFloat4x4(&pf, proj);

	// Fog box height range used by render (the game spans the whole terrain height range) and the largest
	// per-pixel |GPU - CPU| of the last render.
	float boxLo = -1.0f, boxHi = maxH * 1.01f + 1.0f, pixelErr = 0.0f;
	// Optional tall layer for render(): a uniform H, W texture bound at t4 (nullptr = none, as an unbound slot).
	ID3D11ShaderResourceView* testUpper = nullptr;
	float                     testUpperH = 0.0f, testUpperW = 0.0f;
	// Camera looks at the ground point (300, 200, 0) from a given pitch, distance and yaw.
	auto render = [&](float a_pitchDeg, float a_dist, float a_yawDeg, float a_sceneDistAtCentre, float& a_cpu) {
		const float pitch = XMConvertToRadians(a_pitchDeg), yaw = XMConvertToRadians(a_yawDeg);
		const XMFLOAT3 target{ 300.0f, 200.0f, 0.0f };
		const XMFLOAT3 cam{ target.x - std::cos(pitch) * std::sin(yaw) * a_dist, target.y - std::cos(pitch) * std::cos(yaw) * a_dist,
			target.z + std::sin(pitch) * a_dist };
		const XMVECTOR forward = XMVector3Normalize(XMVectorSet(target.x - cam.x, target.y - cam.y, target.z - cam.z, 0));
		const XMMATRIX view = XMMatrixLookToLH(XMVectorZero(), forward, XMVectorSet(0, 0, 1, 0));
		const XMMATRIX vp = XMMatrixMultiply(view, proj);
		XMFLOAT4X4     inv;
		XMStoreFloat4x4(&inv, XMMatrixInverse(nullptr, vp));

		RaymarchShaders::RayCB rc{};
		std::memcpy(rc.invViewProj, &inv, sizeof(rc.invViewProj));
		rc.field[0] = kOrigin;
		rc.field[1] = kOrigin;
		rc.field[2] = kExtent;
		rc.field[3] = 1.0f;
		rc.box[0] = boxLo;
		rc.box[1] = boxHi;
		rc.box[2] = 100000.0f;
		rc.box[3] = kSteps;
		rc.fog[0] = kOpacity;
		rc.fog[1] = kSteepness;
		rc.depth[0] = pf._33;
		rc.depth[1] = pf._43;
		rc.depth[2] = kSize;
		rc.depth[3] = kSize;
		rc.lowRes[0] = kSize;
		rc.lowRes[1] = kSize;
		rc.lowRes[2] = 1.0f;
		rc.posAdjust[0] = cam.x;
		rc.posAdjust[1] = cam.y;
		rc.posAdjust[2] = cam.z;
		ctx->UpdateSubresource(cb, 0, nullptr, &rc, 0, 0);

		// Scene depth: the ground point itself (or a nearer surface when testing occlusion).
		const float sceneDist = a_sceneDistAtCentre > 0.0f ? a_sceneDistAtCentre : a_dist;
		const float devDepth = pf._33 + pf._43 / sceneDist;   // view z equals distance on the centre ray
		std::fill(depthData.begin(), depthData.end(), devDepth);
		ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);

		const float black[4]{};
		ctx->ClearRenderTargetView(rtv, black);
		D3D11_VIEWPORT viewport{ 0, 0, static_cast<float>(kSize), static_cast<float>(kSize), 0, 1 };
		ctx->RSSetViewports(1, &viewport);
		ctx->OMSetRenderTargets(1, &rtv, nullptr);
		ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		ctx->VSSetShader(vs, nullptr, 0);
		ctx->PSSetShader(ps, nullptr, 0);
		ctx->PSSetConstantBuffers(0, 1, &cb);
		ID3D11ShaderResourceView* srvs[2]{ depthSrv, sim.ColumnSrv() };
		ctx->PSSetShaderResources(0, 2, srvs);
		ctx->PSSetShaderResources(4, 1, &testUpper);
		ctx->PSSetSamplers(0, 1, &linear);
		ctx->Draw(3, 0);
		ID3D11ShaderResourceView* none[2]{};
		ctx->PSSetShaderResources(0, 2, none);
		ctx->PSSetShaderResources(4, 1, none);
		const auto img = ReadF32(dev, ctx, out, kSize, kSize, 4);

		// Average the 2x2 centre pixels (jitter differs per pixel) and integrate the matching rays on the CPU.
		float gpu = 0.0f;
		a_cpu = 0.0f;
		pixelErr = 0.0f;
		for (int dy = 0; dy < 2; ++dy) {
			for (int dx = 0; dx < 2; ++dx) {
				const int px = kSize / 2 - 1 + dx, py = kSize / 2 - 1 + dy;
				const float g = img[(static_cast<std::size_t>(py) * kSize + px) * 4];
				gpu += g;
				const XMVECTOR ndc = XMVectorSet((px + 0.5f) / kSize * 2 - 1, 1 - (py + 0.5f) / kSize * 2, devDepth, 1);
				XMFLOAT4       w;
				XMStoreFloat4(&w, XMVector4Transform(ndc, XMLoadFloat4x4(&inv)));
				XMFLOAT3 rel{ w.x / w.w, w.y / w.w, w.z / w.w };
				const float len = std::sqrt(rel.x * rel.x + rel.y * rel.y + rel.z * rel.z);
				float c = CpuFog(cols, cam, { rel.x / len, rel.y / len, rel.z / len }, len);
				if (testUpper) {
					// Tall layer over the flat ground (z = 0): rho(h) = W / (H / 70) x shape(h / H), a = kUpperSteepnessCpp.
					constexpr int kN = 20000;
					double tau = 0.0;
					for (int k = 0; k < kN; ++k) {
						const double t = (k + 0.5) * len / kN;
						const double h = cam.z + rel.z / len * t;
						if (h <= 0.0 || h >= testUpperH) continue;
						tau += kOpacity * testUpperW / (testUpperH / 70.0) * g_upperProfile.Shape(h / testUpperH) * (len / kN) / 70.0;
					}
					c = static_cast<float>(1.0 - (1.0 - c) * std::exp(-tau));
				}
				a_cpu += c;
				pixelErr = std::max(pixelErr, std::fabs(g - c));
			}
		}
		a_cpu /= 4.0f;
		return gpu / 4.0f;
	};

	struct Case
	{
		float pitch, dist, yaw;
		const char* name;
	};
	const Case cases[] = { { 45.0f, 400.0f, 0.0f, "pitch 45, 400 units" }, { 20.0f, 400.0f, 0.0f, "pitch 20, 400 units" },
		{ 45.0f, 900.0f, 0.0f, "pitch 45, 900 units" }, { 45.0f, 400.0f, 120.0f, "pitch 45, yaw 120" },
		{ 75.0f, 250.0f, 300.0f, "pitch 75, 250 units, yaw 300" } };
	for (const auto& c : cases) {
		float      cpu = 0.0f;
		const float gpu = render(c.pitch, c.dist, c.yaw, -1.0f, cpu);
		Check(std::isfinite(gpu) && cpu > 0.02f && std::fabs(gpu - cpu) < 0.03f + 0.08f * cpu, "fog matches the ground integral",
			std::string(c.name) + ": GPU " + std::to_string(gpu) + " vs CPU " + std::to_string(cpu));
	}

	// Tall layer (steam, pooled fog) marched next to the low one: a uniform 200-unit layer of W 0.8 with the flatter
	// profile (a = 1) must add its optical depth to the low layer's, as the CPU integral of both says.
	{
		std::vector<float> up(static_cast<std::size_t>(kRes) * kRes * 2);
		for (std::size_t i = 0; i < up.size(); i += 2) {
			up[i] = 200.0f;
			up[i + 1] = 0.8f;
		}
		D3D11_TEXTURE2D_DESC ud{};
		ud.Width = ud.Height = kRes;
		ud.MipLevels = ud.ArraySize = 1;
		ud.Format = DXGI_FORMAT_R32G32_FLOAT;
		ud.SampleDesc.Count = 1;
		ud.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA    uinit{ up.data(), kRes * sizeof(float) * 2, 0 };
		ID3D11Texture2D*          upTex = nullptr;
		ID3D11ShaderResourceView* upSrv = nullptr;
		dev->CreateTexture2D(&ud, &uinit, &upTex);
		dev->CreateShaderResourceView(upTex, nullptr, &upSrv);
		const float savedHi = boxHi;
		boxHi = 220.0f;
		testUpper = upSrv;
		testUpperH = 200.0f;
		testUpperW = 0.8f;
		float       cpuBoth = 0.0f, cpuLow = 0.0f;
		const float gpuBoth = render(45.0f, 400.0f, 0.0f, -1.0f, cpuBoth);
		testUpper = nullptr;
		const float gpuLow = render(45.0f, 400.0f, 0.0f, -1.0f, cpuLow);
		Check(std::fabs(gpuBoth - cpuBoth) < 0.01f + 0.03f * cpuBoth && gpuBoth > gpuLow + 0.05f, "tall layer marched next to the low one",
			"GPU " + std::to_string(gpuBoth) + " vs CPU " + std::to_string(cpuBoth) + " (low layer alone GPU " + std::to_string(gpuLow) + ")");
		boxHi = savedHi;
		SafeRelease(upSrv);
		SafeRelease(upTex);
	}

	// Reviewer case 2026-11-07: the box spans 2000 units of terrain height while the fog is under 100 units tall;
	// eye 120 units up looking 5 degrees down, so one of the 24 steps is far longer than the layer is tall.
	{
		boxLo = -1000.0f;
		boxHi = 1000.0f;
		float       cpu = 0.0f;
		const float gpu = render(5.0f, 120.0f / std::sin(XMConvertToRadians(5.0f)), 0.0f, -1.0f, cpu);
		// Tolerance: each step integrates the profile exactly but reads the column (H, W) once, and along this
		// grazing ray a step covers about 115 units of ground while the layer's height changes under it. With the
		// two-octave shape the worst pixel is off by 0.029 and the mean by 0.006 (2026-11-10); the earlier shape
		// happened to be nearly flat along this ray (0.0003).
		Check(cpu > 0.05f && std::fabs(gpu - cpu) < 0.015f && pixelErr < 0.04f, "thin layer in a tall box",
			"GPU " + std::to_string(gpu) + " vs CPU " + std::to_string(cpu) + ", worst pixel off by " + std::to_string(pixelErr));
		boxLo = -1.0f;
		boxHi = maxH * 1.01f + 1.0f;
	}

	float      cpuOpen = 0.0f;
	float      cpuNear = 0.0f;
	const float open = render(45.0f, 400.0f, 0.0f, -1.0f, cpuOpen);
	const float nearCut = render(45.0f, 400.0f, 0.0f, 350.0f, cpuNear);   // a surface 50 units before the ground
	Check(nearCut < open * 0.9f, "near surface cuts the fog", "fog " + std::to_string(nearCut) + " vs " + std::to_string(open) + " to the ground");

	// Two-pass path as in game: march at half resolution into RGBA32F, composite at full resolution onto an fp16
	// target with alpha blending (white fog over black, so red = fog). Checked against the composite's own weighting
	// recomputed on the CPU, a full-resolution march, and thin foreground columns.
	{
		const UINT lowW = kSize / 2;
		D3D11_TEXTURE2D_DESC ld{};
		ld.Width = ld.Height = lowW;
		ld.MipLevels = ld.ArraySize = 1;
		ld.SampleDesc.Count = 1;
		ld.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		ld.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
		ID3D11Texture2D*          lowTex = nullptr;
		ID3D11RenderTargetView*   lowRtv = nullptr;
		ID3D11ShaderResourceView* lowSrv = nullptr;
		dev->CreateTexture2D(&ld, nullptr, &lowTex);
		dev->CreateRenderTargetView(lowTex, nullptr, &lowRtv);
		dev->CreateShaderResourceView(lowTex, nullptr, &lowSrv);
		ld.Width = ld.Height = kSize;
		ld.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
		ld.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;   // SRV: the probe reads it
		ID3D11Texture2D*        sceneTex = nullptr;
		ID3D11RenderTargetView* sceneRtv = nullptr;
		dev->CreateTexture2D(&ld, nullptr, &sceneTex);
		dev->CreateRenderTargetView(sceneTex, nullptr, &sceneRtv);
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

		XMFLOAT4X4 inv{};
		XMFLOAT3   cam{};
		XMFLOAT3   camFwd{ 0.0f, 1.0f, 0.0f };   // the composite camera's forward direction
		auto setCamera = [&](const XMFLOAT3& a_cam, const XMFLOAT3& a_target) {
			cam = a_cam;
			const XMVECTOR forward = XMVector3Normalize(XMVectorSet(a_target.x - cam.x, a_target.y - cam.y, a_target.z - cam.z, 0));
			XMStoreFloat3(&camFwd, forward);
			const XMMATRIX vp = XMMatrixMultiply(XMMatrixLookToLH(XMVectorZero(), forward, XMVectorSet(0, 0, 1, 0)), proj);
			XMStoreFloat4x4(&inv, XMMatrixInverse(nullptr, vp));
			return vp;
		};
		float hfCb[4]{};          // height fog constants for setCb (zeros = off)
		float fieldOnCb = 1.0f;   // 0 = no wind field fog (march returns nothing)
		float farCb[4]{};         // far tier constants for setCb (zeros = off)
		float glowDirCb[4]{};     // RayCB sunDir for setCb (w = 0: no glow)
		float glowColCb[4]{};     // RayCB sunColor
		float poolGCb[4]{};       // RayCB poolGrid (far pools)
		float poolFCb[4]{};       // RayCB poolFar (x = 0: off)
		ID3D11ShaderResourceView* testCoarse = nullptr;   // t5 for the probe (far pools)
		auto setCb = [&](float a_scale) {
			RaymarchShaders::RayCB rc{};
			std::memcpy(rc.invViewProj, &inv, sizeof(rc.invViewProj));
			rc.field[0] = kOrigin;
			rc.field[1] = kOrigin;
			rc.field[2] = kExtent;
			rc.field[3] = fieldOnCb;
			std::memcpy(rc.heightFog, hfCb, sizeof(hfCb));
			std::memcpy(rc.farFog, farCb, sizeof(farCb));
			std::memcpy(rc.sunDir, glowDirCb, sizeof(glowDirCb));
			std::memcpy(rc.sunColor, glowColCb, sizeof(glowColCb));
			std::memcpy(rc.poolGrid, poolGCb, sizeof(poolGCb));
			std::memcpy(rc.poolFar, poolFCb, sizeof(poolFCb));
			rc.box[0] = -1.0f;
			rc.box[1] = maxH * 1.01f + 1.0f;
			rc.box[2] = 100000.0f;
			rc.box[3] = kSteps;
			rc.fog[0] = kOpacity;
			rc.fog[1] = kSteepness;
			rc.depth[0] = pf._33;
			rc.depth[1] = pf._43;
			rc.depth[2] = kSize;
			rc.depth[3] = kSize;
			rc.lowRes[0] = kSize * a_scale;
			rc.lowRes[1] = kSize * a_scale;
			rc.lowRes[2] = 1.0f / a_scale;
			for (int c = 0; c < 3; ++c) {
				rc.colorLow[c] = 1.0f;
				rc.colorHigh[c] = 1.0f;
			}
			rc.distFog[0] = 1.0e6f;
			rc.distFog[1] = 2.0e6f;
			rc.distFog[2] = 1.0f;
			rc.distFog[3] = 0.0f;
			rc.posAdjust[0] = cam.x;
			rc.posAdjust[1] = cam.y;
			rc.posAdjust[2] = cam.z;
			ctx->UpdateSubresource(cb, 0, nullptr, &rc, 0, 0);
		};
		const float black[4]{};
		ID3D11ShaderResourceView* none[3]{};
		auto marchInto = [&](ID3D11RenderTargetView* a_rtv, UINT a_w) {
			ctx->ClearRenderTargetView(a_rtv, black);
			D3D11_VIEWPORT v{ 0, 0, static_cast<float>(a_w), static_cast<float>(a_w), 0, 1 };
			ctx->RSSetViewports(1, &v);
			ctx->OMSetRenderTargets(1, &a_rtv, nullptr);
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
		};
		// Returns the composited scene's red channel; a_low gets the half-resolution march.
		auto twoPass = [&](std::vector<float>& a_low) {
			ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);
			setCb(0.5f);
			marchInto(lowRtv, lowW);
			ctx->ClearRenderTargetView(sceneRtv, black);
			D3D11_VIEWPORT fullVp{ 0, 0, static_cast<float>(kSize), static_cast<float>(kSize), 0, 1 };
			ctx->RSSetViewports(1, &fullVp);
			ctx->OMSetRenderTargets(1, &sceneRtv, nullptr);
			ctx->OMSetBlendState(blend, black, 0xFFFFFFFF);
			ctx->PSSetShader(comp, nullptr, 0);
			ID3D11ShaderResourceView* compIn[3]{ depthSrv, sim.ColumnSrv(), lowSrv };
			ctx->PSSetShaderResources(0, 3, compIn);
			ctx->Draw(3, 0);
			ctx->PSSetShaderResources(0, 3, none);
			ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);
			a_low = ReadF32(dev, ctx, lowTex, lowW, lowW, 4);
			D3D11_TEXTURE2D_DESC sdsc{};
			sceneTex->GetDesc(&sdsc);
			sdsc.Usage = D3D11_USAGE_STAGING;
			sdsc.BindFlags = 0;
			sdsc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			ID3D11Texture2D* st = nullptr;
			dev->CreateTexture2D(&sdsc, nullptr, &st);
			ctx->CopyResource(st, sceneTex);
			std::vector<float>       red(static_cast<std::size_t>(kSize) * kSize, -1.0f);
			D3D11_MAPPED_SUBRESOURCE mm{};
			if (SUCCEEDED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &mm))) {
				for (UINT y = 0; y < kSize; ++y) {
					const auto* row = reinterpret_cast<const std::uint16_t*>(static_cast<const std::uint8_t*>(mm.pData) + y * mm.RowPitch);
					for (UINT x = 0; x < kSize; ++x) red[y * kSize + x] = HalfToFloat(row[x * 4]);
				}
				ctx->Unmap(st, 0);
			}
			SafeRelease(st);
			return red;
		};
		// Single full-resolution march (the reference the upsampled result should follow).
		auto fullMarch = [&]() {
			ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);
			setCb(1.0f);
			marchInto(rtv, kSize);
			const auto         img = ReadF32(dev, ctx, out, kSize, kSize, 4);
			std::vector<float> fog(static_cast<std::size_t>(kSize) * kSize);
			for (std::size_t i = 0; i < fog.size(); ++i) fog[i] = img[i * 4];
			return fog;
		};
		// CPU copy of the composite's SceneDistance and neighbour weighting. NaN when no neighbour matches.
		auto sceneDist = [&](int a_x, int a_y) {
			const XMVECTOR ndc = XMVectorSet((a_x + 0.5f) / kSize * 2 - 1, 1 - (a_y + 0.5f) / kSize * 2, depthData[a_y * kSize + a_x], 1);
			XMFLOAT4       w;
			XMStoreFloat4(&w, XMVector4Transform(ndc, XMLoadFloat4x4(&inv)));
			return std::sqrt((w.x * w.x + w.y * w.y + w.z * w.z) / (w.w * w.w));
		};
		auto expectComposite = [&](const std::vector<float>& a_low, int a_x, int a_y) {
			const float myDist = sceneDist(a_x, a_y);
			const float lx = (a_x + 0.5f) / 2.0f - 0.5f, ly = (a_y + 0.5f) / 2.0f - 0.5f;
			const int   bx = static_cast<int>(std::floor(lx)), by = static_cast<int>(std::floor(ly));
			float       sum = 0.0f, wsum = 0.0f;
			for (int y = -1; y <= 1; ++y) {
				for (int x = -1; x <= 1; ++x) {
					const int    cx = std::clamp(bx + x, 0, static_cast<int>(lowW) - 1), cy = std::clamp(by + y, 0, static_cast<int>(lowW) - 1);
					const float* m = &a_low[(static_cast<std::size_t>(cy) * lowW + cx) * 4];
					const float  rel = std::fabs(m[3] - myDist) / std::max(myDist, 1.0f);
					if (rel > 0.15f) continue;
					const float ox = cx - lx, oy = cy - ly;
					const float w = (1.0f - rel / 0.15f) / (1.0f + ox * ox + oy * oy);
					sum += m[0] * w;
					wsum += w;
				}
			}
			return wsum > 1e-5f ? sum / wsum : NAN;
		};

		// Case 1: ground at a fixed view depth, camera pitched 45 degrees at 400 units. Thin columns of very near
		// depth (40 units, above all fog) at an odd x, which the half-res march reads (it samples full x = 2c + 1),
		// and at an even x, which it misses.
		const float pitch = XMConvertToRadians(45.0f);
		const XMFLOAT3 target{ 300.0f, 200.0f, 0.0f };
		setCamera({ target.x, target.y - std::cos(pitch) * 400.0f, target.z + std::sin(pitch) * 400.0f }, target);
		const float groundDepth = pf._33 + pf._43 / 400.0f;
		const float nearDepth = pf._33 + pf._43 / 40.0f;
		const int   colOdd = 41, colEven = 64;
		for (UINT y = 0; y < kSize; ++y) {
			for (UINT x = 0; x < kSize; ++x) {
				const int ix = static_cast<int>(x);
				depthData[y * kSize + x] = ix == colOdd || ix == colEven ? nearDepth : groundDepth;
			}
		}
		std::vector<float> low;
		const auto         scene = twoPass(low);
		// Weighting: every pixel of rows 30..60, columns 20..34 (clear of both columns, where the jittered march
		// varies from pixel to pixel, so a wrong block or weight shows) must equal the CPU recomputation.
		float maxErr = 0.0f, meanFog = 0.0f;
		int   n = 0;
		for (int y = 30; y <= 60; ++y) {
			for (int x = 20; x <= 34; ++x) {
				const float e = expectComposite(low, x, y);
				maxErr = std::max(maxErr, std::isnan(e) ? 1.0f : std::fabs(scene[y * kSize + x] - e));
				meanFog += scene[y * kSize + x];
				++n;
			}
		}
		meanFog /= n;
		Check(maxErr < 0.004f && meanFog > 0.05f, "composite matches its weighting",
			"max |GPU - CPU| " + std::to_string(maxErr) + " over " + std::to_string(n) + " pixels, mean fog " + std::to_string(meanFog));
		// Backlit glow: with the sun straight ahead the fog brightens (white sun, strength 0.3, g 0.6: up to 1.5 x 0.3 over
		// the plain colour, times alpha); with the sun behind the camera it does not change.
		{
			auto meanOf = [&](const std::vector<float>& a_scene) {
				double sum = 0.0;
				for (int y = 30; y <= 60; ++y)
					for (int x = 20; x <= 34; ++x) sum += a_scene[y * kSize + x];
				return static_cast<float>(sum / (31.0 * 15.0));
			};
			std::vector<float> lowG;
			const float plainMean = meanOf(scene);
			const float meanAlpha = plainMean;   // fog colour 1 here, so the scene's red is the alpha
			glowColCb[0] = glowColCb[1] = glowColCb[2] = 1.0f;
			glowColCb[3] = 0.6f;
			glowDirCb[0] = camFwd.x;
			glowDirCb[1] = camFwd.y;
			glowDirCb[2] = camFwd.z;
			glowDirCb[3] = 0.3f;
			const float aheadMean = meanOf(twoPass(lowG));
			glowDirCb[0] = -camFwd.x;
			glowDirCb[1] = -camFwd.y;
			glowDirCb[2] = -camFwd.z;
			const float behindMean = meanOf(twoPass(lowG));
			glowDirCb[3] = 0.0f;
			const bool finite = std::isfinite(aheadMean) && std::isfinite(behindMean);
			Check(finite && aheadMean > plainMean * 1.2f && aheadMean < plainMean + 1.5f * 0.3f * meanAlpha + 1e-3f && std::fabs(behindMean - plainMean) < 1e-4f,
				"backlit glow toward the sun only", "mean fog plain " + std::to_string(plainMean) + ", sun ahead " + std::to_string(aheadMean) +
				", sun behind " + std::to_string(behindMean));
		}
		bool evenMissed = true;
		for (int c = colEven / 2 - 2; c <= colEven / 2 + 1; ++c) {
			evenMissed = evenMissed && low[(static_cast<std::size_t>(kSize / 4) * lowW + c) * 4 + 3] > 100.0f;
		}
		const float onOdd = scene[(kSize / 2) * kSize + colOdd];
		const float onEven = scene[(kSize / 2) * kSize + colEven];
		Check(onOdd >= 0.0f && onOdd < 0.02f, "thin near object stays clear (sampled column)", "fog " + std::to_string(onOdd));
		Check(evenMissed && onEven >= 0.0f && onEven < 0.02f, "thin near object stays clear (missed column)",
			std::string("march missed it: ") + (evenMissed ? "yes" : "no") + ", fog " + std::to_string(onEven));

		// Probe (game ini Probe=1): the compute shader must report what the passes really did. Same sequence as
		// the game (march, probe phase 0, composite, probe phase 1) over a grey scene; its records are compared with
		// the composited scene read back here, the camera, and the blend formula the game log uses.
		{
			ID3DBlob* pcs = Compile(RaymarchShaders::kProbeCS, "cs_5_0");
			ID3D11ComputeShader*       probeCs = nullptr;
			ID3D11Buffer*              probeCb = nullptr;
			ID3D11Buffer*              probeBuf = nullptr;
			ID3D11Buffer*              probeSt = nullptr;
			ID3D11UnorderedAccessView* probeUav = nullptr;
			ID3D11ShaderResourceView*  sceneSrv = nullptr;
			constexpr int kPoints = RaymarchShaders::kProbeCols * RaymarchShaders::kProbeRows;
			constexpr int kRecords = kPoints * RaymarchShaders::kProbeSlots + RaymarchShaders::kProbeExtra;
			bool ok = pcs && SUCCEEDED(dev->CreateComputeShader(pcs->GetBufferPointer(), pcs->GetBufferSize(), nullptr, &probeCs));
			SafeRelease(pcs);
			D3D11_BUFFER_DESC pcbd{ 16, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
			ok = ok && SUCCEEDED(dev->CreateBuffer(&pcbd, nullptr, &probeCb));
			D3D11_BUFFER_DESC bdsc{ kRecords * 16u, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 16 };
			ok = ok && SUCCEEDED(dev->CreateBuffer(&bdsc, nullptr, &probeBuf));
			ok = ok && SUCCEEDED(dev->CreateUnorderedAccessView(probeBuf, nullptr, &probeUav));
			D3D11_BUFFER_DESC sdsc2{ kRecords * 16u, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 16 };
			ok = ok && SUCCEEDED(dev->CreateBuffer(&sdsc2, nullptr, &probeSt));
			ok = ok && SUCCEEDED(dev->CreateShaderResourceView(sceneTex, nullptr, &sceneSrv));
			Check(ok, "probe compute shader", ok ? "compiled, buffers created" : "creation failed");
			if (ok) {
				auto dispatch = [&](float a_phase) {
					const float phase[4]{ a_phase, 0, 0, 0 };
					ctx->UpdateSubresource(probeCb, 0, nullptr, phase, 0, 0);
					ctx->OMSetRenderTargets(0, nullptr, nullptr);
					ID3D11ShaderResourceView* in[6]{ depthSrv, sim.ColumnSrv(), lowSrv, sceneSrv, nullptr, testCoarse };
					ID3D11Buffer*             cbs[2]{ cb, probeCb };
					ctx->CSSetShader(probeCs, nullptr, 0);
					ctx->CSSetConstantBuffers(0, 2, cbs);
					ctx->CSSetShaderResources(0, 6, in);
					ctx->CSSetSamplers(0, 1, &linear);
					ctx->CSSetUnorderedAccessViews(0, 1, &probeUav, nullptr);
					ctx->Dispatch(1, 1, 1);
					ID3D11ShaderResourceView*  noSrv[6]{};
					ID3D11UnorderedAccessView* noUav = nullptr;
					ctx->CSSetShaderResources(0, 6, noSrv);
					ctx->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
				};
				// Plain ground: real depth of the z = 0 plane, so every point's scene height above ground is 0.
				const XMMATRIX vpg = setCamera({ 300.0f, 200.0f - 283.0f, 283.0f }, { 300.0f, 200.0f, 0.0f });
				const XMMATRIX invg = XMMatrixInverse(nullptr, vpg);
				for (UINT y = 0; y < kSize; ++y) {
					for (UINT x = 0; x < kSize; ++x) {
						// Ray through the pixel centre, hit with z = 0, project back for the depth.
						XMFLOAT4 a, b;
						XMStoreFloat4(&a, XMVector4Transform(XMVectorSet((x + 0.5f) / kSize * 2 - 1, 1 - (y + 0.5f) / kSize * 2, 0.5f, 1), invg));
						const XMFLOAT3 pnt{ a.x / a.w, a.y / a.w, a.z / a.w };   // relative to the camera
						const float    t = pnt.z < -1e-4f ? -cam.z / pnt.z : -1.0f;
						if (t <= 0.0f) {
							depthData[y * kSize + x] = 1.0f;
							continue;
						}
						XMStoreFloat4(&b, XMVector4Transform(XMVectorSet(pnt.x * t, pnt.y * t, pnt.z * t, 1), vpg));
						depthData[y * kSize + x] = b.z / b.w;
					}
				}
				ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);
				setCb(0.5f);
				marchInto(lowRtv, lowW);
				const float grey[4]{ 0.25f, 0.25f, 0.25f, 1.0f };
				ctx->ClearRenderTargetView(sceneRtv, grey);
				dispatch(0.0f);
				D3D11_VIEWPORT fullVp{ 0, 0, static_cast<float>(kSize), static_cast<float>(kSize), 0, 1 };
				ctx->RSSetViewports(1, &fullVp);
				ctx->OMSetRenderTargets(1, &sceneRtv, nullptr);
				ctx->OMSetBlendState(blend, black, 0xFFFFFFFF);
				ctx->PSSetShader(comp, nullptr, 0);
				ID3D11ShaderResourceView* compIn[3]{ depthSrv, sim.ColumnSrv(), lowSrv };
				ctx->PSSetShaderResources(0, 3, compIn);
				ctx->Draw(3, 0);
				ctx->PSSetShaderResources(0, 3, none);
				ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);
				dispatch(1.0f);
				ctx->CopyResource(probeSt, probeBuf);
				std::vector<float>       rec(static_cast<std::size_t>(kRecords) * 4);
				D3D11_MAPPED_SUBRESOURCE pm{};
				if (SUCCEEDED(ctx->Map(probeSt, 0, D3D11_MAP_READ, 0, &pm))) {
					std::memcpy(rec.data(), pm.pData, rec.size() * sizeof(float));
					ctx->Unmap(probeSt, 0);
				}
				// Scene read back on the CPU (red channel).
				D3D11_TEXTURE2D_DESC sdsc{};
				sceneTex->GetDesc(&sdsc);
				sdsc.Usage = D3D11_USAGE_STAGING;
				sdsc.BindFlags = 0;
				sdsc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
				ID3D11Texture2D* st = nullptr;
				dev->CreateTexture2D(&sdsc, nullptr, &st);
				ctx->CopyResource(st, sceneTex);
				std::vector<float>       red(static_cast<std::size_t>(kSize) * kSize, -1.0f);
				D3D11_MAPPED_SUBRESOURCE mm{};
				if (SUCCEEDED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &mm))) {
					for (UINT y = 0; y < kSize; ++y) {
						const auto* row = reinterpret_cast<const std::uint16_t*>(static_cast<const std::uint8_t*>(mm.pData) + y * mm.RowPitch);
						for (UINT x = 0; x < kSize; ++x) red[y * kSize + x] = HalfToFloat(row[x * 4]);
					}
					ctx->Unmap(st, 0);
				}
				SafeRelease(st);

				auto r = [&](int a_point, int a_slot) { return &rec[static_cast<std::size_t>((a_point * RaymarchShaders::kProbeSlots + a_slot) * 4)]; };
				float alphaErr = 0.0f, afterErr = 0.0f, blendErr = 0.0f, worstH = 0.0f, beforeErr = 0.0f, maxAlpha = 0.0f;
				int   ground = 0;
				for (int i = 0; i < kPoints; ++i) {
					const int   x = static_cast<int>(r(i, 0)[0]), y = static_cast<int>(r(i, 0)[1]);
					const float after = red[static_cast<std::size_t>(y) * kSize + x];
					const float a = r(i, 5)[3];
					maxAlpha = std::max(maxAlpha, a);
					alphaErr = std::max(alphaErr, std::fabs(a - (after - 0.25f) / 0.75f));
					afterErr = std::max(afterErr, std::fabs(r(i, 7)[0] - after));
					beforeErr = std::max(beforeErr, std::fabs(r(i, 6)[0] - 0.25f));
					blendErr = std::max(blendErr, std::fabs(r(i, 7)[0] - (a * r(i, 5)[0] + (1.0f - a) * r(i, 6)[0])));
					if (r(i, 0)[3] < 1e8f && r(i, 2)[3] > -1e8f) {
						++ground;
						worstH = std::max(worstH, std::fabs(r(i, 2)[2] - r(i, 2)[3]));
					}
				}
				constexpr int kCentre = (RaymarchShaders::kProbeRows / 2) * RaymarchShaders::kProbeCols + RaymarchShaders::kProbeCols / 2;
				const float*  cd = r(kCentre, 1);
				const float*  cw = r(kCentre, 2);
				// CPU ray through the centre point's pixel centre and its hit with z = 0.
				XMFLOAT4 ca;
				XMStoreFloat4(&ca, XMVector4Transform(XMVectorSet((r(kCentre, 0)[0] + 0.5f) / kSize * 2 - 1, 1 - (r(kCentre, 0)[1] + 0.5f) / kSize * 2, 0.5f, 1), invg));
				const XMVECTOR cdir = XMVector3Normalize(XMVectorSet(ca.x / ca.w, ca.y / ca.w, ca.z / ca.w, 0));
				XMFLOAT3 ed;
				XMStoreFloat3(&ed, cdir);
				const float tHit = -cam.z / ed.z;
				const float hx = cam.x + ed.x * tHit, hy = cam.y + ed.y * tHit;
				const float dirErr = std::max({ std::fabs(cd[0] - ed.x), std::fabs(cd[1] - ed.y), std::fabs(cd[2] - ed.z) });
				Check(alphaErr < 0.004f && maxAlpha > 0.05f, "probe alpha is the composited alpha",
					"max |probe - scene| " + std::to_string(alphaErr) + ", largest alpha " + std::to_string(maxAlpha));
				Check(afterErr < 0.002f && beforeErr < 0.002f && blendErr < 0.004f, "probe scene colours and blend check",
					"after off by " + std::to_string(afterErr) + ", before off by " + std::to_string(beforeErr) + ", blend formula off by " + std::to_string(blendErr));
				Check(dirErr < 1e-3f && std::fabs(cw[0] - hx) < 0.5f && std::fabs(cw[1] - hy) < 0.5f && ground > kPoints / 2 && worstH < 2.0f,
					"probe camera ray and ground height",
					"centre dir off by " + std::to_string(dirErr) + ", centre hit (" + std::to_string(cw[0]) + ", " + std::to_string(cw[1]) + ") vs CPU (" +
						std::to_string(hx) + ", " + std::to_string(hy) + "), worst |height above ground| " + std::to_string(worstH) + " over " + std::to_string(ground) + " points");

				// Height fog: the composite's closed form against numeric integration along each probe ray, with
				// the wind field off. Case A: the 45-degree camera over the ground plane, base above the ground (rays
				// cross the base). Case B: a level camera at the sky (scene distance infinite, capped at MaxDistance),
				// so near-level rays take the series branch.
				auto heightCase = [&](const char* a_name, float a_densityPerMetre, float a_falloff, float a_base, float a_maxDist) {
					hfCb[0] = a_densityPerMetre / 70.0f;
					hfCb[1] = 1.0f / a_falloff;
					hfCb[2] = a_base;
					hfCb[3] = a_maxDist;
					fieldOnCb = 0.0f;
					ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);
					setCb(0.5f);
					marchInto(lowRtv, lowW);
					dispatch(0.0f);
					ctx->CopyResource(probeSt, probeBuf);
					std::vector<float>       hr(static_cast<std::size_t>(kRecords) * 4);
					D3D11_MAPPED_SUBRESOURCE hm{};
					if (SUCCEEDED(ctx->Map(probeSt, 0, D3D11_MAP_READ, 0, &hm))) {
						std::memcpy(hr.data(), hm.pData, hr.size() * sizeof(float));
						ctx->Unmap(probeSt, 0);
					}
					auto hrec = [&](int a_point, int a_slot) { return &hr[static_cast<std::size_t>((a_point * RaymarchShaders::kProbeSlots + a_slot) * 4)]; };
					float worst = 0.0f, maxA = 0.0f, minA = 1.0f;
					for (int i = 0; i < kPoints; ++i) {
						const float* d = hrec(i, 1);
						const double L = std::min<double>(hrec(i, 0)[3], a_maxDist);
						constexpr int kN = 20000;
						double        tau = 0.0;
						for (int sIdx = 0; sIdx < kN; ++sIdx) {
							const double t = (sIdx + 0.5) * L / kN;
							const double z = cam.z + d[2] * t;
							tau += a_densityPerMetre / 70.0 * std::exp(-(z - a_base) / a_falloff) * (L / kN);
						}
						const float expect = static_cast<float>(1.0 - std::exp(-tau));
						const float got = hrec(i, 5)[3];
						worst = std::max(worst, std::fabs(got - expect));
						maxA = std::max(maxA, got);
						minA = std::min(minA, got);
					}
					Check(worst < 0.002f && maxA > 0.05f && maxA - minA > 0.02f, a_name,
						"worst |GPU - numeric| " + std::to_string(worst) + ", alpha " + std::to_string(minA) + ".." + std::to_string(maxA));
					hfCb[0] = hfCb[1] = hfCb[2] = hfCb[3] = 0.0f;
					fieldOnCb = 1.0f;
				};
				heightCase("height fog closed form (rays cross the base)", 0.5f, 150.0f, 100.0f, 100000.0f);
				setCamera({ 300.0f, -900.0f, 50.0f }, { 300.0f, 2000.0f, 50.0f });
				std::fill(depthData.begin(), depthData.end(), 1.0f);
				heightCase("height fog closed form (sky, level rays, capped distance)", 0.05f, 400.0f, 0.0f, 3000.0f);

				// Far tier: the composite's layer on the depth-estimated ground plane against numeric integration of
				// the same model (ground plane through the scene point, last FlatLength units, column profile), wind
				// field off, no noise. A flat plane, a 10% slope, and a vertical wall that must carry none.
				{
					const float kFarH = 30.0f, kFlat = 4000.0f;   // thin, so alpha stays mid-range and errors show
					const XMMATRIX vpf = setCamera({ 300.0f, 200.0f - 283.0f, 283.0f }, { 300.0f, 200.0f, 0.0f });
					// Surface: ground z = a x + b y + c, plus an optional wall at y = wallY (facing the camera).
					auto fillDepth = [&](float a_a, float a_b, float a_c, float a_wallY, const XMMATRIX& a_vp) {
						const XMMATRIX a_inv = XMMatrixInverse(nullptr, a_vp);
						for (UINT y = 0; y < kSize; ++y) {
							for (UINT x = 0; x < kSize; ++x) {
								XMFLOAT4 q;
								XMStoreFloat4(&q, XMVector4Transform(XMVectorSet((x + 0.5f) / kSize * 2 - 1, 1 - (y + 0.5f) / kSize * 2, 0.5f, 1), a_inv));
								const XMFLOAT3 d{ q.x / q.w, q.y / q.w, q.z / q.w };   // relative to the camera
								const float den = d.z - a_a * d.x - a_b * d.y;
								float t = std::fabs(den) > 1e-6f ? (a_a * cam.x + a_b * cam.y + a_c - cam.z) / den : -1.0f;
								if (a_wallY > 0.0f && d.y > 1e-6f) {
									const float tw = (a_wallY - cam.y) / d.y;
									if (tw > 0.0f && (t <= 0.0f || tw < t)) t = tw;
								}
								if (t <= 0.0f) {
									depthData[y * kSize + x] = 1.0f;
									continue;
								}
								XMFLOAT4 b;
								XMStoreFloat4(&b, XMVector4Transform(XMVectorSet(d.x * t, d.y * t, d.z * t, 1), a_vp));
								depthData[y * kSize + x] = b.z / b.w;
							}
						}
					};
					auto runFar = [&]() {
						farCb[0] = 1.0f;
						farCb[1] = kFlat;
						farCb[2] = kFarH;
						farCb[3] = 0.0f;
						fieldOnCb = 0.0f;
						ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);
						setCb(0.5f);
						marchInto(lowRtv, lowW);
						dispatch(0.0f);
						ctx->CopyResource(probeSt, probeBuf);
						std::vector<float>       fr(static_cast<std::size_t>(kRecords) * 4);
						D3D11_MAPPED_SUBRESOURCE fm{};
						if (SUCCEEDED(ctx->Map(probeSt, 0, D3D11_MAP_READ, 0, &fm))) {
							std::memcpy(fr.data(), fm.pData, fr.size() * sizeof(float));
							ctx->Unmap(probeSt, 0);
						}
						farCb[0] = farCb[1] = farCb[2] = farCb[3] = 0.0f;
						fieldOnCb = 1.0f;
						return fr;
					};
					auto fRec = [&](const std::vector<float>& a_r, int a_point, int a_slot) {
						return &a_r[static_cast<std::size_t>((a_point * RaymarchShaders::kProbeSlots + a_slot) * 4)];
					};
					// Numeric tau of the model for a ray: vertical height above the plane z = a x + b y + c.
					auto expectTau = [&](const float* a_dir, float a_dist, float a_a, float a_b, float a_c) {
						constexpr int kN = 20000;
						const double  t0 = std::max(0.0, static_cast<double>(a_dist) - kFlat);
						double        tau = 0.0;
						for (int i = 0; i < kN; ++i) {
							const double t = t0 + (i + 0.5) * (a_dist - t0) / kN;
							const double x = cam.x + a_dir[0] * t, y = cam.y + a_dir[1] * t, z = cam.z + a_dir[2] * t;
							const double h = z - (a_a * x + a_b * y + a_c);
							if (h <= 0.0 || h >= kFarH) continue;
							tau += kOpacity * g_lowProfile.Shape(h / kFarH) * ((a_dist - t0) / kN) / 70.0;   // W / (H / 70) = 1
						}
						return tau;
					};
					auto planeCase = [&](const char* a_name, float a_a, float a_b, float a_c) {
						fillDepth(a_a, a_b, a_c, -1.0f, vpf);
						const auto fr = runFar();
						float      worst = 0.0f, maxA = 0.0f, minA = 1.0f;
						int        n = 0;
						for (int i = 0; i < kPoints; ++i) {
							const float* r0 = fRec(fr, i, 0);
							if (!(r0[3] < 1e8f)) continue;
							// Pixels on the screen border have a neighbour off screen on one side; still a plane.
							const float expect = static_cast<float>(1.0 - std::exp(-expectTau(fRec(fr, i, 1), r0[3], a_a, a_b, a_c)));
							worst = std::max(worst, std::fabs(fRec(fr, i, 5)[3] - expect));
							maxA = std::max(maxA, fRec(fr, i, 5)[3]);
							minA = std::min(minA, fRec(fr, i, 5)[3]);
							++n;
						}
						Check(n > kPoints / 2 && worst < 0.002f && maxA > 0.05f && minA < 0.9f, a_name,
							"worst |GPU - numeric| " + std::to_string(worst) + " over " + std::to_string(n) + " points, alpha " + std::to_string(minA) + ".." + std::to_string(maxA));
					};
					planeCase("far tier on flat ground", 0.0f, 0.0f, 0.0f);
					planeCase("far tier on a 10% slope", 0.0f, 0.1f, -20.0f);
					// Slope share (RaymarchShaders.h kFarSlopeNone 0.74, kFarSlopeFull 0.94; hatching and frost on far roofs,
					// report 2026-11-11): a slope carries smoothstep(0.74, 0.94, normal.z) of the layer, so 30 degrees keeps
					// 0.69 of the optical depth and 45 degrees none (it carried all of it under the old 0.7 cut).
					auto slopeCase = [&](const char* a_name, float a_b, float a_c, float a_share) {
						fillDepth(0.0f, a_b, a_c, -1.0f, vpf);
						const auto fr = runFar();
						float      worst = 0.0f, maxA = 0.0f;
						int        n = 0;
						for (int i = 0; i < kPoints; ++i) {
							const float* r0 = fRec(fr, i, 0);
							if (!(r0[3] < 1e8f)) continue;
							const float expect = static_cast<float>(1.0 - std::exp(-a_share * expectTau(fRec(fr, i, 1), r0[3], 0.0f, a_b, a_c)));
							worst = std::max(worst, std::fabs(fRec(fr, i, 5)[3] - expect));
							maxA = std::max(maxA, fRec(fr, i, 5)[3]);
							++n;
						}
						Check(n > kPoints / 2 && worst < 0.004f && (a_share > 0.0f ? maxA > 0.02f : maxA == 0.0f), a_name,
							"share " + std::to_string(a_share) + ", worst |GPU - numeric| " + std::to_string(worst) + " over " + std::to_string(n) + " points, alpha up to " + std::to_string(maxA));
					};
					{
						const float nz30 = std::cos(XMConvertToRadians(30.0f));
						const float x30 = (nz30 - 0.74f) / (0.94f - 0.74f);
						slopeCase("far tier fades on a 30-degree slope", std::tan(XMConvertToRadians(30.0f)), -20.0f, x30 * x30 * (3.0f - 2.0f * x30));
						slopeCase("far tier leaves a 45-degree slope clear", 1.0f, -20.0f, 0.0f);
					}					// Wall at y = 330 in front of the 45-degree camera: wall pixels (distance to the wall plane) carry none.
					fillDepth(0.0f, 0.0f, 0.0f, 330.0f, vpf);
					{
						const auto fr = runFar();
						int        onWall = 0, wallFog = 0, foot = 0, ground = 0, groundFog = 0;
						float      maxFoot = 0.0f, maxGround = 0.0f;
						for (int i = 0; i < kPoints; ++i) {
							const float* r0 = fRec(fr, i, 0);
							const float* w = fRec(fr, i, 2);   // scene point, world
							if (!(r0[3] < 1e8f)) continue;
							// The wall's foot row (a few units up) lies inside the layer and may carry fog either way (since the silhouette
							// fix it takes the wall side and carries none); above 20 units it must not. Every point off the wall must.
							const bool onPlane = std::fabs(w[1] - 330.0f) < 1.0f;
							const float a = fRec(fr, i, 5)[3];
							if (onPlane && w[2] > 20.0f) {
								++onWall;
								wallFog += a > 1e-4f ? 1 : 0;
							} else if (onPlane) {
								++foot;
								maxFoot = std::max(maxFoot, a);
							} else {
								++ground;
								groundFog += a > 0.01f ? 1 : 0;
								maxGround = std::max(maxGround, a);
							}
						}
						Check(onWall >= 5 && wallFog == 0 && ground >= 20 && groundFog == ground && maxFoot <= maxGround, "far tier leaves walls clear",
							std::to_string(onWall) + " wall points (" + std::to_string(wallFog) + " fogged), " + std::to_string(foot) + " at its foot (alpha up to " + std::to_string(maxFoot) + "), " +
								std::to_string(ground) + " ground points (" + std::to_string(groundFog) + " fogged)");
					}

					// Silhouette (Nexus report 2026-11-10, white lines along far ridges and roofs): a plateau top z = 80 fills
					// rows >= kEdgeRow and flat ground z = 0 beyond it shows above. The ground pixels just above the edge have
					// the plateau 2 rows down, nearer by less than the 35% jump test; a plane through both lies along the ray,
					// which then crosses the layer for the whole FlatLength window. They must carry the ground's fog, as the
					// plateau pixels carry the plateau's.
					{
						constexpr UINT  kEdgeRow = 50;   // probe row 48 sits 2 rows above it
						constexpr float kTop = 80.0f;
						const XMMATRIX  vps = setCamera({ 300.0f, -9000.0f, 2000.0f }, { 300.0f, 1000.0f, 0.0f });
						const XMMATRIX  invS = XMMatrixInverse(nullptr, vps);
						// Distance from the camera to the scene point of pixel (x, y), -1 for sky.
						auto sceneDist = [&](UINT a_x, UINT a_y, XMFLOAT3& a_d) {
							XMFLOAT4 q;
							XMStoreFloat4(&q, XMVector4Transform(XMVectorSet((a_x + 0.5f) / kSize * 2 - 1, 1 - (a_y + 0.5f) / kSize * 2, 0.5f, 1), invS));
							a_d = { q.x / q.w, q.y / q.w, q.z / q.w };
							const float zPlane = a_y >= kEdgeRow ? kTop : 0.0f;
							return a_d.z < -1e-6f ? (zPlane - cam.z) / a_d.z : -1.0f;   // in units of a_d
						};
						for (UINT y = 0; y < kSize; ++y) {
							for (UINT x = 0; x < kSize; ++x) {
								XMFLOAT3    d;
								const float t = sceneDist(x, y, d);
								if (t <= 0.0f) {
									depthData[y * kSize + x] = 1.0f;
									continue;
								}
								XMFLOAT4 b;
								XMStoreFloat4(&b, XMVector4Transform(XMVectorSet(d.x * t, d.y * t, d.z * t, 1), vps));
								depthData[y * kSize + x] = b.z / b.w;
							}
						}
						float jump = 0.0f;   // depth jump across the edge, centre column: must be inside the 35% test
						{
							XMFLOAT3 d;
							const float ta = sceneDist(48, 48, d);
							const float la = ta * std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
							const float tb = sceneDist(48, kEdgeRow, d);
							const float lb = tb * std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
							jump = la / lb - 1.0f;
						}
						const auto fr = runFar();
						float      worstEdge = 0.0f, worstTop = 0.0f, maxEdge = 0.0f;
						int        nEdge = 0, nTop = 0;
						for (int i = 0; i < kPoints; ++i) {
							const float* r0 = fRec(fr, i, 0);
							if (!(r0[3] < 1e8f)) continue;
							const int   row = static_cast<int>(r0[1]);
							const float got = fRec(fr, i, 8)[1];   // the far tier's optical depth (alpha saturates at this angle)
							if (row == 48) {
								const float expect = static_cast<float>(expectTau(fRec(fr, i, 1), r0[3], 0.0f, 0.0f, 0.0f));
								worstEdge = std::max(worstEdge, std::fabs(got - expect) / std::max(expect, 1e-3f));
								maxEdge = std::max(maxEdge, got);
								++nEdge;
							} else if (row > static_cast<int>(kEdgeRow) + 4) {
								const float expect = static_cast<float>(expectTau(fRec(fr, i, 1), r0[3], 0.0f, 0.0f, kTop));
								worstTop = std::max(worstTop, std::fabs(got - expect) / std::max(expect, 1e-3f));
								++nTop;
							}
						}
						Check(nEdge >= 5 && nTop >= 5 && jump > 0.05f && jump < 0.35f && worstEdge < 0.02f && worstTop < 0.02f,
							"far tier ignores a nearer ridge below a silhouette",
							"jump " + std::to_string(jump) + ", edge row worst relative tau error " + std::to_string(worstEdge) + " (tau up to " + std::to_string(maxEdge) + ", " +
								std::to_string(nEdge) + " points), plateau worst relative " + std::to_string(worstTop) + " (" + std::to_string(nTop) + " points)");
						setCamera({ 300.0f, 200.0f - 283.0f, 283.0f }, { 300.0f, 200.0f, 0.0f });
					}

					// Near the horizon on flat ground (Codex review 2026-11-10): probe row 61 lies 5 px under the horizon, where
					// the point 2 px up is 67% farther (past the jump test) and the point 2 px down 29% nearer. Both sides continue
					// the plane, so the side must be chosen among those that pass the jump test, or the row loses its fog.
					{
						const XMMATRIX vph = setCamera({ 300.0f, -900.0f, 100.0f }, { 300.0f, 100.0f, 224.0f });
						fillDepth(0.0f, 0.0f, 0.0f, -1.0f, vph);
						const auto fr = runFar();
						float      worst = 0.0f, minRow = 1.0f;
						int        n = 0, nRow = 0;
						for (int i = 0; i < kPoints; ++i) {
							const float* r0 = fRec(fr, i, 0);
							if (!(r0[3] < 1e8f)) continue;
							const float got = fRec(fr, i, 5)[3];
							const float expect = static_cast<float>(1.0 - std::exp(-expectTau(fRec(fr, i, 1), r0[3], 0.0f, 0.0f, 0.0f)));
							worst = std::max(worst, std::fabs(got - expect));
							++n;
							if (static_cast<int>(r0[1]) == 61) {
								minRow = std::min(minRow, got);
								++nRow;
							}
						}
						Check(n >= 20 && nRow >= 5 && minRow > 0.05f && worst < 0.002f, "far tier keeps its fog just under the horizon",
							"worst |GPU - numeric| " + std::to_string(worst) + " over " + std::to_string(n) + " points, row 61 alpha from " + std::to_string(minRow) +
								" (" + std::to_string(nRow) + " points)");
						setCamera({ 300.0f, 200.0f - 283.0f, 283.0f }, { 300.0f, 200.0f, 0.0f });
					}

					// Seam: with the field on, the far share is 1 - the field's round fade at each point of the ray.
					// Camera near the field edge so the scene points lie across the fade (r 0.55..0.95). The far
					// optical depth (probe slot 8) must match numeric integration with the share taken per point.
					{
						const XMMATRIX vps = setCamera({ 1500.0f, -150.0f, 283.0f }, { 1500.0f, 150.0f, 0.0f });
						fillDepth(0.0f, 0.0f, 0.0f, -1.0f, vps);
						farCb[0] = 1.0f;
						farCb[1] = kFlat;
						farCb[2] = kFarH;
						farCb[3] = 0.0f;
						fieldOnCb = 1.0f;
						ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);
						setCb(0.5f);
						marchInto(lowRtv, lowW);
						dispatch(0.0f);
						ctx->CopyResource(probeSt, probeBuf);
						std::vector<float>       sr(static_cast<std::size_t>(kRecords) * 4);
						D3D11_MAPPED_SUBRESOURCE sm{};
						if (SUCCEEDED(ctx->Map(probeSt, 0, D3D11_MAP_READ, 0, &sm))) {
							std::memcpy(sr.data(), sm.pData, sr.size() * sizeof(float));
							ctx->Unmap(probeSt, 0);
						}
						farCb[0] = farCb[1] = farCb[2] = farCb[3] = 0.0f;
						auto share = [&](double a_x, double a_y) {
							const double ux = (a_x - kOrigin) / kExtent - 0.5, uy = (a_y - kOrigin) / kExtent - 0.5;
							const double r = std::sqrt(ux * ux + uy * uy) * 2.0;
							const double t = std::clamp((r - 0.55) / 0.4, 0.0, 1.0);
							return t * t * (3.0 - 2.0 * t);
						};
						float worst = 0.0f, minS = 1.0f, maxS = 0.0f;
						int   n = 0;
						for (int i = 0; i < kPoints; ++i) {
							const float* r0 = fRec(sr, i, 0);
							const float* d = fRec(sr, i, 1);
							const float* r8 = fRec(sr, i, 8);
							if (!(r0[3] < 1e8f) || r8[0] < 0.7f) continue;
							constexpr int kN = 20000;
							const double  dist = r0[3];
							const double  t0 = std::max(0.0, dist - kFlat);
							double        tau = 0.0;
							for (int k = 0; k < kN; ++k) {
								const double t = t0 + (k + 0.5) * (dist - t0) / kN;
								const double h = cam.z + d[2] * t;
								if (h <= 0.0 || h >= kFarH) continue;
								tau += share(cam.x + d[0] * t, cam.y + d[1] * t) * kOpacity * g_lowProfile.Shape(h / kFarH) * ((dist - t0) / kN) / 70.0;
							}
							const float sP = static_cast<float>(share(fRec(sr, i, 2)[0], fRec(sr, i, 2)[1]));
							minS = std::min(minS, sP);
							maxS = std::max(maxS, sP);
							worst = std::max(worst, static_cast<float>(std::fabs(r8[1] - tau) / std::max(tau, 0.05)));
							++n;
						}
						Check(n > kPoints / 2 && minS < 0.3f && maxS > 0.7f && worst < 0.02f, "far tier share across the field edge",
							"worst relative |GPU - numeric| " + std::to_string(worst) + " over " + std::to_string(n) + " points, share at the scene points " +
								std::to_string(minS) + ".." + std::to_string(maxS));
					}

					// Far pools (coarse grid, t5): the composite's optical depth (probe slot 1 w) against numeric integration
					// of the model: fog under the coarse lid, the tall layer's profile (a = 1) above the coarse ground, the density at
					// the ground below it, share 1 - the field's round fade, grid edge fade over two cells.
					{
						constexpr int   kCs = 17;
						constexpr float kCell = 1024.0f, kCo = -8192.0f, kDen = 0.02f, kCap = 600.0f;
						auto makeCoarse = [&](float a_a, float a_b, float a_c, float a_lid, bool a_unknown) {
							std::vector<float> rg(static_cast<std::size_t>(kCs) * kCs * 2);
							for (int y = 0; y < kCs; ++y)
								for (int x = 0; x < kCs; ++x) {
									const float g = a_a * (kCo + x * kCell) + a_b * (kCo + y * kCell) + a_c;
									const std::size_t i = (static_cast<std::size_t>(y) * kCs + x) * 2;
									rg[i] = a_unknown ? -1.0e30f : g;
									rg[i + 1] = a_unknown ? 0.0f : std::max(a_lid - g, 0.0f);
								}
							D3D11_TEXTURE2D_DESC cd{};
							cd.Width = cd.Height = kCs;
							cd.MipLevels = cd.ArraySize = 1;
							cd.Format = DXGI_FORMAT_R32G32_FLOAT;
							cd.SampleDesc.Count = 1;
							cd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
							D3D11_SUBRESOURCE_DATA ci{ rg.data(), kCs * sizeof(float) * 2, 0 };
							ID3D11Texture2D*       ct = nullptr;
							dev->CreateTexture2D(&cd, &ci, &ct);
							dev->CreateShaderResourceView(ct, nullptr, &testCoarse);
							SafeRelease(ct);
						};
						auto shareAt = [&](double a_x, double a_y) {
							if (fieldOnCb < 0.5f) return 1.0;
							const double ux = (a_x - kOrigin) / kExtent - 0.5, uy = (a_y - kOrigin) / kExtent - 0.5;
							const double t = std::clamp((std::sqrt(ux * ux + uy * uy) * 2.0 - 0.55) / 0.4, 0.0, 1.0);
							return t * t * (3.0 - 2.0 * t);
						};
						// a_a, a_b, a_c: coarse ground plane; a_sa, a_sb, a_sc: scene ground plane (depth buffer).
						auto poolCase = [&](const char* a_name, float a_a, float a_b, float a_c, float a_lid, float a_sa, float a_sb, float a_sc,
											float a_tol, bool a_relative, bool a_unknown) {
							makeCoarse(a_a, a_b, a_c, a_lid, a_unknown);
							poolGCb[0] = poolGCb[1] = kCo;
							poolGCb[2] = kCell;
							poolGCb[3] = static_cast<float>(kCs);
							poolFCb[0] = kDen;
							poolFCb[1] = kCap;
							poolFCb[2] = a_lid;
							const XMMATRIX vpp = fieldOnCb > 0.5f ? setCamera({ 1500.0f, -150.0f, 283.0f }, { 1500.0f, 150.0f, 0.0f })
							                                      : setCamera({ 300.0f, 200.0f - 283.0f, 283.0f }, { 300.0f, 200.0f, 0.0f });
							fillDepth(a_sa, a_sb, a_sc, -1.0f, vpp);
							ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);
							setCb(0.5f);
							marchInto(lowRtv, lowW);
							dispatch(0.0f);
							ctx->CopyResource(probeSt, probeBuf);
							std::vector<float>       pr(static_cast<std::size_t>(kRecords) * 4);
							D3D11_MAPPED_SUBRESOURCE pm{};
							if (SUCCEEDED(ctx->Map(probeSt, 0, D3D11_MAP_READ, 0, &pm))) {
								std::memcpy(pr.data(), pm.pData, pr.size() * sizeof(float));
								ctx->Unmap(probeSt, 0);
							}
							std::fill(std::begin(poolGCb), std::end(poolGCb), 0.0f);
							std::fill(std::begin(poolFCb), std::end(poolFCb), 0.0f);
							SafeRelease(testCoarse);
							const double last = kCs - 1.0;
							float        worst = 0.0f, maxTau = 0.0f;
							int          n = 0;
							bool         finite = true;
							for (int i = 0; i < kPoints; ++i) {
								const float* r0 = fRec(pr, i, 0);
								const float* d = fRec(pr, i, 1);
								if (!(r0[3] < 1e8f)) continue;
								finite = finite && std::isfinite(d[3]);
								constexpr int kN = 20000;
								const double  dist = r0[3];
								double        tau = 0.0;
								for (int k = 0; k < kN; ++k) {
									const double t = (k + 0.5) * dist / kN;
									const double x = cam.x + d[0] * t, y = cam.y + d[1] * t, z = cam.z + d[2] * t;
									const double gx = (x - kCo) / kCell, gy = (y - kCo) / kCell;
									if (a_unknown || gx < 0.0 || gy < 0.0 || gx > last || gy > last) continue;
									const double g = a_a * x + a_b * y + a_c;
									const double H = std::min(static_cast<double>(a_lid) - g, static_cast<double>(kCap));
									if (H < 1.0) continue;
									const double h = z - g;
									if (h >= H) continue;
									const double fade = std::clamp(std::min({ gx, gy, last - gx, last - gy }) / 2.0, 0.0, 1.0);
									const double rho = kDen * g_upperProfile.Shape(h < 0.0 ? 0.0 : h / H);
									tau += kOpacity * rho * (dist / kN) / 70.0 * shareAt(x, y) * fade;
								}
								const float err = a_relative ? static_cast<float>(std::fabs(d[3] - tau) / std::max(tau, 0.05)) : static_cast<float>(std::fabs((1.0 - std::exp(-d[3])) - (1.0 - std::exp(-tau))));
								worst = std::max(worst, err);
								maxTau = std::max(maxTau, d[3]);
								++n;
							}
							const bool sized = a_unknown ? maxTau == 0.0f : maxTau > 0.05f;
							Check(finite && n > kPoints / 2 && sized && worst < a_tol, a_name,
								std::string(a_relative ? "worst relative" : "worst alpha") + " |GPU - numeric| " + std::to_string(worst) + " over " + std::to_string(n) +
									" points, largest tau " + std::to_string(maxTau));
						};
						fieldOnCb = 0.0f;
						poolCase("far pools fill to a level lid", 0.0f, 0.0f, 0.0f, 150.0f, 0.0f, 0.0f, 0.0f, 0.002f, false, false);
						poolCase("far pools count air below the coarse ground", 0.0f, 0.0f, 40.0f, 200.0f, 0.0f, 0.0f, 0.0f, 0.002f, false, false);
						poolCase("far pools on a slope keep a level lid", 0.0f, 0.05f, 0.0f, 300.0f, 0.0f, 0.05f, 0.0f, 0.02f, false, false);
						poolCase("far pools none next to unknown cells", 0.0f, 0.0f, 0.0f, 150.0f, 0.0f, 0.0f, 0.0f, 1e-6f, false, true);
						fieldOnCb = 1.0f;
						poolCase("far pools hand over at the field edge", 0.0f, 0.0f, 0.0f, 150.0f, 0.0f, 0.0f, 0.0f, 0.03f, true, false);
						// Field on, general camera: far-plane pixels included (numeric to the grid edge), a depth cap, a raised
						// coarse ground.
						auto poolCase2 = [&](const char* a_name, XMFLOAT3 a_cam, XMFLOAT3 a_target, float a_cGround, float a_lid, float a_cap,
											 float a_tol, bool a_expectNone) {
							makeCoarse(0.0f, 0.0f, a_cGround, a_lid, false);
							poolGCb[0] = poolGCb[1] = kCo;
							poolGCb[2] = kCell;
							poolGCb[3] = static_cast<float>(kCs);
							poolFCb[0] = kDen;
							poolFCb[1] = a_cap;
							poolFCb[2] = a_lid;
							const XMMATRIX vpp = setCamera(a_cam, a_target);
							fillDepth(0.0f, 0.0f, 0.0f, -1.0f, vpp);
							ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);
							setCb(0.5f);
							marchInto(lowRtv, lowW);
							dispatch(0.0f);
							ctx->CopyResource(probeSt, probeBuf);
							std::vector<float>       pr(static_cast<std::size_t>(kRecords) * 4);
							D3D11_MAPPED_SUBRESOURCE pm{};
							if (SUCCEEDED(ctx->Map(probeSt, 0, D3D11_MAP_READ, 0, &pm))) {
								std::memcpy(pr.data(), pm.pData, pr.size() * sizeof(float));
								ctx->Unmap(probeSt, 0);
							}
							std::fill(std::begin(poolGCb), std::end(poolGCb), 0.0f);
							std::fill(std::begin(poolFCb), std::end(poolFCb), 0.0f);
							SafeRelease(testCoarse);
							const double last = kCs - 1.0, lo = kCo, hi = kCo + kCell * last;
							float        worst = 0.0f, maxTau = 0.0f;
							int          n = 0, sky = 0;
							bool         finite = true;
							for (int i = 0; i < kPoints; ++i) {
								const float* r0 = fRec(pr, i, 0);
								const float* d = fRec(pr, i, 1);
								finite = finite && std::isfinite(d[3]);
								// Length to the grid's xy edge (sky pixels end there).
								double tEdge = 1e30;
								for (int ax = 0; ax < 2; ++ax) {
									const double o = ax == 0 ? cam.x : cam.y, v = d[ax];
									if (std::fabs(v) > 1e-9) tEdge = std::min(tEdge, ((v > 0 ? hi : lo) - o) / v);
								}
								const double dist = std::min(static_cast<double>(r0[3]), tEdge);
								sky += r0[3] > tEdge ? 1 : 0;   // the ray reaches the grid edge before its scene point
								constexpr int kN = 40000;
								double        tau = 0.0;
								for (int k = 0; k < kN; ++k) {
									const double t = (k + 0.5) * dist / kN;
									const double x = cam.x + d[0] * t, y = cam.y + d[1] * t, z = cam.z + d[2] * t;
									const double gx = (x - kCo) / kCell, gy = (y - kCo) / kCell;
									if (gx < 0.0 || gy < 0.0 || gx > last || gy > last) continue;
									const double H = std::min(static_cast<double>(a_lid - a_cGround), static_cast<double>(a_cap));
									if (H < 1.0) continue;
									const double h = z - a_cGround;
									if (h >= H) continue;
									const double fade = std::clamp(std::min({ gx, gy, last - gx, last - gy }) / 2.0, 0.0, 1.0);
									const double share = shareAt(x, y);
									const double rho = kDen * g_upperProfile.Shape(h < 0.0 ? 0.0 : h / H);
									tau += kOpacity * rho * (dist / kN) / 70.0 * share * fade;
								}
								worst = std::max(worst, static_cast<float>(std::fabs(d[3] - tau) / std::max(tau, 0.05)));
								maxTau = std::max(maxTau, d[3]);
								++n;
							}
							const bool sized = a_expectNone ? maxTau == 0.0f : maxTau > 0.05f;
							Check(finite && sized && worst < a_tol, a_name,
								"worst relative |GPU - numeric| " + std::to_string(worst) + " over " + std::to_string(n) + " points (" + std::to_string(sky) +
									" past the grid edge), largest tau " + std::to_string(maxTau));
						};
						// Eye inside a 280-unit pool at the field centre, rising rays leave the lid about 3600 units out, inside the grid.
						poolCase2("far pools end at the lid on rising rays", { 0.0f, 0.0f, 100.0f }, { 4000.0f, 0.0f, 300.0f }, 0.0f, 280.0f, 600.0f, 0.006f, false);   // without the lid cut: 0.010 (measured with the old profile, not re-derived)
						// Coarse ground 40 over the scene ground 0, so rays pass below it (the "below" term would run with H = 0).
						poolCase2("far pools off with a depth cap of 0", { 1500.0f, -150.0f, 283.0f }, { 1500.0f, 150.0f, 0.0f }, 40.0f, 280.0f, 0.0f, 1e-6f, true);
						// Coarse ground 200 over a scene ground of 0 (a cliff between coarse samples): the rays end below the coarse
						// ground, where the density at the ground counts, with the field's share across its edge.
						poolCase2("far pools below a raised coarse ground at the field edge", { 1500.0f, -150.0f, 283.0f }, { 1500.0f, 150.0f, 0.0f }, 200.0f, 300.0f, 600.0f, 0.03f, false);
						// Cliff: cells at x <= 0 hold a 200-unit pool (ground 0, lid 200), cells beyond a cliff 1000 units up hold
						// none. Rays that stay above z = 300 must carry no fog (a lid mixed with the cliff top's ground rose to
						// 200 + 800 f across the cliff cell; reviewer 2026-11-08); rays coming down into the pool must.
						{
							std::vector<float> rg(static_cast<std::size_t>(kCs) * kCs * 2);
							for (int y = 0; y < kCs; ++y)
								for (int x = 0; x < kCs; ++x) {
									const bool        low = kCo + x * kCell <= 0.0f;
									const std::size_t i = (static_cast<std::size_t>(y) * kCs + x) * 2;
									rg[i] = low ? 0.0f : 1000.0f;
									rg[i + 1] = low ? 200.0f : 0.0f;
								}
							D3D11_TEXTURE2D_DESC cd{};
							cd.Width = cd.Height = kCs;
							cd.MipLevels = cd.ArraySize = 1;
							cd.Format = DXGI_FORMAT_R32G32_FLOAT;
							cd.SampleDesc.Count = 1;
							cd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
							D3D11_SUBRESOURCE_DATA ci{ rg.data(), kCs * sizeof(float) * 2, 0 };
							ID3D11Texture2D*       ct = nullptr;
							dev->CreateTexture2D(&cd, &ci, &ct);
							dev->CreateShaderResourceView(ct, nullptr, &testCoarse);
							SafeRelease(ct);
							poolGCb[0] = poolGCb[1] = kCo;
							poolGCb[2] = kCell;
							poolGCb[3] = static_cast<float>(kCs);
							poolFCb[0] = kDen;
							poolFCb[1] = kCap;
							poolFCb[2] = 1200.0f;   // a higher pool elsewhere: the highest-lid clip must not be what keeps these rays clear
							fieldOnCb = 0.0f;
							const XMMATRIX vpp = setCamera({ -3000.0f, 0.0f, 300.0f }, { 3000.0f, 0.0f, 420.0f });
							fillDepth(0.0f, 0.0f, 0.0f, -1.0f, vpp);
							ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);
							setCb(0.5f);
							marchInto(lowRtv, lowW);
							dispatch(0.0f);
							ctx->CopyResource(probeSt, probeBuf);
							std::vector<float>       pr(static_cast<std::size_t>(kRecords) * 4);
							D3D11_MAPPED_SUBRESOURCE pm{};
							if (SUCCEEDED(ctx->Map(probeSt, 0, D3D11_MAP_READ, 0, &pm))) {
								std::memcpy(pr.data(), pm.pData, pr.size() * sizeof(float));
								ctx->Unmap(probeSt, 0);
							}
							std::fill(std::begin(poolGCb), std::end(poolGCb), 0.0f);
							std::fill(std::begin(poolFCb), std::end(poolFCb), 0.0f);
							SafeRelease(testCoarse);
							fieldOnCb = 1.0f;
							float maxAbove = 0.0f, maxDown = 0.0f;
							int   above = 0;
							bool  finite = true;
							for (int i = 0; i < kPoints; ++i) {
								const float* d = fRec(pr, i, 1);
								finite = finite && std::isfinite(d[3]);
								if (d[2] >= 0.0f) {
									maxAbove = std::max(maxAbove, d[3]);
									++above;
								} else {
									maxDown = std::max(maxDown, d[3]);
								}
							}
							Check(finite && above >= 5 && maxAbove < 1e-6f && maxDown > 0.01f, "far pools stay under the lid next to a cliff",
								"largest tau on " + std::to_string(above) + " rays above the lid " + std::to_string(maxAbove) + ", on rays coming down " +
									std::to_string(maxDown));
						}
					}
				}
			}
			SafeRelease(probeCs);
			SafeRelease(probeCb);
			SafeRelease(probeUav);
			SafeRelease(probeBuf);
			SafeRelease(probeSt);
			SafeRelease(sceneSrv);
		}

		// Case 2: low camera inside the fog looking along flat ground toward the horizon (real ground-plane depth).
		// Near the horizon the distance changes by more than 15% per pixel; those pixels must still get their fog.
		const XMMATRIX vp2 = setCamera({ 300.0f, -900.0f, 50.0f }, { 300.0f, 2000.0f, -100.0f });
		for (UINT y = 0; y < kSize; ++y) {
			for (UINT x = 0; x < kSize; ++x) {
				const XMVECTOR ndc = XMVectorSet((x + 0.5f) / kSize * 2 - 1, 1 - (y + 0.5f) / kSize * 2, 0.5f, 1);
				XMFLOAT4       w;
				XMStoreFloat4(&w, XMVector4Transform(ndc, XMLoadFloat4x4(&inv)));
				const XMFLOAT3 dir{ w.x / w.w, w.y / w.w, w.z / w.w };
				float          d = 1.0f;
				if (dir.z < -1e-6f) {
					const float t = -cam.z / dir.z;
					XMFLOAT4    clip;
					XMStoreFloat4(&clip, XMVector4Transform(XMVectorSet(dir.x * t, dir.y * t, dir.z * t, 1), vp2));
					d = clip.z / clip.w;
				}
				depthData[y * kSize + x] = d;
			}
		}
		const auto ref = fullMarch();
		const auto horizon = twoPass(low);
		int holes = 0, fogPixels = 0, unmatched = 0;
		for (UINT y = 0; y < kSize; ++y) {
			for (UINT x = 0; x < kSize; ++x) {
				const float r = ref[y * kSize + x];
				if (r < 0.1f) continue;
				++fogPixels;
				unmatched += std::isnan(expectComposite(low, static_cast<int>(x), static_cast<int>(y))) ? 1 : 0;
				holes += horizon[y * kSize + x] < 0.25f * r ? 1 : 0;
			}
		}
		Check(fogPixels > 100 && unmatched > 0 && holes == 0, "horizon keeps its fog",
			std::to_string(holes) + " holes among " + std::to_string(fogPixels) + " fog pixels (" + std::to_string(unmatched) +
				" with no matching neighbour)");

		SafeRelease(blend);
		SafeRelease(sceneRtv);
		SafeRelease(sceneTex);
		SafeRelease(lowSrv);
		SafeRelease(lowRtv);
		SafeRelease(lowTex);
	}

	SafeRelease(linear);
	SafeRelease(cb);
	SafeRelease(depthSrv);
	SafeRelease(depthTex);
	SafeRelease(rtv);
	SafeRelease(out);
	SafeRelease(comp);
	SafeRelease(ps);
	SafeRelease(vs);
	SafeRelease(hSrv);
	SafeRelease(hTex);
	sim.Release();
	SafeRelease(ctx);
	SafeRelease(dev);
	return g_fails;
}
