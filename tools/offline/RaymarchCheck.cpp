// Offline check for the raymarched fog on WARP.
//   For several cameras looking at the same ground point (different pitch, distance and yaw) the GPU fog at the
//   screen centre must match a fine CPU integral of the same column texture along the same ray. That is the
//   "fog stays on the ground whatever the camera does" property, measured. Plus: a near scene surface cuts the fog.

#include "RaymarchShaders.h"
#include "IniEdit.h"
#include "WeatherRules.h"
#include "WorldspaceRules.h"
#include "RegionRules.h"
#include "FogConditions.h"
#include "PlacementRules.h"
#include "WindSim.h"
#include "WindRules.h"
#include "Settings.h"   // the shipping defaults (inline values only; nothing here reads the ini)

#include <DirectXMath.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
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

	// RunWakeChecks runs twice: as it was, then on a field created with [Wind] FogSampling 1, the same limits, the
	// name marked (DEBUG-LEDGER.md 2026-11-15, gate H4). Empty in the first pass, so those lines are as they were.
	const char* g_pass = "";
	int         g_fogSampling = 0;

	void Check(bool a_ok, const char* a_name, const std::string& a_detail)
	{
		std::printf("%s %s%s : %s\n", a_ok ? "PASS" : "FAIL", a_name, g_pass, a_detail.c_str());
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

	ID3DBlob* Compile(const char* a_body, const char* a_target, const D3D_SHADER_MACRO* a_defines = nullptr)
	{
		std::string source = RaymarchShaders::kCommon;
		source += a_body;
		ID3DBlob* code = nullptr;
		ID3DBlob* errors = nullptr;
		const HRESULT hr = D3DCompile(source.data(), source.size(), "GroundFogRay", a_defines, nullptr, "main", a_target,
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
	float CpuFog(const std::vector<float>& a_cols, DirectX::XMFLOAT3 a_cam, DirectX::XMFLOAT3 a_dir, float a_maxDist, float a_opacity = kOpacity)
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
			tau += a_opacity * rho * (step / 70.0f);
		}
		return 1.0f - static_cast<float>(std::exp(-tau));
	}

	// ---- far layer and far pools along exact rays ----
	// The composite's own FarLayerTau and FarPoolTau run in a small compute kernel over given eyes, directions and lengths
	// (no screen in between), against fine CPU integrals of the same models.
	constexpr const char* kFarKernel = R"(
StructuredBuffer<float4>   FarRays : register(t8);   // per ray: eye (relative to PosAdjust) and length, then the direction (t7: the composite's relief)
RWStructuredBuffer<float4> FarOut  : register(u0);   // per ray: far layer tau, its brightness, its mean distance, far pools' tau
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	const float4 a = FarRays[id.x * 2];
	const float4 b = FarRays[id.x * 2 + 1];
	float bright, meanT, brightP, meanP;
	const float tau = FarLayerTau(a.xyz, b.xyz, a.w, bright, meanT);
	const float tauP = FarPoolTau(a.xyz, b.xyz, a.w, brightP, meanP);
	FarOut[id.x] = float4(tau, bright, meanT, tauP);
}
)";

	struct FarRay
	{
		float eye[3];
		float dist;
		float dir[3];
		float pad;
	};

	FarRay RayTo(double a_ex, double a_ey, double a_ez, double a_px, double a_py, double a_pz)
	{
		const double dx = a_px - a_ex, dy = a_py - a_ey, dz = a_pz - a_ez;
		const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
		return { { static_cast<float>(a_ex), static_cast<float>(a_ey), static_cast<float>(a_ez) }, static_cast<float>(len),
			{ static_cast<float>(dx / len), static_cast<float>(dy / len), static_cast<float>(dz / len) }, 0.0f };
	}

	ID3D11ShaderResourceView* MakeRgTexture(ID3D11Device* a_dev, int a_size, const std::vector<float>& a_rg)
	{
		D3D11_TEXTURE2D_DESC d{};
		d.Width = d.Height = static_cast<UINT>(a_size);
		d.MipLevels = d.ArraySize = 1;
		d.Format = DXGI_FORMAT_R32G32_FLOAT;
		d.SampleDesc.Count = 1;
		d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA    init{ a_rg.data(), static_cast<UINT>(a_size * sizeof(float) * 2), 0 };
		ID3D11Texture2D*          tex = nullptr;
		ID3D11ShaderResourceView* srv = nullptr;
		if (SUCCEEDED(a_dev->CreateTexture2D(&d, &init, &tex))) a_dev->CreateShaderResourceView(tex, nullptr, &srv);
		SafeRelease(tex);
		return srv;
	}

	// Land table: a_size vertices a_cell apart from (a_ox, a_oy), ground a_ground(x, y) (NaN = unknown, -1e30 in the
	// table), share 1 (these scenes' land is gentle; the share itself is GridMath::BuildLandTable's, FogShaderCheck.cpp).
	template <class F>
	std::vector<float> LandRg(int a_size, double a_ox, double a_oy, double a_cell, F a_ground, float a_share = 1.0f)
	{
		std::vector<float> rg(static_cast<std::size_t>(a_size) * a_size * 2);
		for (int y = 0; y < a_size; ++y)
			for (int x = 0; x < a_size; ++x) {
				const double      g = a_ground(a_ox + x * a_cell, a_oy + y * a_cell);
				const std::size_t i = (static_cast<std::size_t>(y) * a_size + x) * 2;
				rg[i] = std::isnan(g) ? -1.0e30f : static_cast<float>(g);
				rg[i + 1] = std::isnan(g) ? 0.0f : a_share;
			}
		return rg;
	}

	double Smooth(double a_e0, double a_e1, double a_x)
	{
		const double t = std::clamp((a_x - a_e0) / (a_e1 - a_e0), 0.0, 1.0);
		return t * t * (3.0 - 2.0 * t);
	}

	// The share of fog at distance t along the ray that shows: RaymarchShaders.h FarThinStart (kFarThin0 5000 without a
	// field or with a small one, else the end of the field's fade ring, 0.475 x its side), kFarThin1 30000.
	double ThinStart(double a_fieldSize) { return std::max(5000.0, 0.475 * a_fieldSize); }
	double Thin(double a_t, double a_start = 5000.0) { return 1.0 - Smooth(a_start, 30000.0, a_t); }

	// The march's stretch of a ray (RaymarchShaders.h MarchPixel), on the CPU: the box (the field's square, heights
	// a_zLo..a_zHi), the scene distance, and with a_circle the round fade's outer circle (0.475 x the field's side from
	// its centre). Returns the stretch's length (0 = not marched).
	double MarchSpan(const double a_eye[3], const double a_dir[3], double a_sceneDist, double a_origin, double a_extent, double a_zLo, double a_zHi, bool a_circle)
	{
		const double lo[3]{ a_origin, a_origin, a_zLo }, hi[3]{ a_origin + a_extent, a_origin + a_extent, a_zHi };
		double       tEnter = 0.0, tExit = a_sceneDist;
		for (int k = 0; k < 3; ++k) {
			const double inv = 1.0 / (std::fabs(a_dir[k]) > 1e-6 ? a_dir[k] : 1e-6);
			const double t0 = (lo[k] - a_eye[k]) * inv, t1 = (hi[k] - a_eye[k]) * inv;
			tEnter = std::max(tEnter, std::min(t0, t1));
			tExit = std::min(tExit, std::max(t0, t1));
		}
		if (a_circle) {
			const double ox = a_eye[0] - (a_origin + 0.5 * a_extent), oy = a_eye[1] - (a_origin + 0.5 * a_extent), R = 0.475 * a_extent;
			const double qa = a_dir[0] * a_dir[0] + a_dir[1] * a_dir[1], qc = ox * ox + oy * oy - R * R;
			if (qa > 1e-8) {
				const double qb = ox * a_dir[0] + oy * a_dir[1], disc = qb * qb - qa * qc;
				if (!(disc > 0.0)) return 0.0;
				tEnter = std::max(tEnter, (-qb - std::sqrt(disc)) / qa);
				tExit = std::min(tExit, (-qb + std::sqrt(disc)) / qa);
			} else if (!(qc < 0.0)) {
				return 0.0;
			}
		}
		return std::max(tExit - tEnter, 0.0);
	}

	// Fine CPU integral of the far layer's model along a ray: per unit, FogP.x * FarP.x * shape(h / H) / 70 x share(x, y)
	// for 0 <= h < H above the ground g(x, y) (nothing under it: the scenes keep their rays out of the ground), each
	// step's absorbed share shown by Thin(t). Returns that alpha (a_thin false: no thinning).
	template <class G, class S>
	double LayerAlphaCpu(const FarRay& a_r, double a_opacity, double a_strength, double a_H, G a_ground, S a_share, bool a_thin = true, double a_thinStart = 5000.0)
	{
		constexpr int kN = 40000;
		const double  dt = a_r.dist / kN;
		double        tau = 0.0, shown = 0.0;
		for (int i = 0; i < kN; ++i) {
			const double t = (i + 0.5) * dt;
			const double x = a_r.eye[0] + a_r.dir[0] * t, y = a_r.eye[1] + a_r.dir[1] * t, z = a_r.eye[2] + a_r.dir[2] * t;
			const double h = z - a_ground(x, y);
			if (h < 0.0 || h >= a_H) continue;
			const double d = a_opacity * a_strength * g_lowProfile.Shape(h / a_H) / 70.0 * a_share(x, y) * dt;
			shown += std::exp(-tau) * (1.0 - std::exp(-d)) * (a_thin ? Thin(t, a_thinStart) : 1.0);
			tau += d;
		}
		return shown;
	}

	void RunFarRayChecks(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx)
	{
		const std::string src = std::string(RaymarchShaders::kCompositeCore) + kFarKernel;
		ID3DBlob*            code = Compile(src.c_str(), "cs_5_0");
		ID3D11ComputeShader* cs = nullptr;
		ID3D11Buffer*        cb = nullptr;
		D3D11_BUFFER_DESC    cbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
		const bool           ok = code && SUCCEEDED(a_dev->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &cs)) &&
		                SUCCEEDED(a_dev->CreateBuffer(&cbd, nullptr, &cb));
		SafeRelease(code);
		Check(ok, "far ray kernel", ok ? "compiled" : "creation failed");
		if (!ok) {
			SafeRelease(cs);
			SafeRelease(cb);
			return;
		}
		// Runs the rays; per ray (far layer tau, its brightness, its mean distance, far pools' tau).
		auto run = [&](const RaymarchShaders::RayCB& a_rc, ID3D11ShaderResourceView* a_coarse, ID3D11ShaderResourceView* a_land, const std::vector<FarRay>& a_rays) {
			std::vector<float>         out(a_rays.size() * 4, NAN);
			const UINT                 n = static_cast<UINT>(a_rays.size());
			ID3D11Buffer*              in = nullptr;
			ID3D11Buffer*              res = nullptr;
			ID3D11Buffer*              st = nullptr;
			ID3D11ShaderResourceView*  inSrv = nullptr;
			ID3D11UnorderedAccessView* uav = nullptr;
			D3D11_BUFFER_DESC          ind{ n * 32u, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 16 };
			D3D11_SUBRESOURCE_DATA     init{ a_rays.data(), 0, 0 };
			D3D11_BUFFER_DESC          outd{ n * 16u, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 16 };
			D3D11_BUFFER_DESC          stgd{ n * 16u, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 16 };
			if (SUCCEEDED(a_dev->CreateBuffer(&ind, &init, &in)) && SUCCEEDED(a_dev->CreateShaderResourceView(in, nullptr, &inSrv)) &&
				SUCCEEDED(a_dev->CreateBuffer(&outd, nullptr, &res)) && SUCCEEDED(a_dev->CreateUnorderedAccessView(res, nullptr, &uav)) &&
				SUCCEEDED(a_dev->CreateBuffer(&stgd, nullptr, &st))) {
				a_ctx->UpdateSubresource(cb, 0, nullptr, &a_rc, 0, 0);
				ID3D11ShaderResourceView* srv[4]{ a_coarse, a_land, nullptr, inSrv };   // t5, t6, t7 (relief: none), t8
				a_ctx->CSSetShader(cs, nullptr, 0);
				a_ctx->CSSetConstantBuffers(0, 1, &cb);
				a_ctx->CSSetShaderResources(5, 4, srv);
				a_ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
				a_ctx->Dispatch(n, 1, 1);
				ID3D11ShaderResourceView*  noSrv[4]{};
				ID3D11UnorderedAccessView* noUav = nullptr;
				a_ctx->CSSetShaderResources(5, 4, noSrv);
				a_ctx->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
				a_ctx->CopyResource(st, res);
				D3D11_MAPPED_SUBRESOURCE m{};
				if (SUCCEEDED(a_ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
					std::memcpy(out.data(), m.pData, out.size() * sizeof(float));
					a_ctx->Unmap(st, 0);
				}
			}
			SafeRelease(uav);
			SafeRelease(inSrv);
			SafeRelease(st);
			SafeRelease(res);
			SafeRelease(in);
			return out;
		};
		auto alphaOf = [](double a_tau) { return 1.0 - std::exp(-a_tau); };
		auto fmt = [](double a_v) { return std::to_string(a_v).substr(0, std::to_string(a_v).find('.') + 4); };
		// The game's defaults (Thickness 126, ProfileSteepness 3) with a low strength so alpha stays mid-range.
		constexpr double kH = 126.0, kOp = 0.18;
		auto baseCb = [&](float a_strength) {
			RaymarchShaders::RayCB rc{};
			rc.fog[0] = static_cast<float>(kOp);
			rc.fog[1] = kSteepness;
			rc.farFog[0] = a_strength;
			rc.farFog[1] = 0.0f;
			rc.farFog[2] = static_cast<float>(kH);
			rc.farFog[3] = 0.0f;
			return rc;
		};
		constexpr int    kLand = 193;      // Terrain::kLandSize
		constexpr double kLandCell = 128.0;
		constexpr double kHalf = kLandCell * (kLand - 1) / 2;   // 12288
		auto setLand = [&](RaymarchShaders::RayCB& a_rc, double a_ox, double a_oy) {
			a_rc.landGrid[0] = static_cast<float>(a_ox);
			a_rc.landGrid[1] = static_cast<float>(a_oy);
			a_rc.landGrid[2] = static_cast<float>(kLandCell);
			a_rc.landGrid[3] = static_cast<float>(kLand);
		};
		auto flat = [](double a_z) { return [a_z](double, double) { return a_z; }; };
		auto one = [](double, double) { return 1.0; };

		// 1. Flat known land under a filled table: rays down to the ground at several distances and azimuths, and level rays
		// inside the layer ending at a wall. Tolerance 0.01: the shader weights each 256-unit piece's absorption once (at its
		// absorption centre, density taken as even), the reference every 0.4 units or less.
		{
			auto* land = MakeRgTexture(a_dev, kLand, LandRg(kLand, -kHalf, -kHalf, kLandCell, flat(0.0)));
			auto  rc = baseCb(0.1f);
			setLand(rc, -kHalf, -kHalf);
			std::vector<FarRay> rays;
			for (const double az : { 0.0, 37.0, 90.0 })
				for (const double D : { 1500.0, 3000.0, 6000.0, 9000.0, 12000.0 }) {
					const double a = az * 3.14159265358979 / 180.0;
					rays.push_back(RayTo(0.0, 0.0, 150.0, D * std::sin(a), D * std::cos(a), 0.0));
				}
			for (const double D : { 4000.0, 8000.0, 11000.0 }) rays.push_back(RayTo(0.0, 0.0, 60.0, 0.0, D, 60.0));
			const auto  out = run(rc, nullptr, land, rays);
			double      worst = 0.0, lo = 1.0, hi = 0.0;
			std::string samples;
			for (std::size_t i = 0; i < rays.size(); ++i) {
				const double gpu = alphaOf(out[i * 4]);
				const double cpu = LayerAlphaCpu(rays[i], kOp, 0.1, kH, flat(0.0), one);
				worst = std::max(worst, std::isfinite(gpu) ? std::fabs(gpu - cpu) : 1.0);
				lo = std::min(lo, gpu);
				hi = std::max(hi, gpu);
				if (i < 5 || i >= 15) samples += " d " + std::to_string(static_cast<int>(rays[i].dist)) + " " + fmt(gpu) + "/" + fmt(cpu);
			}
			Check(worst < 0.01 && lo < 0.6 && hi > 0.3, "far layer on flat known land matches a fine integration",
				"worst |GPU - CPU| alpha " + std::to_string(worst) + " over " + std::to_string(rays.size()) + " rays, alpha " + fmt(lo) + ".." + fmt(hi) +
					"; GPU/CPU by length:" + samples);
			SafeRelease(land);
		}

		// 1b. The table's slope share (channel G) scales the layer: rays within 5000 units (no thinning), so G = 0.5 gives
		// half the optical depth of G = 1 and G = 0 none.
		{
			auto rc = baseCb(0.1f);
			setLand(rc, -kHalf, -kHalf);
			const std::vector<FarRay> rays{ RayTo(0.0, 0.0, 150.0, 0.0, 1500.0, 0.0), RayTo(0.0, 0.0, 150.0, 2400.0, 1800.0, 0.0) };
			double                    tau[3][2]{};
			const float               shares[3]{ 1.0f, 0.5f, 0.0f };
			for (int k = 0; k < 3; ++k) {
				auto*      land = MakeRgTexture(a_dev, kLand, LandRg(kLand, -kHalf, -kHalf, kLandCell, flat(0.0), shares[k]));
				const auto out = run(rc, nullptr, land, rays);
				tau[k][0] = out[0];
				tau[k][1] = out[4];
				SafeRelease(land);
			}
			const double halfOff = std::max(std::fabs(tau[1][0] / tau[0][0] - 0.5), std::fabs(tau[1][1] / tau[0][1] - 0.5));
			Check(tau[0][0] > 0.05 && tau[0][1] > 0.05 && halfOff < 1e-3 && tau[2][0] == 0.0 && tau[2][1] == 0.0, "far layer takes the table's slope share",
				"tau with G = 1 / 0.5 / 0: " + std::to_string(tau[0][0]) + " / " + std::to_string(tau[1][0]) + " / " + std::to_string(tau[2][0]) + " and " +
					std::to_string(tau[0][1]) + " / " + std::to_string(tau[1][1]) + " / " + std::to_string(tau[2][1]) + " (ratio off 0.5 by " + std::to_string(halfOff) + ")");
		}

		// 2. No table (LandG all zero): the ground is the level FarP.y. Rays down to that level carry fog as on known land at
		// that height; rays to points 1500 units above it carry none (the continued ground cannot reach them).
		{
			auto rc = baseCb(0.1f);
			rc.farFog[1] = 50.0f;
			std::vector<FarRay> onLevel, above;
			for (const double D : { 2000.0, 6000.0, 12000.0, 20000.0 }) onLevel.push_back(RayTo(0.0, 0.0, 250.0, 0.0, D, 50.0));
			for (const double D : { 8000.0, 16000.0 }) above.push_back(RayTo(0.0, 0.0, 250.0, 0.0, D, 1550.0));
			const auto out = run(rc, nullptr, nullptr, onLevel);
			const auto outA = run(rc, nullptr, nullptr, above);
			double      worst = 0.0, lo = 1.0, maxAbove = 0.0;
			std::string samples;
			for (std::size_t i = 0; i < onLevel.size(); ++i) {
				const double gpu = alphaOf(out[i * 4]);
				const double cpu = LayerAlphaCpu(onLevel[i], kOp, 0.1, kH, flat(50.0), one);
				worst = std::max(worst, std::isfinite(gpu) ? std::fabs(gpu - cpu) : 1.0);
				lo = std::min(lo, gpu);
				samples += " " + fmt(gpu) + "/" + fmt(cpu);
			}
			for (std::size_t i = 0; i < above.size(); ++i) maxAbove = std::max(maxAbove, std::isfinite(outA[i * 4]) ? alphaOf(outA[i * 4]) : 1.0);
			// 0.02: past the table the continued ground has 3 pieces, each weighted once.
			Check(worst < 0.02 && lo > 0.05 && maxAbove < 1e-6, "far layer without a table lies on the level FarP.y",
				"on the level: worst |GPU - CPU| " + std::to_string(worst) + ", GPU/CPU at 2000/6000/12000/20000:" + samples + "; 1500 above it: alpha up to " +
					std::to_string(maxAbove));
		}

		// 3. Continuity (GPT counterexample 2026-11-13): a piece lying about level near the under-ground limit. FarP (1, 0,
		// 126, 0), FogP (0.18, 3), eye at z 0, a level ray of one piece (256 units) over ground 126 rising or falling by 0.126
		// over the piece. The two must differ by less than 0.01 (with a hard limit in place of the ramp: 0 and 0.875, measured
		// 2026-11-13).
		{
			double a[2]{};
			bool   finite = true;   // a NaN from both runs must not pass as "no jump" (GPT review 2026-11-13)
			for (int k = 0; k < 2; ++k) {
				const double s = (k == 0 ? 0.126 : -0.126) / 256.0;
				auto*        land = MakeRgTexture(a_dev, kLand, LandRg(kLand, -kHalf, -kHalf, kLandCell, [s](double, double y) { return 126.0 + s * y; }));
				auto         rc = baseCb(1.0f);
				setLand(rc, -kHalf, -kHalf);
				const auto out = run(rc, nullptr, land, { RayTo(0.0, 0.0, 0.0, 0.0, 256.0, 0.0) });
				finite = finite && std::isfinite(out[0]);
				a[k] = std::isfinite(out[0]) ? alphaOf(out[0]) : 1.0;
				SafeRelease(land);
			}
			Check(finite && std::fabs(a[0] - a[1]) < 0.01, "far layer has no jump at the under-ground limit",
				"ground rising: alpha " + std::to_string(a[0]) + ", falling: " + std::to_string(a[1]));
		}

		// 4. A ray over known land longer than 10240 units still reads the table (64 pieces): the ground stands 1000 units over
		// the level ray (z 60) for its first 10400 units and is 0 after, so all of its fog lies 10460..14000 units out. Read
		// only 40 pieces deep, the rest continues from the high ground and carries next to none.
		{
			const double oy = -kHalf, eyeY = oy + 32.0;
			auto         ground = [&](double, double a_y) { return a_y - eyeY < 10400.0 ? 1000.0 : 0.0; };
			auto         rg = LandRg(kLand, -kHalf, oy, kLandCell, ground);
			auto*        land = MakeRgTexture(a_dev, kLand, rg);
			auto         rc = baseCb(0.1f);
			setLand(rc, -kHalf, oy);
			const FarRay ray = RayTo(0.0, eyeY, 60.0, 0.0, eyeY + 14000.0, 60.0);
			const auto   out = run(rc, nullptr, land, { ray });
			// The table's ground along the ray: linear between its vertices.
			auto tableGround = [&](double, double a_y) {
				const double g = (a_y - oy) / kLandCell;
				const int    j = std::clamp(static_cast<int>(std::floor(g)), 0, kLand - 2);
				const double f = g - j, g0 = rg[(static_cast<std::size_t>(j) * kLand) * 2], g1 = rg[(static_cast<std::size_t>(j + 1) * kLand) * 2];
				return g0 + (g1 - g0) * f;
			};
			const double gpu = std::isfinite(out[0]) ? alphaOf(out[0]) : -1.0;
			const double cpu = LayerAlphaCpu(ray, kOp, 0.1, kH, tableGround, one);
			Check(gpu > 0.2 && std::fabs(gpu - cpu) < 0.02, "far layer reads the table past 10240 units",
				"alpha GPU " + std::to_string(gpu) + " vs CPU " + std::to_string(cpu) + " (fog only 10460..14000 units out)");
			SafeRelease(land);
		}

		// 5. A far tree top: a scene point 750 units above level ground, 229 m (16030 units) out, beyond the known land (the
		// table knows 8192 units around the player). It must carry less than 0.05 (reviewer 2026-11-13: fogged to the top).
		// The ground point under it carries fog (the layer is on out there). The eye stands 120 units up, inside the layer,
		// with the wind field (WorldSize 8192) around it as in game, so the fog right in front of the eye is the field's.
		{
			auto* land = MakeRgTexture(a_dev, kLand, LandRg(kLand, -kHalf, -kHalf, kLandCell, [](double a_x, double a_y) {
				return std::fabs(a_x) <= 8192.0 && std::fabs(a_y) <= 8192.0 ? 0.0 : NAN;
			}));
			auto rc = baseCb(1.0f);
			setLand(rc, -kHalf, -kHalf);
			rc.field[0] = rc.field[1] = -4096.0f;
			rc.field[2] = 8192.0f;
			rc.field[3] = 1.0f;
			const auto   out = run(rc, nullptr, land, { RayTo(0.0, 0.0, 120.0, 0.0, 16030.0, 750.0), RayTo(0.0, 0.0, 120.0, 0.0, 16030.0, 0.0) });
			const double top = std::isfinite(out[0]) ? alphaOf(out[0]) : 1.0, foot = std::isfinite(out[4]) ? alphaOf(out[4]) : 0.0;
			Check(top < 0.05 && foot > 0.05, "far tree top beyond the known land stays clear",
				"alpha at the top (750 up) " + std::to_string(top) + ", at its foot " + std::to_string(foot));
			SafeRelease(land);
		}

		// 6. Share across the field edge: with the field on, the far layer carries 1 - the field's round fade at each point.
		{
			auto* land = MakeRgTexture(a_dev, kLand, LandRg(kLand, -kHalf, -kHalf, kLandCell, flat(0.0)));
			auto  rc = baseCb(1.0f);
			setLand(rc, -kHalf, -kHalf);
			rc.field[0] = rc.field[1] = kOrigin;
			rc.field[2] = kExtent;
			rc.field[3] = 1.0f;
			auto fieldShare = [](double a_x, double a_y) {
				const double r = std::sqrt((a_x - (kOrigin + 0.5 * kExtent)) * (a_x - (kOrigin + 0.5 * kExtent)) + (a_y - (kOrigin + 0.5 * kExtent)) * (a_y - (kOrigin + 0.5 * kExtent))) / (0.5 * kExtent);
				return Smooth(0.55, 0.95, r);
			};
			std::vector<FarRay> rays;
			for (const double y : { 0.0, 300.0, 600.0, 900.0, 1200.0, 1600.0, 2000.0, 2600.0 }) rays.push_back(RayTo(1500.0, -150.0, 283.0, 1500.0, y, 0.0));
			for (const double x : { 900.0, 1200.0, 2000.0, 2400.0 }) rays.push_back(RayTo(1500.0, -150.0, 283.0, x, 150.0, 0.0));
			const auto  out = run(rc, nullptr, land, rays);
			double      worst = 0.0, lo = 1.0, hi = 0.0;
			std::string samples;
			for (std::size_t i = 0; i < rays.size(); ++i) {
				const double gpu = alphaOf(out[i * 4]);
				const double cpu = LayerAlphaCpu(rays[i], kOp, 1.0, kH, flat(0.0), fieldShare);
				worst = std::max(worst, std::isfinite(gpu) ? std::fabs(gpu - cpu) : 1.0);
				lo = std::min(lo, gpu);
				hi = std::max(hi, gpu);
				samples += " " + fmt(gpu) + "/" + fmt(cpu);
			}
			// 0.02: FarPiece takes the share at each 256-unit piece's middle; the ray to (1200, 150) ends 0.04 into the fade
			// ring after two pieces, with its fog at the far end (alpha 0.050 vs 0.033, 2026-11-13).
			Check(worst < 0.02 && lo < 0.1 && hi > 0.3, "far layer share across the field edge",
				"worst |GPU - CPU| alpha " + std::to_string(worst) + " over " + std::to_string(rays.size()) + " rays, alpha " + fmt(lo) + ".." + fmt(hi) +
					"; GPU/CPU:" + samples);
			SafeRelease(land);
		}

		// 6b. The thinning starts at the end of the field's fade ring when that lies past 5000 units (FarThinStart): the
		// field's own fog does not thin with distance, so the far layer must take over at full strength there whatever the
		// field's size. Fields of 12288 (the default: ring 3379..5837 units) and 16384 (4506..7782) round the eye, flat
		// known land, level rays inside the layer to walls 7000 to 12000 units out and rays down to the ground. Against the
		// fine integral with the thinning from max(5000, 0.475 x field); the same integral with the old fixed 5000 is
		// printed beside it and must be further off than the tolerance (so this check fails on the old start).
		{
			auto* land = MakeRgTexture(a_dev, kLand, LandRg(kLand, -kHalf, -kHalf, kLandCell, flat(0.0)));
			double      worst = 0.0, oldGap = 0.0, lo = 1.0, hi = 0.0;
			std::string samples;
			int         count = 0;
			for (const double F : { 12288.0, 16384.0 }) {
				auto rc = baseCb(0.05f);
				setLand(rc, -kHalf, -kHalf);
				rc.field[0] = rc.field[1] = static_cast<float>(-0.5 * F);
				rc.field[2] = static_cast<float>(F);
				rc.field[3] = 1.0f;
				auto fieldShare = [F](double a_x, double a_y) { return Smooth(0.55, 0.95, std::sqrt(a_x * a_x + a_y * a_y) / (0.5 * F)); };
				std::vector<FarRay> rays;
				for (const double D : { 7000.0, 9000.0, 12000.0 }) rays.push_back(RayTo(0.0, 0.0, 60.0, 0.0, D, 60.0));
				for (const double D : { 8000.0, 11000.0 }) rays.push_back(RayTo(0.0, 0.0, 150.0, 0.6 * D, 0.8 * D, 0.0));
				const auto out = run(rc, nullptr, land, rays);
				samples += " field " + std::to_string(static_cast<int>(F)) + ":";
				for (std::size_t i = 0; i < rays.size(); ++i) {
					const double gpu = alphaOf(out[i * 4]);
					const double cpu = LayerAlphaCpu(rays[i], kOp, 0.05, kH, flat(0.0), fieldShare, true, ThinStart(F));
					const double old = LayerAlphaCpu(rays[i], kOp, 0.05, kH, flat(0.0), fieldShare, true, 5000.0);
					worst = std::max(worst, std::isfinite(gpu) ? std::fabs(gpu - cpu) : 1.0);
					if (F > 16000.0) oldGap = std::max(oldGap, cpu - old);
					lo = std::min(lo, gpu);
					hi = std::max(hi, gpu);
					samples += " " + fmt(gpu) + "/" + fmt(cpu) + "/" + fmt(old);
					++count;
				}
			}
			// 0.003: measured 0.0002 (the pieces are 256 units over a ring 2458 and 3277 units wide); the old start is off
			// by 0.034 at 16384 and by 0.012 at 12288.
			Check(worst < 0.003 && oldGap > 0.02 && lo < 0.5 && hi > 0.3, "far layer thins from the fade ring's end at any field size",
				"worst |GPU - CPU| alpha " + std::to_string(worst) + " over " + std::to_string(count) + " rays (tolerance 0.003), alpha " + fmt(lo) + ".." + fmt(hi) +
					"; the old start (5000) is off by up to " + std::to_string(oldGap) + " at 16384; GPU/CPU/CPU with the old start:" + samples);
			SafeRelease(land);
		}

		// 7. Far pools thin out with distance as the far layer: a 150-unit pool over level ground across the coarse grid
		// (65 cells of 512), rays down into it from 300 units up, against the thinned CPU integral (and the unthinned one,
		// which the far rays must miss).
		{
			constexpr int    kCs = 65;
			constexpr double kCc = 512.0, kCo = -kCc * (kCs - 1) / 2, kDen = 0.02, kLid = 150.0;
			std::vector<float> crg(static_cast<std::size_t>(kCs) * kCs * 2);
			for (std::size_t i = 0; i < crg.size(); i += 2) {
				crg[i] = 0.0f;
				crg[i + 1] = static_cast<float>(kLid);
			}
			auto* coarse = MakeRgTexture(a_dev, kCs, crg);
			auto  rc = baseCb(0.0f);
			rc.fog[0] = kOpacity;
			rc.poolGrid[0] = rc.poolGrid[1] = static_cast<float>(kCo);
			rc.poolGrid[2] = static_cast<float>(kCc);
			rc.poolGrid[3] = static_cast<float>(kCs);
			rc.poolFar[0] = static_cast<float>(kDen);
			rc.poolFar[1] = 600.0f;
			rc.poolFar[2] = static_cast<float>(kLid);
			std::vector<FarRay> rays;
			for (const double D : { 2000.0, 6000.0, 10000.0, 14000.0 }) {
				rays.push_back(RayTo(0.0, 0.0, 300.0, D, 0.0, 0.0));
				rays.push_back(RayTo(0.0, 0.0, 300.0, 0.7 * D, 0.7 * D, 0.0));
			}
			const auto out = run(rc, coarse, nullptr, rays);
			// a_field > 0: a field of that side round the eye; the pools carry 1 - its round fade and thin from its ring's end.
			auto       poolAlpha = [&](const FarRay& a_r, bool a_thin, double a_field = 0.0, double a_thinStart = 5000.0) {
                constexpr int kN = 40000;
                const double  dt = a_r.dist / kN;
                double        tau = 0.0, shown = 0.0;
                for (int i = 0; i < kN; ++i) {
                    const double t = (i + 0.5) * dt, z = a_r.eye[2] + a_r.dir[2] * t;
                    if (z < 0.0 || z >= kLid) continue;
                    const double x = a_r.eye[0] + a_r.dir[0] * t, y = a_r.eye[1] + a_r.dir[1] * t;
                    const double share = a_field > 0.0 ? Smooth(0.55, 0.95, std::sqrt(x * x + y * y) / (0.5 * a_field)) : 1.0;
                    const double d = kOpacity * kDen * g_upperProfile.Shape(z / kLid) / 70.0 * share * dt;
                    shown += std::exp(-tau) * (1.0 - std::exp(-d)) * (a_thin ? Thin(t, a_thinStart) : 1.0);
                    tau += d;
                }
                return shown;
			};
			double      worst = 0.0, gapFar = 0.0;
			std::string samples;
			for (std::size_t i = 0; i < rays.size(); ++i) {
				const double gpu = alphaOf(out[i * 4 + 3]);
				const double cpu = poolAlpha(rays[i], true);
				worst = std::max(worst, std::isfinite(gpu) ? std::fabs(gpu - cpu) : 1.0);
				if (i % 2 == 0) samples += " " + fmt(gpu) + "/" + fmt(cpu) + "/" + fmt(poolAlpha(rays[i], false));
				if (i == 6) gapFar = poolAlpha(rays[i], false) - cpu;
			}
			Check(worst < 0.01 && gapFar > 0.05, "far pools thin out with distance",
				"worst |GPU - thinned CPU| alpha " + std::to_string(worst) + " over " + std::to_string(rays.size()) +
					" rays; GPU/thinned/unthinned at 2000/6000/10000/14000:" + samples);
			// 7b. With a field of 16384 round the eye the pools, as the far layer (6b), thin from its fade ring's end (7782
			// units), not from 5000: rays to 9000, 11000 and 14000 units, against the integral with the field's share and
			// that start; the integral with the old start is printed beside it and must be further off than the tolerance.
			{
				constexpr double F = 16384.0;
				rc.field[0] = rc.field[1] = static_cast<float>(-0.5 * F);
				rc.field[2] = static_cast<float>(F);
				rc.field[3] = 1.0f;
				std::vector<FarRay> far;
				for (const double D : { 9000.0, 11000.0, 14000.0 }) {
					far.push_back(RayTo(0.0, 0.0, 300.0, D, 0.0, 0.0));
					far.push_back(RayTo(0.0, 0.0, 300.0, 0.7 * D, 0.7 * D, 0.0));
				}
				const auto  outF = run(rc, coarse, nullptr, far);
				double      worstF = 0.0, oldGap = 0.0;
				std::string samplesF;
				for (std::size_t i = 0; i < far.size(); ++i) {
					const double gpu = alphaOf(outF[i * 4 + 3]);
					const double cpu = poolAlpha(far[i], true, F, ThinStart(F));
					const double old = poolAlpha(far[i], true, F, 5000.0);
					worstF = std::max(worstF, std::isfinite(gpu) ? std::fabs(gpu - cpu) : 1.0);
					oldGap = std::max(oldGap, cpu - old);
					if (i % 2 == 0) samplesF += " " + fmt(gpu) + "/" + fmt(cpu) + "/" + fmt(old);
				}
				// 0.003: measured 0.0002 (the share is read once per piece of about one coarse cell across the 3277-unit ring).
				Check(worstF < 0.003 && oldGap > 0.02, "far pools thin from the fade ring's end at field size 16384",
					"worst |GPU - CPU| alpha " + std::to_string(worstF) + " over " + std::to_string(far.size()) + " rays (tolerance 0.003); the old start (5000) is off by up to " +
						std::to_string(oldGap) + "; GPU/CPU/CPU with the old start at 9000/11000/14000:" + samplesF);
			}
			SafeRelease(coarse);
		}

		// 8. The pixel angle comes from FarN.w (the CPU: 2 / (proj _22 x render height)); 0 there falls back to kFarPixel
		// (0.001). Patches on, a grazing ray where the patch filter is part way: 0 and 0.001 give the same, 0.004 does not.
		{
			auto* land = MakeRgTexture(a_dev, kLand, LandRg(kLand, -kHalf, -kHalf, kLandCell, flat(0.0)));
			auto  rc = baseCb(0.1f);
			setLand(rc, -kHalf, -kHalf);
			rc.farFog[3] = 1.0f;
			const std::vector<FarRay> rays{ RayTo(0.0, 0.0, 150.0, 0.0, 3000.0, 0.0), RayTo(0.0, 0.0, 150.0, 1700.0, 2500.0, 0.0) };
			double                    tau[3][2]{};
			const float               angles[3]{ 0.0f, 0.001f, 0.004f };
			for (int k = 0; k < 3; ++k) {
				rc.farNoise[3] = angles[k];
				const auto out = run(rc, nullptr, land, rays);
				tau[k][0] = out[0];
				tau[k][1] = out[4];
			}
			const double same = std::max(std::fabs(tau[0][0] - tau[1][0]), std::fabs(tau[0][1] - tau[1][1]));
			const double diff = std::max(std::fabs(tau[2][0] - tau[1][0]), std::fabs(tau[2][1] - tau[1][1]));
			Check(same < 1e-6 && diff > 1e-3, "far layer takes the pixel angle from FarN.w",
				"tau with 0 / 0.001 / 0.004: " + std::to_string(tau[0][0]) + " / " + std::to_string(tau[1][0]) + " / " + std::to_string(tau[2][0]));
			SafeRelease(land);
		}

		SafeRelease(cs);
		SafeRelease(cb);
	}

	// ---- column height response on the screen (WindShaders.h CSColumn, SurfaceResponse; judge's plan r11 steps 1 and 2) ----
	// Hand-built columns, the plugin's march and composite pixel shaders at full resolution (scale 1), against fine CPU
	// integrals of the same columns (CpuFog).
	//   1. The fog line on a post: a runner's lane (q 0.691, the ledger's lane mean with no wisps) five texels (320 units)
	//      wide along y, the camera on its centre line 140 units up, a post across the view 290 and 600 units ahead.
	//      Columns undisturbed (H = baseH 51.7, W = baseH / 70), SurfaceResponse 0 (W x q) and 1 (H x q and W x q: the same
	//      mass under a lower top). The height on the post where alpha is 0.5, at vertical optical depths 1.0 and 2.3.
	//      Limits set before measuring: GPU within 2 units of the CPU; the drop from the undisturbed line with 1 at least 2 x
	//      the drop with 0 is printed met / NOT met (the plan's own slab model gives 1.79 at 290 units, optical depth 1.0).
	//   2. The same mass on the screen: pixels looking at the ground along the lane (the ray in the lane's even part, its foot
	//      inside the march's circle), alpha with 1 within 0.01 of alpha with 0.
	//   3. A crest 1.6 x Thickness tall stays inside the march box of WindShaders::LowLayerTop (the box FogRenderer sets);
	//      the old box (Thickness + 2) as the control that clips it.
	//   4. Relief shading (ColorHigh.w = ReliefGain): looking straight down, the colour over relief r is 1 + G r of the
	//      colour beside it (G 0.08: 0.975 and 0.957 for r = -0.309 and -0.536, within 0.005); with G = 0 the composite is
	//      the same bit for bit with the relief texture bound or not, and so is G 0.08 with it unbound.
	void RunSurfaceChecks(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11VertexShader* a_vs, ID3D11PixelShader* a_march,
		ID3D11PixelShader* a_comp, ID3D11SamplerState* a_linear)
	{
		using namespace DirectX;
		constexpr UINT  kN = 96;                     // the screen, square
		constexpr float kBaseH = 51.7f;              // 0.41 x Thickness 126: the layer's mean height (the plan's slab model)
		constexpr float kQ = 0.691f;                 // a runner's lane, mean fog amount (ledger 2026-11-14, 0c, no wisps)
		constexpr int   kLane0 = 30, kLane1 = 34;    // lane texels across x: world -128..192; texel 32's centre is x = 32
		constexpr float kCamX = 32.0f, kCamY = -300.0f;
		constexpr float kLaneIn0 = -96.0f, kLaneIn1 = 160.0f;   // between the lane's outer texel centres: bilinear reads lane only
		const float     w0 = kBaseH / 70.0f;
		const float     thickness = Settings::thickness;
		// FogRenderer's box top over the highest ground (0 here, no steam): hi + max(LowLayerTop + 2, SteamTopMax + 2).
		const float boxOff = WindShaders::LowLayerTop(thickness, 0.0f) + 2.0f;
		const float boxOn = WindShaders::LowLayerTop(thickness, 1.0f) + 2.0f;

		auto makeSrv = [&](DXGI_FORMAT a_fmt, const void* a_data, UINT a_pitch) {
			D3D11_TEXTURE2D_DESC d{};
			d.Width = d.Height = kRes;
			d.MipLevels = d.ArraySize = 1;
			d.Format = a_fmt;
			d.SampleDesc.Count = 1;
			d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			D3D11_SUBRESOURCE_DATA    init{ a_data, a_pitch, 0 };
			ID3D11Texture2D*          tex = nullptr;
			ID3D11ShaderResourceView* srv = nullptr;
			if (SUCCEEDED(a_dev->CreateTexture2D(&d, &init, &tex))) a_dev->CreateShaderResourceView(tex, nullptr, &srv);
			SafeRelease(tex);   // the view keeps it
			return srv;
		};
		// Columns over the check's grid (flat known ground at z = 0): the lane's texel columns get (a_laneH, a_laneW), the rest (a_outH, a_outW).
		auto laneCols = [&](float a_laneH, float a_laneW, float a_outH, float a_outW) {
			std::vector<float> c(static_cast<std::size_t>(kRes) * kRes * 4);
			for (int y = 0; y < kRes; ++y) {
				for (int x = 0; x < kRes; ++x) {
					const bool in = x >= kLane0 && x <= kLane1;
					float*     t = &c[(static_cast<std::size_t>(y) * kRes + x) * 4];
					t[0] = in ? a_laneH : a_outH;
					t[1] = in ? a_laneW : a_outW;
					t[2] = 0.0f;
					t[3] = 1.0f;
				}
			}
			return c;
		};

		D3D11_TEXTURE2D_DESC td{};
		td.Width = td.Height = kN;
		td.MipLevels = td.ArraySize = 1;
		td.SampleDesc.Count = 1;
		td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
		ID3D11Texture2D*          marchTex = nullptr;
		ID3D11Texture2D*          compTex = nullptr;
		ID3D11Texture2D*          depthTex = nullptr;
		ID3D11RenderTargetView*   marchRtv = nullptr;
		ID3D11RenderTargetView*   compRtv = nullptr;
		ID3D11ShaderResourceView* marchSrv = nullptr;
		ID3D11ShaderResourceView* depthSrv = nullptr;
		ID3D11Buffer*             cb = nullptr;
		bool ok = SUCCEEDED(a_dev->CreateTexture2D(&td, nullptr, &marchTex)) && SUCCEEDED(a_dev->CreateRenderTargetView(marchTex, nullptr, &marchRtv)) &&
		          SUCCEEDED(a_dev->CreateShaderResourceView(marchTex, nullptr, &marchSrv)) && SUCCEEDED(a_dev->CreateTexture2D(&td, nullptr, &compTex)) &&
		          SUCCEEDED(a_dev->CreateRenderTargetView(compTex, nullptr, &compRtv));
		td.Format = DXGI_FORMAT_R32_FLOAT;
		td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		std::vector<float>     depth(static_cast<std::size_t>(kN) * kN, 1.0f);
		D3D11_SUBRESOURCE_DATA dinit{ depth.data(), kN * sizeof(float), 0 };
		ok = ok && SUCCEEDED(a_dev->CreateTexture2D(&td, &dinit, &depthTex)) && SUCCEEDED(a_dev->CreateShaderResourceView(depthTex, nullptr, &depthSrv));
		D3D11_BUFFER_DESC cbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
		ok = ok && SUCCEEDED(a_dev->CreateBuffer(&cbd, nullptr, &cb));

		const auto colsU = laneCols(kBaseH, w0, kBaseH, w0);              // undisturbed
		const auto colsN = laneCols(kBaseH, w0 * kQ, kBaseH, w0);         // SurfaceResponse 0: the lane thinner
		const auto colsA = laneCols(kBaseH * kQ, w0 * kQ, kBaseH, w0);    // SurfaceResponse 1: the lane lower, the same W
		const float crestH = WindShaders::LowLayerTop(thickness, 1.0f);   // shape 1 with s at its cap: 201.6
		const float crestW = thickness / 70.0f * WindShaders::kSurfaceMaxCpp;   // its W, rho 1.6 (a bank)
		const auto  colsC = laneCols(crestH, crestW, kBaseH, w0);
		auto*       srvU = makeSrv(DXGI_FORMAT_R32G32B32A32_FLOAT, colsU.data(), kRes * 16);
		auto*       srvN = makeSrv(DXGI_FORMAT_R32G32B32A32_FLOAT, colsN.data(), kRes * 16);
		auto*       srvA = makeSrv(DXGI_FORMAT_R32G32B32A32_FLOAT, colsA.data(), kRes * 16);
		auto*       srvC = makeSrv(DXGI_FORMAT_R32G32B32A32_FLOAT, colsC.data(), kRes * 16);
		ok = ok && srvU && srvN && srvA && srvC;
		Check(ok, "height response screen checks: resources", ok ? "created" : "creation failed");
		if (!ok) {
			SafeRelease(srvU); SafeRelease(srvN); SafeRelease(srvA); SafeRelease(srvC); SafeRelease(cb); SafeRelease(depthSrv); SafeRelease(depthTex);
			SafeRelease(marchSrv); SafeRelease(marchRtv); SafeRelease(compRtv); SafeRelease(compTex); SafeRelease(marchTex);
			return;
		}

		struct View
		{
			XMFLOAT3   cam;
			XMFLOAT4X4 vp, inv;
			float      p33, p43;
		};
		auto makeView = [&](XMFLOAT3 a_cam, XMFLOAT3 a_target, float a_fovDeg, XMVECTOR a_up) {
			View v{};
			v.cam = a_cam;
			const XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(a_fovDeg), 1.0f, 5.0f, 100000.0f);
			XMFLOAT4X4     pf;
			XMStoreFloat4x4(&pf, proj);
			v.p33 = pf._33;
			v.p43 = pf._43;
			const XMVECTOR fwd = XMVector3Normalize(XMVectorSet(a_target.x - a_cam.x, a_target.y - a_cam.y, a_target.z - a_cam.z, 0));
			const XMMATRIX vp = XMMatrixMultiply(XMMatrixLookToLH(XMVectorZero(), fwd, a_up), proj);
			XMStoreFloat4x4(&v.vp, vp);
			XMStoreFloat4x4(&v.inv, XMMatrixInverse(nullptr, vp));
			return v;
		};
		const XMVECTOR upZ = XMVectorSet(0, 0, 1, 0);
		auto pixelDir = [&](const View& a_v, UINT a_x, UINT a_y) {
			const XMVECTOR ndc = XMVectorSet((a_x + 0.5f) / kN * 2 - 1, 1 - (a_y + 0.5f) / kN * 2, 0.5f, 1);
			XMFLOAT4       w;
			XMStoreFloat4(&w, XMVector4Transform(ndc, XMLoadFloat4x4(&a_v.inv)));
			const XMFLOAT3 rel{ w.x / w.w, w.y / w.w, w.z / w.w };
			const float    len = std::sqrt(rel.x * rel.x + rel.y * rel.y + rel.z * rel.z);
			return XMFLOAT3{ rel.x / len, rel.y / len, rel.z / len };
		};
		// Scene: the ground (z = 0) and, with a finite a_postY, a post across the view at y = a_postY (a wall facing the camera).
		// hitZ: the height on the post each pixel sees (NaN: the ground); hitT: the distance to what it sees (inf: nothing).
		std::vector<float> hitZ(depth.size(), NAN), hitT(depth.size(), INFINITY);
		auto fillDepth = [&](const View& a_v, float a_postY) {
			for (UINT y = 0; y < kN; ++y) {
				for (UINT x = 0; x < kN; ++x) {
					const XMFLOAT3 d = pixelDir(a_v, x, y);
					float          t = INFINITY, z = NAN;
					if (std::isfinite(a_postY) && d.y > 1e-6f) {
						const float tp = (a_postY - a_v.cam.y) / d.y;
						const float zp = a_v.cam.z + d.z * tp;
						if (tp > 0.0f && zp >= 0.0f) {
							t = tp;
							z = zp;
						}
					}
					if (d.z < -1e-6f && -a_v.cam.z / d.z < t) {
						t = -a_v.cam.z / d.z;
						z = NAN;
					}
					float dev = 1.0f;
					if (std::isfinite(t)) {
						XMFLOAT4 clip;
						XMStoreFloat4(&clip, XMVector4Transform(XMVectorSet(d.x * t, d.y * t, d.z * t, 1), XMLoadFloat4x4(&a_v.vp)));
						dev = clip.z / clip.w;
					}
					depth[y * kN + x] = dev;
					hitZ[y * kN + x] = z;
					hitT[y * kN + x] = t;
				}
			}
			a_ctx->UpdateSubresource(depthTex, 0, nullptr, depth.data(), kN * sizeof(float), 0);
		};
		auto setCb = [&](const View& a_v, float a_opacity, float a_boxTop, float a_gain) {
			RaymarchShaders::RayCB rc{};
			std::memcpy(rc.invViewProj, &a_v.inv, sizeof(rc.invViewProj));
			rc.field[0] = kOrigin;
			rc.field[1] = kOrigin;
			rc.field[2] = kExtent;
			rc.field[3] = 1.0f;
			rc.box[0] = -1.0f;
			rc.box[1] = a_boxTop;
			rc.box[2] = 100000.0f;
			rc.box[3] = static_cast<float>(kSteps);
			rc.fog[0] = a_opacity;
			rc.fog[1] = kSteepness;
			rc.depth[0] = a_v.p33;
			rc.depth[1] = a_v.p43;
			rc.depth[2] = static_cast<float>(kN);
			rc.depth[3] = static_cast<float>(kN);
			rc.lowRes[0] = static_cast<float>(kN);
			rc.lowRes[1] = static_cast<float>(kN);
			rc.lowRes[2] = 1.0f;
			for (int c = 0; c < 3; ++c) {
				rc.colorLow[c] = 1.0f;
				rc.colorHigh[c] = 1.0f;
			}
			rc.colorHigh[3] = a_gain;
			rc.distFog[0] = 1.0e6f;
			rc.distFog[1] = 2.0e6f;
			rc.distFog[2] = 1.0f;
			rc.posAdjust[0] = a_v.cam.x;
			rc.posAdjust[1] = a_v.cam.y;
			rc.posAdjust[2] = a_v.cam.z;
			a_ctx->UpdateSubresource(cb, 0, nullptr, &rc, 0, 0);
		};
		const float black[4]{};
		auto march = [&](ID3D11ShaderResourceView* a_cols) {
			a_ctx->ClearRenderTargetView(marchRtv, black);
			const D3D11_VIEWPORT vp{ 0.0f, 0.0f, static_cast<float>(kN), static_cast<float>(kN), 0.0f, 1.0f };
			a_ctx->RSSetViewports(1, &vp);
			a_ctx->OMSetRenderTargets(1, &marchRtv, nullptr);
			a_ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);
			a_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			a_ctx->VSSetShader(a_vs, nullptr, 0);
			a_ctx->PSSetShader(a_march, nullptr, 0);
			a_ctx->PSSetConstantBuffers(0, 1, &cb);
			ID3D11ShaderResourceView* in[2]{ depthSrv, a_cols };
			a_ctx->PSSetShaderResources(0, 2, in);
			a_ctx->PSSetSamplers(0, 1, &a_linear);
			a_ctx->Draw(3, 0);
			ID3D11ShaderResourceView* none[2]{};
			a_ctx->PSSetShaderResources(0, 2, none);
			a_ctx->OMSetRenderTargets(0, nullptr, nullptr);
			return ReadF32(a_dev, a_ctx, marchTex, kN, kN, 4);
		};
		// The march into the low target (scale 1), then the composite (no blending: its colour and alpha as written), t7 = a_relief.
		auto composite = [&](ID3D11ShaderResourceView* a_cols, ID3D11ShaderResourceView* a_relief) {
			march(a_cols);
			a_ctx->ClearRenderTargetView(compRtv, black);
			a_ctx->OMSetRenderTargets(1, &compRtv, nullptr);
			a_ctx->PSSetShader(a_comp, nullptr, 0);
			ID3D11ShaderResourceView* in[3]{ depthSrv, a_cols, marchSrv };
			a_ctx->PSSetShaderResources(0, 3, in);
			a_ctx->PSSetShaderResources(7, 1, &a_relief);
			a_ctx->Draw(3, 0);
			ID3D11ShaderResourceView* none[3]{};
			a_ctx->PSSetShaderResources(0, 3, none);
			a_ctx->PSSetShaderResources(7, 1, none);
			a_ctx->OMSetRenderTargets(0, nullptr, nullptr);
			return ReadF32(a_dev, a_ctx, compTex, kN, kN, 4);
		};
		auto f2 = [](double a_v) {
			char b[32];
			std::snprintf(b, sizeof(b), "%.2f", a_v);
			return std::string(b);
		};

		// 1. The fog line on a post.
		{
			// Height on the post where the march's alpha crosses 0.5 down pixel column a_x (rows top to bottom go down the post),
			// linear between the two rows around it; NaN where it does not cross on the post.
			auto gpuLine = [&](const std::vector<float>& a_img, UINT a_x) {
				for (UINT y = 1; y < kN; ++y) {
					const float z0 = hitZ[(y - 1) * kN + a_x], z1 = hitZ[y * kN + a_x];
					if (!std::isfinite(z0) || !std::isfinite(z1)) continue;
					const float a0 = a_img[((y - 1) * kN + a_x) * 4], a1 = a_img[(y * kN + a_x) * 4];
					if (a0 < 0.5f && a1 >= 0.5f) return z0 + (0.5f - a0) / (a1 - a0) * (z1 - z0);
				}
				return NAN;
			};
			// The same on the CPU: bisection on the height, CpuFog from the camera to (kCamX, a_postY, z). 0 when even the foot is under 0.5.
			auto cpuLine = [&](const std::vector<float>& a_cols, const XMFLOAT3& a_cam, float a_postY, float a_opacity) {
				auto alphaAt = [&](float a_z) {
					const float dx = kCamX - a_cam.x, dy = a_postY - a_cam.y, dz = a_z - a_cam.z;
					const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
					return CpuFog(a_cols, a_cam, { dx / len, dy / len, dz / len }, len, a_opacity);
				};
				if (alphaAt(0.0f) < 0.5f) return 0.0f;
				float lo = 0.0f, hi = 120.0f;
				for (int i = 0; i < 40; ++i) {
					const float m = 0.5f * (lo + hi);
					(alphaAt(m) >= 0.5f ? lo : hi) = m;
				}
				return 0.5f * (lo + hi);
			};
			struct Model
			{
				float post, tauV, u, n, a;
			};
			// tools: scratchpad surface/drop_model.py (the judge's slab model, judge_check.py): undisturbed / SurfaceResponse 0 / 1.
			const Model models[] = { { 290.0f, 1.0f, 20.71f, 14.61f, 9.76f }, { 290.0f, 2.3f, 32.77f, 27.77f, 18.58f },
				{ 600.0f, 1.0f, 30.74f, 25.41f, 16.92f }, { 600.0f, 2.3f, 40.22f, 36.48f, 24.45f } };
			const XMFLOAT3 cam{ kCamX, kCamY, 140.0f };
			bool           good = true;
			float          worst = 0.0f;
			std::string    detail, ratios;
			for (const Model& m : models) {
				const float postY = kCamY + m.post;
				const float aimZ = 25.0f;
				const float aimDist = std::sqrt(m.post * m.post + (cam.z - aimZ) * (cam.z - aimZ));
				const View  v = makeView(cam, { kCamX, postY, aimZ }, XMConvertToDegrees(2.0f * std::atan(45.0f / aimDist)), upZ);
				fillDepth(v, postY);
				const float op = m.tauV / w0;
				float       g[3], c[3];
				const std::vector<float>* cols[3]{ &colsU, &colsN, &colsA };
				ID3D11ShaderResourceView* srvs[3]{ srvU, srvN, srvA };
				const float               boxes[3]{ boxOff, boxOff, boxOn };
				for (int k = 0; k < 3; ++k) {
					setCb(v, op, boxes[k], 0.0f);
					g[k] = gpuLine(march(srvs[k]), kN / 2);
					c[k] = cpuLine(*cols[k], cam, postY, op);
					good = good && std::isfinite(g[k]) && std::fabs(g[k] - c[k]) <= 2.0f;
					worst = std::max(worst, std::isfinite(g[k]) ? std::fabs(g[k] - c[k]) : 1e9f);
				}
				good = good && g[2] < g[1] && g[1] < g[0];
				const float dropN = g[0] - g[1], dropA = g[0] - g[2];
				const float ratio = dropA / std::max(dropN, 1e-6f);
				detail += "; " + std::to_string(static_cast<int>(m.post)) + " units, tau " + f2(m.tauV) + ": GPU/CPU " + f2(g[0]) + "/" + f2(c[0]) + ", " + f2(g[1]) +
				          "/" + f2(c[1]) + ", " + f2(g[2]) + "/" + f2(c[2]) + " (model " + f2(m.u) + ", " + f2(m.n) + ", " + f2(m.a) + ")";
				ratios += std::string(ratios.empty() ? "" : "; ") + std::to_string(static_cast<int>(m.post)) + " units, tau " + f2(m.tauV) + ": drop " + f2(dropN) +
				          " -> " + f2(dropA) + " = x " + f2(ratio) + " " + (ratio >= 2.0f ? "met" : "NOT met") + " (model x " +
				          f2((m.u - m.a) / (m.u - m.n)) + ")";
			}
			Check(good, "fog line on a post follows the lane's top (GPU = CPU)",
				"height where alpha is 0.5, undisturbed / SurfaceResponse 0 / 1, limit 2 units, worst " + f2(worst) + detail);
			std::printf("INFO fog line drop with SurfaceResponse 1 against 0, the limit set before measuring (at least x 2) : %s\n", ratios.c_str());
		}

		// 2. Ground pixels along the lane: the same fog with the lower top (the mass is the same in every column).
		{
			const XMFLOAT3 cam{ kCamX, kCamY, 140.0f };
			const View     v = makeView(cam, { kCamX, kCamY + 600.0f, 0.0f }, 60.0f, upZ);
			fillDepth(v, NAN);
			float       worst = 0.0f, lane = 0.0f;
			int         n = 0;
			std::string detail;
			for (const float tauV : { 1.0f, 2.3f }) {
				const float op = tauV / w0;
				setCb(v, op, boxOff, 0.0f);
				const auto imgU = march(srvU);
				const auto imgN = march(srvN);
				setCb(v, op, boxOn, 0.0f);
				const auto imgA = march(srvA);
				float      w = 0.0f, l = 0.0f;
				for (UINT y = 0; y < kN; ++y) {
					for (UINT x = 0; x < kN; ++x) {
						const std::size_t i = static_cast<std::size_t>(y) * kN + x;
						if (!std::isfinite(hitT[i]) || std::isfinite(hitZ[i])) continue;
						const XMFLOAT3 d = pixelDir(v, x, y);
						const float    gx = cam.x + d.x * hitT[i], gy = cam.y + d.y * hitT[i];
						// The whole ray inside the lane's even part, and its foot inside the march's circle (0.475 x kExtent from the
						// centre, a texel of margin): a ray cut off by the circle crosses only part of the column, where the two
						// tops hold different shares of the same mass.
						if (gx < kLaneIn0 || gx > kLaneIn1 || std::hypot(gx, gy) > 0.475f * kExtent - 64.0f) continue;
						w = std::max(w, std::fabs(imgA[i * 4] - imgN[i * 4]));
						l = std::max(l, std::fabs(imgN[i * 4] - imgU[i * 4]));
						++n;
					}
				}
				worst = std::max(worst, w);
				lane = std::max(lane, l);
				detail += "; tau " + f2(tauV) + ": largest |alpha(1) - alpha(0)| " + std::to_string(w) + " (the lane against no lane up to " + std::to_string(l) + ")";
			}
			Check(n > 50 && worst <= 0.01f && lane > 0.02f, "ground seen along the lane keeps its fog",
				std::to_string(n) + " pixel samples, limit 0.01" + detail);
		}

		// 3. A crest 1.6 x Thickness tall (shape 1, s at its cap) inside the box FogRenderer sets with the response on.
		{
			const XMFLOAT3 cam{ kCamX, kCamY, 260.0f };
			const float    op = 1.0f / w0;
			bool           good = true;
			std::string    detail;
			for (const float aimZ : { 150.0f, 190.0f }) {
				const View v = makeView(cam, { kCamX, 0.0f, aimZ }, 20.0f, upZ);
				fillDepth(v, 0.0f);
				setCb(v, op, boxOn, 0.0f);
				const auto imgOn = march(srvC);
				setCb(v, op, boxOff, 0.0f);
				const auto imgOff = march(srvC);
				float      gOn = 0.0f, gOff = 0.0f, cpu = 0.0f;
				for (UINT dy = 0; dy < 2; ++dy) {
					for (UINT dx = 0; dx < 2; ++dx) {
						const UINT        x = kN / 2 - 1 + dx, y = kN / 2 - 1 + dy;
						const std::size_t i = static_cast<std::size_t>(y) * kN + x;
						gOn += imgOn[i * 4] / 4.0f;
						gOff += imgOff[i * 4] / 4.0f;
						const XMFLOAT3 d = pixelDir(v, x, y);
						cpu += CpuFog(colsC, cam, d, hitT[i], op) / 4.0f;
					}
				}
				good = good && std::isfinite(gOn) && std::fabs(gOn - cpu) <= 0.005f + 0.02f * cpu && gOff <= 0.01f && cpu > (aimZ < 170.0f ? 0.2f : 0.01f);
				detail += "; post height " + f2(aimZ) + ": box " + f2(boxOn) + " GPU " + std::to_string(gOn) + ", CPU " + std::to_string(cpu) + ", old box " + f2(boxOff) +
				          " GPU " + std::to_string(gOff);
			}
			Check(good, "a crest 1.6 x Thickness tall is not cut by the march box", "column " + f2(crestH) + " units, W " + f2(crestW) + detail);
		}

		// 4. Relief shading, straight down from 2400 units over uniform columns: the lane texels at relief -0.309, a block at -0.536, a
		// strip at +0.12 (a bank), the rest 0.
		{
			std::vector<float> rel(static_cast<std::size_t>(kRes) * kRes, 0.0f);
			for (int y = 0; y < kRes; ++y) {
				for (int x = 0; x < kRes; ++x) {
					float& r = rel[static_cast<std::size_t>(y) * kRes + x];
					if (x >= kLane0 && x <= kLane1) r = kQ - 1.0f;
					if (x >= 40 && x <= 45 && y >= 40 && y <= 45) r = 0.464f - 1.0f;
					if (x >= 24 && x <= 26) r = 0.12f;
				}
			}
			auto*          relSrv = makeSrv(DXGI_FORMAT_R32_FLOAT, rel.data(), kRes * sizeof(float));
			const XMFLOAT3 cam{ 300.0f, 300.0f, 2400.0f };
			const View     v = makeView(cam, { 300.0f, 300.0f, 0.0f }, 70.0f, XMVectorSet(0, 1, 0, 0));
			fillDepth(v, NAN);
			auto pixelOf = [&](float a_x, float a_y) {
				XMFLOAT4 clip;
				XMStoreFloat4(&clip, XMVector4Transform(XMVectorSet(a_x - cam.x, a_y - cam.y, -cam.z, 1), XMLoadFloat4x4(&v.vp)));
				const int px = static_cast<int>((clip.x / clip.w + 1.0f) * 0.5f * kN);
				const int py = static_cast<int>((1.0f - clip.y / clip.w) * 0.5f * kN);
				return static_cast<std::size_t>(std::clamp(py, 0, static_cast<int>(kN) - 1)) * kN + static_cast<std::size_t>(std::clamp(px, 0, static_cast<int>(kN) - 1));
			};
			const std::size_t pLane = pixelOf(32.0f, 300.0f), pSpot = pixelOf(704.0f, 704.0f), pBank = pixelOf(-416.0f, 300.0f), pPlain = pixelOf(300.0f, -500.0f);
			const float       op = 1.0f / w0;
			bool              good = relSrv != nullptr;
			std::string       detail;
			for (const float gain : { 0.08f, 0.25f }) {
				setCb(v, op, boxOff, gain);
				const auto  img = composite(srvU, relSrv);
				const float plain = img[pPlain * 4];
				const float lane = img[pLane * 4] / plain, spot = img[pSpot * 4] / plain, bank = img[pBank * 4] / plain;
				const float wantLane = 1.0f + gain * (kQ - 1.0f), wantSpot = 1.0f + gain * (0.464f - 1.0f);
				good = good && img[pPlain * 4 + 3] > 0.05f && std::fabs(lane - wantLane) <= 0.005f && std::fabs(spot - wantSpot) <= 0.005f;
				char b[200];
				std::snprintf(b, sizeof(b), "; G %.2f: trough %.4f (want %.4f), deepest %.4f (want %.4f), bank %.4f (want %.4f), plain colour %.4f", gain, lane, wantLane,
					spot, wantSpot, bank, 1.0f + gain * 0.12f, plain);
				detail += b;
			}
			setCb(v, op, boxOff, 0.0f);
			const auto offBound = composite(srvU, relSrv);
			const auto offFree = composite(srvU, nullptr);
			setCb(v, op, boxOff, 0.08f);
			const auto gainFree = composite(srvU, nullptr);
			auto       bits = [](const std::vector<float>& a_a, const std::vector<float>& a_b) {
				return a_a.size() == a_b.size() && std::memcmp(a_a.data(), a_b.data(), a_a.size() * sizeof(float)) == 0;
			};
			const bool same = bits(offBound, offFree) && bits(gainFree, offFree);
			Check(good && same, "relief shading darkens a trough by the gain",
				std::string("G 0 bound / unbound and G 0.08 unbound the same bit for bit: ") + (same ? "yes" : "NO") + detail);
			SafeRelease(relSrv);
		}

		SafeRelease(srvU);
		SafeRelease(srvN);
		SafeRelease(srvA);
		SafeRelease(srvC);
		SafeRelease(cb);
		SafeRelease(depthSrv);
		SafeRelease(depthTex);
		SafeRelease(marchSrv);
		SafeRelease(marchRtv);
		SafeRelease(compRtv);
		SafeRelease(compTex);
		SafeRelease(marchTex);
	}

	// ---- wake evaporation (ini [Wind] WakeEvaporate, WindShaders.h WakeDose) ----
	// The gates G3-G9 were written into DEBUG-LEDGER.md "2026-11-14 15:xx 후류 증발" and committed (35a414f) before this ran;
	// the limits below are those, not chosen after measuring. Conditions: the game's texel (32 units) on a 128-texel field
	// (4096 units, origin 0), R 70, flat ground, no wind, vorticity 2, gusts at GustMinimum, 2 s of the field alone first, a
	// body from 0.1 to 0.55 of the field along +x at 400 units/s, WispAmount 0.5. Frames as WindField::Update takes them: the
	// body's position and speed in real time, the field's dt clamped to 1/15 s and taken in two steps above 1/30 s, the wake's
	// path (distance moved, straight run) from real time. "Untouched" = the same frames and pattern with no body.
	// Screen: the plugin's march and composite at scale 1 (8 x 8 pixels, 2 degrees) aimed at a ground point from 250 units
	// up at 40 degrees, composite alpha averaged over the pixels; the columns are the field's own BuildColumns.
	constexpr int   kWakeRes = 128;
	constexpr float kWakeCell = 32.0f;
	constexpr float kWakeExtent = kWakeRes * kWakeCell;
	constexpr int   kWakeHSize = 33;
	constexpr float kWakeTest = 0.7f;      // the test value (ledger)
	constexpr float kWakeSpeed = 400.0f;   // the game's run as the profiler saw it: 341..442
	constexpr float kWakeOpacity = 0.216f; // 0.18 per metre x the fog weather's 1.2 (FogRenderer rc.fog[0])
	constexpr float kWakeEye = 250.0f;
	constexpr float kWakeTheta = 40.0f;
	constexpr UINT  kWakeN = 8;

	struct WakeSpec
	{
		std::vector<float> frames{ 1.0f / 30.0f };   // real frame times, repeated in this order
		float              e{ 0.0f };                 // WakeEvaporate
		bool               feed{ true };              // false: the same frames with no body (the untouched field)
		float              wisp{ 0.5f };
		float              speed{ kWakeSpeed };
		float              follow{ Settings::actorFollow };
		float              pathEnd{ 0.55f };          // the body stops being fed once its path is (pathEnd - 0.1) x the field long
		int                turn{ 0 };                 // 0 straight; 1: at the field's centre turn to +y; 2: at the centre turn back to -x
		float              after{ 0.0f };             // seconds of frames after the body is gone
		std::function<void(float, const std::vector<float>&)> sample;   // with a sample time: called on frames past it (seconds since the body's first frame, rho)
		float              sampleEvery{ 0.0f };
	};

	struct WakeField
	{
		std::vector<float> rho, delta;
		float              bodyX{ 0.0f }, bodyY{ 0.0f }, hx{ 1.0f }, hy{ 0.0f };   // the body where it was fed last, its heading
		float              x0{ 0.0f }, yMid{ 0.0f }, time{ 0.0f };
		std::vector<std::pair<float, float>> path;   // every position the body was fed at
		bool               finite{ true };
		int                frames{ 0 };
	};

	std::vector<float> ReadWakeChannel(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11Texture2D* a_tex, int a_channels, int a_channel)
	{
		const auto all = ReadF32(a_dev, a_ctx, a_tex, kWakeRes, kWakeRes, a_channels);
		std::vector<float> out(static_cast<std::size_t>(kWakeRes) * kWakeRes);
		for (std::size_t i = 0; i < out.size(); ++i) out[i] = all[i * a_channels + a_channel];
		return out;
	}

	WakeField RunWake(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_heights, const WakeSpec& a_s)
	{
		a_sim.Reset(a_ctx);
		WindShaders::SimCB cb{};
		cb.grid[2] = kWakeCell;
		cb.grid[3] = static_cast<float>(kWakeRes);
		cb.wind[2] = Settings::windRelax;
		cb.wind[3] = 2.0f;
		cb.gust[0] = Settings::gustMinimum;
		cb.gust[1] = Settings::gustWavelength;
		cb.gust[2] = Settings::fogRefill;
		cb.gust[3] = Settings::velocityDissipation;
		cb.shift[3] = static_cast<float>(kWakeHSize);
		cb.counts[2] = Settings::wallSlope;
		cb.counts[3] = a_s.wisp;
		cb.heightGrid[2] = kWakeExtent / (kWakeHSize - 1);
		cb.heightGrid[3] = 1.0f;
		cb.dragParams[0] = Settings::actorPushWalk;
		cb.dragParams[1] = a_s.follow;
		cb.dragParams[3] = Settings::actorPushMinSpeed;
		WakeField out;
		out.x0 = 0.1f * kWakeExtent;
		out.yMid = 0.5f * kWakeExtent + 0.5f * kWakeCell;
		std::size_t f = 0;
		float       t = 0.0f;
		auto frame = [&](float a_realDt, bool a_body, const WindRules::Body& a_b, float a_step, float a_run) {
			const float dt = std::min(a_realDt, WindRules::kLongestFrame);
			const int   steps = WindRules::StepsPerFrame(dt);
			int         count = 0;
			if (a_body && WindRules::Disturbs(a_b, Settings::thickness)) {
				WindRules::SetDrag(cb, count++, a_b, Settings::actorRadius, Settings::actorPush, a_step, a_run);
			}
			cb.counts[0] = static_cast<float>(count);
			cb.dragParams[2] = a_s.e / static_cast<float>(steps);
			cb.shift[2] = dt;
			cb.time[0] = dt / steps;
			for (int s = 1; s <= steps; ++s) {
				cb.time[1] = t + dt * s / steps;
				a_sim.Step(a_ctx, cb, a_heights, Settings::pressureIterations);
			}
			t += dt;
			++out.frames;
		};
		const WindRules::Body none{};
		for (float settled = 0.0f; settled < 2.0f;) {
			const float r = a_s.frames[f++ % a_s.frames.size()];
			frame(r, false, none, 0.0f, 0.0f);
			settled += r;
		}
		float px = out.x0, py = out.yMid, hx = 1.0f, hy = 0.0f, run = 0.0f, length = 0.0f, since = 0.0f, nextSample = 0.0f;
		bool  turned = false;
		auto  sample = [&]() {
			if (!a_s.sample || since + 1e-6f < nextSample) return;
			nextSample += a_s.sampleEvery;
			a_s.sample(since, ReadWakeChannel(a_dev, a_ctx, a_sim.FogTexture(), 2, 0));
		};
		for (int k = 0;; ++k) {
			const float r = a_s.frames[f++ % a_s.frames.size()];
			float       step = 0.0f;
			const float phx = hx, phy = hy;
			if (k > 0) {
				if (!turned && a_s.turn != 0 && px >= 0.5f * kWakeExtent) {
					hx = a_s.turn == 1 ? 0.0f : -1.0f;
					hy = a_s.turn == 1 ? 1.0f : 0.0f;
					turned = true;
				}
				step = a_s.speed * r;
				px += hx * step;
				py += hy * step;
				length += step;
				since += r;
			}
			if (length > (a_s.pathEnd - 0.1f) * kWakeExtent) break;
			const bool straight = run > 0.0f && hx * phx + hy * phy >= WindRules::kWakeTurnCos;
			run = (straight ? run : 0.0f) + step;
			const WindRules::Body body{ px, py, 0.0f, hx * a_s.speed, hy * a_s.speed, 0.0f, false, true, 0.0f };
			frame(r, a_s.feed, body, step, run);
			out.bodyX = px;
			out.bodyY = py;
			out.hx = hx;
			out.hy = hy;
			out.path.emplace_back(px, py);
			sample();
		}
		for (float rest = 0.0f; rest < a_s.after;) {
			const float r = a_s.frames[f++ % a_s.frames.size()];
			frame(r, false, none, 0.0f, 0.0f);
			rest += r;
			since += r;
			sample();
		}
		out.time = t;
		out.rho = ReadWakeChannel(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
		out.delta = ReadWakeChannel(a_dev, a_ctx, a_sim.FogTexture(), 2, 1);
		for (std::size_t i = 0; i < out.rho.size(); ++i) out.finite = out.finite && std::isfinite(out.rho[i]) && std::isfinite(out.delta[i]);
		return out;
	}

	float WakeQ(const WakeField& a_f, std::size_t a_i)
	{
		const float eq = a_f.rho[a_i] - a_f.delta[a_i];
		return eq > 0.05f ? a_f.rho[a_i] / eq : 1.0f;
	}

	bool WakeFullCircle(int a_x, int a_y)
	{
		return 2.0f * std::hypot((a_x + 0.5f) / kWakeRes - 0.5f, (a_y + 0.5f) / kWakeRes - 0.5f) <= 0.55f;
	}

	// Distance behind the body along its heading and across it, for texel (x, y).
	void WakeFrame(const WakeField& a_f, int a_x, int a_y, float& a_behind, float& a_across)
	{
		const float dx = (a_x + 0.5f) * kWakeCell - a_f.bodyX, dy = (a_y + 0.5f) * kWakeCell - a_f.bodyY;
		a_behind = -(dx * a_f.hx + dy * a_f.hy);
		a_across = std::fabs(-dx * a_f.hy + dy * a_f.hx);
	}

	struct WakeStats
	{
		float  trailQ{ NAN };                  // mean q on the body's line 2..6R behind (texel centres)
		int    trailN{ 0 };
		float  lowQ{ 10.0f }, lowBehind{ 0.0f }; // the lowest q inside the full circle and how far behind the body it lies
		float  sideQ{ NAN };                   // mean q 2..6R behind, 3..5R across
		std::vector<float> trailX;             // the trail texels' world x (body heading +x)
	};

	WakeStats StatsOfWake(const WakeField& a_f)
	{
		const float R = WindRules::DragRadius(Settings::actorRadius, 0.0f);
		WakeStats   out;
		double      trail = 0.0, side = 0.0;
		int         sideN = 0;
		for (int y = 0; y < kWakeRes; ++y) {
			for (int x = 0; x < kWakeRes; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * kWakeRes + x;
				const float       q = WakeQ(a_f, i);
				float             b, c;
				WakeFrame(a_f, x, y, b, c);
				if (WakeFullCircle(x, y) && q < out.lowQ) {
					out.lowQ = q;
					out.lowBehind = b;
				}
				if (b >= 2.0f * R && b <= 6.0f * R) {
					if (c < 0.5f * kWakeCell) {
						trail += q;
						++out.trailN;
						out.trailX.push_back((x + 0.5f) * kWakeCell);
					} else if (c >= 3.0f * R && c <= 5.0f * R) {
						side += q;
						++sideN;
					}
				}
			}
		}
		out.trailQ = out.trailN ? static_cast<float>(trail / out.trailN) : NAN;
		out.sideQ = sideN ? static_cast<float>(side / sideN) : NAN;
		return out;
	}

	// Columns of one build: low layer (H, W, ground x known, known) and the tall layer (H, W).
	struct WakeCols
	{
		std::vector<float> low, upper;
		float              top{ 0.0f };   // the march box's top for them
	};

	WakeCols BuildWakeCols(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_heights,
		ID3D11ShaderResourceView* a_pools, float a_poolDepth, float a_response, float a_wisp, float a_time)
	{
		a_sim.BuildColumns(a_ctx, a_heights, Settings::thickness, a_time, 0.0f, 0.0f, a_pools, { 1.0f, 0.0f, 1.0f, 0.0f },
			{ 0.0f, 0.0f, 0.0f, a_pools ? Settings::placePoolDensity : 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { a_wisp, 0.0f, a_response, 0.0f });
		WakeCols out;
		out.low = ReadF32(a_dev, a_ctx, a_sim.ColumnTexture(), kWakeRes, kWakeRes, 4);
		out.upper = ReadF32(a_dev, a_ctx, a_sim.UpperTexture(), kWakeRes, kWakeRes, 2);
		out.top = std::max(WindShaders::LowLayerTop(Settings::thickness, a_response), a_poolDepth) + 2.0f;
		return out;
	}

	// The same from a stored field: written back into the field's fog texture first (rho, delta), then built.
	WakeCols BuildWakeColsFrom(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_heights, const WakeField& a_f,
		ID3D11ShaderResourceView* a_pools, float a_poolDepth, float a_response)
	{
		std::vector<float> fog(a_f.rho.size() * 2);
		for (std::size_t i = 0; i < a_f.rho.size(); ++i) {
			fog[i * 2] = a_f.rho[i];
			fog[i * 2 + 1] = a_f.delta[i];
		}
		a_ctx->UpdateSubresource(a_sim.FogTexture(), 0, nullptr, fog.data(), kWakeRes * 2 * sizeof(float), 0);
		return BuildWakeCols(a_dev, a_ctx, a_sim, a_heights, a_pools, a_poolDepth, a_response, 0.5f, a_f.time);
	}

	// The plugin's march and composite over given columns: composite alpha averaged over a small screen aimed at each ground point.
	class WakeScreen
	{
	public:
		WakeScreen(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11VertexShader* a_vs, ID3D11PixelShader* a_march, ID3D11PixelShader* a_comp,
			ID3D11SamplerState* a_linear) :
			_dev(a_dev), _ctx(a_ctx), _vs(a_vs), _march(a_march), _comp(a_comp), _linear(a_linear)
		{
			D3D11_TEXTURE2D_DESC td{};
			td.Width = td.Height = kWakeN;
			td.MipLevels = td.ArraySize = 1;
			td.SampleDesc.Count = 1;
			td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
			_ok = SUCCEEDED(a_dev->CreateTexture2D(&td, nullptr, &_marchTex)) && SUCCEEDED(a_dev->CreateRenderTargetView(_marchTex, nullptr, &_marchRtv)) &&
			      SUCCEEDED(a_dev->CreateShaderResourceView(_marchTex, nullptr, &_marchSrv)) && SUCCEEDED(a_dev->CreateTexture2D(&td, nullptr, &_compTex)) &&
			      SUCCEEDED(a_dev->CreateRenderTargetView(_compTex, nullptr, &_compRtv));
			td.Format = DXGI_FORMAT_R32_FLOAT;
			td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			std::vector<float>     depth(kWakeN * kWakeN, 1.0f);
			D3D11_SUBRESOURCE_DATA dinit{ depth.data(), kWakeN * sizeof(float), 0 };
			_ok = _ok && SUCCEEDED(a_dev->CreateTexture2D(&td, &dinit, &_depthTex)) && SUCCEEDED(a_dev->CreateShaderResourceView(_depthTex, nullptr, &_depthSrv));
			D3D11_BUFFER_DESC cbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
			_ok = _ok && SUCCEEDED(a_dev->CreateBuffer(&cbd, nullptr, &_cb));
		}
		~WakeScreen()
		{
			SafeRelease(_cb);
			SafeRelease(_depthSrv);
			SafeRelease(_depthTex);
			SafeRelease(_compRtv);
			SafeRelease(_compTex);
			SafeRelease(_marchSrv);
			SafeRelease(_marchRtv);
			SafeRelease(_marchTex);
		}
		bool Ok() const { return _ok; }

		// Composite alpha seen from a_eye units up at a_theta degrees toward ground point (a_x, a_y), the camera standing back along -x.
		float Alpha(const WakeCols& a_cols, float a_x, float a_y, float a_eye, float a_theta, float a_opacity)
		{
			using namespace DirectX;
			ID3D11ShaderResourceView* low = MakeSrv(DXGI_FORMAT_R32G32B32A32_FLOAT, a_cols.low.data(), kWakeRes * 16);
			ID3D11ShaderResourceView* up = MakeSrv(DXGI_FORMAT_R32G32_FLOAT, a_cols.upper.data(), kWakeRes * 8);
			if (!low || !up) {
				SafeRelease(low);
				SafeRelease(up);
				return NAN;
			}
			const float    back = a_eye / std::tan(XMConvertToRadians(a_theta));
			const XMFLOAT3 cam{ a_x - back, a_y, a_eye };
			const XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(2.0f), 1.0f, 5.0f, 100000.0f);
			XMFLOAT4X4     pf;
			XMStoreFloat4x4(&pf, proj);
			const XMVECTOR fwd = XMVector3Normalize(XMVectorSet(back, 0.0f, -a_eye, 0.0f));
			const XMMATRIX vp = XMMatrixMultiply(XMMatrixLookToLH(XMVectorZero(), fwd, XMVectorSet(0, 0, 1, 0)), proj);
			XMFLOAT4X4     vpf, inv;
			XMStoreFloat4x4(&vpf, vp);
			XMStoreFloat4x4(&inv, XMMatrixInverse(nullptr, vp));
			std::vector<float> depth(kWakeN * kWakeN, 1.0f);
			for (UINT y = 0; y < kWakeN; ++y) {
				for (UINT x = 0; x < kWakeN; ++x) {
					XMFLOAT4 w;
					XMStoreFloat4(&w, XMVector4Transform(XMVectorSet((x + 0.5f) / kWakeN * 2 - 1, 1 - (y + 0.5f) / kWakeN * 2, 0.5f, 1), XMLoadFloat4x4(&inv)));
					const float dz = w.z / w.w;
					if (dz < -1e-6f) {
						const float s = -cam.z / dz;
						XMFLOAT4    clip;
						XMStoreFloat4(&clip, XMVector4Transform(XMVectorSet(w.x / w.w * s, w.y / w.w * s, w.z / w.w * s, 1), XMLoadFloat4x4(&vpf)));
						depth[y * kWakeN + x] = clip.z / clip.w;
					}
				}
			}
			_ctx->UpdateSubresource(_depthTex, 0, nullptr, depth.data(), kWakeN * sizeof(float), 0);
			RaymarchShaders::RayCB rc{};
			std::memcpy(rc.invViewProj, &inv, sizeof(rc.invViewProj));
			rc.field[0] = 0.0f;
			rc.field[1] = 0.0f;
			rc.field[2] = kWakeExtent;
			rc.field[3] = 1.0f;
			rc.box[0] = -1.0f;
			rc.box[1] = a_cols.top;
			rc.box[2] = 100000.0f;
			rc.box[3] = static_cast<float>(kSteps);
			rc.fog[0] = a_opacity;
			rc.fog[1] = kSteepness;
			rc.depth[0] = pf._33;
			rc.depth[1] = pf._43;
			rc.depth[2] = static_cast<float>(kWakeN);
			rc.depth[3] = static_cast<float>(kWakeN);
			rc.lowRes[0] = static_cast<float>(kWakeN);
			rc.lowRes[1] = static_cast<float>(kWakeN);
			rc.lowRes[2] = 1.0f;
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
			_ctx->UpdateSubresource(_cb, 0, nullptr, &rc, 0, 0);
			const float black[4]{};
			const D3D11_VIEWPORT vpt{ 0.0f, 0.0f, static_cast<float>(kWakeN), static_cast<float>(kWakeN), 0.0f, 1.0f };
			_ctx->RSSetViewports(1, &vpt);
			_ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);
			_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			_ctx->VSSetShader(_vs, nullptr, 0);
			_ctx->PSSetConstantBuffers(0, 1, &_cb);
			_ctx->PSSetSamplers(0, 1, &_linear);
			ID3D11ShaderResourceView* none[5]{};
			// march
			_ctx->ClearRenderTargetView(_marchRtv, black);
			_ctx->OMSetRenderTargets(1, &_marchRtv, nullptr);
			_ctx->PSSetShader(_march, nullptr, 0);
			ID3D11ShaderResourceView* in[5]{ _depthSrv, low, nullptr, nullptr, up };
			_ctx->PSSetShaderResources(0, 5, in);
			_ctx->Draw(3, 0);
			_ctx->PSSetShaderResources(0, 5, none);
			// composite
			_ctx->ClearRenderTargetView(_compRtv, black);
			_ctx->OMSetRenderTargets(1, &_compRtv, nullptr);
			_ctx->PSSetShader(_comp, nullptr, 0);
			ID3D11ShaderResourceView* cin[5]{ _depthSrv, low, _marchSrv, nullptr, up };
			_ctx->PSSetShaderResources(0, 5, cin);
			_ctx->Draw(3, 0);
			_ctx->PSSetShaderResources(0, 5, none);
			_ctx->OMSetRenderTargets(0, nullptr, nullptr);
			const auto img = ReadF32(_dev, _ctx, _compTex, kWakeN, kWakeN, 4);
			SafeRelease(low);
			SafeRelease(up);
			double sum = 0.0;
			for (UINT i = 0; i < kWakeN * kWakeN; ++i) sum += img[i * 4 + 3];
			return static_cast<float>(sum / (kWakeN * kWakeN));
		}

	private:
		ID3D11ShaderResourceView* MakeSrv(DXGI_FORMAT a_fmt, const void* a_data, UINT a_pitch)
		{
			D3D11_TEXTURE2D_DESC d{};
			d.Width = d.Height = kWakeRes;
			d.MipLevels = d.ArraySize = 1;
			d.Format = a_fmt;
			d.SampleDesc.Count = 1;
			d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			D3D11_SUBRESOURCE_DATA    init{ a_data, a_pitch, 0 };
			ID3D11Texture2D*          tex = nullptr;
			ID3D11ShaderResourceView* srv = nullptr;
			if (SUCCEEDED(_dev->CreateTexture2D(&d, &init, &tex))) _dev->CreateShaderResourceView(tex, nullptr, &srv);
			SafeRelease(tex);
			return srv;
		}

		ID3D11Device*             _dev;
		ID3D11DeviceContext*      _ctx;
		ID3D11VertexShader*       _vs;
		ID3D11PixelShader*        _march;
		ID3D11PixelShader*        _comp;
		ID3D11SamplerState*       _linear;
		ID3D11Texture2D*          _marchTex{ nullptr };
		ID3D11Texture2D*          _compTex{ nullptr };
		ID3D11Texture2D*          _depthTex{ nullptr };
		ID3D11RenderTargetView*   _marchRtv{ nullptr };
		ID3D11RenderTargetView*   _compRtv{ nullptr };
		ID3D11ShaderResourceView* _marchSrv{ nullptr };
		ID3D11ShaderResourceView* _depthSrv{ nullptr };
		ID3D11Buffer*             _cb{ nullptr };
		bool                      _ok{ false };
	};

	ID3D11ShaderResourceView* MakeWakeGrid(ID3D11Device* a_dev, DXGI_FORMAT a_fmt, const std::vector<float>& a_data, UINT a_channels)
	{
		D3D11_TEXTURE2D_DESC d{};
		d.Width = d.Height = kWakeHSize;
		d.MipLevels = d.ArraySize = 1;
		d.Format = a_fmt;
		d.SampleDesc.Count = 1;
		d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA    init{ a_data.data(), kWakeHSize * sizeof(float) * a_channels, 0 };
		ID3D11Texture2D*          tex = nullptr;
		ID3D11ShaderResourceView* srv = nullptr;
		if (SUCCEEDED(a_dev->CreateTexture2D(&d, &init, &tex))) a_dev->CreateShaderResourceView(tex, nullptr, &srv);
		SafeRelease(tex);
		return srv;
	}

	std::string F3(double a_v)
	{
		char b[32];
		std::snprintf(b, sizeof(b), "%.3f", a_v);
		return b;
	}

	void RunWakeChecks(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11VertexShader* a_vs, ID3D11PixelShader* a_march, ID3D11PixelShader* a_comp,
		ID3D11SamplerState* a_linear)
	{
		const float R = WindRules::DragRadius(Settings::actorRadius, 0.0f);
		WindSim     sim;
		std::string error;
		if (!sim.Create(a_dev, kWakeRes, error, g_fogSampling)) {
			Check(false, "wake evaporation checks: field", error);
			return;
		}
		auto*      heights = MakeWakeGrid(a_dev, DXGI_FORMAT_R32_FLOAT, std::vector<float>(kWakeHSize * kWakeHSize, 0.0f), 1);
		WakeScreen screen(a_dev, a_ctx, a_vs, a_march, a_comp, a_linear);
		if (!heights || !screen.Ok()) {
			Check(false, "wake evaporation checks: resources", "creation failed");
			SafeRelease(heights);
			sim.Release();
			return;
		}
		auto pools = [&](float a_depth) {
			std::vector<float> p(static_cast<std::size_t>(kWakeHSize) * kWakeHSize * 4, 0.0f);
			for (std::size_t i = 0; i < p.size(); i += 4) {
				p[i] = 1.0f;
				p[i + 3] = a_depth;
			}
			return MakeWakeGrid(a_dev, DXGI_FORMAT_R32G32B32A32_FLOAT, p, 4);
		};
		// Relative drop of the summed alpha over the trail's ground points, trail against untouched.
		auto drop = [&](const WakeCols& a_trail, const WakeCols& a_untouched, const WakeStats& a_st, float a_y, float a_eye, float& a_aTrail, float& a_aUntouched) {
			double t = 0.0, u = 0.0;
			for (float x : a_st.trailX) {
				t += screen.Alpha(a_trail, x, a_y, a_eye, kWakeTheta, kWakeOpacity);
				u += screen.Alpha(a_untouched, x, a_y, a_eye, kWakeTheta, kWakeOpacity);
			}
			const std::size_t n = std::max<std::size_t>(a_st.trailX.size(), 1);
			a_aTrail = static_cast<float>(t / n);
			a_aUntouched = static_cast<float>(u / n);
			return u > 0.0 ? static_cast<float>(1.0 - t / u) : NAN;
		};

		// G3, G4, G5, G8 over the frame rates; G9 on the 30 fps fields.
		struct Rate
		{
			const char*        name;
			std::vector<float> frames;
			float              qLimit, aLimit;
		};
		const Rate rates[] = { { "30 fps", { 1.0f / 30.0f }, 0.45f, 0.55f }, { "20 fps", { 1.0f / 20.0f }, 0.45f, 0.55f }, { "15 fps", { 1.0f / 15.0f }, 0.45f, 0.55f },
			{ "13 fps", { 1.0f / 13.0f }, 0.55f, 0.45f },
			{ "mixed (1/30 s x 5, then 0.2 s)", { 1.0f / 30.0f, 1.0f / 30.0f, 1.0f / 30.0f, 1.0f / 30.0f, 1.0f / 30.0f, 0.2f }, 0.55f, 0.45f } };
		bool        depthOk = true, screenOk = true, lowOk = true, sideOk = true, finite = true;
		std::string depthText, screenText, lowText, sideText;
		WakeField   on30, off30, none30;
		for (const Rate& r : rates) {
			WakeSpec s;
			s.frames = r.frames;
			s.e = kWakeTest;
			const WakeField on = RunWake(a_dev, a_ctx, sim, heights, s);
			const WakeCols  cOn = BuildWakeCols(a_dev, a_ctx, sim, heights, nullptr, 0.0f, 1.0f, s.wisp, on.time);
			s.e = 0.0f;
			const WakeField off = RunWake(a_dev, a_ctx, sim, heights, s);
			const WakeCols  cOff = BuildWakeCols(a_dev, a_ctx, sim, heights, nullptr, 0.0f, 1.0f, s.wisp, off.time);
			s.feed = false;
			const WakeField none = RunWake(a_dev, a_ctx, sim, heights, s);
			const WakeCols  cNone = BuildWakeCols(a_dev, a_ctx, sim, heights, nullptr, 0.0f, 1.0f, s.wisp, none.time);
			const WakeStats sOn = StatsOfWake(on), sOff = StatsOfWake(off);
			finite = finite && on.finite && off.finite && none.finite;
			float       aOn, aU, aOff, aU2;
			const float dOn = drop(cOn, cNone, sOn, on.yMid, kWakeEye, aOn, aU);
			const float dOff = drop(cOff, cNone, sOn, on.yMid, kWakeEye, aOff, aU2);
			depthOk = depthOk && sOn.trailN > 0 && sOn.trailQ <= r.qLimit;
			screenOk = screenOk && dOn >= r.aLimit;
			lowOk = lowOk && sOn.lowQ >= 0.15f && sOn.lowBehind >= 0.0f;
			sideOk = sideOk && std::fabs(sOn.sideQ - sOff.sideQ) <= 0.02f;
			const std::string sep = depthText.empty() ? "" : "; ";
			depthText += sep + r.name + ": " + F3(sOn.trailQ) + " (limit " + F3(r.qLimit) + "; WakeEvaporate 0: " + F3(sOff.trailQ) + "; " + std::to_string(sOn.trailN) + " texels, " +
			             std::to_string(on.frames) + " frames)";
			screenText += sep + r.name + ": drop " + F3(dOn) + " (limit " + F3(r.aLimit) + "), alpha " + F3(aU) + " -> " + F3(aOn) + "; WakeEvaporate 0: drop " + F3(dOff) +
			              ", alpha -> " + F3(aOff);
			lowText += sep + r.name + ": lowest q " + F3(sOn.lowQ) + " at " + F3(sOn.lowBehind / R) + " R behind (WakeEvaporate 0: " + F3(sOff.lowQ) + " at " +
			           F3(sOff.lowBehind / R) + " R)";
			sideText += sep + r.name + ": " + F3(sOn.sideQ) + " / " + F3(sOff.sideQ);
			if (r.frames.size() == 1 && r.frames[0] == 1.0f / 30.0f) {
				on30 = on;
				off30 = off;
				none30 = none;
			}
		}
		Check(finite && depthOk, "wake: the trail 2-6R behind a runner thins by at least 55 % (45 % at 13 fps and mixed frames)",
			"WakeEvaporate 0.70, run 400, wisps 0.5, mean q on the body's line 2..6R behind: " + depthText);
		Check(finite && screenOk, "wake: the trail's alpha seen at 40 degrees drops by at least 55 % (45 % at 13 fps and mixed frames)",
			"GPU march + composite, eye 250, opacity 0.216/m, SurfaceResponse 1, ground points on the body's line 2..6R behind, 1 - sum alpha (trail) / sum alpha (untouched): " +
				screenText);
		Check(finite && lowOk, "wake: the lowest q stays at or above 0.15 and never ahead of the body", lowText);
		Check(finite && sideOk, "wake: the band beside the trail is unchanged within 0.02", "mean q 2..6R behind, 3..5R across, WakeEvaporate 0.70 / 0: " + sideText);

		// G7a: ground behind the start, beyond 2R + 32 units, within 3R of the body's line, on the 30 fps fields.
		{
			float worst = 0.0f;
			int   n = 0;
			for (int y = 0; y < kWakeRes; ++y) {
				for (int x = 0; x < kWakeRes; ++x) {
					const float wx = (x + 0.5f) * kWakeCell, wy = (y + 0.5f) * kWakeCell;
					if (wx >= on30.x0 - (2.0f * R + 32.0f) || std::fabs(wy - on30.yMid) > 3.0f * R) continue;
					const std::size_t i = static_cast<std::size_t>(y) * kWakeRes + x;
					worst = std::max(worst, std::fabs(WakeQ(on30, i) - WakeQ(off30, i)));
					++n;
				}
			}
			Check(on30.finite && off30.finite && n > 0 && worst <= 0.02f, "wake: ground behind where the run began is untouched",
				std::to_string(n) + " texels more than 2R + 32 behind the start within 3R of the line: largest |q(0.70) - q(0)| " + F3(worst) + " (limit 0.02)");
		}

		// G9: pooled fog on the 30 fps fields.
		{
			struct Pool
			{
				float depth, eye, limit;
				const char* where;
			};
			const Pool cases[] = { { 60.0f, 250.0f, 0.50f, "60 deep, eye 250 (outside the lid)" }, { 200.0f, 250.0f, 0.45f, "200 deep, eye 250 (outside)" },
				{ 500.0f, 250.0f, 0.35f, "500 deep, eye 250 (inside the lid)" }, { 500.0f, 650.0f, 0.35f, "500 deep, eye 650 (outside)" } };
			const WakeStats st = StatsOfWake(on30);
			bool            good = true;
			std::string     text;
			for (const Pool& p : cases) {
				auto* tex = pools(p.depth);
				for (const float k : { 0.0f, 1.0f }) {
					const WakeCols cOn = BuildWakeColsFrom(a_dev, a_ctx, sim, heights, on30, tex, p.depth, k);
					const WakeCols cNone = BuildWakeColsFrom(a_dev, a_ctx, sim, heights, none30, tex, p.depth, k);
					const WakeCols cOff = BuildWakeColsFrom(a_dev, a_ctx, sim, heights, off30, tex, p.depth, k);
					float       aOn, aU, aOff, aU2;
					const float dOn = drop(cOn, cNone, st, on30.yMid, p.eye, aOn, aU);
					const float dOff = drop(cOff, cNone, st, on30.yMid, p.eye, aOff, aU2);
					good = good && tex && dOn >= p.limit;
					char b[256];
					std::snprintf(b, sizeof(b), "%s%s, SurfaceResponse %.0f: drop %.3f (limit %.2f), alpha %.3f -> %.3f; WakeEvaporate 0: drop %.3f", text.empty() ? "" : "; ", p.where,
						k, dOn, p.limit, aU, aOn, dOff);
					text += b;
				}
				SafeRelease(tex);
			}
			Check(good, "wake: pooled fog keeps the accepted trail contrast", "PoolDensity " + F3(Settings::placePoolDensity) + ", 30 fps, 40 degrees: " + text);
		}

		// G5 and G7b: a 90 degree turn at the field's centre (+x, then +y). G7c: the long run (0.1 to 0.9, 47 R). INFO: a U-turn, a walker.
		{
			WakeSpec s;
			s.turn = 1;
			s.pathEnd = 0.55f;
			s.e = kWakeTest;
			const WakeField on = RunWake(a_dev, a_ctx, sim, heights, s);
			s.e = 0.0f;
			const WakeField off = RunWake(a_dev, a_ctx, sim, heights, s);
			const WakeStats st = StatsOfWake(on);
			float           worst = 0.0f;
			int             n = 0;
			for (int y = 0; y < kWakeRes; ++y) {
				for (int x = 0; x < kWakeRes; ++x) {
					const float wx = (x + 0.5f) * kWakeCell, wy = (y + 0.5f) * kWakeCell;
					float       nearest = 1e30f;
					for (const auto& [px, py] : on.path) nearest = std::min(nearest, std::hypot(wx - px, wy - py));
					if (nearest <= 2.0f * R + 64.0f) continue;
					const std::size_t i = static_cast<std::size_t>(y) * kWakeRes + x;
					worst = std::max(worst, std::fabs(WakeQ(on, i) - WakeQ(off, i)));
					++n;
				}
			}
			Check(on.finite && off.finite && st.lowQ >= 0.15f && st.lowBehind >= 0.0f, "wake: a 90 degree turn keeps the lowest q at or above 0.15, not ahead of the body",
				"lowest q " + F3(st.lowQ) + " at " + F3(st.lowBehind / R) + " R behind (WakeEvaporate 0: " + F3(StatsOfWake(off).lowQ) + ")");
			Check(on.finite && off.finite && n > 0 && worst <= 0.02f, "wake: a 90 degree turn leaves the ground off its path untouched",
				std::to_string(n) + " texels more than 2R + 64 from every point of the path: largest |q(0.70) - q(0)| " + F3(worst) + " (limit 0.02)");
		}
		{
			WakeSpec s;
			s.pathEnd = 0.9f;
			s.e = kWakeTest;
			const WakeField on = RunWake(a_dev, a_ctx, sim, heights, s);
			s.e = 0.0f;
			const WakeField off = RunWake(a_dev, a_ctx, sim, heights, s);
			float worst = 0.0f, at = 0.0f, farTake = 0.0f;   // far: the largest 6..40R behind (passed long ago), shown
			std::string profile;
			const int row = static_cast<int>(on.yMid / kWakeCell);
			for (int x = 0; x < kWakeRes; ++x) {
				const float b = on.bodyX - (x + 0.5f) * kWakeCell;
				if (b < 2.0f * R || b > 40.0f * R) continue;
				const std::size_t i = static_cast<std::size_t>(row) * kWakeRes + x;
				const float       take = -std::log(std::max(WakeQ(on, i), 1e-6f) / std::max(WakeQ(off, i), 1e-6f));
				if (take > worst) {
					worst = take;
					at = b / R;
				}
				if (b >= 6.0f * R) farTake = std::max(farTake, take);
				if (x % 8 == 0) profile += (profile.empty() ? "" : ", ") + F3(b / R) + " R: " + F3(take);
			}
			Check(on.finite && off.finite && worst <= 1.5f * kWakeTest, "wake: what a long run evaporates stays bounded",
				"run 0.1 to 0.9 of the field (" + F3((0.8f * kWakeExtent) / R) + " R): largest -ln(q(0.70) / q(0)) on the body's line 2..40R behind " + F3(worst) + " at " + F3(at) +
					" R (limit 1.5 x 0.70 = 1.05); largest 6..40R behind " + F3(farTake) + "; along the line (behind: -ln ratio) " + profile);
			// INFO (ledger 15:xx addition): the same with ActorFollow 0, to see whether the dragged air makes the excess.
			s.follow = 0.0f;
			s.e = kWakeTest;
			const WakeField on0 = RunWake(a_dev, a_ctx, sim, heights, s);
			s.e = 0.0f;
			const WakeField off0 = RunWake(a_dev, a_ctx, sim, heights, s);
			float worst0 = 0.0f, at0 = 0.0f;
			for (int x = 0; x < kWakeRes; ++x) {
				const float b = on0.bodyX - (x + 0.5f) * kWakeCell;
				if (b < 2.0f * R || b > 40.0f * R) continue;
				const std::size_t i = static_cast<std::size_t>(row) * kWakeRes + x;
				const float       take = -std::log(std::max(WakeQ(on0, i), 1e-6f) / std::max(WakeQ(off0, i), 1e-6f));
				if (take > worst0) {
					worst0 = take;
					at0 = b / R;
				}
			}
			std::printf("INFO wake long run with ActorFollow 0 (no limit) : largest -ln(q(0.70) / q(0)) on the body's line 2..40R behind %.3f at %.3f R (with ActorFollow %.0f: %.3f)\n",
				worst0, at0, Settings::actorFollow, worst);
		}
		{
			WakeSpec s;
			s.turn = 2;
			s.pathEnd = 0.7f;   // to the centre (0.4 of the field) and 0.2 of it back
			s.e = kWakeTest;
			const WakeStats on = StatsOfWake(RunWake(a_dev, a_ctx, sim, heights, s));
			s.e = 0.0f;
			const WakeStats off = StatsOfWake(RunWake(a_dev, a_ctx, sim, heights, s));
			WakeSpec w;
			w.speed = 100.0f;
			w.pathEnd = 0.3f;
			w.e = kWakeTest;
			const WakeStats wOn = StatsOfWake(RunWake(a_dev, a_ctx, sim, heights, w));
			w.e = 0.0f;
			const WakeStats wOff = StatsOfWake(RunWake(a_dev, a_ctx, sim, heights, w));
			Check(on.lowQ >= 0.15f, "wake: a U-turn back along its own trail keeps the lowest q at or above 0.15",
				"run 400 to the centre and back along the same line, 30 fps, wisps 0.5 (ledger 15:xx (2): the place is not checked, the body runs onto its own earlier "
				"trail): lowest q " + F3(on.lowQ) + " at " + F3(on.lowBehind / R) + " R behind (WakeEvaporate 0: " + F3(off.lowQ) + ")");
			std::printf("INFO wake (no limit, ledger 2026-11-14 15:xx) : walk 100 (30 fps, wisps 0.5) trail q 2..6R %.3f, lowest %.3f (WakeEvaporate 0: %.3f, %.3f)\n",
				wOn.trailQ, wOn.lowQ, wOff.trailQ, wOff.lowQ);
		}

		// G6: refill at one spot (the field's centre texel on the body's line), 30 fps, no wisps; the body runs on to 0.9 and is gone.
		{
			WakeSpec s;
			s.wisp = 0.0f;
			s.pathEnd = 0.9f;
			s.after = 60.0f;
			s.sampleEvery = 0.25f;
			const float spotX = (kWakeRes / 2 + 0.5f) * kWakeCell;
			const float passAt = (spotX - 0.1f * kWakeExtent) / s.speed;
			const std::size_t spot = static_cast<std::size_t>(kWakeRes / 2) * kWakeRes + kWakeRes / 2;
			auto measure = [&](float a_e, float& a_lowest, float& a_back) {
				a_lowest = 10.0f;
				a_back = -1.0f;
				s.e = a_e;
				s.sample = [&](float a_since, const std::vector<float>& a_rho) {
					const float t = a_since - passAt;
					if (t < 0.0f) return;
					const float q = a_rho[spot];   // no wisps: rho_eq is 1, q = rho
					if (q < a_lowest) {
						a_lowest = q;
						a_back = -1.0f;
					} else if (a_back < 0.0f && q >= 0.95f) {
						a_back = t;
					}
				};
				return RunWake(a_dev, a_ctx, sim, heights, s).finite;
			};
			float lowOn, backOn, lowOff, backOff;
			const bool okOn = measure(kWakeTest, lowOn, backOn);
			const bool okOff = measure(0.0f, lowOff, backOff);
			Check(okOn && okOff && backOn >= 0.0f && backOn <= 45.0f, "wake: a runner's path fills back to within 5 % in 45 s",
				"spot on the path, sampled every 0.25 s from the pass: lowest q " + F3(lowOn) + ", back at 0.95 after " + F3(backOn) + " s (limit 45; -1 = not within 60 s); WakeEvaporate 0: lowest " +
					F3(lowOff) + ", back after " + F3(backOff) + " s");
		}
		SafeRelease(heights);
		sim.Release();
	}
}

namespace
{
	// kBlurPS (ini [Render] FogBlurRadius, 2026-11-16) against a CPU replica of its formula on an 8 x 8 march texture with a
	// depth edge down the middle (scene distance 500 left, 2000 right): radius 1 and 2 at any distance, and radius 1 with
	// a 15 % distance tolerance. The scene distance (.w) must come through unchanged.
	void CheckFogBlur(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx)
	{
		constexpr int N = 8;
		ID3DBlob* vsb = Compile(RaymarchShaders::kVS, "vs_5_0");
		ID3DBlob* bb = Compile(RaymarchShaders::kBlurPS, "ps_5_0");
		if (!vsb || !bb) {
			SafeRelease(vsb);
			SafeRelease(bb);
			Check(false, "fog blur matches its formula", "shaders did not compile");
			return;
		}
		ID3D11VertexShader* vs = nullptr;
		ID3D11PixelShader*  ps = nullptr;
		a_dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
		a_dev->CreatePixelShader(bb->GetBufferPointer(), bb->GetBufferSize(), nullptr, &ps);
		SafeRelease(vsb);
		SafeRelease(bb);
		std::vector<float> in(N * N * 4);
		for (int y = 0; y < N; ++y) {
			for (int x = 0; x < N; ++x) {
				float* t = &in[(y * N + x) * 4];
				t[0] = static_cast<float>((x * 7 + y * 3) % 10) / 10.0f;   // fog amount
				t[1] = 0.2f + 0.05f * x;                                    // brightness
				t[2] = 1000.0f + 10.0f * y;                                 // fog distance
				t[3] = x < N / 2 ? 500.0f : 2000.0f;                        // scene distance
			}
		}
		D3D11_TEXTURE2D_DESC td{};
		td.Width = td.Height = N;
		td.MipLevels = td.ArraySize = 1;
		td.SampleDesc.Count = 1;
		td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA    init{ in.data(), N * 16u, 0 };
		ID3D11Texture2D*          inTex = nullptr;
		ID3D11ShaderResourceView* inSrv = nullptr;
		a_dev->CreateTexture2D(&td, &init, &inTex);
		a_dev->CreateShaderResourceView(inTex, nullptr, &inSrv);
		td.BindFlags = D3D11_BIND_RENDER_TARGET;
		ID3D11Texture2D*        outTex = nullptr;
		ID3D11RenderTargetView* rtv = nullptr;
		a_dev->CreateTexture2D(&td, nullptr, &outTex);
		a_dev->CreateRenderTargetView(outTex, nullptr, &rtv);
		D3D11_BUFFER_DESC cbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
		ID3D11Buffer*     cb = nullptr;
		a_dev->CreateBuffer(&cbd, nullptr, &cb);

		struct Case { int r; float tol; };
		const Case cases[]{ { 1, 0.0f }, { 2, 0.0f }, { 1, 0.15f } };
		double      worst = 0.0;
		bool        kept = true;
		std::string detail;
		for (const Case& k : cases) {
			RaymarchShaders::RayCB rc{};
			rc.lowRes[0] = rc.lowRes[1] = static_cast<float>(N);
			rc.tune[2] = static_cast<float>(k.r);
			rc.tune[3] = k.tol;
			a_ctx->UpdateSubresource(cb, 0, nullptr, &rc, 0, 0);
			a_ctx->OMSetRenderTargets(1, &rtv, nullptr);
			const D3D11_VIEWPORT vp{ 0.0f, 0.0f, static_cast<float>(N), static_cast<float>(N), 0.0f, 1.0f };
			a_ctx->RSSetViewports(1, &vp);
			a_ctx->IASetInputLayout(nullptr);
			a_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			a_ctx->VSSetShader(vs, nullptr, 0);
			a_ctx->PSSetShader(ps, nullptr, 0);
			a_ctx->PSSetConstantBuffers(0, 1, &cb);
			a_ctx->PSSetShaderResources(2, 1, &inSrv);
			a_ctx->Draw(3, 0);
			ID3D11ShaderResourceView* none = nullptr;
			a_ctx->PSSetShaderResources(2, 1, &none);
			const std::vector<float> gpu = ReadF32(a_dev, a_ctx, outTex, N, N, 4);
			double caseWorst = 0.0;
			for (int y = 0; y < N; ++y) {
				for (int x = 0; x < N; ++x) {
					const float* c = &in[(y * N + x) * 4];
					double       sumX = 0.0, sumY = 0.0, sumZ = 0.0, wsum = 0.0, wf = 0.0;
					for (int dy = -k.r; dy <= k.r; ++dy) {
						for (int dx = -k.r; dx <= k.r; ++dx) {
							const int    qx = std::clamp(x + dx, 0, N - 1), qy = std::clamp(y + dy, 0, N - 1);
							const float* n = &in[(qy * N + qx) * 4];
							double       w = (1.0 - std::abs(dx) / (k.r + 1.0)) * (1.0 - std::abs(dy) / (k.r + 1.0));
							if (k.tol > 0.0f) w *= std::clamp(1.0 - std::fabs(n[3] - c[3]) / std::max(c[3], 1.0f) / k.tol, 0.0, 1.0);
							sumX += n[0] * w;
							sumY += n[1] * w * n[0];
							sumZ += n[2] * w * n[0];
							wsum += w;
							wf += w * n[0];
						}
					}
					double ex[4]{ c[0], c[1], c[2], c[3] };
					if (wsum > 1e-6) {
						ex[0] = sumX / wsum;
						if (wf > 1e-6) {
							ex[1] = sumY / wf;
							ex[2] = sumZ / wf;
						}
					}
					const float* g = &gpu[(y * N + x) * 4];
					for (int ch = 0; ch < 4; ++ch) {
						const double scale = ch >= 2 ? std::max(std::fabs(ex[ch]), 1.0) : 1.0;
						caseWorst = std::max(caseWorst, std::fabs(g[ch] - ex[ch]) / scale);
					}
					if (g[3] != c[3]) kept = false;
				}
			}
			worst = std::max(worst, caseWorst);
			detail += "r " + std::to_string(k.r) + " tol " + std::to_string(k.tol) + ": worst " + std::to_string(caseWorst) + "; ";
		}
		Check(worst < 1e-5 && kept, "fog blur matches its formula", detail + (kept ? "scene distance kept" : "scene distance changed"));
		SafeRelease(cb);
		SafeRelease(rtv);
		SafeRelease(outTex);
		SafeRelease(inSrv);
		SafeRelease(inTex);
		SafeRelease(ps);
		SafeRelease(vs);
	}
}

namespace
{
	// kTemporalPS (ini [Render] TemporalBlend, 2026-11-16) against a CPU replica. Identity matrices: a texel's scene point is
	// (ndc x, ndc y, depth) from an eye at 0, and last frame's camera sees it shifted by the PosAdjust difference. Scale 2
	// (8 x 8 march, 16 x 16 depth): blend with and without the clamp, history texels at another distance (not used), no
	// usable history, a camera offset of two texels. Scale 1 (8 x 8 both): an offset of -0.1 pixel must keep each texel's own
	// history (the nearest march pixel), not its left neighbour's.
	int NearestMarchTexelCpu(double a_f, double a_z)
	{
		const int t0 = static_cast<int>(std::floor((a_f - 0.5) / a_z - 0.5));
		int       best = t0;
		double    bestD = 1e30;
		for (int k = -1; k <= 2; ++k) {
			const double d = std::fabs(std::floor((t0 + k + 0.5) * a_z) + 0.5 - a_f);
			if (d < bestD) {
				bestD = d;
				best = t0 + k;
			}
		}
		return best;
	}

	void CheckFogTemporal(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx)
	{
		constexpr int N = 8;
		ID3DBlob* vsb = Compile(RaymarchShaders::kVS, "vs_5_0");
		ID3DBlob* tb = Compile(RaymarchShaders::kTemporalPS, "ps_5_0");
		if (!vsb || !tb) {
			SafeRelease(vsb);
			SafeRelease(tb);
			Check(false, "fog temporal blend matches its formula", "shaders did not compile");
			return;
		}
		ID3D11VertexShader* vs = nullptr;
		ID3D11PixelShader*  ps = nullptr;
		a_dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
		a_dev->CreatePixelShader(tb->GetBufferPointer(), tb->GetBufferSize(), nullptr, &ps);
		SafeRelease(vsb);
		SafeRelease(tb);
		constexpr float kDepth = 0.5f;
		auto makeSrv = [&](const float* a_data, int a_w, DXGI_FORMAT a_fmt, UINT a_pitch, ID3D11Texture2D*& a_tex, ID3D11ShaderResourceView*& a_srv) {
			D3D11_TEXTURE2D_DESC td{};
			td.Width = td.Height = static_cast<UINT>(a_w);
			td.MipLevels = td.ArraySize = 1;
			td.SampleDesc.Count = 1;
			td.Format = a_fmt;
			td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			D3D11_SUBRESOURCE_DATA init{ a_data, a_pitch, 0 };
			a_dev->CreateTexture2D(&td, &init, &a_tex);
			a_dev->CreateShaderResourceView(a_tex, nullptr, &a_srv);
		};
		D3D11_TEXTURE2D_DESC od{};
		od.Width = od.Height = N;
		od.MipLevels = od.ArraySize = 1;
		od.SampleDesc.Count = 1;
		od.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		od.BindFlags = D3D11_BIND_RENDER_TARGET;
		ID3D11Texture2D*        outTex = nullptr;
		ID3D11RenderTargetView* rtv = nullptr;
		a_dev->CreateTexture2D(&od, nullptr, &outTex);
		a_dev->CreateRenderTargetView(outTex, nullptr, &rtv);
		D3D11_BUFFER_DESC cbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
		ID3D11Buffer*     cb = nullptr;
		a_dev->CreateBuffer(&cbd, nullptr, &cb);

		struct Case { const char* name; int scale; float a; float clamp; float valid; float shiftPx; };
		const Case cases[]{ { "blend", 2, 0.25f, 0.0f, 1.0f, 0.0f }, { "blend+clamp", 2, 0.25f, 1.0f, 1.0f, 0.0f }, { "no history", 2, 0.25f, 0.0f, 0.0f, 0.0f },
			{ "camera offset 2 texels", 2, 0.25f, 0.0f, 1.0f, 4.0f }, { "scale 1, offset -0.1 px", 1, 0.25f, 0.0f, 1.0f, -0.1f } };
		constexpr float kTol = 0.1f;
		double      worst = 0.0;
		std::string detail;
		for (const Case& k : cases) {
			const int F = N * k.scale;
			// ndc of texel t's march pixel along one axis (y points up)
			auto ndcOf = [&](int a_t, bool a_y) {
				const int    p = static_cast<int>(std::floor((a_t + 0.5) * k.scale));
				const double n = (p + 0.5) / F * 2.0 - 1.0;
				return a_y ? -n : n;
			};
			std::vector<float> cur(N * N * 4), hist(N * N * 4), depth(F * F, kDepth);
			for (int y = 0; y < N; ++y) {
				for (int x = 0; x < N; ++x) {
					const double nx = ndcOf(x, false), ny = ndcOf(y, true);
					const float  own = static_cast<float>(std::sqrt(nx * nx + ny * ny + kDepth * kDepth));
					float*       c = &cur[(y * N + x) * 4];
					c[0] = static_cast<float>((x * 5 + y * 3) % 10) / 10.0f;
					c[1] = 0.3f + 0.04f * x;
					c[2] = 800.0f + 5.0f * y;
					// a few texels stand alone at another distance (a thin leaf): their clamp has fewer than two neighbours
					c[3] = (x * 3 + y) % 11 == 0 ? own * 1.5f : own;
					float* h = &hist[(y * N + x) * 4];
					h[0] = (x + 2 * y) % 7 == 0 ? 0.0f : static_cast<float>((x * 3 + y * 7) % 10) / 10.0f;   // some texels held no fog
					h[1] = 0.6f - 0.03f * y;
					h[2] = 900.0f + 4.0f * x;
					h[3] = (x + y) % 5 == 0 ? own * 1.5f : own;   // every fifth texel: something else stood there
				}
			}
			ID3D11Texture2D *         curTex = nullptr, *histTex = nullptr, *depthTex = nullptr;
			ID3D11ShaderResourceView *curSrv = nullptr, *histSrv = nullptr, *depthSrv = nullptr;
			makeSrv(cur.data(), N, DXGI_FORMAT_R32G32B32A32_FLOAT, N * 16u, curTex, curSrv);
			makeSrv(hist.data(), N, DXGI_FORMAT_R32G32B32A32_FLOAT, N * 16u, histTex, histSrv);
			makeSrv(depth.data(), F, DXGI_FORMAT_R32_FLOAT, F * 4u, depthTex, depthSrv);
			const double shiftNdc = k.shiftPx * 2.0 / F;
			RaymarchShaders::RayCB rc{};
			for (int d = 0; d < 4; ++d) rc.invViewProj[d * 5] = rc.prevViewProj[d * 5] = 1.0f;
			rc.depth[2] = rc.depth[3] = static_cast<float>(F);
			rc.lowRes[0] = rc.lowRes[1] = static_cast<float>(N);
			rc.lowRes[2] = static_cast<float>(k.scale);
			rc.prevPos[0] = static_cast<float>(shiftNdc);
			rc.prevPos[3] = k.valid;
			rc.temporal[0] = k.a;
			rc.temporal[1] = kTol;
			rc.temporal[2] = k.clamp;
			a_ctx->UpdateSubresource(cb, 0, nullptr, &rc, 0, 0);
			a_ctx->OMSetRenderTargets(1, &rtv, nullptr);
			const D3D11_VIEWPORT vp{ 0.0f, 0.0f, static_cast<float>(N), static_cast<float>(N), 0.0f, 1.0f };
			a_ctx->RSSetViewports(1, &vp);
			a_ctx->IASetInputLayout(nullptr);
			a_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			a_ctx->VSSetShader(vs, nullptr, 0);
			a_ctx->PSSetShader(ps, nullptr, 0);
			a_ctx->PSSetConstantBuffers(0, 1, &cb);
			ID3D11ShaderResourceView* in[4]{ depthSrv, nullptr, curSrv, histSrv };
			a_ctx->PSSetShaderResources(0, 4, in);
			a_ctx->Draw(3, 0);
			ID3D11ShaderResourceView* none[4]{};
			a_ctx->PSSetShaderResources(0, 4, none);
			const std::vector<float> gpu = ReadF32(a_dev, a_ctx, outTex, N, N, 4);
			double caseWorst = 0.0;
			int    used = 0, ownTexel = 0, lone = 0;
			for (int y = 0; y < N; ++y) {
				for (int x = 0; x < N; ++x) {
					const float* c = &cur[(y * N + x) * 4];
					double       ex[4]{ c[0], c[1], c[2], c[3] };
					const double px = ndcOf(x, false) + shiftNdc, py = ndcOf(y, true);
					const double fx = (px * 0.5 + 0.5) * F, fy = (0.5 - py * 0.5) * F;
					const int    qx = NearestMarchTexelCpu(fx, k.scale), qy = NearestMarchTexelCpu(fy, k.scale);
					const double expect = std::sqrt(px * px + py * py + kDepth * kDepth);
					if (k.valid > 0.5f && qx >= 0 && qx < N && qy >= 0 && qy < N) {
						const float* h = &hist[(qy * N + qx) * 4];
						if (std::fabs(h[3] - expect) / std::max(expect, 1.0) <= kTol) {
							double hx = h[0];
							if (k.clamp > 0.5f) {
								double lo = c[0], hi = c[0];
								int    others = 0;
								for (int dy = -1; dy <= 1; ++dy) {
									for (int dx = -1; dx <= 1; ++dx) {
										if (dx == 0 && dy == 0) continue;
										const float* n = &cur[(std::clamp(y + dy, 0, N - 1) * N + std::clamp(x + dx, 0, N - 1)) * 4];
										if (std::fabs(n[3] - c[3]) / std::max(c[3], 1.0f) <= kTol) {
											lo = std::min(lo, static_cast<double>(n[0]));
											hi = std::max(hi, static_cast<double>(n[0]));
											++others;
										}
									}
								}
								if (others >= 2) {
									hx = std::clamp(hx, lo, hi);
								} else {
									++lone;
								}
							}
							ex[0] = hx + (c[0] - hx) * k.a;
							const double wh = (1.0 - k.a) * std::min(hx, static_cast<double>(h[0])), wc = k.a * c[0];
							if (wh + wc > 1e-6) {
								ex[1] = (h[1] * wh + c[1] * wc) / (wh + wc);
								ex[2] = (h[2] * wh + c[2] * wc) / (wh + wc);
							}
							++used;
							if (qx == x && qy == y) ++ownTexel;
						}
					}
					const float* g = &gpu[(y * N + x) * 4];
					for (int ch = 0; ch < 4; ++ch) {
						const double scale = ch >= 2 ? std::max(std::fabs(ex[ch]), 1.0) : 1.0;
						caseWorst = std::max(caseWorst, std::fabs(g[ch] - ex[ch]) / scale);
					}
				}
			}
			worst = std::max(worst, caseWorst);
			detail += std::string(k.name) + ": worst " + std::to_string(caseWorst) + ", history used " + std::to_string(used) + " (own texel " +
			          std::to_string(ownTexel) + (k.clamp > 0.5f ? ", unclamped lone " + std::to_string(lone) : std::string()) + "); ";
			SafeRelease(depthSrv);
			SafeRelease(depthTex);
			SafeRelease(histSrv);
			SafeRelease(histTex);
			SafeRelease(curSrv);
			SafeRelease(curTex);
		}
		Check(worst < 1e-4, "fog temporal blend matches its formula", detail);
		SafeRelease(cb);
		SafeRelease(rtv);
		SafeRelease(outTex);
		SafeRelease(ps);
		SafeRelease(vs);
	}

	// NoisePattern and DepthPick (2026-11-16 tests): MarchJitter and MarchSourcePixel read straight from a pixel shader
	// against CPU replicas. Depth: a different value at every full-res pixel (no ties). Block sizes 2, 4 and 17 / 8 = 2.125
	// (blocks of 2 and 3 pixels); DepthPick 0 must give the old lower-right pixel.
	// (the march core macro is not visible here; kCompositeCore carries it)
	constexpr const char* kPickProbeMain = R"(
float4 main(VSOut i) : SV_TARGET
{
	const int2 p = MarchSourcePixel(i.pos.xy);
	return float4(MarchJitter(i.pos.xy), (float)p.x, (float)p.y, 0.0);
}
)";

	float DepthAt(int a_x, int a_y)
	{
		const unsigned h = (static_cast<unsigned>(a_x) * 73856093u) ^ (static_cast<unsigned>(a_y) * 19349663u);
		return 0.3f + 0.5f * static_cast<float>(h % 9973u) / 9973.0f;
	}

	float IgnCpu(float a_x, float a_y)
	{
		const float d = a_x * 0.06711056f + a_y * 0.00583715f;
		const float v = 52.9829189f * (d - std::floor(d));
		return v - std::floor(v);
	}

	// MarchSourcePixel for texel (a_tx, a_ty), z = full / low, a_f = full size (square)
	std::pair<int, int> PickCpu(int a_tx, int a_ty, float a_z, int a_f, bool a_pick, const std::function<float(int, int)>& a_depth = DepthAt)
	{
		const int fx = static_cast<int>(std::min((a_tx + 0.5f) * a_z, a_f - 1.0f)), fy = static_cast<int>(std::min((a_ty + 0.5f) * a_z, a_f - 1.0f));
		if (!a_pick) return { fx, fy };
		auto lo = [&](int t) { return std::min(static_cast<int>(std::floor(static_cast<float>(t) * a_z)), a_f - 1); };
		auto hi = [&](int t) { return std::max(std::min(static_cast<int>(std::floor(static_cast<float>(t + 1) * a_z)), a_f) - 1, lo(t)); };
		const bool wantMin = ((a_tx + a_ty) & 1) == 0;
		int        bx = std::clamp(fx, lo(a_tx), hi(a_tx)), by = std::clamp(fy, lo(a_ty), hi(a_ty));   // ties keep the lower-right pixel
		float      bd = a_depth(bx, by);
		for (int y = lo(a_ty); y <= hi(a_ty); ++y) {
			for (int x = lo(a_tx); x <= hi(a_tx); ++x) {
				const float d = a_depth(x, y);
				if (wantMin ? d < bd : d > bd) {
					bd = d;
					bx = x;
					by = y;
				}
			}
		}
		return { bx, by };
	}

	void CheckPickAndJitter(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx)
	{
		constexpr int N = 8;
		ID3DBlob* vsb = Compile(RaymarchShaders::kVS, "vs_5_0");
		const std::string probeSrc = std::string(RaymarchShaders::kCompositeCore) + kPickProbeMain;
		ID3DBlob*         pb = Compile(probeSrc.c_str(), "ps_5_0");
		if (!vsb || !pb) {
			SafeRelease(vsb);
			SafeRelease(pb);
			Check(false, "DepthPick and NoisePattern match their formulas", "shaders did not compile");
			return;
		}
		ID3D11VertexShader* vs = nullptr;
		ID3D11PixelShader*  ps = nullptr;
		a_dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
		a_dev->CreatePixelShader(pb->GetBufferPointer(), pb->GetBufferSize(), nullptr, &ps);
		SafeRelease(vsb);
		SafeRelease(pb);
		D3D11_TEXTURE2D_DESC od{};
		od.Width = od.Height = N;
		od.MipLevels = od.ArraySize = 1;
		od.SampleDesc.Count = 1;
		od.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		od.BindFlags = D3D11_BIND_RENDER_TARGET;
		ID3D11Texture2D*        outTex = nullptr;
		ID3D11RenderTargetView* rtv = nullptr;
		a_dev->CreateTexture2D(&od, nullptr, &outTex);
		a_dev->CreateRenderTargetView(outTex, nullptr, &rtv);
		D3D11_BUFFER_DESC cbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
		ID3D11Buffer*     cb = nullptr;
		a_dev->CreateBuffer(&cbd, nullptr, &cb);

		struct Case { const char* name; int full; bool pick; int noise; float phase; };
		const Case cases[]{ { "pick 0, z 2, IGN", 16, false, 0, 0.0f }, { "pick 1, z 2, golden", 16, true, 1, 0.381966f },
			{ "pick 1, z 4, IGN", 32, true, 0, 0.0f }, { "pick 1, z 2.125, golden", 17, true, 1, 0.854102f } };
		constexpr float kFrame = 37.0f;
		int             pickWrong = 0, mins = 0, maxs = 0, wide = 0;
		double          jitterWorst = 0.0;
		std::string     detail;
		for (const Case& k : cases) {
			std::vector<float> depth(k.full * k.full);
			for (int y = 0; y < k.full; ++y) {
				for (int x = 0; x < k.full; ++x) depth[y * k.full + x] = DepthAt(x, y);
			}
			D3D11_TEXTURE2D_DESC td{};
			td.Width = td.Height = static_cast<UINT>(k.full);
			td.MipLevels = td.ArraySize = 1;
			td.SampleDesc.Count = 1;
			td.Format = DXGI_FORMAT_R32_FLOAT;
			td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			D3D11_SUBRESOURCE_DATA    init{ depth.data(), static_cast<UINT>(k.full) * 4u, 0 };
			ID3D11Texture2D*          depthTex = nullptr;
			ID3D11ShaderResourceView* depthSrv = nullptr;
			a_dev->CreateTexture2D(&td, &init, &depthTex);
			a_dev->CreateShaderResourceView(depthTex, nullptr, &depthSrv);
			const float            z = static_cast<float>(k.full) / static_cast<float>(N);
			RaymarchShaders::RayCB rc{};
			rc.fog[2] = kFrame;
			rc.depth[2] = rc.depth[3] = static_cast<float>(k.full);
			rc.lowRes[0] = rc.lowRes[1] = static_cast<float>(N);
			rc.lowRes[2] = z;
			rc.tune2[0] = static_cast<float>(k.noise);
			rc.tune2[1] = k.phase;
			rc.tune2[2] = k.pick ? 1.0f : 0.0f;
			a_ctx->UpdateSubresource(cb, 0, nullptr, &rc, 0, 0);
			a_ctx->OMSetRenderTargets(1, &rtv, nullptr);
			const D3D11_VIEWPORT vp{ 0.0f, 0.0f, static_cast<float>(N), static_cast<float>(N), 0.0f, 1.0f };
			a_ctx->RSSetViewports(1, &vp);
			a_ctx->IASetInputLayout(nullptr);
			a_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			a_ctx->VSSetShader(vs, nullptr, 0);
			a_ctx->PSSetShader(ps, nullptr, 0);
			a_ctx->PSSetConstantBuffers(0, 1, &cb);
			a_ctx->PSSetShaderResources(0, 1, &depthSrv);
			a_ctx->Draw(3, 0);
			ID3D11ShaderResourceView* none = nullptr;
			a_ctx->PSSetShaderResources(0, 1, &none);
			const std::vector<float> gpu = ReadF32(a_dev, a_ctx, outTex, N, N, 4);
			int caseWrong = 0;
			for (int y = 0; y < N; ++y) {
				for (int x = 0; x < N; ++x) {
					const float* g = &gpu[(y * N + x) * 4];
					const auto [px, py] = PickCpu(x, y, z, k.full, k.pick);
					if (static_cast<int>(g[1]) != px || static_cast<int>(g[2]) != py) ++caseWrong;
					if (k.pick) {
						((x + y) & 1) == 0 ? ++mins : ++maxs;
						const int w = std::min(static_cast<int>(std::floor((x + 1) * z)), k.full) - static_cast<int>(std::floor(x * z));
						if (w > 2) ++wide;
					}
					const float  cx = x + 0.5f, cy = y + 0.5f;
					const double ex = k.noise == 1 ? [&] { const float v = IgnCpu(cx, cy) + k.phase; return static_cast<double>(v - std::floor(v)); }()
					                               : static_cast<double>(IgnCpu(cx + 5.588238f * kFrame, cy + 5.588238f * kFrame));
					const double d = std::fabs(g[0] - ex);
					jitterWorst = std::max(jitterWorst, std::min(d, 1.0 - d));   // offsets live on a circle (0 and 1 meet)
				}
			}
			pickWrong += caseWrong;
			detail += std::string(k.name) + ": pick mismatches " + std::to_string(caseWrong) + "; ";
			SafeRelease(depthSrv);
			SafeRelease(depthTex);
		}
		detail += "min texels " + std::to_string(mins) + ", max texels " + std::to_string(maxs) + ", 3-pixel-wide blocks " + std::to_string(wide) +
		          ", worst jitter error " + std::to_string(jitterWorst);
		Check(pickWrong == 0 && jitterWorst < 2e-3 && wide > 0, "DepthPick and NoisePattern match their formulas", detail);
		SafeRelease(cb);
		SafeRelease(rtv);
		SafeRelease(outTex);
		SafeRelease(ps);
		SafeRelease(vs);
	}

	// kTemporalPS with DepthPick 1: a texel's history is found among the 3 x 3 texels around its point's block by the best
	// distance match. Identity matrices as in CheckFogTemporal, every full-res pixel at its own depth. Still camera (block
	// sizes 2 and 2.125): every texel must take its own history. Camera moved two texels: the GPU must match the replica.
	void CheckTemporalDepthPick(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx)
	{
		constexpr int N = 8;
		ID3DBlob* vsb = Compile(RaymarchShaders::kVS, "vs_5_0");
		ID3DBlob* tb = Compile(RaymarchShaders::kTemporalPS, "ps_5_0");
		if (!vsb || !tb) {
			SafeRelease(vsb);
			SafeRelease(tb);
			Check(false, "fog temporal blend with DepthPick finds the matching history", "shaders did not compile");
			return;
		}
		ID3D11VertexShader* vs = nullptr;
		ID3D11PixelShader*  ps = nullptr;
		a_dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
		a_dev->CreatePixelShader(tb->GetBufferPointer(), tb->GetBufferSize(), nullptr, &ps);
		SafeRelease(vsb);
		SafeRelease(tb);
		auto makeSrv = [&](const float* a_data, int a_w, DXGI_FORMAT a_fmt, UINT a_pitch, ID3D11Texture2D*& a_tex, ID3D11ShaderResourceView*& a_srv) {
			D3D11_TEXTURE2D_DESC td{};
			td.Width = td.Height = static_cast<UINT>(a_w);
			td.MipLevels = td.ArraySize = 1;
			td.SampleDesc.Count = 1;
			td.Format = a_fmt;
			td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			D3D11_SUBRESOURCE_DATA init{ a_data, a_pitch, 0 };
			a_dev->CreateTexture2D(&td, &init, &a_tex);
			a_dev->CreateShaderResourceView(a_tex, nullptr, &a_srv);
		};
		D3D11_TEXTURE2D_DESC od{};
		od.Width = od.Height = N;
		od.MipLevels = od.ArraySize = 1;
		od.SampleDesc.Count = 1;
		od.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		od.BindFlags = D3D11_BIND_RENDER_TARGET;
		ID3D11Texture2D*        outTex = nullptr;
		ID3D11RenderTargetView* rtv = nullptr;
		a_dev->CreateTexture2D(&od, nullptr, &outTex);
		a_dev->CreateRenderTargetView(outTex, nullptr, &rtv);
		D3D11_BUFFER_DESC cbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
		ID3D11Buffer*     cb = nullptr;
		a_dev->CreateBuffer(&cbd, nullptr, &cb);

		struct Case { const char* name; int full; float shiftPx; bool still; bool uniform; };
		// "one depth": every pixel at the same depth, as the sky is (Codex review: ties must keep each texel's own history)
		const Case cases[]{ { "still, z 2", 16, 0.0f, true, false }, { "still, z 2.125", 17, 0.0f, true, false }, { "still, one depth, z 2", 16, 0.0f, true, true },
			{ "camera offset 2 texels, z 2", 16, 4.0f, false, false } };
		constexpr float kA = 0.25f, kTol = 0.1f;
		double          worst = 0.0;
		bool            stillOwn = true;
		std::string     detail;
		for (const Case& k : cases) {
			const float                         z = static_cast<float>(k.full) / static_cast<float>(N);
			const std::function<float(int, int)> depthOf = [&](int a_x, int a_y) { return k.uniform ? 0.5f : DepthAt(a_x, a_y); };
			auto                                pointOf = [&](int a_tx, int a_ty) {   // the scene point the texel marched (eye at 0)
                const auto [px, py] = PickCpu(a_tx, a_ty, z, k.full, true, depthOf);
                const double nx = (px + 0.5) / k.full * 2.0 - 1.0, ny = -((py + 0.5) / k.full * 2.0 - 1.0);
                return std::array<double, 3>{ nx, ny, static_cast<double>(depthOf(px, py)) };
			};
			std::vector<float> cur(N * N * 4), hist(N * N * 4), depth(k.full * k.full);
			for (int y = 0; y < k.full; ++y) {
				for (int x = 0; x < k.full; ++x) depth[y * k.full + x] = depthOf(x, y);
			}
			for (int y = 0; y < N; ++y) {
				for (int x = 0; x < N; ++x) {
					const auto  p = pointOf(x, y);
					const float own = static_cast<float>(std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]));
					float*      c = &cur[(y * N + x) * 4];
					c[0] = static_cast<float>((x * 5 + y * 3) % 10) / 10.0f;
					c[1] = 0.3f + 0.04f * x;
					c[2] = 800.0f + 5.0f * y;
					c[3] = own;
					float* h = &hist[(y * N + x) * 4];
					h[0] = 0.05f + 0.09f * static_cast<float>((x * 3 + y * 5) % 10);   // no zeros, no equal diagonal neighbours (reviewer)
					h[1] = 0.6f - 0.03f * y;
					h[2] = 900.0f + 4.0f * x;
					h[3] = own;
				}
			}
			ID3D11Texture2D *         curTex = nullptr, *histTex = nullptr, *depthTex = nullptr;
			ID3D11ShaderResourceView *curSrv = nullptr, *histSrv = nullptr, *depthSrv = nullptr;
			makeSrv(cur.data(), N, DXGI_FORMAT_R32G32B32A32_FLOAT, N * 16u, curTex, curSrv);
			makeSrv(hist.data(), N, DXGI_FORMAT_R32G32B32A32_FLOAT, N * 16u, histTex, histSrv);
			makeSrv(depth.data(), k.full, DXGI_FORMAT_R32_FLOAT, static_cast<UINT>(k.full) * 4u, depthTex, depthSrv);
			const double           shiftNdc = k.shiftPx * 2.0 / k.full;
			RaymarchShaders::RayCB rc{};
			for (int d = 0; d < 4; ++d) rc.invViewProj[d * 5] = rc.prevViewProj[d * 5] = 1.0f;
			rc.depth[2] = rc.depth[3] = static_cast<float>(k.full);
			rc.lowRes[0] = rc.lowRes[1] = static_cast<float>(N);
			rc.lowRes[2] = z;
			rc.prevPos[0] = static_cast<float>(shiftNdc);
			rc.prevPos[3] = 1.0f;
			rc.temporal[0] = kA;
			rc.temporal[1] = kTol;
			rc.tune2[2] = 1.0f;
			a_ctx->UpdateSubresource(cb, 0, nullptr, &rc, 0, 0);
			a_ctx->OMSetRenderTargets(1, &rtv, nullptr);
			const D3D11_VIEWPORT vp{ 0.0f, 0.0f, static_cast<float>(N), static_cast<float>(N), 0.0f, 1.0f };
			a_ctx->RSSetViewports(1, &vp);
			a_ctx->IASetInputLayout(nullptr);
			a_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			a_ctx->VSSetShader(vs, nullptr, 0);
			a_ctx->PSSetShader(ps, nullptr, 0);
			a_ctx->PSSetConstantBuffers(0, 1, &cb);
			ID3D11ShaderResourceView* in[4]{ depthSrv, nullptr, curSrv, histSrv };
			a_ctx->PSSetShaderResources(0, 4, in);
			a_ctx->Draw(3, 0);
			ID3D11ShaderResourceView* none[4]{};
			a_ctx->PSSetShaderResources(0, 4, none);
			const std::vector<float> gpu = ReadF32(a_dev, a_ctx, outTex, N, N, 4);
			double caseWorst = 0.0;
			int    used = 0, own = 0, viaNeighbour = 0;
			for (int y = 0; y < N; ++y) {
				for (int x = 0; x < N; ++x) {
					const float* c = &cur[(y * N + x) * 4];
					double       ex[4]{ c[0], c[1], c[2], c[3] };
					const auto   p = pointOf(x, y);
					const double qx = p[0] + shiftNdc, qy = p[1], qz = p[2];
					const double fx = (qx * 0.5 + 0.5) * k.full, fy = (0.5 - qy * 0.5) * k.full;
					// the texel whose block holds the point (float, as the shader computes it)
					auto blockTexel = [&](double a_f) {
						const float pf = std::floor(static_cast<float>(a_f));
						int         tb = static_cast<int>(std::floor(pf / z));
						if (std::floor(static_cast<float>(tb + 1) * z) <= pf) ++tb;
						if (std::floor(static_cast<float>(tb) * z) > pf) --tb;
						return std::clamp(tb, 0, N - 1);
					};
					const double expect = std::sqrt(qx * qx + qy * qy + qz * qz);
					if (fx >= 0.0 && fx < k.full && fy >= 0.0 && fy < k.full) {
						const int q0x = blockTexel(fx), q0y = blockTexel(fy);
						int       bx = q0x, by = q0y;
						double    be = std::fabs(hist[(q0y * N + q0x) * 4 + 3] - expect) / std::max(expect, 1.0);
						if (be > kTol) for (int dy = -1; dy <= 1; ++dy) {
							for (int dx = -1; dx <= 1; ++dx) {
								if (dx == 0 && dy == 0) continue;
								const int nx = q0x + dx, ny = q0y + dy;
								if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;
								const double e = std::fabs(hist[(ny * N + nx) * 4 + 3] - expect) / std::max(expect, 1.0);
								if (e < be) {
									be = e;
									bx = nx;
									by = ny;
								}
							}
						}
						if (be <= kTol) {
							const float* h = &hist[(by * N + bx) * 4];
							const double hx = h[0];
							ex[0] = hx + (c[0] - hx) * kA;
							const double wh = (1.0 - kA) * hx, wc = kA * c[0];
							if (wh + wc > 1e-6) {
								ex[1] = (h[1] * wh + c[1] * wc) / (wh + wc);
								ex[2] = (h[2] * wh + c[2] * wc) / (wh + wc);
							}
							++used;
							if (bx == x && by == y) ++own;
							if (bx != q0x || by != q0y) ++viaNeighbour;
						}
					}
					const float* g = &gpu[(y * N + x) * 4];
					for (int ch = 0; ch < 4; ++ch) {
						const double scale = ch >= 2 ? std::max(std::fabs(ex[ch]), 1.0) : 1.0;
						caseWorst = std::max(caseWorst, std::fabs(g[ch] - ex[ch]) / scale);
					}
				}
			}
			if (k.still && own != N * N) stillOwn = false;
			worst = std::max(worst, caseWorst);
			detail += std::string(k.name) + ": worst " + std::to_string(caseWorst) + ", history used " + std::to_string(used) + " (own texel " +
			          std::to_string(own) + ", found beside the block's texel " + std::to_string(viaNeighbour) + "); ";
			SafeRelease(depthSrv);
			SafeRelease(depthTex);
			SafeRelease(histSrv);
			SafeRelease(histTex);
			SafeRelease(curSrv);
			SafeRelease(curTex);
		}
		Check(worst < 1e-4 && stillOwn, "fog temporal blend with DepthPick finds the matching history", detail);
		SafeRelease(cb);
		SafeRelease(rtv);
		SafeRelease(outTex);
		SafeRelease(ps);
		SafeRelease(vs);
	}
}

namespace
{
	// SunShadow (2026-11-17): the sun depth kernel (kSunColsCS) against a CPU copy of its algorithm and against the
	// closed forms it should approach, and the march's SunCols lookup / blend helpers against their formulas.
	struct SunGrid
	{
		int                n;
		float              cell;
		std::vector<float> col;   // H, W, ground x known, known
		std::vector<float> up;    // H x known, W x known
	};

	double SunProfileDensity(double h, double H, double a)
	{
		a = std::max(a, 1e-3);
		const double x = std::max(h / H, 0.0);
		return x < 1.0 ? a * std::exp(-a * x) / (1.0 - std::exp(-a)) : 0.0;
	}

	double SunDensityCpu(const SunGrid& g, int qx, int qy, double z, double opacity, double a, bool unfade)
	{
		if (qx < 0 || qy < 0 || qx >= g.n || qy >= g.n) return 0.0;
		const float* c = &g.col[(static_cast<std::size_t>(qy) * g.n + qx) * 4];
		if (c[3] < 0.05f) return 0.0;
		const double h = z - c[2] / c[3];
		if (h < 0.0) return 0.0;
		double fade = 1.0;
		if (unfade) {
			const double ux = (qx + 0.5) / g.n - 0.5, uy = (qy + 0.5) / g.n - 0.5;
			const double r = std::sqrt(ux * ux + uy * uy) * 2.0;
			const double s = std::clamp((r - 0.55) / 0.4, 0.0, 1.0);
			fade = 1.0 - s * s * (3.0 - 2.0 * s);
			if (!(fade > 0.0)) return 0.0;
		}
		double       d = 0.0;
		if (c[0] >= 1.0f && c[1] > 0.0f) d += c[1] * c[3] * SunProfileDensity(h, c[0], a) / c[0];
		const float* u = &g.up[(static_cast<std::size_t>(qy) * g.n + qx) * 2];
		const double upH = u[0] / c[3];
		if (upH >= 1.0 && u[1] > 0.0f) d += u[1] * SunProfileDensity(h, upH, RaymarchShaders::kUpperSteepnessCpp) / upH;
		return opacity * d / std::max(fade, 0.1);
	}

	// The kernel's algorithm in double: four heights, kTaps pieces toward the sun.
	std::array<double, 4> SunColsCpu(const SunGrid& g, int px, int py, const double sun[3], double lowest, double topZ, double longest,
		double opacity, double a, bool unfade, int scale = 1)
	{
		std::array<double, 4> o{};
		const int             nOut = (g.n + scale - 1) / scale;
		const double          ratio = static_cast<double>(g.n) / nOut;
		const double          pcx = (px + 0.5) * ratio, pcy = (py + 0.5) * ratio;   // output texel centre in column texels
		// ground and top bilinear over the known texels around it (kSunColsCS SunColumnAt)
		const double qx = pcx - 0.5, qy = pcy - 0.5;
		const int    ix = static_cast<int>(std::floor(qx)), iy = static_cast<int>(std::floor(qy));
		const double fx = qx - ix, fy = qy - iy;
		double       wSum = 0.0, gSum = 0.0, tSum = 0.0;
		for (int k = 0; k < 4; ++k) {
			const int    ox = k & 1, oy = k >> 1;
			const int    cx = std::clamp(ix + ox, 0, g.n - 1), cy = std::clamp(iy + oy, 0, g.n - 1);
			const float* cc = &g.col[(static_cast<std::size_t>(cy) * g.n + cx) * 4];
			const double w = (ox ? fx : 1.0 - fx) * (oy ? fy : 1.0 - fy) * cc[3];
			if (cc[3] >= 0.05f && w > 0.0) {
				wSum += w;
				gSum += w * cc[2] / cc[3];
				tSum += w * std::max(static_cast<double>(cc[0]), g.up[(static_cast<std::size_t>(cy) * g.n + cx) * 2] / static_cast<double>(cc[3]));
			}
		}
		if (!(wSum > 1e-4)) return o;
		const double ground = gSum / wSum;
		const double top = std::max(tSum / wSum, 1.0);
		double       v[3]{ sun[0], sun[1], std::max(sun[2], lowest) };
		const double len0 = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		for (double& e : v) e /= len0;
		constexpr int kTaps = 12;
		for (int k = 0; k < 4; ++k) {
			const double z0 = ground + top * (k / 3.0);
			const double len = std::min(std::max(topZ - z0, 0.0) / v[2], longest);
			const double nearLen = std::min(len, (3.0 * top + 32.0) / v[2]);
			const int    nearTaps = nearLen < len ? 8 : kTaps;
			double       tau = 0.0;
			for (int s = 0; s < kTaps; ++s) {
				double d, seg;
				if (s < nearTaps) {
					const double u = (s + 0.5) / nearTaps;
					d = nearLen * u * u;
					seg = 2.0 * nearLen * u / nearTaps;
				} else {
					seg = (len - nearLen) / (kTaps - 8);
					d = nearLen + (s - nearTaps + 0.5) * seg;
				}
				const double qx = pcx + v[0] * d / g.cell;
				const double qy = pcy + v[1] * d / g.cell;
				tau += SunDensityCpu(g, static_cast<int>(std::floor(qx)), static_cast<int>(std::floor(qy)), z0 + v[2] * d, opacity, a, unfade) * seg;
			}
			o[k] = tau;
		}
		return o;
	}

	// Runs kSunColsCS on the grid; R32 output (the plugin stores R16, which only rounds).
	std::vector<float> SunColsGpu(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11ComputeShader* a_cs, const SunGrid& g,
		const RaymarchShaders::SunColsCB& a_cb)
	{
		std::vector<float>   out;
		D3D11_TEXTURE2D_DESC td{};
		td.Width = td.Height = static_cast<UINT>(g.n);
		td.MipLevels = td.ArraySize = 1;
		td.SampleDesc.Count = 1;
		td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA cd{ g.col.data(), static_cast<UINT>(g.n * 16), 0 };
		ID3D11Texture2D *colTex = nullptr, *upTex = nullptr, *outTex = nullptr;
		ID3D11ShaderResourceView *colSrv = nullptr, *upSrv = nullptr;
		ID3D11UnorderedAccessView* uav = nullptr;
		a_dev->CreateTexture2D(&td, &cd, &colTex);
		td.Format = DXGI_FORMAT_R32G32_FLOAT;
		D3D11_SUBRESOURCE_DATA ud{ g.up.data(), static_cast<UINT>(g.n * 8), 0 };
		a_dev->CreateTexture2D(&td, &ud, &upTex);
		td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		td.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
		a_dev->CreateTexture2D(&td, nullptr, &outTex);
		D3D11_BUFFER_DESC bd{ sizeof(a_cb), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
		D3D11_SUBRESOURCE_DATA bdata{ &a_cb, 0, 0 };
		ID3D11Buffer* cb = nullptr;
		a_dev->CreateBuffer(&bd, &bdata, &cb);
		if (colTex && upTex && outTex && cb && SUCCEEDED(a_dev->CreateShaderResourceView(colTex, nullptr, &colSrv)) &&
			SUCCEEDED(a_dev->CreateShaderResourceView(upTex, nullptr, &upSrv)) && SUCCEEDED(a_dev->CreateUnorderedAccessView(outTex, nullptr, &uav))) {
			ID3D11ShaderResourceView* in[2]{ colSrv, upSrv };
			a_ctx->CSSetShader(a_cs, nullptr, 0);
			a_ctx->CSSetConstantBuffers(1, 1, &cb);
			a_ctx->CSSetShaderResources(0, 2, in);
			a_ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
			a_ctx->Dispatch((g.n + 7) / 8, (g.n + 7) / 8, 1);
			ID3D11ShaderResourceView*  noSrv[2]{};
			ID3D11UnorderedAccessView* noUav = nullptr;
			a_ctx->CSSetShaderResources(0, 2, noSrv);
			a_ctx->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
			out = ReadF32(a_dev, a_ctx, outTex, g.n, g.n, 4);
		}
		SafeRelease(uav);
		SafeRelease(upSrv);
		SafeRelease(colSrv);
		SafeRelease(cb);
		SafeRelease(outTex);
		SafeRelease(upTex);
		SafeRelease(colTex);
		return out;
	}

	SunGrid SunFlatGrid(int a_n, float a_cell, float a_H, float a_W)
	{
		SunGrid g{ a_n, a_cell, std::vector<float>(static_cast<std::size_t>(a_n) * a_n * 4), std::vector<float>(static_cast<std::size_t>(a_n) * a_n * 2, 0.0f) };
		for (int i = 0; i < a_n * a_n; ++i) {
			g.col[i * 4 + 0] = a_H;
			g.col[i * 4 + 1] = a_W;
			g.col[i * 4 + 2] = 0.0f;
			g.col[i * 4 + 3] = 1.0f;
		}
		return g;
	}

	void CheckSunCols(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx)
	{
		const char* kName = "SunShadow sun depth kernel matches its algorithm and the closed forms";
		ID3DBlob*   b = Compile(RaymarchShaders::kSunColsCS, "cs_5_0");
		ID3D11ComputeShader* cs = nullptr;
		if (!b || FAILED(a_dev->CreateComputeShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &cs))) {
			SafeRelease(b);
			Check(false, kName, "kernel did not compile");
			return;
		}
		SafeRelease(b);
		constexpr int    N = 24;
		constexpr float  kCell = 32.0f;
		constexpr double kOpacityS = 1.0, kA = 3.0, kLowest = 0.05;
		std::string      detail;
		double           worstAlgo = 0.0;   // GPU vs CPU copy, relative to max(1, value)
		auto             runCase = [&](const SunGrid& g, const double sun[3], double topZ, bool unfade = false, int scale = 1) {
            RaymarchShaders::SunColsCB cb{};
            cb.grid[0] = static_cast<float>(g.n);
            cb.grid[1] = g.cell;
            cb.grid[2] = static_cast<float>(kA);
            cb.grid[3] = static_cast<float>(kOpacityS);
            const double len = std::sqrt(sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]);
            cb.sun[0] = static_cast<float>(sun[0] / len);
            cb.sun[1] = static_cast<float>(sun[1] / len);
            cb.sun[2] = static_cast<float>(sun[2] / len);
            cb.sun[3] = static_cast<float>(kLowest);
            cb.top[0] = static_cast<float>(topZ);
            cb.top[1] = 4096.0f;
            cb.top[2] = unfade ? 1.0f : 0.0f;
            cb.origin[2] = static_cast<float>(scale);
            const int nOut = (g.n + scale - 1) / scale;
            const std::vector<float> gpu = SunColsGpu(a_dev, a_ctx, cs, g, cb);
            const double             sn[3]{ cb.sun[0], cb.sun[1], cb.sun[2] };
            for (int y = 0; y < nOut && gpu.size() == static_cast<std::size_t>(g.n) * g.n * 4; ++y) {
                for (int x = 0; x < nOut; ++x) {
                    const auto c = SunColsCpu(g, x, y, sn, kLowest, topZ, 4096.0, kOpacityS, kA, unfade, scale);
                    for (int k = 0; k < 4; ++k) {
                        const double v = gpu[(static_cast<std::size_t>(y) * g.n + x) * 4 + k];
                        worstAlgo = std::max(worstAlgo, std::isfinite(v) ? std::fabs(v - c[k]) / std::max(1.0, std::fabs(c[k])) : 1e9);
                    }
                }
            }
            return gpu;
		};
		auto at = [&](const std::vector<float>& a_v, int x, int y, int k) {
			return a_v.size() == static_cast<std::size_t>(N) * N * 4 ? static_cast<double>(a_v[(static_cast<std::size_t>(y) * N + x) * 4 + k]) : NAN;
		};
		// 1. even layer, sun overhead: the ground's depth is the column's W (12 taps over an a = 3 profile)
		const SunGrid flat = SunFlatGrid(N, kCell, 100.0f, 0.3f);
		const double  up[3]{ 0.0, 0.0, 1.0 };
		const auto    g1 = runCase(flat, up, 100.0);
		const double  vert = at(g1, 12, 12, 0);
		const bool    ok1 = std::fabs(vert - 0.3) < 0.006;
		// 2. even layer, sun at 20 degrees: plane-parallel exact, W / sin(20) (interior texel, the ray stays on the grid)
		const double e20 = 20.0 * 3.14159265358979 / 180.0;
		const double s20[3]{ std::cos(e20), 0.0, std::sin(e20) };
		const auto   g2 = runCase(flat, s20, 100.0);
		const double slant = at(g2, 6, 12, 0);
		const bool   ok2 = std::fabs(slant - 0.3 / std::sin(e20)) < 0.02 * (0.3 / std::sin(e20));
		// 3. monotone: lower sun deeper, higher point shallower
		const double e8 = 8.0 * 3.14159265358979 / 180.0;
		const double s8[3]{ std::cos(e8), 0.0, std::sin(e8) };
		const auto   g3 = runCase(flat, s8, 100.0);
		const bool   ok3 = at(g3, 0, 12, 0) > slant && at(g2, 6, 12, 1) < slant && at(g2, 6, 12, 2) < at(g2, 6, 12, 1) && at(g2, 6, 12, 3) < at(g2, 6, 12, 2);
		// 4. a tall bank toward the sun shades a low column next to it more than a low column far from it
		SunGrid bank = SunFlatGrid(N, kCell, 40.0f, 0.15f);
		for (int y = 0; y < N; ++y) {
			for (int x = 10; x < 13; ++x) {
				bank.col[(static_cast<std::size_t>(y) * N + x) * 4 + 0] = 200.0f;
				bank.col[(static_cast<std::size_t>(y) * N + x) * 4 + 1] = 0.9f;
			}
		}
		const double e10 = 10.0 * 3.14159265358979 / 180.0;
		const double s10[3]{ std::cos(e10), 0.0, std::sin(e10) };
		const auto   g4 = runCase(bank, s10, 200.0);
		const double nearBank = at(g4, 7, 12, 1), farBank = at(g4, 14, 12, 1);
		const bool   ok4 = nearBank > 1.5 * farBank;
		// 5. unknown column: 0; the tall layer adds its own depth
		SunGrid holes = SunFlatGrid(N, kCell, 100.0f, 0.3f);
		holes.col[(static_cast<std::size_t>(12) * N + 12) * 4 + 3] = 0.0f;
		for (int i = 0; i < N * N; ++i) {
			holes.up[i * 2 + 0] = 150.0f;
			holes.up[i * 2 + 1] = 0.2f;
		}
		const auto g5 = runCase(holes, up, 150.0);
		const bool ok5 = at(g5, 12, 12, 0) == 0.0 && at(g5, 12, 12, 3) == 0.0 && std::fabs(at(g5, 5, 5, 0) - 0.5) < 0.012;
		// 6. ground 500 up, half known: the kernel reads ground = z / known and W x known (the sky overhead: W x known = 0.15)
		SunGrid half = SunFlatGrid(N, kCell, 100.0f, 0.3f);
		for (int i = 0; i < N * N; ++i) {
			half.col[i * 4 + 2] = 500.0f * 0.5f;
			half.col[i * 4 + 3] = 0.5f;
		}
		const auto   g6 = runCase(half, up, 600.0);
		const double halfV = at(g6, 12, 12, 0);
		const bool   ok6 = std::fabs(halfV - 0.15) < 0.003;
		// 7. a hill toward the sun: the ray through its ground counts nothing, so its own fog does not matter there
		SunGrid hillA = SunFlatGrid(N, kCell, 40.0f, 0.15f), hillB = hillA;
		for (int y = 0; y < N; ++y) {
			for (int x = 9; x < N; ++x) {
				hillA.col[(static_cast<std::size_t>(y) * N + x) * 4 + 2] = 2000.0f;
				hillB.col[(static_cast<std::size_t>(y) * N + x) * 4 + 2] = 2000.0f;
				hillB.col[(static_cast<std::size_t>(y) * N + x) * 4 + 1] = 0.0f;
			}
		}
		const auto   g7a = runCase(hillA, s10, 2040.0), g7b = runCase(hillB, s10, 2040.0);
		const double hillWith = at(g7a, 6, 12, 0), hillWithout = at(g7b, 6, 12, 0);
		const bool   ok7 = std::fabs(hillWith - hillWithout) < 1e-6;
		// 8. a thin layer under a box top far above (a valley): overhead, the ground still sees about W
		const SunGrid thin = SunFlatGrid(N, kCell, 40.0f, 0.3f);
		const auto    g8 = runCase(thin, up, 3000.0);
		const double  thinV = at(g8, 12, 12, 0);
		const bool    ok8 = std::fabs(thinV - 0.3) < 0.009;
		// 9. columns faded as CSColumn fades them (W x edge), undone (Top.z = 1): a texel in the fade ring sees the whole layer
		SunGrid faded = SunFlatGrid(N, kCell, 100.0f, 0.3f);
		for (int y = 0; y < N; ++y) {
			for (int x = 0; x < N; ++x) {
				const double ux = (x + 0.5) / N - 0.5, uy = (y + 0.5) / N - 0.5;
				const double s = std::clamp((std::sqrt(ux * ux + uy * uy) * 2.0 - 0.55) / 0.4, 0.0, 1.0);
				faded.col[(static_cast<std::size_t>(y) * N + x) * 4 + 1] = static_cast<float>(0.3 * (1.0 - s * s * (3.0 - 2.0 * s)));
			}
		}
		const auto   g9 = runCase(faded, up, 100.0, true);
		const double ringV = at(g9, 3, 12, 0);   // r about 0.71: faded to about 0.57 of W
		const bool   ok9 = std::fabs(ringV - 0.3) < 0.006;
		// 11. ground rising toward the sun at the sun's angle (20 degrees): the ray keeps its height in the layer, so the
		// depth is far more than over flat ground; against a fine integration along the same path (Codex review 2026-11-17)
		SunGrid slope = SunFlatGrid(N, kCell, 40.0f, 0.15f);
		for (int y = 0; y < N; ++y) {
			for (int x = 0; x < N; ++x) {
				slope.col[(static_cast<std::size_t>(y) * N + x) * 4 + 2] = static_cast<float>((x + 0.5) * kCell * std::tan(e20));
			}
		}
		const double slopeTop = N * kCell * std::tan(e20) + 40.0;
		const auto   g11 = runCase(slope, s20, slopeTop);
		double       slopeRef = 0.0;
		{
			const double z0 = 4.5 * kCell * std::tan(e20) + 40.0 / 3.0;
			const double len = std::min((slopeTop - z0) / std::sin(e20), 4096.0);
			constexpr int kFine = 20000;
			for (int s = 0; s < kFine; ++s) {
				const double d = (s + 0.5) * len / kFine;
				const double qx = 4.5 + std::cos(e20) * d / kCell;
				slopeRef += SunDensityCpu(slope, static_cast<int>(std::floor(qx)), 12, z0 + std::sin(e20) * d, kOpacityS, kA, false) * (len / kFine);
			}
		}
		const double slopeV = at(g11, 4, 12, 1);
		const bool   ok11 = std::fabs(slopeV - slopeRef) < 0.25 * slopeRef && slopeV > 2.0 * at(g2, 6, 12, 1);
		// 10. a diagonal sun over the bank: the GPU and the CPU copy agree (x and y not swapped)
		const double sd[3]{ std::cos(e10) * 0.6, std::cos(e10) * 0.8, std::sin(e10) };
		runCase(bank, sd, 200.0);
		runCase(slope, s20, slopeTop, false, 2);   // half resolution, as the plugin runs it: the GPU and the CPU copy agree
		runCase(faded, sd, 100.0, true, 2);
		const bool okAlgo = worstAlgo < 1e-4;
		detail = "overhead ground " + std::to_string(vert) + " (W 0.3); 20 deg " + std::to_string(slant) + " (W / sin " +
		         std::to_string(0.3 / std::sin(e20)) + "); 8 deg " + std::to_string(at(g3, 4, 12, 0)) + ", 20 deg at 1/3, 2/3, 1 of the top " +
		         std::to_string(at(g2, 6, 12, 1)) + ", " + std::to_string(at(g2, 6, 12, 2)) + ", " + std::to_string(at(g2, 6, 12, 3)) +
		         "; low column beside a tall bank " + std::to_string(nearBank) + ", far from it " + std::to_string(farBank) + "; unknown column " +
		         std::to_string(at(g5, 12, 12, 0)) + ", both layers overhead " + std::to_string(at(g5, 5, 5, 0)) + " (0.5); worst GPU vs CPU copy " +
		         std::to_string(worstAlgo);
		detail += "; ground 500 half known " + std::to_string(halfV) + " (0.15); hill toward the sun with / without its fog " + std::to_string(hillWith) + " / " +
		          std::to_string(hillWithout) + "; thin layer under a far top " + std::to_string(thinV) + " (0.3); fade ring undone " + std::to_string(ringV) +
		          " (0.3)";
		detail += "; ground rising toward the sun " + std::to_string(slopeV) + " (fine integration " + std::to_string(slopeRef) + ")";
		Check(ok1 && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 && ok8 && ok9 && ok11 && okAlgo, kName, detail);
		SafeRelease(cs);
	}

	// SunTauField / SunBright / SunRepHeight / SunTauVertical, through the composite core, against their formulas.
	constexpr const char* kSunProbeMain = R"(
float4 main(VSOut i) : SV_TARGET
{
	const int2   p = int2(i.pos.xy);
	const float2 cuv = (float2(p) + 0.5) / 8.0;
	const float  h = p.x * 9.0 - 6.0;   // -6 .. 57 against a top of 40: under the ground, inside and above
	const float  tau = SunTauField(cuv, h, 40.0);
	return float4(tau, SunBright(0.3, tau), SunRepHeight(-10.0 + 3.0 * p.y, 2.0 * p.x - 3.0, 10.0), SunTauVertical(1.0));
}
)";

	void CheckSunHelpers(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx)
	{
		const char* kName = "SunShadow lookup and blend match their formulas";
		constexpr int N = 8;
		ID3DBlob*     vsb = Compile(RaymarchShaders::kVS, "vs_5_0");
		const std::string probeSrc = std::string(RaymarchShaders::kCompositeCore) + kSunProbeMain;
		ID3DBlob*     pb = Compile(probeSrc.c_str(), "ps_5_0");
		if (!vsb || !pb) {
			SafeRelease(vsb);
			SafeRelease(pb);
			Check(false, kName, "shaders did not compile");
			return;
		}
		ID3D11VertexShader* vs = nullptr;
		ID3D11PixelShader*  ps = nullptr;
		a_dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
		a_dev->CreatePixelShader(pb->GetBufferPointer(), pb->GetBufferSize(), nullptr, &ps);
		SafeRelease(vsb);
		SafeRelease(pb);
		D3D11_TEXTURE2D_DESC od{};
		od.Width = od.Height = N;
		od.MipLevels = od.ArraySize = 1;
		od.SampleDesc.Count = 1;
		od.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		od.BindFlags = D3D11_BIND_RENDER_TARGET;
		ID3D11Texture2D*        outTex = nullptr;
		ID3D11RenderTargetView* rtv = nullptr;
		a_dev->CreateTexture2D(&od, nullptr, &outTex);
		a_dev->CreateRenderTargetView(outTex, nullptr, &rtv);
		// SunCols: texel (x, y) holds (1, 2, 3, 4) x (1 + 0.25 y) + 0.1 x; sampled at texel centres, so bilinear = the texel
		std::vector<float> sun(N * N * 4);
		for (int y = 0; y < N; ++y) {
			for (int x = 0; x < N; ++x) {
				for (int k = 0; k < 4; ++k) sun[(y * N + x) * 4 + k] = static_cast<float>((k + 1) * (1.0 + 0.25 * y) + 0.1 * x);
			}
		}
		od.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA sd{ sun.data(), N * 16, 0 };
		ID3D11Texture2D*          sunTex = nullptr;
		ID3D11ShaderResourceView* sunSrv = nullptr;
		a_dev->CreateTexture2D(&od, &sd, &sunTex);
		a_dev->CreateShaderResourceView(sunTex, nullptr, &sunSrv);
		D3D11_SAMPLER_DESC smp{};
		smp.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		smp.AddressU = smp.AddressV = smp.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		ID3D11SamplerState* lin = nullptr;
		a_dev->CreateSamplerState(&smp, &lin);
		D3D11_BUFFER_DESC cbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
		ID3D11Buffer*     cb = nullptr;
		a_dev->CreateBuffer(&cbd, nullptr, &cb);
		double      worst = 0.0;
		std::string detail;
		for (int pass = 0; pass < 2; ++pass) {
			RaymarchShaders::RayCB rc{};
			rc.sunDir[2] = pass == 0 ? 0.5f : 0.01f;   // the second pass is under the lowest slant
			rc.sunShadow[0] = 0.8f;
			rc.sunShadow[1] = 0.7f;
			rc.sunShadow[2] = 0.05f;
			rc.sunShadow[3] = 1.0f;
			a_ctx->UpdateSubresource(cb, 0, nullptr, &rc, 0, 0);
			const FLOAT clear[4]{ 0, 0, 0, 0 };
			a_ctx->ClearRenderTargetView(rtv, clear);
			a_ctx->OMSetRenderTargets(1, &rtv, nullptr);
			const D3D11_VIEWPORT vp{ 0, 0, N, N, 0, 1 };
			a_ctx->RSSetViewports(1, &vp);
			a_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			a_ctx->VSSetShader(vs, nullptr, 0);
			a_ctx->PSSetShader(ps, nullptr, 0);
			a_ctx->PSSetConstantBuffers(0, 1, &cb);
			a_ctx->PSSetShaderResources(12, 1, &sunSrv);
			a_ctx->PSSetSamplers(0, 1, &lin);
			a_ctx->Draw(3, 0);
			ID3D11ShaderResourceView* noSrv = nullptr;
			a_ctx->PSSetShaderResources(12, 1, &noSrv);
			ID3D11RenderTargetView* noRtv = nullptr;
			a_ctx->OMSetRenderTargets(1, &noRtv, nullptr);
			const std::vector<float> out = ReadF32(a_dev, a_ctx, outTex, N, N, 4);
			for (int y = 0; y < N; ++y) {
				for (int x = 0; x < N; ++x) {
					const float* s = &sun[(y * N + x) * 4];
					const double h = x * 9.0 - 6.0;
					const double f = std::max(h, 0.0) / 40.0 * 3.0;
					double       tau;
					if (f >= 3.0) tau = s[3] * std::clamp(4.0 - f, 0.0, 1.0);
					else if (f < 1.0) tau = s[0] + (s[1] - s[0]) * f;
					else if (f < 2.0) tau = s[1] + (s[2] - s[1]) * (f - 1.0);
					else tau = s[2] + (s[3] - s[2]) * (f - 2.0);
					const double bright = 0.3 + (std::exp(-0.8 * tau) - 0.3) * 0.7;
					const double ha = -10.0 + 3.0 * y, hb = 2.0 * x - 3.0, top = 10.0;
					const double rep = 0.5 * (std::clamp(std::min(ha, hb), 0.0, top) + std::clamp(std::max(ha, hb), 0.0, top));
					const double vertTau = 1.0 / std::max(static_cast<double>(rc.sunDir[2]), 0.05);
					const float* o = &out[(y * N + x) * 4];
					worst = std::max({ worst, std::fabs(o[0] - tau) / std::max(1.0, tau), std::fabs(o[1] - bright), std::fabs(o[2] - rep),
						std::fabs(o[3] - vertTau) / vertTau });
				}
			}
		}
		// without SunCols (w = 0) the field depth is 0
		RaymarchShaders::RayCB rc{};
		rc.sunShadow[0] = 0.8f;
		rc.sunShadow[1] = 1.0f;
		rc.sunShadow[2] = 0.05f;
		rc.sunDir[2] = 0.5f;
		a_ctx->UpdateSubresource(cb, 0, nullptr, &rc, 0, 0);
		a_ctx->OMSetRenderTargets(1, &rtv, nullptr);
		a_ctx->PSSetShaderResources(12, 1, &sunSrv);
		a_ctx->Draw(3, 0);
		ID3D11ShaderResourceView* noSrv = nullptr;
		a_ctx->PSSetShaderResources(12, 1, &noSrv);
		ID3D11RenderTargetView* noRtv = nullptr;
		a_ctx->OMSetRenderTargets(1, &noRtv, nullptr);
		const std::vector<float> off = ReadF32(a_dev, a_ctx, outTex, N, N, 4);
		double offMax = 0.0;
		for (int i = 0; i < N * N; ++i) offMax = std::max(offMax, static_cast<double>(std::fabs(off[i * 4 + 0])) + std::fabs(off[i * 4 + 1] - 1.0));
		detail = "worst error " + std::to_string(worst) + " (sun z 0.5 and 0.01, heights under, inside and over the top); without SunCols depth 0 and lit 1: off by " +
		         std::to_string(offMax);
		Check(worst < 1e-4 && offMax < 1e-6, kName, detail);
		SafeRelease(cb);
		SafeRelease(lin);
		SafeRelease(sunSrv);
		SafeRelease(sunTex);
		SafeRelease(rtv);
		SafeRelease(outTex);
		SafeRelease(ps);
		SafeRelease(vs);
	}
}

namespace
{
	// IniEdit::SetValue (the in-game page's ini writer, Settings::WriteValue): the key's line rewritten in place, every
	// other line kept, CRLF kept, a same-named key in another section untouched, a missing key after the section's last
	// key, a missing section at the end, case-insensitive names.
	void CheckIniEdit()
	{
		const std::string in = "; head\r\n[General]\r\nEnable = 1\r\n\r\n[Render]\r\n; opacity note\r\nOpacity = 0.18 ; inline\r\nEnable = 7\r\n\r\n; far fog notes\r\n[FarFog]\r\nEnable = 0\r\n";
		const std::string a = IniEdit::SetValue(in, "render", "OPACITY", "0.5");
		const std::string aWant = "; head\r\n[General]\r\nEnable = 1\r\n\r\n[Render]\r\n; opacity note\r\nOpacity = 0.5 ; inline\r\nEnable = 7\r\n\r\n; far fog notes\r\n[FarFog]\r\nEnable = 0\r\n";
		const std::string b = IniEdit::SetValue(in, "General", "Enable", "0");
		const bool        bOk = b.find("[General]\r\nEnable = 0\r\n") != std::string::npos && b.find("Enable = 7") != std::string::npos &&
		                 b.find("[FarFog]\r\nEnable = 0") != std::string::npos;
		const std::string c = IniEdit::SetValue(in, "Render", "SunShadow", "1");
		const bool        cOk = c.find("Enable = 7\r\nSunShadow = 1\r\n\r\n; far fog notes") != std::string::npos;
		const std::string d = IniEdit::SetValue(in, "Menu", "Shown", "1");
		const bool        dOk = d.size() > in.size() && d.substr(d.size() - 19) == "[Menu]\r\nShown = 1\r\n";
		const std::string e = IniEdit::SetValue("[Render]\nOpacity = 1", "Render", "Opacity", "2");
		const bool        eOk = e == "[Render]\nOpacity = 2\n";
		const std::string f = IniEdit::SetValue("[Render]\nOpacity = 1\nX = 0\nopacity=3\n[Wind]\nOpacity = 9\n", "Render", "Opacity", "5");
		const bool        fOk = f == "[Render]\nOpacity = 5\nX = 0\nopacity = 5\n[Wind]\nOpacity = 9\n";
		Check(a == aWant && bOk && cOk && dOk && eOk && fOk, "ini writer sets one key and keeps the rest",
			std::string("replace in section with inline comment ") + (a == aWant ? "ok" : "WRONG") + ", same key in other sections untouched " +
				(bOk ? "ok" : "WRONG") + ", missing key after the section's last key " + (cOk ? "ok" : "WRONG") + ", missing section appended " +
				(dOk ? "ok" : "WRONG") + ", LF file without a last line end " + (eOk ? "ok" : "WRONG") + ", a key written twice set both times " +
				(fOk ? "ok" : "WRONG"));
	}
}

namespace
{
	// LocalHalo (LocalLights): the closed-form ray integral against a fine numeric sum of the same falloff and fade.
	constexpr const char* kHaloProbeMain = R"(
float4 main(VSOut i) : SV_TARGET
{
	const int2  p = int2(i.pos.xy);
	const float R = 300.0;
	const float3 light = float3(0.0, 2000.0, 0.0);
	const float  side = (p.x - 3.5) * 150.0;          // ray passes at |side| from the light
	const float  dist = p.y == 0 ? 1e9 : (p.y == 1 ? 1500.0 : 2500.0);   // open sky, ends before the light, ends past it
	const float3 dir = normalize(float3(side, 2000.0, 0.0) - float3(0.0, 0.0, 0.0));
	return float4(LocalHalo(float3(0.0, 0.0, 0.0), dir, dist, light, R), 0.0, 0.0, 0.0);
}
)";

	void CheckLocalHalo(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx)
	{
		const char* kName = "LocalLights halo matches a numeric ray integral";
		ID3DBlob*   vsb = Compile(RaymarchShaders::kVS, "vs_5_0");
		const std::string src = std::string(RaymarchShaders::kCompositeCore) + kHaloProbeMain;
		ID3DBlob*   pb = Compile(src.c_str(), "ps_5_0");
		if (!vsb || !pb) {
			SafeRelease(vsb);
			SafeRelease(pb);
			Check(false, kName, "shaders did not compile");
			return;
		}
		ID3D11VertexShader* vs = nullptr;
		ID3D11PixelShader*  ps = nullptr;
		a_dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
		a_dev->CreatePixelShader(pb->GetBufferPointer(), pb->GetBufferSize(), nullptr, &ps);
		SafeRelease(vsb);
		SafeRelease(pb);
		constexpr int        W = 8, H = 3;
		D3D11_TEXTURE2D_DESC od{};
		od.Width = W;
		od.Height = H;
		od.MipLevels = od.ArraySize = 1;
		od.SampleDesc.Count = 1;
		od.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		od.BindFlags = D3D11_BIND_RENDER_TARGET;
		ID3D11Texture2D*        tex = nullptr;
		ID3D11RenderTargetView* rtv = nullptr;
		a_dev->CreateTexture2D(&od, nullptr, &tex);
		a_dev->CreateRenderTargetView(tex, nullptr, &rtv);
		D3D11_BUFFER_DESC cbd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
		RaymarchShaders::RayCB rc{};
		D3D11_SUBRESOURCE_DATA cbi{ &rc, 0, 0 };
		ID3D11Buffer*          cb = nullptr;
		a_dev->CreateBuffer(&cbd, &cbi, &cb);
		a_ctx->OMSetRenderTargets(1, &rtv, nullptr);
		const D3D11_VIEWPORT vp{ 0, 0, W, H, 0, 1 };
		a_ctx->RSSetViewports(1, &vp);
		a_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		a_ctx->VSSetShader(vs, nullptr, 0);
		a_ctx->PSSetShader(ps, nullptr, 0);
		a_ctx->PSSetConstantBuffers(0, 1, &cb);
		a_ctx->Draw(3, 0);
		ID3D11RenderTargetView* none = nullptr;
		a_ctx->OMSetRenderTargets(1, &none, nullptr);
		const std::vector<float> out = ReadF32(a_dev, a_ctx, tex, W, H, 4);
		double worst = 0.0, centre = 0.0, past = 0.0, cut = 0.0;
		for (int y = 0; y < H; ++y) {
			for (int x = 0; x < W; ++x) {
				const double R = 300.0, side = (x - 3.5) * 150.0, dist = y == 0 ? 1e6 : (y == 1 ? 1500.0 : 2500.0);
				const double len = std::sqrt(side * side + 2000.0 * 2000.0);
				const double dx = side / len, dy = 2000.0 / len;
				const double tc = 2000.0 * dy, h = std::sqrt(std::max(2000.0 * 2000.0 - tc * tc, 0.0));
				const double end = std::min(dist, 20000.0);
				constexpr int kN = 400000;
				double        sum = 0.0;
				for (int s = 0; s < kN; ++s) {
					const double tt = (s + 0.5) * end / kN;
					const double px = dx * tt, py = dy * tt;
					const double d2 = px * px + (py - 2000.0) * (py - 2000.0);
					sum += 1.0 / (1.0 + 64.0 * d2 / (R * R)) * end / kN;   // kHaloCore 64 (2026-11-17)
				}
				const double e = std::clamp((h - R) / (2.0 * R), 0.0, 1.0);
				const double want = sum * std::sqrt(64.0 / 8.0) / R * (1.0 - e * e * (3.0 - 2.0 * e));
				const double got = out[(y * W + x) * 4];
				worst = std::max(worst, std::fabs(got - want) / std::max(0.05, want));
				if (y == 0 && (x == 3 || x == 4)) centre = got;
				if (y == 1 && (x == 3 || x == 4)) past = got;
				if (y == 0 && x == 0) cut = got;
			}
		}
		// the open ray 75 units (R / 4) off: 45 % of the 1.11 peak by the design, 0.50 (was > 0.8 with the core of 8)
		Check(worst < 0.03 && centre > 0.4 && centre < 0.65 && past < 0.6 * centre && cut < 0.3 * centre, kName,
			"worst relative error " + std::to_string(worst) + "; ray past the light (open) " + std::to_string(centre) + ", the same ray ending 500 units short " +
				std::to_string(past) + ", a ray passing 525 units off on a light of radius 300 " + std::to_string(cut));
		SafeRelease(cb);
		SafeRelease(rtv);
		SafeRelease(tex);
		SafeRelease(ps);
		SafeRelease(vs);
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
	CheckIniEdit();
	CheckFogBlur(dev, ctx);
	CheckFogTemporal(dev, ctx);
	CheckPickAndJitter(dev, ctx);
	CheckTemporalDepthPick(dev, ctx);
	CheckSunCols(dev, ctx);
	CheckSunHelpers(dev, ctx);
	CheckLocalHalo(dev, ctx);
	{
		// LightFogWeight (H-ll4). Limits from the design: a brazier on the ground 1, a torch in the hand (110 units up) in a
		// layer 126 tall at steepness 3 under 0.1, a light on a post over the layer's top 0, under the ground 1, no layer 0.
		const float ground = RaymarchShaders::LightFogWeight(0.0f, 126.0f, 3.0f);
		const float hand = RaymarchShaders::LightFogWeight(110.0f, 126.0f, 3.0f);
		const float post = RaymarchShaders::LightFogWeight(150.0f, 126.0f, 3.0f);
		const float below = RaymarchShaders::LightFogWeight(-20.0f, 126.0f, 3.0f);
		const float none = RaymarchShaders::LightFogWeight(50.0f, 0.0f, 3.0f);
		Check(ground == 1.0f && hand > 0.0f && hand < 0.1f && post == 0.0f && below == 1.0f && none == 0.0f, "LocalLights weighted by the fog at the light's height",
			"ground " + std::to_string(ground) + ", hand " + std::to_string(hand) + ", post " + std::to_string(post) + ", under the ground " + std::to_string(below) +
				", no layer " + std::to_string(none));
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


	// Day fog share (WeatherRules.h): Azurite's SkyrimFog at noon (sky far = day far 15000) keeps its 2.5, at night
	// (sky far 300000) the boost falls to 1.075, halfway through a transition from clear the day distance is blended;
	// clear weathers (below 1) and unusable input are left alone.
	{
		namespace WR = WeatherRules;
		const float noon = WR::ApplyDayShare(2.5f, WR::DayFogShare(15000.0f, 15000.0f, 1.0f, 15000.0f));
		const float night = WR::ApplyDayShare(2.5f, WR::DayFogShare(15000.0f, 15000.0f, 1.0f, 300000.0f));
		const float half = WR::DayFogShare(15000.0f, 300000.0f, 0.5f, 157500.0f);   // blended day far 157500 = sky far
		const float clearNight = WR::ApplyDayShare(0.45f, WR::DayFogShare(300000.0f, 300000.0f, 1.0f, 300000.0f));
		const float badSky = WR::DayFogShare(15000.0f, 15000.0f, 1.0f, std::numeric_limits<float>::quiet_NaN());
		const float zeroSky = WR::DayFogShare(15000.0f, 15000.0f, 1.0f, 0.0f);
		const float badPct = WR::DayFogShare(15000.0f, 300000.0f, std::numeric_limits<float>::quiet_NaN(), 15000.0f);
		Check(std::fabs(noon - 2.5f) < 1e-5f && std::fabs(night - 1.075f) < 1e-5f && std::fabs(half - 1.0f) < 1e-5f && std::fabs(clearNight - 0.45f) < 1e-6f &&
				  badSky == 1.0f && zeroSky == 1.0f && std::fabs(badPct - 1.0f) < 1e-6f,
			"fog weather boost follows the game's own fog",
			"noon " + std::to_string(noon) + " (2.5), night " + std::to_string(night) + " (1.075), half-way share " + std::to_string(half) +
				", clear night " + std::to_string(clearNight) + ", NaN sky " + std::to_string(badSky) + ", zero sky " + std::to_string(zeroSky) +
				", NaN transition " + std::to_string(badPct));
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
		// Last Seed: the month whose night steam is neither 0 nor full, so the fade of the night shows in the steam.
		const auto before7 = run(6.99f, 7, 0.0f, true, false, false, -1.0f);
		const auto after7 = run(7.01f, 7, 0.0f, true, false, false, -1.0f);
		const auto before20 = run(19.99f, 7, 0.0f, true, false, false, -1.0f);
		const auto after20 = run(20.01f, 7, 0.0f, true, false, false, -1.0f);
		const bool smooth = std::fabs(before7.cold - after7.cold) < 0.02f && std::fabs(before20.cold - after20.cold) < 0.02f && before7.cold > 0.05f;
		const bool ok = same(dawnClear.time, 1.6f) && same(dawnClear.global, 1.6f * 1.25f) && same(noonClear.time, 0.35f) &&
						same(noonClear.global, 0.35f * 0.75f) && same(dawnCloudy.time, 1.3f) && same(dawnWindy.wind, 0.3f) &&
						same(dawnWindy.global, 1.6f * 0.3f) && same(survival.air, 0.2f) && same(survival.cold, 0.375f) && same(snowNoon.cold, 1.0f) &&
						same(noonClear.cold, 0.0f) && same(dawnClear.air, 0.665f) && same(dawnClear.cold, 0.7875f) && same(springCold.cold, 0.0f) && smooth &&
						same(noonClear.evaporation, 0.2f * (1.0f - 0.35f * 0.75f)) && same(dawnClear.evaporation, 0.0f);
		Check(ok, "fog conditions (time, sky, wind, season, cold)",
			"dawn clear " + std::to_string(dawnClear.global) + ", noon " + std::to_string(noonClear.global) + ", dawn cloudy time " +
				std::to_string(dawnCloudy.time) + ", windy " + std::to_string(dawnWindy.global) + ", survival cold " + std::to_string(survival.cold) +
				", snow cold " + std::to_string(snowNoon.cold) + ", spring steam " + std::to_string(springCold.cold) + ", steam 06:59/07:01 " +
				std::to_string(before7.cold) + "/" + std::to_string(after7.cold) + ", 19:59/20:01 " + std::to_string(before20.cold) + "/" + std::to_string(after20.cold));

		// Steam by month without a survival mod (2026-11-14): clear, calm, at night 02:00, dawn 06:30 and noon 13:00.
		// The night is added to the month's air cold (it used to be max(month, 0.6): full steam on clear nights from
		// Midyear to Frostfall, 0.63 / 0.38 / 0.25 in the first three months). Wanted: most in late autumn and early
		// winter, least in summer (a tie at 0 with spring, when the water is colder than the air), summer nights at
		// most 0.35, and a survival mod's cold used as it is whatever the hour.
		const std::array<float, 12> wantNight{ 0.95f, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.325f, 0.2f, 0.7f, 1.0f, 1.0f };
		std::array<float, 12> night{}, dawn{}, noon{};
		bool tableOk = true, addsOk = true, survivalOk = true;
		for (int m = 0; m < 12; ++m) {
			const auto rn = run(2.0f, m, 0.0f, true, false, false, -1.0f);
			const auto rd = run(6.5f, m, 0.0f, true, false, false, -1.0f);
			const auto ry = run(13.0f, m, 0.0f, true, false, false, -1.0f);
			const auto i = static_cast<std::size_t>(m);
			night[i] = rn.cold;
			dawn[i] = rd.cold;
			noon[i] = ry.cold;
			tableOk = tableOk && same(rn.cold, wantNight[i]) && rd.cold <= rn.cold && ry.cold <= rd.cold;
			addsOk = addsOk && std::fabs(rn.air - ry.air - p.nightCold) < 1e-6f && std::fabs(rd.air - ry.air - 0.5f * p.nightCold) < 1e-6f;
			const auto sn = run(2.0f, m, 0.0f, true, false, false, 0.2f);
			const auto sy = run(13.0f, m, 0.0f, true, false, false, 0.2f);
			survivalOk = survivalOk && sn.air == 0.2f && sy.air == 0.2f && sn.cold == sy.cold;
		}
		const float yearMax = *std::max_element(night.begin(), night.end());
		const float yearMin = *std::min_element(night.begin(), night.end());
		const float summerMax = std::max({ night[5], night[6], night[7] });
		const float summerMin = std::min({ night[5], night[6], night[7] });
		bool maxInSeason = false, maxOutside = false;
		for (std::size_t i = 0; i < 12; ++i) {
			const bool lateAutumnWinter = i == 9 || i == 10 || i == 11 || i == 0;   // Frostfall .. Morning Star
			maxInSeason = maxInSeason || (lateAutumnWinter && night[i] == yearMax);
			maxOutside = maxOutside || (!lateAutumnWinter && night[i] == yearMax);
		}
		const bool orderOk = maxInSeason && !maxOutside && summerMin == yearMin && summerMax <= 0.35f &&
							 std::min({ night[11], night[0], night[1] }) >= summerMax && std::min({ night[9], night[10], night[11], night[0] }) >= summerMax;
		auto row = [](const std::array<float, 12>& a_v) {
			std::string s;
			char        buf[16];
			for (float v : a_v) {
				std::snprintf(buf, sizeof(buf), "%s%.3f", s.empty() ? "" : " ", v);
				s += buf;
			}
			return s;
		};
		Check(tableOk && addsOk && survivalOk && orderOk, "steam by month without a survival mod (night adds to the month)",
			"night " + row(night) + " | dawn " + row(dawn) + " | noon " + row(noon) + " | summer night max " + std::to_string(summerMax) + ", table " +
				(tableOk ? "ok" : "WRONG") + ", night adds " + (addsOk ? "ok" : "WRONG") + ", order " + (orderOk ? "ok" : "WRONG") + ", survival " +
				(survivalOk ? "ok" : "WRONG"));
		// [Conditions] NightCold: the shipping default is the header's value, so the table above is what a default ini gives.
		Check(Settings::condNightCold == FogConditions::Params{}.nightCold, "night cold: the ini default is the header's value",
			"Settings " + std::to_string(Settings::condNightCold) + ", FogConditions " + std::to_string(FogConditions::Params{}.nightCold));
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
							 PlacementRules::IsMarshLand("RP_LMarshMud01") && !PlacementRules::IsMarshLand("LTundra01") &&
							 !PlacementRules::IsMarshLand("LRiverMud01");   // a river bed is not marsh (2026-11-13)
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
		PlacementRules::Params p0;
		p0.lake = 0.0f;
		p0.marshLand = 5.0f;
		p0.shoreDistance = 0.0f;
		p0.land = 1.0f;
		const auto o0 = PlacementRules::Build(g, k, n, 64.0f, p0);
		// Marsh land is blended with the dry land around it over 512 units (8 cells of 64): the middle of a marsh as wide
		// as the grid keeps its multiplier, the multiplier rises steadily across the border and no neighbouring cells
		// differ by more than a sixth of the whole step.
		std::vector<Kind> wide(n * n, Kind::kMarshLand);
		const auto        ow = PlacementRules::Build(g, wide, n, 64.0f, p0);
		float              maxStep = 0.0f;
		bool               rising = true;
		const int          nb = 41;   // wide enough for the 8-cell blend on both sides of the border at column 20
		std::vector<float> gb(nb * nb, 0.0f);
		std::vector<Kind>  border(nb * nb, Kind::kLand);
		for (int y = 0; y < nb; ++y)
			for (int x = 20; x < nb; ++x) border[y * nb + x] = Kind::kMarshLand;
		const auto ob = PlacementRules::Build(gb, border, nb, 64.0f, p0);
		for (int x = 1; x < nb; ++x) {
			const float d = ob[20 * nb + x][0] - ob[20 * nb + x - 1][0];
			maxStep = std::max(maxStep, std::fabs(d));
			rising = rising && d >= -1e-4f;
		}
		const float dry = ob[20 * nb + 4][0], wet = ob[20 * nb + 36][0];
		const bool  keepOk = o0[4 * n + 4][0] == 0.0f && std::fabs(ow[4 * n + 6][0] - 5.0f) < 1e-4f && rising && maxStep < 4.0f / 6.0f &&
		                     std::fabs(dry - 1.0f) < 1e-4f && std::fabs(wet - 5.0f) < 1e-4f;
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
			"lake 0 -> " + std::to_string(o0[4 * n + 4][0]) + ", marsh land 5 -> " + std::to_string(ow[4 * n + 6][0]) + " (all marsh), across a border " + std::to_string(dry) + ".." +
				std::to_string(wet) + " largest step " + std::to_string(maxStep) + ", next to water " +
				std::to_string(o0[4 * n + 3][0]) + ", slope spot before/after the move " + std::to_string(before) + "/" + std::to_string(after));
	}

	// Shore summary (PlacementRules::SummarizeShore, the "shore:" diagnostic log line): does bank land exceed the water
	// beside it? A river down column 64 of a 129 grid with the test-ini values of the ledger (land 0.7, shore 1.3 over
	// 700 units = 11 cells, river 1.1). Flat bank 0.91 stays under the river; a 20-degree bank reaches 0.7 x 1.3 x 1.8 =
	// 1.638; a hollow 12 cells out is past the band and counted as land beyond it, one 11 cells out is bank; unknown
	// cells are not counted.
	{
		using PlacementRules::Kind;
		const int   n = 129, mid = 64;
		const float cell = 64.0f;
		PlacementRules::Params pp;
		pp.land = 0.7f;
		pp.shoreFactor = 1.3f;
		pp.shoreDistance = 700.0f;
		pp.lake = 1.4f;
		pp.river = 1.1f;
		pp.freshSteam = 1.0f;
		std::vector<Kind> k(static_cast<std::size_t>(n) * n, Kind::kLand);
		for (int y = 0; y < n; ++y) k[static_cast<std::size_t>(y) * n + mid] = Kind::kRiver;
		auto run = [&](const std::vector<float>& a_g) { return PlacementRules::SummarizeShore(PlacementRules::Build(a_g, k, n, cell, pp), a_g, k, n, cell, pp); };
		auto close = [](float a_v, float a_want) { return std::fabs(a_v - a_want) < 2e-3f; };
		const std::vector<float> flat(static_cast<std::size_t>(n) * n, 0.0f);
		std::vector<float>       steep(flat.size());
		const float              rise = cell * std::tan(20.0f * 3.14159265f / 180.0f);
		for (int y = 0; y < n; ++y)
			for (int x = 0; x < n; ++x) steep[static_cast<std::size_t>(y) * n + x] = rise * static_cast<float>(std::abs(x - mid));
		std::vector<float> hollow12 = flat, hollow11 = flat, unknown = flat;
		hollow12[static_cast<std::size_t>(mid) * n + mid + 12] = -600.0f;
		hollow11[static_cast<std::size_t>(mid) * n + mid + 11] = -600.0f;
		for (int y = 0; y < n; ++y) unknown[static_cast<std::size_t>(y) * n + mid + 1] = -1.0e30f;
		const auto a = run(flat), b = run(steep), c = run(hollow12), d = run(hollow11), e = run(unknown);
		const auto riverI = static_cast<std::size_t>(Kind::kRiver);
		const bool flatOk = a.valid && a.band == 11 && close(a.bankLand, 0.91f) && a.bankAbove == 0 && a.bankExcess == 0.0f && close(a.water[riverI], 1.1f) &&
		                    a.water[static_cast<std::size_t>(Kind::kLake)] < 0.0f && a.water[static_cast<std::size_t>(Kind::kMarshWater)] < 0.0f &&
		                    a.water[static_cast<std::size_t>(Kind::kSea)] < 0.0f && a.water[static_cast<std::size_t>(Kind::kHotSpring)] < 0.0f &&
		                    close(a.steamZ, 1.0f) && a.steamY == 0.0f && close(a.landBeyond, 0.7f) && a.bankMarsh < 0.0f;
		const bool steepOk = close(b.bankLand, 1.638f) && b.bankAbove > 0 && close(b.bankExcess, 0.538f);
		const bool hollowOk = close(c.bankLand, 0.91f) && c.bankAbove == 0 && close(c.landBeyond, 1.26f) && close(d.bankLand, 1.294f) && d.bankAbove == 1 &&
		                      close(d.bankExcess, 0.194f);
		const bool unknownOk = close(e.bankLand, 0.91f) && e.bankAbove == 0 && e.bankCells < a.bankCells;
		Check(flatOk && steepOk && hollowOk && unknownOk, "shore summary (flat bank, 20-degree bank, hollow past and inside the band, unknown cells)",
			"flat: " + PlacementRules::DescribeShore(a, 0.5f, 0.7f) + "; 20 deg: bank " + std::to_string(b.bankLand) + ", " + std::to_string(b.bankAbove) +
				" above, excess " + std::to_string(b.bankExcess) + "; hollow at 12 cells: bank " + std::to_string(c.bankLand) + ", " + std::to_string(c.bankAbove) +
				" above, beyond " + std::to_string(c.landBeyond) + "; at 11 cells: bank " + std::to_string(d.bankLand) + ", " + std::to_string(d.bankAbove) +
				" above, excess " + std::to_string(d.bankExcess) + "; unknown bank column: bank " + std::to_string(e.bankLand) + ", cells " +
				std::to_string(e.bankCells) + " of " + std::to_string(a.bankCells));
	}

	// Bank cap (PlacementRules::Params::bankCap, [Placement] BankCap, off by default): land in the shore band is held to
	// the water it borders. The scenes of the shore summary above (river down column 64, land 0.7, shore 1.3 over 11
	// cells, river 1.1, lake 1.4), each built with the cap off and on. Flat bank and a hollow past the band: not one bit
	// changes. 20-degree bank: the land touching the water comes down from 1.638 to the river's 1.1. Hollows 11 and 10
	// cells out: the band's outermost cell is untouched, one cell further in is lowered by a tenth of its excess. Two
	// kinds of water on either side of one bank cell: the larger (the lake's 1.4), whichever side the lake is on.
	// Nothing is raised anywhere, steam and pool depth are untouched, and the largest step across the band's outer edge
	// is no larger than without the cap; the same on a seeded random grid with every kind and default values.
	{
		using PlacementRules::Kind;
		const int         n = 129, mid = 64;
		const float       cell = 64.0f;
		const std::size_t count = static_cast<std::size_t>(n) * n;
		PlacementRules::Params pp;
		pp.land = 0.7f;
		pp.shoreFactor = 1.3f;
		pp.shoreDistance = 700.0f;
		pp.lake = 1.4f;
		pp.river = 1.1f;
		pp.freshSteam = 1.0f;
		auto at = [&](int a_x, int a_y) { return static_cast<std::size_t>(a_y) * n + a_x; };
		auto isWater = [](Kind a_k) { return a_k == Kind::kLake || a_k == Kind::kRiver || a_k == Kind::kMarshWater || a_k == Kind::kHotSpring || a_k == Kind::kSea; };
		struct Pair
		{
			std::vector<std::array<float, 4>> off, on;
			float raise{ -1.0e30f };   // largest on.x - off.x
			float lower{ 0.0f };       // smallest on.x - off.x
			bool  restSame{ true };    // y, z, w bit-identical
			bool  same{ true };        // x too
			float edgeOff{ 0.0f }, edgeOn{ 0.0f };   // largest step across the band's outer edge
			float stepOff{ 0.0f }, stepOn{ 0.0f };   // largest step between neighbouring land cells anywhere (reported)
		};
		auto run = [&](const std::vector<float>& a_g, const std::vector<Kind>& a_k, PlacementRules::Params a_p) {
			Pair r;
			a_p.bankCap = false;
			r.off = PlacementRules::Build(a_g, a_k, n, cell, a_p);
			a_p.bankCap = true;
			r.on = PlacementRules::Build(a_g, a_k, n, cell, a_p);
			// The band as Build has it: 8-neighbour steps from the known water cells, at most band of them.
			const int               band = std::max(1, static_cast<int>(std::ceil(a_p.shoreDistance / cell)));
			std::vector<int>        dist(count, -1);
			std::deque<std::size_t> queue;
			auto                    known = [&](std::size_t a_i) { return a_g[a_i] > -1.0e29f; };
			for (std::size_t i = 0; i < count; ++i) {
				if (known(i) && isWater(a_k[i])) {
					dist[i] = 0;
					queue.push_back(i);
				}
			}
			auto each = [&](std::size_t a_i, auto&& a_fn) {
				const int x = static_cast<int>(a_i % n), y = static_cast<int>(a_i / n);
				for (int dy = -1; dy <= 1; ++dy) {
					for (int dx = -1; dx <= 1; ++dx) {
						if ((dx == 0 && dy == 0) || x + dx < 0 || y + dy < 0 || x + dx >= n || y + dy >= n) continue;
						a_fn(at(x + dx, y + dy));
					}
				}
			};
			while (!queue.empty()) {
				const std::size_t i = queue.front();
				queue.pop_front();
				if (dist[i] >= band) continue;
				each(i, [&](std::size_t a_j) {
					if (dist[a_j] < 0) {
						dist[a_j] = dist[i] + 1;
						queue.push_back(a_j);
					}
				});
			}
			auto land = [&](std::size_t a_i) { return known(a_i) && !isWater(a_k[a_i]); };
			for (std::size_t i = 0; i < count; ++i) {
				const float d = r.on[i][0] - r.off[i][0];
				r.raise = std::max(r.raise, d);
				r.lower = std::min(r.lower, d);
				r.same = r.same && std::memcmp(&r.on[i][0], &r.off[i][0], sizeof(float)) == 0;
				r.restSame = r.restSame && std::memcmp(&r.on[i][1], &r.off[i][1], 3 * sizeof(float)) == 0;
				if (!land(i)) continue;
				each(i, [&](std::size_t a_j) {
					if (!land(a_j)) return;
					r.stepOff = std::max(r.stepOff, std::fabs(r.off[i][0] - r.off[a_j][0]));
					r.stepOn = std::max(r.stepOn, std::fabs(r.on[i][0] - r.on[a_j][0]));
					if (dist[i] == band && dist[a_j] < 0) {
						r.edgeOff = std::max(r.edgeOff, std::fabs(r.off[i][0] - r.off[a_j][0]));
						r.edgeOn = std::max(r.edgeOn, std::fabs(r.on[i][0] - r.on[a_j][0]));
					}
				});
			}
			return r;
		};
		auto close = [](float a_v, float a_want) { return std::fabs(a_v - a_want) < 2e-3f; };
		auto sound = [](const Pair& a_r) { return a_r.raise <= 0.0f && a_r.restSame && a_r.edgeOn <= a_r.edgeOff; };

		std::vector<Kind> river(count, Kind::kLand);
		for (int y = 0; y < n; ++y) river[at(mid, y)] = Kind::kRiver;
		const std::vector<float> flat(count, 0.0f);
		std::vector<float>       steep(count);
		const float              rise = cell * std::tan(20.0f * 3.14159265f / 180.0f);
		for (int y = 0; y < n; ++y)
			for (int x = 0; x < n; ++x) steep[at(x, y)] = rise * static_cast<float>(std::abs(x - mid));
		std::vector<float> hollow12 = flat, hollow11 = flat, hollow10 = flat, unknown = flat, hollow1 = flat;
		hollow12[at(mid + 12, mid)] = -600.0f;
		hollow11[at(mid + 11, mid)] = -600.0f;
		hollow10[at(mid + 10, mid)] = -600.0f;
		hollow1[at(mid + 1, mid)] = -600.0f;
		for (int y = 0; y < n; ++y) unknown[at(mid + 1, y)] = -1.0e30f;
		const Pair a = run(flat, river, pp), b = run(steep, river, pp), c = run(hollow12, river, pp), d = run(hollow11, river, pp),
				   e = run(hollow10, river, pp), u = run(unknown, river, pp);
		const bool flatOk = a.same && close(a.on[at(mid + 1, mid)][0], 0.91f) && sound(a);
		const bool steepOk = close(b.off[at(mid + 1, mid)][0], 1.638f) && close(b.off[at(mid - 1, mid)][0], 1.638f) &&
		                     b.on[at(mid + 1, mid)][0] <= 1.1f + 2e-3f && b.on[at(mid - 1, mid)][0] <= 1.1f + 2e-3f && b.lower < -0.5f && sound(b);
		const bool hollowOk = c.same && close(c.on[at(mid + 12, mid)][0], 1.26f) && sound(c) &&
		                      close(d.off[at(mid + 11, mid)][0], 1.294f) && d.on[at(mid + 11, mid)][0] == d.off[at(mid + 11, mid)][0] && sound(d) &&
		                      close(e.off[at(mid + 10, mid)][0], 1.329f) && close(e.on[at(mid + 10, mid)][0], 1.306f) && sound(e) && sound(u);

		// Tie: one bank column between a river and a lake, a hollow in it so that its old value is above both.
		std::vector<Kind> riverLake = river, lakeRiver(count, Kind::kLand);
		for (int y = 0; y < n; ++y) {
			riverLake[at(mid + 2, y)] = Kind::kLake;
			lakeRiver[at(mid, y)] = Kind::kLake;
			lakeRiver[at(mid + 2, y)] = Kind::kRiver;
		}
		const Pair t1 = run(hollow1, riverLake, pp), t2 = run(hollow1, lakeRiver, pp), t0 = run(hollow1, river, pp);
		const std::size_t spot = at(mid + 1, mid);
		const bool tieOk = close(t1.off[spot][0], 1.638f) && close(t1.on[spot][0], 1.4f) && close(t2.on[spot][0], 1.4f) && close(t0.on[spot][0], 1.1f) &&
		                   sound(t1) && sound(t2) && sound(t0);

		// Seeded random grid (LCG, seed below): rough heights 0..1500, every kind, some unknown cells, default values.
		std::uint32_t seed = 20261114u;
		auto          next = [&]() {
			seed = seed * 1664525u + 1013904223u;
			return static_cast<float>(seed >> 8) / 16777216.0f;
		};
		std::vector<float> rough(count);
		std::vector<Kind>  mixed(count, Kind::kLand);
		for (auto& h : rough) h = 1500.0f * next();
		for (int y = 0; y < n; ++y) {
			for (int x = 0; x < n; ++x) {
				Kind k = Kind::kLand;
				if (x >= 20 && x < 22) k = Kind::kRiver;
				if (y >= 90 && y < 100 && x >= 40 && x < 70) k = Kind::kLake;
				if (y >= 30 && y < 36 && x >= 60 && x < 66) k = Kind::kMarshWater;
				if (y >= 28 && y < 40 && x >= 66 && x < 80) k = Kind::kMarshLand;
				if (y >= 60 && y < 63 && x >= 100 && x < 103) k = Kind::kHotSpring;
				if (y < 8) k = Kind::kSea;
				mixed[at(x, y)] = k;
				if (k != Kind::kLand && k != Kind::kMarshLand) rough[at(x, y)] = k == Kind::kSea ? 0.0f : 300.0f;
				if (next() < 0.01f) rough[at(x, y)] = -1.0e30f;
			}
		}
		const Pair r = run(rough, mixed, PlacementRules::Params{});
		const bool randomOk = sound(r) && r.lower < 0.0f;

		Check(flatOk && steepOk && hollowOk && tieOk && randomOk,
			"bank cap (off by default; flat bank and far hollow unchanged, 20-degree bank down to the river's value, nothing raised, band edge, two water kinds)",
			"flat: beside the water " + std::to_string(a.on[at(mid + 1, mid)][0]) + (a.same ? ", grid identical" : ", GRID CHANGED") + "; 20 deg: beside the water " +
				std::to_string(b.off[at(mid + 1, mid)][0]) + " -> " + std::to_string(b.on[at(mid + 1, mid)][0]) + ", largest raise " + std::to_string(b.raise) +
				", largest cut " + std::to_string(b.lower) + ", step across the band edge " + std::to_string(b.edgeOff) + " -> " + std::to_string(b.edgeOn) +
				", largest land step " + std::to_string(b.stepOff) + " -> " + std::to_string(b.stepOn) + "; hollow at 12 cells: " +
				std::to_string(c.on[at(mid + 12, mid)][0]) + (c.same ? ", grid identical" : ", GRID CHANGED") + "; at 11: " + std::to_string(d.off[at(mid + 11, mid)][0]) +
				" -> " + std::to_string(d.on[at(mid + 11, mid)][0]) + " (edge step " + std::to_string(d.edgeOff) + " -> " + std::to_string(d.edgeOn) + "); at 10: " +
				std::to_string(e.off[at(mid + 10, mid)][0]) + " -> " + std::to_string(e.on[at(mid + 10, mid)][0]) + " (edge step " + std::to_string(e.edgeOff) + " -> " +
				std::to_string(e.edgeOn) + ", land step " + std::to_string(e.stepOff) + " -> " + std::to_string(e.stepOn) + "); river | bank | lake: " +
				std::to_string(t1.off[spot][0]) + " -> " + std::to_string(t1.on[spot][0]) + ", lake | bank | river: " + std::to_string(t2.on[spot][0]) + ", river only: " +
				std::to_string(t0.on[spot][0]) + "; random grid: largest raise " + std::to_string(r.raise) + ", largest cut " + std::to_string(r.lower) +
				", edge step " + std::to_string(r.edgeOff) + " -> " + std::to_string(r.edgeOn) + ", land step " + std::to_string(r.stepOff) + " -> " +
				std::to_string(r.stepOn) + ", steam and pools " + (r.restSame ? "identical" : "CHANGED"));

		// Parts of the cap no scene above runs (review 2026-11-15): the max(.., m) term (flat marsh land beside a river,
		// ShoreFactor 1 so that the value is the cell's own blended multiplier: 0.7 + 1.3 x 9/16 = 1.431 against the
		// river's 1.1, and the cap must leave the whole grid as it is), the defaults (Params off = as before, Settings on since 0.3.0: FogRenderer copies Settings in), a
		// band of one cell (ShoreDistance = one cell: the only band cell is held to the river's value, the next cell is
		// untouched) and a hot spring as the water (its x is the lake's). The 20-degree profile is printed, not judged:
		// with the cap the land is at the river's value at the water and above it further in (the cap's stated limit).
		std::vector<Kind> marsh = river, spring(count, Kind::kLand);
		for (int y = 0; y < n; ++y) {
			for (int x = mid + 1; x <= mid + 20; ++x) marsh[at(x, y)] = Kind::kMarshLand;
			spring[at(mid, y)] = Kind::kHotSpring;
		}
		PlacementRules::Params noShore = pp, oneCell = pp;
		noShore.shoreFactor = 1.0f;
		oneCell.shoreDistance = cell;
		const Pair ms = run(flat, marsh, noShore), one = run(steep, river, oneCell), hs = run(steep, spring, pp);
		const std::size_t beside = at(mid + 1, mid), second = at(mid + 2, mid);
		const bool marshOk = close(ms.off[beside][0], 1.431f) && ms.same && ms.restSame;
		const bool defaultOk = !PlacementRules::Params{}.bankCap && Settings::placeBankCap;
		const bool oneOk = close(one.off[beside][0], 1.638f) && close(one.on[beside][0], 1.1f) &&
		                   std::memcmp(&one.on[second][0], &one.off[second][0], sizeof(float)) == 0 && one.raise <= 0.0f && one.restSame;
		const bool springOk = close(hs.off[beside][0], 1.638f) && close(hs.on[beside][0], 1.4f) && hs.raise <= 0.0f && hs.restSame;
		std::string profile;
		for (int dx = 1; dx <= 5; ++dx) profile += (dx > 1 ? " / " : "") + std::to_string(b.on[at(mid + dx, mid)][0]);
		Check(marshOk && defaultOk && oneOk && springOk, "bank cap: the cell's own multiplier, defaults (Params off, Settings on), one-cell band, hot spring",
			"marsh land beside a river, ShoreFactor 1: " + std::to_string(ms.off[beside][0]) + " -> " + std::to_string(ms.on[beside][0]) +
				(ms.same ? ", grid identical" : ", GRID CHANGED") + "; defaults: Params " + (PlacementRules::Params{}.bankCap ? "ON" : "off") + ", Settings " +
				(Settings::placeBankCap ? "ON" : "off") + "; band of one cell, 20 deg: " + std::to_string(one.off[beside][0]) + " -> " +
				std::to_string(one.on[beside][0]) + ", next cell " + std::to_string(one.off[second][0]) + " -> " + std::to_string(one.on[second][0]) +
				"; hot spring, 20 deg: " + std::to_string(hs.off[beside][0]) + " -> " + std::to_string(hs.on[beside][0]) +
				"; not judged: 20 deg with the cap, cells 1 to 5 from the river (1.1): " + profile);

		// The shore summary on the two-kind scenes: a bank cell the cap holds to the lake's value is not counted as above
		// the river on its other side, whichever side the river is on (SummarizeShore takes the larger of equally near
		// water, as Build does). Without the cap the hollow cell (1.638) is above both: one cell, excess 0.238, both ways.
		auto sum = [&](const std::vector<std::array<float, 4>>& a_o, const std::vector<Kind>& a_k) { return PlacementRules::SummarizeShore(a_o, hollow1, a_k, n, cell, pp); };
		const auto s1 = sum(t1.on, riverLake), s2 = sum(t2.on, lakeRiver), s1off = sum(t1.off, riverLake), s2off = sum(t2.off, lakeRiver);
		Check(s1.valid && s2.valid && s1.bankAbove == 0 && s2.bankAbove == 0 && s1.bankExcess == 0.0f && s2.bankExcess == 0.0f && s1off.bankAbove == 1 &&
				  s2off.bankAbove == 1 && close(s1off.bankExcess, 0.238f) && close(s2off.bankExcess, 0.238f),
			"shore summary: two water kinds (the larger of equally near water, either order)",
			"cap on: river | bank | lake " + std::to_string(s1.bankAbove) + " above, excess " + std::to_string(s1.bankExcess) + "; lake | bank | river " +
				std::to_string(s2.bankAbove) + " above, excess " + std::to_string(s2.bankExcess) + "; cap off: " + std::to_string(s1off.bankAbove) + " above, excess " +
				std::to_string(s1off.bankExcess) + "; " + std::to_string(s2off.bankAbove) + " above, excess " + std::to_string(s2off.bankExcess));
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

	XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(70.0f), 1.0f, 15.0f, 100000.0f);   // the sky check swaps in a reversed one
	XMFLOAT4X4     pf;
	XMStoreFloat4x4(&pf, proj);

	// Fog box height range used by render (the game spans the whole terrain height range) and the largest
	// per-pixel |GPU - CPU| of the last render.
	float boxLo = -1.0f, boxHi = maxH * 1.01f + 1.0f, pixelErr = 0.0f;
	// Optional tall layer for render(): a uniform H, W texture bound at t4 (nullptr = none, as an unbound slot).
	ID3D11ShaderResourceView* testUpper = nullptr;
	float                     testUpperH = 0.0f, testUpperW = 0.0f;
	// For the fade-circle checks: the point looked at, another march shader, column texture and step count (0 = as usual).
	XMFLOAT3                  testTarget{ 300.0f, 200.0f, 0.0f };
	ID3D11PixelShader*        testPs = nullptr;
	ID3D11ShaderResourceView* testCols = nullptr;
	int                       testSteps = 0;
	// Camera looks at the ground point (300, 200, 0) from a given pitch, distance and yaw.
	auto render = [&](float a_pitchDeg, float a_dist, float a_yawDeg, float a_sceneDistAtCentre, float& a_cpu) {
		const float pitch = XMConvertToRadians(a_pitchDeg), yaw = XMConvertToRadians(a_yawDeg);
		const XMFLOAT3 target = testTarget;
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
		rc.box[3] = static_cast<float>(testSteps > 0 ? testSteps : kSteps);
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
		ctx->PSSetShader(testPs ? testPs : ps, nullptr, 0);
		ctx->PSSetConstantBuffers(0, 1, &cb);
		ID3D11ShaderResourceView* srvs[2]{ depthSrv, testCols ? testCols : sim.ColumnSrv() };
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

	// The march is clipped to the round fade's outer circle (r = 0.95 of the half side: 1945.6 units here, the square's
	// side 2048, its corner 2896), where it used to run to the square's edge. Same scene, the march shader with and
	// without the clip (GROUNDFOG_NO_FADE_CLIP), rays the clip shortens: over the rim from inside (slanting down along an
	// axis and a diagonal, and level inside the layer), in over the rim from an eye outside the circle, and one that
	// runs inside the square but wholly past the circle.
	{
		const D3D_SHADER_MACRO noClip[]{ { "GROUNDFOG_NO_FADE_CLIP", "1" }, { nullptr, nullptr } };
		ID3DBlob*              nb = Compile(RaymarchShaders::kMarchPS, "ps_5_0", noClip);
		ID3D11PixelShader*     psNoClip = nullptr;
		if (nb) dev->CreatePixelShader(nb->GetBufferPointer(), nb->GetBufferSize(), nullptr, &psNoClip);
		SafeRelease(nb);
		struct ClipRay
		{
			XMFLOAT3    cam, target;
			const char* name;
		};
		// Runs one ray (the 2x2 centre pixels' mean, as render): a_ps null = the plugin's march; a_spans = the march's
		// stretch without and with the clip (CPU, the centre ray).
		auto renderRay = [&](const ClipRay& a_r, ID3D11PixelShader* a_ps, int a_steps, float& a_cpu, double* a_spans = nullptr) {
			const double dx = a_r.target.x - a_r.cam.x, dy = a_r.target.y - a_r.cam.y, dz = a_r.target.z - a_r.cam.z;
			const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
			if (a_spans) {
				const double eye[3]{ a_r.cam.x, a_r.cam.y, a_r.cam.z }, dir[3]{ dx / dist, dy / dist, dz / dist };
				a_spans[0] = MarchSpan(eye, dir, dist, kOrigin, kExtent, boxLo, boxHi, false);
				a_spans[1] = MarchSpan(eye, dir, dist, kOrigin, kExtent, boxLo, boxHi, true);
			}
			testTarget = a_r.target;
			testPs = a_ps;
			testSteps = a_steps;
			const float g = render(XMConvertToDegrees(static_cast<float>(std::asin(-dz / dist))), static_cast<float>(dist),
				XMConvertToDegrees(static_cast<float>(std::atan2(dx, dy))), -1.0f, a_cpu);
			testTarget = { 300.0f, 200.0f, 0.0f };
			testPs = nullptr;
			testSteps = 0;
			return g;
		};
		const ClipRay rays[] = {
			{ { 0.0f, 1500.0f, 60.0f }, { 0.0f, 2200.0f, 0.0f }, "down over the rim" },
			{ { 1000.0f, 1000.0f, 60.0f }, { 1600.0f, 1600.0f, 0.0f }, "down over the rim, diagonal" },
			{ { 0.0f, 1740.0f, 10.0f }, { 0.0f, 2600.0f, 10.0f }, "level over the rim" },
			{ { 0.0f, -2600.0f, 60.0f }, { 0.0f, -1700.0f, 0.0f }, "in from outside" },
			{ { -1900.0f, -1900.0f, 60.0f }, { -1150.0f, -1150.0f, 0.0f }, "in from outside, diagonal" },
			{ { -500.0f, 2000.0f, 10.0f }, { 500.0f, 2000.0f, 10.0f }, "along the square's edge, past the circle" },
		};
		constexpr int kFine = 512;
		// Tolerances. Same fog: with 512 steps both marches have converged, so what is left is what the clip cuts off: the
		// trace the sampler's linear filter leaves up to one texel beyond the circle (the texels' own W is 0 from r = 0.95).
		// Here a texel is 64 units (r 0.031): that trace is up to 3 % of the layer in a band 90 units wide; in game (384
		// texels of 32 units over 12288) under 0.1 %. Measured here: 0.0005 in alpha at most; tolerance 0.002. At 24 steps
		// the clipped march must stay within the tolerance the unclipped one has had against the fine CPU integral
		// (0.03 + 0.08 x fog; measured 0.002 clipped, 0.013 unclipped: the same steps over a shorter stretch).
		double      worstSame = 0.0, worst24 = 0.0, worst24Old = 0.0, shortened = 1.0;
		float       maxFog = 0.0f;
		bool        ok24 = psNoClip != nullptr;
		std::string detail;
		for (const auto& r : rays) {
			float        cpu = 0.0f, unused = 0.0f;
			double       spans[2]{};
			const float  clipFine = renderRay(r, nullptr, kFine, cpu, spans), openFine = renderRay(r, psNoClip, kFine, unused);
			const float  clip24 = renderRay(r, nullptr, 0, unused), open24 = renderRay(r, psNoClip, 0, unused);
			worstSame = std::max(worstSame, std::isfinite(clipFine) && std::isfinite(openFine) ? std::fabs(clipFine - openFine) : 1.0);
			worst24 = std::max(worst24, std::isfinite(clip24) ? std::fabs(clip24 - cpu) : 1.0);
			worst24Old = std::max(worst24Old, static_cast<double>(std::fabs(open24 - cpu)));
			ok24 = ok24 && std::isfinite(clip24) && std::fabs(clip24 - cpu) < 0.03f + 0.08f * cpu;
			if (spans[0] > 0.0) shortened = std::min(shortened, spans[1] / spans[0]);
			maxFog = std::max(maxFog, cpu);
			detail += std::string("; ") + r.name + ": stretch " + std::to_string(static_cast<int>(spans[0])) + " -> " + std::to_string(static_cast<int>(spans[1])) +
			          " units, 512 steps clipped/unclipped " + std::to_string(clipFine) + "/" + std::to_string(openFine) + ", 24 steps " + std::to_string(clip24) + "/" +
			          std::to_string(open24) + ", CPU " + std::to_string(cpu);
		}
		Check(ok24 && worstSame < 0.002 && shortened < 0.7 && maxFog > 0.2f, "march clipped to the fade circle gives the same fog",
			"worst |clipped - unclipped| at 512 steps " + std::to_string(worstSame) + " (tolerance 0.002); at 24 steps worst |GPU - CPU| clipped " +
				std::to_string(worst24) + ", unclipped " + std::to_string(worst24Old) + " (tolerance 0.03 + 0.08 x fog)" + detail);

		// A ray that misses the circle is not marched: the column texture replaced by one full of fog out to the square's
		// corners (H 98, W 1 everywhere), a level ray across the square's corner 2757 units from the centre at its nearest
		// (277 units of it inside the square). Unclipped it gathers that fog; clipped it must return exactly 0, i.e. no
		// step was taken (any step would have read fog).
		{
			std::vector<float> full(static_cast<std::size_t>(kRes) * kRes * 4);
			for (std::size_t i = 0; i < full.size(); i += 4) {
				full[i] = 98.0f;
				full[i + 1] = 1.0f;
				full[i + 2] = 0.0f;
				full[i + 3] = 1.0f;
			}
			D3D11_TEXTURE2D_DESC fd{};
			fd.Width = fd.Height = kRes;
			fd.MipLevels = fd.ArraySize = 1;
			fd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			fd.SampleDesc.Count = 1;
			fd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			D3D11_SUBRESOURCE_DATA    finit{ full.data(), kRes * sizeof(float) * 4, 0 };
			ID3D11Texture2D*          fullTex = nullptr;
			ID3D11ShaderResourceView* fullSrv = nullptr;
			dev->CreateTexture2D(&fd, &finit, &fullTex);
			dev->CreateShaderResourceView(fullTex, nullptr, &fullSrv);
			const ClipRay corner{ { 2400.0f, 1500.0f, 30.0f }, { 1500.0f, 2400.0f, 30.0f }, "across the corner" };
			const ClipRay inside{ { 0.0f, 0.0f, 30.0f }, { 0.0f, 600.0f, 30.0f }, "inside" };
			float         unused = 0.0f;
			double        spans[2]{};
			testCols = fullSrv;
			const float missClip = renderRay(corner, nullptr, 0, unused, spans), missOpen = renderRay(corner, psNoClip, 0, unused);
			const float hitClip = renderRay(inside, nullptr, 0, unused);   // the full texture is read at all
			testCols = nullptr;
			Check(fullSrv && psNoClip && missClip == 0.0f && missOpen > 0.05f && hitClip > 0.05f && spans[0] > 200.0 && spans[1] == 0.0,
				"ray that misses the fade circle is not marched",
				"fog-filled square, ray across its corner: stretch " + std::to_string(spans[0]) + " -> " + std::to_string(spans[1]) + " units, fog clipped " +
					std::to_string(missClip) + ", unclipped " + std::to_string(missOpen) + "; a ray inside the circle " + std::to_string(hitClip));
			SafeRelease(fullSrv);
			SafeRelease(fullTex);
		}
		SafeRelease(psNoClip);

		// Step length with RaymarchSteps 24 for level rays (heights open, nothing in the way), every degree of azimuth:
		// WorldSize 8192 clipped to its square (before) against 12288 clipped to its fade circle (now). From the field's
		// centre, and from 320 units off it (the grid trails the player by up to RecenterDistance 256 + one 64-unit cell).
		{
			auto stepRange = [](double a_field, bool a_circle, double a_off, double& a_min, double& a_mean, double& a_max) {
				a_min = 1e30;
				a_max = a_mean = 0.0;
				for (int deg = 0; deg < 360; ++deg) {
					const double a = deg * 3.14159265358979 / 180.0;
					const double eye[3]{ a_off, 0.0, 60.0 }, dir[3]{ std::sin(a), std::cos(a), 0.0 };
					const double len = MarchSpan(eye, dir, 1e9, -0.5 * a_field, a_field, -1000.0, 1000.0, a_circle) / kSteps;
					a_min = std::min(a_min, len);
					a_max = std::max(a_max, len);
					a_mean += len / 360.0;
				}
			};
			double oldMin, oldMean, oldMax, newMin, newMean, newMax, oldOffMax, newOffMax, d0, d1;
			stepRange(8192.0, false, 0.0, oldMin, oldMean, oldMax);
			stepRange(12288.0, true, 0.0, newMin, newMean, newMax);
			stepRange(8192.0, false, 320.0, d0, d1, oldOffMax);
			stepRange(12288.0, true, 320.0, d0, d1, newOffMax);
			auto f1 = [](double a_v) { return std::to_string(a_v).substr(0, std::to_string(a_v).find('.') + 2); };
			// Not "no longer than before": the circle's radius 0.475 x 12288 = 5836.8 is 0.76 % more than the old square's
			// corner 4096 x sqrt 2 = 5792.6, and off the centre along an axis the old longest ray (to the far corner) grew
			// less than the circle's does. Held to within 1 % from the centre and 3 % from 320 units off it; the mean over
			// the directions is longer by a quarter (the square's sides were nearer than its corners).
			Check(newMax <= oldMax * 1.01 && newOffMax <= oldOffMax * 1.03 && std::fabs(newMax - 0.475 * 12288.0 / kSteps) < 0.01,
				"march step at 12288 (fade circle) against the longest at 8192 (square)",
				"24 steps, level rays from the centre: 8192 square " + f1(oldMin) + ".." + f1(oldMax) + " units a step (mean " + f1(oldMean) + "); 12288 circle " +
					f1(newMin) + ".." + f1(newMax) + " (mean " + f1(newMean) + "): longest +" + f1(100.0 * (newMax / oldMax - 1.0)) + " %, mean +" +
					f1(100.0 * (newMean / oldMean - 1.0)) + " %; 320 units off the centre: longest " + f1(oldOffMax) + " -> " + f1(newOffMax) + " (+" +
					f1(100.0 * (newOffMax / oldOffMax - 1.0)) + " %); 12288 without the clip would be " + f1(6144.0 / kSteps) + ".." + f1(6144.0 * 1.41421356 / kSteps));
		}
	}

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
		float landGCb[4]{};       // RayCB landGrid (far layer's land table; zeros = none)
		ID3D11ShaderResourceView* testLand = nullptr;     // t6 for the probe (far layer's land table)
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
			std::memcpy(rc.landGrid, landGCb, sizeof(landGCb));
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
					ID3D11ShaderResourceView* in[7]{ depthSrv, sim.ColumnSrv(), lowSrv, sceneSrv, nullptr, testCoarse, testLand };
					ID3D11Buffer*             cbs[2]{ cb, probeCb };
					ctx->CSSetShader(probeCs, nullptr, 0);
					ctx->CSSetConstantBuffers(0, 2, cbs);
					ctx->CSSetShaderResources(0, 7, in);
					ctx->CSSetSamplers(0, 1, &linear);
					ctx->CSSetUnorderedAccessViews(0, 1, &probeUav, nullptr);
					ctx->Dispatch(1, 1, 1);
					ID3D11ShaderResourceView*  noSrv[7]{};
					ID3D11UnorderedAccessView* noUav = nullptr;
					ctx->CSSetShaderResources(0, 7, noSrv);
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

				// Far layer through the composite and the probe, with its land table at t6 (its numbers along exact rays are
				// checked by RunFarRayChecks): wind field off, no noise. The sky carries none with either depth direction, flat
				// ground just under the horizon keeps its fog. The far pools' checks below share the helpers.
				{
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
					// Probe records with the far layer on: strength, ground z without a table, thickness; the field off.
					auto runFar = [&](float a_strength, float a_level, float a_H) {
						farCb[0] = a_strength;
						farCb[1] = a_level;
						farCb[2] = a_H;
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
					// A flat land table (z = 0) of the plugin's size (193 vertices, 128 apart) centred on (a_x, a_y), bound at t6.
					auto bindFlatTable = [&](float a_x, float a_y) {
						const double half = 128.0 * 96.0;
						SafeRelease(testLand);
						testLand = MakeRgTexture(dev, 193, LandRg(193, a_x - half, a_y - half, 128.0, [](double, double) { return 0.0; }));
						landGCb[0] = static_cast<float>(a_x - half);
						landGCb[1] = static_cast<float>(a_y - half);
						landGCb[2] = 128.0f;
						landGCb[3] = 193.0f;
					};
					auto unbindTable = [&]() {
						SafeRelease(testLand);
						std::fill(std::begin(landGCb), std::end(landGCb), 0.0f);
					};
					auto fRec = [&](const std::vector<float>& a_r, int a_point, int a_slot) {
						return &a_r[static_cast<std::size_t>((a_point * RaymarchShaders::kProbeSlots + a_slot) * 4)];
					};
					// Sky (depth at the far plane: 1, or 0 with reversed depth) gets no far layer. The eye stands inside the layer
					// (z 60, thickness 126) over a known flat table, so a sky ray that ran FarLayerTau would collect fog right in
					// front of it. The ground below the horizon carries fog in both.
					{
						std::string detail;
						bool        allOk = true;
						bindFlatTable(300.0f, -900.0f);
						for (int reversed = 0; reversed < 2; ++reversed) {
							proj = reversed ? XMMatrixPerspectiveFovLH(XMConvertToRadians(70.0f), 1.0f, 100000.0f, 15.0f)
							                : XMMatrixPerspectiveFovLH(XMConvertToRadians(70.0f), 1.0f, 15.0f, 100000.0f);
							XMStoreFloat4x4(&pf, proj);
							const XMMATRIX vps = setCamera({ 300.0f, -900.0f, 60.0f }, { 300.0f, 2000.0f, 60.0f });
							fillDepth(0.0f, 0.0f, 0.0f, -1.0f, vps);
							const float skyDepth = reversed ? 0.0f : 1.0f;
							if (reversed) {
								for (auto& d : depthData) d = d == 1.0f ? 0.0f : d;
							}
							const auto fr = runFar(0.2f, 0.0f, 126.0f);
							int        sky = 0, skyFog = 0, skyFlag = 0, ground = 0, groundFog = 0;
							float      maxSky = 0.0f, minGround = 1.0f;
							for (int i = 0; i < kPoints; ++i) {
								const float* r0 = fRec(fr, i, 0);
								const int    px = static_cast<int>(r0[0]), py = static_cast<int>(r0[1]);
								const float  a = fRec(fr, i, 5)[3];
								if (depthData[static_cast<std::size_t>(py) * kSize + px] == skyDepth) {
									++sky;
									maxSky = std::max(maxSky, a);
									skyFog += a > 0.0f ? 1 : 0;
									skyFlag += fRec(fr, i, 8)[0] > 0.5f && fRec(fr, i, 8)[1] == 0.0f ? 1 : 0;
								} else {
									++ground;
									minGround = std::min(minGround, a);
									groundFog += a > 0.05f ? 1 : 0;
								}
							}
							const bool ok = sky >= 9 && skyFog == 0 && skyFlag == sky && ground >= 9 && groundFog == ground;
							allOk = allOk && ok;
							detail += std::string(reversed ? "reversed depth: " : "normal depth: ") + std::to_string(sky) + " sky points (alpha up to " +
							          std::to_string(maxSky) + ", " + std::to_string(skyFlag) + " flagged sky by the probe), " + std::to_string(ground) +
							          " ground points (alpha from " + std::to_string(minGround) + ")" + (reversed ? "" : "; ");
						}
						proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(70.0f), 1.0f, 15.0f, 100000.0f);
						XMStoreFloat4x4(&pf, proj);
						unbindTable();
						Check(allOk, "far layer gives the sky none (normal and reversed depth)", detail);
						setCamera({ 300.0f, 200.0f - 283.0f, 283.0f }, { 300.0f, 200.0f, 0.0f });
					}

					// Near the horizon on flat ground (Codex review 2026-11-10: the old layer lost row 61, 5 px under the horizon):
					// every ground point's composite alpha against the far layer's CPU integral (thinned with distance), and row 61
					// must carry fog.
					{
						const XMMATRIX vph = setCamera({ 300.0f, -900.0f, 100.0f }, { 300.0f, 100.0f, 224.0f });
						fillDepth(0.0f, 0.0f, 0.0f, -1.0f, vph);
						bindFlatTable(300.0f, -900.0f);
						const auto fr = runFar(0.05f, 0.0f, 30.0f);   // low strength: alpha mid-range, so a wrong value shows
						unbindTable();
						float worst = 0.0f, minRow = 1.0f, maxDist = 0.0f;
						int   n = 0, nRow = 0;
						for (int i = 0; i < kPoints; ++i) {
							const float* r0 = fRec(fr, i, 0);
							const float* d = fRec(fr, i, 1);
							if (depthData[static_cast<std::size_t>(r0[1]) * kSize + static_cast<std::size_t>(r0[0])] == 1.0f) continue;
							const FarRay ray{ { cam.x, cam.y, cam.z }, r0[3], { d[0], d[1], d[2] }, 0.0f };
							const float  expect = static_cast<float>(LayerAlphaCpu(ray, kOpacity, 0.05, 30.0, [](double, double) { return 0.0; }, [](double, double) { return 1.0; }));
							const float  got = fRec(fr, i, 5)[3];
							worst = std::max(worst, std::fabs(got - expect));
							maxDist = std::max(maxDist, r0[3]);
							++n;
							if (static_cast<int>(r0[1]) == 61) {
								minRow = std::min(minRow, got);
								++nRow;
							}
						}
						Check(n >= 20 && nRow >= 5 && minRow > 0.05f && worst < 0.01f, "far layer keeps its fog just under the horizon",
							"worst |GPU - CPU| " + std::to_string(worst) + " over " + std::to_string(n) + " ground points (out to " + std::to_string(maxDist) +
								" units), row 61 alpha from " + std::to_string(minRow) + " (" + std::to_string(nRow) + " points)");
						setCamera({ 300.0f, 200.0f - 283.0f, 283.0f }, { 300.0f, 200.0f, 0.0f });
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

	RunFarRayChecks(dev, ctx);
	RunSurfaceChecks(dev, ctx, vs, ps, comp, linear);
	RunWakeChecks(dev, ctx, vs, ps, comp, linear);
	g_pass = " (FogSampling 1)";
	g_fogSampling = 1;
	std::printf("INFO ---- the wake checks again with FogSampling 1 (INFO lines below are that pass's) ----\n");
	RunWakeChecks(dev, ctx, vs, ps, comp, linear);
	g_pass = "";
	g_fogSampling = 0;

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
