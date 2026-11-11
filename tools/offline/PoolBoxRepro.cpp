// Reproduction attempt for the "flat faces with straight edges across the sky" report (arca.live b/tullius 188814330,
// 2026-11-11; DEBUG-LEDGER.md 2026-11-12). The whole chain the game runs, offline on WARP: terrain heights -> pooled
// fog (PlacementRules::Build on the fine and on the coarse grid) -> fog columns (WindSim::BuildColumns) -> march at
// half resolution -> composite with the far layer and the far pools.
// Usage: PoolBoxRepro <output folder> [valley half width, default 3000] [pitch up in degrees, default 20] [yaw in degrees, default 0]
// Scene: a straight valley along Y with a flat floor (z = 0) and sides rising at kSideSlope; the camera stands on the
// floor at the grid centre and looks along the valley, pitched up. Depth is reversed float32 (as good as exact), so
// nothing here comes from depth precision.
// Cases: "all" = as in game, "nofar" = [FarFog] Pools = 0, "nocoarse" = no coarse pools at all, "nopool" = no pooled fog.
// Prints the fog alpha of the sky by elevation and writes fog_<case>.bin (kSize x kSize float) and sky.bin (1 = sky).

#include "PlacementRules.h"
#include "RaymarchShaders.h"
#include "WindSim.h"

#include <DirectXMath.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <algorithm>
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

	// The game's defaults (Settings.h, dist ini).
	constexpr int   kRes = 256;              // SimResolution
	constexpr float kExtent = 8192.0f;       // WorldSize
	constexpr float kOrigin = -4096.0f;
	constexpr int   kHeightSize = 129;       // HeightResolution: 64 units apart
	constexpr int   kCoarseSize = 65;        // Terrain::kCoarseSize
	constexpr float kCoarseCell = 512.0f;    // Terrain::kCoarseCell
	constexpr float kThickness = 126.0f;
	constexpr float kOpacity = 0.18f;
	constexpr float kSteepness = 3.0f;
	constexpr int   kSteps = 24;
	constexpr float kFlatLength = 4000.0f;
	constexpr float kPoolDensity = 0.1f, kPoolMaxDepth = 280.0f, kPoolMinVisibility = 83.0f, kPoolRadius = 1500.0f;
	constexpr float kCoarsePoolRadius = 6144.0f, kCoarseTolerance = 250.0f;

	constexpr UINT  kSize = 512;
	constexpr float kFovYDeg = 60.0f;
	constexpr float kNear = 15.0f, kFar = 100000.0f;
	constexpr float kEyeZ = 120.0f;
	constexpr float kSideSlope = 0.6f, kSideTop = 4000.0f;
	float           kHalfWidth = 3000.0f;

	float Ground(float a_x, float)
	{
		return std::min(kSideSlope * std::max(std::fabs(a_x) - kHalfWidth, 0.0f), kSideTop);
	}

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

	bool WriteBin(const std::string& a_path, const void* a_data, std::size_t a_bytes)
	{
		std::ofstream f(a_path, std::ios::binary);
		f.write(static_cast<const char*>(a_data), static_cast<std::streamsize>(a_bytes));
		return static_cast<bool>(f);
	}

	ID3D11ShaderResourceView* MakeTexture(ID3D11Device* a_dev, DXGI_FORMAT a_format, UINT a_size, const void* a_data, UINT a_bytesPerTexel)
	{
		D3D11_TEXTURE2D_DESC d{};
		d.Width = d.Height = a_size;
		d.MipLevels = d.ArraySize = 1;
		d.Format = a_format;
		d.SampleDesc.Count = 1;
		d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA    init{ a_data, a_size * a_bytesPerTexel, 0 };
		ID3D11Texture2D*          tex = nullptr;
		ID3D11ShaderResourceView* srv = nullptr;
		if (FAILED(a_dev->CreateTexture2D(&d, &init, &tex)) || FAILED(a_dev->CreateShaderResourceView(tex, nullptr, &srv))) {
			std::fprintf(stderr, "texture creation failed (format %d, size %u)\n", static_cast<int>(a_format), a_size);
			std::exit(1);
		}
		tex->Release();   // the view keeps it
		return srv;
	}
}

int main(int argc, char** argv)
{
	using namespace DirectX;
	if (argc < 2) {
		std::fprintf(stderr, "usage: PoolBoxRepro <output folder> [valley half width] [pitch up, degrees] [yaw, degrees]\n");
		return 2;
	}
	const std::string outDir = argv[1];
	if (argc > 2) kHalfWidth = std::strtof(argv[2], nullptr);
	const float pitchDeg = argc > 3 ? std::strtof(argv[3], nullptr) : 20.0f;
	const float yawDeg = argc > 4 ? std::strtof(argv[4], nullptr) : 0.0f;
	if (!(kHalfWidth >= 100.0f && kHalfWidth <= 12000.0f) || !(pitchDeg >= -30.0f && pitchDeg <= 80.0f) || !(yawDeg >= -180.0f && yawDeg <= 180.0f)) {
		std::fprintf(stderr, "half width 100..12000, pitch -30..80, yaw -180..180\n");
		return 2;
	}
	ID3D11Device*        dev = nullptr;
	ID3D11DeviceContext* ctx = nullptr;
	if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
		std::fprintf(stderr, "WARP device failed\n");
		return 1;
	}

	// 1. Terrain: the fine height grid (64 units) and the coarse grid (512 units), both centred on the camera.
	std::vector<float> fine(static_cast<std::size_t>(kHeightSize) * kHeightSize);
	const float        fineCell = kExtent / (kHeightSize - 1);
	for (int y = 0; y < kHeightSize; ++y)
		for (int x = 0; x < kHeightSize; ++x) fine[static_cast<std::size_t>(y) * kHeightSize + x] = Ground(kOrigin + x * fineCell, kOrigin + y * fineCell);
	const float        coarseOrigin = -0.5f * kCoarseCell * (kCoarseSize - 1);
	std::vector<float> coarse(static_cast<std::size_t>(kCoarseSize) * kCoarseSize);
	for (int y = 0; y < kCoarseSize; ++y)
		for (int x = 0; x < kCoarseSize; ++x)
			coarse[static_cast<std::size_t>(y) * kCoarseSize + x] = Ground(coarseOrigin + x * kCoarseCell, coarseOrigin + y * kCoarseCell);

	// 2. Pooled fog, as FogRenderer's placement job: the fine grid with PoolRadius, the coarse one with CoarsePoolRadius.
	PlacementRules::Params pp{};
	pp.poolRadius = kPoolRadius;
	pp.poolMaxDepth = kPoolMaxDepth;
	PlacementRules::Params cp = pp;
	cp.poolRadius = kCoarsePoolRadius;
	cp.shoreDistance = 0.0f;
	const auto finePlace = PlacementRules::Build(fine, std::vector<PlacementRules::Kind>(fine.size(), PlacementRules::Kind::kLand), kHeightSize, fineCell, pp);
	const auto coarsePlace = PlacementRules::Build(coarse, std::vector<PlacementRules::Kind>(coarse.size(), PlacementRules::Kind::kLand), kCoarseSize, kCoarseCell, cp);
	std::vector<float> coarseRg(coarse.size() * 2);
	float              fineDepthMax = 0.0f, coarseDepthMax = 0.0f, coarseLidMax = -1.0e30f, coarseTop = -1.0e30f;
	for (const auto& p : finePlace) fineDepthMax = std::max(fineDepthMax, p[3]);
	for (std::size_t i = 0; i < coarse.size(); ++i) {
		coarseRg[i * 2] = coarse[i];
		coarseRg[i * 2 + 1] = coarsePlace[i][3];
		coarseDepthMax = std::max(coarseDepthMax, coarsePlace[i][3]);
		if (coarsePlace[i][3] > 0.0f) {
			coarseLidMax = std::max(coarseLidMax, coarse[i] + coarsePlace[i][3]);
			const float wx = coarseOrigin + static_cast<float>(i % kCoarseSize) * kCoarseCell, wy = coarseOrigin + static_cast<float>(i / kCoarseSize) * kCoarseCell;
			if (std::fabs(wx) <= -kOrigin + kCoarseCell && std::fabs(wy) <= -kOrigin + kCoarseCell) coarseTop = std::max(coarseTop, coarse[i] + coarsePlace[i][3]);
		}
	}
	const int cc = (kCoarseSize - 1) / 2;
	std::printf("valley half width %.0f, sides %.2f; pool depth: fine grid up to %.0f, coarse grid up to %.0f (at the centre %.0f), highest coarse lid %.0f\n",
		kHalfWidth, kSideSlope, fineDepthMax, coarseDepthMax, coarsePlace[static_cast<std::size_t>(cc) * kCoarseSize + cc][3], coarseLidMax);
	std::printf("coarse ground along x at the centre row (512 units apart):");
	for (int x = cc - 8; x <= cc + 8; ++x) std::printf(" %.0f", coarse[static_cast<std::size_t>(cc) * kCoarseSize + x]);
	std::printf("\ncoarse pool depth along x at the centre row:");
	for (int x = cc - 8; x <= cc + 8; ++x) std::printf(" %.0f", coarsePlace[static_cast<std::size_t>(cc) * kCoarseSize + x][3]);
	std::printf("\n");

	ID3D11ShaderResourceView* hSrv = MakeTexture(dev, DXGI_FORMAT_R32_FLOAT, kHeightSize, fine.data(), sizeof(float));
	ID3D11ShaderResourceView* pSrv = MakeTexture(dev, DXGI_FORMAT_R32G32B32A32_FLOAT, kHeightSize, finePlace.data(), sizeof(float) * 4);
	ID3D11ShaderResourceView* cSrv = MakeTexture(dev, DXGI_FORMAT_R32G32_FLOAT, kCoarseSize, coarseRg.data(), sizeof(float) * 2);

	// 3. Wind field at rest and the fog columns.
	WindSim     sim;
	std::string error;
	if (!sim.Create(dev, kRes, error)) {
		std::fprintf(stderr, "WindSim: %s\n", error.c_str());
		return 1;
	}
	sim.Reset(ctx);
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
	scb.heightGrid[2] = fineCell;
	scb.heightGrid[3] = 1.0f;
	sim.Step(ctx, scb, hSrv, 5);
	const float poolDensity = PlacementRules::LimitPoolDensity(kPoolDensity, kOpacity, RaymarchShaders::kUpperSteepnessCpp, kPoolMinVisibility);

	auto compile = [&](const char* a_body, const char* a_target) -> ID3DBlob* {
		std::string src = RaymarchShaders::kCommon;
		src += a_body;
		ID3DBlob* code = nullptr;
		ID3DBlob* errs = nullptr;
		if (FAILED(D3DCompile(src.data(), src.size(), "Ray", nullptr, nullptr, "main", a_target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errs))) {
			std::fprintf(stderr, "%s\n", errs ? static_cast<const char*>(errs->GetBufferPointer()) : "compile failed");
			std::exit(1);
		}
		SafeRelease(errs);
		return code;
	};
	ID3DBlob*           vsb = compile(RaymarchShaders::kVS, "vs_5_0");
	ID3DBlob*           psb = compile(RaymarchShaders::kMarchPS, "ps_5_0");
	ID3DBlob*           cpb = compile(RaymarchShaders::kCompositePS, "ps_5_0");
	ID3D11VertexShader* vs = nullptr;
	ID3D11PixelShader*  ps = nullptr;
	ID3D11PixelShader*  comp = nullptr;
	dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
	dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps);
	dev->CreatePixelShader(cpb->GetBufferPointer(), cpb->GetBufferSize(), nullptr, &comp);
	SafeRelease(vsb);
	SafeRelease(psb);
	SafeRelease(cpb);

	const UINT           lowW = kSize / 2;
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
	td.Width = td.Height = kSize;
	td.BindFlags = D3D11_BIND_RENDER_TARGET;
	ID3D11Texture2D*        sceneTex = nullptr;   // black scene, white fog: red = fog alpha after the blend
	ID3D11RenderTargetView* sceneRtv = nullptr;
	dev->CreateTexture2D(&td, nullptr, &sceneTex);
	dev->CreateRenderTargetView(sceneTex, nullptr, &sceneRtv);
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

	// 4. Camera and the scene depth (reversed float32): each pixel's ray walked over the height function.
	const XMFLOAT3 cam{ 0.0f, 0.0f, kEyeZ };
	const float    pitch = XMConvertToRadians(pitchDeg), yaw = XMConvertToRadians(yawDeg);
	const XMVECTOR fwd = XMVectorSet(std::cos(pitch) * std::sin(yaw), std::cos(pitch) * std::cos(yaw), std::sin(pitch), 0.0f);
	const XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(kFovYDeg), 1.0f, kFar, kNear);
	XMFLOAT4X4     pf;
	XMStoreFloat4x4(&pf, proj);
	const XMMATRIX vp = XMMatrixMultiply(XMMatrixLookToLH(XMVectorZero(), fwd, XMVectorSet(0, 0, 1, 0)), proj);
	const XMMATRIX invVp = XMMatrixInverse(nullptr, vp);
	XMFLOAT4X4     inv;
	XMStoreFloat4x4(&inv, invVp);
	std::vector<float>        depthData(static_cast<std::size_t>(kSize) * kSize, 0.0f);
	std::vector<std::uint8_t> sky(depthData.size(), 0);
	std::vector<float>        elev(depthData.size(), 0.0f);
	for (UINT y = 0; y < kSize; ++y) {
		for (UINT x = 0; x < kSize; ++x) {
			XMFLOAT4 q;
			XMStoreFloat4(&q, XMVector4Transform(XMVectorSet((x + 0.5f) / kSize * 2 - 1, 1 - (y + 0.5f) / kSize * 2, 0.5f, 1), invVp));
			XMFLOAT3 d;
			XMStoreFloat3(&d, XMVector3Normalize(XMVectorSet(q.x / q.w, q.y / q.w, q.z / q.w, 0.0f)));
			const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
			elev[i] = XMConvertToDegrees(std::asin(d.z));
			float hit = -1.0f;
			float prev = 0.0f;
			for (float t = 8.0f; t < 60000.0f; t += std::max(8.0f, t * 0.004f)) {
				if (cam.z + d.z * t < Ground(cam.x + d.x * t, cam.y + d.y * t)) {
					float a = prev, b = t;
					for (int k = 0; k < 24; ++k) {
						const float m = 0.5f * (a + b);
						(cam.z + d.z * m < Ground(cam.x + d.x * m, cam.y + d.y * m) ? b : a) = m;
					}
					hit = 0.5f * (a + b);
					break;
				}
				prev = t;
			}
			if (hit <= 0.0f) {
				sky[i] = 1;
				continue;
			}
			XMFLOAT4 b;
			XMStoreFloat4(&b, XMVector4Transform(XMVectorSet(d.x * hit, d.y * hit, d.z * hit, 1), vp));
			depthData[i] = b.z / b.w;
		}
	}
	ID3D11ShaderResourceView* depthSrv = MakeTexture(dev, DXGI_FORMAT_R32_FLOAT, kSize, depthData.data(), sizeof(float));
	WriteBin(outDir + "/sky.bin", sky.data(), sky.size());

	float lo = 1e30f, hi = -1e30f;
	for (const float h : fine) {
		lo = std::min(lo, h);
		hi = std::max(hi, h);
	}
	const float               black[4]{};
	ID3D11ShaderResourceView* none[6]{};
	struct Case
	{
		const char* name;
		bool        finePools, coarseField, farPools;
	};
	const Case cases[] = { { "all", true, true, true }, { "nofar", true, true, false }, { "nocoarse", true, false, false }, { "nopool", false, false, false } };
	std::printf("camera pitch %.0f, yaw %.0f; pool density %.4f; box top %.0f (highest ground in the field %.0f)\n", pitchDeg, yawDeg, poolDensity,
		std::max(hi + kThickness + 2.0f, coarseTop + 2.0f), hi);
	for (const Case& c : cases) {
		const float pd = c.finePools ? poolDensity : 0.0f;
		sim.BuildColumns(ctx, hSrv, kThickness, 0.0f, 0.0f, 0.0f, pSrv, { 1.0f, 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, pd }, { 0.0f, 0.0f, 0.0f, 0.0f },
			{ 0.0f, 0.0f, 0.0f, 0.0f }, c.coarseField ? cSrv : nullptr, { coarseOrigin, coarseOrigin, kCoarseCell, static_cast<float>(kCoarseSize) },
			{ kCoarseTolerance, 2.0f * kPoolMaxDepth, 0.0f, 0.0f });
		RaymarchShaders::RayCB rc{};
		std::memcpy(rc.invViewProj, &inv, sizeof(rc.invViewProj));
		rc.field[0] = kOrigin;
		rc.field[1] = kOrigin;
		rc.field[2] = kExtent;
		rc.field[3] = 1.0f;
		rc.box[0] = lo - 1.0f;
		rc.box[1] = hi + kThickness + 2.0f;
		if (c.coarseField && coarseTop > -1.0e29f) rc.box[1] = std::max(rc.box[1], coarseTop + 2.0f);
		rc.box[2] = kExtent;
		rc.box[3] = static_cast<float>(kSteps);
		rc.fog[0] = kOpacity;
		rc.fog[1] = kSteepness;
		rc.depth[0] = pf._33;
		rc.depth[1] = pf._43;
		rc.depth[2] = static_cast<float>(kSize);
		rc.depth[3] = static_cast<float>(kSize);
		rc.lowRes[0] = static_cast<float>(lowW);
		rc.lowRes[1] = static_cast<float>(lowW);
		rc.lowRes[2] = static_cast<float>(kSize) / lowW;
		for (int k = 0; k < 3; ++k) {
			rc.colorLow[k] = 1.0f;
			rc.colorHigh[k] = 1.0f;
		}
		rc.distFog[0] = 1.0e6f;
		rc.distFog[1] = 2.0e6f;
		rc.distFog[2] = 1.0f;
		rc.posAdjust[0] = cam.x;
		rc.posAdjust[1] = cam.y;
		rc.posAdjust[2] = cam.z;
		rc.farFog[0] = 1.0f;
		rc.farFog[1] = kFlatLength;
		rc.farFog[2] = kThickness;
		rc.farFog[3] = 0.0f;
		rc.poolGrid[0] = rc.poolGrid[1] = coarseOrigin;
		rc.poolGrid[2] = kCoarseCell;
		rc.poolGrid[3] = static_cast<float>(kCoarseSize);
		rc.poolFar[0] = c.farPools ? poolDensity : 0.0f;
		rc.poolFar[1] = 2.0f * kPoolMaxDepth;
		rc.poolFar[2] = coarseLidMax;
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
		ID3D11ShaderResourceView* marchIn[2]{ depthSrv, sim.ColumnSrv() };
		ID3D11ShaderResourceView* upper = sim.UpperSrv();
		ctx->PSSetShaderResources(0, 2, marchIn);
		ctx->PSSetShaderResources(4, 1, &upper);
		ctx->PSSetSamplers(0, 1, &linear);
		ctx->Draw(3, 0);
		ctx->PSSetShaderResources(0, 6, none);

		D3D11_VIEWPORT fullVp{ 0, 0, static_cast<float>(kSize), static_cast<float>(kSize), 0, 1 };
		ctx->ClearRenderTargetView(sceneRtv, black);
		ctx->RSSetViewports(1, &fullVp);
		ctx->OMSetRenderTargets(1, &sceneRtv, nullptr);
		ctx->OMSetBlendState(blend, black, 0xFFFFFFFF);
		ctx->PSSetShader(comp, nullptr, 0);
		ID3D11ShaderResourceView* compIn[3]{ depthSrv, sim.ColumnSrv(), lowSrv };
		ID3D11ShaderResourceView* compUpper[2]{ upper, c.farPools ? cSrv : nullptr };
		ctx->PSSetShaderResources(0, 3, compIn);
		ctx->PSSetShaderResources(4, 2, compUpper);
		ctx->Draw(3, 0);
		ctx->PSSetShaderResources(0, 6, none);
		ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);

		const auto         img = ReadFloats(dev, ctx, sceneTex, kSize, kSize, 4);
		std::vector<float> fog(static_cast<std::size_t>(kSize) * kSize);
		for (std::size_t i = 0; i < fog.size(); ++i) fog[i] = img[i * 4];
		WriteBin(outDir + "/fog_" + std::string(c.name) + ".bin", fog.data(), fog.size() * sizeof(float));

		// Sky fog by elevation (5-degree bins): mean, and the largest step between horizontally adjacent sky pixels.
		std::printf("%-9s sky fog by elevation:", c.name);
		for (int bin = 0; bin < 10; ++bin) {
			double sum = 0.0;
			int    n = 0;
			for (std::size_t i = 0; i < fog.size(); ++i) {
				if (sky[i] && elev[i] >= bin * 5.0f && elev[i] < (bin + 1) * 5.0f) {
					sum += fog[i];
					++n;
				}
			}
			if (n) std::printf(" %d-%d:%.3f", bin * 5, bin * 5 + 5, sum / n);
		}
		float maxStep = 0.0f;
		for (UINT y = 0; y < kSize; ++y)
			for (UINT x = 0; x + 1 < kSize; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
				if (sky[i] && sky[i + 1]) maxStep = std::max(maxStep, std::fabs(fog[i + 1] - fog[i]));
			}
		for (UINT y = 0; y + 1 < kSize; ++y)
			for (UINT x = 0; x < kSize; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
				if (sky[i] && sky[i + kSize]) maxStep = std::max(maxStep, std::fabs(fog[i + kSize] - fog[i]));
			}
		std::printf(" | largest step between neighbouring sky pixels %.3f\n", maxStep);
	}

	SafeRelease(blend);
	SafeRelease(linear);
	SafeRelease(cb);
	SafeRelease(depthSrv);
	SafeRelease(sceneRtv);
	SafeRelease(sceneTex);
	SafeRelease(lowSrv);
	SafeRelease(lowRtv);
	SafeRelease(lowTex);
	SafeRelease(comp);
	SafeRelease(ps);
	SafeRelease(vs);
	SafeRelease(cSrv);
	SafeRelease(pSrv);
	SafeRelease(hSrv);
	sim.Release();
	SafeRelease(ctx);
	SafeRelease(dev);
	return 0;
}
