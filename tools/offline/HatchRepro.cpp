// Reproduction attempt for the "hatching / frost on far walls and roofs" report (arca.live b/tullius 188808534,
// 2026-11-11; DEBUG-LEDGER.md). Runs the real march and composite shaders on WARP over a street-level view of a
// wall with a sloped roof on flat ground and measures, per surface, how much the fog differs from pixel to pixel.
// Usage: HatchRepro <output folder> [near plane, default 15] [vertical field of view in degrees, default 60] [roof slope in degrees, default 45]
// The pixel's angular size sets how far apart neighbouring scene points lie: 768 px over 41 degrees is the pixel size of
// 1080p at a 90-degree horizontal field of view, 31 degrees that of 1440p.
// Prints one line per case and surface; writes fog_<case>.bin (kSize x kSize float, fog alpha) and labels_<dist>.bin
// (0 sky, 1 ground, 2 wall, 3 roof) for the cases named in kDump.
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
	constexpr float kExtent = 4096.0f;     // wind field side, units, centred on the camera
	constexpr float kOrigin = -2048.0f;
	constexpr int   kHeightSize = 17;
	constexpr UINT  kSize = 768;           // 60 degrees over 768 px: 0.078 degrees per pixel (1080p at 90 degrees: 0.055 vertical)
	float           kFovYDeg = 60.0f;      // second optional argument
	float           kNear = 15.0f;         // first optional argument (the game's fNearDistance default is 15)
	constexpr float kFar = 100000.0f;
	float           kRoofDeg = 45.0f;      // third optional argument: the roof's slope (the far layer fades out between 20 and 42 degrees: kFarSlopeFull, kFarSlopeNone)
	constexpr float kEyeZ = 120.0f;
	constexpr float kThickness = 126.0f;   // ini Thickness
	constexpr float kOpacity = 0.18f;      // ini Opacity (the user's value)
	constexpr float kSteepness = 3.0f;     // ini ProfileSteepness
	constexpr int   kSteps = 24;           // ini RaymarchSteps
	constexpr float kFlatLength = 4000.0f; // ini FlatLength
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
}

int main(int argc, char** argv)
{
	using namespace DirectX;
	if (argc < 2) {
		std::fprintf(stderr, "usage: HatchRepro <output folder>\n");
		return 2;
	}
	const std::string outDir = argv[1];
	if (argc > 2) kNear = std::strtof(argv[2], nullptr);
	if (argc > 3) kFovYDeg = std::strtof(argv[3], nullptr);
	if (argc > 4) kRoofDeg = std::strtof(argv[4], nullptr);
	if (!(kRoofDeg >= 5.0f && kRoofDeg <= 85.0f)) {
		std::fprintf(stderr, "roof slope must be 5..85 degrees (got %s)\n", argv[4]);
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
	std::printf("field %.0f units, layer H up to %.1f, %u x %u px over %.0f degrees, near plane %.1f, roof %.0f degrees, march at half resolution, %d steps, %d frames\n", kExtent, maxH, kSize, kSize, kFovYDeg, kNear, kRoofDeg, kSteps, kFrames);
	std::printf("%-22s %-7s %8s %9s %12s %12s %10s\n", "case", "surface", "pixels", "mean fog", "noise 1 frm", "noise avg", "fogged %");

	for (const float dist : kDistances) {
		const float    wallH = 0.22f * dist, roofLen = 0.18f * dist;
		const XMFLOAT3 cam{ 0.0f, 0.0f, kEyeZ };
		const XMVECTOR fwd = XMVector3Normalize(XMVectorSet(0.0f, dist, wallH * 0.55f - kEyeZ, 0.0f));
		const XMMATRIX view = XMMatrixLookToLH(XMVectorZero(), fwd, XMVectorSet(0, 0, 1, 0));

		for (int rev = 0; rev < 2; ++rev) {
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
					if (d.z < -1e-9f) take(-cam.z / d.z, 1);
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
					const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
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

			for (int mode = 0; mode < 3; ++mode) {   // 0 field, 1 far, 2 both
				const char*        modeName = mode == 0 ? "field" : mode == 1 ? "far" : "both";
				std::vector<float> first, mean(static_cast<std::size_t>(kSize) * kSize, 0.0f);
				for (int frame = 0; frame < kFrames; ++frame) {
					RaymarchShaders::RayCB rc{};
					std::memcpy(rc.invViewProj, &inv, sizeof(rc.invViewProj));
					rc.field[0] = kOrigin;
					rc.field[1] = kOrigin;
					rc.field[2] = kExtent;
					rc.field[3] = mode == 1 ? 0.0f : 1.0f;
					rc.box[0] = -1.0f;
					rc.box[1] = kThickness + 2.0f;
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
					if (mode != 0) {
						rc.farFog[0] = 1.0f;
						rc.farFog[1] = kFlatLength;
						rc.farFog[2] = kThickness;
						rc.farFog[3] = 0.0f;   // no layer-height noise: any variation left comes from the plane estimate
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
				}
				const std::string name = std::to_string(static_cast<int>(dist)) + "_" + modeName + "_" + (rev ? "rev" : "z24");
				for (const char* dump : kDump) {
					if (name == dump) {
						WriteBin(outDir + "/fog_" + name + ".bin", first.data(), first.size() * sizeof(float));
						WriteBin(outDir + "/fogavg_" + name + ".bin", mean.data(), mean.size() * sizeof(float));
					}
				}
				const char* surfaces[] = { "sky", "ground", "wall", "roof" };
				for (std::uint8_t s = 1; s <= 3; ++s) {
					int    n1 = 0, n2 = 0, count = 0, fogged = 0;
					double sum = 0.0;
					for (std::size_t i = 0; i < first.size(); ++i) {
						if (label[i] != s) continue;
						++count;
						sum += first[i];
						fogged += first[i] > 0.004f ? 1 : 0;   // one step of an 8-bit colour channel
					}
					const double r1 = HighPassRms(first, label, s, n1);
					const double ra = HighPassRms(mean, label, s, n2);
					std::printf("%-22s %-7s %8d %9.4f %12.5f %12.5f %9.1f%%\n", name.c_str(), surfaces[s], count, count ? sum / count : 0.0, r1, ra,
						count ? 100.0 * fogged / count : 0.0);
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
	SafeRelease(hSrv);
	SafeRelease(hTex);
	sim.Release();
	SafeRelease(ctx);
	SafeRelease(dev);
	return 0;
}
