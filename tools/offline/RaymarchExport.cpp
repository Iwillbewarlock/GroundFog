// Exports the fog columns and GPU raymarch images for comparison with a Blender Cycles reference render.
// Usage: RaymarchExport <output folder>
// Writes: columns.bin (64x64 RGBA float: H units, W metres, ground z * known units, known), cameras.json,
//         gpu_<k>.bin (256x256 float: fog = 1 - transmittance) for each camera.
// Scene: flat ground at z = 0 units, field square [-2048, 2048] units, fog at rest (density 1), base thickness 98.

#include "RaymarchShaders.h"
#include "WindSim.h"

#include <DirectXMath.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#include <cmath>
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

	constexpr int   kRes = 64;
	float kExtent = 4096.0f;   // field side, units; optional second argument
	float kOrigin = -2048.0f;
	constexpr int   kHeightSize = 17;
	constexpr UINT  kSize = 256;

	struct Camera
	{
		float pitchDeg, dist, yawDeg;
	};
	constexpr Camera kCameras[] = { { 45.0f, 400.0f, 0.0f }, { 15.0f, 600.0f, 30.0f }, { 70.0f, 300.0f, 200.0f } };
	constexpr float  kTarget[3] = { 300.0f, 200.0f, 0.0f };
	constexpr float  kFovYDeg = 70.0f;

	std::vector<float> Read(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11Texture2D* a_tex, int a_w, int a_h, int a_ch)
	{
		D3D11_TEXTURE2D_DESC d{};
		a_tex->GetDesc(&d);
		d.Usage = D3D11_USAGE_STAGING;
		d.BindFlags = 0;
		d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ID3D11Texture2D*   st = nullptr;
		std::vector<float> out(static_cast<std::size_t>(a_w) * a_h * a_ch, 0.0f);
		if (FAILED(a_dev->CreateTexture2D(&d, nullptr, &st))) {
			return out;
		}
		a_ctx->CopyResource(st, a_tex);
		D3D11_MAPPED_SUBRESOURCE m{};
		if (SUCCEEDED(a_ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
			for (int y = 0; y < a_h; ++y) {
				std::memcpy(&out[static_cast<std::size_t>(y) * a_w * a_ch], static_cast<const std::uint8_t*>(m.pData) + y * m.RowPitch, sizeof(float) * a_w * a_ch);
			}
			a_ctx->Unmap(st, 0);
		}
		SafeRelease(st);
		return out;
	}

	bool WriteBin(const std::string& a_path, const std::vector<float>& a_data)
	{
		std::ofstream f(a_path, std::ios::binary);
		f.write(reinterpret_cast<const char*>(a_data.data()), static_cast<std::streamsize>(a_data.size() * sizeof(float)));
		return static_cast<bool>(f);
	}
}

int main(int argc, char** argv)
{
	using namespace DirectX;
	if (argc < 2) {
		std::fprintf(stderr, "usage: RaymarchExport <output folder> [field side in units, default 4096]\n");
		return 2;
	}
	const std::string outDir = argv[1];
	if (argc > 2) {
		const float side = std::strtof(argv[2], nullptr);
		if (!(side >= 512.0f && side <= 16384.0f)) {
			std::fprintf(stderr, "field side %s is outside 512..16384 units\n", argv[2]);
			return 2;
		}
		kExtent = side;
		kOrigin = -side / 2.0f;
	}
	ID3D11Device*        dev = nullptr;
	ID3D11DeviceContext* ctx = nullptr;
	if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
		std::fprintf(stderr, "WARP device failed\n");
		return 1;
	}

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
	sim.BuildColumns(ctx, hSrv, 98.0f, 0.0f, 0.0f, 0.0f);
	const auto cols = Read(dev, ctx, sim.ColumnTexture(), kRes, kRes, 4);
	WriteBin(outDir + "/columns.bin", cols);
	float maxH = 0.0f;
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
	ID3D11VertexShader* vs = nullptr;
	ID3D11PixelShader*  ps = nullptr;
	dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
	dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps);
	SafeRelease(vsb);
	SafeRelease(psb);

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
	std::vector<float>        depthData(kSize * kSize, 1.0f);
	D3D11_SUBRESOURCE_DATA    dinit{ depthData.data(), kSize * sizeof(float), 0 };
	ID3D11Texture2D*          depthTex = nullptr;
	ID3D11ShaderResourceView* depthSrv = nullptr;
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

	const XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(kFovYDeg), 1.0f, 15.0f, 100000.0f);
	XMFLOAT4X4     pf;
	XMStoreFloat4x4(&pf, proj);

	std::string json = "{\n  \"units_per_metre\": 70,\n  \"fov_y_deg\": 70,\n  \"size\": 256,\n  \"target\": [300, 200, 0],\n  \"columns\": {\"res\": 64, \"origin\": " + std::to_string(kOrigin) + ", \"extent\": " + std::to_string(kExtent) + "},\n  \"opacity_per_metre\": 1.4,\n  \"steepness\": 3,\n  \"note\": \"DirectX LookToLH: screen right is -X for a camera looking +Y with up +Z; image row 0 is the top\",\n  \"cameras\": [\n";
	for (int k = 0; k < static_cast<int>(std::size(kCameras)); ++k) {
		const auto&    c = kCameras[k];
		const float    pitch = XMConvertToRadians(c.pitchDeg), yaw = XMConvertToRadians(c.yawDeg);
		const XMFLOAT3 cam{ kTarget[0] - std::cos(pitch) * std::sin(yaw) * c.dist, kTarget[1] - std::cos(pitch) * std::cos(yaw) * c.dist,
			kTarget[2] + std::sin(pitch) * c.dist };
		const XMVECTOR fwd = XMVector3Normalize(XMVectorSet(kTarget[0] - cam.x, kTarget[1] - cam.y, kTarget[2] - cam.z, 0));
		const XMMATRIX vp = XMMatrixMultiply(XMMatrixLookToLH(XMVectorZero(), fwd, XMVectorSet(0, 0, 1, 0)), proj);
		XMFLOAT4X4     inv;
		XMStoreFloat4x4(&inv, XMMatrixInverse(nullptr, vp));

		// Depth of the real ground plane z = 0 per pixel (sky where the ray does not hit it).
		for (UINT y = 0; y < kSize; ++y) {
			for (UINT x = 0; x < kSize; ++x) {
				const XMVECTOR ndc = XMVectorSet((x + 0.5f) / kSize * 2 - 1, 1 - (y + 0.5f) / kSize * 2, 0.5f, 1);
				XMFLOAT4       w;
				XMStoreFloat4(&w, XMVector4Transform(ndc, XMLoadFloat4x4(&inv)));
				const XMFLOAT3 dir{ w.x / w.w, w.y / w.w, w.z / w.w };
				float          d = 1.0f;
				if (dir.z < -1e-6f) {
					const float t = -cam.z / dir.z;   // scale on dir to reach z = 0
					const XMVECTOR hit = XMVectorSet(dir.x * t, dir.y * t, dir.z * t, 1);
					XMFLOAT4       clip;
					XMStoreFloat4(&clip, XMVector4Transform(hit, vp));
					d = clip.z / clip.w;
				}
				depthData[y * kSize + x] = d;
			}
		}
		ctx->UpdateSubresource(depthTex, 0, nullptr, depthData.data(), kSize * sizeof(float), 0);

		RaymarchShaders::RayCB rc{};
		std::memcpy(rc.invViewProj, &inv, sizeof(rc.invViewProj));
		rc.field[0] = kOrigin;
		rc.field[1] = kOrigin;
		rc.field[2] = kExtent;
		rc.field[3] = 1.0f;
		rc.box[0] = -1.0f;
		rc.box[1] = maxH * 1.01f + 1.0f;
		rc.box[2] = 100000.0f;
		rc.box[3] = 64.0f;
		rc.fog[0] = 1.4f;
		rc.fog[1] = 3.0f;   // ProfileSteepness
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
		ctx->PSSetSamplers(0, 1, &linear);
		ctx->Draw(3, 0);
		ID3D11ShaderResourceView* none[2]{};
		ctx->PSSetShaderResources(0, 2, none);
		const auto         img = Read(dev, ctx, out, kSize, kSize, 4);
		std::vector<float> fog(kSize * kSize);
		for (UINT i = 0; i < kSize * kSize; ++i) fog[i] = img[i * 4];
		WriteBin(outDir + "/gpu_" + std::to_string(k) + ".bin", fog);

		char line[256];
		std::snprintf(line, sizeof(line), "    {\"pitch\": %.1f, \"dist\": %.1f, \"yaw\": %.1f, \"pos\": [%.4f, %.4f, %.4f]}%s\n", c.pitchDeg, c.dist,
			c.yawDeg, cam.x, cam.y, cam.z, k + 1 < static_cast<int>(std::size(kCameras)) ? "," : "");
		json += line;
	}
	json += "  ]\n}\n";
	std::ofstream(outDir + "/cameras.json") << json;
	std::printf("exported columns (max H %.1f) and %zu GPU images to %s\n", maxH, std::size(kCameras), outDir.c_str());

	SafeRelease(linear);
	SafeRelease(cb);
	SafeRelease(depthSrv);
	SafeRelease(depthTex);
	SafeRelease(rtv);
	SafeRelease(out);
	SafeRelease(ps);
	SafeRelease(vs);
	SafeRelease(hSrv);
	SafeRelease(hTex);
	sim.Release();
	SafeRelease(ctx);
	SafeRelease(dev);
	return 0;
}
