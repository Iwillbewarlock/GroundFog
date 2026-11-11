// Fog columns for a terrain with water and marsh, through the plugin's own placement path, for a Blender look render.
// Usage: PlacementExport <scene.bin> <output folder> [evaporation steamHeight steamDensity poolDensity wisp coarse]
//   wisp = LookP.x (fine structure); coarse = 1 builds coarse pools from scene.bin's optional coarse section
//   (int32 size, float originX, float originY, float cell, size^2 heights) with CoarsePoolRadius 6144, as the plugin.
//   poolDensity is scaled by the radiation strength max(global - evaporation, 0) (<= 2), as FogRenderer does; the pool
//   lid fill is that strength clamped to 0..1. Writes upper.bin too (tall layer, RGBA: H, W, ground * known, known).
//   (SteamP; defaults 0 0 0 = steam only thickens the layer; the plugin uses FogConditions evaporation and the ini values)
// scene.bin (little endian): int32 heightSize, int32 windRes, float origin, float extent, float thickness,
//   float global (FogConditions global), float cold (steam), float wind, int32 usePlacement,
//   heightSize^2 float heights (units; water surface over water), heightSize^2 uint8 kinds (PlacementRules::Kind).
// Writes: columns.bin (windRes^2 RGBA float: H units, W metres, ground z * known units, known), place.bin
//   (heightSize^2 RGBA float, PlacementRules::Build output), columns.json (sizes for the Blender script).
// Same calls as the game: PlacementRules::Build with default Params, WindSim::Step then BuildColumns with the
// placement texture and PlaceCB (global, cold, wind) (FogRenderer.cpp UpdatePlacement / Draw).

#include "PlacementRules.h"
#include "WindSim.h"

#include <d3d11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
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

	template <class T>
	bool ReadValue(std::ifstream& a_f, T& a_v)
	{
		return static_cast<bool>(a_f.read(reinterpret_cast<char*>(&a_v), sizeof(T)));
	}

	bool WriteBin(const std::string& a_path, const void* a_data, std::size_t a_bytes)
	{
		std::ofstream f(a_path, std::ios::binary);
		f.write(static_cast<const char*>(a_data), static_cast<std::streamsize>(a_bytes));
		return static_cast<bool>(f);
	}
}

int main(int argc, char** argv)
{
	if (argc < 3) {
		std::fprintf(stderr, "usage: PlacementExport <scene.bin> <output folder>\n");
		return 2;
	}
	std::ifstream in(argv[1], std::ios::binary);
	std::int32_t  hs = 0, res = 0, usePlace = 0;
	float         origin = 0, extent = 0, thickness = 0, global = 1, cold = 0, wind = 1;
	if (!in || !ReadValue(in, hs) || !ReadValue(in, res) || !ReadValue(in, origin) || !ReadValue(in, extent) || !ReadValue(in, thickness) ||
		!ReadValue(in, global) || !ReadValue(in, cold) || !ReadValue(in, wind) || !ReadValue(in, usePlace)) {
		std::fprintf(stderr, "%s: cannot read the header\n", argv[1]);
		return 2;
	}
	if (hs < 3 || hs > 1025 || res < 16 || res > 1024 || !(extent > 0.0f)) {
		std::fprintf(stderr, "%s: bad sizes (heights %d, wind %d, extent %g)\n", argv[1], hs, res, extent);
		return 2;
	}
	const std::size_t           n = static_cast<std::size_t>(hs) * hs;
	std::vector<float>          heights(n);
	std::vector<std::uint8_t>   kindBytes(n);
	if (!in.read(reinterpret_cast<char*>(heights.data()), static_cast<std::streamsize>(n * sizeof(float))) ||
		!in.read(reinterpret_cast<char*>(kindBytes.data()), static_cast<std::streamsize>(n))) {
		std::fprintf(stderr, "%s: file shorter than %d x %d heights and kinds\n", argv[1], hs, hs);
		return 2;
	}
	const std::string out = argv[2];
	std::array<float, 4> steamParams{ 0.0f, 0.0f, 0.0f, 0.0f };
	for (int i = 0; i < 4 && 3 + i < argc; ++i) steamParams[static_cast<std::size_t>(i)] = std::strtof(argv[3 + i], nullptr);
	const float wispAmount = argc > 7 ? std::strtof(argv[7], nullptr) : 0.0f;
	const bool  useCoarse = argc > 8 && std::atoi(argv[8]) != 0;
	const float strength = std::max(global - steamParams[0], 0.0f);
	steamParams[3] *= std::min(strength, 2.0f);
	const float       cell = extent / static_cast<float>(hs - 1);

	std::vector<PlacementRules::Kind> kinds(n);
	for (std::size_t i = 0; i < n; ++i) kinds[i] = static_cast<PlacementRules::Kind>(std::min<int>(kindBytes[i], 6));
	PlacementRules::Params placeParams;
	placeParams.poolFill = std::clamp(strength, 0.0f, 1.0f);
	const auto place = PlacementRules::Build(heights, kinds, hs, cell, placeParams);
	// Optional coarse grid (wide-valley pools).
	std::int32_t cSize = 0;
	float        cOriginX = 0.0f, cOriginY = 0.0f, cCell = 0.0f;
	std::vector<float> cHeights;
	if (ReadValue(in, cSize) && cSize >= 3 && cSize <= 513 && ReadValue(in, cOriginX) && ReadValue(in, cOriginY) && ReadValue(in, cCell)) {
		cHeights.resize(static_cast<std::size_t>(cSize) * cSize);
		if (!in.read(reinterpret_cast<char*>(cHeights.data()), static_cast<std::streamsize>(cHeights.size() * sizeof(float)))) cHeights.clear();
	}
	std::vector<float> coarseRg;
	if (useCoarse && !cHeights.empty()) {
		PlacementRules::Params cp = placeParams;
		cp.poolRadius = 6144.0f;
		cp.shoreDistance = 0.0f;
		const auto cpools = PlacementRules::Build(cHeights, std::vector<PlacementRules::Kind>(cHeights.size(), PlacementRules::Kind::kLand), cSize, cCell, cp);
		coarseRg.resize(cHeights.size() * 2);
		for (std::size_t i = 0; i < cHeights.size(); ++i) {
			coarseRg[i * 2] = cHeights[i];
			coarseRg[i * 2 + 1] = cpools[i][3];
		}
	}
	WriteBin(out + "/place.bin", place.data(), place.size() * sizeof(place[0]));

	ID3D11Device*        dev = nullptr;
	ID3D11DeviceContext* ctx = nullptr;
	if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
		std::fprintf(stderr, "WARP device failed\n");
		return 1;
	}
	WindSim     sim;
	std::string error;
	if (!sim.Create(dev, res, error)) {
		std::fprintf(stderr, "WindSim: %s\n", error.c_str());
		return 1;
	}
	sim.Reset(ctx);

	auto texture = [&](DXGI_FORMAT a_fmt, const void* a_data, UINT a_pitch, ID3D11Texture2D*& a_tex, ID3D11ShaderResourceView*& a_srv) {
		D3D11_TEXTURE2D_DESC d{};
		d.Width = d.Height = static_cast<UINT>(hs);
		d.MipLevels = d.ArraySize = 1;
		d.Format = a_fmt;
		d.SampleDesc.Count = 1;
		d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA init{ a_data, a_pitch, 0 };
		return SUCCEEDED(dev->CreateTexture2D(&d, &init, &a_tex)) && SUCCEEDED(dev->CreateShaderResourceView(a_tex, nullptr, &a_srv));
	};
	ID3D11Texture2D*          hTex = nullptr;
	ID3D11ShaderResourceView* hSrv = nullptr;
	ID3D11Texture2D*          pTex = nullptr;
	ID3D11ShaderResourceView* pSrv = nullptr;
	if (!texture(DXGI_FORMAT_R32_FLOAT, heights.data(), hs * sizeof(float), hTex, hSrv) ||
		!texture(DXGI_FORMAT_R32G32B32A32_FLOAT, place.data(), hs * sizeof(float) * 4, pTex, pSrv)) {
		std::fprintf(stderr, "texture creation failed\n");
		return 1;
	}

	WindShaders::SimCB scb{};
	scb.grid[0] = origin;
	scb.grid[1] = origin;
	scb.grid[2] = extent / static_cast<float>(res);
	scb.grid[3] = static_cast<float>(res);
	scb.time[0] = 1.0f / 30.0f;
	scb.shift[3] = static_cast<float>(hs);
	scb.counts[2] = 1.2f;
	scb.counts[3] = wispAmount;   // the banks and gaps are part of the field's fog amount
	scb.heightGrid[0] = origin;
	scb.heightGrid[1] = origin;
	scb.heightGrid[2] = cell;
	scb.heightGrid[3] = 1.0f;
	sim.Step(ctx, scb, hSrv, 5);
	ID3D11Texture2D*          cTex = nullptr;
	ID3D11ShaderResourceView* cSrv = nullptr;
	if (!coarseRg.empty()) {
		D3D11_TEXTURE2D_DESC cd{};
		cd.Width = cd.Height = static_cast<UINT>(cSize);
		cd.MipLevels = cd.ArraySize = 1;
		cd.Format = DXGI_FORMAT_R32G32_FLOAT;
		cd.SampleDesc.Count = 1;
		cd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA ci{ coarseRg.data(), static_cast<UINT>(cSize * sizeof(float) * 2), 0 };
		dev->CreateTexture2D(&cd, &ci, &cTex);
		dev->CreateShaderResourceView(cTex, nullptr, &cSrv);
	}
	sim.BuildColumns(ctx, hSrv, thickness, 0.0f, 0.0f, 0.0f, usePlace ? pSrv : nullptr, { global, cold, wind, 0.0f }, steamParams,
		{ 0.0f, 0.0f, 0.0f, 0.0f }, { wispAmount, 0.0f, 0.0f, 0.0f }, cSrv, { cOriginX, cOriginY, cCell, static_cast<float>(cSize) },
		{ 250.0f, 560.0f, 0.0f, 0.0f });
	SafeRelease(cSrv);
	SafeRelease(cTex);
	const auto cols = Read(dev, ctx, sim.ColumnTexture(), res, res, 4);
	WriteBin(out + "/columns.bin", cols.data(), cols.size() * sizeof(float));
	const auto upper = Read(dev, ctx, sim.UpperTexture(), res, res, 2);
	std::vector<float> upperRgba(static_cast<std::size_t>(res) * res * 4);
	float maxUpperH = 0.0f;
	for (std::size_t i = 0; i < static_cast<std::size_t>(res) * res; ++i) {
		upperRgba[i * 4 + 0] = cols[i * 4 + 3] > 1e-3f ? upper[i * 2] / cols[i * 4 + 3] : 0.0f;   // H was stored x known
		upperRgba[i * 4 + 1] = upper[i * 2 + 1];
		upperRgba[i * 4 + 2] = cols[i * 4 + 2];
		upperRgba[i * 4 + 3] = cols[i * 4 + 3];
		maxUpperH = std::max(maxUpperH, upper[i * 2]);
	}
	WriteBin(out + "/upper.bin", upperRgba.data(), upperRgba.size() * sizeof(float));
	// Coarse grid (ground, pool depth) for the far pools (RaymarchShaders FarPoolTau): coarse.bin, size^2 RG float.
	if (!coarseRg.empty()) WriteBin(out + "/coarse.bin", coarseRg.data(), coarseRg.size() * sizeof(float));

	float maxW = 0.0f, maxH = 0.0f;
	for (int i = 0; i < res * res; ++i) {
		maxH = std::max(maxH, cols[i * 4]);
		maxW = std::max(maxW, cols[i * 4 + 1]);
	}
	std::ofstream(out + "/columns.json") << "{\n  \"res\": " << res << ",\n  \"origin\": " << origin << ",\n  \"extent\": " << extent
										 << ",\n  \"height_size\": " << hs << ",\n  \"global\": " << global << ",\n  \"cold\": " << cold
										 << ",\n  \"wind\": " << wind << ",\n  \"placement\": " << usePlace << ",\n  \"max_w\": " << maxW
										 << ",\n  \"max_h\": " << maxH << ",\n  \"coarse_size\": " << (coarseRg.empty() ? 0 : cSize)
										 << ",\n  \"coarse_origin_x\": " << cOriginX << ",\n  \"coarse_origin_y\": " << cOriginY
										 << ",\n  \"coarse_cell\": " << cCell << ",\n  \"pool_density\": " << steamParams[3]
										 << ",\n  \"pool_cap\": " << 560.0f << "\n}\n";
	std::printf("columns %dx%d, placement %s, max H %.1f units, max W %.3f, tall layer max H %.1f\n", res, res, usePlace ? "on" : "off", maxH, maxW, maxUpperH);

	SafeRelease(pSrv);
	SafeRelease(pTex);
	SafeRelease(hSrv);
	SafeRelease(hTex);
	sim.Release();
	SafeRelease(ctx);
	SafeRelease(dev);
	return 0;
}
