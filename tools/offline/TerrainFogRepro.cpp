// The far layer on real Skyrim land (DEBUG-LEDGER.md 2026-11-13). Runs the real march and composite shaders on WARP
// over a height grid exported from the load order's LAND records (tools/offline/land_from_housecarl.py): the player
// stands on the land, the wind field and the far layer's land table are built as the plugin would build them, the
// depth buffer is the land's triangles (128-unit quads, 24-bit depth). Land only: no trees, rocks or buildings.
// Usage: TerrainFogRepro <output folder> <land.f32> <cols> <rows> <origin x> <origin y> <player x> <player y>
//                        <yaw degrees, 0 = north (+y), clockwise> <pitch degrees up> <eye above the ground> <old|off|new> [ideal 0|1]
//   old = FarP.y is the old shader's FlatLength (build against the 0.2.0 header), off = far layer off (0.2.1),
//   new = this header's far layer with its land table.
// Writes scene.rgb (kSize x kSize x 3 bytes: the land lit by a plain sun, the sky), label.u8 (0 sky, 1 land), dist.f32,
// fog_<variant>.u8 (fog alpha x 255: field + far layer, patches on) and, except for off, faronly_<variant>.f32 (the far
// layer alone with an even layer: what "ideal.f32" is compared with). With ideal = 1: ideal.f32, the same even layer
// lying on the true land everywhere, integrated on the CPU along every fourth pixel's ray (-1 elsewhere).

#include "GridMath.h"
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
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{
	using namespace DirectX;

	template <class T>
	void SafeRelease(T*& a_p)
	{
		if (a_p) {
			a_p->Release();
			a_p = nullptr;
		}
	}

	constexpr int   kRes = 64;              // wind field cells per side
	constexpr float kExtent = 8192.0f;      // ini WorldSize
	constexpr int   kHeightSize = 129;      // ini HeightResolution (64 units apart)
	constexpr int   kLandSize = 193;        // the far layer's land table: as fine as the land (128 units), 12288 units either side
	constexpr float kLandCell = 128.0f;
	constexpr int   kSlopeReach = 4;        // slope share: the land's slope over this many vertices either side (512 units)
	constexpr float kSlopeFull = 0.268f;    // tan 15 degrees: all of the layer up to here
	constexpr float kSlopeNone = 0.577f;    // tan 30 degrees: none from here
	constexpr float kGameCell = 4096.0f;
	constexpr int   kLoadedCells = 2;       // loaded land: the player's cell and this many around it (uGridsToLoad 5)
	constexpr UINT  kSize = 768;
	constexpr float kFovYDeg = 59.0f;       // 16:9 at a 90-degree horizontal view, drawn square
	constexpr float kNear = 15.0f;
	constexpr float kFilmFps = 24.0f;
	constexpr float kFilmWind = 90.0f;      // units a second the patches travel (HatchRepro's film)
	constexpr float kFar = 200000.0f;
	constexpr float kThickness = 126.0f;
	constexpr float kOpacity = 0.18f;
	constexpr float kSteepness = 3.0f;
	constexpr int   kSteps = 24;
	constexpr float kLandStep = 128.0f;
	constexpr float kUnitsPerMetre = 70.0f;

	struct Land
	{
		std::vector<float> h;
		int                cols = 0, rows = 0;
		float              ox = 0.0f, oy = 0.0f;

		[[nodiscard]] float At(int a_x, int a_y) const { return h[static_cast<std::size_t>(a_y) * cols + a_x]; }
		// Height on the land's triangles (each quad split along its (0,0)-(1,1) diagonal); false off the grid.
		[[nodiscard]] bool Height(double a_wx, double a_wy, double& a_z) const
		{
			const double u = (a_wx - ox) / kLandStep, v = (a_wy - oy) / kLandStep;
			if (!(u >= 0.0 && v >= 0.0 && u <= cols - 1 && v <= rows - 1)) return false;
			const int    i = std::min(static_cast<int>(u), cols - 2), j = std::min(static_cast<int>(v), rows - 2);
			const double fu = u - i, fv = v - j;
			const double h00 = At(i, j), h10 = At(i + 1, j), h01 = At(i, j + 1), h11 = At(i + 1, j + 1);
			a_z = fu >= fv ? h00 + (h10 - h00) * fu + (h11 - h10) * fv : h00 + (h11 - h01) * fu + (h01 - h00) * fv;
			return true;
		}
	};

	// Ray o + d t (unit d) against one triangle; t or -1.
	double HitTri(const double* o, const double* d, const double* a, const double* b, const double* c)
	{
		const double e1[3]{ b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3]{ c[0] - a[0], c[1] - a[1], c[2] - a[2] };
		const double p[3]{ d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0] };
		const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
		if (std::fabs(det) < 1e-12) return -1.0;
		const double s[3]{ o[0] - a[0], o[1] - a[1], o[2] - a[2] };
		const double u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) / det;
		if (u < -1e-9 || u > 1.0 + 1e-9) return -1.0;
		const double q[3]{ s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0] };
		const double v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
		if (v < -1e-9 || u + v > 1.0 + 1e-9) return -1.0;
		const double t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
		return t > 1e-6 ? t : -1.0;
	}

	// First hit of the ray with the land's triangles, walking the 128-unit quads it crosses; a_n = the triangle's
	// upward unit normal. -1 for none within a_max.
	double CastLand(const Land& a_land, const double* o, const double* d, double a_max, double* a_n)
	{
		double       u = (o[0] - a_land.ox) / kLandStep, v = (o[1] - a_land.oy) / kLandStep;
		const double du = d[0] / kLandStep, dv = d[1] / kLandStep;
		if (!(u >= 0.0 && v >= 0.0 && u < a_land.cols - 1 && v < a_land.rows - 1)) return -1.0;
		int          i = static_cast<int>(u), j = static_cast<int>(v);
		const int    si = du > 0.0 ? 1 : -1, sj = dv > 0.0 ? 1 : -1;
		const double tdu = std::fabs(du) > 1e-12 ? 1.0 / std::fabs(du) : 1e300, tdv = std::fabs(dv) > 1e-12 ? 1.0 / std::fabs(dv) : 1e300;
		double       tu = std::fabs(du) > 1e-12 ? ((du > 0.0 ? i + 1 - u : u - i)) * tdu : 1e300;
		double       tv = std::fabs(dv) > 1e-12 ? ((dv > 0.0 ? j + 1 - v : v - j)) * tdv : 1e300;
		double       t = 0.0;
		while (t < a_max && i >= 0 && j >= 0 && i < a_land.cols - 1 && j < a_land.rows - 1) {
			const double tExit = std::min(std::min(tu, tv), a_max);
			const double h00 = a_land.At(i, j), h10 = a_land.At(i + 1, j), h01 = a_land.At(i, j + 1), h11 = a_land.At(i + 1, j + 1);
			const double top = std::max(std::max(h00, h10), std::max(h01, h11));
			if (std::min(o[2] + d[2] * t, o[2] + d[2] * tExit) <= top + 1e-3) {
				const double x0 = a_land.ox + i * kLandStep, y0 = a_land.oy + j * kLandStep;
				const double p00[3]{ x0, y0, h00 }, p10[3]{ x0 + kLandStep, y0, h10 }, p01[3]{ x0, y0 + kLandStep, h01 }, p11[3]{ x0 + kLandStep, y0 + kLandStep, h11 };
				const double ta = HitTri(o, d, p00, p10, p11), tb = HitTri(o, d, p00, p11, p01);
				double       best = -1.0;
				const double *q0 = p00, *q1 = p10, *q2 = p11;
				if (ta > 0.0) best = ta;
				if (tb > 0.0 && (best < 0.0 || tb < best)) {
					best = tb;
					q1 = p11;
					q2 = p01;
				}
				if (best > 0.0 && best <= a_max) {
					const double e1[3]{ q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2] }, e2[3]{ q2[0] - q0[0], q2[1] - q0[1], q2[2] - q0[2] };
					double       n[3]{ e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
					const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
					const double sgn = n[2] < 0.0 ? -1.0 : 1.0;
					for (int k = 0; k < 3; ++k) a_n[k] = sgn * n[k] / std::max(len, 1e-12);
					return best;
				}
			}
			t = tExit;
			if (tu < tv) {
				tu += tdu;
				i += si;
			} else {
				tv += tdv;
				j += sj;
			}
		}
		return -1.0;
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

	// Set when an output file could not be written: main then exits 1 (review r5 finding 9: a missing folder used to give
	// no files and exit 0).
	bool g_writeFailed = false;

	bool WriteBin(const std::string& a_path, const void* a_data, std::size_t a_bytes)
	{
		std::ofstream f(a_path, std::ios::binary);
		f.write(static_cast<const char*>(a_data), static_cast<std::streamsize>(a_bytes));
		f.close();
		if (!f) {
			std::fprintf(stderr, "could not write %s\n", a_path.c_str());
			g_writeFailed = true;
			return false;
		}
		return true;
	}

	ID3D11ShaderResourceView* MakeTexture(ID3D11Device* a_dev, DXGI_FORMAT a_format, UINT a_size, const void* a_data, UINT a_texel)
	{
		D3D11_TEXTURE2D_DESC d{};
		d.Width = d.Height = a_size;
		d.MipLevels = d.ArraySize = 1;
		d.Format = a_format;
		d.SampleDesc.Count = 1;
		d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA    init{ a_data, a_size * a_texel, 0 };
		ID3D11Texture2D*          tex = nullptr;
		ID3D11ShaderResourceView* srv = nullptr;
		if (SUCCEEDED(a_dev->CreateTexture2D(&d, &init, &tex))) a_dev->CreateShaderResourceView(tex, nullptr, &srv);
		SafeRelease(tex);
		return srv;
	}

	// The profile's density per metre at height h above the ground for an even layer of thickness H (written from the
	// definition, not from the shader: mean density kOpacity over H, falling as exp(-a h / H)).
	double IdealDensity(double h, double H)
	{
		if (h < 0.0 || h >= H) return 0.0;
		return kOpacity * kSteepness * std::exp(-kSteepness * h / H) / (1.0 - std::exp(-static_cast<double>(kSteepness)));
	}
}

int main(int argc, char** argv)
{
	if (argc < 13) {
		std::fprintf(stderr, "usage: TerrainFogRepro <out> <land.f32> <cols> <rows> <origin x> <origin y> <player x> <player y> <yaw> <pitch> <eye above> <old|off|new> [ideal 0|1] [film frames]\n");
		return 2;
	}
	const std::string outDir = argv[1];
	{
		std::error_code ec;
		std::filesystem::create_directories(outDir, ec);
		if (ec || !std::filesystem::is_directory(outDir)) {
			std::fprintf(stderr, "could not create the output folder %s (%s)\n", outDir.c_str(), ec ? ec.message().c_str() : "not a folder");
			return 1;
		}
	}
	Land              land;
	land.cols = static_cast<int>(std::strtol(argv[3], nullptr, 10));
	land.rows = static_cast<int>(std::strtol(argv[4], nullptr, 10));
	land.ox = std::strtof(argv[5], nullptr);
	land.oy = std::strtof(argv[6], nullptr);
	const double playerX = std::strtod(argv[7], nullptr), playerY = std::strtod(argv[8], nullptr);
	const float  yaw = XMConvertToRadians(std::strtof(argv[9], nullptr)), pitch = XMConvertToRadians(std::strtof(argv[10], nullptr));
	const double eyeAbove = std::strtod(argv[11], nullptr);
	const int    variant = std::strcmp(argv[12], "old") == 0 ? 0 : std::strcmp(argv[12], "off") == 0 ? 1 : 2;
	const char*  variantName = variant == 0 ? "old" : variant == 1 ? "off" : "new";
	const bool   wantIdeal = argc > 13 && std::strtol(argv[13], nullptr, 10) != 0;
	// Film: this many frames of the view the player sees (pass 0), the patches travelling with a light breeze;
	// written to video_<variant>.u8 (frames x kSize x kSize bytes, fog alpha x 255).
	const int    filmFrames = argc > 14 ? std::clamp(static_cast<int>(std::strtol(argv[14], nullptr, 10)), 0, 2000) : 0;
	if (land.cols < 2 || land.rows < 2 || land.cols > 20000 || land.rows > 20000) {
		std::fprintf(stderr, "land size %d x %d is not usable\n", land.cols, land.rows);
		return 2;
	}
	land.h.resize(static_cast<std::size_t>(land.cols) * land.rows);
	{
		std::ifstream f(argv[2], std::ios::binary);
		f.read(reinterpret_cast<char*>(land.h.data()), static_cast<std::streamsize>(land.h.size() * sizeof(float)));
		if (!f || f.gcount() != static_cast<std::streamsize>(land.h.size() * sizeof(float))) {
			std::fprintf(stderr, "%s does not hold %d x %d floats\n", argv[2], land.cols, land.rows);
			return 2;
		}
	}
	double groundZ = 0.0;
	if (!land.Height(playerX, playerY, groundZ)) {
		std::fprintf(stderr, "the player (%.0f, %.0f) is off the land grid\n", playerX, playerY);
		return 2;
	}
	const double cam[3]{ playerX, playerY, groundZ + eyeAbove };

	ID3D11Device*        dev = nullptr;
	ID3D11DeviceContext* ctx = nullptr;
	if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
		std::fprintf(stderr, "WARP device failed\n");
		return 1;
	}

	// Fine height grid: the field's ground (64 units apart, centred on the player).
	const float        fieldOx = static_cast<float>(playerX) - 0.5f * kExtent, fieldOy = static_cast<float>(playerY) - 0.5f * kExtent;
	const float        fineCell = kExtent / (kHeightSize - 1);
	std::vector<float> fine(static_cast<std::size_t>(kHeightSize) * kHeightSize);
	float              lo = 1e30f, hi = -1e30f;
	for (int y = 0; y < kHeightSize; ++y) {
		for (int x = 0; x < kHeightSize; ++x) {
			double z = groundZ;
			if (!land.Height(fieldOx + x * fineCell, fieldOy + y * fineCell, z)) {
				std::fprintf(stderr, "the wind field reaches off the land grid: move the player at least %.0f units in from its edge\n", 0.5f * kExtent);
				return 2;
			}
			fine[static_cast<std::size_t>(y) * kHeightSize + x] = static_cast<float>(z);
			lo = std::min(lo, static_cast<float>(z));
			hi = std::max(hi, static_cast<float>(z));
		}
	}
	ID3D11ShaderResourceView* hSrv = MakeTexture(dev, DXGI_FORMAT_R32_FLOAT, kHeightSize, fine.data(), sizeof(float));

	// Land table for the far layer: ground z and the share of the layer the land carries by its slope; snapped to
	// whole vertices around the player; known only on the land the game has loaded (the player's 4096-unit cell and
	// kLoadedCells around it). The slope is taken over kSlopeReach vertices either side, so single facets do not show.
	const float        landHalf = kLandCell * (kLandSize - 1) / 2;
	const float        landOx = std::floor(static_cast<float>(playerX) / kLandCell + 0.5f) * kLandCell - landHalf;
	const float        landOy = std::floor(static_cast<float>(playerY) / kLandCell + 0.5f) * kLandCell - landHalf;
	const double       cellX = std::floor(playerX / kGameCell), cellY = std::floor(playerY / kGameCell);
	const double       knownX0 = (cellX - kLoadedCells) * kGameCell, knownX1 = (cellX + kLoadedCells + 1) * kGameCell;
	const double       knownY0 = (cellY - kLoadedCells) * kGameCell, knownY1 = (cellY + kLoadedCells + 1) * kGameCell;
	// Heights of the known vertices, then the plugin's own builder (GridMath::BuildLandTable) for ground + slope share.
	std::vector<float> landH(static_cast<std::size_t>(kLandSize) * kLandSize, -1.0e30f);
	std::vector<float> landRg(landH.size() * 2, 0.0f);
	int                known = 0, steep = 0;
	for (int y = 0; y < kLandSize; ++y) {
		for (int x = 0; x < kLandSize; ++x) {
			const double wx = landOx + x * kLandCell, wy = landOy + y * kLandCell;
			double       z = 0.0;
			if (wx >= knownX0 && wx <= knownX1 && wy >= knownY0 && wy <= knownY1 && land.Height(wx, wy, z)) {
				landH[static_cast<std::size_t>(y) * kLandSize + x] = static_cast<float>(z);
				++known;
			}
		}
	}
	GridMath::BuildLandTable(landH, kLandSize, kLandCell, -1.0e30f, landRg);
	for (std::size_t i = 0; i < landH.size(); ++i) steep += landH[i] > -1.0e29f && landRg[i * 2 + 1] < 0.5f ? 1 : 0;
	ID3D11ShaderResourceView* cSrv = MakeTexture(dev, DXGI_FORMAT_R32G32_FLOAT, kLandSize, landRg.data(), sizeof(float) * 2);
	std::printf("player (%.0f, %.0f) ground %.0f, eye +%.0f; field ground %.0f..%.0f; land table: %d of %d known, %d of them steep (share under a half)\n",
		playerX, playerY, groundZ, eyeAbove, lo, hi, known, kLandSize * kLandSize, steep);

	// Wind field at rest on this ground.
	WindSim     sim;
	std::string error;
	if (!sim.Create(dev, kRes, error)) {
		std::fprintf(stderr, "WindSim: %s\n", error.c_str());
		return 1;
	}
	sim.Reset(ctx);
	WindShaders::SimCB scb{};
	scb.grid[0] = fieldOx;
	scb.grid[1] = fieldOy;
	scb.grid[2] = kExtent / kRes;
	scb.grid[3] = kRes;
	scb.time[0] = 1.0f / 30.0f;
	scb.shift[3] = kHeightSize;
	scb.counts[2] = 1.2f;
	scb.heightGrid[0] = fieldOx;
	scb.heightGrid[1] = fieldOy;
	scb.heightGrid[2] = fineCell;
	scb.heightGrid[3] = 1.0f;
	sim.Step(ctx, scb, hSrv, 5);
	sim.BuildColumns(ctx, hSrv, kThickness, 0.0f, 0.0f, 0.0f);

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

	// Camera and depth.
	const XMVECTOR fwd = XMVector3Normalize(XMVectorSet(std::sin(yaw) * std::cos(pitch), std::cos(yaw) * std::cos(pitch), std::sin(pitch), 0.0f));
	const XMMATRIX view = XMMatrixLookToLH(XMVectorZero(), fwd, XMVectorSet(0, 0, 1, 0));
	const XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(kFovYDeg), 1.0f, kNear, kFar);
	XMFLOAT4X4     pf;
	XMStoreFloat4x4(&pf, proj);
	const XMMATRIX vp = XMMatrixMultiply(view, proj);
	const XMMATRIX invVp = XMMatrixInverse(nullptr, vp);
	XMFLOAT4X4     inv;
	XMStoreFloat4x4(&inv, invVp);

	const std::size_t         pixels = static_cast<std::size_t>(kSize) * kSize;
	std::vector<float>        depthData(pixels, 1.0f), distData(pixels, -1.0f), ideal(pixels, -1.0f);
	std::vector<std::uint8_t> label(pixels, 0), rgb(pixels * 3, 0);
	const double              sun[3]{ 0.45, 0.30, 0.84 };
	for (UINT y = 0; y < kSize; ++y) {
		for (UINT x = 0; x < kSize; ++x) {
			XMFLOAT4 q;
			XMStoreFloat4(&q, XMVector4Transform(XMVectorSet((x + 0.5f) / kSize * 2 - 1, 1 - (y + 0.5f) / kSize * 2, 0.5f, 1), invVp));
			double       d[3]{ q.x / q.w, q.y / q.w, q.z / q.w };
			const double dl = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
			for (double& c : d) c /= dl;
			double            n[3]{ 0.0, 0.0, 1.0 };
			const double      t = CastLand(land, cam, d, kFar * 0.95, n);
			const std::size_t i = static_cast<std::size_t>(y) * kSize + x;
			if (t <= 0.0) {
				const double up = std::clamp(d[2] * 2.5 + 0.1, 0.0, 1.0);   // sky: pale at the horizon, bluer above
				rgb[i * 3] = static_cast<std::uint8_t>(176.0 - 80.0 * up);
				rgb[i * 3 + 1] = static_cast<std::uint8_t>(190.0 - 72.0 * up);
				rgb[i * 3 + 2] = static_cast<std::uint8_t>(205.0 - 55.0 * up);
				continue;
			}
			label[i] = 1;
			distData[i] = static_cast<float>(t);
			XMFLOAT4 b;
			XMStoreFloat4(&b, XMVector4Transform(XMVectorSet(static_cast<float>(d[0] * t), static_cast<float>(d[1] * t), static_cast<float>(d[2] * t), 1), vp));
			depthData[i] = static_cast<float>(std::round(static_cast<double>(b.z) / b.w * 16777215.0) / 16777215.0);   // D24_UNORM
			// Plain lighting: grass on gentle land, rock on steep, a sun and some sky light.
			const double light = 0.35 + 0.65 * std::max(n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2], 0.0);
			const double rock = std::clamp((0.86 - n[2]) / 0.12, 0.0, 1.0);
			const double base[3]{ 86.0 + (128.0 - 86.0) * rock, 104.0 + (124.0 - 104.0) * rock, 62.0 + (120.0 - 62.0) * rock };
			for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = static_cast<std::uint8_t>(std::clamp(base[c] * light, 0.0, 255.0));
			if (wantIdeal && x % 4 == 0 && y % 4 == 0) {
				// The even layer lying on the true land, integrated in 16-unit steps (the last step ends on the land).
				double       tau = 0.0;
				const double step = 16.0;
				for (double s = 0.5 * step; s < t; s += step) {
					double gz = 0.0;
					if (!land.Height(cam[0] + d[0] * s, cam[1] + d[1] * s, gz)) break;
					tau += IdealDensity(cam[2] + d[2] * s - gz, kThickness) * step / kUnitsPerMetre;
				}
				ideal[i] = static_cast<float>(1.0 - std::exp(-tau));
			}
		}
	}
	ID3D11ShaderResourceView* depthSrv = MakeTexture(dev, DXGI_FORMAT_R32_FLOAT, kSize, depthData.data(), sizeof(float));
	WriteBin(outDir + "/scene.rgb", rgb.data(), rgb.size());
	WriteBin(outDir + "/label.u8", label.data(), label.size());
	WriteBin(outDir + "/dist.f32", distData.data(), distData.size() * sizeof(float));
	if (wantIdeal) WriteBin(outDir + "/ideal.f32", ideal.data(), ideal.size() * sizeof(float));

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
	ID3D11ShaderResourceView* none[7]{};

	// pass 0: field + far layer with patches (what the player sees); pass 1: the far layer alone, even (for the ideal).
	std::ofstream film;
	if (filmFrames) {
		film.open(outDir + "/video_" + variantName + ".u8", std::ios::binary);
		if (!film) {
			std::fprintf(stderr, "could not write %s/video_%s.u8\n", outDir.c_str(), variantName);
			g_writeFailed = true;
		}
	}
	for (int pass = 0; pass < (variant == 1 || filmFrames ? 1 : 2); ++pass) {
		for (int frame = 0; frame < (filmFrames ? filmFrames : 1); ++frame) {
			const float filmTime = frame / kFilmFps, travelX = kFilmWind * filmTime, travelY = 0.35f * kFilmWind * filmTime;
			if (filmFrames) sim.BuildColumns(ctx, hSrv, kThickness, filmTime, travelX, travelY);
			RaymarchShaders::RayCB rc{};
			std::memcpy(rc.invViewProj, &inv, sizeof(rc.invViewProj));
			rc.field[0] = fieldOx;
			rc.field[1] = fieldOy;
			rc.field[2] = kExtent;
			rc.field[3] = pass == 0 ? 1.0f : 0.0f;
			rc.box[0] = lo - 1.0f;
			rc.box[1] = hi + kThickness + 2.0f;
			rc.box[2] = kExtent;
			rc.box[3] = static_cast<float>(kSteps);
			rc.fog[0] = kOpacity;
			rc.fog[1] = kSteepness;
			rc.fog[2] = static_cast<float>(frame);
			rc.farNoise[0] = travelX;
			rc.farNoise[1] = travelY;
			rc.farNoise[2] = filmTime;
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
			rc.posAdjust[0] = static_cast<float>(cam[0]);
			rc.posAdjust[1] = static_cast<float>(cam[1]);
			rc.posAdjust[2] = static_cast<float>(cam[2]);
			rc.farFog[0] = variant == 1 ? 0.0f : 1.0f;
			rc.farFog[1] = variant == 0 ? 4000.0f : static_cast<float>(groundZ);   // old: FlatLength; new: the level used without a table
			rc.farFog[2] = kThickness;
			rc.farFog[3] = pass == 0 ? 1.0f : 0.0f;
			if (variant == 2) SetLandGrid(rc, landOx, landOy, kLandCell, static_cast<float>(kLandSize));
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
			ctx->PSSetShaderResources(0, 7, none);

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
			ctx->PSSetShaderResources(0, 7, none);
			ctx->OMSetBlendState(nullptr, black, 0xFFFFFFFF);

			const auto         img = ReadFloats(dev, ctx, sceneTex, kSize, kSize, 4);
			std::vector<float> fog(pixels);
			for (std::size_t i = 0; i < pixels; ++i) fog[i] = img[i * 4];
			if (pass == 0) {
				std::vector<std::uint8_t> bytes(pixels);
				for (std::size_t i = 0; i < pixels; ++i) bytes[i] = static_cast<std::uint8_t>(std::clamp(fog[i], 0.0f, 1.0f) * 255.0f + 0.5f);
				if (filmFrames) film.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
				else WriteBin(outDir + "/fog_" + variantName + ".u8", bytes.data(), bytes.size());
			} else {
				WriteBin(outDir + "/faronly_" + std::string(variantName) + ".f32", fog.data(), fog.size() * sizeof(float));
			}
		}
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
	SafeRelease(hSrv);
	sim.Release();
	SafeRelease(ctx);
	SafeRelease(dev);
	if (filmFrames) {
		film.close();
		if (!film) {
			std::fprintf(stderr, "could not write %s/video_%s.u8\n", outDir.c_str(), variantName);
			g_writeFailed = true;
		}
	}
	return g_writeFailed ? 1 : 0;
}
