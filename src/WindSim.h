#pragma once

// The shared wind + fog grid on the GPU. No game dependency: the plugin fills SimCB from game state, the
// offline check fills it by hand. All compute state it touches is saved and restored around each call.

#include "ShaderCache.h"
#include "WindShaders.h"

#include <d3d11.h>

#include <array>
#include <string>
#include <vector>

class WindSim
{
public:
	bool Create(ID3D11Device* a_device, int a_resolution, std::string& a_error);
	static std::vector<ShaderCache::Job> ShaderJobs();   // what Create compiles, for the background compile
	void Release();
	[[nodiscard]] bool Ready() const { return _ready; }
	[[nodiscard]] int  Resolution() const { return _res; }

	// Sets velocity 0, pressure 0, fog (rho 1, delta 0).
	void Reset(ID3D11DeviceContext* a_ctx);

	// Moves every field by (a_sx, a_sy) texels: new(x, y) = old(x + sx, y + sy). Exposed texels get the rest state.
	void Shift(ID3D11DeviceContext* a_ctx, int a_sx, int a_sy);

	// One frame. a_heights may be null (no walls). a_pressureIterations Jacobi steps from a zero guess.
	void Step(ID3D11DeviceContext* a_ctx, const WindShaders::SimCB& a_cb, ID3D11ShaderResourceView* a_heights,
		int a_pressureIterations);

	[[nodiscard]] ID3D11ShaderResourceView* VelSrv() const { return _vel[_velCur].srv; }
	[[nodiscard]] ID3D11Texture2D*          FogTexture() const { return _fog[_fogCur].tex; }   // RG32F: x = rho, y = delta
	[[nodiscard]] ID3D11ShaderResourceView* ColumnSrv() const { return _column.srv; }
	[[nodiscard]] ID3D11Texture2D*          ColumnTexture() const { return _column.tex; }
	// The tall layer (steam, pooled fog): H, W per texel, same grid as the columns.
	[[nodiscard]] ID3D11ShaderResourceView* UpperSrv() const { return _upper.srv; }
	[[nodiscard]] ID3D11Texture2D*          UpperTexture() const { return _upper.tex; }

	// Fog columns for the raymarch from the current fog field and the height texture (see CSColumn).
	// a_place: PlacementRules grid at the height texture's size (nullptr = none); a_placeParams: PlaceCB
	// (radiation multiplier, steam cold, wind factor, unused). Defaults leave the columns as before.
	void BuildColumns(ID3D11DeviceContext* a_ctx, ID3D11ShaderResourceView* a_heights, float a_thickness, float a_time,
		float a_travelX, float a_travelY, ID3D11ShaderResourceView* a_place = nullptr, std::array<float, 4> a_placeParams = { 1.0f, 0.0f, 1.0f, 0.0f },
		std::array<float, 4> a_steamParams = { 0.0f, 0.0f, 0.0f, 0.0f },   // SteamP: evaporation, steam height, steam density, pool density
		std::array<float, 4> a_placeGrid = { 0.0f, 0.0f, 0.0f, 0.0f },     // PlaceG: placement texture origin x, y, cell, size (cell 0 = height grid)
		std::array<float, 4> a_look = { 0.0f, 0.0f, 0.0f, 0.0f },          // LookP: wisp amount, region value (RegionRules)
		ID3D11ShaderResourceView* a_coarse = nullptr,                        // coarse pools (ground, depth), RG32F
		std::array<float, 4> a_coarseGrid = { 0.0f, 0.0f, 0.0f, 0.0f },    // CoarseG: origin x, y, cell, size
		std::array<float, 4> a_coarseParams = { 0.0f, 0.0f, 0.0f, 0.0f }); // CoarseP: ground tolerance, depth limit
	[[nodiscard]] ID3D11Texture2D*          VelTexture() const { return _vel[_velCur].tex; }
	[[nodiscard]] ID3D11Texture2D*          WallTexture() const { return _wall.tex; }

private:
	struct Target
	{
		ID3D11Texture2D*           tex{ nullptr };
		ID3D11ShaderResourceView*  srv{ nullptr };
		ID3D11UnorderedAccessView* uav{ nullptr };
	};

	enum Kernel
	{
		kShift,
		kWall,
		kCurl,
		kForce,
		kAdvectVel,
		kDivergence,
		kPressure,
		kSubtract,
		kAdvectFog,
		kColumn,
		kKernelCount
	};

	bool CreateTarget(ID3D11Device* a_device, DXGI_FORMAT a_format, Target& a_out);
	void ReleaseTarget(Target& a_t);
	void SetIter(ID3D11DeviceContext* a_ctx, float a_x, float a_y, float a_z, float a_w);
	void Run(ID3D11DeviceContext* a_ctx, Kernel a_kernel);
	void ShiftOne(ID3D11DeviceContext* a_ctx, Target (&a_pair)[2], int& a_cur, std::array<float, 4> a_fill);

	std::array<ID3D11ComputeShader*, kKernelCount> _cs{};
	ID3D11Buffer*       _cb{ nullptr };
	ID3D11Buffer*       _iter{ nullptr };
	ID3D11Buffer*       _place{ nullptr };   // PlaceCB
	ID3D11SamplerState* _linear{ nullptr };
	Target              _vel[2]{};
	Target              _fog[2]{};
	Target              _pressure[2]{};
	Target              _div{};
	Target              _curl{};
	Target              _wall{};
	Target              _column{};
	Target              _upper{};   // R32G32: tall layer H, W
	int                 _velCur{ 0 };
	int                 _fogCur{ 0 };
	int                 _pressureCur{ 0 };
	int                 _res{ 0 };
	bool                _ready{ false };
	WindShaders::SimCB  _lastCb{};
};
