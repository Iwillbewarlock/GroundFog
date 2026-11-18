#include "WindSim.h"

#include "ComputeGuard.h"

#include <d3dcompiler.h>

#include <cstdlib>
#include <cstring>

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

	constexpr const char* kEntries[] = { "CSShift", "CSWall", "CSCurl", "CSForce", "CSAdvectVel", "CSDivergence",
		"CSPressure", "CSSubtract", "CSAdvectFog", "CSColumn" };

	constexpr UINT kSrvSlots = ComputeGuard::kSrvSlots;
	constexpr UINT kUavSlots = ComputeGuard::kUavSlots;

	void Unbind(ID3D11DeviceContext* a_ctx)
	{
		ID3D11ShaderResourceView*  noSrv[kSrvSlots]{};
		ID3D11UnorderedAccessView* noUav[kUavSlots]{};
		a_ctx->CSSetShaderResources(0, kSrvSlots, noSrv);
		a_ctx->CSSetUnorderedAccessViews(0, kUavSlots, noUav, nullptr);
	}
}

bool WindSim::CreateTarget(ID3D11Device* a_device, DXGI_FORMAT a_format, Target& a_out)
{
	D3D11_TEXTURE2D_DESC desc{};
	desc.Width = desc.Height = static_cast<UINT>(_res);
	desc.MipLevels = desc.ArraySize = 1;
	desc.Format = a_format;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	return SUCCEEDED(a_device->CreateTexture2D(&desc, nullptr, &a_out.tex)) &&
	       SUCCEEDED(a_device->CreateShaderResourceView(a_out.tex, nullptr, &a_out.srv)) &&
	       SUCCEEDED(a_device->CreateUnorderedAccessView(a_out.tex, nullptr, &a_out.uav));
}

void WindSim::ReleaseTarget(Target& a_t)
{
	SafeRelease(a_t.uav);
	SafeRelease(a_t.srv);
	SafeRelease(a_t.tex);
}

// FogSampling 0 is WindShaders::kSource as it stands; Create and ShaderJobs use the same text for each mode.
std::string WindSim::Source(int a_fogSampling)
{
	return a_fogSampling == 1 ? std::string("#define FOG_SAMPLING 1\n") + WindShaders::kSource : std::string(WindShaders::kSource);
}

std::vector<ShaderCache::Job> WindSim::ShaderJobs(int a_fogSampling)
{
	std::vector<ShaderCache::Job> jobs;
	for (int k = 0; k < kKernelCount; ++k) jobs.push_back({ Source(a_fogSampling), kEntries[k], "cs_5_0", "GroundFogWind" });
	return jobs;
}

bool WindSim::Create(ID3D11Device* a_device, int a_resolution, std::string& a_error, int a_fogSampling)
{
	Release();
	_res = a_resolution;
	_fogSampling = a_fogSampling == 1 ? 1 : 0;
	const std::string source = Source(_fogSampling);
	for (int k = 0; k < kKernelCount; ++k) {
		ID3DBlob* code = nullptr;
		ID3DBlob* errors = nullptr;
		const HRESULT hr = ShaderCache::Compile(source, kEntries[k], "cs_5_0", "GroundFogWind", &code, &errors);
		if (FAILED(hr)) {
			a_error = std::string(kEntries[k]) + ": " + (errors ? static_cast<const char*>(errors->GetBufferPointer()) : "no message");
			SafeRelease(errors);
			SafeRelease(code);
			Release();
			return false;
		}
		SafeRelease(errors);
		const HRESULT made = a_device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &_cs[k]);
		SafeRelease(code);
		if (FAILED(made)) {
			a_error = std::string(kEntries[k]) + ": CreateComputeShader failed";
			Release();
			return false;
		}
	}

	D3D11_BUFFER_DESC cbDesc{ sizeof(WindShaders::SimCB), D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
	D3D11_BUFFER_DESC iterDesc{ 16, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };
	D3D11_BUFFER_DESC placeDesc{ 96, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0 };   // PlaceCB: PlaceP, SteamP, PlaceG, LookP, CoarseG, CoarseP
	D3D11_SAMPLER_DESC sd{};
	sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	sd.MaxLOD = D3D11_FLOAT32_MAX;
	bool ok = SUCCEEDED(a_device->CreateBuffer(&cbDesc, nullptr, &_cb)) &&
	          SUCCEEDED(a_device->CreateBuffer(&iterDesc, nullptr, &_iter)) &&
	          SUCCEEDED(a_device->CreateBuffer(&placeDesc, nullptr, &_place)) &&
	          SUCCEEDED(a_device->CreateSamplerState(&sd, &_linear));
	for (int i = 0; ok && i < 2; ++i) {
		ok = CreateTarget(a_device, DXGI_FORMAT_R16G16_FLOAT, _vel[i]) &&
		     CreateTarget(a_device, DXGI_FORMAT_R32G32_FLOAT, _fog[i]) &&  // rho, delta; fp16 rounds away slow recovery
		     CreateTarget(a_device, DXGI_FORMAT_R32_FLOAT, _pressure[i]);
	}
	ok = ok && CreateTarget(a_device, DXGI_FORMAT_R32_FLOAT, _div) &&
	     CreateTarget(a_device, DXGI_FORMAT_R16_FLOAT, _curl) &&
	     CreateTarget(a_device, DXGI_FORMAT_R16_FLOAT, _wall) &&
	     CreateTarget(a_device, DXGI_FORMAT_R32G32B32A32_FLOAT, _column) &&
	     CreateTarget(a_device, DXGI_FORMAT_R32G32_FLOAT, _upper) &&
	     CreateTarget(a_device, DXGI_FORMAT_R16_FLOAT, _relief);   // typed R16 UAV, as _wall and _curl
	if (!ok) {
		a_error = "texture, buffer or sampler creation failed";
		Release();
		return false;
	}
	_ready = true;
	return true;
}

void WindSim::Release()
{
	for (auto*& cs : _cs) SafeRelease(cs);
	SafeRelease(_cb);
	SafeRelease(_iter);
	SafeRelease(_place);
	SafeRelease(_linear);
	for (auto& t : _vel) ReleaseTarget(t);
	for (auto& t : _fog) ReleaseTarget(t);
	for (auto& t : _pressure) ReleaseTarget(t);
	ReleaseTarget(_div);
	ReleaseTarget(_curl);
	ReleaseTarget(_wall);
	ReleaseTarget(_column);
	ReleaseTarget(_upper);
	ReleaseTarget(_relief);
	_ready = false;
}

void WindSim::SetIter(ID3D11DeviceContext* a_ctx, float a_x, float a_y, float a_z, float a_w)
{
	const float v[4]{ a_x, a_y, a_z, a_w };
	a_ctx->UpdateSubresource(_iter, 0, nullptr, v, 0, 0);
}

void WindSim::Run(ID3D11DeviceContext* a_ctx, Kernel a_kernel)
{
	const UINT groups = static_cast<UINT>((_res + WindShaders::kGroup - 1) / WindShaders::kGroup);
	a_ctx->CSSetShader(_cs[a_kernel], nullptr, 0);
	a_ctx->Dispatch(groups, groups, 1);
	Unbind(a_ctx);
}

void WindSim::Reset(ID3D11DeviceContext* a_ctx)
{
	if (!_ready) {
		return;
	}
	const float zero[4]{ 0, 0, 0, 0 };
	const float fog[4]{ 1, 0, 0, 0 };   // rho 1 (the undisturbed layer), delta 0
	for (int i = 0; i < 2; ++i) {
		a_ctx->ClearUnorderedAccessViewFloat(_vel[i].uav, zero);
		a_ctx->ClearUnorderedAccessViewFloat(_pressure[i].uav, zero);
		a_ctx->ClearUnorderedAccessViewFloat(_fog[i].uav, fog);
	}
	a_ctx->ClearUnorderedAccessViewFloat(_div.uav, zero);
	a_ctx->ClearUnorderedAccessViewFloat(_curl.uav, zero);
	a_ctx->ClearUnorderedAccessViewFloat(_wall.uav, zero);
	a_ctx->ClearUnorderedAccessViewFloat(_column.uav, zero);
	a_ctx->ClearUnorderedAccessViewFloat(_upper.uav, zero);
	a_ctx->ClearUnorderedAccessViewFloat(_relief.uav, zero);
}

void WindSim::BuildColumns(ID3D11DeviceContext* a_ctx, ID3D11ShaderResourceView* a_heights, float a_thickness, float a_time,
	float a_travelX, float a_travelY, ID3D11ShaderResourceView* a_place, std::array<float, 4> a_placeParams, std::array<float, 4> a_steamParams,
	std::array<float, 4> a_placeGrid, std::array<float, 4> a_look, ID3D11ShaderResourceView* a_coarse, std::array<float, 4> a_coarseGrid,
	std::array<float, 4> a_coarseParams)
{
	if (!_ready) {
		return;
	}
	const ComputeGuard guard(a_ctx);
	_lastCb.heightGrid[3] = a_heights ? 1.0f : 0.0f;
	a_ctx->UpdateSubresource(_cb, 0, nullptr, &_lastCb, 0, 0);
	SetIter(a_ctx, a_thickness, a_time, a_travelX, a_travelY);
	a_ctx->CSSetConstantBuffers(0, 1, &_cb);
	a_ctx->CSSetConstantBuffers(1, 1, &_iter);
	a_ctx->CSSetShaderResources(1, 1, &_fog[_fogCur].srv);
	a_ctx->CSSetShaderResources(4, 1, &_wall.srv);
	a_ctx->CSSetShaderResources(6, 1, &a_heights);
	a_placeParams[3] = a_place ? 1.0f : 0.0f;
	if (!a_coarse) a_coarseGrid[2] = 0.0f;   // no texture: no coarse pools
	const std::array<float, 24> placeCb{ a_placeParams[0], a_placeParams[1], a_placeParams[2], a_placeParams[3], a_steamParams[0], a_steamParams[1],
		a_steamParams[2], a_steamParams[3], a_placeGrid[0], a_placeGrid[1], a_placeGrid[2], a_placeGrid[3], a_look[0], a_look[1], a_look[2], a_look[3],
		a_coarseGrid[0], a_coarseGrid[1], a_coarseGrid[2], a_coarseGrid[3], a_coarseParams[0], a_coarseParams[1], a_coarseParams[2], a_coarseParams[3] };
	a_ctx->UpdateSubresource(_place, 0, nullptr, placeCb.data(), 0, 0);
	a_ctx->CSSetConstantBuffers(2, 1, &_place);
	a_ctx->CSSetShaderResources(8, 1, &a_place);
	a_ctx->CSSetShaderResources(9, 1, &a_coarse);
	a_ctx->CSSetUnorderedAccessViews(4, 1, &_column.uav, nullptr);
	a_ctx->CSSetUnorderedAccessViews(5, 1, &_upper.uav, nullptr);
	a_ctx->CSSetUnorderedAccessViews(6, 1, &_relief.uav, nullptr);   // Run's Unbind clears u0..u6 (ComputeGuard::kUavSlots)
	Run(a_ctx, kColumn);
}

void WindSim::ShiftOne(ID3D11DeviceContext* a_ctx, Target (&a_pair)[2], int& a_cur, std::array<float, 4> a_fill)
{
	SetIter(a_ctx, a_fill[0], a_fill[1], a_fill[2], a_fill[3]);
	ID3D11ShaderResourceView* src = a_pair[a_cur].srv;
	a_ctx->CSSetShaderResources(7, 1, &src);
	a_ctx->CSSetUnorderedAccessViews(4, 1, &a_pair[1 - a_cur].uav, nullptr);
	Run(a_ctx, kShift);
	a_cur = 1 - a_cur;
}

void WindSim::Shift(ID3D11DeviceContext* a_ctx, int a_sx, int a_sy)
{
	if (!_ready || (a_sx == 0 && a_sy == 0)) {
		return;
	}
	if (std::abs(a_sx) >= _res || std::abs(a_sy) >= _res) {
		Reset(a_ctx);
		return;
	}
	const ComputeGuard guard(a_ctx);
	_lastCb.shift[0] = static_cast<float>(a_sx);
	_lastCb.shift[1] = static_cast<float>(a_sy);
	_lastCb.grid[3] = static_cast<float>(_res);
	a_ctx->UpdateSubresource(_cb, 0, nullptr, &_lastCb, 0, 0);
	a_ctx->CSSetConstantBuffers(0, 1, &_cb);
	a_ctx->CSSetConstantBuffers(1, 1, &_iter);
	ShiftOne(a_ctx, _vel, _velCur, { 0, 0, 0, 1 });  // w = 1: copy edge velocity instead of filling
	ShiftOne(a_ctx, _fog, _fogCur, { 1, 0, 0, 0 });   // rho 1, delta 0 (w must stay 0: fill, not edge copy); pressure is not carried between frames
	_lastCb.shift[0] = _lastCb.shift[1] = 0.0f;
}

void WindSim::Step(ID3D11DeviceContext* a_ctx, const WindShaders::SimCB& a_cb, ID3D11ShaderResourceView* a_heights,
	int a_pressureIterations)
{
	if (!_ready) {
		return;
	}
	const ComputeGuard guard(a_ctx);
	_lastCb = a_cb;
	_lastCb.grid[3] = static_cast<float>(_res);
	_lastCb.heightGrid[3] = a_heights ? _lastCb.heightGrid[3] : 0.0f;
	a_ctx->UpdateSubresource(_cb, 0, nullptr, &_lastCb, 0, 0);
	a_ctx->CSSetConstantBuffers(0, 1, &_cb);
	a_ctx->CSSetConstantBuffers(1, 1, &_iter);
	a_ctx->CSSetSamplers(0, 1, &_linear);

	auto srv = [&](UINT a_slot, ID3D11ShaderResourceView* a_view) { a_ctx->CSSetShaderResources(a_slot, 1, &a_view); };
	auto uav = [&](UINT a_slot, ID3D11UnorderedAccessView* a_view) { a_ctx->CSSetUnorderedAccessViews(a_slot, 1, &a_view, nullptr); };

	// walls
	srv(6, a_heights);
	uav(2, _wall.uav);
	Run(a_ctx, kWall);

	// curl
	srv(0, _vel[_velCur].srv);
	uav(2, _curl.uav);
	Run(a_ctx, kCurl);

	// forces
	srv(0, _vel[_velCur].srv);
	srv(4, _wall.srv);
	srv(5, _curl.srv);
	uav(0, _vel[1 - _velCur].uav);
	Run(a_ctx, kForce);
	_velCur = 1 - _velCur;

	// advect velocity
	srv(0, _vel[_velCur].srv);
	srv(4, _wall.srv);
	uav(0, _vel[1 - _velCur].uav);
	Run(a_ctx, kAdvectVel);
	_velCur = 1 - _velCur;

	// divergence (with blast sources)
	srv(0, _vel[_velCur].srv);
	srv(4, _wall.srv);
	uav(3, _div.uav);
	Run(a_ctx, kDivergence);

	// pressure: Jacobi from a zero guess every frame (GPU Gems 1 ch. 38). Nothing carries over, so a source whose
	// sum on the grid is not exactly zero (cut by a wall or the edge) cannot build up a constant over time.
	const float zero[4]{ 0, 0, 0, 0 };
	a_ctx->ClearUnorderedAccessViewFloat(_pressure[_pressureCur].uav, zero);
	for (int i = 0; i < a_pressureIterations; ++i) {
		srv(2, _pressure[_pressureCur].srv);
		srv(3, _div.srv);
		srv(4, _wall.srv);
		uav(2, _pressure[1 - _pressureCur].uav);
		Run(a_ctx, kPressure);
		_pressureCur = 1 - _pressureCur;
	}

	// subtract gradient
	srv(0, _vel[_velCur].srv);
	srv(2, _pressure[_pressureCur].srv);
	srv(4, _wall.srv);
	uav(0, _vel[1 - _velCur].uav);
	Run(a_ctx, kSubtract);
	_velCur = 1 - _velCur;

	// advect fog with the projected velocity
	srv(0, _vel[_velCur].srv);
	srv(1, _fog[_fogCur].srv);
	srv(4, _wall.srv);
	uav(1, _fog[1 - _fogCur].uav);
	Run(a_ctx, kAdvectFog);
	_fogCur = 1 - _fogCur;
}
