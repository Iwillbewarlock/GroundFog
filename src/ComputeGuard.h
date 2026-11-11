#pragma once

#include <d3d11.h>

// Saves and restores the compute bindings our dispatches touch (shader, t0-t9, u0-u5, b0-b2, s0), so the game's
// (and Community Shaders') compute state is unchanged afterwards. Used by WindSim and the raymarch probe.
struct ComputeGuard
{
	static constexpr UINT kSrvSlots = 10;
	static constexpr UINT kCbSlots = 3;
	static constexpr UINT kUavSlots = 6;

	ID3D11DeviceContext*       ctx;
	ID3D11ComputeShader*       cs{ nullptr };
	ID3D11ShaderResourceView*  srv[kSrvSlots]{};
	ID3D11UnorderedAccessView* uav[kUavSlots]{};
	ID3D11Buffer*              cb[kCbSlots]{};
	ID3D11SamplerState*        sampler{ nullptr };

	explicit ComputeGuard(ID3D11DeviceContext* a_ctx) :
		ctx(a_ctx)
	{
		ctx->CSGetShader(&cs, nullptr, nullptr);
		ctx->CSGetShaderResources(0, kSrvSlots, srv);
		ctx->CSGetUnorderedAccessViews(0, kUavSlots, uav);
		ctx->CSGetConstantBuffers(0, kCbSlots, cb);
		ctx->CSGetSamplers(0, 1, &sampler);
	}
	~ComputeGuard()
	{
		ID3D11ShaderResourceView* noSrv[kSrvSlots]{};
		ctx->CSSetShaderResources(0, kSrvSlots, noSrv);  // free our textures before re-binding UAVs
		ctx->CSSetUnorderedAccessViews(0, kUavSlots, uav, nullptr);
		ctx->CSSetShaderResources(0, kSrvSlots, srv);
		ctx->CSSetConstantBuffers(0, kCbSlots, cb);
		ctx->CSSetSamplers(0, 1, &sampler);
		ctx->CSSetShader(cs, nullptr, 0);
		Rel(cs);
		for (auto*& p : srv) Rel(p);
		for (auto*& p : uav) Rel(p);
		for (auto*& p : cb) Rel(p);
		Rel(sampler);
	}
	ComputeGuard(const ComputeGuard&) = delete;
	ComputeGuard& operator=(const ComputeGuard&) = delete;

private:
	template <class T>
	static void Rel(T*& a_p)
	{
		if (a_p) {
			a_p->Release();
			a_p = nullptr;
		}
	}
};
