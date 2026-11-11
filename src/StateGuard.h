#pragma once

// Saves every pipeline binding the fog pass changes and puts it back on destruction, so the game's
// cached render state (and Community Shaders' bindings) stay valid. Restore order: output merger
// first, then shader resources, because binding a render target can silently unbind an SRV of the
// same resource (ID3D11DeviceContext::OMSetRenderTargets docs).
class StateGuard
{
public:
	static constexpr UINT kSrvSlots = 6;  // t0..t5 on VS and PS (t4: the tall fog layer, t5: the coarse grid)
	static constexpr UINT kCbSlots = 1;   // b0 on VS and PS

	struct Snapshot
	{
		std::array<ID3D11RenderTargetView*, D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT> rtvs{};
		ID3D11DepthStencilView*   dsv{ nullptr };
		ID3D11DepthStencilState*  depthState{ nullptr };
		UINT                      stencilRef{ 0 };
		ID3D11BlendState*         blendState{ nullptr };
		std::array<FLOAT, 4>      blendFactor{};
		UINT                      sampleMask{ 0 };
		ID3D11RasterizerState*    rasterState{ nullptr };
		std::array<D3D11_VIEWPORT, D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE> viewports{};
		UINT                      viewportCount{ 0 };
		D3D11_PRIMITIVE_TOPOLOGY  topology{ D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED };
		ID3D11InputLayout*        inputLayout{ nullptr };
		ID3D11VertexShader*       vs{ nullptr };
		ID3D11PixelShader*        ps{ nullptr };
		ID3D11GeometryShader*     gs{ nullptr };
		ID3D11HullShader*         hs{ nullptr };
		ID3D11DomainShader*       ds{ nullptr };
		std::array<ID3D11ShaderResourceView*, kSrvSlots> vsSrv{};
		std::array<ID3D11ShaderResourceView*, kSrvSlots> psSrv{};
		std::array<ID3D11Buffer*, kCbSlots>             vsCb{};
		std::array<ID3D11Buffer*, kCbSlots>             psCb{};
		ID3D11SamplerState*                             vsSampler{ nullptr };
		ID3D11SamplerState*                             psSampler{ nullptr };

		void Capture(ID3D11DeviceContext* a_ctx)
		{
			a_ctx->OMGetRenderTargets(static_cast<UINT>(rtvs.size()), rtvs.data(), &dsv);
			a_ctx->OMGetDepthStencilState(&depthState, &stencilRef);
			a_ctx->OMGetBlendState(&blendState, blendFactor.data(), &sampleMask);
			a_ctx->RSGetState(&rasterState);
			viewportCount = static_cast<UINT>(viewports.size());
			a_ctx->RSGetViewports(&viewportCount, viewports.data());
			a_ctx->IAGetPrimitiveTopology(&topology);
			a_ctx->IAGetInputLayout(&inputLayout);
			a_ctx->VSGetShader(&vs, nullptr, nullptr);
			a_ctx->PSGetShader(&ps, nullptr, nullptr);
			a_ctx->GSGetShader(&gs, nullptr, nullptr);
			a_ctx->HSGetShader(&hs, nullptr, nullptr);
			a_ctx->DSGetShader(&ds, nullptr, nullptr);
			a_ctx->VSGetShaderResources(0, kSrvSlots, vsSrv.data());
			a_ctx->PSGetShaderResources(0, kSrvSlots, psSrv.data());
			a_ctx->VSGetConstantBuffers(0, kCbSlots, vsCb.data());
			a_ctx->PSGetConstantBuffers(0, kCbSlots, psCb.data());
			a_ctx->VSGetSamplers(0, 1, &vsSampler);
			a_ctx->PSGetSamplers(0, 1, &psSampler);
		}

		void Apply(ID3D11DeviceContext* a_ctx) const
		{
			a_ctx->OMSetRenderTargets(static_cast<UINT>(rtvs.size()), rtvs.data(), dsv);
			a_ctx->OMSetDepthStencilState(depthState, stencilRef);
			a_ctx->OMSetBlendState(blendState, blendFactor.data(), sampleMask);
			a_ctx->RSSetState(rasterState);
			if (viewportCount > 0) {
				a_ctx->RSSetViewports(viewportCount, viewports.data());
			}
			a_ctx->IASetPrimitiveTopology(topology);
			a_ctx->IASetInputLayout(inputLayout);
			a_ctx->VSSetShader(vs, nullptr, 0);
			a_ctx->PSSetShader(ps, nullptr, 0);
			a_ctx->GSSetShader(gs, nullptr, 0);
			a_ctx->HSSetShader(hs, nullptr, 0);
			a_ctx->DSSetShader(ds, nullptr, 0);
			a_ctx->VSSetShaderResources(0, kSrvSlots, vsSrv.data());
			a_ctx->PSSetShaderResources(0, kSrvSlots, psSrv.data());
			a_ctx->VSSetConstantBuffers(0, kCbSlots, vsCb.data());
			a_ctx->PSSetConstantBuffers(0, kCbSlots, psCb.data());
			a_ctx->VSSetSamplers(0, 1, &vsSampler);
			a_ctx->PSSetSamplers(0, 1, &psSampler);
		}

		// Same bindings? Compares pointers only (viewport values too).
		[[nodiscard]] bool SameAs(const Snapshot& a_other) const
		{
			const auto sameViewports = viewportCount == a_other.viewportCount &&
			                           std::memcmp(viewports.data(), a_other.viewports.data(), sizeof(D3D11_VIEWPORT) * viewportCount) == 0;
			return rtvs == a_other.rtvs && dsv == a_other.dsv && depthState == a_other.depthState &&
			       stencilRef == a_other.stencilRef && blendState == a_other.blendState &&
			       blendFactor == a_other.blendFactor && sampleMask == a_other.sampleMask &&
			       rasterState == a_other.rasterState && sameViewports && topology == a_other.topology &&
			       inputLayout == a_other.inputLayout && vs == a_other.vs && ps == a_other.ps && gs == a_other.gs &&
			       hs == a_other.hs && ds == a_other.ds && vsSrv == a_other.vsSrv && psSrv == a_other.psSrv &&
			       vsCb == a_other.vsCb && psCb == a_other.psCb && vsSampler == a_other.vsSampler && psSampler == a_other.psSampler;
		}

		void Release()
		{
			auto rel = [](auto*& a_p) {
				if (a_p) {
					a_p->Release();
					a_p = nullptr;
				}
			};
			for (auto*& p : rtvs) rel(p);
			rel(dsv);
			rel(depthState);
			rel(blendState);
			rel(rasterState);
			rel(inputLayout);
			rel(vs);
			rel(ps);
			rel(gs);
			rel(hs);
			rel(ds);
			for (auto*& p : vsSrv) rel(p);
			for (auto*& p : psSrv) rel(p);
			for (auto*& p : vsCb) rel(p);
			for (auto*& p : psCb) rel(p);
			rel(vsSampler);
			rel(psSampler);
		}
	};

	explicit StateGuard(ID3D11DeviceContext* a_ctx) :
		_ctx(a_ctx)
	{
		_saved.Capture(_ctx);
	}

	~StateGuard()
	{
		_saved.Apply(_ctx);
		_saved.Release();
	}

	StateGuard(const StateGuard&) = delete;
	StateGuard& operator=(const StateGuard&) = delete;

	[[nodiscard]] const Snapshot& Saved() const { return _saved; }

private:
	ID3D11DeviceContext* _ctx;
	Snapshot             _saved;
};
