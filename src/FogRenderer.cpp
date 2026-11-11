#include "PCH.h"

#include <future>

#include "FogRenderer.h"

#include "RaymarchShaders.h"
#include "ShaderCache.h"
#include "WindSim.h"
#include "GpuTimer.h"
#include "Settings.h"
#include "StateGuard.h"
#include "ComputeGuard.h"
#include "FogConditions.h"
#include "FogLight.h"
#include "GameFog.h"
#include "PlacementRules.h"
#include "WeatherFog.h"
#include "Terrain.h"
#include "WindField.h"
#include "WorldspaceRules.h"
#include "RegionRules.h"
#include "EditorID.h"

namespace FogRenderer
{
	namespace
	{
		// This frame's camera, lighting and field facts, filled in Draw and read by the raymarch and the logs.
		struct FrameParams
		{
			float viewProj[16];
			float posAdjust[4];
			float depthParams[4];      // x = proj _33, y = proj _43
			float fogColor[4];         // rgb lit fog colour
			float distFog[4];          // x = near, y = far, z = power, w = max
			float distFogColorNear[4];
			float distFogColorFar[4];
			float fluid[4];            // x, y = wind/fog field origin, z = extent (units), w = 1 if the field is ready
		};

		// Why the raymarch drew nothing this frame. Each reason is logged once (warn), not every frame; the
		// draw is tried again on every later frame, so a transient failure (a target lost on resize) recovers.
		enum class NoDraw
		{
			kResources,   // raymarch shaders or constant buffer missing
			kDepth,       // the main depth has no shader resource view
			kTarget,      // the reduced-resolution target could not be created
			kMatrix,      // the camera matrix cannot be inverted
			kNothing,     // no wind field (or no known ground) while far fog and height fog are off
			kMap,         // the constant buffer could not be written
			kCount
		};
		std::array<bool, static_cast<std::size_t>(NoDraw::kCount)> g_noDrawLogged{};

		void LogNoDraw(NoDraw a_why, const char* a_text)
		{
			auto& logged = g_noDrawLogged[static_cast<std::size_t>(a_why)];
			if (!logged) {
				logged = true;
				logger::warn("fog not drawn: {} (logged once per loaded game; tried again every frame)", a_text);
			}
		}

		// Periodic state lines are for diagnosis only: off in a normal play session.
		bool kDiagLogs() { return Settings::selfTest || Settings::profiler; }

		ID3D11BlendState*         g_blend{ nullptr };
		ID3D11RasterizerState*    g_raster{ nullptr };
		ID3D11DepthStencilState*  g_depthOff{ nullptr };
		ID3D11SamplerState*       g_linear{ nullptr };
		ID3D11VertexShader*       g_rayVs{ nullptr };
		ID3D11PixelShader*        g_rayMarch{ nullptr };
		ID3D11PixelShader*        g_rayComposite{ nullptr };
		ID3D11Buffer*             g_rayCb{ nullptr };
		ID3D11Texture2D*          g_lowTex{ nullptr };
		ID3D11RenderTargetView*   g_lowRtv{ nullptr };
		ID3D11ShaderResourceView* g_lowSrv{ nullptr };
		UINT                      g_lowW{ 0 };
		UINT                      g_lowH{ 0 };
		ID3D11Texture2D*          g_heightTex{ nullptr };
		ID3D11ShaderResourceView* g_heightSrv{ nullptr };
		int                       g_heightSize{ 0 };
		GpuTimer                  g_timer;
		GpuTimer                  g_windTimer;   // wind + fog field step and the column build
		std::chrono::steady_clock::time_point g_lastGpuLog{};
		bool                      g_created{ false };
		bool                      g_failed{ false };
		Stats                     g_stats{};
		bool                      g_logCamera{ false };
		std::uint64_t             g_frame{ 0 };
		float                     g_mainNear43{ 0.0f };
		bool                      g_haveMainCamera{ false };
		std::chrono::steady_clock::time_point g_lastRejectLog{};
		std::chrono::steady_clock::time_point g_lastTerrainLog{};
		constexpr float           kMainCameraMaxDistance = 3000.0f;  // camera to player, game units
		std::uint64_t             g_drawnFrame{ ~0ull };
		float                     g_weatherFactor{ 1.0f };   // WeatherFog::CurrentFactor of this frame
		FogConditions::Result     g_cond{};                   // this frame's conditions
		// The [Worldspaces] rule of the player's worldspace (WorldspaceRules.h); looked up when the worldspace changes
		// and after a load. Render thread only.
		WorldspaceRules::Rule     g_regionRule{};
		std::string               g_regionText{ "none" };   // for the log and the self test
		const RE::TESWorldSpace*  g_regionSpace{ nullptr };
		bool                      g_regionKnown{ false };
		std::vector<WorldspaceRules::Entry> g_regionIni;
		bool                      g_regionIniRead{ false };
		// The [Regions] value of the player's weather region (RegionRules.h), eased over a few seconds when the region
		// changes. Render thread only.
		constexpr float           kGeoEaseSeconds = 2.0f;     // time constant: 95% of the way in about 6 s
		constexpr float           kGeoGapSeconds = 1.0f;      // no drawn frame for this long (loading screen, interior, a
		                                                      // worldspace set to off): take the new value at once
		float                     g_geoTarget{ 0.0f };
		float                     g_geo{ 0.0f };            // this frame's value, added to the placement multiplier
		float                     g_geoFarTarget{ 0.0f };
		float                     g_geoFar{ 0.0f };         // this frame's value for the far fog (RegionRules::Entry::far)
		const RE::TESRegion*      g_geoRegion{ nullptr };
		bool                      g_geoKnown{ false };      // false: look the region up again and take its value at once
		std::string               g_geoText{ "none" };      // for the log and the self test
		std::vector<RegionRules::Entry> g_geoIni;
		bool                      g_geoIniRead{ false };
		std::chrono::steady_clock::time_point g_geoTime{};
		// The last region the sky named, and the worldspace it was in. A forced weather (console fw, quests, weather
		// mods) clears Sky::region; the place has not changed, so its value stays until the sky names a region again or
		// the player leaves the worldspace. Limit: walking into another region under a forced weather keeps the old one.
		const RE::TESRegion*      g_geoHeld{ nullptr };
		const RE::TESWorldSpace*  g_geoHeldSpace{ nullptr };
		RE::NiPoint3              g_geoHeldPos{};         // where the player was when the sky last named it
		bool                      g_geoWasHeld{ false };  // the value in use came from the kept region (for the log line)
		// A kept region is dropped once the player is this far from where the sky last named it: a fast travel or a
		// door under a forced weather must not carry the old place along (reviewer 2026-11-11). Two exterior cells.
		constexpr float           kGeoHeldMaxDistance = 8192.0f;
		FogConditions::Sky        g_condSky{};                // their inputs, for logs
		float                     g_condWind{ -1.0f };        // low-passed Sky::windSpeed for the conditions (< 0: take the next)
		std::chrono::steady_clock::time_point g_condWindTime{};
		std::chrono::steady_clock::time_point g_condWindSnapUntil{};   // until then the conditions take the raw wind
		ID3D11Texture2D*          g_placeTex{ nullptr };      // PlacementRules grid, height-grid size, RGBA32F
		ID3D11ShaderResourceView* g_placeSrv{ nullptr };
		ID3D11Texture2D*          g_coarseTex{ nullptr };    // coarse pools: ground, depth (RG32F)
		ID3D11ShaderResourceView* g_coarseSrv{ nullptr };
		std::array<float, 4>      g_coarseGrid{ 0.0f, 0.0f, 0.0f, 0.0f };   // uploaded coarse texture: origin x, y, cell, size
		int                       g_placeSize{ 0 };
		std::chrono::steady_clock::time_point g_placeBuilt{};
		float                     g_placeOriginX{ 0.0f };   // grid origin the placement texture was built for
		float                     g_placeOriginY{ 0.0f };
		std::uint32_t             g_placeGeneration{ 0 };   // Terrain::Generation() at the last build
		bool                      g_placeForce{ true };     // rebuild on the next frame (ini reloaded)
		float                     g_placeFill{ -1.0f };     // PlacementRules poolFill baked into the last build
		std::vector<float>        g_poolEased;              // eased pool depths of the last build (PlacementRules::EasePools)
		bool                      g_poolSettling{ false };  // eased depths still off their targets: rebuild again (throttled)
		std::array<float, 3>      g_placeMax{ 1.0f, 0.0f, 0.0f };   // largest x, y, z in the last build (probe W bound)
		std::array<std::uint32_t, 8> g_placeCounts{};        // cells per kind in the last build
		// Mean x of the last applied build, for the log. Cells with unknown height count with their 1.0, and the cell
		// counts beside it in the log line are taken when a build is requested, so the two can be one build apart.
		float                        g_placeMean{ 0.0f };
		std::array<float, 4>         g_glowLog{};            // last glow: direction toward the sun, strength (for the log)
		void DropPlaceJob();   // waits for and drops a running placement build (defined with UpdatePlacement)
		// Height fog base: the player's height, smoothed so climbing a hill does not visibly lift the fog.
		float                     g_heightFogBase{ 0.0f };
		bool                      g_heightFogBaseSet{ false };
		std::chrono::steady_clock::time_point g_heightFogBaseTime{};
		constexpr float           kHeightFogBaseSeconds = 8.0f;   // time constant of that smoothing

		// Probe (ini Probe=1): GPU readback of 63 screen points every 2 s, logged as [PROBE] lines (kProbeCS).
		constexpr int  kProbeRing = 3;
		constexpr auto kProbeInterval = std::chrono::seconds(2);
		constexpr int  kProbePoints = RaymarchShaders::kProbeCols * RaymarchShaders::kProbeRows;
		constexpr int  kProbeSlots = RaymarchShaders::kProbeSlots;
		constexpr int  kProbeRecords = kProbePoints * kProbeSlots + RaymarchShaders::kProbeExtra;
		struct ProbeMeta
		{
			bool          pending{ false };
			int           id{ 0 };
			std::uint64_t frame{ 0 };
			std::uint64_t framesSince{ 0 };
			UINT          presentsSince{ 0 };
			std::uint64_t drawsSince{ 0 };
			float         wMax{ 0.0f };   // CSColumn W bound at capture (radiation part plus steam columns; see the capture site)
			int           sweepStep{ -1 };
			float         sweepPitch{ 0.0f };
			float         playerPitch{ 0.0f };
			bool          haveNi{ false };
			float         niCol0[3]{};
			float         niRow0[3]{};
			std::string   header;   // CPU-side camera and target facts at capture time
		};
		ID3D11ComputeShader*                  g_probeCs{ nullptr };
		ID3D11Buffer*                         g_probeCb{ nullptr };
		ID3D11Buffer*                         g_probeBuf{ nullptr };
		ID3D11UnorderedAccessView*            g_probeUav{ nullptr };
		std::array<ID3D11Buffer*, kProbeRing> g_probeStaging{};
		std::array<ProbeMeta, kProbeRing>     g_probeMeta{};
		int                                   g_probeNext{ 0 };
		int                                   g_probeId{ 0 };
		std::chrono::steady_clock::time_point g_lastProbe{};
		std::uint64_t                         g_probeLastFrame{ 0 };
		UINT                                  g_probeLastPresents{ 0 };
		std::uint64_t                         g_probeLastDraws{ 0 };

		// Probe=2: the plugin turns the player's look pitch itself and probes once per step (Actor::SetLooking on
		// the main thread through the SKSE task queue), then puts the pitch back. The logged dir.z shows which sign
		// looks down.
		constexpr float         kSweepPitch[] = { -0.25f, 0.0f, 0.35f, 0.7f, 1.05f, 1.4f, 0.7f, 0.0f };
		constexpr int           kSweepSteps = static_cast<int>(std::size(kSweepPitch));
		constexpr auto          kSweepHold = std::chrono::milliseconds(2500);
		constexpr auto          kSweepProbeAt = std::chrono::milliseconds(1500);
		constexpr std::uint64_t kSweepWaitDraws = 300;
		struct Sweep
		{
			int                                   step{ -1 };
			bool                                  done{ false };
			bool                                  probed{ false };
			float                                 original{ 0.0f };
			std::uint64_t                         startAt{ 0 };   // g_stats.draws at which the sweep starts
			std::chrono::steady_clock::time_point stepStart{};
			std::chrono::steady_clock::time_point lastUpdate{};
		};
		Sweep g_sweep;
		// A game load bumps the generation (queued pitch tasks of the old one do nothing) and re-arms the sweep.
		std::atomic<std::uint32_t> g_sweepGen{ 0 };
		std::atomic<bool>          g_sweepLoaded{ false };
		constexpr auto             kSweepMaxGap = std::chrono::seconds(1);      // no fog draw this long: abort
		constexpr auto             kSweepMaxWait = std::chrono::seconds(4);     // a step's probe still missing: abort

		template <class T>
		void SafeRelease(T*& a_p)
		{
			if (a_p) {
				a_p->Release();
				a_p = nullptr;
			}
		}

		ID3DBlob* Compile(const char* a_body, const char* a_target)
		{
			std::string source = RaymarchShaders::kCommon;
			source += a_body;
			ID3DBlob*  code = nullptr;
			ID3DBlob*  errors = nullptr;
			const auto hr = ShaderCache::Compile(source, "main", a_target, "GroundFogRay", &code, &errors);
			if (FAILED(hr)) {
				logger::error("raymarch shader {} compile failed (hr {:08X}): {}", a_target, static_cast<std::uint32_t>(hr),
					errors ? static_cast<const char*>(errors->GetBufferPointer()) : "no message");
				SafeRelease(errors);
				SafeRelease(code);
				return nullptr;
			}
			SafeRelease(errors);
			return code;
		}

		void ReleaseAll()
		{
			SafeRelease(g_blend);
			SafeRelease(g_raster);
			SafeRelease(g_depthOff);
			SafeRelease(g_linear);
			SafeRelease(g_rayVs);
			SafeRelease(g_rayMarch);
			SafeRelease(g_rayComposite);
			SafeRelease(g_rayCb);
			SafeRelease(g_lowSrv);
			SafeRelease(g_lowRtv);
			SafeRelease(g_lowTex);
			SafeRelease(g_heightSrv);
			SafeRelease(g_heightTex);
			g_heightSize = 0;
			DropPlaceJob();
			SafeRelease(g_placeSrv);
			SafeRelease(g_placeTex);
			g_placeSize = 0;
			SafeRelease(g_coarseSrv);
			SafeRelease(g_coarseTex);
			g_coarseGrid = { 0.0f, 0.0f, 0.0f, 0.0f };
			SafeRelease(g_probeCs);
			SafeRelease(g_probeCb);
			SafeRelease(g_probeUav);
			SafeRelease(g_probeBuf);
			for (auto*& b : g_probeStaging) SafeRelease(b);
			for (auto& m : g_probeMeta) m.pending = false;
			g_timer.Release();
			g_windTimer.Release();
		}

		// Any failure here leaves the fog pass off for the session (Ready() does not try again); it is logged once.
		bool CreateResources(ID3D11Device* a_device)
		{
			{
				ID3DBlob* vsb = Compile(RaymarchShaders::kVS, "vs_5_0");
				ID3DBlob* mb = Compile(RaymarchShaders::kMarchPS, "ps_5_0");
				ID3DBlob* cb = Compile(RaymarchShaders::kCompositePS, "ps_5_0");
				bool      ok = vsb && mb && cb;
				ok = ok && SUCCEEDED(a_device->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &g_rayVs));
				ok = ok && SUCCEEDED(a_device->CreatePixelShader(mb->GetBufferPointer(), mb->GetBufferSize(), nullptr, &g_rayMarch));
				ok = ok && SUCCEEDED(a_device->CreatePixelShader(cb->GetBufferPointer(), cb->GetBufferSize(), nullptr, &g_rayComposite));
				SafeRelease(vsb);
				SafeRelease(mb);
				SafeRelease(cb);
				g_stats.shadersCompiled = ok;
				if (!ok) {
					logger::error("raymarch shaders are not available; the fog pass is off");
					return false;
				}
				D3D11_BUFFER_DESC rd{ sizeof(RaymarchShaders::RayCB), D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
				if (FAILED(a_device->CreateBuffer(&rd, nullptr, &g_rayCb))) {
					logger::error("raymarch constant buffer creation failed; the fog pass is off");
					return false;
				}
			}

			D3D11_BLEND_DESC blend{};
			auto&            rt = blend.RenderTarget[0];
			rt.BlendEnable = TRUE;
			rt.SrcBlend = D3D11_BLEND_SRC_ALPHA;
			rt.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
			rt.BlendOp = D3D11_BLEND_OP_ADD;
			rt.SrcBlendAlpha = D3D11_BLEND_ZERO;
			rt.DestBlendAlpha = D3D11_BLEND_ONE;  // keep the scene's alpha channel as it was
			rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
			rt.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
			if (FAILED(a_device->CreateBlendState(&blend, &g_blend))) {
				logger::error("fog blend state creation failed");
				return false;
			}

			D3D11_RASTERIZER_DESC raster{};
			raster.FillMode = D3D11_FILL_SOLID;
			raster.CullMode = D3D11_CULL_NONE;
			raster.DepthClipEnable = TRUE;
			if (FAILED(a_device->CreateRasterizerState(&raster, &g_raster))) {
				logger::error("fog rasterizer state creation failed");
				return false;
			}

			D3D11_DEPTH_STENCIL_DESC depth{};
			depth.DepthEnable = FALSE;
			depth.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
			depth.DepthFunc = D3D11_COMPARISON_ALWAYS;
			if (FAILED(a_device->CreateDepthStencilState(&depth, &g_depthOff))) {
				logger::error("fog depth state creation failed");
				return false;
			}
			D3D11_SAMPLER_DESC sd{};
			sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
			sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
			sd.MaxLOD = D3D11_FLOAT32_MAX;
			if (FAILED(a_device->CreateSamplerState(&sd, &g_linear))) {
				logger::error("fog sampler creation failed");
				return false;
			}
			{
				// Probe resources; a failure only disables Probe=1.
				ID3DBlob* pb = Compile(RaymarchShaders::kProbeCS, "cs_5_0");
				bool      pok = pb && SUCCEEDED(a_device->CreateComputeShader(pb->GetBufferPointer(), pb->GetBufferSize(), nullptr, &g_probeCs));
				SafeRelease(pb);
				D3D11_BUFFER_DESC pcb{ 16, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0 };
				pok = pok && SUCCEEDED(a_device->CreateBuffer(&pcb, nullptr, &g_probeCb));
				D3D11_BUFFER_DESC sb{ kProbeRecords * 16u, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 16 };
				pok = pok && SUCCEEDED(a_device->CreateBuffer(&sb, nullptr, &g_probeBuf));
				pok = pok && SUCCEEDED(a_device->CreateUnorderedAccessView(g_probeBuf, nullptr, &g_probeUav));
				D3D11_BUFFER_DESC st{ kProbeRecords * 16u, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, 16 };
				for (auto*& b : g_probeStaging) {
					pok = pok && SUCCEEDED(a_device->CreateBuffer(&st, nullptr, &b));
				}
				if (!pok) {
					logger::error("probe resources failed; Probe=1 does nothing");
					SafeRelease(g_probeCs);
					SafeRelease(g_probeCb);
					SafeRelease(g_probeUav);
					SafeRelease(g_probeBuf);
					for (auto*& b : g_probeStaging) SafeRelease(b);
				}
			}

			g_timer.Create(a_device);
			g_windTimer.Create(a_device);
			logger::info("fog pass resources ready");
			return true;
		}

		bool Create(ID3D11Device* a_device)
		{
			const auto start = std::chrono::steady_clock::now();
			if (CreateResources(a_device)) {
				const auto [cached, asked] = ShaderCache::PrecompileCount();
				logger::info("fog resources created in {:.1f} ms on the render thread (background compile {:.0f} ms, {} of {} shaders cached)",
					std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), ShaderCache::PrecompileMs(), cached, asked);
				return true;
			}
			ReleaseAll();
			return false;
		}

		// Self-test probe: every VS/PS texture slot and PS constant buffer slot, wider than the guard,
		// so a binding the fog pass knocked out anywhere (e.g. D3D auto-unbinding an SRV) shows up.
		struct Probe
		{
			std::array<ID3D11ShaderResourceView*, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT> vsSrv{};
			std::array<ID3D11ShaderResourceView*, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT> psSrv{};
			std::array<ID3D11Buffer*, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT>       psCb{};

			void Capture(ID3D11DeviceContext* a_ctx)
			{
				a_ctx->VSGetShaderResources(0, static_cast<UINT>(vsSrv.size()), vsSrv.data());
				a_ctx->PSGetShaderResources(0, static_cast<UINT>(psSrv.size()), psSrv.data());
				a_ctx->PSGetConstantBuffers(0, static_cast<UINT>(psCb.size()), psCb.data());
			}
			[[nodiscard]] bool SameAs(const Probe& a_o) const { return vsSrv == a_o.vsSrv && psSrv == a_o.psSrv && psCb == a_o.psCb; }
			void Release()
			{
				for (auto*& p : vsSrv) SafeRelease(p);
				for (auto*& p : psSrv) SafeRelease(p);
				for (auto*& p : psCb) SafeRelease(p);
			}
		};

		bool EnsureHeightTexture(ID3D11Device* a_device, int a_size)
		{
			if (g_heightTex && g_heightSize == a_size) {
				return true;
			}
			SafeRelease(g_heightSrv);
			SafeRelease(g_heightTex);
			D3D11_TEXTURE2D_DESC desc{};
			desc.Width = static_cast<UINT>(a_size);
			desc.Height = static_cast<UINT>(a_size);
			desc.MipLevels = 1;
			desc.ArraySize = 1;
			desc.Format = DXGI_FORMAT_R32_FLOAT;
			desc.SampleDesc.Count = 1;
			desc.Usage = D3D11_USAGE_DEFAULT;
			desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			if (FAILED(a_device->CreateTexture2D(&desc, nullptr, &g_heightTex)) ||
				FAILED(a_device->CreateShaderResourceView(g_heightTex, nullptr, &g_heightSrv))) {
				logger::error("ground height texture {}x{} creation failed", a_size, a_size);
				SafeRelease(g_heightTex);
				return false;
			}
			g_heightSize = a_size;
			return true;
		}

		bool Ready(ID3D11Device* a_device)
		{
			if (g_failed) {
				return false;
			}
			if (!g_created) {
				g_created = Create(a_device);
				g_failed = !g_created;
			}
			return g_created;
		}

		// Fog colours in the space the scene is lit in. Settings::colorSpace 1 = Community Shaders Linear Lighting:
		// the installed CS converts weather colours with the piecewise sRGB EOTF (Common/TransferFunctions.hlsli,
		// ColorManagement.hlsli SRGBToWorking), each colour on its own before any mixing. 0 = the colours as given.
		float ToWorking(float a_c)
		{
			const int space = Settings::colorSpace >= 0 ? Settings::colorSpace : WeatherFog::AutoColorSpace();
			if (space != 1) {
				return a_c;
			}
			const float u = std::fabs(a_c);
			const float lin = u <= 0.04045f ? u / 12.92f : std::pow((u + 0.055f) / 1.055f, 2.4f);
			return std::copysign(lin, a_c);
		}

		RE::NiColor ToWorking(const RE::NiColor& a_c) { return { ToWorking(a_c.red), ToWorking(a_c.green), ToWorking(a_c.blue) }; }

		void CopyColour(float (&a_out)[4], const RE::NiColor& a_c, float a_scale)
		{
			a_out[0] = a_c.red * a_scale;
			a_out[1] = a_c.green * a_scale;
			a_out[2] = a_c.blue * a_scale;
			a_out[3] = 1.0f;
		}

		std::array<float, 2> g_lightLog{};   // sun elevation (degrees), sun weight in the fog colour: for the state line

		// Sun elevation in degrees (0 when unknown) and how much of a day it is (0 at night, 1 by day). The sun's light
		// points from the sun into the scene; Community Shaders' Sky Sync turns it to the moon at night, so the clock
		// decides day and night (as FillGlow).
		void SunHeight(float& a_elevationDeg, float& a_day)
		{
			a_elevationDeg = 0.0f;
			a_day = 0.0f;
			auto* sky = RE::Sky::GetSingleton();
			if (!sky || !sky->sun || !sky->sun->light) {
				return;
			}
			const auto& d = sky->sun->light->GetWorldDirection();
			const float len = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
			if (!(len > 1e-3f) || !std::isfinite(len)) {
				return;
			}
			float hour = 12.0f;
			if (auto* cal = RE::Calendar::GetSingleton()) {
				hour = cal->GetHour();
			}
			auto smooth = [](float a_e0, float a_e1, float a_x) {
				const float t = std::clamp((a_x - a_e0) / (a_e1 - a_e0), 0.0f, 1.0f);
				return t * t * (3.0f - 2.0f * t);
			};
			a_day = smooth(5.0f, 7.0f, hour) * (1.0f - smooth(18.5f, 20.5f, hour));
			a_elevationDeg = std::asin(std::clamp(-d.z / len, -1.0f, 1.0f)) * 57.29578f;
		}

		void FillLighting(FrameParams& a_cb)
		{
			auto* sky = RE::Sky::GetSingleton();
			if (!sky) {
				a_cb.fogColor[0] = a_cb.fogColor[1] = a_cb.fogColor[2] = ToWorking(0.6f) * Settings::brightness;
				a_cb.distFog[0] = 1.0e6f;
				a_cb.distFog[1] = 2.0e6f;
				return;
			}
			using Color = RE::TESWeather::ColorTypes;
			// Colour from the light that reaches the fog (FogLight.h): the sky's and the sun's chromaticities, the sun's
			// by its share of the light at this sun height where it reaches the layer. Brightness is kept as it was
			// tuned in game: the luminance of ambient x 0.65 + sunlight x 0.35 (light adds linearly, so the colours
			// are converted to the working space first).
			const auto  ambient = ToWorking(sky->skyColor[Color::kAmbient]);
			const auto  sun = ToWorking(sky->skyColor[Color::kSunlight]);
			constexpr float kBrightSun = 0.35f;
			const FogLight::Rgb skyRgb{ ambient.red, ambient.green, ambient.blue };
			const FogLight::Rgb sunRgb{ sun.red, sun.green, sun.blue };
			const float luminance = FogLight::Luminance(skyRgb) * (1.0f - kBrightSun) + FogLight::Luminance(sunRgb) * kBrightSun;
			float       elevation = 0.0f;
			float       day = 0.0f;
			SunHeight(elevation, day);
			const float weight = FogLight::SunWeight(elevation, day, Settings::sunMix);
			const auto  rgb = FogLight::Colour(skyRgb, sunRgb, weight, luminance);
			g_lightLog = { elevation, weight };
			CopyColour(a_cb.fogColor, RE::NiColor{ rgb[0], rgb[1], rgb[2] }, Settings::brightness);
			CopyColour(a_cb.distFogColorNear, ToWorking(sky->skyColor[Color::kFogNear]), 1.0f);
			CopyColour(a_cb.distFogColorFar, ToWorking(sky->skyColor[Color::kFogFar]), 1.0f);
			// GameFogScale stretches the weathers' fog distances; undo what was applied to the current weather so our
			// colour ramp keeps the record's own (weathers another mod changed are not scaled, GameFog::ScaleFor = 1).
			const float scale = std::max(GameFog::ScaleFor(sky->currentWeather), 0.01f);
			a_cb.distFog[0] = sky->fogNear / scale;
			a_cb.distFog[1] = std::max(sky->fogFar, sky->fogNear + 1.0f) / scale;
			a_cb.distFog[2] = Settings::fogPower;
			a_cb.distFog[3] = Settings::fogMax;
		}

		bool EnsureLowTarget(ID3D11Device* a_device, UINT a_w, UINT a_h)
		{
			if (g_lowTex && g_lowW == a_w && g_lowH == a_h) {
				return true;
			}
			SafeRelease(g_lowSrv);
			SafeRelease(g_lowRtv);
			SafeRelease(g_lowTex);
			D3D11_TEXTURE2D_DESC desc{};
			desc.Width = a_w;
			desc.Height = a_h;
			desc.MipLevels = desc.ArraySize = 1;
			desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;  // distances exceed the fp16 range (65504)
			desc.SampleDesc.Count = 1;
			desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
			if (FAILED(a_device->CreateTexture2D(&desc, nullptr, &g_lowTex)) ||
				FAILED(a_device->CreateRenderTargetView(g_lowTex, nullptr, &g_lowRtv)) ||
				FAILED(a_device->CreateShaderResourceView(g_lowTex, nullptr, &g_lowSrv))) {
				// Logged once by the caller (NoDraw::kTarget); the next frame tries again.
				SafeRelease(g_lowSrv);
				SafeRelease(g_lowRtv);
				SafeRelease(g_lowTex);
				return false;
			}
			g_lowW = a_w;
			g_lowH = a_h;
			return true;
		}

		// General 4x4 inverse (cofactors), row-major. Returns false for a singular matrix.
		bool Invert4x4(const float (&m)[16], float (&out)[16])
		{
			float inv[16];
			inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
			inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
			inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
			inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
			inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
			inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
			inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
			inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
			inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
			inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
			inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
			inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
			inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
			inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
			inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
			inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
			const float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
			if (std::fabs(det) < 1e-20f || !std::isfinite(det)) {
				return false;
			}
			for (int i = 0; i < 16; ++i) {
				out[i] = inv[i] / det;
			}
			return true;
		}

		void SetLookPitch(float a_pitch)
		{
			const std::uint32_t gen = g_sweepGen.load();
			if (const auto* tasks = SKSE::GetTaskInterface()) {
				tasks->AddTask([a_pitch, gen]() {
					if (g_sweepGen.load() != gen) {
						return;   // a game was loaded since: leave its pitch alone
					}
					if (auto* p = RE::PlayerCharacter::GetSingleton()) {
						p->SetLooking(a_pitch);
					}
				});
			}
		}

		void AbortSweep(const char* a_why)
		{
			if (g_sweepLoaded.load()) {
				return;   // a game was loaded: its pitch is not ours to restore; UpdateSweep re-arms
			}
			if (g_sweep.step >= 0 && !g_sweep.done) {
				SetLookPitch(g_sweep.original);
				g_sweep.done = true;
				logger::warn("[PROBE] sweep stopped at step {} ({}); look pitch put back to {:.3f}", g_sweep.step, a_why, g_sweep.original);
			}
		}

		// Advances the Probe=2 pitch sweep; called once per drawn frame before the fog draw. A step moves on only
		// after its probe was submitted; a pause in the fog draw or a missing probe stops the sweep and puts the
		// pitch back.
		void UpdateSweep(RE::PlayerCharacter* a_player)
		{
			if (g_sweepLoaded.exchange(false)) {
				g_sweep = {};   // the loaded game has its own pitch; arm again
			}
			if (Settings::probe != 2) {
				AbortSweep("Probe is no longer 2");
				g_sweep = {};
				return;
			}
			if (g_sweep.done) {
				return;
			}
			const auto now = std::chrono::steady_clock::now();
			if (g_sweep.step >= 0 && now - g_sweep.lastUpdate > kSweepMaxGap) {
				AbortSweep("the fog draw paused over 1 s: indoors, a menu or loading");
				return;
			}
			g_sweep.lastUpdate = now;
			if (g_sweep.step < 0) {
				if (g_sweep.startAt == 0) {
					g_sweep.startAt = g_stats.draws + kSweepWaitDraws;
					logger::info("[PROBE] sweep armed: starts after {} more drawn frames; do not move the mouse", kSweepWaitDraws);
				}
				if (g_stats.draws < g_sweep.startAt) {
					return;
				}
				g_sweep.original = a_player->GetAngleX();
				g_sweep.step = 0;
				g_sweep.stepStart = now;
				g_sweep.probed = false;
				logger::info("[PROBE] sweep start: look pitch {:.3f} saved, {} steps of {} ms", g_sweep.original, kSweepSteps,
					static_cast<long long>(kSweepHold.count()));
			} else if (now - g_sweep.stepStart >= kSweepHold) {
				if (!g_sweep.probed) {
					if (now - g_sweep.stepStart >= kSweepHold + kSweepMaxWait) {
						AbortSweep("no probe for this step");
					}
					return;   // keep holding this pitch until its probe is in
				}
				if (++g_sweep.step >= kSweepSteps) {
					SetLookPitch(g_sweep.original);
					g_sweep.done = true;
					logger::info("[PROBE] sweep done, look pitch put back to {:.3f}", g_sweep.original);
					return;
				}
				g_sweep.stepStart = now;
				g_sweep.probed = false;
			}
			SetLookPitch(kSweepPitch[g_sweep.step]);
		}

		UINT PresentCount()
		{
			auto* renderer = RE::BSGraphics::Renderer::GetSingleton();
			auto* chain = renderer ? reinterpret_cast<IDXGISwapChain*>(renderer->GetRuntimeData().renderWindows[0].swapChain) : nullptr;
			UINT  count = 0;
			if (!chain || FAILED(chain->GetLastPresentCount(&count))) {
				return 0;
			}
			return count;
		}

		// CPU facts for a probe: the viewport the game had bound at the hook, the targets, both camera sources.
		std::string ProbeHeader(ID3D11DeviceContext* a_ctx, RE::BSGraphics::RenderTargetData& a_main,
			RE::BSGraphics::DepthStencilData& a_depth, const FrameParams& a_cb, ProbeMeta& a_meta)
		{
			UINT           vpCount = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
			D3D11_VIEWPORT vps[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE]{};
			a_ctx->RSGetViewports(&vpCount, vps);
			D3D11_TEXTURE2D_DESC dd{};
			D3D11_TEXTURE2D_DESC sd{};
			if (a_depth.texture) reinterpret_cast<ID3D11Texture2D*>(a_depth.texture)->GetDesc(&dd);
			if (a_main.texture) reinterpret_cast<ID3D11Texture2D*>(a_main.texture)->GetDesc(&sd);
			D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
			if (a_depth.depthSRV) reinterpret_cast<ID3D11ShaderResourceView*>(a_depth.depthSRV)->GetDesc(&srvDesc);

			auto*        shadow = RE::BSGraphics::RendererShadowState::GetSingleton();
			const auto&  view = shadow->GetRuntimeData().cameraData.getEye();
			const float* v = reinterpret_cast<const float*>(&view.viewMat);
			const float* pr = reinterpret_cast<const float*>(&view.projMat);

			std::string skyFog = "none";
			if (auto* sky = RE::Sky::GetSingleton()) {
				using Color = RE::TESWeather::ColorTypes;
				const auto& fn = sky->skyColor[Color::kFogNear];
				const auto& ff = sky->skyColor[Color::kFogFar];
				skyFog = std::format("raw near ({:.3f}, {:.3f}, {:.3f}) far ({:.3f}, {:.3f}, {:.3f}), shader near ({:.3f}, {:.3f}, {:.3f}) far ({:.3f}, {:.3f}, {:.3f}), fog near {:.0f} far {:.0f}, ColorSpace {}",
					fn.red, fn.green, fn.blue, ff.red, ff.green, ff.blue, a_cb.distFogColorNear[0], a_cb.distFogColorNear[1], a_cb.distFogColorNear[2],
					a_cb.distFogColorFar[0], a_cb.distFogColorFar[1], a_cb.distFogColorFar[2], sky->fogNear, sky->fogFar, Settings::colorSpace);
			}
			std::string ni = "none";
			a_meta.haveNi = false;
			if (auto* cam = RE::Main::WorldRootCamera()) {
				const auto& t = cam->world.translate;
				const auto& r = cam->world.rotate;
				const auto& fr = cam->GetRuntimeData2().viewFrustum;
				for (int k = 0; k < 3; ++k) {
					a_meta.niCol0[k] = r.entry[k][0];
					a_meta.niRow0[k] = r.entry[0][k];
				}
				a_meta.haveNi = true;
				ni = std::format("pos ({:.1f}, {:.1f}, {:.1f}) col0 ({:.3f}, {:.3f}, {:.3f}) row0 ({:.3f}, {:.3f}, {:.3f}) frustum l {:.3f} r {:.3f} t {:.3f} b {:.3f} near {:.2f} far {:.0f} ortho {}",
					t.x, t.y, t.z, r.entry[0][0], r.entry[1][0], r.entry[2][0], r.entry[0][0], r.entry[0][1], r.entry[0][2],
					fr.fLeft, fr.fRight, fr.fTop, fr.fBottom, fr.fNear, fr.fFar, fr.bOrtho);
			}
			return std::format("hookVP {}: ({:.0f}, {:.0f}, {:.0f} x {:.0f}, depth {:.2f}..{:.2f}) | depth tex {}x{} fmt {} samples {} srvFmt {} srvDim {} | scene tex {}x{} fmt {} | posAdjust ({:.1f}, {:.1f}, {:.1f}) | NiCamera {} | viewMat fwd col2 ({:.3f}, {:.3f}, {:.3f}) row3 ({:.1f}, {:.1f}, {:.1f}) | proj _11 {:.4f} _22 {:.4f} _31 {:.5f} _32 {:.5f} _33 {:.7f} _34 {:.3f} _43 {:.4f} _44 {:.3f} | used viewProj row2 ({:.4f}, {:.4f}, {:.4f}, {:.4f}) | sky fog {}",
				vpCount, vps[0].TopLeftX, vps[0].TopLeftY, vps[0].Width, vps[0].Height, vps[0].MinDepth, vps[0].MaxDepth,
				dd.Width, dd.Height, static_cast<int>(dd.Format), dd.SampleDesc.Count, static_cast<int>(srvDesc.Format), static_cast<int>(srvDesc.ViewDimension),
				sd.Width, sd.Height, static_cast<int>(sd.Format),
				a_cb.posAdjust[0], a_cb.posAdjust[1], a_cb.posAdjust[2], ni,
				v[2], v[6], v[10], v[12], v[13], v[14],
				pr[0], pr[5], pr[8], pr[9], pr[10], pr[11], pr[14], pr[15],
				a_cb.viewProj[8], a_cb.viewProj[9], a_cb.viewProj[10], a_cb.viewProj[11], skyFog);
		}

		// One probe dispatch (phase 0 before the composite, 1 after). Targets are unbound so the scene and the
		// low-res march can be read; the caller's StateGuard restores them, ComputeGuard restores the compute
		// bindings. Debug only.
		bool RunProbe(ID3D11DeviceContext* a_ctx, float a_phase, ID3D11ShaderResourceView* a_depthSrv, ID3D11ShaderResourceView* a_sceneSrv)
		{
			const ComputeGuard guard(a_ctx);
			a_ctx->OMSetRenderTargets(0, nullptr, nullptr);
			D3D11_MAPPED_SUBRESOURCE mapped{};
			if (FAILED(a_ctx->Map(g_probeCb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
				return false;
			}
			const float phase[4]{ a_phase, 0.0f, 0.0f, 0.0f };
			std::memcpy(mapped.pData, phase, sizeof(phase));
			a_ctx->Unmap(g_probeCb, 0);
			ID3D11ShaderResourceView* srv[6]{ a_depthSrv, WindField::ColumnSrv(), g_lowSrv, a_sceneSrv, WindField::UpperSrv(), g_coarseSrv };   // RayCB poolFar.x = 0 switches the far pools off
			ID3D11Buffer*             cbs[2]{ g_rayCb, g_probeCb };
			a_ctx->CSSetShader(g_probeCs, nullptr, 0);
			a_ctx->CSSetConstantBuffers(0, 2, cbs);
			a_ctx->CSSetShaderResources(0, 6, srv);
			a_ctx->CSSetSamplers(0, 1, &g_linear);
			a_ctx->CSSetUnorderedAccessViews(0, 1, &g_probeUav, nullptr);
			a_ctx->Dispatch(1, 1, 1);
			return true;
		}

		float AngleDeg(const float* a_a, const float* a_b)
		{
			const float la = std::sqrt(a_a[0] * a_a[0] + a_a[1] * a_a[1] + a_a[2] * a_a[2]);
			const float lb = std::sqrt(a_b[0] * a_b[0] + a_b[1] * a_b[1] + a_b[2] * a_b[2]);
			if (la < 1e-6f || lb < 1e-6f) {
				return -1.0f;
			}
			const float c = std::clamp((a_a[0] * a_b[0] + a_a[1] * a_b[1] + a_a[2] * a_b[2]) / (la * lb), -1.0f, 1.0f);
			return std::acos(c) * 57.29578f;
		}

		void LogProbe(const ProbeMeta& a_meta, const std::vector<float>& a_v)
		{
			auto rec = [&](int a_point, int a_slot) { return &a_v[static_cast<std::size_t>((a_point * kProbeSlots + a_slot) * 4)]; };
			const float* extra = &a_v[static_cast<std::size_t>(kProbePoints * kProbeSlots * 4)];
			logger::info("[PROBE] #{} sweep step {} commanded pitch {:.3f} player angleX {:.3f}", a_meta.id, a_meta.sweepStep,
				a_meta.sweepPitch, a_meta.playerPitch);
			logger::info("[PROBE] #{} frame {} (+{} frames, +{} presents, +{} draws since the last probe) {}",
				a_meta.id, a_meta.frame, a_meta.framesSince, a_meta.presentsSince, a_meta.drawsSince, a_meta.header);
			logger::info("[PROBE] #{} eyeRel ({:.2f}, {:.2f}, {:.2f}) box z {:.0f}..{:.0f} maxDist {:.0f} steps {:.0f} opacity {:.3f}",
				a_meta.id, extra[0], extra[1], extra[2], extra[3], extra[4], extra[5], extra[6], extra[7]);

			const float        opacity = extra[7];
			const float        wMax = a_meta.wMax;
			int                overBound = 0;
			int                groundLike = 0;
			int                farFogged = 0;
			float              maxErr = 0.0f;
			std::vector<float> heights;
			for (int i = 0; i < kProbePoints; ++i) {
				const float* r0 = rec(i, 0);
				const float* r1 = rec(i, 1);
				const float* r2 = rec(i, 2);
				const float* r3 = rec(i, 3);
				const float* r4 = rec(i, 4);
				const float* r5 = rec(i, 5);
				const float* r6 = rec(i, 6);
				const float* r7 = rec(i, 7);
				const float* r8 = rec(i, 8);
				const float  a = r5[3];
				float        err = 0.0f;
				for (int c = 0; c < 3; ++c) {
					err = std::max(err, std::fabs(r7[c] - (a * r5[c] + (1.0f - a) * r6[c])));
				}
				maxErr = std::max(maxErr, err);
				// Most fog a steep ray can collect in one column: Opacity * Wmax / |dir.z|. Only judged where |dir.z| >= 0.3:
				// shallow rays cross many columns of different ground height, where one column is no limit (bound 1).
				const float bound = std::fabs(r1[2]) >= 0.3f ? 1.0f - std::exp(-opacity * wMax / std::fabs(r1[2])) : 1.0f;
				if (r4[0] > bound + 0.02f) {   // near field alone; the far and height tiers add on top
					++overBound;
				}
				const bool  known = r2[3] > -1e8f;   // in the column field, ground known, scene point finite (shader)
				const float h = known ? r2[2] - r2[3] : 0.0f;
				if (known) {
					heights.push_back(h);
				}
				logger::info("[PROBE] #{} p{:02} px ({:.0f}, {:.0f}) d {:.7f} dist {:.0f} dir ({:.3f}, {:.3f}, {:.3f}) z {:.0f} ground {:.0f} h {} | col H {:.0f} W {:.2f} k {:.2f} | march {:.3f} comp {:.3f} bound {:.3f} | src ({:.2f}, {:.2f}, {:.2f}) before ({:.3f}, {:.3f}, {:.3f}) after ({:.3f}, {:.3f}, {:.3f}) blendErr {:.3f} | far nz {:.2f} tau {:.3f} share {:.2f} heightTau {:.3f} poolTau {:.3f}",
					a_meta.id, i, r0[0], r0[1], r0[2], r0[3], r1[0], r1[1], r1[2], r2[2], r2[3], known ? std::format("{:.0f}", h) : std::string("unknown"),
					r3[0], r3[1], r3[3], r4[0], a, bound, r5[0], r5[1], r5[2], r6[0], r6[1], r6[2], r7[0], r7[1], r7[2], err,
					r8[0], r8[1], r8[2], r8[3], r1[3]);
				if (r8[0] >= 0.7f) {
					++groundLike;
				}
				if (r8[1] > 1e-3f) {
					++farFogged;
				}
			}
			float median = 0.0f;
			if (!heights.empty()) {
				std::nth_element(heights.begin(), heights.begin() + static_cast<std::ptrdiff_t>(heights.size() / 2), heights.end());
				median = heights[heights.size() / 2];
			}
			constexpr int kCentre = (RaymarchShaders::kProbeRows / 2) * RaymarchShaders::kProbeCols + RaymarchShaders::kProbeCols / 2;
			const float*  centreDir = rec(kCentre, 1);
			const float   angCol = a_meta.haveNi ? AngleDeg(centreDir, a_meta.niCol0) : -1.0f;
			const float   angRow = a_meta.haveNi ? AngleDeg(centreDir, a_meta.niRow0) : -1.0f;
			logger::info("[PROBE] #{} summary: comp alpha over the one-column bound at {} of {} points (|dir.z| >= 0.3 only), blend error max {:.3f}, scene height above ground median {:.0f} units over {} points, centre ray vs NiCamera col0 {:.1f} deg / row0 {:.1f} deg; far tier: {} points ground-like, {} fogged",
				a_meta.id, overBound, kProbePoints, maxErr, median, heights.size(), angCol, angRow, groundLike, farFogged);
		}

		// Reads back finished probes without stalling (DO_NOT_WAIT); called from every Draw.
		void CollectProbes(ID3D11DeviceContext* a_ctx)
		{
			for (int k = 0; k < kProbeRing; ++k) {
				auto& meta = g_probeMeta[k];
				if (!meta.pending || !g_probeStaging[k]) {
					continue;
				}
				D3D11_MAPPED_SUBRESOURCE mapped{};
				const HRESULT hr = a_ctx->Map(g_probeStaging[k], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped);
				if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
					continue;
				}
				meta.pending = false;
				if (FAILED(hr)) {
					logger::warn("[PROBE] #{} readback failed (hr {:08X})", meta.id, static_cast<std::uint32_t>(hr));
					continue;
				}
				std::vector<float> values(static_cast<std::size_t>(kProbeRecords) * 4);
				std::memcpy(values.data(), mapped.pData, values.size() * sizeof(float));
				a_ctx->Unmap(g_probeStaging[k], 0);
				LogProbe(meta, values);
			}
		}

		void UpdateRegion(const RE::TESWorldSpace* a_space)
		{
			if (g_regionKnown && a_space == g_regionSpace) {
				return;
			}
			if (a_space != g_regionSpace) {
				// The region kept for a forced weather belongs to the worldspace just left: coming back to another part of
				// it must not bring that region back (GPT review 2026-11-11). Interiors return before this function, so a
				// visit indoors keeps it.
				g_geoHeld = nullptr;
				g_geoHeldSpace = nullptr;
			}
			g_regionSpace = a_space;
			g_regionKnown = true;
			if (!g_regionIniRead) {
				g_regionIniRead = true;
				for (const auto& [key, value] : Settings::Section("Worldspaces")) {
					WorldspaceRules::Rule rule;
					if (!WorldspaceRules::ParseRule(value, rule)) {
						logger::warn("[Worldspaces] {} = '{}' is not off, default, a number >= 0 or 'always <number>'; skipped", key, value);
						continue;
					}
					g_regionIni.push_back({ WeatherRules::NormalizeOverrideKey(key), rule });
				}
			}
			constexpr std::size_t kMaxParents = 8;   // parent links are one or two deep in the game's data; a loop would not end
			std::vector<WorldspaceRules::Space> chain;
			for (const auto* w = a_space; w && chain.size() < kMaxParents; w = w->parentWorld) {
				WorldspaceRules::Space s;
				s.editorID = EditorIDOf(w);
				if (const auto* file = w->GetFile(0)) {
					s.plugin = std::string(file->GetFilename());
				}
				s.localFormID = w->GetLocalFormID();
				chain.push_back(std::move(s));
			}
			std::string how;
			g_regionRule = WorldspaceRules::Resolve(chain, g_regionIni, WorldspaceRules::BuiltIn(), &how);
			const std::string name = chain.empty() ? std::string("(none)") : chain.front().editorID.empty() ? WeatherRules::FormKey(chain.front().plugin, chain.front().localFormID) : chain.front().editorID;
			g_regionText = std::format("{}: {}{} from {}", name, WorldspaceRules::KindName(g_regionRule.kind),
				g_regionRule.kind == WorldspaceRules::Kind::kScale || g_regionRule.kind == WorldspaceRules::Kind::kAlways ? std::format(" {:.2f}", g_regionRule.value) : std::string(), how);
			logger::info("worldspace {}", g_regionText);
		}

		// The region the sky is taking its weather from (Sky::region) and its [Regions] value. Once per drawn frame.
		void UpdateGeo()
		{
			const auto* sky = RE::Sky::GetSingleton();
			const RE::TESRegion* region = sky ? sky->region : nullptr;
			bool held = false;
			const auto* player = RE::PlayerCharacter::GetSingleton();
			const RE::NiPoint3 pos = player ? player->GetPosition() : RE::NiPoint3{};
			if (region) {
				g_geoHeld = region;
				g_geoHeldSpace = g_regionSpace;
				g_geoHeldPos = pos;
			} else if (g_geoHeld && g_geoHeldSpace == g_regionSpace) {
				const float dx = pos.x - g_geoHeldPos.x, dy = pos.y - g_geoHeldPos.y;
				if (player && dx * dx + dy * dy <= kGeoHeldMaxDistance * kGeoHeldMaxDistance) {
					region = g_geoHeld;
					held = true;
				} else {
					g_geoHeld = nullptr;   // moved away without the sky naming a region: the old one no longer describes the place
				}
			}
			const auto now = std::chrono::steady_clock::now();
			// Also after a gap without drawn frames: fast travel, a door or a load screen put the player in another region
			// at once, and easing from the old value showed as fog thinning for seconds after arriving (reviewer 2026-11-11).
			const bool first = !g_geoKnown || std::chrono::duration<float>(now - g_geoTime).count() > kGeoGapSeconds;
			if (first || region != g_geoRegion || held != g_geoWasHeld) {
				g_geoRegion = region;
				g_geoWasHeld = held;
				g_geoKnown = true;
				if (!g_geoIniRead) {
					g_geoIniRead = true;
					for (const auto& [key, value] : Settings::Section("Regions")) {
						float add = 0.0f;
						bool  clamped = false;
						if (!RegionRules::ParseAdd(value, add, &clamped)) {
							logger::warn("[Regions] {} = '{}' is not a number or 'default'; skipped", key, value);
							continue;
						}
						if (clamped) {
							logger::warn("[Regions] {} = '{}' is outside {} .. {}; using {}", key, value, RegionRules::kMinAdd, RegionRules::kMaxAdd, add);
						}
						g_geoIni.push_back(RegionRules::IniEntry(key, add));
					}
				}
				std::string how = "none";
				std::string name = "(no region)";
				g_geoTarget = 0.0f;
				g_geoFarTarget = 0.0f;
				if (region) {
					RegionRules::Region r;
					r.editorID = EditorIDOf(region);
					if (const auto* file = region->GetFile(0)) {
						r.plugin = std::string(file->GetFilename());
					}
					r.localFormID = region->GetLocalFormID();
					g_geoTarget = RegionRules::AddFor(r, g_geoIni, RegionRules::BuiltIn(), &how, &g_geoFarTarget);
					name = r.editorID.empty() ? WeatherRules::FormKey(r.plugin, r.localFormID) : r.editorID + " (" + WeatherRules::FormKey(r.plugin, r.localFormID) + ")";
				}
				g_geoText = std::format("{}{}: {:+.2f} (far fog {:+.2f}) from {}", name, held ? " (kept: the sky names none now)" : "", g_geoTarget,
					g_geoFarTarget, how);
				logger::info("weather region {}", g_geoText);
			}
			if (first) {
				g_geo = g_geoTarget;   // a loaded game or a changed ini starts at its own value
				g_geoFar = g_geoFarTarget;
			} else {
				const float dt = std::min(std::chrono::duration<float>(now - g_geoTime).count(), 1.0f);
				const float k = 1.0f - std::exp(-std::max(dt, 0.0f) / kGeoEaseSeconds);
				g_geo += (g_geoTarget - g_geo) * k;
				g_geoFar += (g_geoFarTarget - g_geoFar) * k;
			}
			g_geoTime = now;
		}

		// The region value in use: none in an "always" worldspace, whose amount is fixed (and where the sky may still hold
		// the region the player came from: which one is not known, reviewer 2026-11-11).
		float GeoNow() { return g_regionRule.kind == WorldspaceRules::Kind::kAlways ? 0.0f : g_geo; }
		float GeoFarNow() { return g_regionRule.kind == WorldspaceRules::Kind::kAlways ? 0.0f : g_geoFar; }

		// This frame's fog conditions from the clock, the sky and (when installed) Survival Mode Improved.
		void UpdateConditions()
		{
			g_cond = {};
			if (!Settings::conditions) {
				return;
			}
			FogConditions::Sky in;
			if (auto* cal = RE::Calendar::GetSingleton()) {
				in.hour = cal->GetHour();
				in.month = static_cast<int>(cal->GetMonth());
			}
			if (auto* sky = RE::Sky::GetSingleton()) {
				// Gusts (Sky::windSpeed changes every few seconds) must not pump the fog: wind thins radiation fog over
				// minutes as it mixes the layer (in game 2026-11-08: global 0.49..0.93 between log lines 5 s apart, the fog
				// came and went within seconds). Low-pass with kConditionWindSeconds; the wind field still moves with the
				// raw wind.
				// A load (Reset) or a gap of more than 2 s (the loading screen of a door or worldspace change, conditions switched
				// off) starts again from the current wind (GPT review 2026-11-08: the old calm value took about 28 s to
				// let a windy save's wind thin the fog).
				constexpr float kConditionWindSeconds = 60.0f;
				const auto  now = std::chrono::steady_clock::now();
				const float gap = std::chrono::duration<float>(now - g_condWindTime).count();
				g_condWindTime = now;
				const float raw = std::isfinite(sky->windSpeed) ? sky->windSpeed : 0.0f;
				// For 10 s after that it follows the raw wind: the sky reports 0 for the first frames after a load and
				// then the weather's wind (in game 2026-11-08: the filter started at 0 and took over a minute to rise).
				constexpr float kConditionWindSnapSeconds = 10.0f;
				if (g_condWind < 0.0f || !(gap >= 0.0f && gap <= 2.0f)) {
					g_condWindSnapUntil = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<float>(kConditionWindSnapSeconds));
				}
				if (now < g_condWindSnapUntil) {
					g_condWind = raw;
				} else {
					g_condWind += (raw - g_condWind) * (1.0f - std::exp(-gap / kConditionWindSeconds));
				}
				in.windSpeed = g_condWind;
				if (const auto* w = sky->currentWeather) {
					const auto f = w->data.flags;
					in.pleasant = f.any(RE::TESWeather::WeatherDataFlag::kPleasant);
					in.cloudy = f.any(RE::TESWeather::WeatherDataFlag::kCloudy);
					in.rainy = f.any(RE::TESWeather::WeatherDataFlag::kRainy);
					in.snow = f.any(RE::TESWeather::WeatherDataFlag::kSnow);
				}
			}
			if (Settings::condUseSurvival) {
				// Survival Mode Improved writes its ambient temperature here (NeedCold.h; blizzard = 450).
				// SMI stops updating it while survival mode is off, so the value only counts when Survival_ModeEnabled
				// (the Creation Club global SMI reads, Utility.h IsSurvivalEnabled) is 1.
				static RE::TESGlobal* const smi = RE::TESForm::LookupByEditorID<RE::TESGlobal>("SMI_CurrentAmbientTemp");
				static RE::TESGlobal* const enabled = RE::TESForm::LookupByEditorID<RE::TESGlobal>("Survival_ModeEnabled");
				if (smi && enabled && enabled->value == 1.0f && std::isfinite(smi->value)) {
					in.survivalCold = std::clamp(smi->value / 450.0f, 0.0f, 1.0f);
				}
			}
			FogConditions::Params p;
			p.nightLevel = Settings::condNight;
			p.morningPeak = Settings::condMorningPeak;
			p.afternoonLevel = Settings::condAfternoon;
			p.windLow = Settings::condWindLow;
			p.windHigh = Settings::condWindHigh;
			p.windSuppress = Settings::condWindSuppress;
			p.evaporation = Settings::condEvaporation;
			g_condSky = in;
			g_cond = FogConditions::Compute(in, p);
			if (g_regionRule.kind == WorldspaceRules::Kind::kAlways) {
				g_cond = {};   // a fixed amount: no time of day, wind or season, no steam over cold water (hot springs keep theirs)
			}
		}

		// Radiation-fog strength after evaporation (global, wind included): the pooled fog's density scale, and its
		// lid fill (0 = at the basin floor, 1 = at the rim) clamped to 0..1.
		float RadiationStrength() { return std::max(g_cond.global - g_cond.evaporation, 0.0f); }
		float PoolFill() { return std::clamp(RadiationStrength(), 0.0f, 1.0f); }
		// Pool density this frame, near and far: PoolDensity x strength, held to the visibility floor. CSColumn scales the
		// near pools by up to 1 + 0.6 x WispAmount, so the floor counts that peak (GPT review 2026-11-08: 40 m without it);
		// the far pools take the same density, so the two still meet at the field's edge.
		float PoolDensityNow()
		{
			return PlacementRules::LimitPoolDensity(Settings::placePoolDensity * std::min(RadiationStrength(), 2.0f),
				Settings::opacity * g_weatherFactor * (1.0f + 0.6f * Settings::wispAmount), RaymarchShaders::kUpperSteepnessCpp, Settings::placePoolMinVisibility);
		}

		// The placement grid is built on a worker thread (PlacementRules::Build with pools took 2.2 ms on the render
		// thread). One job at a time on copies of the inputs; requests made while it runs merge into the next one. The
		// texture keeps the grid origin it was built for (PlaceGrid in PlaceCB), so CSColumn reads it at the right
		// world spots while a newer build is on its way.
		struct PlaceJob
		{
			std::vector<std::array<float, 4>> grid;
			float                             originX{ 0.0f };
			float                             originY{ 0.0f };
			float                             cell{ 64.0f };
			int                               size{ 0 };
			std::uint32_t                     generation{ 0 };
			float                             fill{ 1.0f };
			// Coarse pools (wide valleys), built in the same job when its inputs are set (coarseSize > 0).
			std::vector<std::array<float, 4>> coarse;   // x = ground (or unknown), w = pool depth
			float                             coarseOriginX{ 0.0f };
			float                             coarseOriginY{ 0.0f };
			int                               coarseSize{ 0 };
			std::uint32_t                     coarseGeneration{ 0 };
		};
		std::future<PlaceJob> g_placeJob;            // the running build, if any
		float                 g_placeReqOriginX{ 0.0f };   // inputs of the last requested build
		float                 g_placeReqOriginY{ 0.0f };
		std::uint32_t         g_placeReqGeneration{ 0 };
		std::array<float, 4>  g_placeGrid{ 0.0f, 0.0f, 0.0f, 0.0f };   // uploaded texture: origin x, y, cell, size
		std::uint32_t             g_coarseGen{ 0 };          // Terrain::CoarseGeneration of the uploaded coarse texture
		std::vector<float>        g_coarseEased;             // eased coarse depths
		std::vector<std::array<float, 4>> g_coarseTarget;    // last applied coarse build before easing (easing-only steps)
		float                     g_coarseOriginTX{ 0.0f };  // its origin
		float                     g_coarseOriginTY{ 0.0f };
		std::uint32_t             g_coarseGenT{ 0 };
		float                     g_coarseTop{ -1.0e30f };   // highest coarse pool top inside the field (march box)
		float                     g_coarseLidMax{ -1.0e30f };   // highest coarse pool top anywhere (far pools)

		bool PlaceJobRunning() { return g_placeJob.valid(); }
		std::vector<std::array<float, 4>> g_placeTarget;   // last applied build before easing (for easing-only steps)

		// The uploaded placement belongs to this terrain (a worldspace change or reset bumps the generation; until a
		// new build lands, nothing from the old world is used; GPT review 2026-11-08).
		bool PlaceUsable() { return Settings::placement && g_placeSrv && g_placeGrid[2] > 0.0f && g_placeGeneration == Terrain::Generation(); }

		// The region value for the field around the player. Its built-in number has the water-and-terrain overlap taken
		// out; without a usable placement grid ([Placement] Enable = 0, or not built yet) nothing overlaps, so the far
		// value applies there too (GPT review 2026-11-11).
		float GeoNearNow() { return PlaceUsable() ? GeoNow() : GeoFarNow(); }

		// Takes a finished build: eases its pools from the last uploaded ones (shifted by the grid move between the two
		// builds) and uploads it with its own origin.
		void ApplyPlaceJob(ID3D11DeviceContext* a_ctx, PlaceJob& a_job, std::chrono::steady_clock::time_point a_now)
		{
			// A job without a coarse part (coarse build deferred) keeps easing the last coarse target, so the coarse pools do
			// not stop half way (GPT review 2026-11-08).
			if (a_job.coarseSize == 0 && !g_coarseTarget.empty() && g_coarseGenT == Terrain::CoarseGeneration()) {
				a_job.coarse = g_coarseTarget;
				a_job.coarseOriginX = g_coarseOriginTX;
				a_job.coarseOriginY = g_coarseOriginTY;
				a_job.coarseSize = Terrain::kCoarseSize;
				a_job.coarseGeneration = g_coarseGenT;
			}
			g_placeTarget = a_job.grid;
			if (a_job.coarseSize > 0) {
				g_coarseTarget = a_job.coarse;
				g_coarseOriginTX = a_job.coarseOriginX;
				g_coarseOriginTY = a_job.coarseOriginY;
				g_coarseGenT = a_job.coarseGeneration;
			}
			// At most one throttle period per step: time spent settled before a new target must not count toward it
			// (GPT review 2026-11-08: after 10 s at rest the first step took 63% of a change at once). Taken once, before
			// the build time is updated (the coarse part read it after and always got 0; GPT review 2026-11-08).
			const float dt = std::min(std::chrono::duration<float>(a_now - g_placeBuilt).count(), 0.5f);
			{
				constexpr float kPoolEaseSeconds = 2.0f;
				const bool  same = g_placeSize == a_job.size && g_placeGeneration == a_job.generation && g_poolEased.size() == a_job.grid.size();
				const int   dx = same ? static_cast<int>(std::lround((a_job.originX - g_placeOriginX) / a_job.cell)) : 0;
				const int   dy = same ? static_cast<int>(std::lround((a_job.originY - g_placeOriginY) / a_job.cell)) : 0;
				if (!same) g_poolEased.clear();
				const float left = PlacementRules::EasePools(g_poolEased, a_job.size, dx, dy, a_job.grid, 1.0f - std::exp(-std::max(dt, 0.0f) / kPoolEaseSeconds));
				g_poolSettling = left > 1.0f;
			}
			a_ctx->UpdateSubresource(g_placeTex, 0, nullptr, a_job.grid.data(), static_cast<UINT>(a_job.size * sizeof(float) * 4), 0);
			g_placeMax = { 0.0f, 0.0f, 0.0f };
			double      sum = 0.0;
			std::size_t used = 0;
			for (const auto& c : a_job.grid) {
				for (std::size_t ch = 0; ch < 3; ++ch) g_placeMax[ch] = std::max(g_placeMax[ch], c[ch]);
				if (c[0] > 0.0f) {
					sum += c[0];
					++used;
				}
			}
			g_placeMean = used ? static_cast<float>(sum / static_cast<double>(used)) : 0.0f;
			g_placeBuilt = a_now;
			g_placeOriginX = a_job.originX;
			g_placeOriginY = a_job.originY;
			g_placeGeneration = a_job.generation;
			g_placeFill = a_job.fill;
			g_placeGrid = { a_job.originX, a_job.originY, a_job.cell, static_cast<float>(a_job.size) };
			if (a_job.coarseSize > 0 && g_coarseTex) {
				const bool  same = g_coarseGen == a_job.coarseGeneration && g_coarseEased.size() == a_job.coarse.size();
				const int   dx = same ? static_cast<int>(std::lround((a_job.coarseOriginX - g_coarseGrid[0]) / Terrain::kCoarseCell)) : 0;
				const int   dy = same ? static_cast<int>(std::lround((a_job.coarseOriginY - g_coarseGrid[1]) / Terrain::kCoarseCell)) : 0;
				if (!same) g_coarseEased.clear();
				const float coarseLeft = PlacementRules::EasePools(g_coarseEased, a_job.coarseSize, dx, dy, a_job.coarse, 1.0f - std::exp(-std::max(dt, 0.0f) / 2.0f));
				g_poolSettling = g_poolSettling || coarseLeft > 1.0f;
				std::vector<float> rg(a_job.coarse.size() * 2);
				for (std::size_t i = 0; i < a_job.coarse.size(); ++i) {
					const bool knownCell = Terrain::IsKnown(a_job.coarse[i][0]);
					rg[i * 2] = knownCell ? a_job.coarse[i][0] : Terrain::kUnknown;   // CSColumn needs all four corners known
					rg[i * 2 + 1] = knownCell ? a_job.coarse[i][3] : 0.0f;
				}
				a_ctx->UpdateSubresource(g_coarseTex, 0, nullptr, rg.data(), static_cast<UINT>(a_job.coarseSize * sizeof(float) * 2), 0);
				g_coarseGrid = { a_job.coarseOriginX, a_job.coarseOriginY, Terrain::kCoarseCell, static_cast<float>(a_job.coarseSize) };
				g_coarseGen = a_job.coarseGeneration;
				// Highest pool top inside the near field (the march box must hold it: a coarse lid can come from rims
				// outside the field, above every fine height there).
				g_coarseTop = -1.0e30f;
				g_coarseLidMax = -1.0e30f;
				const float fx0 = g_placeGrid[0], fy0 = g_placeGrid[1], fx1 = fx0 + g_placeGrid[2] * (g_placeGrid[3] - 1.0f), fy1 = fy0 + g_placeGrid[2] * (g_placeGrid[3] - 1.0f);
				// The shader mixes the four corners' lids (ground + depth, also of corners without a pool) wherever the mix
				// has a pool, so every cell next to a pooled one counts (GPT review 2026-11-08).
				const int cs = a_job.coarseSize;
				auto pooled = [&](int a_x, int a_y) {
					return a_x >= 0 && a_y >= 0 && a_x < cs && a_y < cs && rg[(static_cast<std::size_t>(a_y) * cs + a_x) * 2 + 1] >= 1.0f;
				};
				for (int y = 0; y < cs; ++y) {
					for (int x = 0; x < cs; ++x) {
						const float wx = a_job.coarseOriginX + x * Terrain::kCoarseCell, wy = a_job.coarseOriginY + y * Terrain::kCoarseCell;
						bool touches = false;
						for (int dy2 = -1; dy2 <= 1 && !touches; ++dy2)
							for (int dx2 = -1; dx2 <= 1 && !touches; ++dx2) touches = pooled(x + dx2, y + dy2);
						const std::size_t i = static_cast<std::size_t>(y) * cs + x;
						if (!touches || !Terrain::IsKnown(rg[i * 2])) continue;
						const float top = rg[i * 2] + rg[i * 2 + 1];
						// Far pools mix only corners with a pool (CoarsePoolAt: weight x saturate(depth), so a fading depth
						// between 0 and 1 counts too; GPT review 2026-11-08), so those cells bound them (a neighbour without a
						// pool can be a ridge; reviewer 2026-11-08).
						if (rg[i * 2 + 1] > 0.0f) g_coarseLidMax = std::max(g_coarseLidMax, top);
						if (wx < fx0 - Terrain::kCoarseCell || wx > fx1 + Terrain::kCoarseCell || wy < fy0 - Terrain::kCoarseCell || wy > fy1 + Terrain::kCoarseCell) continue;
						g_coarseTop = std::max(g_coarseTop, top);
					}
				}
			}
		}

		bool CoarseUsable() { return Settings::coarsePools && PlaceUsable() && g_coarseSrv && g_coarseGrid[2] > 0.0f && g_coarseGen == Terrain::CoarseGeneration(); }

		// Waits for a running build and drops it (resource release, reset).
		void DropPlaceJob()
		{
			if (g_placeJob.valid()) {
				g_placeJob.wait();
				g_placeJob = {};
			}
		}

		// The coarse pool texture (created once).
		bool EnsureCoarseTexture(ID3D11Device* a_device)
		{
			if (g_coarseTex) return true;
			D3D11_TEXTURE2D_DESC d{};
			d.Width = d.Height = static_cast<UINT>(Terrain::kCoarseSize);
			d.MipLevels = d.ArraySize = 1;
			d.Format = DXGI_FORMAT_R32G32_FLOAT;
			d.SampleDesc.Count = 1;
			d.Usage = D3D11_USAGE_DEFAULT;
			d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			if (FAILED(a_device->CreateTexture2D(&d, nullptr, &g_coarseTex)) || FAILED(a_device->CreateShaderResourceView(g_coarseTex, nullptr, &g_coarseSrv))) {
				logger::error("coarse pool texture creation failed; wide-valley pools off");
				SafeRelease(g_coarseTex);
				return false;
			}
			return true;
		}

		// Rebuilds the placement grid from the terrain kinds and heights (at most twice a second while they change).
		void UpdatePlacement(ID3D11Device* a_device, ID3D11DeviceContext* a_ctx, const Terrain::Grid& a_grid)
		{
			if (!Settings::placement) {
				return;
			}
			const auto now = std::chrono::steady_clock::now();
			if (!g_placeTex || g_placeSize != a_grid.size) {
				DropPlaceJob();
				SafeRelease(g_placeSrv);
				SafeRelease(g_placeTex);
				D3D11_TEXTURE2D_DESC d{};
				d.Width = d.Height = static_cast<UINT>(a_grid.size);
				d.MipLevels = d.ArraySize = 1;
				d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
				d.SampleDesc.Count = 1;
				d.Usage = D3D11_USAGE_DEFAULT;
				d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
				if (FAILED(a_device->CreateTexture2D(&d, nullptr, &g_placeTex)) || FAILED(a_device->CreateShaderResourceView(g_placeTex, nullptr, &g_placeSrv))) {
					logger::error("placement texture {}x{} creation failed; placement off", a_grid.size, a_grid.size);
					SafeRelease(g_placeTex);
					return;
				}
				g_placeSize = a_grid.size;
				g_placeGrid = { 0.0f, 0.0f, 0.0f, 0.0f };   // nothing uploaded yet: CSColumn reads no placement
				std::vector<std::array<float, 4>> blank(static_cast<std::size_t>(a_grid.size) * a_grid.size, { 1.0f, 0.0f, 0.0f, 0.0f });
				a_ctx->UpdateSubresource(g_placeTex, 0, nullptr, blank.data(), static_cast<UINT>(a_grid.size * sizeof(float) * 4), 0);
				g_placeForce = true;
			}
			// 1. A finished build is applied (only when it still belongs to this grid size and terrain generation).
			if (g_placeJob.valid() && g_placeJob.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
				PlaceJob job;
				try {
					job = g_placeJob.get();
				} catch (const std::exception& e) {
					logger::error("placement build failed ({}); keeping the last placement", e.what());
					job.size = -1;
				}
				if (job.coarseSize > 0 && job.coarseGeneration != Terrain::CoarseGeneration()) job.coarseSize = 0;
				if (job.size == g_placeSize && job.generation == Terrain::Generation()) {
					ApplyPlaceJob(a_ctx, job, now);
				}
			}
			if (g_placeJob.valid()) {
				return;   // one build at a time; whatever changed meanwhile is looked at when it is done
			}
			// 2. A new build when the grid moved (at once), or kinds or the pool fill changed (twice a second); pools still
			//    easing toward the last build only re-ease its cached target (no Build; GPT review 2026-11-08).
			const float fill = PoolFill();
			const bool fillChanged = std::fabs(fill - g_placeFill) > 0.02f;
			const bool moved = a_grid.originX != g_placeReqOriginX || a_grid.originY != g_placeReqOriginY ||
			                   Terrain::Generation() != g_placeReqGeneration || g_placeForce;
			const bool coarse = Settings::coarsePools && EnsureCoarseTexture(a_device);
			const bool coarseChanged = coarse && Terrain::CoarseDirty();
			const bool throttled = now - g_placeBuilt < std::chrono::milliseconds(500);
			if (!moved && !Terrain::KindsDirty() && !fillChanged && !coarseChanged) {
				if (g_poolSettling && !throttled && g_placeTarget.size() == static_cast<std::size_t>(g_placeSize) * g_placeSize) {
					PlaceJob ease;
					ease.grid = g_placeTarget;
					ease.originX = g_placeOriginX;
					ease.originY = g_placeOriginY;
					ease.cell = g_placeGrid[2];
					ease.size = g_placeSize;
					ease.generation = g_placeGeneration;
					ease.fill = g_placeFill;
					if (!g_coarseTarget.empty() && g_coarseGenT == Terrain::CoarseGeneration()) {
						ease.coarse = g_coarseTarget;
						ease.coarseOriginX = g_coarseOriginTX;
						ease.coarseOriginY = g_coarseOriginTY;
						ease.coarseSize = Terrain::kCoarseSize;
						ease.coarseGeneration = g_coarseGenT;
					}
					ApplyPlaceJob(a_ctx, ease, now);
				}
				return;
			}
			if (!moved && throttled) {
				return;
			}
			const auto heights = Terrain::Heights();
			const auto kinds = Terrain::Kinds();
			std::vector<float> ground(heights.begin(), heights.end());
			std::vector<PlacementRules::Kind> k(kinds.size());
			g_placeCounts = {};
			for (std::size_t i = 0; i < kinds.size(); ++i) {
				k[i] = static_cast<PlacementRules::Kind>(kinds[i]);
				if (Terrain::IsKnown(heights[i])) {
					++g_placeCounts[std::min<std::size_t>(kinds[i], 7)];
				}
			}
			PlacementRules::Params pp;
			pp.lake = Settings::placeLake;
			pp.river = Settings::placeRiver;
			pp.marshWater = Settings::placeMarshWater;
			pp.marshLand = Settings::placeMarshLand;
			pp.hotSpring = Settings::placeHotSpring;
			pp.seaRadiation = Settings::placeSeaRadiation;
			pp.seaSteam = Settings::placeSeaSteam;
			pp.freshSteam = Settings::placeFreshSteam;
			pp.shoreFactor = Settings::placeShoreFactor;
			pp.shoreDistance = Settings::placeShoreDistance;
			pp.valleyStrength = Settings::placeValleyStrength;
			pp.valleyScale = Settings::placeValleyScale;
			pp.valleyRadius = Settings::placeValleyRadius;
			pp.land = Settings::placeLand;
			pp.poolRadius = Settings::placePoolRadius;
			pp.poolMaxDepth = Settings::placePoolMaxDepth;
			pp.poolFill = fill;
			PlaceJob job;
			job.originX = a_grid.originX;
			job.originY = a_grid.originY;
			job.cell = a_grid.cell;
			job.size = a_grid.size;
			job.generation = Terrain::Generation();
			job.fill = fill;
			PlacementRules::Params coarseParams = pp;
			if (coarse) {
				const auto& cg = Terrain::CoarseGrid();
				const auto  ch = Terrain::CoarseHeights();
				// Built only once at most 10% of the cells within 12 of the centre are still waiting to be asked: a half-filled
				// grid made open pools on slopes at the fill's edge (reviewer 2026-11-08). Pending, not failed, counts: cells
				// the game cannot answer (land not loaded) must not hold the gate shut; 12 cells (6144 units, + up to 1024 from
				// the player) stays inside the loaded land. The 5 s failed-cell retry reopens them briefly; deferred jobs keep
				// easing the last coarse target meanwhile, so that does not show.
				bool filled = false;
				if (cg.size == Terrain::kCoarseSize && ch.size() == static_cast<std::size_t>(cg.size) * cg.size) {
					const int c = (cg.size - 1) / 2;
					int       inside = 0, pending = 0;
					for (int y = c - 12; y <= c + 12; ++y)
						for (int x = c - 12; x <= c + 12; ++x) {
							if ((x - c) * (x - c) + (y - c) * (y - c) > 144) continue;
							++inside;
							pending += ch[static_cast<std::size_t>(y) * cg.size + x] == Terrain::kUnknown ? 1 : 0;
						}
					filled = pending * 10 <= inside;
				}
				if (filled) {
					job.coarse.assign(ch.size(), { 0.0f, 0.0f, 0.0f, 0.0f });
					for (std::size_t i = 0; i < ch.size(); ++i) job.coarse[i][0] = ch[i];
					job.coarseOriginX = cg.originX;
					job.coarseOriginY = cg.originY;
					job.coarseSize = cg.size;
					job.coarseGeneration = Terrain::CoarseGeneration();
					coarseParams.poolRadius = Settings::coarsePoolRadius;
					coarseParams.shoreDistance = 0.0f;
				}
				Terrain::ClearCoarseDirty();
			}
			Terrain::ClearKindsDirty();
			g_placeForce = false;
			g_placeReqOriginX = a_grid.originX;
			g_placeReqOriginY = a_grid.originY;
			g_placeReqGeneration = job.generation;
			g_placeFill = fill;   // requested; a fill change during the build is caught after it
			g_placeJob = std::async(std::launch::async, [job, ground = std::move(ground), k = std::move(k), pp, coarseParams]() mutable {
				job.grid = PlacementRules::Build(ground, k, job.size, job.cell, pp);
				if (job.coarseSize > 0) {
					std::vector<float> cgz(job.coarse.size());
					for (std::size_t i = 0; i < cgz.size(); ++i) cgz[i] = job.coarse[i][0];
					const auto pools = PlacementRules::Build(cgz, std::vector<PlacementRules::Kind>(cgz.size(), PlacementRules::Kind::kLand), job.coarseSize,
						Terrain::kCoarseCell, coarseParams);
					for (std::size_t i = 0; i < cgz.size(); ++i) job.coarse[i][3] = pools[i][3];
				}
				return job;
			});
		}

		// Tallest steam column CSColumn can build this frame (units): SteamHeight x min(steam, 3), steam bounded by the
		// channel maxima of the last placement build.
		float SteamTopMax()
		{
			if (!PlaceUsable() || Settings::placeSteamHeight <= 0.0f) {
				return 0.0f;
			}
			const float steam = (g_placeMax[1] + g_placeMax[2] * g_cond.cold) * g_cond.wind;
			return Settings::placeSteamHeight * std::clamp(steam, 0.0f, 3.0f);
		}

		std::string DescribeConditions()
		{
			return std::format("conditions: hour {:.1f} month {} wind {:.2f}{} -> time {:.2f} wind {:.2f} global {:.2f}, air cold {:.2f} steam {:.2f}{} | placement cells: land {} marsh land {} lake {} river {} marsh water {} hot spring {} sea {}, mean multiplier {:.3f}",
				g_condSky.hour, g_condSky.month, g_condSky.windSpeed, g_condSky.pleasant ? " clear" : g_condSky.snow ? " snow" : g_condSky.rainy ? " rain" : g_condSky.cloudy ? " cloudy" : "",
				g_cond.time, g_cond.wind, g_cond.global, g_cond.air, g_cond.cold, g_condSky.survivalCold >= 0.0f ? std::format(" (survival {:.2f})", g_condSky.survivalCold) : std::string(),
				g_placeCounts[0], g_placeCounts[1], g_placeCounts[2], g_placeCounts[3], g_placeCounts[4], g_placeCounts[5], g_placeCounts[6], g_placeMean) +
			       std::format(" | toward sun ({:.2f}, {:.2f}, {:.2f}) glow {:.2f} | sun {:.1f} deg, weight in the fog colour {:.2f}", g_glowLog[0],
					   g_glowLog[1], g_glowLog[2], g_glowLog[3], g_lightLog[0], g_lightLog[1]);
		}

		// Backlit glow: direction toward the sun from the sun's light (its world direction points from the sun into the
		// scene), the sunlight colour of the weather. Community Shaders' Sky Sync turns that light to the moon at night,
		// so the glow also needs the clock to say it is day (GPT design review 2026-11-08), and the sun above the horizon.
		void FillGlow(RaymarchShaders::RayCB& a_rc)
		{
			a_rc.sunDir[3] = 0.0f;
			if (Settings::glowStrength <= 0.0f) {
				return;
			}
			auto* sky = RE::Sky::GetSingleton();
			if (!sky || !sky->sun || !sky->sun->light) {
				return;
			}
			const auto& d = sky->sun->light->GetWorldDirection();
			const float len = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
			if (!(len > 1e-3f) || !std::isfinite(len)) {
				return;
			}
			const float sx = -d.x / len, sy = -d.y / len, sz = -d.z / len;
			float hour = 12.0f;
			if (auto* cal = RE::Calendar::GetSingleton()) {
				hour = cal->GetHour();
			}
			auto smooth = [](float a_e0, float a_e1, float a_x) {
				const float t = std::clamp((a_x - a_e0) / (a_e1 - a_e0), 0.0f, 1.0f);
				return t * t * (3.0f - 2.0f * t);
			};
			const float day = smooth(5.0f, 7.0f, hour) * (1.0f - smooth(18.5f, 20.5f, hour));
			const float up = smooth(-0.02f, 0.12f, sz);
			using Color = RE::TESWeather::ColorTypes;
			const auto sun = ToWorking(sky->skyColor[Color::kSunlight]);
			a_rc.sunDir[0] = sx;
			a_rc.sunDir[1] = sy;
			a_rc.sunDir[2] = sz;
			// No sun glow in an "always" worldspace: those are places without a sun to light the fog (Blackreach, Soul Cairn).
			a_rc.sunDir[3] = g_regionRule.kind == WorldspaceRules::Kind::kAlways ? 0.0f : Settings::glowStrength * day * up;
			a_rc.sunColor[0] = sun.red;
			a_rc.sunColor[1] = sun.green;
			a_rc.sunColor[2] = sun.blue;
			a_rc.sunColor[3] = std::clamp(Settings::glowAnisotropy, 0.0f, 0.95f);
			g_glowLog = { sx, sy, sz, a_rc.sunDir[3] };
		}

		// Raymarch draw under its own state guard. Returns false if nothing was drawn.
		// Limit: uses the whole target as the scene area, so dynamic resolution / upscaling (scene drawn into part
		// of the texture) misplaces the fog; upgrade path: read the game's dynamic-resolution ratios.
		bool DrawRaymarchGuarded(ID3D11Device* a_device, ID3D11DeviceContext* a_ctx, RE::BSGraphics::RenderTargetData& a_main,
			RE::BSGraphics::DepthStencilData& a_depth, const FrameParams& a_cb, const D3D11_VIEWPORT& a_vp)
		{
			if (!g_rayVs || !g_rayMarch || !g_rayComposite || !g_rayCb) {
				LogNoDraw(NoDraw::kResources, "the raymarch shaders or their constant buffer are missing");
				return false;
			}
			if (!a_depth.depthSRV) {
				LogNoDraw(NoDraw::kDepth, "the main depth buffer has no shader resource view to read");
				return false;
			}
			// The wind field may be missing (first frames, Wind Enable=0): the far and height tiers still draw.
			const bool farTiers = Settings::farFog || (Settings::heightFog && Settings::heightFogDensity > 0.0f);
			bool       fieldOk = WindField::ColumnSrv() != nullptr && a_cb.fluid[3] > 0.5f;
			const UINT fullW = static_cast<UINT>(a_vp.Width);
			const UINT fullH = static_cast<UINT>(a_vp.Height);
			const UINT lowW = std::max(1u, static_cast<UINT>(fullW * Settings::raymarchScale));
			const UINT lowH = std::max(1u, static_cast<UINT>(fullH * Settings::raymarchScale));
			if (!EnsureLowTarget(a_device, lowW, lowH)) {
				LogNoDraw(NoDraw::kTarget, std::format("the {}x{} raymarch target could not be created", lowW, lowH).c_str());
				return false;
			}

			RaymarchShaders::RayCB rc{};
			if (!Invert4x4(a_cb.viewProj, rc.invViewProj)) {
				LogNoDraw(NoDraw::kMatrix, "the camera matrix cannot be inverted");
				return false;
			}
			// Vertical range of the fog box from the known ground heights.
			float lo = 1e30f;
			float hi = -1e30f;
			for (const float h : Terrain::Heights()) {
				if (Terrain::IsKnown(h)) {
					lo = std::min(lo, h);
					hi = std::max(hi, h);
				}
			}
			if (lo > hi) {
				fieldOk = false;
				lo = hi = 0.0f;
			}
			if (!fieldOk && !farTiers) {
				LogNoDraw(NoDraw::kNothing, "no wind field or no known ground (normal for the first frames after a load, or with "
				                            "[Wind] Enable=0) while [FarFog] and [HeightFog] are off");
				return false;
			}
			rc.field[0] = a_cb.fluid[0];
			rc.field[1] = a_cb.fluid[1];
			rc.field[2] = a_cb.fluid[2];
			rc.field[3] = fieldOk ? 1.0f : 0.0f;
			rc.box[0] = lo - 1.0f;
			rc.box[1] = hi + std::max(Settings::thickness + 2.0f, SteamTopMax() + 2.0f);   // CSColumn: the low layer's H <= Thickness
			if (CoarseUsable() && g_coarseTop > -1.0e29f) rc.box[1] = std::max(rc.box[1], g_coarseTop + 2.0f);   // wide-valley pool lids
			rc.box[2] = a_cb.fluid[2];   // never march further than the field size
			rc.box[3] = static_cast<float>(Settings::raymarchSteps);
			rc.fog[0] = Settings::opacity * g_weatherFactor;
			rc.fog[1] = Settings::profileSteepness;
			rc.fog[2] = static_cast<float>(g_frame % 64);
			rc.fog[3] = Settings::debugView > 0 ? 1.0f : 0.0f;
			rc.depth[0] = a_cb.depthParams[0];
			rc.depth[1] = a_cb.depthParams[1];
			rc.depth[2] = static_cast<float>(fullW);
			rc.depth[3] = static_cast<float>(fullH);
			rc.lowRes[0] = static_cast<float>(lowW);
			rc.lowRes[1] = static_cast<float>(lowH);
			rc.lowRes[2] = static_cast<float>(fullW) / static_cast<float>(lowW);
			for (int c = 0; c < 3; ++c) {
				// The march's brightness is the height fraction (0 at the ground, 1 at the layer's top): 0.9 at the
				// ground keeps the look the fog had in game before the profile rewrite (its range was 0.915 to 1.05).
				rc.colorLow[c] = a_cb.fogColor[c] * 0.9f;
				rc.colorHigh[c] = a_cb.fogColor[c] * 1.05f;
				rc.distColorNear[c] = a_cb.distFogColorNear[c];
				rc.distColorFar[c] = a_cb.distFogColorFar[c];
			}
			for (int c = 0; c < 4; ++c) {
				rc.distFog[c] = a_cb.distFog[c];
			}
			FillGlow(rc);
			rc.posAdjust[0] = a_cb.posAdjust[0];
			rc.posAdjust[1] = a_cb.posAdjust[1];
			rc.posAdjust[2] = a_cb.posAdjust[2];
			if (Settings::heightFog && Settings::heightFogDensity > 0.0f) {
				// Base = the player's feet height, low-pass filtered (time constant kHeightFogBaseSeconds).
				// Limit: one base for the whole view, so a far valley and a far plateau share it.
				const auto now = std::chrono::steady_clock::now();
				if (auto* player = RE::PlayerCharacter::GetSingleton()) {
					const float target = player->GetPositionZ();
					if (!g_heightFogBaseSet || !std::isfinite(g_heightFogBase)) {
						g_heightFogBase = target;
						g_heightFogBaseSet = true;
					} else {
						const float dt = std::min(std::chrono::duration<float>(now - g_heightFogBaseTime).count(), 1.0f);
						g_heightFogBase += (target - g_heightFogBase) * (1.0f - std::exp(-dt / kHeightFogBaseSeconds));
					}
				}
				g_heightFogBaseTime = now;
				// per metre -> per unit, times the worldspace's amount (a number or "always <number>"; height fog has no
				// weather or time rules of its own, so both scale it the same way)
				const float regionAmount = g_regionRule.kind == WorldspaceRules::Kind::kScale || g_regionRule.kind == WorldspaceRules::Kind::kAlways ? g_regionRule.value : 1.0f;
				rc.heightFog[0] = Settings::heightFogDensity * regionAmount / 70.0f;
				rc.heightFog[1] = 1.0f / Settings::heightFogFalloff;
				rc.heightFog[2] = g_heightFogBase + Settings::heightFogBaseOffset;
				rc.heightFog[3] = Settings::heightFogMaxDistance;
			}
			if (Settings::farFog) {
				// Radiation-fog conditions as CSColumn has them on plain dry ground (no placement grid out there), so the
				// two meet at the field edge: G x (Land + region) (with placement) minus evaporation (GPT review 2026-11-08).
				// The region's far value: the placed field's value has the water-and-terrain overlap taken out, and there is
				// no such grid out here. Limit: on dry ground the two differ at the field edge in every built-in region
				// (coast 0.7 vs 1.0, marsh 0.7 vs 1.2, Reach 1.0 vs 1.1, forests by 0.05); the far tier fades in over the
				// field edge, so it shows as a gradient (not looked at in game).
				const float farLand = PlaceUsable() ? Settings::placeLand : 1.0f;
				rc.farFog[0] = std::max(g_cond.global * RegionRules::Combined(farLand, GeoFarNow()) - g_cond.evaporation, 0.0f);
				rc.farFog[1] = Settings::farFogFlatLength;
				rc.farFog[2] = Settings::thickness;
				rc.farFog[3] = Settings::farFogNoiseAmount;
				rc.farNoise[0] = WindField::GustX();
				rc.farNoise[1] = WindField::GustY();
				rc.farNoise[2] = WindField::SimTime();
			}
			// Far pools: the coarse grid's pools beyond the field, at the field's pool density (CSColumn SteamP.w).
			if (Settings::farFog && Settings::farPools && Settings::placement && CoarseUsable() && g_coarseLidMax > -1.0e29f && Settings::placePoolMaxDepth >= 1.0f) {
				for (int c = 0; c < 4; ++c) rc.poolGrid[c] = g_coarseGrid[c];
				rc.poolFar[0] = PoolDensityNow();
				rc.poolFar[1] = 2.0f * Settings::placePoolMaxDepth;
				rc.poolFar[2] = g_coarseLidMax;
			}

			// Probe due? The header reads the viewport the game had bound, so it is taken before the guard changes state.
			int        probeSlot = -1;
			const auto probeNow = std::chrono::steady_clock::now();
			const bool sweepDue = Settings::probe == 2 && g_sweep.step >= 0 && !g_sweep.done && !g_sweep.probed &&
			                      probeNow - g_sweep.stepStart >= kSweepProbeAt;
			const bool timedDue = Settings::probe == 1 && probeNow - g_lastProbe >= kProbeInterval;
			if ((sweepDue || timedDue) && g_probeCs && !g_probeMeta[g_probeNext].pending) {
				probeSlot = g_probeNext;
				g_lastProbe = probeNow;
				auto& pm = g_probeMeta[probeSlot];
				pm.sweepStep = sweepDue ? g_sweep.step : -1;
				pm.sweepPitch = sweepDue ? kSweepPitch[g_sweep.step] : 0.0f;
				if (auto* pl = RE::PlayerCharacter::GetSingleton()) {
					pm.playerPitch = pl->GetAngleX();
				}
				g_probeMeta[probeSlot].header = ProbeHeader(a_ctx, a_main, a_depth, a_cb, g_probeMeta[probeSlot]);
				// CSColumn: W <= Thickness / 70 x place, place = G x p.x + (p.y + p.z x cold) x wind (the sum of the
				// channel maxima bounds it from above); without the placement grid place = G.
				// Steam columns (SteamHeight > 0) add steamH / 70 x SteamDensity on top, up to SteamTopMax().
				const bool  placed = PlaceUsable();
				const float steam = (g_placeMax[1] + g_placeMax[2] * g_cond.cold) * g_cond.wind;
				const bool  columns = placed && Settings::placeSteamHeight > 0.0f;
				const float place = placed ? g_cond.global * RegionRules::Combined(g_placeMax[0], GeoNow()) + (columns ? 0.0f : steam)
				                           : g_cond.global * RegionRules::Combined(1.0f, GeoFarNow());
				// Two layers marched together: the low one (W <= Thickness / 70 x place) plus the tall one (steam
				// columns up to SteamTopMax, pooled fog up to PoolMaxDepth x PoolDensity x strength).
				// PlacementRules holds a pool cell to 2 x PoolMaxDepth (a pit below its pool's floor).
				const float pools = placed ? 2.0f * Settings::placePoolMaxDepth / 70.0f * Settings::placePoolDensity * std::min(RadiationStrength(), 2.0f) : 0.0f;
				// Every part scales with the fog amount rho <= kMaxFogCpp (the pools with 1 + 0.6 (rho - 1)).
				constexpr float kRhoMax = WindShaders::kMaxFogCpp;
				g_probeMeta[probeSlot].wMax = Settings::thickness / 70.0f * std::max(place, 0.0f) * kRhoMax +
				                              (columns ? SteamTopMax() / 70.0f * Settings::placeSteamDensity * kRhoMax : 0.0f) +
				                              pools * (1.0f + 0.6f * (kRhoMax - 1.0f));
			}

			const StateGuard guard(a_ctx);
			D3D11_MAPPED_SUBRESOURCE mapped{};
			if (FAILED(a_ctx->Map(g_rayCb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
				LogNoDraw(NoDraw::kMap, "the raymarch constant buffer could not be written");
				return false;
			}
			std::memcpy(mapped.pData, &rc, sizeof(rc));
			a_ctx->Unmap(g_rayCb, 0);

			const bool timed = Settings::profiler || Settings::selfTest;
			if (timed) {
				g_timer.Begin(a_ctx);
			}
			const FLOAT blendFactor[4]{ 0.0f, 0.0f, 0.0f, 0.0f };
			ID3D11ShaderResourceView* none[3]{};

			// 1. march into the reduced-resolution target (no depth buffer bound, depth is read as a texture)
			a_ctx->OMSetRenderTargets(1, &g_lowRtv, nullptr);
			a_ctx->OMSetDepthStencilState(g_depthOff, 0);
			a_ctx->OMSetBlendState(nullptr, blendFactor, 0xFFFFFFFF);
			a_ctx->RSSetState(g_raster);
			const D3D11_VIEWPORT lowVp{ 0.0f, 0.0f, static_cast<float>(lowW), static_cast<float>(lowH), 0.0f, 1.0f };
			a_ctx->RSSetViewports(1, &lowVp);
			a_ctx->IASetInputLayout(nullptr);
			a_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			a_ctx->GSSetShader(nullptr, nullptr, 0);
			a_ctx->HSSetShader(nullptr, nullptr, 0);
			a_ctx->DSSetShader(nullptr, nullptr, 0);
			a_ctx->VSSetShader(g_rayVs, nullptr, 0);
			a_ctx->PSSetShader(g_rayMarch, nullptr, 0);
			a_ctx->PSSetConstantBuffers(0, 1, &g_rayCb);
			// The game's D3D objects are typed as CommonLib's REX::W32 mirrors of the D3D11 interfaces (the same COM objects).
			auto* const depthSrv = reinterpret_cast<ID3D11ShaderResourceView*>(a_depth.depthSRV);
			auto* const mainSrv = reinterpret_cast<ID3D11ShaderResourceView*>(a_main.SRV);
			auto* const mainRtv = reinterpret_cast<ID3D11RenderTargetView*>(a_main.RTV);
			ID3D11ShaderResourceView* marchIn[2]{ depthSrv, WindField::ColumnSrv() };
			ID3D11ShaderResourceView* upperIn = WindField::UpperSrv();
			a_ctx->PSSetShaderResources(0, 2, marchIn);
			a_ctx->PSSetShaderResources(4, 1, &upperIn);
			a_ctx->PSSetSamplers(0, 1, &g_linear);
			a_ctx->Draw(3, 0);
			a_ctx->PSSetShaderResources(0, 3, none);
			a_ctx->PSSetShaderResources(4, 1, none);
			bool probeOk = false;
			if (probeSlot >= 0) {
				probeOk = RunProbe(a_ctx, 0.0f, depthSrv, mainSrv);
			}

			// 2. composite onto the scene at full resolution
			a_ctx->OMSetRenderTargets(1, &mainRtv, nullptr);
			a_ctx->OMSetBlendState(g_blend, blendFactor, 0xFFFFFFFF);
			a_ctx->RSSetViewports(1, &a_vp);
			a_ctx->PSSetShader(g_rayComposite, nullptr, 0);
			ID3D11ShaderResourceView* compIn[3]{ depthSrv, WindField::ColumnSrv(), g_lowSrv };
			ID3D11ShaderResourceView* compUpper[2]{ upperIn, rc.poolFar[0] > 0.0f ? g_coarseSrv : nullptr };
			a_ctx->PSSetShaderResources(0, 3, compIn);
			a_ctx->PSSetShaderResources(4, 2, compUpper);
			a_ctx->Draw(3, 0);
			a_ctx->PSSetShaderResources(0, 3, none);
			a_ctx->PSSetShaderResources(4, 2, none);
			if (probeSlot >= 0 && probeOk && RunProbe(a_ctx, 1.0f, depthSrv, mainSrv)) {
				a_ctx->CopyResource(g_probeStaging[probeSlot], g_probeBuf);
				if (g_probeMeta[probeSlot].sweepStep >= 0) {
					g_sweep.probed = true;   // this step's probe is submitted
				}
				auto&      meta = g_probeMeta[probeSlot];
				const UINT presents = PresentCount();
				meta.pending = true;
				meta.id = ++g_probeId;
				meta.frame = g_frame;
				meta.framesSince = g_frame - g_probeLastFrame;
				meta.presentsSince = presents - g_probeLastPresents;
				meta.drawsSince = g_stats.draws - g_probeLastDraws;
				g_probeLastFrame = g_frame;
				g_probeLastPresents = presents;
				g_probeLastDraws = g_stats.draws;
				g_probeNext = (g_probeNext + 1) % kProbeRing;
			}

			if (timed) {
				g_timer.End(a_ctx);
			}
			return true;
		}
	}

	void NextFrame()
	{
		++g_frame;
		// The player-view function only renders the main camera, so remember its projection here; draw calls
		// from other accumulators (menus, UI 3D) carry other cameras and are rejected against it.
		if (auto* shadowState = RE::BSGraphics::RendererShadowState::GetSingleton()) {
			const auto& view = shadowState->GetRuntimeData().cameraData.getEye();
			const float* p = reinterpret_cast<const float*>(&view.projMat);
			g_mainNear43 = p[14];
			g_haveMainCamera = true;
		}
	}

	void Draw(const char* a_stage, bool a_ownTarget, int a_site)
	{
		++g_stats.calls;
		if (Settings::ReloadIfChanged()) {
			g_logCamera = true;
			g_placeForce = true;
			GameFog::RequestApply(Settings::gameFogScale);
			WeatherFog::RequestBuild();
			g_regionIni.clear();   // [Worldspaces] may have changed: read it and look the rule up again
			g_regionIniRead = false;
			g_regionKnown = false;
			g_geoIni.clear();      // [Regions] likewise
			g_geoIniRead = false;
			g_geoKnown = false;
		}
		if (!Settings::enable || Settings::probe != 2) {
			AbortSweep(Settings::enable ? "Probe is no longer 2" : "fog switched off");
		}
		if (!Settings::enable) {
			return;
		}
		auto* renderer = RE::BSGraphics::Renderer::GetSingleton();
		auto* shadowState = RE::BSGraphics::RendererShadowState::GetSingleton();
		auto* player = RE::PlayerCharacter::GetSingleton();
		if (!renderer || !shadowState || !player) {
			++g_stats.skipNotReady;
			return;
		}
		auto* device = reinterpret_cast<ID3D11Device*>(renderer->GetRuntimeData().forwarder);
		auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(renderer->GetRuntimeData().context);
		if (!device || !ctx || !Ready(device)) {
			++g_stats.skipNotReady;
			return;
		}
		if (g_probeCs) {
			CollectProbes(ctx);
		}

		// The scene is not always in kMAIN: the game's state cache names the target it is drawing into, and
		// Community Shaders picks its scene target the same way (CS Deferred.cpp:262, 317). Fixed kMAIN drew
		// into a buffer that never reached the screen (2026-11-06, DebugView=2 test).
		const auto& shadowRt = shadowState->GetRuntimeData();
		const auto  rawRtIndex = static_cast<std::uint32_t>(shadowRt.renderTargets[0]);
		auto        rtIndex = rawRtIndex;
		if (rtIndex >= RE::RENDER_TARGET::kTOTAL) {
			rtIndex = RE::RENDER_TARGET::kMAIN;
		}
		// Depth is always the main depth: at some effects calls the game has the pre-water copy (index 7,
		// kPOST_ZPREPASS_COPY) selected, which does not contain the water surface (seen 2026-11-07).
		const auto dsIndex = static_cast<std::uint32_t>(RE::RENDER_TARGET_DEPTHSTENCIL::kMAIN);
		auto& main = renderer->GetRuntimeData().renderTargets[rtIndex];
		auto& depth = renderer->GetDepthStencilData().depthStencils[dsIndex];
		if (rtIndex != g_stats.rtIndex) {
			logger::info("scene target index {} (kMAIN is {}), depth index {}", rtIndex,
				static_cast<std::uint32_t>(RE::RENDER_TARGET::kMAIN), dsIndex);
			g_stats.rtIndex = rtIndex;
			g_stats.dsIndex = dsIndex;
		}
		if (!main.RTV || !depth.views[0]) {
			++g_stats.skipNotReady;
			return;
		}

		// Draw only into the main view, never into reflections or cubemaps that pass through the same code.
		// The game binds targets lazily, so the D3D binding at a hook can still be the previous pass's; the state
		// cache's slot-0 target is what the game is drawing into now.
		if (rawRtIndex != static_cast<std::uint32_t>(RE::RENDER_TARGET::kMAIN)) {
			++g_stats.skipNotMainTarget;
			return;
		}
		(void)a_ownTarget;

		// Once per frame; the frame number comes from the player-view hook (FogRenderer::NextFrame).
		if (g_drawnFrame == g_frame) {
			++g_stats.duplicates;
			return;
		}

		// Main camera only: near the player and with the projection the player view used this frame.
		{
			const auto& view = shadowState->GetRuntimeData().cameraData.getEye();
			const auto& adjust = shadowState->GetRuntimeData().posAdjust.getEye();
			const auto  pp = player->GetPosition();
			const float dx = adjust.x - pp.x;
			const float dy = adjust.y - pp.y;
			const float dz = adjust.z - pp.z;
			const float near43 = reinterpret_cast<const float*>(&view.projMat)[14];
			const bool  nearPlayer = dx * dx + dy * dy + dz * dz < kMainCameraMaxDistance * kMainCameraMaxDistance;
			const bool  sameProj = g_haveMainCamera && std::fabs(near43 - g_mainNear43) <= 0.01f * std::fabs(g_mainNear43) + 1e-4f;
			// Projection is not compared: the water pass in the player view uses a different near plane (35.4)
			// than the world effects (15.0), seen 2026-11-07. Menu / UI 3D cameras sit far from the player.
			(void)sameProj;
			if (Settings::cameraFilter && !nearPlayer) {
				++g_stats.skipOtherCamera;
				// Say why, at most every 3 s, so a wrong filter shows up in the log.
				const auto now = std::chrono::steady_clock::now();
				if (kDiagLogs() && now - g_lastRejectLog > std::chrono::seconds(3)) {
					g_lastRejectLog = now;
					logger::info("skip non-main camera at site {}: distance {:.0f} (limit {:.0f}), near43 {:.4f} vs main {:.4f} (have main {}), frame {}",
						a_site, std::sqrt(dx * dx + dy * dy + dz * dz), kMainCameraMaxDistance, near43, g_mainNear43, g_haveMainCamera, g_frame);
				}
				return;
			}
		}

		auto* cell = player->GetParentCell();
		auto* space = player->GetWorldspace();
		if (!cell || cell->IsInteriorCell() || !space) {
			++g_stats.skipNoWorld;
			return;
		}
		// Before the terrain work, so a worldspace set to off costs no height queries (reviewer 2026-11-11). Terrain::Update
		// resets itself when the worldspace it is given changes, so skipping it here leaves nothing stale.
		UpdateRegion(space);
		if (g_regionRule.kind == WorldspaceRules::Kind::kOff) {
			++g_stats.skipNoWorld;   // [Worldspaces] off: as a place with no world to draw into
			++g_stats.skipWorldspaceOff;
			return;
		}
		if (!Terrain::Update(player->GetPosition(), space)) {
			++g_stats.skipNoWorld;
			return;
		}

		if (Settings::placement && Settings::coarsePools) {
			Terrain::UpdateCoarse(player->GetPosition());   // wide-valley pools (a few cells a frame)
		}
		const auto& grid = Terrain::GetGrid();
		{
			// Terrain state every 5 s: with no known ground the near field has no fog columns. Diagnosis only
			// (EnableSelfTest or EnableProfiler): five lines every 5 s is too much for a normal play session.
			const auto now = std::chrono::steady_clock::now();
			if (kDiagLogs() && now - g_lastTerrainLog > std::chrono::seconds(5)) {
				g_lastTerrainLog = now;
				const auto& ts = Terrain::GetStats();
				const auto  pp = player->GetPosition();
				const auto& ws = WindField::GetStats();
				logger::info("{}", WeatherFog::Describe());
				logger::info("{}", DescribeConditions());
				{
					const auto ch = Terrain::CoarseHeights();
					const auto cknown = std::count_if(ch.begin(), ch.end(), [](float h) { return Terrain::IsKnown(h); });
					const auto cfailed = std::count(ch.begin(), ch.end(), Terrain::kFailed);
					logger::info("coarse terrain: {} of {} cells known, {} failed; coarse pools {}", cknown, ch.size(), cfailed, CoarseUsable() ? "on" : "off");
					logger::info("fog drift {:.2f} m/s; pool density {:.3f} (asked {:.3f}, weather x{:.2f})", WindField::DriftSpeed() / 70.0f, PoolDensityNow(),
						Settings::placePoolDensity * std::min(RadiationStrength(), 2.0f), g_weatherFactor);
				}
				logger::info("terrain: {} of {} cells unknown, {} queries, {} failed, origin ({:.0f}, {:.0f}) cell {:.0f}, player ({:.0f}, {:.0f}, {:.0f}), draws {}; wind steps {} shifts {} actors {} blasts {} impacts {}",
					ts.unknownCells, grid.size * grid.size, ts.queries, ts.failures, grid.originX, grid.originY, grid.cell, pp.x, pp.y, pp.z,
					g_stats.draws, ws.steps, ws.shifts, ws.lastActors, ws.lastBlasts, ws.impacts);
			}
		}
		if (!EnsureHeightTexture(device, grid.size)) {
			++g_stats.skipNotReady;
			return;
		}
		if (Terrain::Dirty()) {
			const auto heights = Terrain::Heights();
			ctx->UpdateSubresource(g_heightTex, 0, nullptr, heights.data(), static_cast<UINT>(grid.size * sizeof(float)), 0);
			Terrain::ClearDirty();
		}

		D3D11_TEXTURE2D_DESC mainDesc{};
		if (main.texture) {
			reinterpret_cast<ID3D11Texture2D*>(main.texture)->GetDesc(&mainDesc);
		}

		D3D11_VIEWPORT vp{};
		if (a_ownTarget) {
			// Under dynamic resolution or an upscaler the scene covers only the top-left part of the target and is
			// enlarged later; drawing over the whole target put the fog of the upper-left view on the whole screen
			// (probe 2026-11-07: hook viewport 1280x800 on 2560x1600, scene depth only there). Scene area = target x
			// the game's dynamic-resolution ratios, which Community Shaders' upscaler also writes (CS Upscaling.cpp
			// 911-934). The viewport bound at the hook may be a previous pass's, so it is only compared and logged.
			// Limit: each area change recreates the low-res target (fine for fixed upscaler ratios; a per-frame
			// dynamic resolution would reallocate often; upgrade path: allocate the full size once).
			float ratioW = 1.0f;
			float ratioH = 1.0f;
			if (auto* state = RE::BSGraphics::State::GetSingleton()) {
				ratioW = state->GetRuntimeData().dynamicResolutionWidthRatio;
				ratioH = state->GetRuntimeData().dynamicResolutionHeightRatio;
			}
			if (!(ratioW > 0.05f && ratioW <= 1.0f)) ratioW = 1.0f;
			if (!(ratioH > 0.05f && ratioH <= 1.0f)) ratioH = 1.0f;
			const auto  texW = static_cast<float>(mainDesc.Width);
			const auto  texH = static_cast<float>(mainDesc.Height);
			const float areaW = std::max(1.0f, std::floor(texW * ratioW + 0.5f));
			const float areaH = std::max(1.0f, std::floor(texH * ratioH + 0.5f));
			vp = { 0.0f, 0.0f, areaW, areaH, 0.0f, 1.0f };

			// Log the area with its inputs when any of them changes, at most every 5 s.
			D3D11_VIEWPORT bound{};
			UINT           boundCount = 1;
			ctx->RSGetViewports(&boundCount, &bound);
			const std::array<float, 6> key{ areaW, areaH, texW, texH, boundCount ? bound.Width : 0.0f, boundCount ? bound.Height : 0.0f };
			static std::array<float, 6>                  s_lastKey{};
			static std::chrono::steady_clock::time_point s_lastLog{};
			static int                                   s_changes = 0;
			const auto                                   nowLog = std::chrono::steady_clock::now();
			if (key != s_lastKey) {
				++s_changes;
				if (kDiagLogs() && nowLog - s_lastLog >= std::chrono::seconds(5)) {
					logger::info("scene area {:.0f}x{:.0f} of a {:.0f}x{:.0f} target (dynamic resolution ratios {:.4f} x {:.4f}; viewport bound at the hook {:.0f}x{:.0f}{}; {} change(s) since the last line)",
						areaW, areaH, texW, texH, ratioW, ratioH, key[4], key[5],
						(key[4] == areaW && key[5] == areaH) ? ", same" : ", DIFFERENT", s_changes);
					s_lastKey = key;
					s_lastLog = nowLog;
					s_changes = 0;
				}
			}
		} else {
			UINT vpCount = 1;
			ctx->RSGetViewports(&vpCount, &vp);
			if (vpCount == 0) {
				vp = {};
			}
		}
		if (vp.Width < 1.0f || vp.Height < 1.0f) {
			++g_stats.skipNotReady;
			return;
		}

		const auto& view = shadowState->GetRuntimeData().cameraData.getEye();
		const auto& adjust = shadowState->GetRuntimeData().posAdjust.getEye();

		FrameParams cb{};
		{
			// Row-vector matrices: clip = rel * view * proj. DTD's sparkle multiplies viewMat and projMat itself.
			const float* v = reinterpret_cast<const float*>(&view.viewMat);
			const float* pr = reinterpret_cast<const float*>(&view.projMat);
			float        vp[16]{};
			for (int r = 0; r < 4; ++r) {
				for (int c = 0; c < 4; ++c) {
					float sum = 0.0f;
					for (int k = 0; k < 4; ++k) {
						sum += v[r * 4 + k] * pr[k * 4 + c];
					}
					vp[r * 4 + c] = sum;
				}
			}
			const float* chosen = Settings::matrixMode == 0 ? reinterpret_cast<const float*>(&view.viewProjMat)
			                    : Settings::matrixMode == 2 ? reinterpret_cast<const float*>(&view.viewProjMatrixUnjittered)
			                                                : vp;
			std::memcpy(cb.viewProj, chosen, sizeof(cb.viewProj));
			if (g_logCamera || g_stats.draws == 0) {
				float diff = 0.0f;
				const float* stored = reinterpret_cast<const float*>(&view.viewProjMat);
				for (int i = 0; i < 16; ++i) {
					diff = std::max(diff, std::fabs(stored[i] - vp[i]));
				}
				logger::info("camera: MatrixMode={} max |viewProjMat - viewMat*projMat| = {:.5f}; viewMat row3 {:.1f} {:.1f} {:.1f} {:.1f}",
					Settings::matrixMode, diff, v[12], v[13], v[14], v[15]);
			}
		}
		cb.posAdjust[0] = adjust.x;
		cb.posAdjust[1] = adjust.y;
		cb.posAdjust[2] = adjust.z;
		g_weatherFactor = WeatherFog::CurrentFactor();   // also for "always": it keeps the weather blend current for the way out
		if (g_regionRule.kind == WorldspaceRules::Kind::kAlways) {
			// The weather is replaced; the Community Shaders factor is the user's own scale and stays.
			g_weatherFactor = g_regionRule.value * WeatherFog::CsFactor();
		} else if (g_regionRule.kind == WorldspaceRules::Kind::kScale) {
			g_weatherFactor *= g_regionRule.value;
		}

		// projMat as stored: row-vector convention, so clip.z = z*_33 + _43 and clip.w = z*_34.
		const auto& proj = view.projMat;
		const float* p = reinterpret_cast<const float*>(&proj);
		const float  p33 = p[10];
		const float  p43 = p[14];
		g_stats.proj34 = p[11];
		g_stats.proj44 = p[15];
		g_stats.readOnlyDepth = depth.readOnlyViews[0] != nullptr;
		g_stats.depthSrv = depth.depthSRV != nullptr;
		cb.depthParams[0] = p33;
		cb.depthParams[1] = p43;
		FillLighting(cb);

		// Step the shared wind + fog field (own compute state guard) and build the raymarch's fog columns from it.
		UpdateGeo();
		UpdateConditions();
		UpdatePlacement(device, ctx, grid);
		const bool windTimed = Settings::profiler || Settings::selfTest;
		if (windTimed) {
			g_windTimer.Begin(ctx);
		}
		const bool fieldOk = WindField::Update(device, ctx, g_heightSrv);
		cb.fluid[0] = WindField::OriginX();
		cb.fluid[1] = WindField::OriginY();
		cb.fluid[2] = std::max(WindField::Extent(), 1.0f);
		cb.fluid[3] = fieldOk && WindField::Ready() ? 1.0f : 0.0f;
		if (fieldOk) {
			WindField::BuildColumns(ctx, g_heightSrv, PlaceUsable() ? g_placeSrv : nullptr,
				{ g_cond.global, g_cond.cold, g_cond.wind, 0.0f },
				{ g_cond.evaporation, Settings::placement ? Settings::placeSteamHeight : 0.0f, Settings::placeSteamDensity,
					Settings::placement ? PoolDensityNow() : 0.0f },
				g_placeGrid, { Settings::wispAmount, GeoNearNow(), 0.0f, 0.0f }, CoarseUsable() ? g_coarseSrv : nullptr, g_coarseGrid,
				{ Settings::coarseTolerance, 2.0f * Settings::placePoolMaxDepth, 0.0f, 0.0f });
		}
		if (windTimed) {
			g_windTimer.End(ctx);
		}
		// EnableProfiler=1: GPU cost every 10 s, then the averages start over.
		if (Settings::profiler) {
			const auto nowGpu = std::chrono::steady_clock::now();
			if (nowGpu - g_lastGpuLog >= std::chrono::seconds(10)) {
				g_lastGpuLog = nowGpu;
				logger::info("[GPU] fog draw avg {:.3f} ms max {:.3f} ms ({} samples) | wind step + columns avg {:.3f} ms max {:.3f} ms ({} samples) | wind {}x{}, {} pressure iterations, scene area {:.0f}x{:.0f}, raymarch scale {:.2f}, steps {}",
					g_timer.AverageMs(), g_timer.MaxMs(), g_timer.Samples(), g_windTimer.AverageMs(), g_windTimer.MaxMs(), g_windTimer.Samples(),
					Settings::simResolution, Settings::simResolution, Settings::pressureIterations, vp.Width, vp.Height, Settings::raymarchScale,
					Settings::raymarchSteps);
				g_timer.ResetStats();
				g_windTimer.ResetStats();
			}
		}
		g_stats.viewportW = vp.Width;
		g_stats.viewportH = vp.Height;
		g_stats.targetW = mainDesc.Width;
		g_stats.targetH = mainDesc.Height;
		g_stats.minDepth = vp.MinDepth;
		g_stats.maxDepth = vp.MaxDepth;

		StateGuard::Snapshot before{};
		Probe                probeBefore{};
		if (Settings::selfTest) {
			before.Capture(ctx);
			probeBefore.Capture(ctx);
		}

		// The screen raymarch is the only fog draw: when it cannot draw, no fog this frame (the reason is logged once).
		UpdateSweep(player);
		const bool drawn = DrawRaymarchGuarded(device, ctx, main, depth, cb, vp);

		if (Settings::selfTest) {
			StateGuard::Snapshot after{};
			after.Capture(ctx);
			++g_stats.stateChecks;
			if (!before.SameAs(after)) {
				++g_stats.stateMismatches;
				if (g_stats.stateMismatches == 1) {
					logger::warn("state check: bindings after the fog pass differ from before it (stage {})", a_stage);
				}
			}
			Probe probeAfter{};
			probeAfter.Capture(ctx);
			if (!probeBefore.SameAs(probeAfter)) {
				++g_stats.probeMismatches;
				if (g_stats.probeMismatches == 1) {
					logger::warn("state probe: a texture or constant buffer slot outside the guard changed (stage {})", a_stage);
				}
			}
			probeAfter.Release();
			probeBefore.Release();
			after.Release();
			before.Release();
		}

		if (drawn && (g_stats.draws == 0 || g_logCamera)) {
			g_logCamera = false;
			const float* vpm = cb.viewProj;
			const float half = grid.cell * static_cast<float>((grid.size - 1) / 2);
			logger::info("camera: posAdjust ({:.1f}, {:.1f}, {:.1f}) player ({:.1f}, {:.1f}, {:.1f}) height grid centre ({:.1f}, {:.1f})",
				adjust.x, adjust.y, adjust.z, player->GetPosition().x, player->GetPosition().y, player->GetPosition().z, grid.originX + half, grid.originY + half);
			logger::info("camera: proj _33={} _43={} depth {} with distance, viewProj row0 {:.4f} {:.4f} {:.4f} {:.4f} row3 {:.4f} {:.4f} {:.4f} {:.4f}",
				p33, p43, p43 < 0.0f ? "grows" : "shrinks (reversed)", vpm[0], vpm[1], vpm[2], vpm[3], vpm[12], vpm[13], vpm[14], vpm[15]);
		}
		if (drawn) {
			if (g_stats.draws == 0) {
				logger::info("first fog draw at stage '{}': viewport {}x{} on a {}x{} target", a_stage,
					vp.Width, vp.Height, mainDesc.Width, mainDesc.Height);
			}
			++g_stats.draws;
			g_drawnFrame = g_frame;
			if (g_stats.drawSite != a_site) {
				logger::info("fog drawn from site {} ({})", a_site, a_stage);
				g_stats.drawSite = a_site;
			}
		}
	}

	void PrecompileShaders()
	{
		auto jobs = WindSim::ShaderJobs();
		auto add = [&](const char* a_common, const char* a_body, const char* a_target, const char* a_name) {
			jobs.push_back({ std::string(a_common) + a_body, "main", a_target, a_name });
		};
		add(RaymarchShaders::kCommon, RaymarchShaders::kVS, "vs_5_0", "GroundFogRay");
		add(RaymarchShaders::kCommon, RaymarchShaders::kMarchPS, "ps_5_0", "GroundFogRay");
		add(RaymarchShaders::kCommon, RaymarchShaders::kCompositePS, "ps_5_0", "GroundFogRay");
		add(RaymarchShaders::kCommon, RaymarchShaders::kProbeCS, "cs_5_0", "GroundFogRay");
		if (!ShaderCache::StartPrecompile(std::move(jobs))) {
			logger::warn("background shader compile could not start; the first outdoor frame compiles them");
		}
	}

	void Reset()
	{
		Terrain::Reset();
		g_noDrawLogged.fill(false);   // each reason is logged again once per loaded game (written by the render thread too: a lost flag only repeats a line)
		g_heightFogBaseSet = false;   // a load can move the player anywhere
		g_condWind = -1.0f;           // the conditions' wind starts from the loaded game's wind
		g_regionKnown = false;        // look the worldspace rule up again (and log it once per loaded game)
		g_geoKnown = false;           // and the weather region's value, without easing from the old place
		g_geoHeld = nullptr;          // a load can move the player anywhere
		WeatherFog::OnGameLoad();
		g_sweepGen.fetch_add(1);   // queued pitch tasks of the old game do nothing
		g_sweepLoaded.store(true);
	}

	const Stats&  GetStats() { return g_stats; }
	std::string   WorldspaceRule() { return g_regionText; }
	std::string   RegionValue() { return std::format("{} (in use {:+.2f}, far fog {:+.2f})", g_geoText, GeoNearNow(), GeoFarNow()); }
	double        GpuAverageMs() { return g_timer.AverageMs(); }
	double        GpuMaxMs() { return g_timer.MaxMs(); }
	std::uint32_t GpuSamples() { return g_timer.Samples(); }
	double        WindGpuAverageMs() { return g_windTimer.AverageMs(); }
	double        WindGpuMaxMs() { return g_windTimer.MaxMs(); }
	std::uint32_t WindGpuMaxSample() { return g_windTimer.MaxSample(); }
	std::uint32_t WindGpuSamples() { return g_windTimer.Samples(); }
}
