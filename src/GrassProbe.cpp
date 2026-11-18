#include "PCH.h"

#include "GrassProbe.h"

#include "CallScan.h"
#include "FogRenderer.h"
#include "GrassProbeDefer.h"
#include "GrassProbeMath.h"
#include "Settings.h"

#include <Windows.h>

#include <fstream>

// How the probe works (all of it only while [Debug] GrassProbe is 1 or 2):
//   - Every direct call to BSShaderAccumulator::RenderBatches (99963 / 106609) and Main_RenderDepth (100421 / 107139)
//     in SkyrimSE.exe's .text is redirected to a thunk here at SKSEPluginLoad, the way Hooks.cpp hooks RenderEffects
//     (call-site rewrite through the SKSE trampoline).
//   - Community Shaders installs its own call-site hooks later, at kPostPostLoad (CS XSEPlugin.cpp:81-92), among them
//     Main_RenderWorld + 0x84 (Deferred.h:157, a RenderBatches call) and PlayerView + 0x395 (TerrainBlending.h:125, the
//     Main_RenderDepth call). It reads the call it replaces, which by then points at our trampoline slot, and calls
//     that: CS's thunk -> ours -> the game.
//   - Each RenderBatches call is recorded (range, group, flags, accumulator mode, bound depth). The first run (probe1,
//     DEBUG-LEDGER.md 2026-11-15 evening) showed grass reaching the main depth in the depth prepass (inside
//     Main_RenderDepth, accumulator mode 0xC) in rounds of: range 0x2B, the grass range, group 1, group 9, group 4.
//     That matches BSShaderAccumulator::FinishAccumulating_ShadowMapOrMask (Nukem9/SkyrimSETest @3289163
//     BSShaderAccumulator.cpp L393-435: 0x2B, grass, group 1, group 9 in one function, then a "Decals" function that
//     renders group 4; in 1.6.1170 the first four calls sit in the function at RVA 14B450C, the group-4 call in the one
//     at RVA 14B3D50).
//   - On a comparison frame (one every 5 s), the main depth is copied into staging textures right before and right
//     after the grass call of the first two prepass rounds, and once more at the first RenderEffects call with the main
//     target (the fog's draw point). Later frames map them without waiting (dropped after 8 tries); a CPU loop over
//     every second pixel of the scene area counts what the grass call wrote and what of it survives to the final depth.
//   - GrassProbe = 2 also, on every second comparison frame, holds the prepass grass call back at its place and makes
//     it from our group-9 thunk right after that round's group-9 call returns, through the same function pointer the
//     grass site calls, with the game's saved arguments; the depth is copied right before it (the round's depth
//     without its grass). Bookkeeping, fallbacks and drops: GrassProbeDefer.h. A deferral needs the render-state
//     precheck of the last normal comparison to have passed (the state cache fields of StateSet were the same at the
//     grass call's place and at the replay point in every round).
//   - The copies change no pipeline state: only CopyResource, Map and Unmap, which bind nothing.
namespace GrassProbe
{
	namespace
	{
		namespace M = GrassProbeMath;
		namespace D = GrassProbeDefer;

		constexpr std::uint32_t kGrassFirst = 0x5C000030;
		constexpr std::uint32_t kGrassLast = 0x5C00005C;
		constexpr std::uint32_t kFlagCubeMap = 1u << 3;        // RE::BSShaderAccumulator::RENDER_FLAG::kCubeMap
		constexpr int           kRounds = D::kRounds;
		constexpr std::size_t   kCallsKept = 384;              // entries recorded per frame ...
		constexpr std::size_t   kCallsReserved = 128;          // ... the last ones only for effects, prepass, grass and replay entries
		constexpr std::size_t   kAccsKept = 8;                 // distinct accumulators named per frame
		constexpr auto          kLogEvery = std::chrono::seconds(5);
		constexpr std::uint32_t kPollFrames = 8;               // non-blocking Map tries, then the comparison is dropped
		constexpr UINT          kSampleStride = 2;             // every second pixel in x and y
		constexpr std::size_t   kEntriesPerLine = 12;
		constexpr std::uint8_t  kNoIndex = 0xFF;
		constexpr std::uint8_t  kModeUnknown = 0xFE;           // the accumulator's vtable is not BSShaderAccumulator's: fields not read
		constexpr std::size_t   kJumpSlotBytes = 14;           // SKSE trampoline slot: FF 25 00000000 + absolute target
		constexpr int           kProbeCols = 9;                // the fog's [PROBE] grid (RaymarchShaders.h kProbeCS)
		constexpr int           kProbeRows = 7;
		constexpr float         kRadToDeg = 57.2957795f;
		// Dump image scale: log distance from 10 to 1e6 units onto 0..65535.
		constexpr float kDumpNear = 10.0f;
		constexpr float kDumpFar = 1.0e6f;

		// Staging copies of one comparison: before / after the grass call of rounds 0 and 1, and the final depth.
		enum Slot : int
		{
			kPre0,
			kPost0,
			kPre1,
			kPost1,
			kFinal,
			kSlots
		};
		constexpr int           PreSlot(int a_round) { return 2 * a_round; }
		constexpr int           PostSlot(int a_round) { return 2 * a_round + 1; }
		constexpr std::uint32_t Bit(int a_slot) { return 1u << a_slot; }

		using BatchFn = void (*)(void*, std::uint32_t, std::uint32_t, std::uint32_t, int);
		using RoundState = D::RoundState;

		enum class Dsv : std::uint8_t
		{
			kNone,
			kMain,
			kOther
		};

		enum class Kind : std::uint8_t
		{
			kBatches,   // a RenderBatches call
			kEffects,   // a RenderEffects call (the fog's draw point)
			kDepth,     // Main_RenderDepth entered
			kReplay     // mode 2: a held grass call made here
		};

		struct Entry
		{
			Kind          kind;
			std::uint32_t first;
			std::uint32_t last;
			std::uint32_t flags;
			std::int32_t  group;
			std::uint8_t  site;
			std::uint8_t  acc;
			std::uint8_t  mode;
			std::uint8_t  ssDepth;
			std::uint8_t  ssRt;
			Dsv           dsv;
			bool          prepass;
			bool          grass;
			bool          snap;       // batches: copied before and after; effects: comparison issued
			bool          fog;        // effects: the fog drew at this call
			bool          deferred;   // batches: held back here (mode 2), made later (R entry)
		};

		enum class FogSrv : std::uint8_t
		{
			kNull,
			kMain,
			kOther
		};

		// One frame's record, between two water-effects calls (the player view's end). Fixed size: no allocation.
		struct Window
		{
			std::array<Entry, kCallsKept>         calls{};
			std::array<const void*, kAccsKept>    accs{};
			std::array<std::uint8_t, kAccsKept>   accMode{};
			std::array<bool, kAccsKept>           accMain{};
			std::array<std::uintptr_t, kAccsKept> accVtable{};
			std::uint32_t                         count{ 0 };
			std::uint32_t                         dropped{ 0 };
			std::uint32_t                         batchCalls{ 0 };   // RenderBatches calls, kept or not
			std::uint32_t                         accCount{ 0 };
			std::uint32_t                         grass{ 0 };
			std::uint32_t                         grassPrepass{ 0 };
			std::uint32_t                         nearGrass{ 0 };
			std::uint32_t                         effects{ 0 };
			std::uint32_t                         prepassCalls{ 0 };
			std::int32_t                          lastEffects{ -1 };  // entry index of the latest RenderEffects call
			std::int32_t                          lastEntry{ -1 };    // entry index of the RenderBatches call being dispatched
			int                                   fogSite{ -2 };      // where the fog drew this frame (-2 = not seen)
			// Kept prepass rounds on an armed frame.
			std::array<std::uint32_t, kRounds>    roundFlags{};
			std::array<int, kRounds>              grassSite{};
			std::array<int, kRounds>              replaySite{};
			std::array<std::string, kRounds>      deferSkip{};       // own copies: a reason's source text can change before the readback
			std::array<int, kRounds>              srvOther{};        // -1 not read, 0 kMAIN view on the main texture, 1 swapped (CS)
			std::array<M::StateSet, kRounds>      stGrass{};         // at the grass call (normal) or the skip point (deferred)
			std::array<M::StateSet, kRounds>      stG9{};            // right after the group-9 call returned (the replay point)
			std::array<M::StateSet, kRounds>      stDecals{};        // at the group-4 call after the group-9 call
			std::array<M::StateSet, kRounds>      stReplay{};        // right after the held call returned (deferred)
			// Arming (decided at the first prepass grass call) and the copies taken.
			bool          armed{ false };
			bool          deferFrame{ false };
			std::uint64_t armIndex{ 0 };
			float         armP33{ 0.0f };
			float         armP43{ 0.0f };
			std::uint32_t taken{ 0 };
			bool          copyFail{ false };
			std::uint32_t rejDepth{ 0 };
			std::uint32_t rejAcc{ 0 };
			std::uint32_t rejOtherAcc{ 0 };
			std::uint32_t rejExtraRound{ 0 };
			std::uint32_t rejNotDue{ 0 };
			std::uint32_t rejNoEffects{ 0 };
			std::uint32_t rejTexture{ 0 };
			std::uint32_t rejOff{ 0 };
			bool          compared{ false };
			std::uint32_t cmpEffectsNo{ 0 };    // which RenderEffects call of the frame (1-based) issued the comparison
			std::uint64_t probeIdBefore{ 0 };
			// Why RenderEffects calls did not issue the comparison (the first reason that applied).
			std::uint32_t cmpNotArmed{ 0 };
			std::uint32_t cmpDone{ 0 };
			std::uint32_t cmpBusy{ 0 };
			std::uint32_t cmpNotMainRt{ 0 };
			std::uint32_t cmpOtherDepth{ 0 };
			std::uint32_t cmpIncomplete{ 0 };

			void Reset()
			{
				count = dropped = batchCalls = accCount = 0;
				grass = grassPrepass = nearGrass = 0;
				effects = prepassCalls = 0;
				lastEffects = lastEntry = -1;
				fogSite = -2;
				roundFlags.fill(0);
				grassSite.fill(-1);
				replaySite.fill(-1);
				for (auto& s : deferSkip) s.clear();
				srvOther.fill(-1);
				stGrass.fill({});
				stG9.fill({});
				stDecals.fill({});
				stReplay.fill({});
				armed = deferFrame = false;
				armIndex = 0;
				armP33 = armP43 = 0.0f;
				taken = 0;
				copyFail = false;
				rejDepth = rejAcc = rejOtherAcc = rejExtraRound = rejNotDue = rejNoEffects = rejTexture = rejOff = 0;
				compared = false;
				cmpEffectsNo = 0;
				probeIdBefore = 0;
				cmpNotArmed = cmpDone = cmpBusy = cmpNotMainRt = cmpOtherDepth = cmpIncomplete = 0;
			}
		};

		// One issued comparison, from the frame that copied it to its readback.
		struct Pending
		{
			bool          active{ false };
			std::uint64_t frame{ 0 };
			std::uint64_t index{ 0 };          // 1-based count of issued comparisons
			int           mode{ 1 };
			bool          deferFrame{ false };
			bool          exterior{ false };
			int           site{ -1 };
			std::uint32_t taken{ 0 };
			std::uint32_t rounds{ 0 };
			std::array<RoundState, kRounds>    roundState{};
			std::array<std::uint32_t, kRounds> roundFlags{};
			std::array<int, kRounds>           grassSite{};
			std::array<int, kRounds>           replaySite{};
			std::array<std::string, kRounds>   deferSkip{};
			std::array<int, kRounds>           srvOther{};
			float         armP33{ 0.0f };      // projection at the prepass (logged)
			float         armP43{ 0.0f };
			float         effP33{ 0.0f };      // projection at the effects call (used for distances)
			float         effP43{ 0.0f };
			std::array<float, 3> cam{};        // camera position (posAdjust) and forward (viewMat column 2) at the effects call
			std::array<float, 3> fwd{};
			UINT          areaW{ 0 };
			UINT          areaH{ 0 };
			std::uint32_t polls{ 0 };
			int           fogSite{ -2 };
			bool          drawStageWater{ false };
			FogSrv        fogSrv{ FogSrv::kNull };
			int           fogSrvFormat{ -1 };
			int           dsvFormat{ -1 };
			bool          fogGiven{ false };   // the fog drew at the comparison's call: its constants below
			float         fogP33{ 0.0f };
			float         fogP43{ 0.0f };
			std::array<float, 4> fogVpZ{};
			std::array<float, 4> fogVpW{};
			bool          fogProbeRequested{ false };
			std::uint64_t fogProbeId{ 0 };     // the [PROBE] readback submitted at that call's draw (0 = none)
		};

		bool                       g_active{ false };          // hooks written (load time)
		bool                       g_noEffects{ false };       // no RenderEffects hook: no draw point to compare at
		std::atomic<DWORD>         g_renderThread{ 0 };        // the thread the water-effects hook runs on
		std::atomic<std::uint32_t> g_otherThread{ 0 };         // RenderBatches calls on any other thread (passed on, not recorded)
		thread_local int           t_prepass{ 0 };             // > 0 while inside Main_RenderDepth on this thread
		std::uint64_t              g_frame{ 0 };
		Window                     g_win;
		Pending                    g_pending;
		D::Tracker                 g_tracker;                   // render thread only
		std::chrono::steady_clock::time_point g_lastOrderLog{};
		std::chrono::steady_clock::time_point g_lastArm{};      // when the last comparison frame was armed
		bool                       g_armedOnce{ false };
		std::uint64_t              g_issued{ 0 };               // comparisons issued (the alternation and dump cadence count these)
		std::uint64_t              g_dumpsWritten{ 0 };
		std::uint64_t              g_fogProbes{ 0 };            // [PROBE] readbacks the fog submitted at comparison calls
		std::uintptr_t             g_accVtable{ 0 };            // BSShaderAccumulator's vtable
		// Hooked RenderBatches sites and the slot each call pointed at right after our write (to see a later rewrite).
		std::array<std::uintptr_t, kMaxBatchSites> g_siteAddr{};
		std::array<std::uintptr_t, kMaxBatchSites> g_siteSlot{};
		std::size_t                g_siteCount{ 0 };
		// The render-state precheck from the last normal comparison, per round (mode 2 may defer a round only when that
		// round was checked and passed there, in the same context).
		D::Precheck                g_precheck;
		std::uint64_t              g_normalStateIndex{ 0 };
		std::array<M::StateSet, kRounds>    g_normalDecals{};   // the group-4 entry state of the last normal comparison
		std::array<std::uint64_t, kRounds>  g_normalWrote{};    // its "wrote" totals (from the readback)
		std::array<bool, kRounds>           g_normalWroteKnown{};
		std::uint64_t              g_normalWroteIndex{ 0 };

		std::array<ID3D11Texture2D*, kSlots> g_stage{};         // staging copies, same description as the main depth
		const void*                g_stageFor{ nullptr };       // the main depth texture they were made for
		D3D11_TEXTURE2D_DESC       g_stageDesc{};               // its description
		bool                       g_compareFailed{ false };    // staging or mapping not possible: no comparisons this session
		bool                       g_noTextureLogged{ false };
		bool                       g_lateLogged{ false };
		bool                       g_dropLogged{ false };

		template <class T>
		void SafeRelease(T*& a_p)
		{
			if (a_p) {
				a_p->Release();
				a_p = nullptr;
			}
		}

		std::uintptr_t Rva(std::uintptr_t a_address) { return a_address - REL::Module::get().base(); }

		std::uintptr_t FunctionRva(std::uintptr_t a_address)
		{
			DWORD64    imageBase = 0;
			const auto* entry = RtlLookupFunctionEntry(static_cast<DWORD64>(a_address), &imageBase, nullptr);
			return entry ? static_cast<std::uintptr_t>(entry->BeginAddress) : 0;
		}

		// True when the whole 5-byte call at a_site lies inside one function's unwind range. A function that calls
		// another adjusts the stack, so it is not a leaf and must have an unwind entry (x64 ABI); an E8 byte with no
		// such entry around it, or one whose rel32 runs past the function's end, is not a real call site.
		bool InsideFunction(std::uintptr_t a_site)
		{
			DWORD64    imageBase = 0;
			const auto* entry = RtlLookupFunctionEntry(static_cast<DWORD64>(a_site), &imageBase, nullptr);
			return entry && imageBase + entry->BeginAddress <= a_site && a_site + 5 <= imageBase + entry->EndAddress;
		}

		// The call at hooked site a_index still points at our trampoline slot (no plugin rewrote it after us, as
		// Community Shaders does at kPostPostLoad with the calls it wraps). A detour at a function's first instruction is
		// not seen here.
		bool SiteStillOurs(int a_index)
		{
			if (a_index < 0 || static_cast<std::size_t>(a_index) >= g_siteCount) {
				return false;
			}
			const auto site = g_siteAddr[static_cast<std::size_t>(a_index)];
			const std::span<const std::uint8_t> code{ reinterpret_cast<const std::uint8_t*>(site), 5 };
			return CallScan::CallTarget(code, site, 0) == g_siteSlot[static_cast<std::size_t>(a_index)];
		}

		std::uint8_t Small(std::uint32_t a_v) { return a_v > 0xFE ? kNoIndex : static_cast<std::uint8_t>(a_v); }

		bool SameDesc(const D3D11_TEXTURE2D_DESC& a_a, const D3D11_TEXTURE2D_DESC& a_b)
		{
			return a_a.Width == a_b.Width && a_a.Height == a_b.Height && a_a.MipLevels == a_b.MipLevels && a_a.ArraySize == a_b.ArraySize &&
			       a_a.Format == a_b.Format && a_a.SampleDesc.Count == a_b.SampleDesc.Count && a_a.SampleDesc.Quality == a_b.SampleDesc.Quality &&
			       a_a.BindFlags == a_b.BindFlags;
		}

		M::StateSet CaptureState()
		{
			M::StateSet s{};
			auto*       shadow = RE::BSGraphics::RendererShadowState::GetSingleton();
			if (!shadow) {
				return s;
			}
			const auto& ss = shadow->GetRuntimeData();
			s.valid = true;
			s.depthMode = static_cast<std::uint32_t>(ss.depthStencilDepthMode);
			s.stencilMode = ss.depthStencilStencilMode;
			s.cull = ss.rasterStateCullMode;
			s.depthBias = ss.rasterStateDepthBiasMode;
			s.blend = ss.alphaBlendMode;
			s.alphaToCoverage = ss.alphaBlendAlphaToCoverage;
			s.alphaTest = ss.alphaTestEnabled;
			s.alphaRef = ss.alphaTestRef;
			return s;
		}

		std::string StateText(const M::StateSet& a_s)
		{
			if (!a_s.valid) {
				return "not recorded";
			}
			return std::format("depthMode {} stencilMode {} cull {} depthBias {} blend {} alphaToCoverage {} alphaTest {} alphaRef {:.6f}", a_s.depthMode,
				a_s.stencilMode, a_s.cull, a_s.depthBias, a_s.blend, a_s.alphaToCoverage, a_s.alphaTest ? 1 : 0, a_s.alphaRef);
		}

		std::string DiffText(std::uint32_t a_mask)
		{
			if (a_mask & M::kStateInvalid) {
				return "a set was not recorded";
			}
			std::string s;
			for (int i = 0; i < M::kStateFields; ++i) {
				if (a_mask & (1u << i)) {
					s += std::format("{}{}", s.empty() ? "" : ", ", M::kStateNames[static_cast<std::size_t>(i)]);
				}
			}
			return s.empty() ? std::string("none") : s;
		}

		void ReleaseStaging()
		{
			for (auto*& t : g_stage) {
				SafeRelease(t);
			}
			g_stageFor = nullptr;
			g_stageDesc = {};
			g_pending.active = false;   // the textures it would read are gone
		}

		// Staging textures for a_source, made on first use and again when the main depth's texture or description
		// changes. Same description as the source except usage, bind and CPU flags, so CopyResource is legal (v1 copied
		// the main depth into a staging texture the same way). A depth texture must be copied whole (no partial copy of
		// a depth-stencil resource): each one is the full texture, about 33 MB at 2560x1600 in format 19.
		bool EnsureStaging(ID3D11Device* a_device, ID3D11Texture2D* a_source)
		{
			D3D11_TEXTURE2D_DESC desc{};
			a_source->GetDesc(&desc);
			if (g_stageFor == a_source && SameDesc(desc, g_stageDesc) && g_stage[kFinal]) {
				return true;
			}
			ReleaseStaging();
			if (desc.SampleDesc.Count != 1 || desc.ArraySize != 1 || desc.MipLevels != 1 || M::CodecOf(desc.Format) == M::Codec::kNone) {
				logger::warn("grassprobe: main depth is format {} samples {} array {} mips {}: not readable by the CPU loop; comparisons off",
					static_cast<int>(desc.Format), desc.SampleDesc.Count, desc.ArraySize, desc.MipLevels);
				g_compareFailed = true;
				return false;
			}
			D3D11_TEXTURE2D_DESC sd = desc;
			sd.Usage = D3D11_USAGE_STAGING;
			sd.BindFlags = 0;
			sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			sd.MiscFlags = 0;
			for (int s = 0; s < kSlots; ++s) {
				const HRESULT hr = a_device->CreateTexture2D(&sd, nullptr, &g_stage[s]);
				if (FAILED(hr) || !g_stage[s]) {
					logger::error("grassprobe: staging texture {} of {} ({}x{} format {}) could not be created (hr {:08X}); comparisons off", s + 1, static_cast<int>(kSlots),
						sd.Width, sd.Height, static_cast<int>(sd.Format), static_cast<std::uint32_t>(hr));
					g_stage[s] = nullptr;
					ReleaseStaging();
					g_compareFailed = true;
					return false;
				}
			}
			g_stageFor = a_source;
			g_stageDesc = desc;
			const double mb = static_cast<double>(desc.Width) * desc.Height * M::TexelBytes(M::CodecOf(desc.Format)) / 1.0e6;
			logger::info("grassprobe: {} staging copies created: {}x{} format {} bind {:X} ({} bytes per texel, {:.1f} MB each, {:.0f} MB together)",
				static_cast<int>(kSlots), desc.Width, desc.Height, static_cast<int>(desc.Format), desc.BindFlags, M::TexelBytes(M::CodecOf(desc.Format)), mb,
				mb * static_cast<double>(static_cast<int>(kSlots)));
			return true;
		}

		std::uint8_t AccSlot(const void* a_acc, std::uint8_t a_mode, bool a_main, std::uintptr_t a_vtableRva)
		{
			for (std::uint32_t i = 0; i < g_win.accCount; ++i) {
				if (g_win.accs[i] == a_acc) {
					return static_cast<std::uint8_t>(i);
				}
			}
			if (g_win.accCount >= kAccsKept) {
				return kNoIndex;
			}
			const auto i = g_win.accCount++;
			g_win.accs[i] = a_acc;
			g_win.accMode[i] = a_mode;
			g_win.accMain[i] = a_main;
			g_win.accVtable[i] = a_vtableRva;
			return static_cast<std::uint8_t>(i);
		}

		// Slot for one more entry, or null: the last kCallsReserved slots are kept for important entries.
		Entry* NextEntry(bool a_important)
		{
			const auto limit = a_important ? kCallsKept : kCallsKept - kCallsReserved;
			if (g_win.count >= limit) {
				++g_win.dropped;
				return nullptr;
			}
			return &g_win.calls[g_win.count++];
		}

		Entry* CurrentEntry()
		{
			return g_win.lastEntry >= 0 && static_cast<std::uint32_t>(g_win.lastEntry) < g_win.count ? &g_win.calls[static_cast<std::size_t>(g_win.lastEntry)] :
			                                                                                             nullptr;
		}

		Dsv BoundDsv(ID3D11DeviceContext* a_ctx, const void* a_mainTex)
		{
			ID3D11DepthStencilView* dsv = nullptr;
			a_ctx->OMGetRenderTargets(0, nullptr, &dsv);
			if (!dsv) {
				return Dsv::kNone;
			}
			ID3D11Resource* res = nullptr;
			dsv->GetResource(&res);
			const Dsv kind = res && a_mainTex && static_cast<const void*>(res) == a_mainTex ? Dsv::kMain : Dsv::kOther;
			SafeRelease(res);
			dsv->Release();
			return kind;
		}

		// True on the render thread once it is known. Checked before any setting is read: settings are written there.
		bool OnRenderThread()
		{
			const DWORD self = GetCurrentThreadId();
			const DWORD render = g_renderThread.load(std::memory_order_relaxed);
			return render != 0 && self == render;
		}

		bool ComparisonDue(std::chrono::steady_clock::time_point a_now) { return !g_pending.active && (!g_armedOnce || a_now - g_lastArm >= kLogEvery); }

		// Copies the main depth into staging slot a_slot (an armed frame only; the texture must be the one the staging
		// textures were made for).
		void CopySlot(int a_slot)
		{
			auto* renderer = RE::BSGraphics::Renderer::GetSingleton();
			auto* ctx = renderer ? reinterpret_cast<ID3D11DeviceContext*>(renderer->GetRuntimeData().context) : nullptr;
			auto* mainTex = renderer ? reinterpret_cast<ID3D11Texture2D*>(renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGET_DEPTHSTENCIL::kMAIN].texture) : nullptr;
			if (!ctx || !mainTex || static_cast<const void*>(mainTex) != g_stageFor || a_slot < 0 || a_slot >= kSlots || !g_stage[a_slot]) {
				g_win.copyFail = true;
				return;
			}
			ctx->CopyResource(g_stage[a_slot], mainTex);
			g_win.taken |= Bit(a_slot);
		}

		// 1 when the kMAIN depth view currently shows another resource than the kMAIN texture (Community Shaders' Terrain
		// Blending points it at its blended depth during the prepass), 0 when it shows the texture, -1 when not readable.
		int SrvOther()
		{
			auto* renderer = RE::BSGraphics::Renderer::GetSingleton();
			if (!renderer) {
				return -1;
			}
			auto& depth = renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGET_DEPTHSTENCIL::kMAIN];
			auto* srv = reinterpret_cast<ID3D11ShaderResourceView*>(depth.depthSRV);
			if (!srv) {
				return -1;
			}
			ID3D11Resource* res = nullptr;
			srv->GetResource(&res);
			const int other = res && static_cast<const void*>(res) == static_cast<const void*>(depth.texture) ? 0 : 1;
			SafeRelease(res);
			return other;
		}

		void DropLogged(int a_round, const char* a_why)
		{
			if (a_round < 0) {
				return;
			}
			if (!g_dropLogged) {
				g_dropLogged = true;
				logger::warn("grassprobe: frame {}: the held grass call of round {} was not made ({}); grass may be missing from this frame's prepass depth. "
							 "Mode 2 acts as mode 1 from now on (logged once)",
					g_frame, a_round, a_why);
			}
		}

		// First prepass grass call of the frame: arm the frame for a comparison when one is due.
		void TryArm(std::uint32_t a_ssDepth, std::uint32_t a_mode, std::uint32_t a_flags)
		{
			if (a_ssDepth != static_cast<std::uint32_t>(RE::RENDER_TARGET_DEPTHSTENCIL::kMAIN)) {
				++g_win.rejDepth;
				return;
			}
			if (a_mode == kModeUnknown || a_mode == kNoIndex || (a_flags & kFlagCubeMap) != 0) {
				++g_win.rejAcc;
				return;
			}
			if (g_noEffects) {
				++g_win.rejNoEffects;
				return;
			}
			if (g_compareFailed) {
				++g_win.rejOff;
				return;
			}
			const auto now = std::chrono::steady_clock::now();
			if (!ComparisonDue(now)) {
				++g_win.rejNotDue;
				return;
			}
			auto* renderer = RE::BSGraphics::Renderer::GetSingleton();
			auto* shadow = RE::BSGraphics::RendererShadowState::GetSingleton();
			auto* device = renderer ? reinterpret_cast<ID3D11Device*>(renderer->GetRuntimeData().forwarder) : nullptr;
			auto* ctx = renderer ? reinterpret_cast<ID3D11DeviceContext*>(renderer->GetRuntimeData().context) : nullptr;
			auto* mainTex = renderer ? reinterpret_cast<ID3D11Texture2D*>(renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGET_DEPTHSTENCIL::kMAIN].texture) : nullptr;
			if (!mainTex || !device || !ctx || !shadow) {
				++g_win.rejTexture;
				if (!g_noTextureLogged) {
					g_noTextureLogged = true;
					logger::warn("grassprobe: main depth texture {}, device {}, context {}: no comparison", static_cast<const void*>(mainTex),
						static_cast<const void*>(device), static_cast<const void*>(ctx));
				}
				return;
			}
			if (!EnsureStaging(device, mainTex)) {
				++g_win.rejOff;
				return;
			}
			g_win.armed = true;
			g_win.armIndex = g_issued + 1;
			g_win.deferFrame = M::DeferredComparison(Settings::grassProbe, g_win.armIndex);
			g_armedOnce = true;
			g_lastArm = now;   // a frame that arms but does not compare waits the full interval too: no copies every frame
			const auto&  view = shadow->GetRuntimeData().cameraData.getEye();   // getEye is not const
			const float* p = reinterpret_cast<const float*>(&view.projMat);
			g_win.armP33 = p[10];
			g_win.armP43 = p[14];
		}

		// GrassProbeDefer::Dispatch's environment in game: copies, records and the game's calls.
		struct Env
		{
			const D::Args& cur;

			void Game(const D::Args& a_a) const
			{
				reinterpret_cast<BatchFn>(a_a.fn)(a_a.acc, a_a.first, a_a.last, a_a.flags, a_a.group);
			}

			void Seen(const D::Call& a_c, const D::Before& a_b) const
			{
				g_win.lastEntry = -1;
				if (Settings::grassProbe == 0) {
					return;
				}
				auto* renderer = RE::BSGraphics::Renderer::GetSingleton();
				auto* shadow = RE::BSGraphics::RendererShadowState::GetSingleton();
				if (!renderer || !shadow) {
					return;
				}
				auto*       ctx = reinterpret_cast<ID3D11DeviceContext*>(renderer->GetRuntimeData().context);
				auto* const mainTex = reinterpret_cast<ID3D11Texture2D*>(renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGET_DEPTHSTENCIL::kMAIN].texture);
				// Fields are read only from an object whose vtable is BSShaderAccumulator's (renderMode at 0x150 in
				// CommonLibSSE-NG's SE/AE layout); anything else is logged as mode FE with its vtable.
				auto*          acc = static_cast<RE::BSShaderAccumulator*>(cur.acc);
				std::uint32_t  mode = kNoIndex;
				bool           isMain = false;
				std::uintptr_t vtable = 0;
				if (acc) {
					vtable = *reinterpret_cast<const std::uintptr_t*>(cur.acc);
					mode = vtable == g_accVtable ? static_cast<std::uint32_t>(acc->GetRuntimeData().renderMode) : kModeUnknown;
					isMain = acc == RE::BSShaderAccumulator::GetMainAccumulator();
				}
				const auto& ss = shadow->GetRuntimeData();
				const auto  ssDepth = static_cast<std::uint32_t>(ss.depthStencil);
				const auto  ssRt = static_cast<std::uint32_t>(ss.renderTargets[0]);
				const Dsv   dsv = ctx ? BoundDsv(ctx, mainTex) : Dsv::kNone;
				++g_win.batchCalls;
				if (!a_c.grass && cur.first <= kGrassLast && cur.last >= kGrassFirst) {
					++g_win.nearGrass;
				}
				if (Entry* entry = NextEntry(a_c.grass || a_c.prepass)) {
					*entry = { Kind::kBatches, cur.first, cur.last, cur.flags, cur.group, Small(static_cast<std::uint32_t>(cur.site)),
						AccSlot(cur.acc, Small(mode), isMain, vtable ? vtable - REL::Module::get().base() : 0), Small(mode), Small(ssDepth), Small(ssRt), dsv,
						a_c.prepass, a_c.grass, false, false, false };
					g_win.lastEntry = static_cast<std::int32_t>(g_win.count - 1);
				}
				if (a_c.grass) {
					++g_win.grass;
					if (a_c.prepass) ++g_win.grassPrepass;
				}
				if (a_b.otherAcc) ++g_win.rejOtherAcc;
				if (a_b.extraRound) ++g_win.rejExtraRound;
				if (a_b.round == 0) {
					TryArm(ssDepth, mode, cur.flags);
				}
				if (!g_win.armed) {
					return;
				}
				if (a_b.round >= 0) {
					const auto r = static_cast<std::size_t>(a_b.round);
					g_win.roundFlags[r] = cur.flags;
					g_win.grassSite[r] = cur.site;
					g_win.srvOther[r] = SrvOther();
				}
				if (a_b.decalsRound >= 0) {
					g_win.stDecals[static_cast<std::size_t>(a_b.decalsRound)] = CaptureState();
				}
			}

			D::Gate GateFor(int a_round) const
			{
				D::Gate g{};
				g.armed = Settings::grassProbe != 0 && g_win.armed;
				g.deferFrame = g_win.deferFrame;
				g.modeAllows = Settings::grassProbe == 2;
				g.stateBlock = g_precheck.Block(a_round);   // valid until the next Record / Invalidate; Decided copies it
				return g;
			}

			bool SiteIntact(int a_site) const { return SiteStillOurs(a_site); }

			void Decided(int a_round, const char* a_whyNot) const
			{
				const auto r = static_cast<std::size_t>(a_round);
				if (!a_whyNot) {
					g_win.stGrass[r] = CaptureState();   // the skip point
					if (Entry* e = CurrentEntry()) e->deferred = true;
					return;
				}
				if (g_win.deferFrame) {
					g_win.deferSkip[r] = a_whyNot;
					logger::info("grassprobe: frame {} round {}: deferral not allowed: {}", g_frame, a_round, a_whyNot);
				}
			}

			void BeforeGrass(int a_round) const
			{
				g_win.stGrass[static_cast<std::size_t>(a_round)] = CaptureState();
				if (Entry* e = CurrentEntry()) e->snap = true;
				CopySlot(PreSlot(a_round));
			}

			void AfterGrass(int a_round) const { CopySlot(PostSlot(a_round)); }

			void G9Returned(int a_round) const
			{
				if (g_win.armed) {
					g_win.stG9[static_cast<std::size_t>(a_round)] = CaptureState();
				}
			}

			void ReplayBegin(int a_round) const { CopySlot(PreSlot(a_round)); }

			void ReplayEnd(int a_round, bool a_planned, int a_site) const
			{
				const auto r = static_cast<std::size_t>(a_round);
				CopySlot(PostSlot(a_round));
				g_win.stReplay[r] = CaptureState();
				g_win.replaySite[r] = a_site;
				if (Entry* e = NextEntry(true)) {
					*e = {};
					e->kind = Kind::kReplay;
					e->site = Small(static_cast<std::uint32_t>(g_win.grassSite[r]));
					e->first = kGrassFirst;
					e->last = kGrassLast;
					e->flags = g_win.roundFlags[r];
					e->group = -1;
					e->prepass = t_prepass > 0;
					e->grass = true;
				}
				if (!a_planned && !g_lateLogged) {
					g_lateLogged = true;
					logger::warn("grassprobe: frame {}: round {} closed without a group-9 call; the held grass call was made before the call at site {} on the same "
								 "accumulator instead. Mode 2 acts as mode 1 from now on (logged once)",
						g_frame, a_round, a_site);
				}
			}

			void Dropped(int a_round) const { DropLogged(a_round, "another accumulator, another depth target or a call outside the prepass came first"); }
		};

		template <std::size_t I>
		struct BatchSite
		{
			static void thunk(void* a_accumulator, std::uint32_t a_firstPass, std::uint32_t a_lastPass, std::uint32_t a_renderFlags, int a_groupIndex)
			{
				const DWORD render = g_renderThread.load(std::memory_order_relaxed);
				if (render == 0 || GetCurrentThreadId() != render) {
					if (render != 0) g_otherThread.fetch_add(1, std::memory_order_relaxed);
					func(a_accumulator, a_firstPass, a_lastPass, a_renderFlags, a_groupIndex);
					return;
				}
				const D::Args args{ func.address(), a_accumulator, a_firstPass, a_lastPass, a_renderFlags, a_groupIndex, static_cast<int>(I) };
				D::Call       call{};
				call.site = static_cast<int>(I);
				call.acc = a_accumulator;
				call.group = a_groupIndex;
				call.grass = a_firstPass == kGrassFirst && a_lastPass == kGrassLast;
				call.prepass = t_prepass > 0;
				const auto* shadow = RE::BSGraphics::RendererShadowState::GetSingleton();
				call.mainDepth = shadow && static_cast<std::uint32_t>(shadow->GetRuntimeData().depthStencil) ==
				                               static_cast<std::uint32_t>(RE::RENDER_TARGET_DEPTHSTENCIL::kMAIN);
				Env env{ args };
				D::Dispatch(g_tracker, call, args, env);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		template <std::size_t I>
		struct DepthSite
		{
			static void thunk(bool a_1, bool a_2)
			{
				++t_prepass;
				if (OnRenderThread() && Settings::grassProbe != 0) {
					++g_win.prepassCalls;
					if (auto* e = NextEntry(true)) {
						*e = {};
						e->kind = Kind::kDepth;
					}
				}
				func(a_1, a_2);
				--t_prepass;
				if (t_prepass == 0 && OnRenderThread()) {
					g_tracker.PrepassEnd();   // a held call waits for the next call, which makes or drops it
				}
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		template <template <std::size_t> class Site, std::size_t... I>
		void WriteSites(const std::vector<std::uintptr_t>& a_sites, std::index_sequence<I...>)
		{
			((I < a_sites.size() ? (void)(Site<I>::func = SKSE::GetTrampoline().write_call<5>(a_sites[I], Site<I>::thunk)) : (void)0), ...);
		}

		// Every direct call to a_target in .text, logged; the first a_max that lie inside a function are returned for hooking.
		std::vector<std::uintptr_t> FindSites(const char* a_name, std::uintptr_t a_target, std::size_t a_max)
		{
			const auto text = REL::Module::get().segment(REL::Segment::textx);
			const std::span<const std::uint8_t> code{ reinterpret_cast<const std::uint8_t*>(text.address()), text.size() };
			const auto start = std::chrono::steady_clock::now();
			const auto calls = CallScan::FindCallsTo(code, text.address(), a_target);
			const auto jumps = CallScan::FindJumpsTo(code, text.address(), a_target);
			const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			std::vector<std::uintptr_t> sites;
			std::size_t                 outside = 0;
			for (const auto& hit : calls) {
				const auto  site = text.address() + hit.offset;
				const char* note = "";
				if (!InsideFunction(site)) {
					note = " (not hooked: not inside one function's unwind range, so not a real call)";
					++outside;
				} else if (sites.size() >= a_max) {
					note = " (not hooked: over the limit)";
				} else {
					sites.push_back(site);
				}
				logger::info("grassprobe:   {} site: call at RVA {:X} in the function at RVA {:X}{}", a_name, Rva(site), FunctionRva(site), note);
			}
			for (const auto& hit : jumps) {
				const auto site = text.address() + hit.offset;
				logger::info("grassprobe:   {} tail jump at RVA {:X} in the function at RVA {:X} (not hooked: its calls are missing from the order lines)",
					a_name, Rva(site), FunctionRva(site));
			}
			logger::info("grassprobe: {} at RVA {:X}: {} calls ({} outside any function) and {} tail jumps in .text ({:.1f} ms); calls hooked: {}", a_name,
				Rva(a_target), calls.size(), outside, jumps.size(), ms, sites.size());
			return sites;
		}

		// What the call at Main_RenderWorld + 0x84 (AE) / 0x8E (SE) points at when we load. Community Shaders rewrites it
		// later, at kPostPostLoad (Deferred.h:157, "Main_RenderWorld_Start"), and then calls whatever was there: our slot if
		// the scan hooked it. Only described here, never hooked by this function.
		void DescribeRenderWorldStart(std::uintptr_t a_batches)
		{
			const auto site = REL::RelocationID(99938, 106583).address() + REL::Relocate(0x8E, 0x84);
			const auto text = REL::Module::get().segment(REL::Segment::textx);
			const std::span<const std::uint8_t> code{ reinterpret_cast<const std::uint8_t*>(site), 5 };
			const auto target = CallScan::CallTarget(code, site, 0);
			const auto batchRenderer = REL::RelocationID(100852, 107642).address();
			std::string what;
			if (target == 0) {
				what = std::format("no direct call (byte {:02X})", code[0]);
			} else if (target == a_batches) {
				what = "BSShaderAccumulator::RenderBatches (in the scan's list)";
			} else if (target == batchRenderer) {
				what = "BSBatchRenderer::RenderBatches (not hooked; not in the order lines)";
			} else if (target >= text.address() && target < text.address() + text.size()) {
				what = std::format("the game function at RVA {:X} (not hooked)", Rva(target));
			} else {
				// A plugin that loaded before us rewrote it: read its trampoline slot if the page is plainly readable.
				MEMORY_BASIC_INFORMATION mbi{};
				std::uintptr_t           jumpTo = 0;
				constexpr DWORD          kReadable = PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_READONLY | PAGE_READWRITE;
				if (VirtualQuery(reinterpret_cast<const void*>(target), &mbi, sizeof(mbi)) == sizeof(mbi) && mbi.State == MEM_COMMIT &&
					(mbi.Protect & kReadable) != 0 && (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0 &&
					target + kJumpSlotBytes <= reinterpret_cast<std::uintptr_t>(mbi.BaseAddress) + mbi.RegionSize) {
					const auto* b = reinterpret_cast<const std::uint8_t*>(target);
					if (b[0] == 0xFF && b[1] == 0x25 && b[2] == 0 && b[3] == 0 && b[4] == 0 && b[5] == 0) {
						std::memcpy(&jumpTo, b + 6, sizeof(jumpTo));
					}
				}
				char    module[MAX_PATH]{};
				HMODULE owner = nullptr;
				if (jumpTo && GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
								  reinterpret_cast<LPCSTR>(jumpTo), &owner)) {
					GetModuleFileNameA(owner, module, MAX_PATH);
				}
				what = std::format("a call another plugin rewrote before us (slot {:X}, jumps to {:X} in {}): its calls are not in the order lines", target,
					jumpTo, module[0] ? module : "an unknown module");
			}
			logger::info("grassprobe: at our load the call at Main_RenderWorld RVA {:X} calls {}; Community Shaders rewrites it at kPostPostLoad and then calls "
						 "what was there",
				Rva(site), what);
		}

		std::string RoundsText()
		{
			std::string s;
			for (int r = 0; r < kRounds; ++r) {
				s += std::format("{}r{} {}", r ? ", " : "", r, D::RoundStateText(g_tracker.State(r)));
			}
			return s;
		}

		void LogOrder()
		{
			const auto other = g_otherThread.exchange(0, std::memory_order_relaxed);
			logger::info("grassprobe: order frame {}: {} RenderBatches calls, {} RenderEffects calls, {} Main_RenderDepth calls ({} entries not kept, {} "
						 "RenderBatches calls on other threads since the last line); grass range {} (in prepass {}), overlapping ranges {}; prepass rounds {} "
						 "(kept {}: {}); armed {} (index {}, deferred frame {}) (skipped: depth not main {}, accumulator {}, other accumulator {}, extra rounds {}, "
						 "not due {}, no effects hook {}, texture {}, off {}); copies {:02X}{}; comparison issued {} (skipped: not armed {}, done {}, busy {}, "
						 "target not main {}, other depth {}, incomplete {}); fog drawn at site {}",
				g_frame, g_win.batchCalls, g_win.effects, g_win.prepassCalls, g_win.dropped, other, g_win.grass, g_win.grassPrepass, g_win.nearGrass,
				g_tracker.Rounds(), std::min<std::uint32_t>(g_tracker.Rounds(), kRounds), RoundsText(), g_win.armed ? 1 : 0, g_win.armIndex, g_win.deferFrame ? 1 : 0,
				g_win.rejDepth, g_win.rejAcc, g_win.rejOtherAcc, g_win.rejExtraRound, g_win.rejNotDue, g_win.rejNoEffects, g_win.rejTexture, g_win.rejOff,
				g_win.taken, g_win.copyFail ? " (a copy failed)" : "", g_win.compared ? 1 : 0, g_win.cmpNotArmed, g_win.cmpDone, g_win.cmpBusy,
				g_win.cmpNotMainRt, g_win.cmpOtherDepth, g_win.cmpIncomplete,
				Settings::drawStage == Settings::DrawStage::kAfterWaterEffects ? std::string("water (DrawStage 1, not tracked)") :
				g_win.fogSite == -2                                            ? std::string("none") :
				                                                                 std::to_string(g_win.fogSite));
			std::string accs;
			for (std::uint32_t i = 0; i < g_win.accCount; ++i) {
				accs += std::format(" a{}={:X} m{:X} vt{:X}{}", i, reinterpret_cast<std::uintptr_t>(g_win.accs[i]), g_win.accMode[i], g_win.accVtable[i],
					g_win.accMain[i] ? " main" : "");
			}
			logger::info("grassprobe: order frame {} accumulators:{} (kMAIN depth index {}, kMAIN target index {})", g_frame, accs.empty() ? " none" : accs,
				static_cast<std::uint32_t>(RE::RENDER_TARGET_DEPTHSTENCIL::kMAIN), static_cast<std::uint32_t>(RE::RENDER_TARGET::kMAIN));
			for (std::uint32_t start = 0; start < g_win.count; start += kEntriesPerLine) {
				std::string line;
				const auto  end = std::min<std::uint32_t>(g_win.count, start + static_cast<std::uint32_t>(kEntriesPerLine));
				for (std::uint32_t i = start; i < end; ++i) {
					const auto& e = g_win.calls[i];
					const char* sep = i == start ? "" : " | ";
					if (e.kind == Kind::kEffects) {
						line += std::format("{}{}:E{}{}{}", sep, i, e.site, e.snap ? " C" : "", e.fog ? " F" : "");
						continue;
					}
					if (e.kind == Kind::kDepth) {
						line += std::format("{}{}:D", sep, i);
						continue;
					}
					if (e.kind == Kind::kReplay) {
						line += std::format("{}{}:R s{} f{:X} g{} {:08X}-{:08X}{}", sep, i, e.site, e.flags, e.group, e.first, e.last, e.prepass ? " P" : "");
						continue;
					}
					line += std::format("{}{}:s{} a{} m{:X} f{:X} g{} {:08X}-{:08X} d{} z{} r{}{}{}{}{}", sep, i, e.site, e.acc == kNoIndex ? -1 : e.acc,
						e.mode, e.flags, e.group, e.first, e.last, e.dsv == Dsv::kMain ? 'M' : e.dsv == Dsv::kOther ? 'o' : '-', e.ssDepth, e.ssRt,
						e.prepass ? " P" : "", e.grass ? " G" : "", e.snap ? " S" : "", e.deferred ? " X" : "");
				}
				logger::info("grassprobe: order frame {} calls {}-{}: {}", g_frame, start, end - 1, line);
			}
		}

		// Render-state lines of an armed frame (logged at its end): the precheck's two sets per round, and on a deferred
		// frame the state after the replay against the last normal comparison's group-4 entry.
		void LogStates()
		{
			for (int r = 0; r < kRounds; ++r) {
				const auto ri = static_cast<std::size_t>(r);
				const auto st = g_tracker.State(r);
				if (st == RoundState::kNone) {
					continue;
				}
				const bool          deferred = st != RoundState::kNormal;
				const std::uint32_t d = M::StateDiff(g_win.stGrass[ri], g_win.stG9[ri]);
				std::string         extra;
				if (deferred) {
					const std::uint32_t dr = M::StateDiff(g_win.stReplay[ri], g_normalDecals[ri]);
					extra = std::format(" | after the replay {} | vs the group-4 entry of normal comparison #{} ({}): {}", StateText(g_win.stReplay[ri]),
						g_normalStateIndex, StateText(g_normalDecals[ri]), dr == 0 ? std::string("equal") : "differs: " + DiffText(dr));
				} else {
					extra = std::format(" | at the group-4 call {}", StateText(g_win.stDecals[ri]));
				}
				logger::info("grassprobe: state frame {} round {} ({}; srv of kMAIN shows {}): at the {} {} | after the group-9 call {} | {}{}", g_frame, r,
					D::RoundStateText(st), g_win.srvOther[ri] < 0 ? "?" : g_win.srvOther[ri] ? "another resource (swapped)" : "the kMAIN texture",
					deferred ? "skip point" : "grass call", StateText(g_win.stGrass[ri]), StateText(g_win.stG9[ri]),
					d == 0 ? std::string("equal") : "differs: " + DiffText(d), extra);
			}
		}

		// The precheck's context now: the setting and the player's place (exterior: its worldspace; interior: its cell).
		D::Context CurrentContext()
		{
			D::Context c{};
			c.mode = Settings::grassProbe;
			auto* player = RE::PlayerCharacter::GetSingleton();
			auto* cell = player ? player->GetParentCell() : nullptr;
			c.exterior = cell && !cell->IsInteriorCell();
			c.place = c.exterior ? static_cast<const void*>(player->GetWorldspace()) : static_cast<const void*>(cell);
			return c;
		}

		// A normal comparison updates the precheck, per round: checked = the round's grass call was made at its place
		// with both state sets recorded; passed = the two sets were equal. A frame with any deferred round does not count.
		void UpdatePrecheck()
		{
			bool                             any = false;
			std::array<bool, kRounds>        checked{};
			std::array<bool, kRounds>        passed{};
			std::array<std::string, kRounds> diffs{};
			for (int r = 0; r < kRounds; ++r) {
				const auto ri = static_cast<std::size_t>(r);
				const auto st = g_tracker.State(r);
				if (st == RoundState::kNone) {
					continue;
				}
				if (st != RoundState::kNormal) {
					return;   // a deferred frame does not update the precheck
				}
				any = true;
				const std::uint32_t d = M::StateDiff(g_win.stGrass[ri], g_win.stG9[ri]);
				checked[ri] = (d & M::kStateInvalid) == 0;
				passed[ri] = d == 0;
				diffs[ri] = DiffText(d);
			}
			if (!any) {
				return;
			}
			g_precheck.Record(CurrentContext(), g_issued, checked, passed, diffs);
			g_normalStateIndex = g_issued;
			g_normalDecals = g_win.stDecals;
		}

		// The mapped staging copies of one comparison and how to reach a texel.
		struct Maps
		{
			std::array<D3D11_MAPPED_SUBRESOURCE, kSlots> m{};
			std::uint32_t                                taken{ 0 };
			M::Codec                                     codec{ M::Codec::kNone };
			std::uint32_t                                bytes{ 0 };

			bool Has(int a_slot) const { return (taken & Bit(a_slot)) != 0 && m[static_cast<std::size_t>(a_slot)].pData; }
			const std::uint8_t* Texel(int a_slot, UINT a_x, UINT a_y) const
			{
				const auto& s = m[static_cast<std::size_t>(a_slot)];
				return static_cast<const std::uint8_t*>(s.pData) + static_cast<std::size_t>(a_y) * s.RowPitch + static_cast<std::size_t>(a_x) * bytes;
			}
			std::uint32_t Key(int a_slot, UINT a_x, UINT a_y) const { return M::RawKey(codec, Texel(a_slot, a_x, a_y)); }
			float         Raw(int a_slot, UINT a_x, UINT a_y) const { return M::DecodeDepth(codec, Texel(a_slot, a_x, a_y)); }
		};

		std::string BandsText(const std::array<std::uint32_t, M::kBands>& a_v, const std::array<std::uint32_t, M::kBands>& a_n)
		{
			std::string s;
			for (int i = 0; i < M::kBands; ++i) {
				s += std::format("{}{} {}/{}", i ? " " : "", M::kBandNames[static_cast<std::size_t>(i)], a_v[static_cast<std::size_t>(i)],
					a_n[static_cast<std::size_t>(i)]);
			}
			return s;
		}

		std::uint64_t Sum(const std::array<std::uint32_t, M::kBands>& a_v)
		{
			std::uint64_t s = 0;
			for (const auto v : a_v) s += v;
			return s;
		}

		std::uint16_t DumpValue(float a_raw, bool a_grows, bool a_flip, float a_p33, float a_p43)
		{
			if (M::IsFarValue(a_raw, a_grows)) {
				return 0xFFFF;
			}
			const float z = M::Distance(a_raw, a_flip, a_p33, a_p43);
			if (!(z > 0.0f) || z >= M::kNoDistance) {
				return 0;   // a bad distance (wrong convention) shows as black
			}
			const float t = std::log(std::clamp(z, kDumpNear, kDumpFar) / kDumpNear) / std::log(kDumpFar / kDumpNear);
			return static_cast<std::uint16_t>(std::lround(std::clamp(t, 0.0f, 1.0f) * 65535.0f));
		}

		// Raw texel at (x, y) of the final copy: value and bytes in hex, for the pixel lines.
		std::string PixelText(const Maps& a_maps, const char* a_name, int a_x, int a_y, UINT a_w, UINT a_h)
		{
			if (a_x < 0 || a_y < 0 || static_cast<UINT>(a_x) >= a_w || static_cast<UINT>(a_y) >= a_h) {
				return std::format("{} ({}, {}) outside the scene area", a_name, a_x, a_y);
			}
			const auto* t = a_maps.Texel(kFinal, static_cast<UINT>(a_x), static_cast<UINT>(a_y));
			std::string hex;
			for (std::uint32_t b = 0; b < a_maps.bytes; ++b) {
				hex += std::format("{:02X}", t[b]);
			}
			return std::format("{} ({}, {}) d {:.7f} bytes {}", a_name, a_x, a_y, M::DecodeDepth(a_maps.codec, t), hex);
		}

		bool Replayed(RoundState a_s) { return a_s == RoundState::kAfterG9 || a_s == RoundState::kLate; }

		// One numbered pair of full-resolution PGM images of the scene area into the SKSE log folder (a hitch of tens of
		// ms; row buffers allocated here, once per dump). pre: the round-0 copy right before its grass call (normal frame)
		// or right before the held call was made (deferred frame: without round 0's grass); mask: the grass mask over the
		// kept rounds.
		void WriteDump(const Maps& a_maps, UINT a_w, UINT a_h, bool a_grows, bool a_flip, float a_p33, float a_p43)
		{
			if (!a_maps.Has(PreSlot(0))) {
				logger::warn("grassprobe: dump skipped: no round-0 copy on frame {}", g_pending.frame);
				return;
			}
			auto dir = logger::log_directory();
			if (!dir) {
				logger::warn("grassprobe: no SKSE log folder; dump skipped");
				return;
			}
			const bool    deferredKind = Replayed(g_pending.roundState[0]);
			const auto    nn = M::DumpSlot(g_dumpsWritten);
			const auto    prePath = *dir / std::format("GroundFog_grassprobe_{:02}_pre.pgm", nn);
			const auto    maskPath = *dir / std::format("GroundFog_grassprobe_{:02}_mask.pgm", nn);
			std::ofstream pre(prePath, std::ios::binary | std::ios::trunc);
			std::ofstream mask(maskPath, std::ios::binary | std::ios::trunc);
			if (!pre || !mask) {
				logger::warn("grassprobe: could not open {} / {} for writing; dump skipped", prePath.string(), maskPath.string());
				return;
			}
			const char* kind = deferredKind ? "deferred" : "normal";
			const char* what = deferredKind ? "grass-free depth (round 0, right before the held grass call was made)" :
			                                  "pre-grass depth (round 0, right before its grass call)";
			pre << std::format("P5\n# GroundFog grass probe frame {} (#{}), {} frame: {} as log distance, 0 = {} units or a bad distance, 65535 = {} units or the "
							   "far value; convention {}\n{} {}\n65535\n",
				g_pending.frame, g_pending.index, kind, what, kDumpNear, kDumpFar, a_grows ? "grows" : "reversed", a_w, a_h);
			mask << std::format("P5\n# GroundFog grass probe frame {} (#{}), {} frame: grass mask over the kept prepass rounds: {} no grass write, {} final depth at "
								"the far value, {} grass wrote and the final value is another (more than 1 % off), {} grass wrote and the final value is within 1 %, "
								"{} grass wrote and is still the final value\n{} {}\n255\n",
				g_pending.frame, g_pending.index, kind, M::kMaskNoGrass, M::kMaskFinalFar, M::kMaskGrassCovered, M::kMaskGrassNear, M::kMaskGrassVisible, a_w, a_h);
			std::vector<std::uint8_t> preRow(static_cast<std::size_t>(a_w) * 2);
			std::vector<std::uint8_t> maskRow(a_w);
			for (UINT y = 0; y < a_h; ++y) {
				for (UINT x = 0; x < a_w; ++x) {
					const auto v = DumpValue(a_maps.Raw(PreSlot(0), x, y), a_grows, a_flip, a_p33, a_p43);
					preRow[2 * static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(v >> 8);   // PGM 16-bit samples are big-endian
					preRow[2 * static_cast<std::size_t>(x) + 1] = static_cast<std::uint8_t>(v & 0xFF);
					const auto  kF = a_maps.Key(kFinal, x, y);
					const float dF = a_maps.Raw(kFinal, x, y);
					const float zF = M::Distance(dF, a_flip, a_p33, a_p43);
					bool        wrote = false;
					bool        visible = false;
					bool        nearHit = false;
					for (int r = 0; r < kRounds; ++r) {
						if (a_maps.Has(PreSlot(r)) && a_maps.Has(PostSlot(r))) {
							const auto kPre = a_maps.Key(PreSlot(r), x, y);
							const auto kPost = a_maps.Key(PostSlot(r), x, y);
							if (kPre != kPost) {
								wrote = true;
								visible = visible || kPost == kF;
								nearHit = nearHit || M::NearEqual(M::Distance(a_maps.Raw(PostSlot(r), x, y), a_flip, a_p33, a_p43), zF);
							}
						}
					}
					maskRow[x] = M::MaskValue(wrote, visible, nearHit, M::IsFarValue(dF, a_grows));
				}
				pre.write(reinterpret_cast<const char*>(preRow.data()), static_cast<std::streamsize>(preRow.size()));
				mask.write(reinterpret_cast<const char*>(maskRow.data()), static_cast<std::streamsize>(maskRow.size()));
			}
			if (!pre || !mask) {
				logger::warn("grassprobe: writing the dump failed ({} / {})", prePath.string(), maskPath.string());
				return;
			}
			++g_dumpsWritten;
			logger::info("grassprobe: dump {:02} frame {} (#{}, {} frame) written: {} and {} ({}x{}; pre = {}, log distance 0 = {} units or bad, 65535 = {} units or far; "
						 "mask {} no grass write, {} final at the far value, {} grass wrote but covered by E (more than 1 % off), {} grass wrote and E is within 1 %, "
						 "{} grass visible at E)",
				nn, g_pending.frame, g_pending.index, kind, prePath.string(), maskPath.string(), a_w, a_h, what, kDumpNear, kDumpFar, M::kMaskNoGrass,
				M::kMaskFinalFar, M::kMaskGrassCovered, M::kMaskGrassNear, M::kMaskGrassVisible);
		}

		const char* FogSrvText()
		{
			switch (g_pending.fogSrv) {
			case FogSrv::kMain:
				return "1";
			case FogSrv::kOther:
				return "0 (another texture)";
			default:
				return "0 (null view)";
			}
		}

		void Analyse(const Maps& a_maps)
		{
			const UINT w = std::min(g_pending.areaW, g_stageDesc.Width);
			const UINT h = std::min(g_pending.areaH, g_stageDesc.Height);
			if (a_maps.bytes == 0 || !a_maps.Has(kFinal) || w == 0 || h == 0) {
				return;
			}
			const auto  start = std::chrono::steady_clock::now();
			const float p33 = g_pending.effP33;
			const float p43 = g_pending.effP43;
			std::array<bool, kRounds> pair{};
			for (int r = 0; r < kRounds; ++r) {
				pair[static_cast<std::size_t>(r)] = a_maps.Has(PreSlot(r)) && a_maps.Has(PostSlot(r));
			}

			// Pass A: raw values of the final depth and the direction of every change a grass call made.
			M::RawStats                         raw{};
			std::array<M::GrassCounts, kRounds> gc{};
			std::uint64_t                       r1VsR0 = 0;
			std::uint64_t                       samples = 0;
			const bool                          haveR1VsR0 = a_maps.Has(PostSlot(0)) && a_maps.Has(PreSlot(1));
			for (UINT y = 0; y < h; y += kSampleStride) {
				for (UINT x = 0; x < w; x += kSampleStride) {
					++samples;
					raw.Add(a_maps.Raw(kFinal, x, y));
					for (int r = 0; r < kRounds; ++r) {
						if (pair[static_cast<std::size_t>(r)]) {
							M::VoteWrite(gc[static_cast<std::size_t>(r)], a_maps.Key(PreSlot(r), x, y), a_maps.Key(PostSlot(r), x, y));
						}
					}
					if (haveR1VsR0 && a_maps.Key(PostSlot(0), x, y) != a_maps.Key(PreSlot(1), x, y)) {
						++r1VsR0;
					}
				}
			}
			std::uint64_t smaller = 0;
			std::uint64_t larger = 0;
			for (const auto& c : gc) {
				smaller += c.smaller;
				larger += c.larger;
			}
			const M::Convention conv = M::DecideConvention(smaller, larger, raw.atZero, raw.atOne, samples, p43);
			const bool          flip = conv.grows != M::DepthGrows(p43);

			// Pass B: bands under that convention.
			std::array<std::uint32_t, M::kBands>                      n{};
			std::array<std::uint32_t, M::kBands>                      anyWrote{};
			std::array<std::uint32_t, M::kBands>                      anyVisible{};
			std::array<std::uint32_t, M::kBands>                      anyNear{};
			std::array<std::array<std::uint32_t, M::kBands>, kRounds> beforeDiff{};
			std::uint64_t                                             r0Wrote = 0;
			std::uint64_t                                             r0WroteKeptInPre1 = 0;
			const bool                                                havePre1 = a_maps.Has(PreSlot(1));
			for (UINT y = 0; y < h; y += kSampleStride) {
				for (UINT x = 0; x < w; x += kSampleStride) {
					const auto  kF = a_maps.Key(kFinal, x, y);
					const float dF = a_maps.Raw(kFinal, x, y);
					const float zF = M::Distance(dF, flip, p33, p43);
					const int   band = M::FinalBand(dF, conv.grows, flip, p33, p43);
					++n[static_cast<std::size_t>(band)];
					bool wrote = false;
					bool visible = false;
					bool nearHit = false;
					for (int r = 0; r < kRounds; ++r) {
						const auto ri = static_cast<std::size_t>(r);
						if (!a_maps.Has(PreSlot(r))) {
							continue;
						}
						const auto kPre = a_maps.Key(PreSlot(r), x, y);
						beforeDiff[ri][static_cast<std::size_t>(band)] += kPre != kF ? 1u : 0u;
						if (!pair[ri]) {
							continue;
						}
						const auto  kPost = a_maps.Key(PostSlot(r), x, y);
						const float dPost = a_maps.Raw(PostSlot(r), x, y);
						const float zPost = M::Distance(dPost, flip, p33, p43);
						const int   own = M::FinalBand(dPost, conv.grows, flip, p33, p43);
						if (M::AddGrass(gc[ri], band, own, kPre, kPost, kF, zPost, zF)) {
							wrote = true;
							visible = visible || kPost == kF;
							nearHit = nearHit || (kPost != kF && M::NearEqual(zPost, zF));
							if (r == 0) {
								++r0Wrote;
								if (havePre1 && a_maps.Key(PreSlot(1), x, y) == kPost) ++r0WroteKeptInPre1;
							}
						}
					}
					anyWrote[static_cast<std::size_t>(band)] += wrote ? 1u : 0u;
					anyVisible[static_cast<std::size_t>(band)] += visible ? 1u : 0u;
					anyNear[static_cast<std::size_t>(band)] += (wrote && !visible && nearHit) ? 1u : 0u;
				}
			}
			const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

			int deferredRounds = 0;
			int afterG9 = 0;
			int late = 0;
			int droppedRounds = 0;
			for (const auto st : g_pending.roundState) {
				deferredRounds += st == RoundState::kDeferred || st == RoundState::kAfterG9 || st == RoundState::kLate || st == RoundState::kDropped ? 1 : 0;
				afterG9 += st == RoundState::kAfterG9 ? 1 : 0;
				late += st == RoundState::kLate ? 1 : 0;
				droppedRounds += st == RoundState::kDropped ? 1 : 0;
			}
			std::string fogDrew;
			if (g_pending.drawStageWater) {
				fogDrew = "fog drawn after the water effects (DrawStage 1, site not tracked)";
			} else if (g_pending.fogSite == -2) {
				fogDrew = "fog not drawn at a RenderEffects call this frame";
			} else {
				fogDrew = std::format("fog drew at site {}", g_pending.fogSite);
			}
			std::string fogGiven = "fog constants: not drawn at the comparison's call";
			if (g_pending.fogGiven) {
				const auto& z = g_pending.fogVpZ;
				const auto& wc = g_pending.fogVpW;
				fogGiven = std::format("fog given DepthP _33 {:.7f} _43 {:.4f} (sky test only), MatrixMode {}, viewProj z column ({:.5f}, {:.5f}, {:.5f}, {:.4f}) "
									   "w column ({:.5f}, {:.5f}, {:.5f}, {:.4f}) (distance = unprojection through its inverse)",
					g_pending.fogP33, g_pending.fogP43, Settings::matrixMode, z[0], z[1], z[2], z[3], wc[0], wc[1], wc[2], wc[3]);
			}
			const auto& f = g_pending.fwd;
			const float yaw = std::atan2(f[0], f[1]) * kRadToDeg;
			const float pitch = std::asin(std::clamp(f[2], -1.0f, 1.0f)) * kRadToDeg;
			logger::info("grassprobe: compare frame {} (#{}): mode {} planned deferred {}, rounds deferred {} (replayed after group 9 {}, late {}, dropped {}); exterior {}; "
						 "effects site {}; camera ({:.1f}, {:.1f}, {:.1f}) forward ({:.3f}, {:.3f}, {:.3f}) yaw {:.1f} pitch {:.1f} deg; texture {}x{} fmt {} srv fmt {} "
						 "dsv fmt {}; fog srv resource is the copied texture {}; scene area {}x{} stride {} samples {} (counts below are samples, x4 for pixels); "
						 "prepass rounds {} (kept {}); proj at prepass _33 {:.7f} _43 {:.4f}, at effects _33 {:.7f} _43 {:.4f}; {}; {}; read back after {} frame(s), "
						 "CPU {:.2f} ms",
				g_pending.frame, g_pending.index, g_pending.mode, g_pending.deferFrame ? 1 : 0, deferredRounds, afterG9, late, droppedRounds,
				g_pending.exterior ? 1 : 0, g_pending.site, g_pending.cam[0], g_pending.cam[1], g_pending.cam[2], f[0], f[1], f[2], yaw, pitch,
				g_stageDesc.Width, g_stageDesc.Height, static_cast<int>(g_stageDesc.Format), g_pending.fogSrvFormat, g_pending.dsvFormat, FogSrvText(), w, h,
				kSampleStride, samples, g_pending.rounds, std::min<std::uint32_t>(g_pending.rounds, kRounds), g_pending.armP33, g_pending.armP43, p33, p43,
				fogDrew, fogGiven, g_pending.polls, ms);

			std::string hist;
			for (int i = 0; i < M::kHistBuckets; ++i) {
				hist += std::format("{}{}", i ? " " : "", raw.hist[static_cast<std::size_t>(i)]);
			}
			logger::info("grassprobe: raw frame {} final depth: min {} max {} at-or-below-1e-7 {} at-or-above-0.9999999 {} outside-0..1 {} of {}; tenths "
						 "[0-0.1 .. 0.9-1] {}",
				g_pending.frame, raw.n ? std::format("{:.7f}", raw.min) : std::string("none"), raw.n ? std::format("{:.7f}", raw.max) : std::string("none"),
				raw.atZero, raw.atOne, raw.outside, samples, hist);
			const float fw = static_cast<float>(w);
			const float fh = static_cast<float>(h);
			const int   cx = M::ProbeGridPixel(4, kProbeCols, fw);
			const int   cy = M::ProbeGridPixel(3, kProbeRows, fh);
			const int   ty = M::ProbeGridPixel(0, kProbeRows, fh);
			const std::string probeRef = g_pending.fogProbeId ? std::format("[PROBE] #{} (p31 = centre, p04 = top)", g_pending.fogProbeId) :
			                             g_pending.fogProbeRequested ? std::string("none (requested; the fog's draw at the comparison's call did not submit one)") :
			                                                           std::string("none (not requested on this comparison)");
			logger::info("grassprobe: raw frame {} pixels: {} | {} | {}; fog probe for this frame {}", g_pending.frame, PixelText(a_maps, "centre", cx, cy, w, h),
				PixelText(a_maps, "top", cx, ty, w, h), PixelText(a_maps, "row0", cx, 0, w, h), probeRef);
			logger::info("grassprobe: convention frame {}: {} by rule {}; votes: writes went smaller {} larger {}; clear thresholds raw<=1e-7 {} raw>=0.9999999 {} of {}; "
						 "projection _43 {:.4f} says {}; distances below = _43 / ({} - _33); bad = negative or no distance",
				g_pending.frame, conv.grows ? "grows" : "reversed", M::RuleName(conv.rule), smaller, larger, raw.atZero, raw.atOne, samples, p43,
				M::DepthGrows(p43) ? "grows" : "reversed", flip ? "(1 - raw)" : "raw");

			const bool deferredFrame = deferredRounds > 0;
			for (int r = 0; r < kRounds; ++r) {
				const auto ri = static_cast<std::size_t>(r);
				const auto st = g_pending.roundState[ri];
				if (st == RoundState::kNone) {
					continue;
				}
				const std::string skip = !g_pending.deferSkip[ri].empty() ? std::format(" (deferral skipped: {})", g_pending.deferSkip[ri]) : std::string();
				const std::string srv = g_pending.srvOther[ri] < 0 ? "?" : g_pending.srvOther[ri] ? "swapped" : "main";
				if (pair[ri]) {
					std::string prev;
					if (st != RoundState::kNormal) {
						prev = g_normalWroteKnown[ri] ? std::format(" | last normal comparison #{} wrote {} in this round", g_normalWroteIndex, g_normalWrote[ri]) :
						                                std::string(" | no normal comparison counted yet for this round");
					}
					logger::info("grassprobe: grass frame {} round {} ({}{}) flags {:X} site s{}{} srv {}: wrote {} (total {}){} | still visible at E {} | near-equal at E {} | "
								 "wrote smaller {} larger {}",
						g_pending.frame, r, D::RoundStateText(st), skip, g_pending.roundFlags[ri], g_pending.grassSite[ri],
						g_pending.replaySite[ri] >= 0 ? std::format(" replayed at s{}", g_pending.replaySite[ri]) : std::string(), srv, BandsText(gc[ri].wrote, n),
						Sum(gc[ri].wrote), prev, BandsText(gc[ri].visible, n), BandsText(gc[ri].nearEqual, n), gc[ri].smaller, gc[ri].larger);
					logger::info("grassprobe: grass frame {} round {} by the grass's own distance: wrote {} | still visible at E {}", g_pending.frame, r,
						BandsText(gc[ri].wroteOwn, n), BandsText(gc[ri].visibleOwn, n));
				} else {
					logger::info("grassprobe: grass frame {} round {} ({}{}): copies missing (taken {:02X}), not counted", g_pending.frame, r, D::RoundStateText(st),
						skip, g_pending.taken);
				}
				if (a_maps.Has(PreSlot(r))) {
					const char* what = !Replayed(st) ? "pre-grass depth (right before the grass call)" :
					                   r == 0       ? "grass-free depth (right before the held call was made)" :
					                                  "depth before the held call (holds earlier rounds' grass)";
					logger::info("grassprobe: deferred {} frame {} round {}: {} differs from final {} (total {} of {})", Replayed(st) ? 1 : 0, g_pending.frame, r, what,
						BandsText(beforeDiff[ri], n), Sum(beforeDiff[ri]), samples);
				}
			}
			logger::info("grassprobe: grass frame {} ({} frame) kept rounds together: wrote {} | still visible at E {} | near-equal at E {} | round 1 before vs round 0 "
						 "after differ {} | of round 0's grass-written samples, round 1's before-copy still equals round 0's after-copy in {} of {}",
				g_pending.frame, deferredFrame ? "deferred" : "normal", BandsText(anyWrote, n), BandsText(anyVisible, n), BandsText(anyNear, n),
				haveR1VsR0 ? std::to_string(r1VsR0) : std::string("n/a"), havePre1 ? std::to_string(r0WroteKeptInPre1) : std::string("n/a"), r0Wrote);

			if (!deferredFrame) {
				for (int r = 0; r < kRounds; ++r) {
					const auto ri = static_cast<std::size_t>(r);
					g_normalWroteKnown[ri] = pair[ri];
					g_normalWrote[ri] = pair[ri] ? Sum(gc[ri].wrote) : 0;
				}
				g_normalWroteIndex = g_pending.index;
			}
			if (Settings::grassProbeDump && M::DumpDue(g_pending.mode, g_pending.index, g_pending.exterior)) {
				WriteDump(a_maps, w, h, conv.grows, flip, p33, p43);
			}
		}

		void PollReadback(ID3D11DeviceContext* a_ctx)
		{
			if (!g_pending.active) {
				return;
			}
			if (g_pending.polls >= kPollFrames) {
				logger::warn("grassprobe: readback not ready after {} frames, dropped (frame {})", kPollFrames, g_pending.frame);
				g_pending.active = false;
				return;
			}
			++g_pending.polls;
			Maps maps{};
			maps.codec = M::CodecOf(g_stageDesc.Format);
			maps.bytes = M::TexelBytes(maps.codec);
			std::uint32_t mapped = 0;
			auto unmapAll = [&]() {
				for (int s = 0; s < kSlots; ++s) {
					if (mapped & Bit(s)) {
						a_ctx->Unmap(g_stage[static_cast<std::size_t>(s)], 0);
					}
				}
			};
			for (int s = 0; s < kSlots; ++s) {
				if ((g_pending.taken & Bit(s)) == 0) {
					continue;
				}
				if (!g_stage[static_cast<std::size_t>(s)]) {
					unmapAll();
					g_pending.active = false;
					return;
				}
				const HRESULT hr = a_ctx->Map(g_stage[static_cast<std::size_t>(s)], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &maps.m[static_cast<std::size_t>(s)]);
				if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
					unmapAll();
					return;
				}
				if (FAILED(hr)) {
					unmapAll();
					logger::error("grassprobe: mapping staging copy {} failed (hr {:08X}); comparisons off", s, static_cast<std::uint32_t>(hr));
					g_compareFailed = true;
					g_pending.active = false;
					return;
				}
				mapped |= Bit(s);
			}
			maps.taken = mapped;
			Analyse(maps);
			unmapAll();
			g_pending.active = false;
		}

		void InspectViews(RE::BSGraphics::DepthStencilData& a_depth, const void* a_mainTex)
		{
			if (auto* dsv = reinterpret_cast<ID3D11DepthStencilView*>(a_depth.views[0])) {
				D3D11_DEPTH_STENCIL_VIEW_DESC dd{};
				dsv->GetDesc(&dd);
				g_pending.dsvFormat = static_cast<int>(dd.Format);
			}
			auto* srv = reinterpret_cast<ID3D11ShaderResourceView*>(a_depth.depthSRV);
			if (!srv) {
				g_pending.fogSrv = FogSrv::kNull;
				return;
			}
			D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
			srv->GetDesc(&sd);
			g_pending.fogSrvFormat = static_cast<int>(sd.Format);
			ID3D11Resource* res = nullptr;
			srv->GetResource(&res);
			g_pending.fogSrv = res && static_cast<const void*>(res) == a_mainTex ? FogSrv::kMain : FogSrv::kOther;
			SafeRelease(res);
		}
	}

	void Install(bool a_effectsHooked)
	{
		if (REL::Module::IsVR()) {
			logger::warn("grassprobe: Skyrim VR is out of scope; probe off");
			return;
		}
		const auto batches = REL::RelocationID(99963, 106609).address();
		const auto renderDepth = REL::RelocationID(100421, 107139).address();
		g_accVtable = RE::VTABLE_BSShaderAccumulator[0].address();
		DescribeRenderWorldStart(batches);
		const auto batchSites = FindSites("RenderBatches", batches, kMaxBatchSites);
		if (batchSites.empty()) {
			logger::error("grassprobe: no direct call to BSShaderAccumulator::RenderBatches found (another plugin may have rewritten them); probe off");
			return;
		}
		WriteSites<BatchSite>(batchSites, std::make_index_sequence<kMaxBatchSites>{});
		g_siteCount = std::min(batchSites.size(), kMaxBatchSites);
		for (std::size_t i = 0; i < g_siteCount; ++i) {
			g_siteAddr[i] = batchSites[i];
			const std::span<const std::uint8_t> code{ reinterpret_cast<const std::uint8_t*>(batchSites[i]), 5 };
			g_siteSlot[i] = CallScan::CallTarget(code, batchSites[i], 0);
		}
		const auto depthSites = FindSites("Main_RenderDepth", renderDepth, kMaxDepthSites);
		if (depthSites.empty()) {
			logger::warn("grassprobe: no direct call to Main_RenderDepth found: the P mark (inside the depth prepass) is never set, so no comparison frame "
						 "is ever armed; order lines only");
		} else {
			WriteSites<DepthSite>(depthSites, std::make_index_sequence<kMaxDepthSites>{});
		}
		g_noEffects = !a_effectsHooked;
		if (g_noEffects) {
			logger::warn("grassprobe: no RenderEffects call is hooked, so there is no point to compare at: order lines only, no copies");
		}
		if (Settings::drawStage == Settings::DrawStage::kAfterWaterEffects) {
			logger::info("grassprobe: DrawStage = 1: the fog draws after the water effects, outside the recorded frame; no F marks, no fog constants or [PROBE]");
		}
		g_active = true;
		logger::info("grassprobe: installed ({} RenderBatches calls, {} Main_RenderDepth calls hooked); mode {} ({}); comparisons every {} s", batchSites.size(),
			depthSites.size(), Settings::grassProbe,
			Settings::grassProbe == 2 ? "every second comparison holds the prepass grass calls back to after group 9, once the state precheck passed" : "copies only",
			std::chrono::duration_cast<std::chrono::seconds>(kLogEvery).count());
	}

	void AtEffects(void* a_accumulator, int a_site)
	{
		(void)a_accumulator;
		if (!g_active || !OnRenderThread()) {
			return;
		}
		DropLogged(g_tracker.Drop(), "the fog's draw point came first");
		if (Settings::grassProbe == 0) {
			return;
		}
		++g_win.effects;
		Entry* entry = NextEntry(true);
		if (entry) {
			*entry = {};
			entry->kind = Kind::kEffects;
			entry->site = Small(static_cast<std::uint32_t>(a_site));
			g_win.lastEffects = static_cast<std::int32_t>(g_win.count - 1);
		} else {
			g_win.lastEffects = -1;
		}
		if (!g_win.armed) {
			++g_win.cmpNotArmed;
			return;
		}
		if (g_win.compared) {
			++g_win.cmpDone;
			return;
		}
		if (g_pending.active) {
			++g_win.cmpBusy;
			return;
		}
		auto* renderer = RE::BSGraphics::Renderer::GetSingleton();
		auto* shadow = RE::BSGraphics::RendererShadowState::GetSingleton();
		if (!renderer || !shadow) {
			return;
		}
		// The main view only, as the fog draw decides it (FogRenderer::Draw): the state cache's slot-0 target is kMAIN.
		if (static_cast<std::uint32_t>(shadow->GetRuntimeData().renderTargets[0]) != static_cast<std::uint32_t>(RE::RENDER_TARGET::kMAIN)) {
			++g_win.cmpNotMainRt;
			return;
		}
		auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(renderer->GetRuntimeData().context);
		auto& depth = renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGET_DEPTHSTENCIL::kMAIN];
		auto* mainTex = reinterpret_cast<ID3D11Texture2D*>(depth.texture);
		D3D11_TEXTURE2D_DESC desc{};
		if (mainTex) {
			mainTex->GetDesc(&desc);
		}
		if (!ctx || !mainTex || static_cast<const void*>(mainTex) != g_stageFor || !SameDesc(desc, g_stageDesc)) {
			++g_win.cmpOtherDepth;
			return;
		}
		if (g_win.copyFail || (g_win.taken & (Bit(kPre0) | Bit(kPost0))) != (Bit(kPre0) | Bit(kPost0))) {
			++g_win.cmpIncomplete;
			return;
		}
		ctx->CopyResource(g_stage[kFinal], mainTex);
		g_win.taken |= Bit(kFinal);

		const auto&  view = shadow->GetRuntimeData().cameraData.getEye();   // as FogRenderer::Draw reads it
		const float* p = reinterpret_cast<const float*>(&view.projMat);
		const float* v = reinterpret_cast<const float*>(&view.viewMat);
		const auto&  adjust = shadow->GetRuntimeData().posAdjust.getEye();
		const auto   ratios = FogRenderer::ReadSceneRatios();   // the same place FogRenderer::Draw reads them
		const float  ratioW = ratios.width;
		const float  ratioH = ratios.height;

		g_pending = {};
		g_pending.active = true;
		g_pending.frame = g_frame;
		g_pending.index = ++g_issued;
		g_pending.mode = Settings::grassProbe;
		g_pending.deferFrame = g_win.deferFrame;
		auto* player = RE::PlayerCharacter::GetSingleton();
		auto* cell = player ? player->GetParentCell() : nullptr;
		g_pending.exterior = cell && !cell->IsInteriorCell();
		g_pending.site = a_site;
		g_pending.taken = g_win.taken;
		g_pending.rounds = g_tracker.Rounds();
		for (int r = 0; r < kRounds; ++r) {
			g_pending.roundState[static_cast<std::size_t>(r)] = g_tracker.State(r);
		}
		g_pending.roundFlags = g_win.roundFlags;
		g_pending.grassSite = g_win.grassSite;
		g_pending.replaySite = g_win.replaySite;
		g_pending.deferSkip = g_win.deferSkip;
		g_pending.srvOther = g_win.srvOther;
		g_pending.armP33 = g_win.armP33;
		g_pending.armP43 = g_win.armP43;
		g_pending.effP33 = p[10];
		g_pending.effP43 = p[14];
		g_pending.cam = { adjust.x, adjust.y, adjust.z };
		g_pending.fwd = { v[2], v[6], v[10] };   // viewMat column 2: the view direction in world space (FogRenderer's probe header)
		g_pending.areaW = std::max(1u, static_cast<UINT>(std::floor(static_cast<float>(desc.Width) * ratioW + 0.5f)));
		g_pending.areaH = std::max(1u, static_cast<UINT>(std::floor(static_cast<float>(desc.Height) * ratioH + 0.5f)));
		g_pending.drawStageWater = Settings::drawStage == Settings::DrawStage::kAfterWaterEffects;
		InspectViews(depth, mainTex);
		g_win.probeIdBefore = FogRenderer::GetStats().probeId;
		if (M::FogProbeDue(Settings::grassProbe, g_pending.index, g_fogProbes) && !g_pending.drawStageWater) {
			FogRenderer::RequestProbe();   // served by the fog's draw of this very call or by none (FogRenderer::Draw)
			g_pending.fogProbeRequested = true;
		}
		UpdatePrecheck();
		g_win.compared = true;
		g_win.cmpEffectsNo = g_win.effects;
		if (entry) {
			entry->snap = true;   // printed as C: the comparison was issued here
		}
	}

	void NoteFogDraw(int a_site)
	{
		if (!g_active || !OnRenderThread() || Settings::grassProbe == 0) {
			return;
		}
		g_win.fogSite = a_site;
		if (g_win.lastEffects >= 0 && static_cast<std::uint32_t>(g_win.lastEffects) < g_win.count) {
			g_win.calls[g_win.lastEffects].fog = true;
		}
		// Only the draw of the RenderEffects call that issued the comparison (NoteFogDraw follows that call's AtEffects in
		// the same hook, so the call count is still the one recorded at issue).
		if (g_pending.active && g_pending.frame == g_frame && g_win.compared && g_win.effects == g_win.cmpEffectsNo && !g_pending.fogGiven) {
			const auto& fs = FogRenderer::GetStats();
			g_pending.fogGiven = true;
			g_pending.fogP33 = fs.depthP33;
			g_pending.fogP43 = fs.depthP43;
			for (int i = 0; i < 4; ++i) {
				g_pending.fogVpZ[static_cast<std::size_t>(i)] = fs.viewProjZ[i];
				g_pending.fogVpW[static_cast<std::size_t>(i)] = fs.viewProjW[i];
			}
			if (fs.probeId != g_win.probeIdBefore) {
				g_pending.fogProbeId = fs.probeId;
				++g_fogProbes;
			}
		}
	}

	void EndFrame()
	{
		if (!g_active) {
			return;
		}
		const DWORD self = GetCurrentThreadId();
		DWORD       render = g_renderThread.load(std::memory_order_relaxed);
		if (render == 0) {
			g_renderThread.store(self, std::memory_order_relaxed);
			render = self;
			logger::info("grassprobe: render thread {}", self);
		}
		if (self != render) {
			return;
		}
		DropLogged(g_tracker.Drop(), "the frame ended first");
		if (Settings::grassProbe != 0) {
			if (g_pending.active && g_pending.frame == g_frame) {
				g_pending.fogSite = g_win.fogSite;
			}
			if (g_win.armed) {
				LogStates();
				if (!g_win.compared) {
					logger::info("grassprobe: frame {} was armed (index {}, deferred frame {}: {}) but not compared ({} RenderEffects calls: busy {}, target not main {}, "
								 "other depth {}, incomplete copies {}); its copies are discarded and the next comparison keeps index {}",
						g_frame, g_win.armIndex, g_win.deferFrame ? 1 : 0, RoundsText(), g_win.effects, g_win.cmpBusy, g_win.cmpNotMainRt, g_win.cmpOtherDepth,
						g_win.cmpIncomplete, g_issued + 1);
				}
			}
			if (auto* renderer = RE::BSGraphics::Renderer::GetSingleton()) {
				if (auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(renderer->GetRuntimeData().context)) {
					PollReadback(ctx);
				}
			}
			// The frame that issued a comparison is the one logged, so its S, X, R and C marks match the compare line's
			// frame; with no comparison, a frame every 6 s, so the reasons show.
			const auto now = std::chrono::steady_clock::now();
			if (g_win.compared || g_win.armed || now - g_lastOrderLog >= kLogEvery + std::chrono::seconds(1)) {
				g_lastOrderLog = now;
				LogOrder();
			}
		} else if (g_stageFor || g_stage[kFinal]) {
			ReleaseStaging();   // switched off at run time: give the depth-sized textures back
			logger::info("grassprobe: GrassProbe set to 0: staging textures released; the hooks only pass calls on");
		}
		g_tracker.EndFrame();   // this frame's round structure decides whether mode 2 may defer next frame
		// Every frame, the setting 0 included, so 2 -> 0 -> 2 or a cell change between two comparisons is seen.
		const D::Context ctx = CurrentContext();
		if (g_precheck.Invalidate(ctx)) {
			logger::info("grassprobe: frame {}: state precheck of normal comparison #{} dropped (context changed: mode {} -> {}, {} -> {}); no deferral until a "
						 "new normal comparison passes",
				g_frame, g_precheck.Index(), g_precheck.Ctx().mode, ctx.mode, g_precheck.Ctx().exterior ? "exterior worldspace" : "interior cell",
				ctx.exterior ? "exterior worldspace" : ctx.place ? "interior cell" : "no cell");
		}
		++g_frame;
		g_win.Reset();
	}
}
