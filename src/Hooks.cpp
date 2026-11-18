#include "PCH.h"

#include "Hooks.h"

#include "CallScan.h"

#include <Windows.h>
#include "FogRenderer.h"
#include "GrassProbe.h"
#include "SelfTest.h"
#include "Settings.h"

namespace Hooks
{
	namespace
	{
		// Address Library IDs (SE, AE), from Community Shaders' FrameAnnotations.cpp / Deferred.h:
		//   Main_RenderPlayerView         35560 / 36559
		//   Main_RenderWaterEffects       35561 / 36560
		//   Main_RenderWorld (body)       99938 / 106583
		//   BSShaderAccumulator::RenderEffects 99940 / 106585
		// The call offsets are not hard-coded: they are found by scanning, because the exe is
		// encrypted on disk and the offsets could not be read offline.
		constexpr std::size_t kMaxScanBytes = 0x4000;

		SiteReport  g_effects{ "before RenderEffects" };
		SiteReport  g_water{ "after RenderWaterEffects" };
		OrderReport g_order{};
		constexpr std::size_t kEventsKept = 48;

		void NoteEvent(char a_event)
		{
			if (g_order.firstEvents.size() < kEventsKept) {
				g_order.firstEvents.push_back(a_event);
			}
		}

		std::uintptr_t Rva(std::uintptr_t a_address) { return a_address - REL::Module::get().base(); }

		// The bytes of the function that starts at a_address, bounded by its unwind entry (.pdata) and
		// by the end of .text, so the scan never strays into the next function or off the section.
		std::span<const std::uint8_t> CodeAt(std::uintptr_t a_address)
		{
			DWORD64    imageBase = 0;
			const auto* entry = RtlLookupFunctionEntry(static_cast<DWORD64>(a_address), &imageBase, nullptr);
			std::size_t length = 0;
			if (entry && imageBase + entry->BeginAddress == a_address && entry->EndAddress > entry->BeginAddress) {
				length = entry->EndAddress - entry->BeginAddress;
			} else {
				logger::warn("no unwind entry starts at RVA {:X}; scanning at most {:X} bytes", Rva(a_address), kMaxScanBytes);
				length = kMaxScanBytes;
			}
			const auto text = REL::Module::get().segment(REL::Segment::textx);
			const auto textEnd = text.address() + text.size();
			if (a_address < text.address() || a_address >= textEnd) {
				logger::error("RVA {:X} is outside .text; not scanned", Rva(a_address));
				return {};
			}
			length = std::min({ length, kMaxScanBytes, static_cast<std::size_t>(textEnd - a_address) });
			return { reinterpret_cast<const std::uint8_t*>(a_address), length };
		}

		constexpr std::size_t kMaxEffectSites = 8;

		// One thunk per hooked RenderEffects call site; each keeps the original target it replaced.
		template <std::size_t I>
		struct RenderEffectsSite
		{
			static void thunk(void* a_accumulator, std::uint32_t a_flags)
			{
				++g_order.effectsCalls;
				NoteEvent(static_cast<char>('0' + I));
				GrassProbe::AtEffects(a_accumulator, static_cast<int>(I));
				if (Settings::drawStage == Settings::DrawStage::kBeforeEffects) {
					const auto drawsBefore = FogRenderer::GetStats().draws;
					FogRenderer::Draw(g_effects.name, true, static_cast<int>(I));
					if (FogRenderer::GetStats().draws != drawsBefore) {
						GrassProbe::NoteFogDraw(static_cast<int>(I));
					}
					SelfTest::Tick();
				}
				func(a_accumulator, a_flags);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct RenderEffectsCall
		{
			static void thunk(void* a_accumulator, std::uint32_t a_flags)
			{
				++g_order.effectsCalls;
				NoteEvent('E');
				GrassProbe::AtEffects(a_accumulator, 0);
				if (Settings::drawStage == Settings::DrawStage::kBeforeEffects) {
					const auto drawsBefore = FogRenderer::GetStats().draws;
					FogRenderer::Draw(g_effects.name, true, 0);
					if (FogRenderer::GetStats().draws != drawsBefore) {
						GrassProbe::NoteFogDraw(0);
					}
					SelfTest::Tick();
				}
				func(a_accumulator, a_flags);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct RenderWaterEffectsCall
		{
			static void thunk()
			{
				func();
				FogRenderer::NextFrame();  // the player view runs once per frame
				GrassProbe::EndFrame();    // does nothing unless [Debug] GrassProbe installed it
				++g_order.waterCalls;
				NoteEvent('W');
				if (Settings::drawStage == Settings::DrawStage::kAfterWaterEffects) {
					FogRenderer::Draw(g_water.name, true, -1);
					SelfTest::Tick();
				}
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		// Finds the first call to a_target inside each of a_hosts and writes our thunk there.
		template <class Hook>
		void InstallAt(SiteReport& a_report, std::initializer_list<std::uintptr_t> a_hosts, std::uintptr_t a_target, bool a_hasFallback = false)
		{
			a_report.expected = Rva(a_target);
			for (const auto host : a_hosts) {
				const auto hits = CallScan::FindCallsTo(CodeAt(host), host, a_target);
				if (hits.empty()) {
					continue;
				}
				a_report.found = true;
				a_report.hits = hits.size();
				const auto site = host + hits.front().offset;
				a_report.site = Rva(site);
				if (hits.size() > 1) {
					logger::warn("{}: {} calls found in the host at RVA {:X}; hooking the first at RVA {:X}", a_report.name,
						hits.size(), Rva(host), a_report.site);
				}
				// The scan matched only calls whose target is the original game function, so no other
				// plugin has rewritten this call. (A call already redirected elsewhere would not match.)
				const auto previous = SKSE::GetTrampoline().write_call<5>(site, Hook::thunk);
				Hook::func = previous;
				a_report.previous = Rva(previous);
				a_report.written = previous == a_target;
				a_report.hostRva = Rva(host);
				logger::info("{}: hooked call at RVA {:X} in the function at RVA {:X} (calls RVA {:X}, {} bytes scanned)",
					a_report.name, a_report.site, a_report.hostRva, a_report.previous, CodeAt(host).size());
				return;
			}
			if (a_hasFallback) {
				logger::info("{}: no call to RVA {:X} in the functions scanned first; searching all of .text", a_report.name, a_report.expected);
			} else {
				logger::error("{}: no call to RVA {:X} found in the scanned functions; this draw stage is unavailable",
					a_report.name, a_report.expected);
			}
		}

		template <std::size_t... I>
		void InstallEffectSites(const std::vector<std::uintptr_t>& a_sites, std::index_sequence<I...>)
		{
			((I < a_sites.size() ? (void)(RenderEffectsSite<I>::func =
											 SKSE::GetTrampoline().write_call<5>(a_sites[I], RenderEffectsSite<I>::thunk))
								 : (void)0),
				...);
		}

		// RenderEffects is not called from the two functions scanned first, so look through all of .text.
		// O(size of .text), about 24 MB, once at load.
		void InstallEffectsAnywhere(std::uintptr_t a_target)
		{
			const auto text = REL::Module::get().segment(REL::Segment::textx);
			const std::span<const std::uint8_t> code{ reinterpret_cast<const std::uint8_t*>(text.address()), text.size() };
			const auto start = std::chrono::steady_clock::now();
			const auto hits = CallScan::FindCallsTo(code, text.address(), a_target);
			const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			logger::info("{}: whole .text scan ({} MB, {:.1f} ms) found {} calls to RVA {:X}", g_effects.name,
				text.size() >> 20, ms, hits.size(), Rva(a_target));
			std::vector<std::uintptr_t> sites;
			for (const auto& hit : hits) {
				const auto site = text.address() + hit.offset;
				DWORD64    imageBase = 0;
				const auto* entry = RtlLookupFunctionEntry(static_cast<DWORD64>(site), &imageBase, nullptr);
				const auto  fnRva = entry ? static_cast<std::uintptr_t>(entry->BeginAddress) : 0;
				logger::info("  site {}: call at RVA {:X} in the function at RVA {:X}", sites.size(), Rva(site), fnRva);
				if (sites.size() < kMaxEffectSites) {
					sites.push_back(site);
				}
			}
			if (sites.empty()) {
				logger::error("{}: no call to RVA {:X} found anywhere in .text; this draw stage is unavailable", g_effects.name, Rva(a_target));
				return;
			}
			InstallEffectSites(sites, std::make_index_sequence<kMaxEffectSites>{});
			g_effects.found = true;
			g_effects.written = true;
			g_effects.hits = hits.size();
			g_effects.site = Rva(sites.front());
			g_effects.previous = Rva(a_target);
		}
	}

	void Install()
	{
		// 14 bytes per hooked call, up to 10 calls; [Debug] GrassProbe adds room for its own (GrassProbe.h).
		SKSE::AllocTrampoline(Settings::grassProbe != 0 ? 256 + GrassProbe::kTrampolineBytes : 256);

		const auto playerView = REL::RelocationID(35560, 36559).address();
		const auto renderWorld = REL::RelocationID(99938, 106583).address();
		const auto renderEffects = REL::RelocationID(99940, 106585).address();
		const auto waterEffects = REL::RelocationID(35561, 36560).address();

		InstallAt<RenderEffectsCall>(g_effects, { renderWorld, playerView }, renderEffects, true);
		if (!g_effects.written) {
			InstallEffectsAnywhere(renderEffects);
		}
		InstallAt<RenderWaterEffectsCall>(g_water, { playerView }, waterEffects);
		if (!g_water.written) {
			// Frames are counted in that call (FogRenderer::NextFrame): without it every draw after the first looks
			// like a second draw of the same frame and is skipped.
			logger::error("the water-effects call could not be hooked (another plugin may have redirected it): frames cannot be "
						  "counted, so no fog will be drawn");
		}

		if (Settings::grassProbe != 0) {
			if (!g_water.written) {
				logger::error("grassprobe: the water-effects call is not hooked, so the probe never sees a frame end; probe off");
			} else {
				GrassProbe::Install(g_effects.written);
			}
		}

		const auto& chosen = Settings::drawStage == Settings::DrawStage::kBeforeEffects ? g_effects : g_water;
		if (!chosen.written) {
			logger::error("the chosen draw stage '{}' could not be hooked: no fog will be drawn. Try DrawStage={} in GroundFog.ini",
				chosen.name, Settings::drawStage == Settings::DrawStage::kBeforeEffects ? 1 : 0);
		}
	}

	const SiteReport&  Effects() { return g_effects; }
	const SiteReport&  Water() { return g_water; }
	const OrderReport& Order() { return g_order; }
}
