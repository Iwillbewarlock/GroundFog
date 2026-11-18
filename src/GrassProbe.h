#pragma once

// [Debug] GrassProbe = 1 / 2: a log-only probe of the scene depth around the game's grass draws (DEBUG-LEDGER.md
// 2026-11-15, H-probe-rev, H-pregrass-2..6). 1 copies the main depth before and after the depth prepass's grass calls
// and at the fog's draw point; 2 also, on every second comparison, moves the prepass grass call to after its round's
// group-9 call to get the depth without grass. Nothing is installed and nothing runs when GrassProbe is 0 at load.
namespace GrassProbe
{
	// Calls hooked at most: BSShaderAccumulator::RenderBatches sites, then Main_RenderDepth sites.
	inline constexpr std::size_t kMaxBatchSites = 64;
	inline constexpr std::size_t kMaxDepthSites = 4;
	// Trampoline space the probe may use (14 bytes per hooked call); Hooks::Install adds it when the probe is on.
	inline constexpr std::size_t kTrampolineBytes = (kMaxBatchSites + kMaxDepthSites) * 14;

	// At SKSE load, after the trampoline is allocated; only when Settings::grassProbe != 0. a_effectsHooked: a RenderEffects
	// call is hooked (without one there is no point to compare at, so nothing is copied).
	void Install(bool a_effectsHooked);
	// From the RenderEffects hooks, before the fog draws: copies the final depth and issues the comparison of an armed
	// frame; on some comparisons asks the fog for a [PROBE] readback at the draw that follows.
	void AtEffects(void* a_accumulator, int a_site);
	// From the RenderEffects hooks, after FogRenderer::Draw, when it drew: marks that call in the order line (F) and
	// keeps the constants the fog shader was given.
	void NoteFogDraw(int a_site);
	// From the player view's water-effects hook, once per frame: reads a finished comparison back, logs the frame.
	void EndFrame();
}
