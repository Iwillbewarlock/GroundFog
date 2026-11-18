#include "PCH.h"

#include "SelfTest.h"

#include "FogRenderer.h"
#include "Hooks.h"
#include "Settings.h"
#include "Terrain.h"
#include "WindField.h"

#include <Windows.h>

namespace SelfTest
{
	namespace
	{
		constexpr std::uint64_t kFramesToWait = 600;   // about 10 s at 60 fps of drawn frames outdoors
		constexpr double        kDrawBudgetMs = 0.7;   // PLAN.md section 10
		constexpr double        kFillBudgetMs = 4.0;   // one frame of CPU height queries, worst case

		bool g_reported{ false };
		int  g_pass{ 0 };
		int  g_fail{ 0 };

		void Result(bool a_ok, std::string_view a_name, std::string_view a_detail)
		{
			(a_ok ? g_pass : g_fail)++;
			logger::info("[SELFTEST] {} {} : {}", a_ok ? "PASS" : "FAIL", a_name, a_detail);
		}

		void Info(std::string_view a_name, std::string_view a_detail)
		{
			logger::info("[SELFTEST] INFO {} : {}", a_name, a_detail);
		}

		void ReportSite(const Hooks::SiteReport& a_site, bool a_required)
		{
			const auto detail = std::format("found={} written={} hits={} site=RVA {:X} in function RVA {:X}, calls RVA {:X} (expected {:X})",
				a_site.found, a_site.written, a_site.hits, a_site.site, a_site.hostRva, a_site.previous, a_site.expected);
			// Whether the hook is the right place is only shown later, by the draw counts and the in-game look.
			if (a_required) {
				Result(a_site.written, std::format("hook written '{}'", a_site.name), detail);
			} else {
				Info(std::format("hook '{}'", a_site.name), detail);
			}
		}
	}

	void ReportLoad()
	{
		if (!Settings::selfTest) {
			return;
		}
		const bool effectsChosen = Settings::drawStage == Settings::DrawStage::kBeforeEffects;
		ReportSite(Hooks::Effects(), effectsChosen);
		ReportSite(Hooks::Water(), !effectsChosen);
		Info("Community Shaders", GetModuleHandleW(L"CommunityShaders.dll") ? "loaded" : "not loaded");
	}

	void Tick()
	{
		if (!Settings::selfTest || g_reported) {
			return;
		}
		const auto& fs = FogRenderer::GetStats();
		// Count only calls made while there was an outdoor world to draw into.
		const auto worldCalls = fs.calls - fs.skipNoWorld;
		// Wait for real draws: right after a load the world is not ready and every count would be meaningless.
		if (fs.draws < kFramesToWait) {
			return;
		}
		g_reported = true;

		logger::info("[SELFTEST] ---- report after {} calls outdoors ----", worldCalls);

		const bool effectsChosen = Settings::drawStage == Settings::DrawStage::kBeforeEffects;
		const auto& site = effectsChosen ? Hooks::Effects() : Hooks::Water();
		const auto& order = Hooks::Order();
		const auto  stageCalls = effectsChosen ? order.effectsCalls : order.waterCalls;
		Result(site.written && stageCalls > 0, "hook runs", std::format("'{}' ran {} times", site.name, stageCalls));

		Result(fs.draws > 0, "fog drawn", std::format("{} draws of {} outdoor calls", fs.draws, worldCalls));
		const double notMain = worldCalls ? static_cast<double>(fs.skipNotMainTarget) / static_cast<double>(worldCalls) : 1.0;
		if (effectsChosen) {
			Result(notMain <= 0.5, "main target bound at the hook",
				std::format("{:.0f}% of outdoor calls had another render target bound", notMain * 100.0));
		} else {
			Info("main target", "the player-view hook binds the main target itself");
		}
		Info("skips", std::format("otherCamera={} notMain={} noReadOnlyDepth={} notReady={} sameFrame={} indoors/noWorld={}",
						  fs.skipOtherCamera, fs.skipNotMainTarget, fs.skipNoReadOnlyDepth, fs.skipNotReady, fs.duplicates, fs.skipNoWorld));
		Info("weather region", FogRenderer::RegionValue());
		Info("worldspace rule", std::format("{} ({} calls skipped in worldspaces set to off)", FogRenderer::WorldspaceRule(), fs.skipWorldspaceOff));

		Result(fs.shadersCompiled, "shaders compile", "raymarch vs_5_0 and the march and composite ps_5_0 from the embedded source");
		Result(fs.depthSrv, "depth readable",
			std::format("depth SRV={} (the raymarch reads it), readOnly DSV={} (not needed)", fs.depthSrv, fs.readOnlyDepth));
		Result(std::fabs(fs.proj34 - 1.0f) < 1e-3f && std::fabs(fs.proj44) < 1e-3f, "projection layout",
			std::format("_34={} _44={} (linear depth assumes 1 and 0)", fs.proj34, fs.proj44));
		Result(fs.stateChecks > 0 && fs.stateMismatches == 0 && fs.probeMismatches == 0, "state restored",
			std::format("{} checks, {} guard mismatches, {} wider-probe mismatches (VS/PS t0-t127, PS b0-b13)", fs.stateChecks,
				fs.stateMismatches, fs.probeMismatches));
		Info("viewport", std::format("{}x{} on a {}x{} target, depth range {}..{}", fs.viewportW, fs.viewportH, fs.targetW,
							 fs.targetH, fs.minDepth, fs.maxDepth));
		// Where the dynamic-resolution ratios were read (SceneRatio.h). Not found = CommonLib's fixed offset, which on
		// one user's runtime read the TAA table and shrank the fog to 1911x1280 of a 3440x1440 screen.
		Result(fs.ratioTableFound, "dynamic resolution ratios located",
			std::format("{:.4f} x {:.4f} at State+0x{:X} (CommonLib's offset 0x{:X}){}", fs.ratioW, fs.ratioH, fs.ratioOffset, fs.ratioExpected,
				fs.ratioTableFound ? "" : "; the TAA table was not found, so CommonLib's offset is used unchecked on this runtime (see the warning line with the floats)"));

		const auto samples = FogRenderer::GpuSamples();
		Result(samples > 0 && FogRenderer::GpuAverageMs() <= kDrawBudgetMs, "GPU time",
			std::format("avg {:.3f} ms, max {:.3f} ms over {} samples (budget {:.1f} ms)", FogRenderer::GpuAverageMs(),
				FogRenderer::GpuMaxMs(), samples, kDrawBudgetMs));
		Info("wind GPU time", std::format("step + columns avg {:.3f} ms, max {:.3f} ms (sample {}) over {} samples; fog + wind avg {:.3f} ms",
			FogRenderer::WindGpuAverageMs(), FogRenderer::WindGpuMaxMs(), FogRenderer::WindGpuMaxSample(), FogRenderer::WindGpuSamples(),
			FogRenderer::GpuAverageMs() + FogRenderer::WindGpuAverageMs()));

		const auto& ts = Terrain::GetStats();
		const auto& grid = Terrain::GetGrid();
		const auto  cells = static_cast<std::uint32_t>(grid.size * grid.size);
		Result(ts.fillMsMax <= kFillBudgetMs, "height fill time",
			std::format("max {:.2f} ms in one frame, {:.1f} ms total, {} queries", ts.fillMsMax, ts.fillMsTotal, ts.queries));
		Result(ts.unknownCells < cells / 4, "heights known",
			std::format("{} of {} cells unknown, {} failed queries, {} recenters (last shift {}x{} cells), {} worldspace resets",
				ts.unknownCells, cells, ts.failures, ts.recenters, ts.lastShiftX, ts.lastShiftY, ts.worldspaceResets));
		Info("water", std::format("{} queries raised to the water surface, largest raise {:.1f} units", ts.waterRaised, ts.waterRaiseMax));
		if (ts.verifyChecks > 0) {
			Result(ts.verifyMaxDiff < 1.0f, "cache matches the game after a recenter",
				std::format("{} cells asked again, largest difference {:.2f} units", ts.verifyChecks, ts.verifyMaxDiff));
		} else {
			Info("cache after a recenter", "not checked: walk at least 300 units during the test");
		}

		const auto& ws = WindField::GetStats();
		Result(!Settings::enableWind || (ws.created && ws.steps > 0 && ws.resets * 100 <= ws.steps), "wind field steps",
			std::format("created={} steps={} shifts={} resets={} {}", ws.created, ws.steps, ws.shifts, ws.resets, ws.error));
		Info("wind inputs", std::format("Sky windSpeed {:.3f}..{:.3f}, windAngle {:.3f}..{:.3f} (units not documented), last actors {}, last blasts {}, impacts queued {}",
								ws.windSpeedMin, ws.windSpeedMax, ws.windAngleMin, ws.windAngleMax, ws.lastActors, ws.lastBlasts, ws.impacts));

		Info("hook order", std::format("RenderEffects calls={} RenderWaterEffects calls={} first events '{}'",
							   order.effectsCalls, order.waterCalls, order.firstEvents));

		logger::info("[SELFTEST] ---- {} PASS, {} FAIL ----", g_pass, g_fail);
	}
}
