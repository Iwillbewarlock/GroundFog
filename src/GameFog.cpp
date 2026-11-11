#include "PCH.h"

#include "GameFog.h"

#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace GameFog
{
	namespace
	{
		std::unordered_map<RE::FormID, RE::TESWeather::FogData> g_original;   // main thread only
		std::unordered_map<RE::FormID, RE::TESWeather::FogData> g_written;    // what Apply last wrote
		bool                                                    g_collected{ false };
		std::atomic<float>                                      g_applied{ 1.0f };
		std::atomic<float>                                      g_requested{ 1.0f };
		std::atomic<bool>                                       g_pending{ false };   // a queued Apply has not run yet
		std::mutex                                              g_scaledLock;
		std::unordered_set<RE::FormID>                          g_scaled;   // weathers holding our scaled values
	}

	void Apply(float a_scale)
	{
		if (!(a_scale > 0.0f) || !std::isfinite(a_scale)) {
			logger::warn("game fog: scale {} is not a positive number; left as it is", a_scale);
			return;
		}
		auto* data = RE::TESDataHandler::GetSingleton();
		if (!data) {
			logger::warn("game fog: no data handler yet; scale {} not applied", a_scale);
			return;
		}
		auto& weathers = data->GetFormArray<RE::TESWeather>();
		if (!g_collected) {
			g_original.reserve(weathers.size());
			for (auto* weather : weathers) {
				if (weather) {
					g_original.emplace(weather->GetFormID(), weather->fogData);
				}
			}
			g_collected = true;
		}
		std::unordered_set<RE::FormID> scaled;
		std::size_t changed = 0;
		std::size_t skipped = 0;
		for (auto* weather : weathers) {
			if (!weather) {
				continue;
			}
			const auto it = g_original.find(weather->GetFormID());
			if (it == g_original.end()) {
				continue;   // created after the first Apply: left alone
			}
			// Someone else (e.g. Community Shaders' weather editor) changed the record after our last write: theirs wins.
			if (const auto w = g_written.find(weather->GetFormID());
				w != g_written.end() && std::memcmp(&w->second, &weather->fogData, sizeof(RE::TESWeather::FogData)) != 0) {
				++skipped;
				continue;
			}
			auto fog = it->second;
			fog.dayNear *= a_scale;
			fog.dayFar *= a_scale;
			fog.nightNear *= a_scale;
			fog.nightFar *= a_scale;
			weather->fogData = fog;
			g_written[weather->GetFormID()] = fog;
			scaled.insert(weather->GetFormID());
			++changed;
		}
		{
			const std::scoped_lock lock(g_scaledLock);
			g_scaled = std::move(scaled);
		}
		g_applied = a_scale;
		logger::info("game fog: near/far distances of {} weathers set to {:.2f} x their record values ({} left alone: changed by another mod)",
			changed, a_scale, skipped);
	}

	void RequestApply(float a_scale)
	{
		g_requested = a_scale;
		if (a_scale == g_applied.load() && !g_pending.load()) {
			return;
		}
		if (g_pending.exchange(true)) {
			return;   // the queued task reads g_requested when it runs
		}
		if (const auto* tasks = SKSE::GetTaskInterface()) {
			tasks->AddTask([]() {
				// Apply the newest request; a request that arrives while Apply runs is picked up by the next pass.
				for (int pass = 0; pass < 4; ++pass) {
					const float want = g_requested.load();
					if (want != g_applied.load()) {
						Apply(want);
					}
					if (g_requested.load() == want) {
						break;
					}
				}
				g_pending = false;
				if (g_requested.load() != g_applied.load()) {
					RequestApply(g_requested.load());   // arrived after the last pass
				}
			});
		} else {
			g_pending = false;
		}
	}

	float ScaleFor(const RE::TESWeather* a_weather)
	{
		if (!a_weather) {
			return 1.0f;
		}
		const std::scoped_lock lock(g_scaledLock);
		return g_scaled.contains(a_weather->GetFormID()) ? g_applied.load() : 1.0f;
	}
}
