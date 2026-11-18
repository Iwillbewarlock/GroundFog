#include "PCH.h"

#include "WeatherFog.h"

#include "EditorID.h"
#include "GameFog.h"
#include "Settings.h"
#include "WeatherRules.h"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace WeatherFog
{
	namespace
	{
		std::mutex                                   g_lock;    // guards the table and the CS factor
		std::unordered_map<RE::FormID, float>        g_factor;  // weather -> factor
		std::unordered_map<RE::FormID, std::string>  g_name;    // weather -> EditorID (for logs)
		// Written on the main thread (Build), read on the render thread. Settings scalars that Build reads may be
		// re-read by an ini reload at the same moment; aligned float stores do not tear on x64, a reload only
		// mixes old and new values for one Build, and the reload queues another Build.
		std::atomic<float>                           g_csFactor{ 1.0f };
		std::atomic<Renderer>                        g_renderer{ Renderer::kVanilla };
		std::atomic<int>                             g_autoColorSpace{ 0 };
		std::atomic<bool>                            g_resetSmooth{ false };   // a game was loaded
		std::chrono::steady_clock::time_point        g_lastCsCheck{};          // render thread only
		std::filesystem::file_time_type              g_csWriteTime{};          // render thread only
		bool                                         g_csChecked{ false };     // render thread only
		bool                                         g_csExisted{ false };     // render thread only
		std::atomic<bool>                            g_pending{ false };
		float                                        g_smoothed{ -1.0f };   // render thread only
		std::chrono::steady_clock::time_point        g_lastTime{};
		constexpr float                              kSmoothSeconds = 3.0f;

		// Community Shaders' settings files as the game sees them (MO2 maps overwrite into Data). Absolute, next to
		// the game exe: a relative path depends on the working folder the launcher left (Settings.cpp IniPath).
		const std::filesystem::path& CsDir()
		{
			static const std::filesystem::path dir = [] {
				wchar_t     exe[MAX_PATH]{};
				const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
				const std::filesystem::path base =
					(n > 0 && n < MAX_PATH) ? std::filesystem::path(exe).parent_path() : std::filesystem::current_path();
				return base / L"Data" / L"SKSE" / L"Plugins" / L"CommunityShaders";
			}();
			return dir;
		}
		std::filesystem::path CsSettings() { return CsDir() / L"SettingsUser.json"; }
		std::filesystem::path CsDefault() { return CsDir() / L"SettingsDefault.json"; }

		// The JSON object that follows `"section"`. A small scan, not a JSON parser: enough for CS's flat settings
		// blocks; returns false when the section is missing.
		bool JsonBlock(const std::string& a_text, std::string_view a_section, std::string& a_block)
		{
			const auto s = a_text.find("\"" + std::string(a_section) + "\"");
			if (s == std::string::npos) {
				return false;
			}
			const auto open = a_text.find('{', s);
			if (open == std::string::npos) {
				return false;
			}
			int         depth = 0;
			std::size_t close = open;
			for (; close < a_text.size(); ++close) {
				if (a_text[close] == '{') ++depth;
				if (a_text[close] == '}' && --depth == 0) break;
			}
			a_block = a_text.substr(open, close - open);
			return true;
		}

		// Finds `"key": value` inside a section; returns false when the section or key is missing.
		bool JsonNumberIn(const std::string& a_text, std::string_view a_section, std::string_view a_key, double& a_out)
		{
			std::string block;
			if (!JsonBlock(a_text, a_section, block)) {
				return false;
			}
			const auto        k = block.find("\"" + std::string(a_key) + "\"");
			if (k == std::string::npos) {
				return false;
			}
			const auto colon = block.find(':', k);
			if (colon == std::string::npos) {
				return false;
			}
			std::size_t p = colon + 1;
			while (p < block.size() && (block[p] == ' ' || block[p] == '\t' || block[p] == '\r' || block[p] == '\n')) ++p;
			if (block.compare(p, 4, "true") == 0) { a_out = 1.0; return true; }
			if (block.compare(p, 5, "false") == 0) { a_out = 0.0; return true; }
			char* end = nullptr;
			a_out = std::strtod(block.c_str() + p, &end);
			return end != block.c_str() + p;
		}

		void DetectRenderer()
		{
			const bool cs = GetModuleHandleW(L"CommunityShaders.dll") != nullptr;
			const HMODULE d3d = GetModuleHandleW(L"d3d11.dll");
			const bool enb = d3d && GetProcAddress(d3d, "ENBGetSDKVersion") != nullptr;
			// ENB first: Community Shaders stays loaded but switches its features off when it finds ENB.
			g_renderer = enb ? Renderer::kENB : cs ? Renderer::kCommunityShaders : Renderer::kVanilla;
			logger::info("weather fog: renderer {} (Community Shaders {}, ENB {})",
				enb ? "ENB" : cs ? "Community Shaders" : "vanilla", cs ? "loaded" : "not loaded", enb ? "found" : "not found");
		}

		std::string ReadText(const std::filesystem::path& a_path)
		{
			std::ifstream file(a_path, std::ios::binary);
			if (!file) {
				return {};
			}
			std::stringstream ss;
			ss << file.rdbuf();
			return ss.str();
		}

		// A value from CS's settings as CS merges them (State.cpp Load): a section in SettingsUser replaces the whole
		// SettingsDefault section, and a key missing from the section CS uses takes the feature's struct default
		// (a_fallback). Limit: CS's override packs and .user overrides are not read (upgrade path: read CS's
		// Overrides folder in its order).
		double CsValue(const std::string& a_default, const std::string& a_user, std::string_view a_section, std::string_view a_key, double a_fallback)
		{
			std::string block;
			const std::string& source = JsonBlock(a_user, a_section, block) ? a_user : a_default;
			double v = a_fallback;
			return JsonNumberIn(source, a_section, a_key, v) ? v : a_fallback;
		}

		// CS height fog density relative to the ini reference, when CS draws the fog; and whether CS Linear Lighting is on.
		void ReadCommunityShaders()
		{
			float csFactor = 1.0f;
			int   colorSpace = 0;
			if (g_renderer.load() != Renderer::kCommunityShaders) {
				g_csFactor = csFactor;
				g_autoColorSpace = colorSpace;
				return;
			}
			const std::string def = ReadText(CsDefault());
			const std::string user = ReadText(CsSettings());
			if (def.empty() && user.empty()) {
				logger::info("weather fog: CS settings not readable ({}, {}); CS link off", CsDefault().string(), CsSettings().string());
				g_csFactor = csFactor;
				g_autoColorSpace = colorSpace;
				return;
			}
			// Fallbacks are CS's struct defaults (LinearLighting.h, ExponentialHeightFog.h; boot list: not disabled).
			const bool   llDisabledAtBoot = CsValue(def, user, "Disable at Boot", "LinearLighting", 0.0) != 0.0;
			const bool   llOn = !llDisabledAtBoot && CsValue(def, user, "Linear Lighting", "enableLinearLighting", 0.0) != 0.0;
			colorSpace = llOn ? 1 : 0;
			const bool   ehfDisabledAtBoot = CsValue(def, user, "Disable at Boot", "ExponentialHeightFog", 0.0) != 0.0;
			const bool   ehfOn = !ehfDisabledAtBoot && CsValue(def, user, "Exponential Height Fog", "enabled", 0.0) != 0.0;
			const double density = CsValue(def, user, "Exponential Height Fog", "fogDensity", 0.005);
			const double noVanilla = CsValue(def, user, "Exponential Height Fog", "disableVanillaFog", 1.0);
			if (Settings::weatherFollowCs && ehfOn && Settings::weatherCsReference > 0.0f) {
				csFactor = std::clamp(static_cast<float>(density) / Settings::weatherCsReference, 0.0f, 4.0f);
			}
			g_csFactor = csFactor;
			g_autoColorSpace = colorSpace;
			logger::info("weather fog: CS Linear Lighting {}, height fog {} (density {:.4f}, vanilla fog {}), CS factor {:.2f}",
				llOn ? "on" : "off", ehfOn ? "on" : "off", density, noVanilla != 0.0 ? "off" : "on", csFactor);
		}

		WeatherRules::Params MakeParams()
		{
			WeatherRules::Params p;
			int bad = 0;
			p.rules = WeatherRules::ParseRules(Settings::GetString("Weather", "Rules", WeatherRules::kDefaultRules), &bad);
			if (bad > 0) {
				logger::warn("weather fog: {} entries of [Weather] Rules are not 'word:number' and were skipped", bad);
			}
			for (const auto& [key, value] : Settings::Section("WeatherOverrides")) {
				float f = 0.0f;
				if (!WeatherRules::ParseFloat(value, f)) {
					logger::warn("weather fog: [WeatherOverrides] {} = '{}' is not a number; skipped", key, value);
					continue;
				}
				p.overrides.push_back({ WeatherRules::NormalizeOverrideKey(key), f });
			}
			p.flagPleasant = Settings::weatherFlagPleasant;
			p.flagCloudy = Settings::weatherFlagCloudy;
			p.flagRainy = Settings::weatherFlagRainy;
			p.flagSnow = Settings::weatherFlagSnow;
			p.farReference = Settings::weatherFarReference;
			return p;
		}

		// WeatherRules::DayFogShare for the sky now (user 2026-11-14 under Azurite's SkyrimFog at 06:00: "일반 안개 없이
		// 땅안개만 있으니 이상하네"). Both distances carry GameFogScale, so the ratio does not.
		float DayFogShare(const RE::Sky* a_sky)
		{
			if (!a_sky || !a_sky->currentWeather) {
				return 1.0f;
			}
			const float now = a_sky->currentWeather->fogData.dayFar;
			const float last = a_sky->lastWeather ? a_sky->lastWeather->fogData.dayFar : now;
			return WeatherRules::DayFogShare(now, last, a_sky->currentWeatherPct, a_sky->fogFar);
		}

		float FactorOf(const RE::TESWeather* a_w)
		{
			if (!a_w) {
				return 1.0f;
			}
			const auto it = g_factor.find(a_w->GetFormID());
			return it == g_factor.end() ? 1.0f : it->second;
		}
	}

	void Build()
	{
		DetectRenderer();
		auto* data = RE::TESDataHandler::GetSingleton();
		if (!data) {
			return;
		}
		const auto params = MakeParams();
		std::unordered_map<RE::FormID, float>       factors;
		std::unordered_map<RE::FormID, std::string> names;
		int noEditorId = 0;
		for (auto* w : data->GetFormArray<RE::TESWeather>()) {
			if (!w) {
				continue;
			}
			WeatherRules::Weather info;
			info.editorID = EditorIDOf(w);
			if (const auto* file = w->GetFile(0)) {
				info.plugin = std::string(file->GetFilename());
			}
			info.localFormID = w->GetLocalFormID();
			const auto flags = w->data.flags;
			info.pleasant = flags.any(RE::TESWeather::WeatherDataFlag::kPleasant);
			info.cloudy = flags.any(RE::TESWeather::WeatherDataFlag::kCloudy);
			info.rainy = flags.any(RE::TESWeather::WeatherDataFlag::kRainy);
			info.snow = flags.any(RE::TESWeather::WeatherDataFlag::kSnow);
			info.dayFar = w->fogData.dayFar / std::max(GameFog::ScaleFor(w), 0.01f);
			const auto r = WeatherRules::FactorFor(info, params);
			noEditorId += info.editorID.empty() ? 1 : 0;
			factors[w->GetFormID()] = r.factor;
			names[w->GetFormID()] = info.editorID.empty() ? std::format("{}|{:06X}", info.plugin, info.localFormID) : info.editorID;
			logger::info("weather fog: {:<32} {}|{:06X} flags {}{}{}{} dayFar {:>7.0f} -> {:.2f} ({})", names[w->GetFormID()], info.plugin,
				info.localFormID, info.pleasant ? "P" : "-", info.cloudy ? "C" : "-", info.rainy ? "R" : "-", info.snow ? "S" : "-", info.dayFar,
				r.factor, r.how);
		}
		ReadCommunityShaders();
		{
			const std::scoped_lock lock(g_lock);
			g_factor = std::move(factors);
			g_name = std::move(names);
		}
		logger::info("weather fog: {} weathers, {} without an EditorID (they fall back to flags; po3 Tweaks or a similar EditorID mod fixes that)",
			g_factor.size(), noEditorId);
	}

	void RequestBuild()
	{
		if (g_pending.exchange(true)) {
			return;
		}
		if (const auto* tasks = SKSE::GetTaskInterface()) {
			tasks->AddTask([]() {
				g_pending = false;
				Build();
			});
		} else {
			g_pending = false;
		}
	}

	// CS settings changed on disk (CS menu save, or the file created or removed): rebuild. Runs even with the weather
	// link off, because the automatic colour space comes from the same file. Limit: CS's in-memory per-weather
	// values are not followed.
	static void WatchCsSettings()
	{
		const auto t = std::chrono::steady_clock::now();
		if (g_renderer.load() != Renderer::kCommunityShaders || t - g_lastCsCheck <= std::chrono::seconds(5)) {
			return;
		}
		g_lastCsCheck = t;
		std::error_code ec;
		const auto      wt = std::filesystem::last_write_time(CsSettings(), ec);
		const bool      exists = !ec;
		// The first check also rebuilds once: the file may have changed between the startup Build and this check
		// (e.g. a CS menu save indoors, where the fog does not run).
		if (!g_csChecked || exists != g_csExisted || (exists && wt != g_csWriteTime)) {
			RequestBuild();
		}
		g_csChecked = true;
		g_csExisted = exists;
		if (exists) {
			g_csWriteTime = wt;
		}
	}

	float CurrentFactor()
	{
		WatchCsSettings();
		if (!Settings::weatherFog) {
			return 1.0f;
		}
		float target = 1.0f;
		if (auto* sky = RE::Sky::GetSingleton()) {
			const std::scoped_lock lock(g_lock);
			const float now = FactorOf(sky->currentWeather);
			const float last = sky->lastWeather ? FactorOf(sky->lastWeather) : now;
			const float pct = std::clamp(std::isfinite(sky->currentWeatherPct) ? sky->currentWeatherPct : 1.0f, 0.0f, 1.0f);
			target = WeatherRules::ApplyDayShare(last + (now - last) * pct, DayFogShare(sky)) * g_csFactor.load();
		}
		const auto t = std::chrono::steady_clock::now();
		if (g_resetSmooth.exchange(false)) {
			g_smoothed = -1.0f;   // a loaded game starts at its own weather, not a blend from the last one
		}
		if (g_smoothed < 0.0f) {
			g_smoothed = target;
		} else {
			const float dt = std::min(std::chrono::duration<float>(t - g_lastTime).count(), 1.0f);
			g_smoothed += (target - g_smoothed) * (1.0f - std::exp(-dt / kSmoothSeconds));
		}
		g_lastTime = t;
		return g_smoothed;
	}

	float CsFactor()
	{
		return Settings::weatherFog ? g_csFactor.load() : 1.0f;
	}

	std::string Describe()
	{
		auto* sky = RE::Sky::GetSingleton();
		if (!sky) {
			return "no sky";
		}
		const std::scoped_lock lock(g_lock);
		auto name = [&](const RE::TESWeather* a_w) -> std::string {
			if (!a_w) return "none";
			const auto it = g_name.find(a_w->GetFormID());
			return it == g_name.end() ? std::format("{:08X}", a_w->GetFormID()) : it->second;
		};
		return std::format("weather {} ({:.2f}) <- {} ({:.2f}) at {:.0f}%, game fog far {:.0f} (day share {:.2f}), CS x{:.2f}, applied {:.2f}",
			name(sky->currentWeather), FactorOf(sky->currentWeather), name(sky->lastWeather), FactorOf(sky->lastWeather),
			sky->currentWeatherPct * 100.0f, sky->fogFar, DayFogShare(sky), g_csFactor.load(), g_smoothed);
	}

	Renderer DetectedRenderer() { return g_renderer.load(); }
	int      AutoColorSpace() { return g_autoColorSpace.load(); }
	void     OnGameLoad() { g_resetSmooth = true; }
}
