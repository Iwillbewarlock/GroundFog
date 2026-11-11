#pragma once

// Weather -> fog factor rules, kept free of game types so the offline harness can test them.
// Order: an ini override (EditorID or "plugin|formid"), then the first keyword found in the EditorID, then the
// weather's classification flags times a term from its daytime fog distance.

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace WeatherRules
{
	// First match wins, so longer and stronger words come first ("FogRain" is fog, "SnowClear" is snow).
	inline constexpr const char* kDefaultRules =
		"fog:2.5, mist:2.2, haze:1.6, volcanicash:1.5, blizzard:1.4, snowstorm:1.4, storm:1.3, rain:1.2, drizzle:1.1, snow:0.9, overcast:0.9, "
		"cloud:0.8, clear:0.45, sunny:0.45";

	struct Rule
	{
		std::string word;   // lower case
		float       factor{ 1.0f };
	};

	struct Override
	{
		std::string key;   // lower case EditorID, or "plugin.esp|00abcd" (local form id, hex, lower case)
		float       factor{ 1.0f };
	};

	struct Weather
	{
		std::string   editorID;
		std::string   plugin;   // file that defines the form
		std::uint32_t localFormID{ 0 };
		bool          pleasant{ false };
		bool          cloudy{ false };
		bool          rainy{ false };
		bool          snow{ false };
		float         dayFar{ 0.0f };   // record value before any GameFogScale
	};

	struct Params
	{
		std::vector<Rule>     rules;
		std::vector<Override> overrides;
		float                 flagPleasant{ 0.5f };
		float                 flagCloudy{ 0.9f };
		float                 flagRainy{ 1.2f };
		float                 flagSnow{ 1.0f };
		float                 farReference{ 100000.0f };   // units; fog ending here counts as 1
		float                 minFactor{ 0.0f };
		float                 maxFactor{ 4.0f };
	};

	struct Result
	{
		float       factor{ 1.0f };
		std::string how;   // "override", "word fog", "flags", ...
	};

	inline std::string Lower(std::string_view a_s)
	{
		std::string out(a_s);
		std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return out;
	}

	inline std::string Trim(std::string_view a_s)
	{
		const auto b = a_s.find_first_not_of(" \t\r\n");
		if (b == std::string_view::npos) {
			return {};
		}
		const auto e = a_s.find_last_not_of(" \t\r\n");
		return std::string(a_s.substr(b, e - b + 1));
	}

	inline bool ParseFloat(std::string_view a_s, float& a_out)
	{
		const auto t = Trim(a_s);
		if (t.empty()) {
			return false;
		}
		const auto r = std::from_chars(t.data(), t.data() + t.size(), a_out);
		return r.ec == std::errc{} && r.ptr == t.data() + t.size() && std::isfinite(a_out);
	}

	// "fog:2.5, mist:2.2" -> rules in that order. Entries without a word or a valid number are skipped and counted.
	inline std::vector<Rule> ParseRules(std::string_view a_text, int* a_bad = nullptr)
	{
		std::vector<Rule> out;
		int               bad = 0;
		std::size_t       pos = 0;
		while (pos <= a_text.size()) {
			const auto        comma = a_text.find(',', pos);
			const std::string item = Trim(a_text.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos));
			pos = comma == std::string_view::npos ? a_text.size() + 1 : comma + 1;
			if (item.empty()) {
				continue;
			}
			const auto colon = item.find(':');
			Rule       r;
			if (colon == std::string::npos || (r.word = Lower(Trim(item.substr(0, colon)))).empty() || !ParseFloat(item.substr(colon + 1), r.factor)) {
				++bad;
				continue;
			}
			out.push_back(std::move(r));
		}
		if (a_bad) {
			*a_bad = bad;
		}
		return out;
	}

	inline std::string FormKey(std::string_view a_plugin, std::uint32_t a_localFormID)
	{
		char buf[16]{};
		const auto r = std::to_chars(buf, buf + sizeof(buf), a_localFormID & 0x00FFFFFFu, 16);
		return Lower(a_plugin) + "|" + std::string(buf, r.ptr);
	}

	// An [WeatherOverrides] key as FactorFor compares it: "Plugin.esp|0053B6", "plugin.esp|0x53b6" -> FormKey form;
	// anything without '|' -> lower-case EditorID. A form id that is not hex is kept as text and matches nothing.
	inline std::string NormalizeOverrideKey(const std::string& a_key)
	{
		const auto bar = a_key.find('|');
		if (bar == std::string::npos) {
			return Lower(Trim(a_key));
		}
		std::string hex = Trim(a_key.substr(bar + 1));
		if (hex.rfind("0x", 0) == 0 || hex.rfind("0X", 0) == 0) {
			hex = hex.substr(2);
		}
		std::uint32_t id = 0;
		const auto    r = std::from_chars(hex.data(), hex.data() + hex.size(), id, 16);
		if (hex.empty() || r.ec != std::errc{} || r.ptr != hex.data() + hex.size()) {
			return Lower(a_key);
		}
		return FormKey(Trim(a_key.substr(0, bar)), id);
	}

	inline Result FactorFor(const Weather& a_w, const Params& a_p)
	{
		auto clampF = [&](float a_f) { return std::clamp(a_f, a_p.minFactor, a_p.maxFactor); };
		const std::string id = Lower(a_w.editorID);
		const std::string form = FormKey(a_w.plugin, a_w.localFormID);
		for (const auto& o : a_p.overrides) {
			if ((!id.empty() && o.key == id) || o.key == form) {
				return { clampF(o.factor), "override" };
			}
		}
		if (!id.empty()) {
			for (const auto& r : a_p.rules) {
				if (id.find(r.word) != std::string::npos) {
					return { clampF(r.factor), "word " + r.word };
				}
			}
		}
		float flag = 1.0f;
		if (a_w.snow) {
			flag = a_p.flagSnow;
		} else if (a_w.rainy) {
			flag = a_p.flagRainy;
		} else if (a_w.cloudy) {
			flag = a_p.flagCloudy;
		} else if (a_w.pleasant) {
			flag = a_p.flagPleasant;
		}
		// Shorter fog distance = foggier; square root and a 0.5..2 clamp keep odd records from dominating.
		const float dist = a_w.dayFar > 1.0f ? std::clamp(std::sqrt(a_p.farReference / a_w.dayFar), 0.5f, 2.0f) : 1.0f;
		return { clampF(flag * dist), a_w.editorID.empty() ? "flags (no EditorID)" : "flags" };
	}
}
