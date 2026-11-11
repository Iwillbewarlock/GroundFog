#pragma once

// Per-worldspace fog rules ([Worldspaces] in the ini), kept free of game types so the offline harness can test them.
//   Name = off          no fog in that worldspace
//   Name = 0.5          the usual rules (weather, time of day), amount x 0.5
//   Name = always 1.0   a fixed amount whatever the weather and the hour (water and hollows still carry more)
//   Name = default      the usual rules; cancels a built-in entry
// Name is the worldspace's EditorID or "plugin|formid" (as [WeatherOverrides]). The nearest worldspace with an entry
// decides: the worldspace itself (ini line, then the built-in table), then its parent the same way (city worldspaces
// under Tamriel), and so on; with none, the usual rules.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "WeatherRules.h"

namespace WorldspaceRules
{
	enum class Kind
	{
		kDefault,   // weather and conditions as everywhere
		kOff,
		kScale,     // usual rules x value
		kAlways     // value replaces the weather factor; time, wind and season do not apply
	};

	struct Rule
	{
		Kind  kind{ Kind::kDefault };
		float value{ 1.0f };

		bool operator==(const Rule&) const = default;
	};

	struct Entry
	{
		std::string key;   // as WeatherRules::NormalizeOverrideKey
		Rule        rule;
	};

	struct Space
	{
		std::string   editorID;
		std::string   plugin;   // file that defines the form
		std::uint32_t localFormID{ 0 };
	};

	inline constexpr float kMaxValue = 4.0f;   // ceiling of one entry (the weather factor has the same one)

	// Places with no sky to burn the fog off, asked for on the mod page (2026-11-11). Amounts not tuned in game.
	inline std::vector<Entry> BuiltIn()
	{
		return {
			{ "blackreach", { Kind::kAlways, 1.0f } },
			{ "dlc01soulcairn", { Kind::kAlways, 1.5f } },
		};
	}

	// The value of one ini line. False for anything else (empty, a word, a negative or non-finite number).
	// Finite numbers above kMaxValue are held to it; one too large for a float (1e39) is not a number here.
	inline bool ParseRule(std::string_view a_text, Rule& a_out)
	{
		const std::string t = WeatherRules::Lower(WeatherRules::Trim(a_text));
		if (t == "off") {
			a_out = { Kind::kOff, 0.0f };
			return true;
		}
		if (t == "default") {
			a_out = { Kind::kDefault, 1.0f };
			return true;
		}
		constexpr std::string_view kAlways = "always";
		const bool always = t.rfind(kAlways, 0) == 0;
		if (always && (t.size() == kAlways.size() || (t[kAlways.size()] != ' ' && t[kAlways.size()] != '\t'))) {
			return false;   // "always" alone, or "always1"
		}
		float v = 0.0f;
		if (!WeatherRules::ParseFloat(always ? std::string_view(t).substr(kAlways.size()) : std::string_view(t), v) || v < 0.0f) {
			return false;
		}
		a_out = { always ? Kind::kAlways : Kind::kScale, v > kMaxValue ? kMaxValue : v };
		return true;
	}

	inline const Entry* Find(const Space& a_space, const std::vector<Entry>& a_entries)
	{
		const std::string id = WeatherRules::Lower(a_space.editorID);
		const std::string form = WeatherRules::FormKey(a_space.plugin, a_space.localFormID);
		for (const auto& e : a_entries) {
			if ((!id.empty() && e.key == id) || e.key == form) {
				return &e;
			}
		}
		return nullptr;
	}

	// a_chain: the worldspace, then its parents outward. The nearest worldspace with an entry decides; on the same
	// worldspace the ini comes before the built-in table, and an ini "default" ends the search there. a_how says where
	// the rule came from, for the log.
	inline Rule Resolve(std::span<const Space> a_chain, const std::vector<Entry>& a_ini, const std::vector<Entry>& a_builtIn,
		std::string* a_how = nullptr)
	{
		for (std::size_t i = 0; i < a_chain.size(); ++i) {
			for (const auto* table : { &a_ini, &a_builtIn }) {
				if (const auto* e = Find(a_chain[i], *table)) {
					if (a_how) {
						const auto& sp = a_chain[i];
						*a_how = std::string(table == &a_ini ? "ini" : "built-in") + (i > 0 ? " (parent " : " (") +
								 (sp.editorID.empty() ? WeatherRules::FormKey(sp.plugin, sp.localFormID) : sp.editorID) + ")";
					}
					return e->rule;
				}
			}
		}
		if (a_how) {
			*a_how = "none";
		}
		return {};
	}

	inline const char* KindName(Kind a_kind)
	{
		switch (a_kind) {
		case Kind::kOff:
			return "off";
		case Kind::kScale:
			return "scale";
		case Kind::kAlways:
			return "always";
		default:
			return "default";
		}
	}
}
