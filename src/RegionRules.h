#pragma once

// Geography by the game's weather regions ([Regions] in the ini), kept free of game types so the offline harness can
// test it. A region's value is ADDED to the water-and-terrain multiplier (PlacementRules) of every spot while the
// player is in that region, before the weather and the time of day multiply the result:
//   amount = (water and terrain + region) x weather x time
// The far fog has no water-and-terrain grid, only dry land, so it takes the region's second value (Entry::farAdd).
// Evidence for the built-in table: LORE-FOG.md (the game's own fog-weather chance per region, UESP lore, Met Office /
// NWS / AMS, and Tardif & Rasmussen 2007: fog frequency within one area spans about 0.5 to 2 times its mean).
// Name is the region's EditorID (needs an EditorID mod such as po3 Tweaks) or "plugin|formid".

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "WeatherRules.h"

namespace RegionRules
{
	struct Entry
	{
		std::string key;   // as WeatherRules::NormalizeOverrideKey
		float       add{ 0.0f };   // for the placed field around the player
		float       farAdd{ 0.0f };   // for the far fog, beyond the placement grid ("far" is a windows.h macro)
	};

	struct Region
	{
		std::string   editorID;
		std::string   plugin;   // file that defines the form
		std::uint32_t localFormID{ 0 };
	};

	inline constexpr float kMinAdd = -1.0f;
	inline constexpr float kMaxAdd = 1.0f;

	// By form id, so it works without an EditorID mod. Two values per region:
	//   far  = what the region gains over dry land (LORE-FOG.md). Used as is by the far fog, which has no
	//          water-and-terrain grid out there.
	//   add  = far MINUS what the water-and-terrain rules already give around the player, so the same reason is not
	//          counted twice, never below 0, to the nearest 0.05. "Already give" = the mean placement multiplier of
	//          the grid around the player (shore and valley factors included; the "mean multiplier" of the
	//          diagnostic log) less the 0.7 of dry land: the MEDIAN of three places per region, in game 2026-11-11
	//          (LORE-FOG.md has the table).
	// Limits: three places per region; a spot with no water in a region whose value is 0 gets nothing extra near
	// the player (its far fog still does). The overlap was measured with the default [Placement] multipliers
	// (Land 0.7, MarshLand 2.0, Lake 1.8, Sea 1.2): other multipliers change it and these numbers do not follow.
	inline std::vector<Entry> BuiltIn()
	{
		auto key = [](std::uint32_t a_id) { return WeatherRules::FormKey("Skyrim.esm", a_id); };
		return {
			{ key(0x0C5855), 0.0f, 0.5f },    // WeatherTundraMarsh: fog weather 20%, "the fog that surrounds the marshes"; near: marsh water and land give +1.40
			{ key(0x1046C8), 0.0f, 0.5f },    // WeatherTundraMarshNoPrecip
			{ key(0x0C5856), 0.3f, 0.4f },    // WeatherReach: fog weather 32% (the most), canyons; near: its rivers and valleys give +0.11
			{ key(0x07042F), 0.3f, 0.4f },    // WeatherMarkarth (in the Reach)
			{ key(0x02A72D), 0.25f, 0.3f },   // WeatherPineForest: fog weather 10%, "a fine mist"; near: valleys give +0.03
			{ key(0x10474D), 0.25f, 0.3f },   // WeatherFalkreath
			{ key(0x0C5852), 0.0f, 0.3f },    // WeatherCoast: fog weather 15%, sea fog reaches the coast; near: the sea gives +0.46
			{ key(0x10FE79), 0.0f, 0.3f },    // WeatherCoastFog
			{ key(0x0C5854), 0.0f, 0.2f },    // WeatherVolcanicTundra: fog weather 15%; near: hot-spring cells (lake value) give +0.29
			{ key(0x1046C7), 0.0f, 0.2f },    // WeatherVolcanicTundraNoPrecip
			{ key(0x0C5857), 0.15f, 0.2f },   // WeatherFallForest: fog weather 10%; near: +0.03 (one of three places had a lake: +0.42)
			{ key(0x1046CC), 0.15f, 0.2f },   // WeatherFallForestNoPrecip
			{ key(0x07042E), 0.15f, 0.2f },   // WeatherRiften (in the fall forest)
			{ key(0x10FDC5), 0.15f, 0.2f },   // WeatherFFRiften
		};
	}

	// The value of one ini line: a finite number, with an optional leading '+', held to kMinAdd..kMaxAdd
	// (a_clamped says it was outside). "default" = 0 (cancels a built-in entry). False for anything else.
	inline bool ParseAdd(std::string_view a_text, float& a_out, bool* a_clamped = nullptr)
	{
		std::string t = WeatherRules::Lower(WeatherRules::Trim(a_text));
		if (a_clamped) {
			*a_clamped = false;
		}
		if (t == "default") {
			a_out = 0.0f;
			return true;
		}
		if (!t.empty() && t.front() == '+') {
			t.erase(t.begin());
			if (t.empty() || t.front() == '+' || t.front() == '-') {
				return false;   // "+", "++1", "+-1"
			}
		}
		float v = 0.0f;
		if (!WeatherRules::ParseFloat(t, v)) {
			return false;
		}
		a_out = std::clamp(v, kMinAdd, kMaxAdd) + 0.0f;   // + 0: "-0" is 0
		if (a_clamped) {
			*a_clamped = v < kMinAdd || v > kMaxAdd;
		}
		return true;
	}

	// The entry of one ini line: its number for the field around the player and for the far fog alike.
	inline Entry IniEntry(const std::string& a_key, float a_add)
	{
		return { WeatherRules::NormalizeOverrideKey(a_key), a_add, a_add };
	}

	inline const Entry* Find(const Region& a_region, const std::vector<Entry>& a_entries)
	{
		const std::string id = WeatherRules::Lower(a_region.editorID);
		const std::string form = WeatherRules::FormKey(a_region.plugin, a_region.localFormID);
		for (const auto& e : a_entries) {
			if ((!id.empty() && e.key == id) || e.key == form) {
				return &e;
			}
		}
		return nullptr;
	}

	// The ini before the built-in table; 0 with neither. a_how says where the value came from, for the log.
	// Returns the value for the placed field; a_far gets the one for the far fog (an ini line sets both to its number).
	inline float AddFor(const Region& a_region, const std::vector<Entry>& a_ini, const std::vector<Entry>& a_builtIn, std::string* a_how = nullptr,
		float* a_far = nullptr)
	{
		const Entry* e = Find(a_region, a_ini);
		const char*  how = "ini";
		if (!e) {
			e = Find(a_region, a_builtIn);
			how = e ? "built-in" : "none";
		}
		if (a_how) *a_how = how;
		if (a_far) *a_far = e ? e->farAdd : 0.0f;
		return e ? e->add : 0.0f;
	}

	// The multiplier a spot ends up with: water-and-terrain value plus the region's, never below 0.
	inline float Combined(float a_place, float a_add)
	{
		return std::max(a_place + a_add, 0.0f);
	}
}
