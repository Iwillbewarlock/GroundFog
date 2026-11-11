#pragma once

// Realistic fog conditions from what the game knows (no temperature or humidity in Skyrim): time of day, sky
// clarity, wind and season, after the meteorology of radiation and steam fog. Pure functions, tested offline.
//   Radiation fog: forms on clear, calm nights as the ground cools; thickest shortly after sunrise; gone 2-4 h after
//   sunrise; winds above about 2.5 m/s (5 kn) prevent or clear it; clouds keep the heat in; most common in autumn
//   and winter (long nights). UK Met Office "What is fog?", US NWS fog guide, NWS Binghamton valley fog.
//   Steam fog (sea smoke, hot springs): cold air over much warmer water, 5-40 C colder (Saunders 1964, QJRMS);
//   light winds favour it. Water warms and cools later than the air (large heat capacity), so open water steams in
//   late autumn and early winter and hardly in spring, when the air is already warmer than the water.

#include <algorithm>
#include <array>
#include <cmath>

namespace FogConditions
{
	struct Params
	{
		float nightLevel{ 1.2f };       // radiation fog multiplier through the night
		float morningPeak{ 1.6f };      // just after sunrise (about 06:30)
		float afternoonLevel{ 0.35f };  // burnt off (11:00 - 16:00)
		float clarityPleasant{ 1.0f };  // how much the time curve applies under each sky
		float clarityCloudy{ 0.5f };
		float clarityRainy{ 0.25f };
		float claritySnow{ 0.4f };
		float windLow{ 0.3f };          // Sky::windSpeed where wind starts to clear the fog ...
		float windHigh{ 0.8f };         // ... and where it clears the most
		float windSuppress{ 0.7f };     // share cleared at windHigh
		std::array<float, 12> season{ 1.25f, 1.2f, 1.05f, 1.0f, 0.9f, 0.8f, 0.75f, 0.8f, 1.0f, 1.15f, 1.25f, 1.3f };   // Morning Star .. Evening Star
		// Air cold proxy (no survival mod): by month, and at night (fades in 19-22 h, out 05-08 h).
		std::array<float, 12> airCold{ 0.6f, 0.6f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.6f, 0.6f };
		float nightCold{ 0.6f };
		// How cold open water is by month, on the air scale; it lags the air by one to two months.
		std::array<float, 12> waterCold{ 0.35f, 0.45f, 0.5f, 0.45f, 0.3f, 0.15f, 0.05f, 0.0f, 0.05f, 0.15f, 0.25f, 0.3f };
		float steamSpan{ 0.4f };   // air colder than the water by this much = full steam
		// By day the weakest radiation fog evaporates first: this much (times 1 - global) is subtracted from each
		// spot's multiplier, so fog stays over water and marsh at midday and is gone from dry ground.
		float evaporation{ 0.2f };
	};

	struct Sky
	{
		float hour{ 12.0f };       // 0..24
		int   month{ 0 };          // 0 = Morning Star (January) .. 11 = Evening Star
		float windSpeed{ 0.0f };   // Sky::windSpeed
		bool  pleasant{ false };
		bool  cloudy{ false };
		bool  rainy{ false };
		bool  snow{ false };
		// Ambient cold from a survival mod, 0 (warm) .. 1 (blizzard-cold); below 0 = none, use the proxy below.
		// Survival Mode Improved: SMI_CurrentAmbientTemp / 450 (its blizzard level, NeedCold.h WEATHER_TEMPS).
		float survivalCold{ -1.0f };
	};

	struct Result
	{
		float global{ 1.0f };   // radiation-fog multiplier: time (by clarity) x wind x season
		float wind{ 1.0f };     // the wind part alone (also thins steam fog)
		float cold{ 0.0f };     // 0..1, steam over open water: how much colder the air is than the water
		float air{ 0.0f };      // 0..1, the air's cold (survival mod or proxy)
		float evaporation{ 0.0f };   // subtracted from the radiation multiplier (0 when global >= 1)
		float time{ 1.0f };     // the time part after clarity
	};

	inline float Smoothstep(float a_e0, float a_e1, float a_x)
	{
		if (!(a_e1 > a_e0)) {
			return a_x >= a_e1 ? 1.0f : 0.0f;
		}
		const float t = std::clamp((a_x - a_e0) / (a_e1 - a_e0), 0.0f, 1.0f);
		return t * t * (3.0f - 2.0f * t);
	}

	// Piecewise linear over the day: night -> peak after sunrise -> burnt off by late morning -> rising at dusk.
	inline float TimeCurve(float a_hour, const Params& a_p)
	{
		float h = std::fmod(a_hour, 24.0f);
		if (h < 0.0f) h += 24.0f;
		const float mid = 0.5f * (a_p.afternoonLevel + a_p.nightLevel);
		const std::array<float, 9> hours{ 0.0f, 4.5f, 6.5f, 9.0f, 11.0f, 16.0f, 19.0f, 21.5f, 24.0f };
		const std::array<float, 9> level{ a_p.nightLevel, a_p.nightLevel * 1.1f, a_p.morningPeak, 0.5f * (a_p.morningPeak + a_p.afternoonLevel),
			a_p.afternoonLevel, a_p.afternoonLevel, mid, a_p.nightLevel, a_p.nightLevel };
		for (std::size_t i = 1; i < hours.size(); ++i) {
			if (h <= hours[i]) {
				const float t = (h - hours[i - 1]) / (hours[i] - hours[i - 1]);
				return level[i - 1] + (level[i] - level[i - 1]) * t;
			}
		}
		return a_p.nightLevel;
	}

	inline float Clarity(const Sky& a_s, const Params& a_p)
	{
		if (a_s.snow) return a_p.claritySnow;
		if (a_s.rainy) return a_p.clarityRainy;
		if (a_s.cloudy) return a_p.clarityCloudy;
		if (a_s.pleasant) return a_p.clarityPleasant;
		return a_p.clarityCloudy;
	}

	inline Result Compute(const Sky& a_s, const Params& a_p)
	{
		Result r;
		const float clarity = std::clamp(Clarity(a_s, a_p), 0.0f, 1.0f);
		r.time = 1.0f + (TimeCurve(a_s.hour, a_p) - 1.0f) * clarity;
		r.wind = 1.0f - std::clamp(a_p.windSuppress, 0.0f, 1.0f) * Smoothstep(a_p.windLow, a_p.windHigh, std::isfinite(a_s.windSpeed) ? a_s.windSpeed : 0.0f);
		const int   m = ((a_s.month % 12) + 12) % 12;
		const float season = a_p.season[static_cast<std::size_t>(m)];
		r.global = std::clamp(r.time * r.wind * season, 0.0f, 4.0f);
		// Air cold: the survival mod's ambient cold when there is one, else snow weather, night and the month.
		// Night fades in and out over three hours each side, so the steam does not jump at a fixed hour.
		const float h = std::fmod(std::fmod(a_s.hour, 24.0f) + 24.0f, 24.0f);
		const float night = a_p.nightCold * std::max(1.0f - Smoothstep(5.0f, 8.0f, h), Smoothstep(19.0f, 22.0f, h));
		const std::size_t mi = static_cast<std::size_t>(m);
		r.air = a_s.survivalCold >= 0.0f ? std::clamp(a_s.survivalCold, 0.0f, 1.0f)
		                                  : std::clamp(std::max({ a_s.snow ? 1.0f : 0.0f, night, a_p.airCold[mi] }), 0.0f, 1.0f);
		// Steam needs air colder than the water, not cold air alone.
		r.cold = std::clamp((r.air - a_p.waterCold[mi]) / std::max(a_p.steamSpan, 0.01f), 0.0f, 1.0f);
		r.evaporation = std::max(a_p.evaporation, 0.0f) * std::max(1.0f - r.global, 0.0f);
		return r;
	}
}
