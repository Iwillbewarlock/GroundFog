#pragma once

// Colour of the fog from the light that reaches it. No game types, so the offline check can test it.
//
// Fog droplets scatter visible light without absorbing it and without favouring a wavelength (single-scattering
// albedo ~ 1, Angstrom exponent ~ 0: Hu & Stamnes 1993, J. Climate 6(4); Slingo 1989, J. Atmos. Sci. 46(10)), so a
// fog layer has no colour of its own: it shows the colour of what lights it. Two things light a ground layer:
// the sky (blue) and, where it reaches the layer, the direct sun (red near the horizon, white when high).
//
// Share of each: the direct beam's part of the irradiance on a horizontal surface under a clear sky, from the
// Bird & Hulstrom clear sky model (SERI/TR-642-761, 1981) run with its baseline atmosphere through pvlib 0.16.1
// (clearsky.bird; ozone 0.3 atm-cm, water 1.5 cm, aerosol optical depth 0.1, albedo 0.2), 2026-11-10:
//     sun elevation (deg)      2      5      10     20     30     45     60
//     direct / global          0.52   0.53   0.64   0.76   0.82   0.86   0.88
// Even a low sun gives about half of the light, where it reaches the ground. What keeps a dawn fog blue is that
// it does not reach it: trees, hills and the far shore hide a sun a few degrees up (photographers' accounts:
// fog in shade reads blue, fog in the first sun reads gold or pink). kSunClearsLow..High is that horizon, an
// assumption, not a measurement: no direct light on the layer below 1 degree, all of it from 8 degrees.
//
// The weights mix the two lights' chromaticities (colour at unit luminance), not their game intensities: the
// game's sunlight colour is far brighter than its ambient colour, so mixing intensities let the sun's hue win at
// any weight (a salmon fog at a clear dawn, in game 2026-11-10). Brightness is kept apart and is the caller's.

#include <algorithm>
#include <array>
#include <cmath>

namespace FogLight
{
	using Rgb = std::array<float, 3>;

	inline constexpr float kSunClearsLow = 1.0f;    // degrees: below this the layer gets no direct sun
	inline constexpr float kSunClearsHigh = 8.0f;   // degrees: from here on it gets all of it

	inline float Luminance(const Rgb& a_c) { return 0.2126f * a_c[0] + 0.7152f * a_c[1] + 0.0722f * a_c[2]; }

	// Direct share of the horizontal irradiance at a sun elevation in degrees (the table above, linear between rows).
	inline float DirectShare(float a_elevationDeg)
	{
		constexpr std::array<float, 8> kElev{ 0.0f, 5.0f, 10.0f, 20.0f, 30.0f, 45.0f, 60.0f, 90.0f };
		constexpr std::array<float, 8> kShare{ 0.52f, 0.53f, 0.64f, 0.76f, 0.82f, 0.86f, 0.88f, 0.89f };
		const float e = std::clamp(a_elevationDeg, 0.0f, 90.0f);
		for (std::size_t i = 1; i < kElev.size(); ++i) {
			if (e <= kElev[i]) {
				const float t = (e - kElev[i - 1]) / (kElev[i] - kElev[i - 1]);
				return kShare[i - 1] + (kShare[i] - kShare[i - 1]) * t;
			}
		}
		return kShare.back();
	}

	// Weight of the direct sun in the fog's colour: its share of the light, times how much of the layer the sun
	// reaches, times a_day (0 at night, when the "sun" light may be the moon), times the user's a_sunMix (1 = as is).
	inline float SunWeight(float a_elevationDeg, float a_day, float a_sunMix)
	{
		const float t = std::clamp((a_elevationDeg - kSunClearsLow) / (kSunClearsHigh - kSunClearsLow), 0.0f, 1.0f);
		const float reached = t * t * (3.0f - 2.0f * t);
		return std::clamp(DirectShare(a_elevationDeg) * reached * std::clamp(a_day, 0.0f, 1.0f) * a_sunMix, 0.0f, 1.0f);
	}

	// The fog's colour: the sky's and the sun's chromaticities mixed by a_sunWeight, at luminance a_luminance.
	// A black input light (luminance ~ 0) has no chromaticity: the other light's is used, or grey when both are black.
	inline Rgb Colour(const Rgb& a_sky, const Rgb& a_sun, float a_sunWeight, float a_luminance)
	{
		constexpr float kBlack = 1e-5f;
		const float ls = Luminance(a_sky);
		const float lu = Luminance(a_sun);
		float       w = std::clamp(a_sunWeight, 0.0f, 1.0f);
		if (!(lu > kBlack)) w = 0.0f;
		if (!(ls > kBlack)) w = lu > kBlack ? 1.0f : 0.0f;
		Rgb out{};
		for (std::size_t c = 0; c < 3; ++c) {
			const float sky = ls > kBlack ? a_sky[c] / ls : 1.0f;
			const float sun = lu > kBlack ? a_sun[c] / lu : 1.0f;
			out[c] = std::max(a_luminance, 0.0f) * ((1.0f - w) * sky + w * sun);
		}
		return out;
	}
}
