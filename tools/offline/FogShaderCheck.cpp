// Offline check for GroundFog, no game needed. Prints PASS/FAIL lines; exit code 1 on any FAIL.
//   1. CallScan finds E8 calls (normal, last-byte boundary, no hit)
//   2. GridMath::Shift keeps world positions (normal, shift larger than grid, bad size)
//   3. RunWindSimChecks (WindSimCheck.cpp) and RunRaymarchChecks (RaymarchCheck.cpp): the wind and fog field,
//      the fog columns and the screen raymarch on WARP, and the CPU rules they use.

#include "CallScan.h"
#include "FogLight.h"
#include "GridMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{
	int g_fail = 0;

	void Check(bool a_ok, const char* a_name, const std::string& a_detail)
	{
		std::printf("%s %s : %s\n", a_ok ? "PASS" : "FAIL", a_name, a_detail.c_str());
		if (!a_ok) {
			++g_fail;
		}
	}

	void PutCall(std::vector<std::uint8_t>& a_code, std::size_t a_at, std::uintptr_t a_base, std::uintptr_t a_target)
	{
		const auto rel = static_cast<std::int32_t>(static_cast<std::intptr_t>(a_target) -
		                                           static_cast<std::intptr_t>(a_base + a_at + 5));
		a_code[a_at] = 0xE8;
		std::memcpy(&a_code[a_at + 1], &rel, 4);
	}

	void TestCallScan()
	{
		constexpr std::uintptr_t base = 0x140001000;
		constexpr std::uintptr_t target = 0x140123450;
		std::vector<std::uint8_t> code(64, 0x90);
		PutCall(code, 7, base, target);
		PutCall(code, 20, base, target + 16);  // a call elsewhere must not match
		auto hits = CallScan::FindCallsTo(code, base, target);
		Check(hits.size() == 1 && hits[0].offset == 7, "callscan normal", "one call at offset 7, " + std::to_string(hits.size()) + " found");

		PutCall(code, code.size() - 5, base, target);  // last full instruction
		code.push_back(0xE8);                           // a lone E8 in the last byte must be ignored
		hits = CallScan::FindCallsTo(code, base, target);
		Check(hits.size() == 2 && hits[1].offset == code.size() - 6, "callscan boundary", std::to_string(hits.size()) + " hits");

		std::vector<std::uint8_t> none(32, 0x90);
		Check(CallScan::FindCallsTo(none, base, target).empty() && CallScan::FindCallsTo({}, base, target).empty(),
			"callscan no hit", "empty for code without the call");
	}

	void TestShift()
	{
		constexpr int     n = 4;
		std::vector<float> g(n * n);
		for (int i = 0; i < n * n; ++i) {
			g[i] = static_cast<float>(i);
		}
		GridMath::Shift(g, n, 1, 2, -1.0f);  // new(x,y) = old(x+1, y+2)
		const bool okMoved = g[0] == 9.0f && g[1 * n + 2] == 15.0f && g[0 * n + 3] == -1.0f && g[2 * n + 0] == -1.0f;
		Check(okMoved, "shift normal", "new(0,0)=old(1,2)=9, exposed cells filled");

		GridMath::Shift(g, n, -10, 0, -1.0f);
		bool allFilled = true;
		for (float v : g) {
			allFilled = allFilled && v == -1.0f;
		}
		Check(allFilled, "shift beyond grid", "every cell refilled");

		std::vector<float> bad(5, 3.0f);
		GridMath::Shift(bad, n, 1, 0, -1.0f);
		Check(bad[0] == 3.0f, "shift bad size", "left untouched");
	}

	// FogLight: the fog takes the colour of what lights it. A blue sky and an orange low sun, as the game gives them
	// at a clear dawn (the sun's colour several times brighter than the ambient one).
	void TestFogLight()
	{
		const FogLight::Rgb sky{ 0.20f, 0.26f, 0.40f };
		const FogLight::Rgb sun{ 1.60f, 0.80f, 0.30f };
		auto sat = [](const FogLight::Rgb& a_c) {
			const float hi = std::max({ a_c[0], a_c[1], a_c[2] });
			const float lo = std::min({ a_c[0], a_c[1], a_c[2] });
			return hi > 0.0f ? (hi - lo) / hi : 0.0f;
		};
		// Normal: a sun 30 degrees up by day lights the layer with its share of the light (0.82).
		const float wDay = FogLight::SunWeight(30.0f, 1.0f, 1.0f);
		const auto  day = FogLight::Colour(sky, sun, wDay, 0.5f);
		Check(std::fabs(wDay - 0.82f) < 1e-3f && std::fabs(FogLight::Luminance(day) - 0.5f) < 1e-4f && day[0] > day[2],
			"fog light by day", "sun weight " + std::to_string(wDay) + ", luminance " + std::to_string(FogLight::Luminance(day)) +
									", red/blue " + std::to_string(day[0] / day[2]));
		// Boundary: a sun on the horizon does not reach the layer: the fog is the sky's colour, at the asked luminance;
		// at 4.5 degrees (half way up the horizon band) it reaches half of it.
		const float wDawn = FogLight::SunWeight(0.5f, 1.0f, 1.0f);
		const auto  dawn = FogLight::Colour(sky, sun, wDawn, 0.5f);
		const float wLow = FogLight::SunWeight(4.5f, 1.0f, 1.0f);
		const auto  low = FogLight::Colour(sky, sun, wLow, 0.5f);
		Check(wDawn == 0.0f && dawn[2] > dawn[0] && std::fabs(FogLight::Luminance(dawn) - 0.5f) < 1e-4f && std::fabs(wLow - 0.5f * 0.529f) < 5e-3f &&
				  sat(low) < sat(sun) * 0.6f,
			"fog light at dawn", "weight at 0.5 deg " + std::to_string(wDawn) + ", at 4.5 deg " + std::to_string(wLow) + ", saturation there " +
									 std::to_string(sat(low)) + " (sun " + std::to_string(sat(sun)) + ", sky " + std::to_string(sat(sky)) + ")");
		// Failure: night (the light may be the moon) and black inputs give no weight to the sun and no NaN.
		const float wNight = FogLight::SunWeight(40.0f, 0.0f, 1.0f);
		const auto  black = FogLight::Colour({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 0.7f, 0.3f);
		const auto  noSky = FogLight::Colour({ 0.0f, 0.0f, 0.0f }, sun, 0.0f, 0.3f);
		Check(wNight == 0.0f && std::isfinite(black[0]) && std::fabs(black[0] - 0.3f) < 1e-5f && std::isfinite(noSky[0]) &&
				  std::fabs(FogLight::Luminance(noSky) - 0.3f) < 1e-4f,
			"fog light at night and with black inputs", "night weight " + std::to_string(wNight) + ", black -> grey " + std::to_string(black[0]) +
															", no sky -> luminance " + std::to_string(FogLight::Luminance(noSky)));
	}
}

int RunWindSimChecks();   // WindSimCheck.cpp, returns failures
int RunRaymarchChecks();  // RaymarchCheck.cpp

int main()
{
	TestCallScan();
	TestShift();
	TestFogLight();
	g_fail += RunWindSimChecks();
	g_fail += RunRaymarchChecks();
	std::printf("%s (%d failures)\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
	return g_fail == 0 ? 0 : 1;
}
