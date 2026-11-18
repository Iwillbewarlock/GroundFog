// Offline check of src/SceneRatio.h: finding the dynamic-resolution ratios after the TAA Halton table in a fake
// BSGraphics::State. Covers CommonLib's offset right (1.6.1170), 8 bytes early (the 3440x1440 report of 2026-11-18),
// the search edges both ways, ratios that equal the table's last pair (Codex review 2026-11-18), no table, the
// fallback and cached paths of Resolve, and the ratio clamp. Called from FogShaderCheck's main.

#include "SceneRatio.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
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

	// Halton(2) and Halton(3), indices 1..8, two rows of 8 as the game keeps them.
	constexpr float kTable[16] = { 1.0f / 2, 1.0f / 4, 3.0f / 4, 1.0f / 8, 5.0f / 8, 3.0f / 8, 7.0f / 8, 1.0f / 16,
		1.0f / 3, 2.0f / 3, 1.0f / 9, 4.0f / 9, 7.0f / 9, 2.0f / 9, 5.0f / 9, 8.0f / 9 };

	constexpr std::size_t kExpected = 0x104;   // where CommonLib puts the width ratio on AE (0x60 + 0xA4)
	// Just what Resolve reads past the expected offset, so a read beyond it would leave the buffer.
	constexpr std::size_t kStateSize = kExpected + SceneRatio::kSearch + 8;

	// A fake State with the table ending at a_ratioAt (when a_table) and the two ratios there.
	std::vector<std::byte> MakeState(std::size_t a_ratioAt, float a_w, float a_h, bool a_table = true, std::size_t a_size = kStateSize + 32)
	{
		std::vector<std::byte> s(a_size, std::byte{ 0 });
		if (a_table) {
			std::memcpy(s.data() + a_ratioAt - sizeof(kTable), kTable, sizeof(kTable));
		}
		std::memcpy(s.data() + a_ratioAt, &a_w, 4);
		std::memcpy(s.data() + a_ratioAt + 4, &a_h, 4);
		return s;
	}

	float At(const std::vector<std::byte>& a_s, std::size_t a_off)
	{
		float f;
		std::memcpy(&f, a_s.data() + a_off, 4);
		return f;
	}

	float Area(float a_size, float a_ratio) { return std::floor(a_size * SceneRatio::Usable(a_ratio) + 0.5f); }

	std::string Off(const std::optional<std::size_t>& a_o) { return a_o ? std::to_string(*a_o) : std::string("none"); }

	std::optional<std::size_t> Find(const std::vector<std::byte>& a_s) { return SceneRatio::FindRatioOffset(a_s.data(), a_s.size(), kExpected); }
}

int RunSceneRatioChecks()
{
	constexpr float k59 = 5.0f / 9.0f;
	constexpr float k89 = 8.0f / 9.0f;

	{   // CommonLib right: found where expected, full screen.
		const auto s = MakeState(kExpected, 1.0f, 1.0f);
		const auto o = Find(s);
		Check(o == kExpected && Area(3440, At(s, *o)) == 3440 && Area(1440, At(s, *o + 4)) == 1440, "scene ratio offset right",
			"found at " + Off(o) + ", expected " + std::to_string(kExpected));
	}
	{   // The report: ratios 8 bytes later than CommonLib thinks. The old read gives 1911x1280, the search gives the screen.
		const auto  s = MakeState(kExpected + 8, 1.0f, 1.0f);
		const float oldW = Area(3440, At(s, kExpected));
		const float oldH = Area(1440, At(s, kExpected + 4));
		const auto  o = Find(s);
		Check(oldW == 1911 && oldH == 1280, "scene ratio report reproduced",
			"old read " + std::to_string(static_cast<int>(oldW)) + "x" + std::to_string(static_cast<int>(oldH)) + " (log: 1911x1280)");
		Check(o == kExpected + 8 && Area(3440, At(s, *o)) == 3440 && Area(1440, At(s, *o + 4)) == 1440, "scene ratio offset 8 late",
			"found at " + Off(o) + ", ratios there give the whole 3440x1440");
	}
	{   // Upscaler ratios after a shifted table are still read.
		const auto s = MakeState(kExpected + 8, 0.5f, 0.5f);
		const auto o = Find(s);
		Check(o && At(s, *o) == 0.5f && At(s, *o + 4) == 0.5f, "scene ratio upscaler", "found at " + Off(o) + ", 0.5 x 0.5");
	}
	{   // Edges: 16 bytes away either way is searched, 20 either way is not; 8 bytes early also works.
		const auto p16 = Find(MakeState(kExpected + 16, 1.0f, 1.0f));
		const auto m16 = Find(MakeState(kExpected - 16, 1.0f, 1.0f));
		const auto p20 = Find(MakeState(kExpected + 20, 1.0f, 1.0f, true, kStateSize + 64));
		const auto m20 = Find(MakeState(kExpected - 20, 1.0f, 1.0f));
		const auto m8 = Find(MakeState(kExpected - 8, 1.0f, 1.0f));
		Check(p16 == kExpected + 16 && m16 == kExpected - 16 && !p20 && !m20 && m8 == kExpected - 8, "scene ratio search edges",
			"+16 " + Off(p16) + ", -16 " + Off(m16) + ", +20 " + Off(p20) + ", -20 " + Off(m20) + ", -8 " + Off(m8));
	}
	{   // Ratios equal to the table's last pair (5/9, 8/9) are not taken for the table (Codex review): with the table in
		// place the nearest place wins; with the table not filled yet nothing is found; with the table 8 bytes early the
		// row in front of the real ratios is found.
		const auto right = Find(MakeState(kExpected, k59, k89));
		const auto empty = Find(MakeState(kExpected, k59, k89, false));
		const auto early = Find(MakeState(kExpected - 8, k59, k89));
		Check(right == kExpected && !empty && early == kExpected - 8, "scene ratio ratios equal the table tail",
			"table right " + Off(right) + ", table empty " + Off(empty) + ", table 8 early " + Off(early));
	}
	{   // No table, or a size too small to hold the read: nothing found.
		const std::vector<std::byte> zeros(kStateSize, std::byte{ 0 });
		const auto s = MakeState(kExpected + 8, 1.0f, 1.0f);
		const auto none = SceneRatio::FindRatioOffset(zeros.data(), zeros.size(), kExpected);
		const auto cut = SceneRatio::FindRatioOffset(s.data(), kExpected + 8, kExpected);   // ratios would end past a_size
		const auto null = SceneRatio::FindRatioOffset(nullptr, kStateSize, kExpected);
		Check(!none && !cut && !null, "scene ratio not found", "zeros " + Off(none) + ", cut short " + Off(cut) + ", null " + Off(null));
	}
	{   // Resolve: found (in a buffer exactly as long as it may read), fallback to the expected offset, a cached place.
		const auto found = MakeState(kExpected + 8, 0.5f, 0.75f, true, kStateSize);
		const auto rf = SceneRatio::Resolve(found.data(), kExpected, std::nullopt);
		const auto nothing = MakeState(kExpected, 0.667f, 0.667f, false, kStateSize);
		const auto rn = SceneRatio::Resolve(nothing.data(), kExpected, std::nullopt);
		const auto rc = SceneRatio::Resolve(found.data(), kExpected, kExpected + 8);
		const auto r0 = SceneRatio::Resolve(nullptr, kExpected, std::nullopt);
		const bool ok = rf.found && rf.offset == kExpected + 8 && rf.width == 0.5f && rf.height == 0.75f &&
		                !rn.found && rn.offset == kExpected && rn.width == 0.667f && rn.height == 0.667f &&
		                rc.found && rc.offset == kExpected + 8 && rc.width == 0.5f && !r0.found && r0.width == 1.0f;
		Check(ok, "scene ratio resolve",
			"found " + std::to_string(rf.offset) + " " + std::to_string(rf.width) + "x" + std::to_string(rf.height) + ", fallback " +
				std::to_string(rn.offset) + " " + std::to_string(rn.width) + ", cached " + std::to_string(rc.offset) + ", null " +
				std::to_string(r0.width));
	}
	{   // The clamp keeps (0.05, 1] and turns the rest into the whole target.
		const float nan = std::numeric_limits<float>::quiet_NaN();
		const bool  ok = SceneRatio::Usable(nan) == 1.0f && SceneRatio::Usable(0.05f) == 1.0f && SceneRatio::Usable(0.0f) == 1.0f &&
		                SceneRatio::Usable(1.5f) == 1.0f && SceneRatio::Usable(1.0f) == 1.0f && SceneRatio::Usable(0.667f) == 0.667f;
		Check(ok, "scene ratio clamp", "NaN, 0, 0.05, 1.5 -> 1; 1 and 0.667 kept");
	}
	return g_fail;
}
