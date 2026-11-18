#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <optional>

// Where BSGraphics::State keeps the dynamic-resolution width/height ratios, found from the bytes around them instead of
// trusting one fixed offset per runtime.
//
// The game stores its TAA jitter table (Halton bases 2 and 3, two rows of 8 floats) right before the two ratios; the
// base-3 row is 1/3, 2/3, 1/9, 4/9, 7/9, 2/9, 5/9, 8/9. A user's log (2026-11-18, runtime not yet known: "viewport
// 1911x1280 on a 3440x1440 target", no upscaler) is exactly 3440 x 5/9 by 1440 x 8/9: CommonLib's offset landed on the
// last two table entries, 8 bytes before the ratios, so the fog covered only the top-left part of the screen. The
// ratios are the two floats right after the whole base-3 row (the full row, not only its last pair, so ratios that
// happen to be 5/9 and 8/9 are not taken for the table; Codex review 2026-11-18).
namespace SceneRatio
{
	inline constexpr float kHalton3[8] = { 1.0f / 3, 2.0f / 3, 1.0f / 9, 4.0f / 9, 7.0f / 9, 2.0f / 9, 5.0f / 9, 8.0f / 9 };
	inline constexpr float kTolerance = 1e-5f;
	inline constexpr std::ptrdiff_t kSearch = 0x10;   // bytes either side of the expected offset (the seen shift is 8)
	inline constexpr std::ptrdiff_t kRowBytes = sizeof(kHalton3);

	// Byte offset from a_base of the width ratio: the float right after the base-3 row, looked for at a_expected,
	// then +-4, +-8 ... up to kSearch (nearest first). a_size bounds the reads. nullopt when the row is not there.
	inline std::optional<std::size_t> FindRatioOffset(const std::byte* a_base, std::size_t a_size, std::size_t a_expected)
	{
		if (!a_base) {
			return std::nullopt;
		}
		const auto expected = static_cast<std::ptrdiff_t>(a_expected);
		const auto size = static_cast<std::ptrdiff_t>(a_size);
		for (std::ptrdiff_t step = 0; step <= kSearch; step += 4) {
			for (const std::ptrdiff_t at : { expected + step, expected - step }) {
				if (at >= kRowBytes && at + 8 <= size) {
					float row[8];
					std::memcpy(row, a_base + at - kRowBytes, sizeof(row));
					bool same = true;
					for (int i = 0; i < 8 && same; ++i) {
						same = std::fabs(row[i] - kHalton3[i]) <= kTolerance;
					}
					if (same) {
						return static_cast<std::size_t>(at);
					}
				}
				if (step == 0) {
					break;   // +0 and -0 are the same place
				}
			}
		}
		return std::nullopt;
	}

	// A usable ratio is in (0.05, 1]; anything else (unset, garbage, NaN) means the whole target.
	inline float Usable(float a_ratio)
	{
		return (a_ratio > 0.05f && a_ratio <= 1.0f) ? a_ratio : 1.0f;
	}

	struct Ratios
	{
		float       width{ 1.0f };
		float       height{ 1.0f };
		std::size_t offset{ 0 };   // where they were read
		bool        found{ false };   // the base-3 row was found in front of them
	};

	// The ratios at a_cached when given (a place found before), else after the base-3 row near a_expected, else at
	// a_expected itself (CommonLib's offset). Reads a_base from a_expected - kSearch - kRowBytes to a_expected + kSearch + 8.
	inline Ratios Resolve(const std::byte* a_base, std::size_t a_expected, std::optional<std::size_t> a_cached)
	{
		Ratios out;
		if (!a_base) {
			return out;
		}
		const auto found = a_cached ? a_cached : FindRatioOffset(a_base, a_expected + kSearch + 8, a_expected);
		out.found = found.has_value();
		out.offset = found.value_or(a_expected);
		float raw[2];
		std::memcpy(raw, a_base + out.offset, sizeof(raw));
		out.width = Usable(raw[0]);
		out.height = Usable(raw[1]);
		return out;
	}
}
