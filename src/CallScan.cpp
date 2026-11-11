#include "CallScan.h"

#include <cstring>

namespace CallScan
{
	namespace
	{
		constexpr std::uint8_t kCallRel32 = 0xE8;
		constexpr std::size_t  kCallSize = 5;
	}

	std::uintptr_t CallTarget(std::span<const std::uint8_t> a_code, std::uintptr_t a_base, std::size_t a_offset)
	{
		if (a_offset + kCallSize > a_code.size() || a_code[a_offset] != kCallRel32) {
			return 0;
		}
		std::int32_t rel{};
		std::memcpy(&rel, a_code.data() + a_offset + 1, sizeof(rel));
		return a_base + a_offset + kCallSize + static_cast<std::intptr_t>(rel);
	}

	std::vector<Hit> FindCallsTo(std::span<const std::uint8_t> a_code, std::uintptr_t a_base, std::uintptr_t a_target)
	{
		std::vector<Hit> hits;
		if (a_code.size() < kCallSize) {
			return hits;
		}
		for (std::size_t i = 0; i + kCallSize <= a_code.size(); ++i) {
			if (CallTarget(a_code, a_base, i) == a_target) {
				hits.push_back({ i, a_target });
			}
		}
		return hits;
	}
}
