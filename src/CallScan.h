#pragma once

#include <cstdint>
#include <span>
#include <vector>

// Finds direct `call rel32` (E8) instructions inside a block of code bytes.
// The game exe is DRM-encrypted on disk, so this runs on the decrypted code in memory.
// A byte scan can hit an E8 that is really part of another instruction; requiring the call to
// land exactly on a known function start makes such false hits very unlikely (not impossible).
namespace CallScan
{
	struct Hit
	{
		std::size_t   offset;  // from the start of the scanned block
		std::uintptr_t target;
	};

	// Target of a 5-byte E8 call at a_site, or 0 if a_code[a_offset] is not E8.
	std::uintptr_t CallTarget(std::span<const std::uint8_t> a_code, std::uintptr_t a_base, std::size_t a_offset);

	// Every E8 call in a_code (loaded at a_base) whose target equals a_target.
	std::vector<Hit> FindCallsTo(std::span<const std::uint8_t> a_code, std::uintptr_t a_base, std::uintptr_t a_target);

	// Every E9 `jmp rel32` (a tail call) in a_code whose target equals a_target. Same false-hit caveat as above.
	std::vector<Hit> FindJumpsTo(std::span<const std::uint8_t> a_code, std::uintptr_t a_base, std::uintptr_t a_target);
}
