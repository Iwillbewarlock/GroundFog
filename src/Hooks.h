#pragma once

namespace Hooks
{
	struct SiteReport
	{
		const char*    name{ "" };
		bool           found{ false };
		bool           written{ false };  // the trampoline call was written over the original call
		std::uintptr_t hostRva{ 0 };       // function the call was found in
		std::size_t    hits{ 0 };       // matching calls in the scanned function
		std::uintptr_t site{ 0 };       // RVA of the call instruction
		std::uintptr_t expected{ 0 };   // RVA of the function it should call
		std::uintptr_t previous{ 0 };   // RVA of what the call pointed to before we wrote it
	};

	struct OrderReport
	{
		std::uint64_t effectsCalls{ 0 };
		std::uint64_t waterCalls{ 0 };
		std::string   firstEvents;  // first calls in order: E = RenderEffects call, W = RenderWaterEffects call
	};

	// Installs at SKSE load. Each site is found by scanning the decrypted code for a call to the expected function.
	void Install();

	const SiteReport&  Effects();
	const SiteReport&  Water();
	const OrderReport& Order();
}
