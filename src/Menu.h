#pragma once

// In-game settings page on SKSE Menu Framework (optional: without it GroundFog runs as before and has no page).
namespace Menu
{
	// Registers the page once the framework's DLL is loaded (call at kDataLoaded). a_running: the fog's hooks are
	// installed (false when the game started with Enable = 0: the page then says a restart is needed after enabling).
	void Register(bool a_running);
}
