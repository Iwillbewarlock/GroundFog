#pragma once

namespace FogRenderer
{
	struct Stats
	{
		std::uint64_t calls{ 0 };            // times the draw hook ran
		std::uint64_t draws{ 0 };
		std::uint64_t skipNotMainTarget{ 0 };  // bound target was not the main colour buffer (reflections, cubemaps)
		std::uint64_t skipNoWorld{ 0 };        // no player, interior, no terrain, or the worldspace is off
		std::uint64_t skipWorldspaceOff{ 0 };  // of those: [Worldspaces] off
		std::uint64_t skipNotReady{ 0 };
		std::uint64_t skipNoReadOnlyDepth{ 0 };
		std::uint64_t duplicates{ 0 };
		std::uint64_t skipOtherCamera{ 0 };   // a call that carried a non-main camera (menus, UI 3D)         // second call in the same frame, not drawn
		std::uint64_t probeMismatches{ 0 };    // wider slot probe changed across the pass
		std::uint64_t stateChecks{ 0 };
		std::uint64_t stateMismatches{ 0 };
		int           drawSite{ -2 };
		std::uint32_t rtIndex{ 0xFFFFFFFF };   // render target the game named for slot 0 at the hook
		std::uint32_t dsIndex{ 0xFFFFFFFF };
		bool          shadersCompiled{ false };
		bool          readOnlyDepth{ false };
		bool          depthSrv{ false };
		float         proj34{ 0.0f };
		float         proj44{ 0.0f };
		float         viewportW{ 0.0f };
		float         viewportH{ 0.0f };
		std::uint32_t targetW{ 0 };
		std::uint32_t targetH{ 0 };
		float         minDepth{ 0.0f };
		float         maxDepth{ 0.0f };
		// What the shaders were given at the last draw (read by the grass probe): DepthP.x / .y (proj _33 / _43; the
		// shaders use them only for the sky test) and the z and w columns of the viewProj the scene distance is
		// unprojected with (InvViewProj = its inverse).
		float         depthP33{ 0.0f };
		float         depthP43{ 0.0f };
		float         viewProjZ[4]{};
		float         viewProjW[4]{};
		std::uint64_t probeId{ 0 };   // id of the last [PROBE] readback submitted (0 = none yet)
	};

	// Draws the ground fog (screen raymarch) into the main colour buffer. a_stage names the hook for the log.
	// a_ownTarget: the hook is inside the player-view function, so bind the main target and viewport
	// ourselves instead of requiring them to be bound already.
	// a_site: which hooked call drew (-1 = player-view water hook, 0.. = RenderEffects call sites).
	void Draw(const char* a_stage, bool a_ownTarget, int a_site);

	// Called once per frame from the player-view hook; Draw runs at most once per frame number.
	void NextFrame();

	void Reset();
	// The grass probe ([Debug] GrassProbe) asks for one [PROBE] readback at this frame's draw, also with Probe=0, so the
	// depth the shader reads through its view is logged for the frame the probe copied. Dropped at the frame's end.
	void RequestProbe();
	void PrecompileShaders();   // kDataLoaded: compile every shader on a background thread

	const Stats& GetStats();
	std::string  RegionValue();      // "WeatherReach (skyrim.esm|c5856): +0.30 (far fog +0.40) from built-in (in use +0.30, far fog +0.40)"
	std::string  WorldspaceRule();   // "Tamriel: default from none", for the self test (render-thread value, read after draws)
	double       GpuAverageMs();
	double       GpuMaxMs();
	std::uint32_t GpuSamples();
	double       WindGpuAverageMs();
	double       WindGpuMaxMs();
	std::uint32_t WindGpuMaxSample();   // which sample (1 = first) was the slowest
	std::uint32_t WindGpuSamples();
}
