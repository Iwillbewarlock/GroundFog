#pragma once

// Thins the game's own weather fog by scaling every weather record's fog near/far distances (FNAM), keeping the
// records' original values so any scale, including 1, can be applied again. No ESP, no shader patch.
namespace GameFog
{
	// Main thread only (forms are edited). a_scale > 1 pushes the fog farther away; 1 restores the records.
	void Apply(float a_scale);

	// From any thread: queues Apply on the main thread when a_scale differs from the scale last applied.
	void RequestApply(float a_scale);

	// Any thread: the scale our last Apply put on this weather (1 when that Apply skipped it or never ran). A change
	// another mod makes after that Apply is not seen here until the next Apply.
	float ScaleFor(const RE::TESWeather* a_weather);
}
