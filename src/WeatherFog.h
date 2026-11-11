#pragma once

// Follows the weather (Azurite, NAT or any other weather mod) and the renderer in use (Community Shaders or ENB):
// a fog factor per weather from WeatherRules, blended through weather transitions, times a factor from Community
// Shaders' height fog density when CS draws the game's fog.
namespace WeatherFog
{
	enum class Renderer
	{
		kVanilla,
		kCommunityShaders,
		kENB
	};

	// Main thread (kDataLoaded, ini reload): detect the renderer, read CS settings, build the weather table and log it.
	void Build();

	// From any thread: queues Build on the main thread (after an ini change).
	void RequestBuild();

	// Render thread, once per frame: the factor for the current sky (transition blend, smoothed over a few seconds).
	float CurrentFactor();

	// The Community Shaders part of CurrentFactor alone (1 with [Weather] Enable=0 or FollowCommunityShaders=0).
	float CsFactor();

	// For logs: "SkyrimFog -> 2.50 (word fog), 100%".
	std::string Describe();

	Renderer DetectedRenderer();

	// 1 when fog colours should be converted to linear (CS Linear Lighting on), 0 otherwise.
	int AutoColorSpace();

	// Any thread: a game was loaded; the next factor starts at the new weather instead of blending from the old.
	void OnGameLoad();
}
