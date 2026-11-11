#pragma once

// Values read from Data/SKSE/Plugins/GroundFog.ini at load. Defaults are the PLAN.md section 11
// starting values (game units; 1 unit is about 1.43 cm, not verified).
namespace Settings
{
	enum class DrawStage : int
	{
		kBeforeEffects = 0,     // just before BSShaderAccumulator::RenderEffects is called
		kAfterWaterEffects = 1  // just after Main_RenderWaterEffects returns
	};

	inline bool      enable{ true };
	inline bool      selfTest{ false };
	inline bool      profiler{ false };
	inline DrawStage drawStage{ DrawStage::kBeforeEffects };
	// 0 = normal. 1 = plain magenta wherever the near-field fog is (raymarch composite).
	inline int       debugView{ 0 };
	// Camera matrix: 0 = cameraData.viewProjMat, 1 = viewMat x projMat (DTD's way), 2 = viewProjMatrixUnjittered.
	inline int       matrixMode{ 1 };
	// 1 = draw only when the call carries the main camera (near the player, same projection as the player view).
	inline bool      cameraFilter{ true };
	// 1 = every 2 s read back camera rays, scene heights and fog alpha at 63 screen points into the log ([PROBE]).
	// (2, the pitch sweep, is a development mode the ini reader no longer accepts.)
	inline int       probe{ 0 };

	// Render: wind-driven ground fog drawn by a screen raymarch through the fog columns around the player
	inline float worldSize{ 8192.0f };   // fog field side, units (about 117 m)
	inline float thickness{ 126.0f };    // 1.8 m: the layer's highest mounds stay below eye level; shallow fog (METAR MIFG) is fog under 2 m deep
	inline float opacity{ 0.18f };   // per metre; 1.4 was a white wall at eye height (Blender render 2026-11-07)
	inline float brightness{ 1.0f };
	inline float sunMix{ 1.0f };   // multiplies the direct sun's weight in the fog colour (FogLight.h): 1 = as the light is, 0 = sky colour only
	inline float fogPower{ 1.0f };
	inline float fogMax{ 1.0f };
	// Fog colour space: -1 = auto (linear when Community Shaders Linear Lighting is on), 0 = weather colours as given,
	// 1 = converted like CS Linear Lighting (sRGB -> linear).
	inline int   colorSpace{ -1 };
	inline float raymarchScale{ 0.5f };   // march resolution relative to the screen
	inline int   raymarchSteps{ 24 };
	inline float profileSteepness{ 3.0f }; // fog density falls as exp(-a h/H) up the column (0 = even with height)

	// Height fog: exponential fog from the ground to the horizon, under and beyond the wind field,
	// evaluated in closed form per pixel. Density falls by e every HeightFogFalloff above the base height.
	inline bool  heightFog{ false };   // off: Community Shaders' height fog usually covers this
	inline float heightFogDensity{ 0.002f };        // per metre at the base height
	inline float heightFogFalloff{ 1000.0f };       // units (about 14 m)
	inline float heightFogBaseOffset{ 0.0f };       // units above the (smoothed) ground height under the player
	inline float heightFogMaxDistance{ 150000.0f }; // units; rays to the sky stop here
	// Far tier: the same thin ground layer beyond the wind field, over the whole visible range,
	// laid on the local ground plane estimated from the depth buffer at each pixel (closed form, no simulation).
	inline bool  farFog{ true };
	inline float farFogFlatLength{ 4000.0f };   // units in front of the scene point where its ground plane is trusted
	inline bool  farPools{ true };              // [FarFog] Pools: the coarse grid's pooled fog beyond the field
	inline float farFogNoiseAmount{ 1.0f };     // 0 = even layer, 1 = the same patches as the near field
	// Realistic conditions (FogConditions.h): radiation fog by time of day (scaled by sky clarity), wind and season;
	// steam over water by the air's cold (Survival Mode Improved's ambient temperature when installed).
	inline bool  conditions{ true };
	inline float condNight{ 1.2f };
	inline float condMorningPeak{ 1.6f };
	inline float condAfternoon{ 0.35f };
	inline float condWindLow{ 0.3f };
	inline float condWindHigh{ 0.8f };
	inline float condWindSuppress{ 0.7f };
	inline bool  condUseSurvival{ true };
	// Placement (PlacementRules.h): where fog collects in the near field (water, marsh, shores, valleys, hot springs, sea).
	inline bool  placement{ true };
	inline float placeLake{ 1.8f };
	inline float placeRiver{ 1.4f };
	inline float placeMarshWater{ 2.2f };
	inline float placeMarshLand{ 2.0f };
	inline float placeHotSpring{ 1.5f };
	inline float placeSeaRadiation{ 1.2f };
	inline float placeSeaSteam{ 1.5f };
	inline float placeFreshSteam{ 1.0f };
	inline float placeShoreFactor{ 1.5f };
	inline float placeShoreDistance{ 700.0f };
	inline float placeValleyStrength{ 0.35f };
	inline float placeValleyScale{ 150.0f };
	inline float placeValleyRadius{ 2048.0f };
	inline float placeLand{ 0.7f };
	inline float placeSteamHeight{ 150.0f };   // units of steam column per unit of steam (0 = old: steam only thickens the layer)
	inline float placeSteamDensity{ 2.0f };   // photos: hot-spring plumes turn opaque within 1-3 m
	inline float condEvaporation{ 0.2f };
	inline float placePoolRadius{ 1500.0f };   // units per closing pass (fills hollows up to 2x this wide); 0 = off
	inline float placePoolMaxDepth{ 280.0f };
	inline float placePoolMinVisibility{ 83.0f };   // [Placement] PoolMinVisibility: metres inside a pool at the densest (0 = no floor)
	inline float placePoolDensity{ 0.1f };   // Blender: 0.3 gave about 13 m visibility inside a pool; 0.1 about 40 m (dense radiation fog 50-200 m)
	inline float wispAmount{ 0.5f };          // [Wind] WispAmount: fine structure inside the layer (0 = smooth)
	inline bool  coarsePools{ true };        // [Placement] CoarsePools: pools in valleys wider than the fine grid
	inline float coarsePoolRadius{ 6144.0f }; // [Placement] CoarsePoolRadius: closing window per pass on the coarse grid (units)
	inline float coarseTolerance{ 250.0f };   // [Placement] CoarseTolerance: coarse pools fade where fine and coarse ground differ by this
	inline float glowStrength{ 0.2f };   // [Render] GlowStrength: backlit glow toward the sun (0 = off)
	inline float glowAnisotropy{ 0.6f };  // [Render] GlowAnisotropy: Henyey-Greenstein g (photos 0.65-0.85; lower is softer)
	// Weather link (WeatherFog.cpp): fog amount times a factor per weather (Azurite, NAT, any weather mod), blended
	// through weather transitions, times Community Shaders' height fog density / CsReference when CS draws the fog.
	inline bool  weatherFog{ true };
	inline float weatherFlagPleasant{ 0.5f };    // weathers no rule matched: by classification flag ...
	inline float weatherFlagCloudy{ 0.9f };
	inline float weatherFlagRainy{ 1.2f };
	inline float weatherFlagSnow{ 1.0f };
	inline float weatherFarReference{ 100000.0f };   // ... times sqrt(this / daytime fog distance), 0.5..2
	inline bool  weatherFollowCs{ true };
	inline float weatherCsReference{ 0.015f };   // CS height fog density that counts as 1
	// Weather fog near/far distances (TESWeather FNAM) times this; > 1 starts the game fog farther away. 1 = records.
	inline float gameFogScale{ 1.0f };

	// Wind + fog field (PLAN.md 16.6). Lengths in game units, speeds in units/s (70 units is about 1 m).
	inline bool  enableWind{ true };
	inline bool  enableWalls{ true };
	inline int   simResolution{ 256 };
	inline int   pressureIterations{ 20 };
	inline float windScale{ 105.0f };        // Sky::windSpeed -> units/s. 300 moved the fog at 2-2.6 m/s in game (windSpeed 0.4-0.6, 2026-11-08); real fog drifts about 0.3-1 m/s (video estimates)
	inline float windRelax{ 0.5f };          // 1/s pull toward base wind + gusts
	inline float vorticity{ 2.0f };
	inline float gustAmount{ 0.6f };         // gust strength as a fraction of the base wind
	inline float gustMinimum{ 40.0f };       // gusts even in still weather
	inline float gustWavelength{ 1200.0f };
	inline float fogRefill{ 0.1f };          // 1/s: a hole in the fog fills back in over about 1 / this seconds
	inline float velocityDissipation{ 0.2f };
	inline float wallSlope{ 1.2f };          // ground rise per unit run that counts as a wall
	inline int   maxActors{ 32 };
	inline float actorRadius{ 70.0f };       // 1 m: reach of a body's drag on the air and radius of a runner's push
	inline float actorFollow{ 4.0f };        // 1/s
	inline float actorPush{ 0.35f };         // fog on a runner's path thins to exp(-this): 0.35 -> about 70 % (0 = off)
	inline float actorPushMinSpeed{ 220.0f }; // units/s; no push below (walking is about 80-150), full at 1.5 x
	inline float blastDepth{ 4.0f };         // divergence integrated over an impact's pulse: the fog at the centre thins to about exp(-this)
	inline float blastDuration{ 0.6f };      // s: length of an impact's pulse
	inline float blastRadius{ 105.0f };      // 1.5 m: the hole's radius for a projectile hit and a shout (x 1.5); explosions use their own
	inline float blastPush{ 150.0f };        // units/s the air gains along the hit direction over the pulse

	// Terrain height cache
	inline int   heightResolution{ 129 };   // 64-unit height spacing over WorldSize 8192
	inline float recenterDistance{ 256.0f };
	inline int   heightRetriesPerFrame{ 64 };

	void Load();

	// Thread-safe copies of raw ini text (the ini is re-read on the render thread).
	std::string GetString(const char* a_section, const char* a_key, const char* a_default);
	std::vector<std::pair<std::string, std::string>> Section(const char* a_section);   // keys in lower case
	void LogAll();

	// Re-reads the ini when its write time changed, checked at most every 2 s. Lets a test switch fog on and off
	// (Enable) or change values while the game runs. Returns true when it reloaded.
	bool ReloadIfChanged();
}
