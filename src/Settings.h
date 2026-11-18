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
	// 0 = normal. 1 = plain magenta wherever the near-field fog is (raymarch composite). 2 = every pixel marches its
	// own ray at full resolution, 64 steps (CompositeM; diagnosis only, heavy). 3 = the same at RaymarchSteps: 0 against
	// 3 shows what the half-resolution upsampling changes, 3 against 2 what the step count changes.
	inline int       debugView{ 0 };
	inline int       menuLanguage{ 0 };   // [General] MenuLanguage: the settings page's language, 0 English, 1 Korean
	inline int       shadowMatrixMode{ 0 };   // [General] ShadowMatrixMode (diagnosis of DebugView 4, 0 - 3; 0 = Community Shaders' convention)
	// Camera matrix: 0 = cameraData.viewProjMat, 1 = viewMat x projMat (DTD's way), 2 = viewProjMatrixUnjittered.
	inline int       matrixMode{ 1 };
	// 1 = draw only when the call carries the main camera (near the player, same projection as the player view).
	inline bool      cameraFilter{ true };
	// 1 = every 2 s read back camera rays, scene heights and fog alpha at 63 screen points into the log ([PROBE]).
	// (2, the pitch sweep, is a development mode the ini reader no longer accepts.)
	inline int       probe{ 0 };
	// [Debug] GrassProbe: 1 = log-only grass depth probe (GrassProbe.cpp): at load hooks the calls to
	// BSShaderAccumulator::RenderBatches and Main_RenderDepth; every 5 s copies the main depth right before and after
	// the depth prepass's grass calls and at the fog's draw point and compares them ("grassprobe:" lines). 2 = the same,
	// and every second comparison makes the prepass grass call after its round's group-9 call instead (the depth
	// without grass). The hooks follow the value at load (0 = nothing installed); set to 0 later, they only pass calls on.
	inline int       grassProbe{ 0 };
	// [Debug] GrassProbeDump: 1 = with the probe running, write a numbered pair of images (depth before grass, grass
	// mask) as PGM into the SKSE log folder on every 6th comparison in an exterior cell, at most 8 pairs (ring).
	inline bool      grassProbeDump{ false };
	// [Debug] FixedNoisePhase (diagnosis only, 2026-11-16 H-phase test): -1 = the march jitter's phase follows the frame
	// number (frame % 64, normal); 0 - 63 = that phase every frame, so the per-pixel jitter pattern stops changing.
	// Only the value passed to the shaders changes; the frame counter itself keeps counting.
	inline int       fixedNoisePhase{ -1 };
	// [Render] OwnRayTaps (2026-11-16 test, bush flicker): a full-resolution pixel whose distance-matching low-res taps
	// amount to fewer than this many equal ones marches its own ray (faded in over one tap below it). 0 = off, as before.
	inline float     ownRayTaps{ 0.0f };
	// [Render] UpsampleKernel (2026-11-16 test, bush flicker): 0 = the 3x3 window from floor with weights 1 / (1 + d^2), as
	// before; 1 = the window centred on the nearest texel with tent weights reaching 0 at 1.5 texels (no step at texel borders).
	inline int       upsampleKernel{ 1 };   // default 1 since 2026-11-16 (with TemporalBlend: no 4-px blocks)
	// [Render] FogBlurRadius (2026-11-16 test, bush flicker): blur the low-res march over (2r+1)^2 texels before the composite
	// (0 = no blur pass, as before; 1 or 2). FogBlurTolerance: neighbours whose scene distance differs by more than this share
	// are left out (fading over it); 0 = taken at any distance.
	inline int       fogBlurRadius{ 0 };
	inline float     fogBlurTolerance{ 0.0f };
	// [Render] TemporalBlend (2026-11-16 test, bush flicker): blend the low-res march with last frame's (reprojected) by
	// this weight of the new frame (0 = off, as before; 0.1 - 0.3: averaged over about 1 / weight frames).
	// TemporalTolerance: history whose scene distance differs by more than this share is not used.
	// TemporalClamp: 1 = clamp the history's fog amount to this frame's 3x3 neighbours first (fewer trails), 0 = no clamp.
	inline float     temporalBlend{ 0.1f };  // default 0.1 since 2026-11-16 (in-game: bush flicker -97 %, no trails seen)
	inline float     temporalTolerance{ 0.1f };
	inline bool      temporalClamp{ true };
	// [Render] NoisePattern (2026-11-16 test): 0 = the march offset as before (IGN moved per frame), 1 = the same pattern
	// shifted by a golden-ratio phase per frame (even spread of each pixel's offsets over time for the temporal blend).
	inline int       noisePattern{ 0 };
	// [Render] DepthPick (2026-11-16 test): 0 = each low-res texel marches its block's lower right pixel, as before;
	// 1 = checkerboard: the nearest pixel of the block on even texels, the farthest on odd ones (ThinFogStart is off then).
	inline int       depthPick{ 0 };
	// [Render] SunShadow (2026-11-17): 0 = off (as before); above 0 = the fog shades itself from the sun: the direct sun
	// reaching a fog point is exp(-SunShadow x the fog's optical depth toward the sun), so tops are sunlit and depths
	// show the sky-lit colour, most with a low sun (0 - 4; 1 = the single-scattering value). No effect at night.
	inline float     sunShadow{ 0.0f };
	// [Render] ShadeBrightness: brightness of fully shaded fog against sunlit fog when the sun carries its full share
	// of the fog's colour (0 - 1); its hue is the sky's.
	inline float     shadeBrightness{ 0.6f };
	// [Render] LocalLights (2026-11-17): 0 = off; 1 - 16 = the nearest that many game point lights (torches, fires) glow in
	// the fog. LocalLightStrength scales the halo (round, centred on the light, fading as rays pass farther from it).
	inline int       localLights{ 0 };
	inline float     localLightStrength{ 0.5f };
	// [Render] LocalLightReach: how far the glow spreads, as a multiple of each light's radius (0.25 - 3; 1 = as tuned)
	inline float     localLightReach{ 1.0f };
	// [Render] BankAmount (2026-11-17): 0 = off; up to 1 = the fog lies thicker and thinner over thousands of units
	// (0.45 to 1.3 x the layer), drifting slowly (after roxorvic's fog Banks).
	inline float     bankAmount{ 1.0f };

	// Render: wind-driven ground fog drawn by a screen raymarch through the fog columns around the player
	inline float worldSize{ 12288.0f };  // fog field side, units (about 176 m); with simResolution 384 and heightResolution 193: 32-unit wind texels, 64-unit height cells
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
	inline int   raymarchSteps{ 24 };     // over the ray's stretch inside the fade circle (RaymarchShaders.h MarchPixel): a level ray from the field centre has 243-unit steps at WorldSize 12288
	inline float profileSteepness{ 3.0f }; // fog density falls as exp(-a h/H) up the column (0 = even with height)
	// [Render] ThinFogStart, units from the eye: from here (all of it from twice this) thin low things (grass blades,
	// bushes) take the fog of the ground right behind them (RaymarchShaders.h ThinShare). 0 = off; 600 is the value
	// meant for release once the offline gates and a game test have passed (plan r12 step 2).
	inline float thinFogStart{ 0.0f };
	// [Render] GrassCover, units from the eye: from here (all of it from twice this) thin low plants standing in the layer take
	// the fog along their own ray continued to the ground behind them, by a share from how much of their neighbourhood is
	// ground (RaymarchShaders.h kCoverPS, GrassCoverM). 0 = off (no extra pass). Prototype 2026-11-15, offline only.
	inline float grassCover{ 0.0f };
	// [Render] ReliefGain: fog over a trail a little darker, over a bank a little brighter, by this much per unit of the
	// column height factor's departure s - 1 (RaymarchShaders.h CompositeOut; needs [Wind] SurfaceResponse > 0). 0 = off;
	// 0.08 is the value meant for release once the offline gates and a game test have passed (judge's plan r11 step 2: a
	// runner's trail 2.5 to 4 % darker).
	inline float reliefGain{ 0.0f };

	// Height fog: exponential fog from the ground to the horizon, under and beyond the wind field,
	// evaluated in closed form per pixel. Density falls by e every HeightFogFalloff above the base height.
	inline bool  heightFog{ false };   // off: Community Shaders' height fog usually covers this
	inline float heightFogDensity{ 0.002f };        // per metre at the base height
	inline float heightFogFalloff{ 1000.0f };       // units (about 14 m)
	inline float heightFogBaseOffset{ 0.0f };       // units above the (smoothed) ground height under the player
	inline float heightFogMaxDistance{ 150000.0f }; // units; rays to the sky stop here
	// Far tier: the same thin ground layer beyond the wind field, over the whole visible range, integrated along each
	// pixel's ray over the ground of the land table (Terrain), continued past the known land (no simulation).
	inline bool  farFog{ true };
	inline bool  farPools{ true };             // [FarFog] Pools: the coarse grid's pooled fog beyond the field
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
	inline bool  placeBankCap{ true };   // land at the water's edge held to that water's value, fading out across the shore band (PlacementRules::Build)
	inline float placeValleyStrength{ 0.35f };
	inline float placeValleyScale{ 150.0f };
	inline float placeValleyRadius{ 2048.0f };
	inline float placeLand{ 0.7f };
	inline float placeSteamHeight{ 150.0f };   // units of steam column per unit of steam (0 = old: steam only thickens the layer)
	inline float placeSteamDensity{ 2.0f };   // photos: hot-spring plumes turn opaque within 1-3 m
	inline float condEvaporation{ 0.2f };
	inline float condNightCold{ 0.13f };   // [Conditions] NightCold: FogConditions::Params::nightCold (no survival mod: added to the month's cold at night)
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
	inline int   simResolution{ 384 };      // 32-unit texels over WorldSize 12288
	inline int   pressureIterations{ 20 };
	inline float windScale{ 105.0f };        // Sky::windSpeed -> units/s. 300 moved the fog at 2-2.6 m/s in game (windSpeed 0.4-0.6, 2026-11-08); real fog drifts about 0.3-1 m/s (video estimates)
	// windRelax 0.5 and velocityDissipation 0.2: half of each (a wake that lasts twice as long) was measured and not
	// taken, the swirls vorticity confinement feeds then burst (WindSimCheck.cpp TestLongRunStable).
	inline float windRelax{ 0.5f };          // 1/s pull toward base wind + gusts
	// [Wind] InputSmoothing (2026-11-16 test): seconds over which the field follows the game's Sky wind (exponential,
	// time constant). 0 = the game's value every frame, as before. The game's frame-to-frame changes are logged either
	// way ("wind input:" lines every 5 s with EnableProfiler or EnableSelfTest).
	inline float windInputSmoothing{ 0.0f };
	inline float vorticity{ 2.0f };
	inline float gustAmount{ 0.6f };         // gust strength as a fraction of the base wind
	inline float gustMinimum{ 40.0f };       // gusts even in still weather
	inline float gustWavelength{ 1200.0f };
	inline float fogRefill{ 0.06f };         // 1/s: a hole in the fog fills back in over about 1 / this seconds (0.06: 17 s; measured: a runner's trail back at 0.9 after 24 s, a hit's hole after 38 s)
	inline float velocityDissipation{ 0.2f };
	inline float wallSlope{ 1.2f };          // ground rise per unit run that counts as a wall
	inline int   maxActors{ 32 };
	inline float actorRadius{ 70.0f };       // 1 m: reach of a body's drag on the air and radius of its push; a body wider than 1.4 m gets its own (WindRules::DragRadius)
	inline float actorFollow{ 6.0f };        // 1/s: how fast the air at a body takes the body's speed
	inline float actorPush{ 0.35f };         // fog on a runner's path thins to about exp(-this) on average: 0.35 -> 70 %, thinnest (about 50 %) just behind the runner (0 = off)
	inline float actorPushWalk{ 0.3f };      // share of actorPush a walker already gets: 0.3 x 0.35 -> the path thins to about 90 %
	inline float actorPushMinSpeed{ 220.0f }; // units/s; the running push starts here and is full at 1.5 x; the walker's share fades in from 0.18 x to 0.36 x (about 40 to 79 at 220)
	inline float wakeEvaporate{ 0.7f };      // [Wind] WakeEvaporate: fog evaporated in a moving body's wake, on its path: exp(-this x the push's gate), never under 0.15 x the layer by the wake alone (0 = off; 0.7 the test value, DEBUG-LEDGER.md 2026-11-14 15:xx)
	inline int   fogSampling{ 0 };          // [Wind] FogSampling: how CSAdvectFog reads the carried fog: 0 = bilinear (as before), 1 = Catmull-Rom held to the 4 nearest texels, on the thin side only and only in air a body drags (WindShaders.h)
	inline float blastDepth{ 4.0f };         // divergence integrated over an impact's pulse: the fog at the centre thins to about exp(-this)
	inline float blastDuration{ 0.6f };      // s: length of an impact's pulse
	inline float blastRadius{ 105.0f };      // 1.5 m: the hole's radius for a projectile hit and a shout (x 1.5); explosions use their own
	inline float blastPush{ 150.0f };        // units/s the air gains along the hit direction over the pulse (a shout: x 1.5)
	inline float explosionRadiusScale{ 0.8f }; // an explosion's hole radius = its own radius x this (measured: the fog is at half out to 0.93 of the explosion's radius; 0.5, as it was: 0.55 - 0.59)
	inline float fireClear{ 0.0f };          // [Wind] FireClear (2026-11-17): 0 = off; campfires, braziers and forges clear the fog round them at this rate (0.3 = a few seconds)
	inline float torchClear{ 0.01f };         // [Wind] TorchClear (2026-11-17): 0 = off; a held torch thins the fog round its holder at this rate
	inline float torchReach{ 60.0f };        // [Wind] TorchReach: the radius (units) of that thinning
	inline float fireEvaporate{ 3.0f };      // the part of blastDepth a fire hit boils off instead of pushing aside (at most all of it)
	// [Wind] SurfaceResponse k: where the field thins or piles up the fog relative to its pattern (q = rho / rho_eq) the layer's
	// top sinks or rises with it: column height x q^k held to 0.15..1.6, the column's mass kept (WindShaders.h CSColumn;
	// judge's plan r11 step 1a). 0 = off (heights from the shape alone, as before); 1.0 is the value meant for release once
	// the offline gates and a game test have passed. Gusts move q too, so with it on the top undulates in wind with nobody
	// near (measured std(q) 0.05 on flat ground, 0.15 beside a cliff; the user accepted that, 2026-11-14).
	// Held to 0..1 (Settings.cpp): s also lowers a pool's lid (CSColumn), which makes the pooled fog under it denser by
	// depth / lid. Up to k = 1 that stays within 1 + 0.6 x WispAmount, the peak FogRenderer.cpp PoolDensityNow's
	// visibility floor counts (WindSimCheck scans it); above 1 it does not (1.31 at 1.25, 1.39 at 1.5: review r5).
	inline float surfaceResponse{ 0.0f };

	// Terrain height cache
	inline int   heightResolution{ 193 };   // 64-unit height spacing over WorldSize 12288
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

	// Sets one key in the ini file (the in-game page, Menu.cpp): replaces the value on the key's line inside its section,
	// keeping every other line and comment, or adds the key at the end of the section (or a new section). The hot reload
	// then reads it. Returns false (logged) when the file cannot be read or written.
	bool WriteValue(const char* a_section, const char* a_key, const std::string& a_value);

	// The settings page holds the hot reload for the next half second, every frame one of its controls is held; a held
	// reload runs at the first check after that (also when the page closed with a control held).
	void HoldReload();
}
