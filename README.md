# GroundFog

An SKSE plugin for Skyrim Special Edition that draws a thin layer of fog on the ground outdoors, moved by a small
real-time wind simulation around the player. One DLL and one ini; no ESP, nothing stored in saves, no shader files of
other mods changed.

Mod page: https://www.nexusmods.com/skyrimspecialedition/mods/198571

## What it does

- A fog field about 176 m wide around the player, carried by a 2D fluid simulation on the GPU (384 x 384 cells).
  Running actors, spell projectiles and explosions push it aside; cliffs and steep slopes block it.
- Beyond that field, a thin far layer to the horizon and pooled fog in hollows and valleys of the loaded terrain.
- The amount follows the time of day, the weather, water and wet ground, the worldspace (`[Worldspaces]`) and the
  game's weather regions (`[Regions]`).
- The fog has no colour of its own: it takes the sky's and the sun's light.
- Its own draw passes in D3D11 after the game's scene; works with or without Community Shaders or ENB.

Settings are in `dist/SKSE/Plugins/GroundFog.ini`, each with a comment. The ini is read again while the game runs.

## Requirements

- SKSE64 and Address Library for SKSE Plugins.
- Runtimes: built for Skyrim SE and AE through CommonLibSSE-NG. Run by the author on 1.6.1170; 1.7.104 loaded and
  drew fog in a short test. Skyrim VR is not built from this branch.
- Optional: powerofthree's Tweaks, for region and worldspace names in the log and for EditorID keys in the ini.

## Building

Windows, Visual Studio 2022 or later with the C++ desktop workload, CMake, Ninja and vcpkg.

```
git clone --recursive https://github.com/Iwillbewarlock/GroundFog.git
cd GroundFog
set VCPKG_ROOT=C:\path\to\vcpkg
build.cmd
```

`build.cmd` configures the `release` preset, builds `build\release\GroundFog.dll` and runs the offline checks
(`FogShaderCheck`: the real shaders on the WARP software device plus the CPU-side rules). It stops when a check fails.

Offline tools in `tools/offline` (none of them needs the game):

| Tool | What it is for |
|---|---|
| `FogShaderCheck` | the checks `build.cmd` runs |
| `RaymarchExport`, `PlacementExport` | fog columns and images for the Blender Cycles comparison in `tools/blender` |
| `HatchRepro` (`build-hatch.cmd`) | per-surface pixel noise on a far wall and roof |
| `PoolBoxRepro` (`build-poolbox.cmd`) | pooled fog in a valley through the whole chain |
| `SkyMeshRepro` (`build-skymesh.cmd`) | a sky mesh that writes depth, under the shaders of three commits |

## Layout

- `src/` the plugin. `FogRenderer.cpp` draws; `WindSim.*` and `WindShaders.h` are the fluid simulation and the fog
  columns; `RaymarchShaders.h` is the march and the composite; `PlacementRules.h`, `RegionRules.h`,
  `WorldspaceRules.h`, `WeatherRules.h` and `FogConditions.h` decide where and when fog forms; `Hooks.cpp` and
  `CallScan.cpp` find the draw points.
- `dist/` the ini as shipped.
- `extern/CommonLibSSE-NG` submodule (alandtse/CommonLibVR, v16.1.0).

## License

GPL-3.0, see `COPYING.txt`. The plugin links CommonLibSSE-NG statically, which is licensed GPL-3.0-or-later with a
modding exception and a linking exception (see its `EXCEPTIONS.md`); a plugin that links it has to be under the GPL
as well. Version 0.1.0 of the mod was built against the older MIT-licensed CommonLibSSE-NG (v4.26.1).

## Credits

- yomotsu, ground-fog web demo (https://yomotsu.github.io/ground-fog/): the idea of a thin 2D fog field that is
  inflated in height and raymarched, and of moving that fog on a 2D fluid grid. Early builds followed the demo
  closely; the released code was rewritten from the sources below.
- Jos Stam, "Stable Fluids" (SIGGRAPH 1999), and Mark Harris, "Fast Fluid Dynamics Simulation on the GPU" (GPU Gems,
  chapter 38): the fluid method. Walls follow Crane, Llamas and Tariq (GPU Gems 3, chapter 30).
- Richard Bird and Roland Hulstrom, "A Simplified Clear Sky Model for Direct and Diffuse Insolation on Horizontal
  Surfaces" (SERI, 1981), computed with pvlib: the sun's and the sky's share of the light for the fog's colour.
- Community Shaders team: engine addresses for the draw points, and the way Community Shaders gathers scene lights
  (Light Limit Fix) and reads the sun's shadow maps, which GroundFog follows.
- roxorvic (Victor), Ground Fog in his Community Shaders fork (skyrim-community-shaders, branch fluid-field/fog-v3,
  GPL-3.0): the formulas for the fog's sun shading, the light glow, the large fog banks and fires clearing the fog.
  Used with permission.
- NearMidnightNow, Dynamic Terrain Deformation (GPL-3.0): its heat-source detection (fire-spell hazards and held
  torches) is used for fire spells and torches clearing the fog, with permission; earlier a reference for projectile
  impact and terrain height lookups.
- QTR-Modding, SKSE Menu Framework 3 (GPL-3.0): its client header `include/SKSEMenuFramework.h`, for the optional
  in-game settings page.
- powerofthree: the `GetFormEditorID` export of po3 Tweaks, called for names when it is installed.
- CommonLibSSE-NG contributors.
