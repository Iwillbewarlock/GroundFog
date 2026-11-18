cmake_minimum_required(VERSION 3.21)

project(GroundFog VERSION 0.3.1 LANGUAGES CXX)

set(ENABLE_SKYRIM_SE ON  CACHE BOOL "" FORCE)
set(ENABLE_SKYRIM_AE ON  CACHE BOOL "" FORCE)
set(ENABLE_SKYRIM_VR OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(SKSE_SUPPORT_XBYAK ON CACHE BOOL "" FORCE)

add_subdirectory(extern/CommonLibSSE-NG CommonLibSSE EXCLUDE_FROM_ALL)

add_library(${PROJECT_NAME} SHARED)

target_sources(${PROJECT_NAME} PRIVATE
    src/plugin.cpp
    src/Settings.cpp
    src/CallScan.cpp
    src/Hooks.cpp
    src/GrassProbe.cpp
    src/Terrain.cpp
    src/FogRenderer.cpp
    src/GpuTimer.cpp
    src/SelfTest.cpp
    src/WindSim.cpp
    src/WindField.cpp
    src/GameFog.cpp
    src/WeatherFog.cpp
    src/Menu.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
# Sources are UTF-8: the settings page's Korean text must reach ImGui as UTF-8 whatever the system code page (2026-11-17).
target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
target_precompile_headers(${PROJECT_NAME} PRIVATE include/PCH.h)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CommonLibSSE
    d3d11.lib
    d3dcompiler.lib
    dxguid.lib
)

option(BUILD_OFFLINE_TOOLS "Build the offline shader check" ON)

if(BUILD_OFFLINE_TOOLS)
    # Offline checks: compiles the wind and raymarch shaders and runs them on WARP, plus the CPU-side rules.
    add_executable(FogShaderCheck tools/offline/FogShaderCheck.cpp tools/offline/WindSimCheck.cpp tools/offline/RaymarchCheck.cpp tools/offline/SceneRatioCheck.cpp src/CallScan.cpp src/WindSim.cpp)
    target_include_directories(FogShaderCheck PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(FogShaderCheck PRIVATE cxx_std_23)
    target_compile_definitions(FogShaderCheck PRIVATE GROUNDFOG_OFFLINE)
    target_link_libraries(FogShaderCheck PRIVATE d3d11.lib d3dcompiler.lib dxguid.lib)
    # Writes fog columns and GPU raymarch images for the Blender Cycles comparison (tools/blender).
    add_executable(RaymarchExport tools/offline/RaymarchExport.cpp src/WindSim.cpp)
    target_include_directories(RaymarchExport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(RaymarchExport PRIVATE cxx_std_23)
    target_link_libraries(RaymarchExport PRIVATE d3d11.lib d3dcompiler.lib dxguid.lib)
    # Street-level wall and roof under the real shaders: per-surface pixel noise (hatching report 2026-11-11).
    add_executable(HatchRepro tools/offline/HatchRepro.cpp src/WindSim.cpp)
    target_include_directories(HatchRepro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(HatchRepro PRIVATE cxx_std_23)
    target_link_libraries(HatchRepro PRIVATE d3d11.lib d3dcompiler.lib dxguid.lib)
    # A valley with pooled fog under the whole chain (terrain, pools, columns, march, composite): flat faces in the sky (report 2026-11-12).
    add_executable(PoolBoxRepro tools/offline/PoolBoxRepro.cpp src/WindSim.cpp)
    target_include_directories(PoolBoxRepro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(PoolBoxRepro PRIVATE cxx_std_23)
    target_link_libraries(PoolBoxRepro PRIVATE d3d11.lib d3dcompiler.lib dxguid.lib)
    add_executable(PlacementExport tools/offline/PlacementExport.cpp src/WindSim.cpp)
    target_include_directories(PlacementExport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(PlacementExport PRIVATE cxx_std_23)
    target_link_libraries(PlacementExport PRIVATE d3d11.lib d3dcompiler.lib dxguid.lib)
    # A runner's trail by heading on the grid, fed as WindField::Update feeds the field (ledger 2026-11-15; not part of ALL: build-wakeheading.cmd).
    add_executable(WakeHeading EXCLUDE_FROM_ALL tools/offline/WakeHeading.cpp src/WindSim.cpp)
    target_include_directories(WakeHeading PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(WakeHeading PRIVATE cxx_std_23)
    target_compile_definitions(WakeHeading PRIVATE GROUNDFOG_OFFLINE NOMINMAX)
    target_link_libraries(WakeHeading PRIVATE d3d11.lib d3dcompiler.lib dxguid.lib)
    # The grass depth probe's pure parts (GrassProbeMath.h, CallScan::FindJumpsTo) and a depth copy round trip on WARP.
    # Not part of ALL and not run by build.cmd, so build.cmd's PASS/FAIL lines stay as they are: build-grassprobe.cmd.
    add_executable(GrassProbeCheck EXCLUDE_FROM_ALL tools/offline/GrassProbeCheck.cpp src/CallScan.cpp)
    target_include_directories(GrassProbeCheck PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(GrassProbeCheck PRIVATE cxx_std_23)
    target_compile_definitions(GrassProbeCheck PRIVATE NOMINMAX)
    target_link_libraries(GrassProbeCheck PRIVATE d3d11.lib)
endif()
