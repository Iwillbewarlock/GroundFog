#pragma once

// With EnableSelfTest=1, writes one PASS/FAIL block to GroundFog.log after kFramesToWait drawn frames
// outdoors (and a first block about the hooks right after load). Claude reads the log to judge the fog pass.
namespace SelfTest
{
	void ReportLoad();  // hooks and CS detection, at kDataLoaded
	void Tick();        // once per call of the chosen draw stage
}
