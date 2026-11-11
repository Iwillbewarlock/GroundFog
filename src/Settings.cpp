#include "PCH.h"

#include "Settings.h"

#include <Windows.h>

#include <filesystem>
#include <fstream>
#include <map>

namespace Settings
{
	namespace
	{
		// Absolute path next to the game exe (a relative path is not resolved against the game folder).
		const std::wstring& IniPath()
		{
			static const std::wstring path = [] {
				wchar_t     exe[MAX_PATH]{};
				const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
				const std::filesystem::path dir =
					(n > 0 && n < MAX_PATH) ? std::filesystem::path(exe).parent_path() : std::filesystem::current_path();
				return (dir / L"Data" / L"SKSE" / L"Plugins" / L"GroundFog.ini").wstring();
			}();
			return path;
		}

		// The ini is parsed by hand: PrivateProfileRedirector (installed in TAKEALOOK) hooks the
		// GetPrivateProfileString family and keeps an in-memory copy, so edits made while the game runs
		// were never seen (2026-11-06). Keys are "section|key", lower-case ASCII.
		std::map<std::string, std::string> g_values;
		std::map<std::string, std::string> g_snapshot;   // copy for other threads, under g_snapshotLock
		std::mutex                         g_snapshotLock;

		std::string Lower(std::string a_text)
		{
			for (auto& c : a_text) {
				c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
			}
			return a_text;
		}

		constexpr const char* kSpace = " \t\r\n";

		std::string Trim(const std::string& a_text)
		{
			const auto first = a_text.find_first_not_of(kSpace);
			if (first == std::string::npos) {
				return {};
			}
			const auto last = a_text.find_last_not_of(kSpace);
			return a_text.substr(first, last - first + 1);
		}

		bool ParseFile()
		{
			g_values.clear();
			std::ifstream in{ std::filesystem::path{ IniPath() } };
			if (!in) {
				return false;
			}
			std::string line;
			std::string section;
			while (std::getline(in, line)) {
				if (const auto cut = line.find_first_of(";#"); cut != std::string::npos) {
					line.resize(cut);
				}
				line = Trim(line);
				if (line.empty()) {
					continue;
				}
				if (line.front() == '[' && line.back() == ']') {
					section = Lower(Trim(line.substr(1, line.size() - 2)));
					continue;
				}
				const auto eq = line.find('=');
				if (eq == std::string::npos) {
					continue;
				}
				g_values[section + "|" + Lower(Trim(line.substr(0, eq)))] = Trim(line.substr(eq + 1));
			}
			return true;
		}

		std::wstring ReadRaw(const wchar_t* a_section, const wchar_t* a_key)
		{
			const auto key = Lower(SKSE::stl::utf16_to_utf8(a_section).value_or("")) + "|" +
			                 Lower(SKSE::stl::utf16_to_utf8(a_key).value_or(""));
			const auto it = g_values.find(key);
			return it == g_values.end() ? std::wstring{} : SKSE::stl::utf8_to_utf16(it->second).value_or(L"");
		}

		void ReadFloat(const wchar_t* a_section, const wchar_t* a_key, float& a_out, float a_min, float a_max)
		{
			const auto raw = ReadRaw(a_section, a_key);
			if (raw.empty()) {
				return;
			}
			wchar_t*    end = nullptr;
			const float value = std::wcstof(raw.c_str(), &end);
			if (end == raw.c_str() || !std::isfinite(value)) {
				logger::warn("ini [{}] {}: '{}' is not a number, keeping {}",
					SKSE::stl::utf16_to_utf8(a_section).value_or("?"), SKSE::stl::utf16_to_utf8(a_key).value_or("?"),
					SKSE::stl::utf16_to_utf8(raw).value_or("?"), a_out);
				return;
			}
			const float clamped = std::clamp(value, a_min, a_max);
			if (clamped != value) {
				logger::warn("ini [{}] {}: {} is outside [{}, {}], using {}",
					SKSE::stl::utf16_to_utf8(a_section).value_or("?"), SKSE::stl::utf16_to_utf8(a_key).value_or("?"),
					value, a_min, a_max, clamped);
			}
			a_out = clamped;
		}

		void ReadInt(const wchar_t* a_section, const wchar_t* a_key, int& a_out, int a_min, int a_max)
		{
			float value = static_cast<float>(a_out);
			ReadFloat(a_section, a_key, value, static_cast<float>(a_min), static_cast<float>(a_max));
			a_out = static_cast<int>(std::lround(value));
		}

		void ReadBool(const wchar_t* a_section, const wchar_t* a_key, bool& a_out)
		{
			int value = a_out ? 1 : 0;
			ReadInt(a_section, a_key, value, 0, 1);
			a_out = value != 0;
		}
	}

	void Load()
	{
		const auto found = ParseFile();
		logger::info("ini {}: {}", SKSE::stl::utf16_to_utf8(IniPath()).value_or("?"), found ? "found" : "MISSING, using defaults");
		ReadBool(L"General", L"Enable", enable);
		ReadBool(L"General", L"EnableSelfTest", selfTest);
		ReadBool(L"General", L"EnableProfiler", profiler);
		int stage = static_cast<int>(drawStage);
		ReadInt(L"General", L"DrawStage", stage, 0, 1);
		drawStage = static_cast<DrawStage>(stage);
		ReadInt(L"General", L"DebugView", debugView, 0, 1);
		ReadInt(L"General", L"MatrixMode", matrixMode, 0, 2);
		ReadBool(L"General", L"CameraFilter", cameraFilter);
		ReadInt(L"General", L"Probe", probe, 0, 1);

		ReadFloat(L"Render", L"WorldSize", worldSize, 512.0f, 16384.0f);
		ReadFloat(L"Render", L"Thickness", thickness, 1.0f, 2000.0f);
		ReadFloat(L"Render", L"Opacity", opacity, 0.0f, 20.0f);
		ReadFloat(L"Render", L"Brightness", brightness, 0.0f, 10.0f);
		ReadFloat(L"Render", L"SunMix", sunMix, 0.0f, 2.0f);
		ReadFloat(L"Render", L"GlowStrength", glowStrength, 0.0f, 2.0f);
		ReadFloat(L"Render", L"GlowAnisotropy", glowAnisotropy, 0.0f, 0.95f);
		ReadFloat(L"Render", L"FogPower", fogPower, 0.1f, 10.0f);
		ReadFloat(L"Render", L"FogMax", fogMax, 0.0f, 1.0f);
		ReadInt(L"Render", L"ColorSpace", colorSpace, -1, 1);
		ReadFloat(L"Render", L"RaymarchScale", raymarchScale, 0.25f, 1.0f);
		ReadInt(L"Render", L"RaymarchSteps", raymarchSteps, 4, 64);
		ReadFloat(L"Render", L"ProfileSteepness", profileSteepness, 0.0f, 10.0f);

		ReadBool(L"FarFog", L"Enable", farFog);
		ReadFloat(L"FarFog", L"FlatLength", farFogFlatLength, 100.0f, 100000.0f);
		ReadFloat(L"FarFog", L"NoiseAmount", farFogNoiseAmount, 0.0f, 1.0f);
		ReadBool(L"FarFog", L"Pools", farPools);
		ReadBool(L"Conditions", L"Enable", conditions);
		ReadFloat(L"Conditions", L"NightLevel", condNight, 0.0f, 4.0f);
		ReadFloat(L"Conditions", L"MorningPeak", condMorningPeak, 0.0f, 4.0f);
		ReadFloat(L"Conditions", L"AfternoonLevel", condAfternoon, 0.0f, 4.0f);
		ReadFloat(L"Conditions", L"WindLow", condWindLow, 0.0f, 5.0f);
		ReadFloat(L"Conditions", L"WindHigh", condWindHigh, 0.0f, 5.0f);
		ReadFloat(L"Conditions", L"WindSuppress", condWindSuppress, 0.0f, 1.0f);
		ReadBool(L"Conditions", L"UseSurvivalTemperature", condUseSurvival);
		ReadBool(L"Placement", L"Enable", placement);
		ReadFloat(L"Placement", L"Lake", placeLake, 0.0f, 10.0f);
		ReadFloat(L"Placement", L"River", placeRiver, 0.0f, 10.0f);
		ReadFloat(L"Placement", L"MarshWater", placeMarshWater, 0.0f, 10.0f);
		ReadFloat(L"Placement", L"MarshLand", placeMarshLand, 0.0f, 10.0f);
		ReadFloat(L"Placement", L"HotSpringSteam", placeHotSpring, 0.0f, 10.0f);
		ReadFloat(L"Placement", L"Sea", placeSeaRadiation, 0.0f, 10.0f);
		ReadFloat(L"Placement", L"SeaSmoke", placeSeaSteam, 0.0f, 10.0f);
		ReadFloat(L"Placement", L"RiverSmoke", placeFreshSteam, 0.0f, 10.0f);
		ReadFloat(L"Placement", L"ShoreFactor", placeShoreFactor, 0.0f, 10.0f);
		ReadFloat(L"Placement", L"ShoreDistance", placeShoreDistance, 0.0f, 10000.0f);
		ReadFloat(L"Placement", L"ValleyStrength", placeValleyStrength, 0.0f, 2.0f);
		ReadFloat(L"Placement", L"ValleyScale", placeValleyScale, 10.0f, 5000.0f);
		ReadFloat(L"Placement", L"ValleyRadius", placeValleyRadius, 128.0f, 8192.0f);
		ReadFloat(L"Placement", L"Land", placeLand, 0.0f, 10.0f);
		ReadFloat(L"Placement", L"SteamHeight", placeSteamHeight, 0.0f, 600.0f);
		ReadFloat(L"Placement", L"SteamDensity", placeSteamDensity, 0.0f, 5.0f);
		ReadFloat(L"Conditions", L"Evaporation", condEvaporation, 0.0f, 2.0f);
		ReadFloat(L"Placement", L"PoolRadius", placePoolRadius, 0.0f, 4000.0f);
		ReadFloat(L"Placement", L"PoolMaxDepth", placePoolMaxDepth, 0.0f, 600.0f);
		ReadFloat(L"Placement", L"PoolDensity", placePoolDensity, 0.0f, 5.0f);
		ReadFloat(L"Placement", L"PoolMinVisibility", placePoolMinVisibility, 0.0f, 5000.0f);
		ReadFloat(L"Wind", L"WispAmount", wispAmount, 0.0f, 1.0f);
		ReadBool(L"Placement", L"CoarsePools", coarsePools);
		ReadFloat(L"Placement", L"CoarsePoolRadius", coarsePoolRadius, 1024.0f, 12000.0f);
		ReadFloat(L"Placement", L"CoarseTolerance", coarseTolerance, 20.0f, 2000.0f);
		ReadBool(L"Weather", L"Enable", weatherFog);
		ReadFloat(L"Weather", L"FlagPleasant", weatherFlagPleasant, 0.0f, 4.0f);
		ReadFloat(L"Weather", L"FlagCloudy", weatherFlagCloudy, 0.0f, 4.0f);
		ReadFloat(L"Weather", L"FlagRainy", weatherFlagRainy, 0.0f, 4.0f);
		ReadFloat(L"Weather", L"FlagSnow", weatherFlagSnow, 0.0f, 4.0f);
		ReadFloat(L"Weather", L"FarReference", weatherFarReference, 1000.0f, 1000000.0f);
		ReadBool(L"Weather", L"FollowCommunityShaders", weatherFollowCs);
		ReadFloat(L"Weather", L"CsReferenceDensity", weatherCsReference, 0.0001f, 1.0f);
		ReadBool(L"HeightFog", L"Enable", heightFog);
		ReadFloat(L"HeightFog", L"Density", heightFogDensity, 0.0f, 0.2f);
		ReadFloat(L"HeightFog", L"FalloffHeight", heightFogFalloff, 50.0f, 50000.0f);
		ReadFloat(L"HeightFog", L"BaseOffset", heightFogBaseOffset, -20000.0f, 20000.0f);
		ReadFloat(L"HeightFog", L"MaxDistance", heightFogMaxDistance, 1000.0f, 1000000.0f);
		ReadFloat(L"HeightFog", L"GameFogScale", gameFogScale, 0.25f, 20.0f);

		ReadBool(L"Wind", L"Enable", enableWind);
		ReadBool(L"Wind", L"Walls", enableWalls);
		ReadInt(L"Wind", L"Resolution", simResolution, 64, 512);
		simResolution = std::max(64, (simResolution / 8) * 8);
		ReadInt(L"Wind", L"PressureIterations", pressureIterations, 1, 80);
		ReadFloat(L"Wind", L"WindScale", windScale, 0.0f, 5000.0f);
		ReadFloat(L"Wind", L"WindRelax", windRelax, 0.0f, 10.0f);
		ReadFloat(L"Wind", L"Vorticity", vorticity, 0.0f, 20.0f);
		ReadFloat(L"Wind", L"GustAmount", gustAmount, 0.0f, 3.0f);
		ReadFloat(L"Wind", L"GustMinimum", gustMinimum, 0.0f, 1000.0f);
		ReadFloat(L"Wind", L"GustWavelength", gustWavelength, 100.0f, 10000.0f);
		ReadFloat(L"Wind", L"FogRefill", fogRefill, 0.0f, 5.0f);
		ReadFloat(L"Wind", L"VelocityDissipation", velocityDissipation, 0.0f, 10.0f);
		ReadFloat(L"Wind", L"WallSlope", wallSlope, 0.1f, 20.0f);
		ReadInt(L"Wind", L"MaxActors", maxActors, 0, 32);
		ReadFloat(L"Wind", L"ActorRadius", actorRadius, 10.0f, 1000.0f);
		ReadFloat(L"Wind", L"ActorFollow", actorFollow, 0.0f, 50.0f);
		ReadFloat(L"Wind", L"ActorPush", actorPush, 0.0f, 3.0f);
		ReadFloat(L"Wind", L"ActorPushMinSpeed", actorPushMinSpeed, 0.0f, 2000.0f);
		ReadFloat(L"Wind", L"BlastDepth", blastDepth, 0.0f, 20.0f);
		ReadFloat(L"Wind", L"BlastDuration", blastDuration, 0.1f, 5.0f);
		ReadFloat(L"Wind", L"BlastRadius", blastRadius, 10.0f, 1000.0f);
		ReadFloat(L"Wind", L"BlastPush", blastPush, 0.0f, 2000.0f);

		ReadInt(L"Terrain", L"HeightResolution", heightResolution, 9, 257);
		{
			// The wind grid must split each height cell into whole texels so a grid move is a whole-texel shift.
			const int cells = heightResolution - 1;
			const int fixedRes = std::max(cells, static_cast<int>(std::lround(static_cast<float>(simResolution) / cells)) * cells);
			if (fixedRes != simResolution) {
				logger::warn("ini [Wind] Resolution={} is not a multiple of HeightResolution-1 ({}); using {}", simResolution, cells, fixedRes);
				simResolution = std::min(fixedRes, 1024);
			}
		}
		ReadFloat(L"Terrain", L"RecenterDistance", recenterDistance, 64.0f, 4096.0f);
		ReadInt(L"Terrain", L"HeightRetriesPerFrame", heightRetriesPerFrame, 0, 4096);
		{
			const std::scoped_lock lock(g_snapshotLock);
			g_snapshot = g_values;
		}
	}

	namespace
	{
		FILETIME LastWrite()
		{
			WIN32_FILE_ATTRIBUTE_DATA data{};
			if (!GetFileAttributesExW(IniPath().c_str(), GetFileExInfoStandard, &data)) {
				return {};
			}
			return data.ftLastWriteTime;
		}

		FILETIME                              g_lastWrite{};
		bool                                  g_haveWrite{ false };
		std::chrono::steady_clock::time_point g_lastCheck{};
		constexpr auto                        kCheckInterval = std::chrono::seconds(2);
	}

	bool ReloadIfChanged()
	{
		const auto now = std::chrono::steady_clock::now();
		if (g_haveWrite && now - g_lastCheck < kCheckInterval) {
			return false;
		}
		g_lastCheck = now;
		const FILETIME write = LastWrite();
		if (!g_haveWrite) {
			g_lastWrite = write;
			g_haveWrite = true;
			return false;
		}
		if (CompareFileTime(&write, &g_lastWrite) == 0) {
			return false;
		}
		g_lastWrite = write;
		Load();
		logger::info("ini changed on disk: reloaded");
		LogAll();
		return true;
	}

	void LogAll()
	{
		logger::info("settings: Enable={} SelfTest={} Profiler={} DrawStage={} DebugView={} MatrixMode={} CameraFilter={} Probe={}",
			enable, selfTest, profiler, static_cast<int>(drawStage), debugView, matrixMode, cameraFilter, probe);
		logger::info("settings: WorldSize={} Thickness={} Opacity={} ProfileSteepness={}", worldSize, thickness, opacity, profileSteepness);
		logger::info("settings: Brightness={} SunMix={} FogPower={} FogMax={} ColorSpace={} RaymarchScale={} RaymarchSteps={}", brightness,
			sunMix, fogPower, fogMax, colorSpace, raymarchScale, raymarchSteps);
		logger::info("settings: Wind Enable={} Walls={} Resolution={} PressureIterations={} WindScale={} Vorticity={} GustAmount={} WallSlope={}",
			enableWind, enableWalls, simResolution, pressureIterations, windScale, vorticity, gustAmount, wallSlope);
		logger::info("settings: HeightResolution={} RecenterDistance={} HeightRetriesPerFrame={}",
			heightResolution, recenterDistance, heightRetriesPerFrame);
		logger::info("settings: FarFog Enable={} FlatLength={} NoiseAmount={} Pools={}; PoolMinVisibility={}", farFog, farFogFlatLength, farFogNoiseAmount, farPools, placePoolMinVisibility);
		logger::info("settings: Conditions Enable={} Night={} MorningPeak={} Afternoon={} Wind {}..{} x{} Survival={}", conditions, condNight,
			condMorningPeak, condAfternoon, condWindLow, condWindHigh, condWindSuppress, condUseSurvival);
		logger::info("settings: Placement Enable={} Lake={} River={} MarshWater={} MarshLand={} HotSpring={} Sea={} SeaSmoke={} RiverSmoke={} Shore {} over {} Valley {} / {} within {} Land={} SteamHeight={} SteamDensity={} Evaporation={} Pool {} / {} x{}",
			placement, placeLake, placeRiver, placeMarshWater, placeMarshLand, placeHotSpring, placeSeaRadiation, placeSeaSteam, placeFreshSteam,
			placeShoreFactor, placeShoreDistance, placeValleyStrength, placeValleyScale, placeValleyRadius, placeLand, placeSteamHeight,
			placeSteamDensity, condEvaporation, placePoolRadius, placePoolMaxDepth, placePoolDensity);
		logger::info("settings: CoarsePools={} CoarsePoolRadius={} CoarseTolerance={} WispAmount={} GlowStrength={} GlowAnisotropy={}", coarsePools,
			coarsePoolRadius, coarseTolerance, wispAmount, glowStrength, glowAnisotropy);
		logger::info("settings: Weather Enable={} Flags P{} C{} R{} S{} FarReference={} FollowCommunityShaders={} CsReferenceDensity={}",
			weatherFog, weatherFlagPleasant, weatherFlagCloudy, weatherFlagRainy, weatherFlagSnow, weatherFarReference, weatherFollowCs, weatherCsReference);
		logger::info("settings: HeightFog Enable={} Density={} FalloffHeight={} BaseOffset={} MaxDistance={} GameFogScale={}",
			heightFog, heightFogDensity, heightFogFalloff, heightFogBaseOffset, heightFogMaxDistance, gameFogScale);
	}

	std::string GetString(const char* a_section, const char* a_key, const char* a_default)
	{
		std::string key = Lower(a_section);   // ParseFile stores "section|key" in lower case
		key += "|";
		for (const char* c = a_key; *c; ++c) key += static_cast<char>(std::tolower(static_cast<unsigned char>(*c)));
		const std::scoped_lock lock(g_snapshotLock);
		const auto it = g_snapshot.find(key);
		return it == g_snapshot.end() ? std::string(a_default) : it->second;
	}

	std::vector<std::pair<std::string, std::string>> Section(const char* a_section)
	{
		const std::string prefix = Lower(a_section) + "|";
		std::vector<std::pair<std::string, std::string>> out;
		const std::scoped_lock lock(g_snapshotLock);
		for (auto it = g_snapshot.lower_bound(prefix); it != g_snapshot.end() && it->first.compare(0, prefix.size(), prefix) == 0; ++it) {
			out.emplace_back(it->first.substr(prefix.size()), it->second);
		}
		return out;
	}
}
