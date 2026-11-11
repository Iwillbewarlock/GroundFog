#include "PCH.h"

#include "FogRenderer.h"
#include "GameFog.h"
#include "WeatherFog.h"
#include "Hooks.h"
#include "SelfTest.h"
#include "WindField.h"
#include "Settings.h"

namespace
{
	void SetupLog()
	{
		auto path = logger::log_directory();
		if (!path) {
			return;
		}
		*path /= "GroundFog.log";
		auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path->string(), true);
		auto log = std::make_shared<spdlog::logger>("global", std::move(sink));
		log->set_level(spdlog::level::info);
		log->flush_on(spdlog::level::info);
		spdlog::set_default_logger(std::move(log));
		spdlog::set_pattern("[%H:%M:%S.%e] [%l] %v");
	}

	void OnMessage(SKSE::MessagingInterface::Message* a_msg)
	{
		switch (a_msg->type) {
		case SKSE::MessagingInterface::kDataLoaded:
			WindField::RegisterEvents();
			if (Settings::gameFogScale != 1.0f) {
				GameFog::Apply(Settings::gameFogScale);   // 1 = records untouched
			}
			WeatherFog::Build();
			SelfTest::ReportLoad();
			break;
		case SKSE::MessagingInterface::kPreLoadGame:
		case SKSE::MessagingInterface::kNewGame:
			FogRenderer::Reset();
			WindField::Reset();
			break;
		default:
			break;
		}
	}
}

SKSEPluginInfo(
	.Version = REL::Version{ 0, 2, 0, 0 },
	.Name = "GroundFog"sv,
	.RuntimeCompatibility = SKSE::VersionIndependence::AddressLibrary)

SKSEPluginLoad(const SKSE::LoadInterface* a_skse)
{
	SetupLog();
	// Skyrim VR (runtime 1.4.x) is not supported: this build has no VR addresses. Decline before anything looks one up.
	if (const auto runtime = a_skse->RuntimeVersion(); runtime[0] == 1 && runtime[1] == 4) {
		logger::error("GroundFog does not support Skyrim VR (runtime {}); not loaded", runtime.string());
		return false;
	}
	SKSE::Init(a_skse, false);  // false: keep our own log set up above
	logger::info("GroundFog 0.2.0 loading on runtime {}", a_skse->RuntimeVersion().string());

	Settings::Load();
	if (Settings::enable) {
		// As early as possible: the renderer's first draw comes before kDataLoaded (in game 2026-11-08: 2.6 s on the render
		// thread with nothing cached yet). D3DCompile needs no device.
		FogRenderer::PrecompileShaders();
	}
	Settings::LogAll();
	if (!Settings::enable) {
		logger::info("Enable=0 in GroundFog.ini: nothing installed");
		return true;
	}

	Hooks::Install();
	WindField::Install();

	if (auto* messaging = SKSE::GetMessagingInterface(); !messaging || !messaging->RegisterListener(OnMessage)) {
		logger::warn("could not register the SKSE message listener; save loads will not reset the height cache");
	}
	return true;
}
