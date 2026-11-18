#include "PCH.h"

#include <filesystem>
#include <format>
#include <map>

// SKSE Menu Framework 3 client header (QTR-Modding/SKSE-Menu-Framework-3 resources/SKSEMenuFramework.h, GPL-3.0):
// header only, resolves the framework's functions at run time, so nothing is linked.
#include "SKSEMenuFramework.h"

#include "Menu.h"
#include "Settings.h"

// The page edits the Settings values directly while a control is dragged (the fog follows at once; the renderer reads
// them every frame) and writes the ini when the control is released, so the normal hot reload (Settings::ReloadIfChanged)
// takes the same values and they survive a restart. Only settings the renderer reads every frame are offered; the rest
// stay in the ini.
namespace Menu
{
	namespace
	{
		namespace UI = ImGuiMCP;

		bool g_running = true;
		bool g_anyActive = false;   // a control was being dragged or typed in this frame

		// One piece of text in each menu language ([General] MenuLanguage: 0 English, 1 Korean). Korean shows only when
		// SKSE Menu Framework's font has Korean on (its ini [Fonts] EnableKorean, on by default).
		struct Tr
		{
			const char* en;
			const char* ko;
		};

		const char* T(const Tr& a_text) { return Settings::menuLanguage == 1 ? a_text.ko : a_text.en; }

		// The label in the current language with the ini key as its ImGui ID ("###"): a control keeps its state when the
		// language changes.
		std::string Label(const Tr& a_text, const char* a_key) { return std::format("{}###{}", T(a_text), a_key); }

		void Hint(const Tr& a_text)
		{
			if (UI::IsItemHovered(0)) {
				UI::SetTooltip("%s", T(a_text));
			}
		}

		std::string Number(float a_value) { return std::format("{}", a_value); }

		void Float(const Tr& a_label, const char* a_section, const char* a_key, float& a_value, float a_min, float a_max, const char* a_format,
			const Tr& a_hint)
		{
			// AlwaysClamp: a Ctrl+click typed value is not held to the slider's range otherwise (Codex review 2026-11-17:
			// a raymarch scale of 100 asked for a target 100 x the screen); the clamp after it covers any other way in
			// clamped only when edited: an ini value outside the slider's range (but inside the loader's) stays as it is
			// until the control is touched (Codex review 2026-11-17)
			if (UI::SliderFloat(Label(a_label, a_key).c_str(), &a_value, a_min, a_max, a_format, UI::ImGuiSliderFlags_AlwaysClamp)) {
				a_value = std::clamp(a_value, a_min, a_max);
			}
			g_anyActive = g_anyActive || UI::IsItemActive();
			const bool done = UI::IsItemDeactivatedAfterEdit();
			Hint(a_hint);
			if (done) {
				Settings::WriteValue(a_section, a_key, Number(a_value));
			}
		}

		// As Float, but the value reaches the renderer only when the control is released: the raymarch scale resizes the
		// render targets, so a drag would rebuild them every frame (advisor 2026-11-17).
		void FloatOnRelease(const Tr& a_label, const char* a_section, const char* a_key, float& a_value, float a_min, float a_max, const char* a_format,
			const Tr& a_hint)
		{
			static std::map<std::string, float> pending;   // by ini key (the label changes with the language)
			auto        it = pending.find(a_key);
			float       v = it != pending.end() ? it->second : a_value;
			if (UI::SliderFloat(Label(a_label, a_key).c_str(), &v, a_min, a_max, a_format, UI::ImGuiSliderFlags_AlwaysClamp)) {
				pending[a_key] = std::clamp(v, a_min, a_max);
			}
			const bool active = UI::IsItemActive();
			g_anyActive = g_anyActive || active;
			const bool done = UI::IsItemDeactivatedAfterEdit();
			Hint(a_hint);
			if (done) {
				a_value = std::clamp(v, a_min, a_max);
				pending.erase(a_key);
				Settings::WriteValue(a_section, a_key, Number(a_value));
			} else if (!active) {
				pending.erase(a_key);
			}
		}

		void Int(const Tr& a_label, const char* a_section, const char* a_key, int& a_value, int a_min, int a_max, const Tr& a_hint)
		{
			if (UI::SliderInt(Label(a_label, a_key).c_str(), &a_value, a_min, a_max, "%d", UI::ImGuiSliderFlags_AlwaysClamp)) {
				a_value = std::clamp(a_value, a_min, a_max);
			}
			g_anyActive = g_anyActive || UI::IsItemActive();
			const bool done = UI::IsItemDeactivatedAfterEdit();
			Hint(a_hint);
			if (done) {
				Settings::WriteValue(a_section, a_key, std::to_string(a_value));
			}
		}

		void Flag(const Tr& a_label, const char* a_section, const char* a_key, bool& a_value, const Tr& a_hint)
		{
			const bool changed = UI::Checkbox(Label(a_label, a_key).c_str(), &a_value);
			Hint(a_hint);
			if (changed) {
				Settings::WriteValue(a_section, a_key, a_value ? "1" : "0");
			}
		}

		void FlagInt(const Tr& a_label, const char* a_section, const char* a_key, int& a_value, const Tr& a_hint)
		{
			bool on = a_value != 0;
			const bool changed = UI::Checkbox(Label(a_label, a_key).c_str(), &on);
			Hint(a_hint);
			if (changed) {
				a_value = on ? 1 : 0;
				Settings::WriteValue(a_section, a_key, on ? "1" : "0");
			}
		}

		// A list choice saved as its index (the items '\0'-separated, ending in "\0\0").
		void Choice(const Tr& a_label, const char* a_section, const char* a_key, int& a_value, const Tr& a_items)
		{
			int value = a_value;
			if (UI::Combo(Label(a_label, a_key).c_str(), &value, T(a_items), -1) && value != a_value) {
				a_value = value;
				Settings::WriteValue(a_section, a_key, std::to_string(value));
			}
		}

		bool Header(const Tr& a_text, const char* a_id) { return UI::CollapsingHeader(Label(a_text, a_id).c_str(), 0); }

		void __stdcall Render()
		{
			g_anyActive = false;
			UI::TextWrapped("%s", T({ "Changes apply at once and are saved to GroundFog.ini when a control is released.",
									  "바꾼 값은 바로 적용되고, 조절을 마치고 손을 떼면 GroundFog.ini에 저장됩니다." }));
			if (!g_running) {
				UI::TextWrapped("%s", T({ "GroundFog was off when the game started: tick Enable, then restart the game.",
										  "게임을 시작할 때 GroundFog가 꺼져 있었습니다. '켜기'를 체크한 뒤 게임을 다시 시작하세요." }));
			}
			UI::Separator();
			if (Header({ "Fog", "안개" }, "HeaderFog")) {
				Flag({ "Enable", "켜기" }, "General", "Enable", Settings::enable, { "Draw the ground fog.", "땅안개를 그립니다." });
				Float({ "Opacity", "짙기" }, "Render", "Opacity", Settings::opacity, 0.0f, 2.0f, "%.3f",
					{ "How thick the fog is (per metre).", "안개가 얼마나 짙은지 (1 m당)." });
				Float({ "Thickness", "층 높이" }, "Render", "Thickness", Settings::thickness, 20.0f, 600.0f, "%.0f",
					{ "How tall the layer is (units; 70 = 1 m).", "안개층의 높이 (게임 단위, 70 = 1 m)." });
				Float({ "Brightness", "밝기" }, "Render", "Brightness", Settings::brightness, 0.0f, 3.0f, "%.2f",
					{ "Overall brightness of the fog.", "안개 전체의 밝기." });
				Float({ "Profile steepness", "바닥 쏠림" }, "Render", "ProfileSteepness", Settings::profileSteepness, 0.0f, 10.0f, "%.1f",
					{ "Higher keeps more of the fog near the ground; 0 = even with height.",
						"높을수록 안개가 바닥 가까이에 몰립니다. 0 = 높이와 상관없이 고르게." });
				Float({ "Large banks", "큰 안개 덩어리" }, "Render", "BankAmount", Settings::bankAmount, 0.0f, 1.0f, "%.2f",
					{ "The fog lies thicker in some places and thinner in others over hundreds of metres (0 = off).",
						"수백 m에 걸쳐 안개가 어떤 곳은 짙게, 어떤 곳은 옅게 깔립니다 (0 = 끔)." });
				Float({ "Fires clear fog", "불이 안개를 걷음" }, "Wind", "FireClear", Settings::fireClear, 0.0f, 2.0f, "%.2f",
					{ "Campfires, braziers, forges and fire spells' burning ground clear the fog round them (0 = off; 0.3 = within a few seconds).",
						"모닥불, 화로, 대장간 화덕, 불 마법이 남긴 바닥 불이 둘레의 안개를 걷어 냅니다 (0 = 끔, 0.3 = 몇 초 안에)." });
				Float({ "Torches thin fog", "횃불이 안개를 옅게 함" }, "Wind", "TorchClear", Settings::torchClear, 0.0f, 0.2f, "%.3f",
					{ "A torch in someone's hand thins the fog a little round them (0 = off).", "횃불을 든 사람 둘레의 안개가 조금 옅어집니다 (0 = 끔)." });
				Float({ "Torch reach", "횃불 범위" }, "Wind", "TorchReach", Settings::torchReach, 20.0f, 300.0f, "%.0f",
					{ "Radius of that thinning (units; 70 = 1 m).", "옅어지는 범위의 반지름 (게임 단위, 70 = 1 m)." });
			}
			if (Header({ "Quality", "품질" }, "HeaderQuality")) {
				FloatOnRelease({ "Raymarch scale", "레이마치 해상도" }, "Render", "RaymarchScale", Settings::raymarchScale, 0.25f, 1.0f, "%.2f",
					{ "Resolution of the fog against the screen. Higher is sharper and costs more GPU time.",
						"화면 대비 안개의 해상도. 높을수록 선명하지만 GPU 시간이 더 듭니다." });
				Int({ "Raymarch steps", "레이마치 단계 수" }, "Render", "RaymarchSteps", Settings::raymarchSteps, 8, 48,
					{ "Samples along each ray.", "시선 하나를 따라 안개를 몇 번 계산하는지." });
				Float({ "Temporal blend", "프레임 섞기" }, "Render", "TemporalBlend", Settings::temporalBlend, 0.0f, 1.0f, "%.2f",
					{ "Weight of the new frame against the last ones (0 = off). Lower is smoother over bushes, slower to follow changes.",
						"새 프레임을 이전 프레임들과 섞는 비율 (0 = 끔). 낮을수록 덤불 위가 부드럽지만 변화를 늦게 따라갑니다." });
				FlagInt({ "Smooth upsampling", "부드러운 확대" }, "Render", "UpsampleKernel", Settings::upsampleKernel,
					{ "Smooth weights when the fog is stretched to the screen.", "안개를 화면 크기로 늘릴 때 픽셀을 부드럽게 섞습니다." });
				FlagInt({ "Checkerboard depth", "체커보드 깊이" }, "Render", "DepthPick", Settings::depthPick,
					{ "Alternate the nearest and farthest pixel per fog texel: cleaner fog at bush and leaf edges.",
						"안개 텍셀마다 가장 가까운 픽셀과 가장 먼 픽셀을 번갈아 씁니다. 덤불과 나뭇잎 가장자리의 안개가 깔끔해집니다." });
				FlagInt({ "Golden-ratio noise", "황금비 노이즈" }, "Render", "NoisePattern", Settings::noisePattern,
					{ "Spread the sample offsets evenly over frames.", "계산하는 위치를 프레임마다 고르게 흩뜨립니다." });
			}
			if (Header({ "Light", "빛" }, "HeaderLight")) {
				Float({ "Sun shadow", "햇빛 가림" }, "Render", "SunShadow", Settings::sunShadow, 0.0f, 4.0f, "%.2f",
					{ "The fog shades itself from the sun: lit tops, bluer depths (0 = off; 1 = physical).",
						"안개가 스스로 햇빛을 가립니다. 윗면은 밝고 깊은 곳은 푸르게 (0 = 끔, 1 = 물리 기준)." });
				Float({ "Shade brightness", "그늘 밝기" }, "Render", "ShadeBrightness", Settings::shadeBrightness, 0.0f, 1.0f, "%.2f",
					{ "Brightness of fully shaded fog.", "햇빛이 완전히 가려진 안개의 밝기." });
				Float({ "Glow strength", "햇빛 번짐 세기" }, "Render", "GlowStrength", Settings::glowStrength, 0.0f, 2.0f, "%.2f",
					{ "Glow of the fog toward the sun.", "해 쪽을 볼 때 안개가 빛나는 정도." });
				Float({ "Glow spread", "햇빛 번짐 집중도" }, "Render", "GlowAnisotropy", Settings::glowAnisotropy, 0.0f, 0.95f, "%.2f",
					{ "How tightly the glow gathers around the sun (higher is tighter).", "햇빛 번짐이 해 둘레에 얼마나 모이는지 (높을수록 좁게)." });
				Int({ "Local lights", "주변 광원 수" }, "Render", "LocalLights", Settings::localLights, 0, 16,
					{ "Torches and fires that light the fog (0 = off).", "안개를 비추는 횃불과 불의 수 (0 = 끔)." });
				Float({ "Local light strength", "광원 세기" }, "Render", "LocalLightStrength", Settings::localLightStrength, 0.0f, 4.0f, "%.2f",
					{ "Brightness of the glow around lights (weaker by day).", "광원 둘레 빛의 밝기 (낮에는 약해짐)." });
				Float({ "Local light reach", "광원 범위" }, "Render", "LocalLightReach", Settings::localLightReach, 0.25f, 3.0f, "%.2f",
					{ "How far the glow spreads, times each light's radius.", "빛이 퍼지는 거리 (각 광원 반경의 배수)." });
			}
			if (Header({ "Debug", "디버그" }, "HeaderDebug")) {
				Choice({ "Language", "언어" }, "General", "MenuLanguage", Settings::menuLanguage, { "English\0한국어\0\0", "English\0한국어\0\0" });
				Choice({ "Debug view", "디버그 화면" }, "General", "DebugView", Settings::debugView,
					{ "Off\0Near fog in magenta\0Full resolution, 64 steps\0Full resolution, raymarch steps\0Sun shadow map\0\0",
						"끔\0가까운 안개를 자홍색으로\0전체 해상도, 64단계\0전체 해상도, 레이마치 단계 수\0햇빛 그림자 맵\0\0" });
				Int({ "Shadow matrix mode", "그림자 행렬 방식" }, "General", "ShadowMatrixMode", Settings::shadowMatrixMode, 0, 3,
					{ "Diagnosis for the sun shadow map view (Debug view 4): 0 = Community Shaders' convention.",
						"햇빛 그림자 맵 화면(디버그 화면 4) 진단용. 0 = Community Shaders 방식." });
				Flag({ "Profiler", "성능 기록" }, "General", "EnableProfiler", Settings::profiler,
					{ "Log GPU and CPU times every 10 s.", "10초마다 GPU와 CPU 시간을 로그에 남깁니다." });
			}
			// While a control is held, the ini is not re-read: a reload set off by the last save would put the old value
			// back into the one being edited (and could be what its release saves; Codex review 2026-11-17). The hold
			// lapses on its own when the page stops drawing (the menu closed with a control held).
			if (g_anyActive) {
				Settings::HoldReload();
			}
		}
	}

	void Register(bool a_running)
	{
		g_running = a_running;
		if (!SKSEMenuFramework::IsInstalled() || !GetMenuFrameworkModule()) {
			logger::info("[Menu] SKSE Menu Framework not found: no settings page (edit GroundFog.ini instead)");
			return;
		}
		SKSEMenuFramework::SetSection("GroundFog");
		SKSEMenuFramework::AddSectionItem("Settings", Render);
		logger::info("[Menu] settings page registered with SKSE Menu Framework");
	}
}
