#pragma once

// EditorID of a form at runtime: the form's own (works when an EditorID mod patches it in), else po3 Tweaks' export
// GetFormEditorID(formID) (signature as powerof3's CLibUtil editorID.hpp), else empty.
inline std::string EditorIDOf(const RE::TESForm* a_form)
{
	if (!a_form) {
		return {};
	}
	if (const char* e = a_form->GetFormEditorID(); e && *e) {
		return e;
	}
	using GetEditorID = const char* (*)(std::uint32_t);
	static const GetEditorID po3 = []() -> GetEditorID {
		const HMODULE tweaks = GetModuleHandleW(L"po3_Tweaks");
		return tweaks ? reinterpret_cast<GetEditorID>(GetProcAddress(tweaks, "GetFormEditorID")) : nullptr;
	}();
	if (po3) {
		if (const char* e = po3(a_form->GetFormID()); e && *e) {
			return e;
		}
	}
	return {};
}
