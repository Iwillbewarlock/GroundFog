#pragma once

// Text edit of one ini key, without the file or game types, so the offline check can test it (Settings::WriteValue uses it).

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace IniEdit
{
	inline std::string Lower(std::string a_text)
	{
		for (auto& c : a_text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		return a_text;
	}

	inline std::string Trim(const std::string& a_text)
	{
		constexpr const char* kSpace = " \t\r\n";
		const auto            first = a_text.find_first_not_of(kSpace);
		if (first == std::string::npos) return {};
		return a_text.substr(first, a_text.find_last_not_of(kSpace) - first + 1);
	}

	// a_text with [a_section] a_key set to a_value: the key's line is rewritten in place (its name as written and any
	// comment after the value are kept, every other line untouched, line ends as the file has them: CRLF when it has
	// any), every such line when the key is there more than once (the reader keeps the last; Codex review 2026-11-17);
	// a missing key goes right after the section's last key (or its header), a missing section at the end.
	// Sections and keys match without regard to case, as the reader (Settings.cpp ParseFile) does.
	inline std::string SetValue(const std::string& a_text, const std::string& a_section, const std::string& a_key, const std::string& a_value)
	{
		const std::string        nl = a_text.find("\r\n") != std::string::npos ? "\r\n" : "\n";
		std::vector<std::string> lines;
		for (std::size_t start = 0; start < a_text.size();) {
			const auto  end = a_text.find('\n', start);
			std::string line = a_text.substr(start, end == std::string::npos ? std::string::npos : end - start);
			if (!line.empty() && line.back() == '\r') line.pop_back();
			lines.push_back(line);
			if (end == std::string::npos) break;
			start = end + 1;
		}
		const std::string want = Lower(Trim(a_section)), key = Lower(Trim(a_key));
		std::string       section;
		long long         insertAt = -1;   // after the wanted section's header or its last key
		bool              done = false;
		for (std::size_t i = 0; i < lines.size(); ++i) {
			std::string bare = lines[i];
			const auto  cut = bare.find_first_of(";#");
			std::string comment = cut == std::string::npos ? std::string{} : bare.substr(cut);
			if (cut != std::string::npos) bare.resize(cut);
			const std::string t = Trim(bare);
			if (!t.empty() && t.front() == '[' && t.back() == ']') {
				section = Lower(Trim(t.substr(1, t.size() - 2)));
				if (section == want) insertAt = static_cast<long long>(i) + 1;
				continue;
			}
			if (section != want) continue;
			const auto eq = t.find('=');
			if (eq == std::string::npos) continue;
			insertAt = static_cast<long long>(i) + 1;
			if (Lower(Trim(t.substr(0, eq))) != key) continue;
			lines[i] = Trim(t.substr(0, eq)) + " = " + a_value + (comment.empty() ? "" : " " + comment);
			done = true;
		}
		if (!done) {
			const std::string add = Trim(a_key) + " = " + a_value;
			if (insertAt >= 0) {
				lines.insert(lines.begin() + insertAt, add);
			} else {
				lines.push_back("[" + Trim(a_section) + "]");
				lines.push_back(add);
			}
		}
		std::string out;
		for (const auto& l : lines) out += l + nl;
		return out;
	}
}
