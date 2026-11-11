#pragma once

// D3DCompile with a cache that a background thread can fill while the game loads, so the first outdoor frame only
// creates the shaders from compiled code. No game dependency (the offline checks use it too and simply never start
// the background job). D3DCompile needs no device and is safe off the render thread.

#include <d3dcompiler.h>

#include <chrono>
#include <cstddef>
#include <future>
#include <map>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace ShaderCache
{
	struct Job
	{
		std::string source;
		std::string entry;
		std::string target;
		std::string name;
	};

	namespace detail
	{
		struct State
		{
			std::mutex                       lock;
			std::map<std::string, ID3DBlob*> blobs;   // lives as long as the process (nothing to release at exit)
			std::shared_future<void>         job;
			double                           ms{ 0.0 };
			std::size_t                      jobs{ 0 };
		};

		// Never destroyed: a static's destructor would wait for the job or release blobs after d3dcompiler unloaded.
		inline State& Get()
		{
			static State* const s = new State;
			return *s;
		}

		inline std::string Key(const std::string& a_source, const char* a_entry, const char* a_target)
		{
			return std::string(a_entry) + '|' + a_target + '|' + a_source;
		}

		inline HRESULT Run(const std::string& a_source, const char* a_entry, const char* a_target, const char* a_name, ID3DBlob** a_code, ID3DBlob** a_errors)
		{
			return D3DCompile(a_source.data(), a_source.size(), a_name, nullptr, nullptr, a_entry, a_target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, a_code, a_errors);
		}
	}

	// Compiled code for this source, entry and target (the caller releases it). Waits for a running background job,
	// then takes its result; anything it did not produce (or that failed there) is compiled here, so errors are
	// reported by the caller as before.
	inline HRESULT Compile(const std::string& a_source, const char* a_entry, const char* a_target, const char* a_name, ID3DBlob** a_code, ID3DBlob** a_errors)
	{
		auto& s = detail::Get();
		std::shared_future<void> job;
		{
			const std::lock_guard guard(s.lock);
			job = s.job;
		}
		if (job.valid()) job.wait();
		{
			const std::lock_guard guard(s.lock);
			const auto            it = s.blobs.find(detail::Key(a_source, a_entry, a_target));
			if (it != s.blobs.end()) {
				it->second->AddRef();
				*a_code = it->second;
				if (a_errors) *a_errors = nullptr;
				return S_OK;
			}
		}
		return detail::Run(a_source, a_entry, a_target, a_name, a_code, a_errors);
	}

	// Starts compiling in the background (once). False if the thread could not start; the first use then compiles
	// on the render thread, as without the cache.
	inline bool StartPrecompile(std::vector<Job> a_jobs)
	{
		auto&                 s = detail::Get();
		const std::lock_guard guard(s.lock);
		if (s.job.valid()) return true;
		s.jobs = a_jobs.size();
		try {
			s.job = std::async(std::launch::async, [jobs = std::move(a_jobs)]() {
				auto&      state = detail::Get();
				const auto start = std::chrono::steady_clock::now();
				for (const auto& j : jobs) {
					ID3DBlob* code = nullptr;
					ID3DBlob* errors = nullptr;
					const HRESULT hr = detail::Run(j.source, j.entry.c_str(), j.target.c_str(), j.name.c_str(), &code, &errors);
					if (errors) errors->Release();
					if (FAILED(hr) || !code) {
						if (code) code->Release();
						continue;   // compiled again at first use, which logs the error
					}
					const std::lock_guard lockState(state.lock);
					state.blobs.emplace(detail::Key(j.source, j.entry.c_str(), j.target.c_str()), code);
				}
				const auto            end = std::chrono::steady_clock::now();
				const std::lock_guard lockState(state.lock);
				state.ms = std::chrono::duration<double, std::milli>(end - start).count();
			}).share();
		} catch (const std::system_error&) {
			return false;
		}
		return true;
	}

	// Background job's time (0 until it finished), and shaders cached / asked for.
	inline double PrecompileMs()
	{
		auto&                 s = detail::Get();
		const std::lock_guard guard(s.lock);
		return s.ms;
	}
	inline std::pair<std::size_t, std::size_t> PrecompileCount()
	{
		auto&                 s = detail::Get();
		const std::lock_guard guard(s.lock);
		return { s.blobs.size(), s.jobs };
	}
}
