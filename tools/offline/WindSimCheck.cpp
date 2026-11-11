// Offline checks for the wind + fog grid (WindSim) on WARP. Same shaders as the plugin.
//   1. rest: no input for 300 steps keeps density at 1 and air still, no NaN
//   2. blast: a blast opens a hole in the fog that recovers afterwards
//   3. walls: a cliff stops the flow and keeps a fog hole from crossing it (control run without the cliff)

#include "WindSim.h"

#include <DirectXPackedVector.h>
#include <d3d11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{
	int g_failures = 0;

	void Check(bool a_ok, const char* a_name, const std::string& a_detail)
	{
		std::printf("%s %s : %s\n", a_ok ? "PASS" : "FAIL", a_name, a_detail.c_str());
		if (!a_ok) {
			++g_failures;
		}
	}

	template <class T>
	void SafeRelease(T*& a_p)
	{
		if (a_p) {
			a_p->Release();
			a_p = nullptr;
		}
	}

	constexpr int   kRes = 64;
	constexpr float kCell = 16.0f;
	constexpr float kDt = 1.0f / 30.0f;
	constexpr int   kHeightSize = 17;
	constexpr float kHeightCell = kRes * kCell / (kHeightSize - 1);

	float Half(std::uint16_t a_h) { return DirectX::PackedVector::XMConvertHalfToFloat(a_h); }

	// Reads one channel of a 32-bit float texture with a_channels channels.
	std::vector<float> Read32(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11Texture2D* a_tex, int a_channels, int a_channel)
	{
		D3D11_TEXTURE2D_DESC desc{};
		a_tex->GetDesc(&desc);
		desc.Usage = D3D11_USAGE_STAGING;
		desc.BindFlags = 0;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ID3D11Texture2D* staging = nullptr;
		std::vector<float> out(kRes * kRes, NAN);
		if (FAILED(a_dev->CreateTexture2D(&desc, nullptr, &staging))) {
			return out;
		}
		a_ctx->CopyResource(staging, a_tex);
		D3D11_MAPPED_SUBRESOURCE m{};
		if (SUCCEEDED(a_ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m))) {
			for (int y = 0; y < kRes; ++y) {
				const auto* row = reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(m.pData) + y * m.RowPitch);
				for (int x = 0; x < kRes; ++x) {
					out[y * kRes + x] = row[x * a_channels + a_channel];
				}
			}
			a_ctx->Unmap(staging, 0);
		}
		if (staging) {
			staging->Release();
		}
		return out;
	}

	// Reads channel a_channel of a 16-bit float texture with a_channels channels.
	std::vector<float> Read(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11Texture2D* a_tex, int a_channels, int a_channel)
	{
		D3D11_TEXTURE2D_DESC desc{};
		a_tex->GetDesc(&desc);
		desc.Usage = D3D11_USAGE_STAGING;
		desc.BindFlags = 0;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ID3D11Texture2D* staging = nullptr;
		std::vector<float> out(kRes * kRes, NAN);
		if (FAILED(a_dev->CreateTexture2D(&desc, nullptr, &staging))) {
			return out;
		}
		a_ctx->CopyResource(staging, a_tex);
		D3D11_MAPPED_SUBRESOURCE m{};
		if (SUCCEEDED(a_ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m))) {
			for (int y = 0; y < kRes; ++y) {
				const auto* row = reinterpret_cast<const std::uint16_t*>(static_cast<const std::uint8_t*>(m.pData) + y * m.RowPitch);
				for (int x = 0; x < kRes; ++x) {
					out[y * kRes + x] = Half(row[x * a_channels + a_channel]);
				}
			}
			a_ctx->Unmap(staging, 0);
		}
		SafeRelease(staging);
		return out;
	}

	WindShaders::SimCB BaseCB()
	{
		WindShaders::SimCB cb{};
		cb.grid[0] = 0.0f;
		cb.grid[1] = 0.0f;
		cb.grid[2] = kCell;
		cb.grid[3] = kRes;
		cb.time[0] = kDt;
		cb.wind[2] = 0.5f;   // relax 1/s
		cb.wind[3] = 0.0f;   // no vorticity confinement in the checks (deterministic)
		cb.gust[1] = 600.0f;
		cb.gust[2] = 0.1f;   // fog refill
		cb.gust[3] = 0.2f;   // velocity dissipation
		cb.shift[3] = kHeightSize;
		cb.counts[2] = 1.2f; // wall rise threshold
		cb.counts[3] = 0.0f; // uniform equilibrium (no wisps) unless a check sets it
		cb.heightGrid[0] = 0.0f;
		cb.heightGrid[1] = 0.0f;
		cb.heightGrid[2] = kHeightCell;
		cb.heightGrid[3] = 1.0f;
		cb.dragParams[0] = 100.0f;
		cb.dragParams[1] = 4.0f;
		return cb;
	}

	float At(const std::vector<float>& a_v, int a_x, int a_y) { return a_v[a_y * kRes + a_x]; }

	// The plugin's impact pulse (WindField.cpp Pulse) with its default settings.
	constexpr float kBlastDepth = 4.0f;
	constexpr float kBlastDuration = 0.6f;
	constexpr float kBlastPush = 150.0f;
	constexpr int   kBlastSteps = 18;   // kBlastDuration / kDt

	float Pulse(float a_age)
	{
		return a_age < 0.0f || a_age > kBlastDuration ? 0.0f : (1.0f - std::cos(6.2831853f * a_age / kBlastDuration)) / kBlastDuration;
	}

	// One impact at (a_x, a_y) of radius a_radius, a_age seconds into its pulse, pushing along +x.
	void SetBlast(WindShaders::SimCB& a_cb, float a_x, float a_y, float a_radius, float a_age)
	{
		a_cb.counts[1] = 1.0f;
		a_cb.blasts[0][0] = a_x;
		a_cb.blasts[0][1] = a_y;
		a_cb.blasts[0][2] = 1.0f / (a_radius * a_radius);
		a_cb.blasts[0][3] = kBlastDepth * Pulse(a_age);
		a_cb.blastKicks[0][0] = kBlastPush * Pulse(a_age);
	}

	bool AllFinite(const std::vector<float>& a_v)
	{
		return std::all_of(a_v.begin(), a_v.end(), [](float a_f) { return std::isfinite(a_f); });
	}

	ID3D11ShaderResourceView* MakeHeights(ID3D11Device* a_dev, bool a_cliff, ID3D11Texture2D*& a_tex)
	{
		std::vector<float> h(kHeightSize * kHeightSize, 0.0f);
		if (a_cliff) {
			for (int y = 0; y < kHeightSize; ++y) {
				for (int x = 9; x < kHeightSize; ++x) {
					h[y * kHeightSize + x] = 2000.0f;
				}
			}
		}
		D3D11_TEXTURE2D_DESC d{};
		d.Width = d.Height = kHeightSize;
		d.MipLevels = d.ArraySize = 1;
		d.Format = DXGI_FORMAT_R32_FLOAT;
		d.SampleDesc.Count = 1;
		d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA init{ h.data(), kHeightSize * sizeof(float), 0 };
		ID3D11ShaderResourceView* srv = nullptr;
		if (SUCCEEDED(a_dev->CreateTexture2D(&d, &init, &a_tex))) {
			a_dev->CreateShaderResourceView(a_tex, nullptr, &srv);
		}
		return srv;
	}

	void TestRest(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_flat)
	{
		a_sim.Reset(a_ctx);
		auto cb = BaseCB();
		for (int i = 0; i < 300; ++i) {
			cb.time[1] = i * kDt;
			a_sim.Step(a_ctx, cb, a_flat, 20);
		}
		const auto density = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
		const auto vx = Read(a_dev, a_ctx, a_sim.VelTexture(), 2, 0);
		double sum = 0.0;
		float  maxV = 0.0f;
		for (float d : density) sum += d;
		for (float v : vx) maxV = std::max(maxV, std::fabs(v));
		const double mean = sum / density.size();
		Check(AllFinite(density) && AllFinite(vx) && std::fabs(mean - 1.0) < 0.05 && maxV < 1.0f, "wind rest",
			"mean density " + std::to_string(mean) + ", max |vx| " + std::to_string(maxV));
	}

	void TestBlast(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_flat)
	{
		a_sim.Reset(a_ctx);
		auto        cb = BaseCB();
		const float cx = kRes * kCell * 0.5f;
		cb.gust[2] = 0.0f;   // no refill during the pulse: the fog is moved, not removed
		for (int i = 0; i < kBlastSteps; ++i) {
			SetBlast(cb, cx, cx, 4.0f * kCell, (i + 0.5f) * kDt);
			a_sim.Step(a_ctx, cb, a_flat, 20);
		}
		const auto  during = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
		const float holeDensity = At(during, kRes / 2, kRes / 2);
		double      sum = 0.0;
		float       lowest = 10.0f, highest = 0.0f;
		for (float d : during) {
			sum += d;
			lowest = std::min(lowest, d);
			highest = std::max(highest, d);
		}
		const double mean = sum / during.size();
		cb.counts[1] = 0.0f;
		cb.gust[2] = 0.1f;
		for (int i = 0; i < 900; ++i) {
			a_sim.Step(a_ctx, cb, a_flat, 20);
		}
		const auto  after = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
		const float recovered = At(after, kRes / 2, kRes / 2);
		// exp(-BlastDepth) = 0.018 at the centre if the projected divergence equalled the source exactly.
		Check(AllFinite(during) && holeDensity < 0.3f, "blast opens a hole", "fog at the centre " + std::to_string(holeDensity));
		Check(lowest >= 0.0f && highest <= WindShaders::kMaxFogCpp + 1e-4f, "fog amount stays in range", "lowest " + std::to_string(lowest) + ", highest " + std::to_string(highest));
		// The source's integral is zero: what leaves the centre piles on the rim. Semi-Lagrangian transport is not
		// exactly conservative, so allow a few percent.
		Check(std::fabs(mean - 1.0) < 0.03, "blast moves the fog, does not remove it", "mean fog over the grid " + std::to_string(mean));
		// Refill toward 1 at 0.1/s: 30 s leaves e^-3 = 5 % of a hole at most.
		Check(AllFinite(after) && recovered > 0.9f, "hole recovers", "fog at the centre after 30 s " + std::to_string(recovered));
	}

	// One actor crosses the grid along +x through the centre. Returns the density at the centre texel when the
	// actor has reached the far end, and (a_after) the density there a_restSeconds later.
	float RunActorCase(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_flat, float a_speed,
		float a_restSeconds, float& a_after)
	{
		constexpr float kHalfSqrtPi = 0.88622693f;
		constexpr float kPush = 0.35f;       // Settings::actorPush default
		constexpr float kMinSpeed = 220.0f;  // Settings::actorPushMinSpeed default
		a_sim.Reset(a_ctx);
		auto        cb = BaseCB();
		const float extent = kRes * kCell;
		const float y = extent * 0.5f + kCell * 0.5f;
		const float x0 = extent * 0.2f;
		const float x1 = extent * 0.8f;
		cb.dragParams[2] = kPush / (kHalfSqrtPi * cb.dragParams[0]);
		cb.dragParams[3] = kMinSpeed;
		cb.counts[0] = 1.0f;
		cb.drags[0][1] = y;
		cb.drags[0][2] = a_speed;
		const int steps = static_cast<int>((x1 - x0) / (a_speed * kDt));
		for (int i = 0; i < steps; ++i) {
			cb.drags[0][0] = x0 + a_speed * kDt * i;
			a_sim.Step(a_ctx, cb, a_flat, 20);
		}
		const auto  during = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
		const float onPath = At(during, kRes / 2, kRes / 2);
		cb.counts[0] = 0.0f;
		for (int i = 0; i < static_cast<int>(a_restSeconds / kDt); ++i) {
			a_sim.Step(a_ctx, cb, a_flat, 20);
		}
		const auto after = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
		a_after = AllFinite(during) && AllFinite(after) ? At(after, kRes / 2, kRes / 2) : -1.0f;
		return onPath;
	}

	void TestActorPush(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_flat)
	{
		float       runAfter = 0.0f;
		float       walkAfter = 0.0f;
		const float run = RunActorCase(a_dev, a_ctx, a_sim, a_flat, 370.0f, 30.0f, runAfter);
		const float walk = RunActorCase(a_dev, a_ctx, a_sim, a_flat, 100.0f, 0.0f, walkAfter);
		// exp(-0.35) = 0.70 on the path; the band allows for the refill and spreading since the actor passed.
		Check(run > 0.55f && run < 0.9f, "runner thins the fog on the path", "density at the centre " + std::to_string(run));
		Check(walkAfter >= 0.0f && walk > 0.97f, "walker leaves the fog alone", "density at the centre " + std::to_string(walk));
		Check(runAfter > 0.95f, "runner's path fills back in", "density at the centre after 30 s " + std::to_string(runAfter));
	}

	// Wind blows +x toward a cliff at x ~ 34 texels. A blast hole is opened upwind; with the cliff the hole must
	// not appear behind it, without the cliff (control) it must.
	float RunWallCase(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_heights,
		float& a_behindSpeed, float& a_wallSpeed)
	{
		a_sim.Reset(a_ctx);
		auto cb = BaseCB();
		cb.wind[0] = 300.0f;
		cb.wind[2] = 2.0f;
		for (int i = 0; i < 90; ++i) {   // let the flow settle
			a_sim.Step(a_ctx, cb, a_heights, 30);
		}
		// Track the lowest density behind the cliff line over time: the hole travels with the wind and can
		// leave the window before the end.
		float minBehindOverTime = 10.0f;
		for (int i = 0; i < 90; ++i) {
			SetBlast(cb, 20.0f * kCell, kRes * kCell * 0.5f, 3.0f * kCell, (i + 0.5f) * kDt);   // rate 0 after the pulse
			a_sim.Step(a_ctx, cb, a_heights, 30);
			if (i % 5 == 4) {
				const auto d = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
				for (int y = 8; y < kRes - 8; ++y) {
					for (int x = 40; x < kRes - 2; ++x) {
						minBehindOverTime = std::min(minBehindOverTime, At(d, x, y));
					}
				}
			}
		}
		const auto density = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
		const auto vx = Read(a_dev, a_ctx, a_sim.VelTexture(), 2, 0);
		const auto vy = Read(a_dev, a_ctx, a_sim.VelTexture(), 2, 1);
		const auto walls = Read(a_dev, a_ctx, a_sim.WallTexture(), 1, 0);
		float      maxWallSpeed = 0.0f;
		for (int i = 0; i < kRes * kRes; ++i) {
			if (walls[i] > 0.5f) {
				maxWallSpeed = std::max(maxWallSpeed, std::hypot(vx[i], vy[i]));
			}
		}
		a_wallSpeed = maxWallSpeed;
		float      minBehind = 10.0f;
		double     speed = 0.0;
		int        n = 0;
		for (int y = 8; y < kRes - 8; ++y) {
			for (int x = 40; x < 56; ++x) {
				minBehind = std::min(minBehind, At(density, x, y));
				speed += std::fabs(At(vx, x, y));
				++n;
			}
		}
		a_behindSpeed = static_cast<float>(speed / n);
		(void)minBehind;
		return minBehindOverTime;
	}

	void TestWalls(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_flat,
		ID3D11ShaderResourceView* a_cliff)
	{
		float       speedOpen = 0.0f;
		float       speedWall = 0.0f;
		float       wallSpeedOpen = 0.0f;
		float       wallSpeedCliff = 0.0f;
		const float holeOpen = RunWallCase(a_dev, a_ctx, a_sim, a_flat, speedOpen, wallSpeedOpen);
		const auto  wall = Read(a_dev, a_ctx, a_sim.WallTexture(), 1, 0);
		const float wallsOpen = static_cast<float>(std::count_if(wall.begin(), wall.end(), [](float a_w) { return a_w > 0.5f; }));
		const float holeWall = RunWallCase(a_dev, a_ctx, a_sim, a_cliff, speedWall, wallSpeedCliff);
		const auto  wall2 = Read(a_dev, a_ctx, a_sim.WallTexture(), 1, 0);
		const float wallsCliff = static_cast<float>(std::count_if(wall2.begin(), wall2.end(), [](float a_w) { return a_w > 0.5f; }));
		Check(wallsOpen == 0.0f && wallsCliff >= kRes, "cliff makes walls", "wall texels flat " + std::to_string(wallsOpen) + ", cliff " + std::to_string(wallsCliff));
		Check(holeOpen < 0.85f, "control: hole crosses without a cliff", "min density behind " + std::to_string(holeOpen) + ", speed " + std::to_string(speedOpen));
		// Air above the plateau keeps blowing (the base wind acts everywhere), so speed behind the cliff is not the
		// test; the test is no flow inside the wall and no fog hole crossing it.
		Check(holeWall > 0.95f && wallSpeedCliff < 1e-3f, "cliff blocks flow and fog",
			"min density behind " + std::to_string(holeWall) + ", max speed in wall texels " + std::to_string(wallSpeedCliff) +
				", speed behind " + std::to_string(speedWall) + " vs " + std::to_string(speedOpen) + " without cliff");
	}

	void TestShiftDirection(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_flat)
	{
		// Mark the field with a blast hole, then shift by (+3, 0): new(x, y) must equal old(x + 3, y).
		a_sim.Reset(a_ctx);
		auto cb = BaseCB();
		for (int i = 0; i < 10; ++i) {
			SetBlast(cb, 30.0f * kCell, 20.0f * kCell, 3.0f * kCell, (i + 0.5f) * kDt);
			a_sim.Step(a_ctx, cb, a_flat, 20);
		}
		const auto before = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
		a_sim.Shift(a_ctx, 3, 0);
		const auto after = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
		float maxErr = 0.0f;
		for (int y = 0; y < kRes; ++y) {
			for (int x = 0; x + 3 < kRes; ++x) {
				maxErr = std::max(maxErr, std::fabs(At(after, x, y) - At(before, x + 3, y)));
			}
		}
		float exposed = 0.0f;
		for (int y = 0; y < kRes; ++y) {
			exposed = std::max(exposed, std::fabs(At(after, kRes - 1, y) - 1.0f));
		}
		Check(maxErr < 1e-5f && exposed < 1e-5f, "shift direction", "max |new(x) - old(x+3)| " + std::to_string(maxErr) + ", exposed strip off rest by " + std::to_string(exposed));
	}

	void TestGustsDivergenceFree(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_flat)
	{
		a_sim.Reset(a_ctx);
		auto cb = BaseCB();
		cb.gust[0] = 200.0f;
		cb.wind[2] = 2.0f;
		for (int i = 0; i < 60; ++i) {
			cb.time[1] = i * kDt;
			a_sim.Step(a_ctx, cb, a_flat, 30);
		}
		const auto vx = Read(a_dev, a_ctx, a_sim.VelTexture(), 2, 0);
		const auto vy = Read(a_dev, a_ctx, a_sim.VelTexture(), 2, 1);
		double divSum = 0.0, speedSum = 0.0;
		int    n = 0;
		for (int y = 2; y < kRes - 2; ++y) {
			for (int x = 2; x < kRes - 2; ++x) {
				const float div = (At(vx, x + 1, y) - At(vx, x - 1, y) + At(vy, x, y + 1) - At(vy, x, y - 1)) * 0.5f;  // per texel
				divSum += std::fabs(div);
				speedSum += std::hypot(At(vx, x, y), At(vy, x, y));
				++n;
			}
		}
		const double meanSpeed = speedSum / n;
		const double ratio = meanSpeed > 0.0 ? (divSum / n) / meanSpeed : 1.0;
		Check(meanSpeed > 5.0 && ratio < 0.05, "gusts are divergence-free", "mean speed " + std::to_string(meanSpeed) + ", mean |div| per texel / speed " + std::to_string(ratio));
	}

	void TestLongRunStable(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_cliff)
	{
		// Shipping defaults: vorticity 2, gusts 0.6 x wind + 40, wind 300, walls on, 50 s.
		a_sim.Reset(a_ctx);
		auto cb = BaseCB();
		cb.wind[0] = 300.0f;
		cb.wind[3] = 2.0f;
		cb.gust[0] = 0.6f * 300.0f + 40.0f;
		cb.gust[1] = 1200.0f;
		for (int i = 0; i < 1500; ++i) {
			cb.time[1] = i * kDt;
			cb.time[2] += 300.0f * kDt;
			a_sim.Step(a_ctx, cb, a_cliff, 20);
		}
		const auto vx = Read(a_dev, a_ctx, a_sim.VelTexture(), 2, 0);
		const auto vy = Read(a_dev, a_ctx, a_sim.VelTexture(), 2, 1);
		const auto d = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
		float maxV = 0.0f;
		for (int i = 0; i < kRes * kRes; ++i) {
			maxV = std::max(maxV, std::hypot(vx[i], vy[i]));
		}
		Check(AllFinite(vx) && AllFinite(vy) && AllFinite(d) && maxV < 3000.0f, "long run stays bounded", "max speed " + std::to_string(maxV) + " after 50 s with defaults");
	}

	void TestBlastOutward(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_flat)
	{
		a_sim.Reset(a_ctx);
		auto        cb = BaseCB();
		const float c = kRes * kCell * 0.5f;
		const float R = 4.0f * kCell;
		SetBlast(cb, c, c, R, 0.5f * kBlastDuration);   // the pulse's peak
		cb.blastKicks[0][0] = 0.0f;                     // no direction: the radial part alone
		for (int i = 0; i < 3; ++i) {
			a_sim.Step(a_ctx, cb, a_flat, 30);
		}
		const auto vx = Read(a_dev, a_ctx, a_sim.VelTexture(), 2, 0);
		const auto vy = Read(a_dev, a_ctx, a_sim.VelTexture(), 2, 1);
		// How well the projection reproduces the wanted divergence at the centre (Jacobi from zero, 30 iterations).
		{
			const int   m = kRes / 2;
			const float div = (At(vx, m + 1, m) - At(vx, m - 1, m) + At(vy, m, m + 1) - At(vy, m, m - 1)) / (2.0f * kCell);
			const float wanted = cb.blasts[0][3];   // Ricker = 1 at the centre
			Check(div > 0.5f * wanted && div < 1.5f * wanted, "projected divergence follows the source",
				"at the centre " + std::to_string(div) + " 1/s, source " + std::to_string(wanted) + " 1/s");
		}
		double radial = 0.0;
		int    n = 0;
		for (int k = 0; k < 16; ++k) {
			const float a = k * 0.3927f;
			const float wx = c + std::cos(a) * R;
			const float wy = c + std::sin(a) * R;
			const int   x = static_cast<int>(wx / kCell);
			const int   y = static_cast<int>(wy / kCell);
			radial += At(vx, x, y) * std::cos(a) + At(vy, x, y) * std::sin(a);
			++n;
		}
		Check(radial / n > 5.0, "blast pushes air outward", "mean radial speed at r = R " + std::to_string(radial / n));
	}

	// The equilibrium pattern rides the base wind. In a steady wind with no gusts the air never departs from the
	// base wind, so the carried departure delta must stay 0 (the pattern is not blurred by the transport). With
	// gusts the pattern is dragged around: delta must appear.
	void TestPatternRidesTheWind(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_flat)
	{
		auto run = [&](float a_gust) {
			a_sim.Reset(a_ctx);
			auto cb = BaseCB();
			cb.wind[0] = 200.0f;
			cb.wind[2] = 2.0f;
			cb.gust[0] = a_gust;
			cb.counts[3] = 0.5f;
			for (int i = 0; i < 1200; ++i) {   // 40 s: what the start-up (air at rest, wind picking up) left has refilled
				cb.time[1] = i * kDt;
				cb.time[2] += 200.0f * kDt;
				a_sim.Step(a_ctx, cb, a_flat, 20);
			}
			const auto delta = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 1);
			const auto rho = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
			float      worst = 0.0f, lo = 10.0f, hi = 0.0f;
			for (int y = 4; y < kRes - 4; ++y) {
				for (int x = 4; x < kRes - 4; ++x) {
					worst = std::max(worst, std::fabs(At(delta, x, y)));
					lo = std::min(lo, At(rho, x, y));
					hi = std::max(hi, At(rho, x, y));
				}
			}
			return std::array<float, 3>{ worst, lo, hi };
		};
		const auto steady = run(0.0f);
		const auto gusty = run(200.0f);
		// Not exactly 0: the pattern also changes slowly in time (EqFog's time term), which the transport leaves to
		// the refill; that lag is about (rate of change) / FogRefill, measured 0.015.
		Check(steady[0] < 0.03f && steady[2] - steady[1] > 0.3f, "pattern rides a steady wind unchanged",
			"largest |delta| " + std::to_string(steady[0]) + ", fog amount " + std::to_string(steady[1]) + ".." + std::to_string(steady[2]));
		Check(gusty[0] > 0.02f && gusty[1] >= 0.0f && gusty[2] <= WindShaders::kMaxFogCpp + 1e-4f, "gusts drag the pattern",
			"largest |delta| " + std::to_string(gusty[0]) + ", fog amount " + std::to_string(gusty[1]) + ".." + std::to_string(gusty[2]));
	}

	// Impacts repeated in a corner, where the grid edge cuts the source's inflow ring: with the pressure solved
	// from zero each frame nothing may build up.
	void TestCornerImpactsBounded(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_flat)
	{
		a_sim.Reset(a_ctx);
		auto cb = BaseCB();
		for (int i = 0; i < 600; ++i) {
			SetBlast(cb, 2.0f * kCell, 2.0f * kCell, 4.0f * kCell, std::fmod((i + 0.5f) * kDt, kBlastDuration));
			a_sim.Step(a_ctx, cb, a_flat, 20);
		}
		const auto vx = Read(a_dev, a_ctx, a_sim.VelTexture(), 2, 0);
		const auto vy = Read(a_dev, a_ctx, a_sim.VelTexture(), 2, 1);
		const auto d = Read32(a_dev, a_ctx, a_sim.FogTexture(), 2, 0);
		float      maxV = 0.0f;
		for (int i = 0; i < kRes * kRes; ++i) {
			maxV = std::max(maxV, std::hypot(vx[i], vy[i]));
		}
		// 20 s of impacts blow the thinned fog across the grid, so the far corner need not be at 1; the check is
		// that speed and fog stay bounded.
		const auto [lo, hi] = std::minmax_element(d.begin(), d.end());
		Check(AllFinite(vx) && AllFinite(vy) && AllFinite(d) && maxV < 3000.0f && *lo >= 0.0f && *hi <= WindShaders::kMaxFogCpp + 1e-4f,
			"repeated corner impacts stay bounded",
			"max speed " + std::to_string(maxV) + " after 20 s, fog " + std::to_string(*lo) + ".." + std::to_string(*hi) + ", at the far corner " +
				std::to_string(At(d, kRes - 8, kRes - 8)));
	}

	void TestComputeStateRestored(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, WindSim& a_sim, ID3D11ShaderResourceView* a_flat)
	{
		// Bind a stand-in SRV and UAV the "game" left on the compute stage; Step must leave them there.
		D3D11_TEXTURE2D_DESC d{};
		d.Width = d.Height = 4;
		d.MipLevels = d.ArraySize = 1;
		d.Format = DXGI_FORMAT_R32_FLOAT;
		d.SampleDesc.Count = 1;
		d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
		ID3D11Texture2D*           t1 = nullptr;
		ID3D11Texture2D*           t2 = nullptr;
		ID3D11ShaderResourceView*  srv = nullptr;
		ID3D11UnorderedAccessView* uav = nullptr;
		a_dev->CreateTexture2D(&d, nullptr, &t1);
		a_dev->CreateTexture2D(&d, nullptr, &t2);
		a_dev->CreateShaderResourceView(t1, nullptr, &srv);
		a_dev->CreateUnorderedAccessView(t2, nullptr, &uav);
		a_ctx->CSSetShaderResources(3, 1, &srv);
		a_ctx->CSSetUnorderedAccessViews(1, 1, &uav, nullptr);
		a_sim.Step(a_ctx, BaseCB(), a_flat, 5);
		ID3D11ShaderResourceView*  srvAfter = nullptr;
		ID3D11UnorderedAccessView* uavAfter = nullptr;
		a_ctx->CSGetShaderResources(3, 1, &srvAfter);
		a_ctx->CSGetUnorderedAccessViews(1, 1, &uavAfter);
		const bool same = srvAfter == srv && uavAfter == uav;
		Check(same, "compute state restored", same ? "CS t3 and u1 unchanged across Step" : "CS bindings changed");
		ID3D11ShaderResourceView*  none = nullptr;
		ID3D11UnorderedAccessView* noneU = nullptr;
		a_ctx->CSSetShaderResources(3, 1, &none);
		a_ctx->CSSetUnorderedAccessViews(1, 1, &noneU, nullptr);
		SafeRelease(srvAfter);
		SafeRelease(uavAfter);
		SafeRelease(srv);
		SafeRelease(uav);
		SafeRelease(t1);
		SafeRelease(t2);
	}
}

int RunWindSimChecks()
{
	ID3D11Device*        dev = nullptr;
	ID3D11DeviceContext* ctx = nullptr;
	if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) {
		Check(false, "wind WARP device", "D3D11CreateDevice failed");
		return g_failures;
	}
	WindSim     sim;
	std::string error;
	const bool  created = sim.Create(dev, kRes, error);
	Check(created, "wind compute shaders", created ? "9 kernels compiled" : error);
	if (created) {
		ID3D11Texture2D* flatTex = nullptr;
		ID3D11Texture2D* cliffTex = nullptr;
		auto*            flat = MakeHeights(dev, false, flatTex);
		auto*            cliff = MakeHeights(dev, true, cliffTex);
		TestRest(dev, ctx, sim, flat);
		TestBlast(dev, ctx, sim, flat);
		TestActorPush(dev, ctx, sim, flat);
		TestWalls(dev, ctx, sim, flat, cliff);
		TestShiftDirection(dev, ctx, sim, flat);
		TestGustsDivergenceFree(dev, ctx, sim, flat);
		TestLongRunStable(dev, ctx, sim, cliff);
		TestBlastOutward(dev, ctx, sim, flat);
		TestPatternRidesTheWind(dev, ctx, sim, flat);
		TestCornerImpactsBounded(dev, ctx, sim, flat);
		TestComputeStateRestored(dev, ctx, sim, flat);
		SafeRelease(flat);
		SafeRelease(cliff);
		SafeRelease(flatTex);
		SafeRelease(cliffTex);
	}
	sim.Release();
	SafeRelease(ctx);
	SafeRelease(dev);
	return g_failures;
}
