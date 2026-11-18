// Offline measuring tool (build-wakeheading.cmd; DEBUG-LEDGER.md 2026-11-15): the profiler line "fog around player" (WindField.cpp, WindRules::SummarizeTrail) computed
// on the plugin's wind field on WARP, fed the way WindField::Update feeds it (frame dt clamped to 1/15, two steps above 1/30,
// the body's velocity, step and run from real time, wind vector, gusts, gust travel, the grid shifted with the player).
// usage: WakeHeading <cases.txt>     one case a line:
//   name res speed fps wind windRel heading e shift jitter mode follow feed [probe0 probeEvery gustMin vort relax startX startY cell
//   fogSampling seconds]   (fogSampling = [Wind] FogSampling of WindSim::Create; seconds = length of a game-mode run, default 8)
//   windRel: degrees from the heading to where the wind blows (0 tail, 90 cross, 180 head); heading: degrees on the grid
//   shift 0/1: recentre the grid like Terrain (player more than 256 from the centre, 64-unit cells)
//   jitter: frame times 1/fps x (1 +- jitter), the body moved by the mean frame (its measured speed is then noisy)
//   mode calib: RaymarchCheck RunWake's path (0.1 -> 0.55 of the field on a texel row), one read at the end
//   mode game: 8 s, probes at 3..8 s
#include <string>
#include <utility>
#include <vector>
#include "Settings.h"
#include "WindRules.h"
#include "WindSim.h"

#include <d3d11.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

namespace
{
	float kCell = 32.0f;   // per case (cell field)
	constexpr float kHCell = 64.0f;
	constexpr float kRecentre = 256.0f;

	struct Case
	{
		std::string name, mode;
		int         res{ 384 };
		float       speed{ 340 }, fps{ 15 }, wind{ 0 }, windRel{ 90 }, heading{ 45 }, e{ 0.7f }, jitter{ 0 }, follow{ 6 };
		int         shift{ 1 }, feed{ 1 };
		float       probe0{ 3.0f }, probeEvery{ 1.0f }, gustMin{ -1.0f }, vort{ -1.0f }, relax{ -1.0f }, startX{ 5.0f }, startY{ 11.0f }, cell{ 32.0f };
		int         adv{ 0 };   // [Wind] FogSampling
		float       seconds{ 8.0f };
	};

	struct Probe
	{
		float t{ 0 };
		WindRules::TrailSummary s;
		float c{ NAN };
		int   cn{ 0 };
		float trough{ NAN }, troughMin{ NAN }, troughOff{ NAN };
		float band[4]{ NAN, NAN, NAN, NAN };   // the profiler's band by distance: 2-3R, 3-4R, 4-5R, 5-6R
		float speed{ 0 };
		float u[6]{ NAN, NAN, NAN, NAN, NAN, NAN };   // air speed along the heading on the body line, bands 0-1R .. 5-6R behind
		float uAcross{ NAN };                          // mean across-heading air speed on the body line, 2..6R
		float prof[9]{ NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN };   // q by across offset -4..+4 cells (bins of one cell), 2..6R behind
		double sumDelta{ 0 };                          // sum of delta over the grid
		float  gridMin{ 10.0f }, gridMax{ -10.0f }, gridMean{ NAN };   // q over the whole grid
		float  maxB{ NAN }, maxC{ NAN }, minB{ NAN }, minC{ NAN };       // where: units behind the body, units to the left of its heading
	};

	float HalfToFloat(unsigned short a_h)
	{
		const unsigned sign = (a_h >> 15) & 1u, ex = (a_h >> 10) & 0x1fu, man = a_h & 0x3ffu;
		float v;
		if (ex == 0) v = std::ldexp(static_cast<float>(man), -24);
		else if (ex == 31) v = man ? NAN : INFINITY;
		else v = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(ex) - 25);
		return sign ? -v : v;
	}

	// R16G16_FLOAT velocity -> floats (x, y per texel)
	std::vector<float> ReadVel(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11Texture2D* a_tex, int a_res)
	{
		D3D11_TEXTURE2D_DESC desc{};
		a_tex->GetDesc(&desc);
		desc.Usage = D3D11_USAGE_STAGING;
		desc.BindFlags = 0;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ID3D11Texture2D* st = nullptr;
		a_dev->CreateTexture2D(&desc, nullptr, &st);
		a_ctx->CopyResource(st, a_tex);
		D3D11_MAPPED_SUBRESOURCE m{};
		std::vector<float>       vel(static_cast<std::size_t>(a_res) * a_res * 2, 0.0f);
		if (SUCCEEDED(a_ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
			for (int y = 0; y < a_res; ++y) {
				const unsigned short* row = reinterpret_cast<const unsigned short*>(static_cast<const char*>(m.pData) + y * m.RowPitch);
				for (int x = 0; x < a_res * 2; ++x) vel[static_cast<std::size_t>(y) * a_res * 2 + x] = HalfToFloat(row[x]);
			}
			a_ctx->Unmap(st, 0);
		}
		st->Release();
		return vel;
	}

	// The air on the body line, the cross profile of q and the grid sum of delta (added for the fogadv experiment).
	void MeasureExtra(Probe& a_p, const std::vector<float>& a_fog, const std::vector<float>& a_vel, int a_res, float a_ox, float a_oy, float a_bx, float a_by,
		float a_vx, float a_vy, float a_R)
	{
		const float len = std::max(std::hypot(a_vx, a_vy), 1e-6f);
		const float fx = a_vx / len, fy = a_vy / len;
		double uSum[6]{}, aSum = 0.0, pSum[9]{};
		int    uN[6]{}, aN = 0, pN[9]{};
		double qSum = 0.0;
		for (int y = 0; y < a_res; ++y) {
			for (int x = 0; x < a_res; ++x) {
				const std::size_t at = (static_cast<std::size_t>(y) * a_res + x) * 2;
				a_p.sumDelta += a_fog[at + 1];
				{
					const float eqAll = a_fog[at] - a_fog[at + 1];
					const float qAll = eqAll > 0.05f ? a_fog[at] / eqAll : 1.0f;
					const float ddx = a_ox + (x + 0.5f) * kCell - a_bx, ddy = a_oy + (y + 0.5f) * kCell - a_by;
					if (qAll < a_p.gridMin) {
						a_p.gridMin = qAll;
						a_p.minB = -(ddx * fx + ddy * fy);
						a_p.minC = -ddx * fy + ddy * fx;
					}
					if (qAll > a_p.gridMax) {
						a_p.gridMax = qAll;
						a_p.maxB = -(ddx * fx + ddy * fy);
						a_p.maxC = -ddx * fy + ddy * fx;
					}
					qSum += qAll;
				}
				const float dx = a_ox + (x + 0.5f) * kCell - a_bx, dy = a_oy + (y + 0.5f) * kCell - a_by;
				const float b = -(dx * fx + dy * fy);
				if (b < 0.0f || b > 6.0f * a_R) continue;
				const float cs = -dx * fy + dy * fx;
				if (std::fabs(cs) < 0.5f * kCell) {
					const int k = std::clamp(static_cast<int>(b / a_R), 0, 5);
					uSum[k] += a_vel[at] * fx + a_vel[at + 1] * fy;
					++uN[k];
					if (b >= 2.0f * a_R) {
						aSum += -a_vel[at] * fy + a_vel[at + 1] * fx;
						++aN;
					}
				}
				if (b >= 2.0f * a_R) {
					const int bin = static_cast<int>(std::floor(cs / kCell + 0.5f)) + 4;
					if (bin >= 0 && bin < 9) {
						const float eq = a_fog[at] - a_fog[at + 1];
						pSum[bin] += eq > 0.05f ? a_fog[at] / eq : 1.0f;
						++pN[bin];
					}
				}
			}
		}
		for (int k = 0; k < 6; ++k) a_p.u[k] = uN[k] ? static_cast<float>(uSum[k] / uN[k]) : NAN;
		a_p.uAcross = aN ? static_cast<float>(aSum / aN) : NAN;
		a_p.gridMean = static_cast<float>(qSum / (static_cast<double>(a_res) * a_res));
		for (int k = 0; k < 9; ++k) a_p.prof[k] = pN[k] ? static_cast<float>(pSum[k] / pN[k]) : NAN;
	}

	std::vector<float> ReadFog(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, ID3D11Texture2D* a_tex, int a_res)
	{
		D3D11_TEXTURE2D_DESC desc{};
		a_tex->GetDesc(&desc);
		desc.Usage = D3D11_USAGE_STAGING;
		desc.BindFlags = 0;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ID3D11Texture2D* st = nullptr;
		a_dev->CreateTexture2D(&desc, nullptr, &st);
		a_ctx->CopyResource(st, a_tex);
		D3D11_MAPPED_SUBRESOURCE m{};
		std::vector<float>       fog(static_cast<std::size_t>(a_res) * a_res * 2);
		if (SUCCEEDED(a_ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
			for (int y = 0; y < a_res; ++y) {
				std::memcpy(&fog[static_cast<std::size_t>(y) * a_res * 2], static_cast<const char*>(m.pData) + y * m.RowPitch, static_cast<std::size_t>(a_res) * 8);
			}
			a_ctx->Unmap(st, 0);
		}
		st->Release();
		return fog;
	}

	Probe Measure(const std::vector<float>& a_fog, int a_res, float a_ox, float a_oy, float a_bx, float a_by, float a_vx, float a_vy, float a_R)
	{
		Probe p;
		p.speed = std::hypot(a_vx, a_vy);
		p.s = WindRules::SummarizeTrail(
			a_res, a_res, a_ox + 0.5f * kCell, a_oy + 0.5f * kCell, kCell, a_bx, a_by, a_vx, a_vy, a_R,
			[&](int a_x, int a_y, float& a_rho, float& a_delta) {
				const float* f = &a_fog[(static_cast<std::size_t>(a_y) * a_res + a_x) * 2];
				a_rho = f[0];
				a_delta = f[1];
			},
			[](int, int) { return 0.0f; }, [](int, int) { return false; });
		const float len = std::max(p.speed, 1e-6f);
		const float fx = a_vx / len, fy = a_vy / len;
		constexpr int kBins = 9;   // 2R..6R in 32-unit bins (280 units)
		constexpr float kBin = 32.0f;
		float         binMin[kBins], binOff[kBins];
		bool          binSet[kBins]{};
		double        cSum = 0.0, bandSum[4]{};
		int           bandN[4]{};
		for (int y = 0; y < a_res; ++y) {
			for (int x = 0; x < a_res; ++x) {
				const float dx = a_ox + (x + 0.5f) * kCell - a_bx, dy = a_oy + (y + 0.5f) * kCell - a_by;
				const float b = -(dx * fx + dy * fy);
				if (b < 2.0f * a_R || b > 6.0f * a_R) continue;
				const float cs = -dx * fy + dy * fx;   // signed, to the left of the heading
				if (std::fabs(cs) > 3.0f * a_R) continue;
				const float* f = &a_fog[(static_cast<std::size_t>(y) * a_res + x) * 2];
				const float  eq = f[0] - f[1];
				const float  q = eq > 0.05f ? f[0] / eq : 1.0f;
				if (std::fabs(cs) < 0.5f * kCell) {
					cSum += q;
					++p.cn;
				}
				if (std::fabs(cs) <= 0.5f * a_R) {
					const int k = std::clamp(static_cast<int>((b - 2.0f * a_R) / a_R), 0, 3);
					bandSum[k] += q;
					++bandN[k];
				}
				const int bin = std::clamp(static_cast<int>((b - 2.0f * a_R) / kBin), 0, kBins - 1);
				if (!binSet[bin] || q < binMin[bin]) {
					binSet[bin] = true;
					binMin[bin] = q;
					binOff[bin] = cs;
				}
			}
		}
		p.c = p.cn ? static_cast<float>(cSum / p.cn) : NAN;
		for (int k = 0; k < 4; ++k) p.band[k] = bandN[k] ? static_cast<float>(bandSum[k] / bandN[k]) : NAN;
		double tSum = 0.0, oSum = 0.0;
		int    n = 0;
		float  tMin = 10.0f;
		for (int i = 0; i < kBins; ++i) {
			if (!binSet[i]) continue;
			tSum += binMin[i];
			oSum += binOff[i];
			tMin = std::min(tMin, binMin[i]);
			++n;
		}
		if (n) {
			p.trough = static_cast<float>(tSum / n);
			p.troughOff = static_cast<float>(oSum / n);
			p.troughMin = tMin;
		}
		return p;
	}

	int RunCase(ID3D11Device* a_dev, ID3D11DeviceContext* a_ctx, const Case& a_c)
	{
		const bool  calib = a_c.mode == "calib";
		const int   res = a_c.res;
		const float extent = res * kCell;
		const int   hSize = calib ? 33 : static_cast<int>(extent / kHCell) + 1;
		const float hCell = extent / (hSize - 1);
		WindSim     sim;
		std::string err;
		if (!sim.Create(a_dev, res, err, a_c.adv)) {
			std::printf("CASE %s: create failed: %s\n", a_c.name.c_str(), err.c_str());
			return 4;
		}
		std::vector<float>   h(static_cast<std::size_t>(hSize) * hSize, 0.0f);
		D3D11_TEXTURE2D_DESC d{};
		d.Width = d.Height = static_cast<UINT>(hSize);
		d.MipLevels = d.ArraySize = 1;
		d.Format = DXGI_FORMAT_R32_FLOAT;
		d.SampleDesc.Count = 1;
		d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		D3D11_SUBRESOURCE_DATA    init{ h.data(), static_cast<UINT>(hSize * sizeof(float)), 0 };
		ID3D11Texture2D*          ht = nullptr;
		ID3D11ShaderResourceView* hs = nullptr;
		a_dev->CreateTexture2D(&d, &init, &ht);
		a_dev->CreateShaderResourceView(ht, nullptr, &hs);

		const float R = WindRules::DragRadius(Settings::actorRadius, 0.0f);
		const float ha = a_c.heading * 0.01745329252f;
		const float hx = calib ? 1.0f : std::cos(ha), hy = calib ? 0.0f : std::sin(ha);
		const float wa = (a_c.heading + a_c.windRel) * 0.01745329252f;
		const float windX = a_c.wind * std::cos(wa), windY = a_c.wind * std::sin(wa);
		float       ox = 0.0f, oy = 0.0f;

		WindShaders::SimCB cb{};
		cb.grid[2] = kCell;
		cb.grid[3] = static_cast<float>(res);
		cb.wind[0] = windX;
		cb.wind[1] = windY;
		cb.wind[2] = a_c.relax >= 0.0f ? a_c.relax : Settings::windRelax;
		cb.wind[3] = a_c.vort >= 0.0f ? a_c.vort : Settings::vorticity;
		cb.gust[0] = Settings::gustAmount * a_c.wind + (a_c.gustMin >= 0.0f ? a_c.gustMin : Settings::gustMinimum);
		cb.gust[1] = Settings::gustWavelength;
		cb.gust[2] = Settings::fogRefill;
		cb.gust[3] = Settings::velocityDissipation;
		cb.shift[3] = static_cast<float>(hSize);
		cb.counts[2] = Settings::wallSlope;
		cb.counts[3] = Settings::wispAmount;
		cb.heightGrid[2] = hCell;
		cb.heightGrid[3] = 1.0f;
		cb.dragParams[0] = Settings::actorPushWalk;
		cb.dragParams[1] = a_c.follow;
		cb.dragParams[3] = Settings::actorPushMinSpeed;
		sim.Reset(a_ctx);

		std::mt19937                          rng(12345);   // fixed seed: the frame-time jitter
		std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
		const float                           base = 1.0f / a_c.fps;
		const std::vector<float> pat = a_c.jitter == -1.0f ? std::vector<float>{ 0.8f, 0.8f, 0.8f, 0.8f, 1.8f } :
		                               a_c.jitter == -2.0f ? std::vector<float>{ 0.7f, 0.7f, 1.6f } :
		                               a_c.jitter == -3.0f ? std::vector<float>{ 0.6f, 1.4f } : std::vector<float>{};
		std::size_t patAt = 0;
		auto realFrame = [&]() { return !pat.empty() ? base * pat[patAt++ % pat.size()] : a_c.jitter > 0.0f ? base * (1.0f + a_c.jitter * uni(rng)) : base; };

		float t = 0.0f, gx = 0.0f, gy = 0.0f;
		int   frames = 0, shifts = 0;
		auto  frame = [&](float a_realDt, bool a_body, const WindRules::Body& a_b, float a_step, float a_run) {
			const float dt = std::min(a_realDt, WindRules::kLongestFrame);
			const int   steps = WindRules::StepsPerFrame(dt);
			int         count = 0;
			if (a_body && WindRules::Disturbs(a_b, Settings::thickness)) {
				WindRules::SetDrag(cb, count++, a_b, Settings::actorRadius, Settings::actorPush, a_step, a_run);
			}
			cb.grid[0] = ox;
			cb.grid[1] = oy;
			cb.heightGrid[0] = ox;
			cb.heightGrid[1] = oy;
			cb.counts[0] = static_cast<float>(count);
			cb.dragParams[2] = a_c.e / static_cast<float>(steps);
			cb.shift[2] = dt;
			cb.time[0] = dt / steps;
			for (int s = 1; s <= steps; ++s) {
				const float part = static_cast<float>(s) / static_cast<float>(steps);
				cb.time[1] = t + dt * part;
				cb.time[2] = gx + windX * dt * part;
				cb.time[3] = gy + windY * dt * part;
				sim.Step(a_ctx, cb, hs, Settings::pressureIterations);
			}
			t += dt;
			gx += windX * dt;
			gy += windY * dt;
			++frames;
		};
		const WindRules::Body none{};
		for (float settled = 0.0f; settled < 2.0f;) {
			const float r = realFrame();
			frame(r, false, none, 0.0f, 0.0f);
			settled += r;
		}
		// start: calib as RunWake; game: near the centre (the grid follows the player), or 5.5 s before the centre without shift
		float px, py;
		if (calib) {
			px = 0.1f * extent;
			py = 0.5f * extent + 0.5f * kCell;
		} else if (a_c.shift) {
			px = 0.5f * extent + a_c.startX;
			py = 0.5f * extent + a_c.startY;
		} else {
			px = 0.5f * extent + 5.0f - hx * a_c.speed * 5.5f;
			py = 0.5f * extent + 11.0f - hy * a_c.speed * 5.5f;
		}
		const float        limit = calib ? 0.45f * extent : 1e9f;
		const float        seconds = calib ? 1e9f : a_c.seconds;
		float              run = 0.0f, length = 0.0f, since = 0.0f, nextProbe = a_c.probe0, lastVx = 0.0f, lastVy = 0.0f, lastX = px, lastY = py;
		std::vector<Probe> probes;
		double             gateSum = 0.0;
		int                gateN = 0;
		for (int k = 0;; ++k) {
			const float r = realFrame();
			float       step = 0.0f;
			if (k > 0) {
				step = a_c.speed * (a_c.jitter != 0.0f ? base : r);
				px += hx * step;
				py += hy * step;
				length += step;
				since += r;
			}
			if (length > limit || since > seconds + 1e-4f) break;
			if (a_c.shift && !calib) {
				const float half = 0.5f * extent;
				if (std::fabs(px - (ox + half)) > kRecentre || std::fabs(py - (oy + half)) > kRecentre) {
					const int dx = static_cast<int>(std::lround((px - half - ox) / kHCell));
					const int dy = static_cast<int>(std::lround((py - half - oy) / kHCell));
					sim.Shift(a_ctx, dx * static_cast<int>(std::lround(kHCell / kCell)), dy * static_cast<int>(std::lround(kHCell / kCell)));
					ox += dx * kHCell;
					oy += dy * kHCell;
					++shifts;
				}
			}
			run += step;
			// FillActors: velocity = position change / real seconds (0 on the first frame); step = speed x seconds
			const float meas = k > 0 ? step / r : a_c.speed;
			const WindRules::Body body{ px, py, 0.0f, hx * meas, hy * meas, 0.0f, false, true, 0.0f };
			frame(r, a_c.feed != 0, body, step, run);
			gateSum += WindRules::DragGate(meas, cb.dragParams[0], cb.dragParams[3]);
			++gateN;
			lastVx = hx * meas;
			lastVy = hy * meas;
			lastX = px;
			lastY = py;
			if (!calib && since + 1e-4f >= nextProbe) {
				const std::vector<float> fogNow = ReadFog(a_dev, a_ctx, sim.FogTexture(), res);
				Probe p = Measure(fogNow, res, ox, oy, px, py, lastVx, lastVy, R);
				MeasureExtra(p, fogNow, ReadVel(a_dev, a_ctx, sim.VelTexture(), res), res, ox, oy, px, py, lastVx, lastVy, R);
				p.t = since;
				probes.push_back(p);
				nextProbe += a_c.probeEvery;
			}
		}
		if (calib) {
			const std::vector<float> fogNow = ReadFog(a_dev, a_ctx, sim.FogTexture(), res);
			Probe p = Measure(fogNow, res, ox, oy, lastX, lastY, lastVx, lastVy, R);
			MeasureExtra(p, fogNow, ReadVel(a_dev, a_ctx, sim.VelTexture(), res), res, ox, oy, lastX, lastY, lastVx, lastVy, R);
			p.t = since;
			probes.push_back(p);
		}
		std::printf("CASE %s: res %d speed %.0f fps %.1f wind %.0f rel %.0f heading %.0f e %.2f shift %d jitter %.2f mode %s follow %.1f feed %d | frames %d shifts %d mean gate %.3f | adv %d\n",
			a_c.name.c_str(), res, a_c.speed, a_c.fps, a_c.wind, a_c.windRel, a_c.heading, a_c.e, a_c.shift, a_c.jitter, a_c.mode.c_str(), a_c.follow, a_c.feed, frames,
			shifts, gateN ? gateSum / gateN : 0.0, a_c.adv);
		double sP = 0, sPmin = 0, sC = 0, sT = 0, sO = 0, sA = 0, sS = 0, sU[6]{}, sProf[9]{}, sDelta = 0;
		float  gMin = 10, gMax = -10, gMinB = NAN, gMinC = NAN, gMaxB = NAN, gMaxC = NAN;
		double gMean = 0;
		float  loP = 10, hiP = -10, loMin = 10, loC = 10, hiC = -10, loT = 10, hiT = -10;
		for (const Probe& p : probes) {
			std::printf("  probe t %.2f speed %.0f: P n %d mean %.3f min %.3f | ahead n %d mean %.3f min %.3f | side n %d mean %.3f | C n %d %.3f | T mean %.3f min %.3f off %+.0f | band 2-3R %.3f 3-4R %.3f 4-5R %.3f 5-6R %.3f\n",
				p.t, p.speed, p.s.behind.n, p.s.behind.qMean, p.s.behind.qMin, p.s.ahead.n, p.s.ahead.qMean, p.s.ahead.qMin, p.s.side.n, p.s.side.qMean, p.cn, p.c, p.trough,
				p.troughMin, p.troughOff, p.band[0], p.band[1], p.band[2], p.band[3]);
			std::printf("    air along 0-1R %.0f 1-2R %.0f 2-3R %.0f 3-4R %.0f 4-5R %.0f 5-6R %.0f across %.1f | prof -4..+4 cells %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f | sum delta %.2f\n",
				p.u[0], p.u[1], p.u[2], p.u[3], p.u[4], p.u[5], p.uAcross, p.prof[0], p.prof[1], p.prof[2], p.prof[3], p.prof[4], p.prof[5], p.prof[6], p.prof[7], p.prof[8],
				p.sumDelta);
			for (int k = 0; k < 6; ++k) sU[k] += p.u[k];
			for (int k = 0; k < 9; ++k) sProf[k] += p.prof[k];
			sDelta += p.sumDelta;
			if (p.gridMin < gMin) {
				gMin = p.gridMin;
				gMinB = p.minB;
				gMinC = p.minC;
			}
			if (p.gridMax > gMax) {
				gMax = p.gridMax;
				gMaxB = p.maxB;
				gMaxC = p.maxC;
			}
			gMean += p.gridMean;
			sP += p.s.behind.qMean;
			sPmin += p.s.behind.qMin;
			sC += p.c;
			sT += p.trough;
			sO += p.troughOff;
			sA += p.s.ahead.qMean;
			sS += p.s.side.qMean;
			loP = std::min(loP, p.s.behind.qMean);
			hiP = std::max(hiP, p.s.behind.qMean);
			loMin = std::min(loMin, p.s.behind.qMin);
			loC = std::min(loC, p.c);
			hiC = std::max(hiC, p.c);
			loT = std::min(loT, p.trough);
			hiT = std::max(hiT, p.trough);
		}
		const double n = static_cast<double>(std::max<std::size_t>(probes.size(), 1));
		std::printf("SUM %s | P %.3f (%.3f..%.3f) Pmin mean %.3f lowest %.3f | C %.3f (%.3f..%.3f) | T %.3f (%.3f..%.3f) off %+.0f | ahead %.3f side %.3f | probes %d\n",
			a_c.name.c_str(), sP / n, loP, hiP, sPmin / n, loMin, sC / n, loC, hiC, sT / n, loT, hiT, sO / n, sA / n, sS / n, static_cast<int>(probes.size()));
		std::printf("SUMX %s adv %d | C %.3f | air along 0-1R %.0f 1-2R %.0f 2-3R %.0f 3-4R %.0f 4-5R %.0f 5-6R %.0f | prof -4..+4 cells %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f | sum delta %.2f | grid q min %.3f max %.3f mean %.4f | min at %.0f behind %.0f left, max at %.0f behind %.0f left\n",
			a_c.name.c_str(), a_c.adv, sC / n, sU[0] / n, sU[1] / n, sU[2] / n, sU[3] / n, sU[4] / n, sU[5] / n, sProf[0] / n, sProf[1] / n, sProf[2] / n, sProf[3] / n,
			sProf[4] / n, sProf[5] / n, sProf[6] / n, sProf[7] / n, sProf[8] / n, sDelta / n, gMin, gMax, gMean / n, gMinB, gMinC, gMaxB, gMaxC);
		std::fflush(stdout);
		hs->Release();
		ht->Release();
		sim.Release();
		return 0;
	}
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::printf("usage: WakeHeading <cases.txt>\n");
		return 2;
	}
	ID3D11Device*        dev = nullptr;
	ID3D11DeviceContext* ctx = nullptr;
	if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) return 3;
	std::ifstream in(argv[1]);
	if (!in) {
		std::printf("cannot open the case file: %s\n", argv[1]);
		return 2;
	}
	std::string   line;
	int           rc = 0;
	while (std::getline(in, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream ss(line);
		Case               c;
		if (!(ss >> c.name >> c.res >> c.speed >> c.fps >> c.wind >> c.windRel >> c.heading >> c.e >> c.shift >> c.jitter >> c.mode >> c.follow >> c.feed)) {
			std::printf("bad case line: %s\n", line.c_str());
			rc = 2;
			continue;
		}
		ss >> c.probe0 >> c.probeEvery >> c.gustMin >> c.vort >> c.relax >> c.startX >> c.startY >> c.cell >> c.adv >> c.seconds;
		kCell = c.cell;
		rc = std::max(rc, RunCase(dev, ctx, c));
	}
	ctx->Release();
	dev->Release();
	return rc;
}
