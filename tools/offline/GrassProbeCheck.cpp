// Offline check of the grass depth probe's pure parts (src/GrassProbeMath.h, CallScan::FindJumpsTo) and of the depth
// copy chain it uses in game (identical-description copy, typeless staging texture, Map) on WARP. Prints PASS/FAIL
// lines; exit code 1 on any FAIL. Not run by build.cmd (its lines must stay as they are): build-grassprobe.cmd.

#include "CallScan.h"
#include "GrassProbeDefer.h"
#include "GrassProbeMath.h"

#include <Windows.h>
#include <d3d11.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace
{
	namespace M = GrassProbeMath;

	int g_fail = 0;

	void Check(bool a_ok, const char* a_name, const std::string& a_detail)
	{
		std::printf("%s %s : %s\n", a_ok ? "PASS" : "FAIL", a_name, a_detail.c_str());
		if (!a_ok) {
			++g_fail;
		}
	}

	// A standard projection like the game's (near 15 units, as logged in game 2026-11-07; far 353840 assumed).
	constexpr float kNear = 15.0f;
	constexpr float kFar = 353840.0f;
	const float     kP33 = kFar / (kFar - kNear);
	const float     kP43 = -kNear * kFar / (kFar - kNear);

	float DepthAt(float a_z) { return kP33 + kP43 / a_z; }

	void TestLinearDepth()
	{
		float worst = 0.0f;
		for (const float z : { 20.0f, 600.0f, 2000.0f, 6000.0f, 100000.0f }) {
			const float back = M::LinearDepth(DepthAt(z), kP33, kP43);
			worst = std::max(worst, std::fabs(back - z) / z);
		}
		Check(worst < 0.01f, "grassprobe linear depth normal", "round trip 20..100000 units, worst relative error " + std::to_string(worst));
		Check(M::LinearDepth(kP33, kP33, kP43) == M::kNoDistance, "grassprobe linear depth boundary", "depth equal to _33 gives no distance");
		// Reversed projection: d' = 1 - d, _33' = 1 - _33, _43' = -_43.
		const float rz = M::LinearDepth(1.0f - DepthAt(1500.0f), 1.0f - kP33, -kP43);
		Check(std::fabs(rz - 1500.0f) < 15.0f && !M::DepthGrows(-kP43) && M::DepthGrows(kP43), "grassprobe linear depth reversed",
			"1500 units back as " + std::to_string(rz));
	}

	void TestBands()
	{
		const bool edges = M::Band(0.0f) == 0 && M::Band(599.9f) == 0 && M::Band(600.0f) == 1 && M::Band(1023.9f) == 1 && M::Band(1024.0f) == 2 &&
		                   M::Band(1999.0f) == 2 && M::Band(2000.0f) == 3 && M::Band(5999.0f) == 3 && M::Band(6000.0f) == M::kFarBand && M::Band(1e8f) == M::kFarBand;
		Check(edges, "grassprobe bands normal", "0 / 600 / 1024 / 2000 / 6000 edges fall in the upper band");
		Check(M::Band(-5.0f) == M::kBadBand && M::Band(std::numeric_limits<float>::quiet_NaN()) == M::kBadBand && M::Band(M::kNoDistance) == M::kBadBand,
			"grassprobe bands failure", "negative, NaN and no distance go to the bad band, not to 6000+");
	}

	void TestDecode()
	{
		auto d24 = [](std::uint32_t a_v) {
			std::uint8_t b[4];
			std::memcpy(b, &a_v, 4);
			return M::DecodeDepth(M::Codec::kD24S8, b);
		};
		Check(d24(0xFFFFFFFFu) == 1.0f && d24(0x12000000u) == 0.0f && std::fabs(d24(0x07800000u) - 0.5f) < 1e-6f, "grassprobe decode d24",
			"stencil byte ignored, 0xFFFFFF = 1, 0x800000 = 0.5");
		std::uint8_t f8[8]{};
		const float  v = 0.75f;
		std::memcpy(f8, &v, 4);
		f8[4] = 0xAB;
		const std::uint16_t h = 0x8000;
		std::uint8_t        h2[2];
		std::memcpy(h2, &h, 2);
		Check(M::DecodeDepth(M::Codec::kF32S8, f8) == 0.75f && std::fabs(M::DecodeDepth(M::Codec::kD16, h2) - 0.50001f) < 1e-4f &&
		          M::DecodeDepth(M::Codec::kNone, f8) == 0.0f && M::TexelBytes(M::Codec::kF32S8) == 8 && M::TexelBytes(M::Codec::kNone) == 0,
			"grassprobe decode other", "f32+s8 first float, d16, unknown codec reads 0 and has no size");
		Check(M::CodecOf(DXGI_FORMAT_R24G8_TYPELESS) == M::Codec::kD24S8 && M::CodecOf(DXGI_FORMAT_R32_TYPELESS) == M::Codec::kF32 &&
		          M::CodecOf(DXGI_FORMAT_R32G8X24_TYPELESS) == M::Codec::kF32S8 && M::CodecOf(DXGI_FORMAT_R8G8B8A8_UNORM) == M::Codec::kNone,
			"grassprobe codec of format", "typeless depth formats map, a colour format does not");
	}

	// One sample of two float depths (D32 layout): keys are the raw bits, as GrassProbe.cpp reads an R32 texture.
	void AccF(M::Counts& a_c, float a_pre, float a_fin, float a_p33, float a_p43)
	{
		std::uint8_t tPre[4];
		std::uint8_t tFin[4];
		std::memcpy(tPre, &a_pre, 4);
		std::memcpy(tFin, &a_fin, 4);
		M::Accumulate(a_c, a_pre, a_fin, M::RawKey(M::Codec::kF32, tPre), M::RawKey(M::Codec::kF32, tFin), a_p33, a_p43);
	}

	void TestAccumulate()
	{
		M::Counts c{};
		AccF(c, DepthAt(1000.0f), DepthAt(1000.0f), kP33, kP43);   // untouched pixel
		AccF(c, DepthAt(1100.0f), DepthAt(1000.0f), kP33, kP43);   // grass blade at 1000 over ground at 1100
		AccF(c, DepthAt(1005.0f), DepthAt(1000.0f), kP33, kP43);   // shifted surface: gap 0.5 %
		AccF(c, 1.0f, DepthAt(300.0f), kP33, kP43);                // snapshot empty, final near
		AccF(c, 1.0f, 1.0f, kP33, kP43);                           // sky in both
		const bool normal = c.samples[1] == 3 && c.differ[1] == 2 && c.tinyGap[1] == 1 && c.preNearer[1] == 0 && c.samples[0] == 1 && c.differ[0] == 1 &&
		                    c.preFar[0] == 1 && c.samples[M::kSkyBand] == 1 && c.differ[M::kSkyBand] == 0 && c.preFar[M::kSkyBand] == 1;
		Check(normal, "grassprobe accumulate normal", "same / grass / shifted / empty snapshot / sky land in their bands");

		M::Counts n{};
		const float d900 = DepthAt(900.0f);
		AccF(n, DepthAt(500.0f), d900, kP33, kP43);                                         // snapshot nearer than final: not a subset
		AccF(n, std::nextafter(d900, 2.0f), d900, kP33, kP43);                               // one float step farther: differs, not nearer
		Check(n.preNearer[1] == 1 && n.differ[1] == 2 && n.Total(n.preNearer) == 1, "grassprobe accumulate nearer",
			"a nearer snapshot is counted; a one-step change differs (bit-exact) without being nearer");

		M::Counts r{};
		const float rp33 = 1.0f - kP33;
		const float rp43 = -kP43;
		AccF(r, 1.0f - DepthAt(1100.0f), 1.0f - DepthAt(1000.0f), rp33, rp43);   // reversed: grass over ground
		AccF(r, 1.0f - DepthAt(500.0f), 1.0f - DepthAt(900.0f), rp33, rp43);     // reversed: snapshot nearer
		AccF(r, 0.0f, 0.0f, rp33, rp43);                                          // reversed sky
		Check(r.differ[1] == 2 && r.preNearer[1] == 1 && r.samples[M::kSkyBand] == 1 && r.preFar[M::kSkyBand] == 1, "grassprobe accumulate reversed",
			"far value 0 and nearer direction follow the sign of _43");
		Check(M::Percent(0, 0) == 0.0 && M::Percent(1, 4) == 25.0, "grassprobe percent", "0 of nothing is 0, 1 of 4 is 25");
	}

	void TestKeys()
	{
		auto key24 = [](std::uint32_t a_v) {
			std::uint8_t b[4];
			std::memcpy(b, &a_v, 4);
			return M::RawKey(M::Codec::kD24S8, b);
		};
		// Stencil-only change: same key; one D24 step: different key, ordered like the depth.
		Check(key24(0x07123456u) == key24(0xFF123456u) && key24(0x00123457u) != key24(0x00123456u) && key24(0x00123457u) > key24(0x00123456u),
			"grassprobe keys d24", "stencil byte ignored, one depth step differs");
		const float   neg = -0.0f;
		std::uint8_t  nb[4];
		std::memcpy(nb, &neg, 4);
		std::uint8_t  hb[4];
		const float   half = 0.5f;
		std::memcpy(hb, &half, 4);
		std::uint8_t  ob[4];
		const float   one = 1.0f;
		std::memcpy(ob, &one, 4);
		Check(M::RawKey(M::Codec::kF32, nb) == 0 && M::RawKey(M::Codec::kF32, hb) < M::RawKey(M::Codec::kF32, ob), "grassprobe keys float",
			"-0 reads as 0, float bits order like the depth");
		Check(M::QuantKey24(1.0f) == 0xFFFFFFu && M::QuantKey24(-1.0f) == 0 && M::QuantKey24(0x800000 / 16777215.0f) == 0x800000u && M::NearerKey(1, 2, true) &&
		          M::NearerKey(2, 1, false) && !M::NearerKey(2, 2, true),
			"grassprobe keys across codecs", "D24 grid: 1 -> FFFFFF, clamped below 0, exact for a D24 value; nearer follows the direction");
	}

	// ---- probe v2 helpers ----

	void TestKeyToDepth()
	{
		std::uint8_t f[8]{};
		const float  v = 0.0423f;
		std::memcpy(f, &v, 4);
		f[4] = 0x5A;   // stencil byte of an R32G8X24 texel
		std::uint8_t        d24[4]{};
		const std::uint32_t r24 = 0xAB800000u;   // stencil AB, depth 0x800000
		std::memcpy(d24, &r24, 4);
		Check(M::KeyToDepth(M::Codec::kF32S8, M::RawKey(M::Codec::kF32S8, f)) == v &&
		          M::KeyToDepth(M::Codec::kD24S8, M::RawKey(M::Codec::kD24S8, d24)) == M::DecodeDepth(M::Codec::kD24S8, d24),
			"grassprobe key to depth normal", "float and D24 keys decode back to the texel's depth");
		Check(M::KeyToDepth(M::Codec::kD24S8, 0xFFFFFFu) == 1.0f && M::KeyToDepth(M::Codec::kD16, 0xFFFFu) == 1.0f && M::KeyToDepth(M::Codec::kF32, 0u) == 0.0f,
			"grassprobe key to depth edge", "largest D24 / D16 key is 1, float key 0 is 0");
		Check(M::KeyToDepth(M::Codec::kNone, 0x3F800000u) == 0.0f, "grassprobe key to depth failure", "unknown codec reads 0");
	}

	void TestRawStats()
	{
		M::RawStats s{};
		for (const float x : { 0.05f, 0.15f, 0.15f, 0.95f }) s.Add(x);
		Check(s.n == 4 && s.min == 0.05f && s.max == 0.95f && s.hist[0] == 1 && s.hist[1] == 2 && s.hist[9] == 1 && s.atZero == 0 && s.atOne == 0,
			"grassprobe raw stats normal", "range and tenths of 0.05, 0.15, 0.15, 0.95");
		M::RawStats e{};
		e.Add(0.0f);
		e.Add(1.0f);
		e.Add(0.1f);
		Check(e.atZero == 1 && e.atOne == 1 && e.hist[0] == 1 && e.hist[9] == 1 && e.hist[1] == 1 && e.n == 3, "grassprobe raw stats edge",
			"exact 0 and 1 are counted and land in the first and last tenth, 0.1 in the second");
		M::RawStats b{};
		b.Add(std::numeric_limits<float>::quiet_NaN());
		b.Add(-0.5f);
		b.Add(1.5f);
		Check(b.outside == 3 && b.n == 0 && b.atZero == 0 && b.atOne == 0, "grassprobe raw stats failure", "NaN, negative and above-1 values only count as outside");
	}

	void TestConvention()
	{
		const auto g = M::DecideConvention(9900, 50, 0, 0, 1000, -15.0f);
		const auto r = M::DecideConvention(0, 17025, 0, 0, 256000, -15.0f);   // probe1-like: later draws only raised the value
		Check(g.grows && g.rule == M::Rule::kWrites && !r.grows && r.rule == M::Rule::kWrites, "grassprobe convention writes",
			"writes going smaller -> grows, larger -> reversed, whatever _43 says");
		const auto at256 = M::DecideConvention(0, 256, 0, 0, 1000, -15.0f);
		const auto at255 = M::DecideConvention(0, 255, 0, 0, 1000, -15.0f);
		const auto at99 = M::DecideConvention(990, 10, 0, 0, 1000, 15.0f);
		const auto under99 = M::DecideConvention(989, 11, 0, 0, 1000, 15.0f);
		Check(!at256.grows && at256.rule == M::Rule::kWrites && at255.rule == M::Rule::kProjection && at99.grows && at99.rule == M::Rule::kWrites &&
		          under99.rule == M::Rule::kProjection && !under99.grows,
			"grassprobe convention thresholds", "exactly 256 votes and exactly 99 % decide; 255 votes and 98.9 % fall through to the projection");
		const auto few = M::DecideConvention(0, 255, 0, 3000, 256000, -15.0f);
		const auto split = M::DecideConvention(600, 400, 5000, 0, 256000, -15.0f);
		Check(few.grows && few.rule == M::Rule::kClear && !split.grows && split.rule == M::Rule::kClear, "grassprobe convention edge",
			"255 votes are too few and a 60/40 split decides nothing: the clear thresholds decide (raw>=0.9999999 -> grows, raw<=1e-7 -> reversed)");
		const auto both = M::DecideConvention(0, 0, 10, 10, 1000, -15.0f);
		const auto empty = M::DecideConvention(0, 0, 0, 0, 0, 15.0f);
		const auto thin = M::DecideConvention(0, 0, 0, 9, 1000, -15.0f);   // 0.9 % at the far value: under the 1 % share
		Check(both.grows && both.rule == M::Rule::kProjection && !empty.grows && empty.rule == M::Rule::kProjection && thin.rule == M::Rule::kProjection,
			"grassprobe convention failure", "both clear values, no data or too few clear samples: the sign of _43 decides");
	}

	void TestThresholds()
	{
		Check(M::IsFarValue(1.0f, true) && M::IsFarValue(0xFFFFFE / 16777215.0f, true) && M::IsFarValue(0xFFFFFD / 16777215.0f, true) &&
		          !M::IsFarValue(0xFFFFFC / 16777215.0f, true) && M::IsFarValue(0.0f, false) && M::IsFarValue(1e-7f, false) && !M::IsFarValue(2e-7f, false),
			"grassprobe far thresholds", "grows: >= 0.9999999f (= 1 - 2^-23: D24 FFFFFE and FFFFFD count, FFFFFC not); reversed: <= 1e-7");
		Check(!M::IsFarValue(0.0f, true) && !M::IsFarValue(1.0f, false), "grassprobe far thresholds failure", "the other convention's far value is not far");
		Check(std::fabs(M::KeyToDepth(M::Codec::kD16, 0x8000u) - 32768.0f / 65535.0f) < 1e-7f && M::KeyToDepth(M::Codec::kD16, 0u) == 0.0f,
			"grassprobe key to depth d16", "D16 key 0x8000 is 32768 / 65535, key 0 is 0");
	}

	void TestDistance()
	{
		float worst = 0.0f;
		for (const float z : { 84.0f, 256.0f, 2000.0f, 6000.0f, 50000.0f }) {
			const float rev = 1.0f - DepthAt(z);   // what probe1 read: 1 - standard, i.e. reversed Z of the same planes
			worst = std::max(worst, std::fabs(M::Distance(rev, true, kP33, kP43) - z) / z);
		}
		Check(worst < 0.005f && std::fabs(M::Distance(DepthAt(1500.0f), false, kP33, kP43) - M::LinearDepth(DepthAt(1500.0f), kP33, kP43)) < 0.5f,
			"grassprobe distance normal", "flipped read of 1 - d gives 84..50000 units back (worst " + std::to_string(worst) + "), unflipped equals LinearDepth");
		// A reversed buffer built from another assumption: infinite far plane, d = near / z. The 1 - d rule only holds for
		// the same near and far planes; here it must be visibly wrong far out, so this check proves the rule's limit.
		const float infFar50k = M::Distance(kNear / 50000.0f, true, kP33, kP43);
		const float infFar84 = M::Distance(kNear / 84.0f, true, kP33, kP43);
		Check(std::fabs(infFar50k - 50000.0f) / 50000.0f > 0.05f && std::fabs(infFar84 - 84.0f) / 84.0f < 0.001f, "grassprobe distance other reversed projection",
			"infinite-far reversed d = near / z read by the 1 - d rule: 84 units back as " + std::to_string(infFar84) + ", 50000 as " + std::to_string(infFar50k) +
				" (wrong by more than 5 %, as expected)");
		Check(M::Distance(kP33, false, kP33, kP43) == M::kNoDistance && M::Distance(1.0f - kP33, true, kP33, kP43) == M::kNoDistance, "grassprobe distance edge",
			"a value on the pole gives no distance, flipped or not");
		Check(M::FinalBand(1.0f, true, false, kP33, kP43) == M::kSkyBand && M::FinalBand(0.0f, false, true, kP33, kP43) == M::kSkyBand &&
		          M::FinalBand(1.0f - DepthAt(300.0f), false, true, kP33, kP43) == 0 && M::FinalBand(1.0f - DepthAt(3000.0f), false, true, kP33, kP43) == 3,
			"grassprobe final band", "far value per convention is sky; reversed 300 / 3000 units land in 0-600 / 2000-6000");
		Check(M::FinalBand(0.0f, true, false, kP33, kP43) == 0 && M::FinalBand(1.5f, false, false, kP33, kP43) == M::kBadBand, "grassprobe final band failure",
			"a reversed sky read as growing lands in the nearest band (the v1 symptom); a raw value above 1 read unflipped gives a negative distance: bad band");
	}

	void TestGrassCounts()
	{
		M::GrassCounts c{};
		M::VoteWrite(c, 100, 200);   // reversed: nearer is larger
		M::VoteWrite(c, 300, 100);
		M::VoteWrite(c, 50, 50);
		const bool w1 = M::AddGrass(c, 1, 0, 100, 200, 200, 500.0f, 500.0f);    // wrote, still final
		const bool w2 = M::AddGrass(c, 1, 0, 100, 200, 250, 500.0f, 400.0f);    // wrote, covered later
		const bool w3 = M::AddGrass(c, 2, 2, 70, 70, 90, 1500.0f, 1500.0f);     // untouched
		const bool w4 = M::AddGrass(c, 1, 0, 100, 200, 201, 500.0f, 503.0f);    // wrote, main pass redrew it 0.6 % off
		Check(w1 && w2 && !w3 && w4 && c.wrote[1] == 3 && c.visible[1] == 1 && c.nearEqual[1] == 1 && c.wroteOwn[0] == 3 && c.visibleOwn[0] == 1 && c.wrote[2] == 0 &&
		          c.larger == 1 && c.smaller == 1,
			"grassprobe grass counts normal", "wrote / visible / near-equal / covered / untouched, own-distance bands and the vote directions");
		Check(!M::NearEqual(500.0f, 506.0f) && M::NearEqual(500.0f, 505.0f) && !M::NearEqual(500.0f, M::kNoDistance) && !M::NearEqual(-1.0f, -1.0f),
			"grassprobe near-equal edge", "1 % of the final distance is the limit; no distance or a negative one is never near");
		Check(!M::AddGrass(c, -1, 0, 1, 2, 2, 1.0f, 1.0f) && !M::AddGrass(c, M::kBands, 0, 1, 2, 2, 1.0f, 1.0f) && !M::AddGrass(c, 1, M::kBands, 1, 2, 2, 1.0f, 1.0f) &&
		          c.wrote[1] == 3,
			"grassprobe grass counts failure", "a band out of range counts nothing");
		Check(M::MaskValue(true, true, false, false) == M::kMaskGrassVisible && M::MaskValue(true, false, false, false) == M::kMaskGrassCovered &&
		          M::MaskValue(true, false, true, false) == M::kMaskGrassNear && M::MaskValue(false, false, false, true) == M::kMaskFinalFar &&
		          M::MaskValue(false, false, false, false) == M::kMaskNoGrass && M::MaskValue(true, true, true, true) == 255,
			"grassprobe mask values", "visible 255, near 160, covered 96, sky 32, none 0; a grass write wins over the far flag");
	}

	void TestState()
	{
		M::StateSet a{ true, 3, 0, 1, 0, 0, 0, true, 0.5f };
		M::StateSet b = a;
		Check(M::StateDiff(a, b) == 0, "grassprobe state equal", "the same set differs in nothing");
		b.cull = 0;
		b.alphaRef = 0.50000006f;
		const auto d = M::StateDiff(a, b);
		Check(d == ((1u << 2) | (1u << 7)) && std::string(M::kStateNames[2]) == "cull" && std::string(M::kStateNames[7]) == "alphaRef", "grassprobe state differs",
			"cull and a one-ulp alphaRef change are both reported");
		M::StateSet none{};
		Check(M::StateDiff(a, none) == M::kStateInvalid && M::StateDiff(none, none) == M::kStateInvalid, "grassprobe state failure",
			"a set that was not recorded never passes the precheck");
	}

	void TestCadence()
	{
		Check(M::ProbeGridPixel(4, 9, 1280.0f) == 640 && M::ProbeGridPixel(3, 7, 800.0f) == 400 && M::ProbeGridPixel(0, 7, 800.0f) == 57 &&
		          M::ProbeGridPixel(8, 9, 1280.0f) == 1208,
			"grassprobe probe grid", "p31 = (640, 400), p04 row 57, last column 1208, as the [PROBE] lines show at 1280x800");
		Check(M::ProbeGridPixel(1, 0, 800.0f) == 0 && M::ProbeGridPixel(1, 7, 0.0f) == 0, "grassprobe probe grid failure", "no cells or no size gives pixel 0");
		Check(!M::DeferredComparison(1, 2) && M::DeferredComparison(2, 2) && !M::DeferredComparison(2, 3) && !M::DeferredComparison(2, 0) && M::DeferredComparison(2, 6),
			"grassprobe deferred cadence", "mode 2 defers even comparisons only; mode 1 and index 0 never");
		Check(M::DumpDue(1, 6, true) && M::DumpDue(1, 12, true) && !M::DumpDue(1, 6, false) && !M::DumpDue(1, 5, true) && !M::DumpDue(1, 0, true),
			"grassprobe dump cadence", "mode 1: every 6th comparison in an exterior");
		int normal = 0;
		int deferred = 0;
		int dumps = 0;
		for (std::uint64_t i = 1; dumps < 8 && i < 200; ++i) {
			if (M::DumpDue(2, i, true)) {
				++dumps;
				(M::DeferredComparison(2, i) ? deferred : normal) += 1;
			}
		}
		Check(dumps == 8 && normal >= 3 && deferred >= 3 && !M::DumpDue(2, 1, true) && M::DumpDue(2, 6, true) && M::DumpDue(2, 13, true),
			"grassprobe dump cadence mode 2", "8 consecutive dumps hold " + std::to_string(normal) + " normal and " + std::to_string(deferred) + " deferred frames");
		Check(M::FogProbeDue(1, 1, 0) && M::FogProbeDue(1, 44, 2) && !M::FogProbeDue(1, 44, 3) && M::FogProbeDue(1, 48, 3) && M::FogProbeDue(2, 13, 3) &&
		          !M::FogProbeDue(2, 12, 3) && !M::FogProbeDue(1, 0, 0),
			"grassprobe fog probe cadence", "every comparison until 3 [PROBE]s were submitted, then on the dump cadence (mode 2 alternates); index 0 never");
		Check(M::DumpSlot(0) == 0 && M::DumpSlot(7) == 7 && M::DumpSlot(8) == 0 && M::DumpSlot(17) == 1, "grassprobe dump ring", "8 pairs, the 9th overwrites the first");
	}

	// ---- deferral bookkeeping (GrassProbeDefer.h) driven by call sequences ----
	int g_accA = 0;   // stand-ins for accumulators: only their addresses matter
	int g_accB = 0;
	constexpr std::uint32_t kGrassLo = 0x5C000030;
	constexpr std::uint32_t kGrassHi = 0x5C00005C;

	struct Step
	{
		int           site;
		void*         acc;
		std::uint32_t first;
		std::uint32_t last;
		std::uint32_t flags;
		int           group;
		bool          prepass;
		bool          mainDepth;
	};

	Step RangeCall(std::uint32_t a_flags) { return { 24, &g_accA, 0x2B, 0x4000002B, a_flags, -1, true, true }; }
	Step GrassCall(std::uint32_t a_flags) { return { 25, &g_accA, kGrassLo, kGrassHi, a_flags, -1, true, true }; }
	Step GroupCall(int a_site, int a_group, std::uint32_t a_flags) { return { a_site, &g_accA, 1, 0x5C006074, a_flags, a_group, true, true }; }

	// One prepass round as probe1's frame 13174 shows it: range, grass, group 1, group 9, group 4.
	std::vector<Step> FullRound(std::uint32_t a_flags, std::uint32_t a_decalFlags)
	{
		return { RangeCall(a_flags), GrassCall(a_flags), GroupCall(26, 1, a_flags), GroupCall(27, 9, a_flags), GroupCall(17, 4, a_decalFlags) };
	}

	struct SimEnv
	{
		std::vector<GrassProbeDefer::Args> made;   // calls actually made, in order
		GrassProbeDefer::Gate              gate{};
		int                                rewrittenSite{ -1 };
		int                                deferred{ 0 };
		int                                planned{ 0 };
		int                                late{ 0 };
		int                                drops{ 0 };
		int                                onMakeSite{ -1 };   // while this site's call is made, run onMake (a nested call)
		std::function<void()>              onMake;

		void Game(const GrassProbeDefer::Args& a_a)
		{
			made.push_back(a_a);
			if (onMake && a_a.site == onMakeSite) {
				auto f = std::move(onMake);
				onMake = nullptr;
				f();
			}
		}
		void Seen(const GrassProbeDefer::Call&, const GrassProbeDefer::Before&) {}
		GrassProbeDefer::Gate GateFor(int) const { return gate; }
		bool SiteIntact(int a_site) const { return a_site >= 0 && a_site != rewrittenSite; }
		void Decided(int, const char* a_whyNot)
		{
			if (!a_whyNot) ++deferred;
		}
		void BeforeGrass(int) {}
		void AfterGrass(int) {}
		void G9Returned(int) {}
		void ReplayBegin(int) {}
		void ReplayEnd(int, bool a_planned, int) { (a_planned ? planned : late) += 1; }
		void Dropped(int a_round)
		{
			if (a_round >= 0) ++drops;
		}
	};

	void Run(GrassProbeDefer::Tracker& a_t, SimEnv& a_e, const Step& a_s)
	{
		const GrassProbeDefer::Call c{ a_s.site, a_s.acc, a_s.group, a_s.first == kGrassLo && a_s.last == kGrassHi, a_s.prepass, a_s.mainDepth };
		const GrassProbeDefer::Args a{ 0x1000u + static_cast<std::uintptr_t>(a_s.site), a_s.acc, a_s.first, a_s.last, a_s.flags, a_s.group, a_s.site };
		GrassProbeDefer::Dispatch(a_t, c, a, a_e);
	}

	void RunAll(GrassProbeDefer::Tracker& a_t, SimEnv& a_e, const std::vector<Step>& a_steps)
	{
		for (const auto& s : a_steps) Run(a_t, a_e, s);
	}

	std::string MadeText(const std::vector<GrassProbeDefer::Args>& a_made)
	{
		std::string s;
		for (const auto& a : a_made) s += (s.empty() ? "" : " ") + std::to_string(a.site);
		return s;
	}

	// A normal comparison frame (copies, no deferral) followed by the frame end: the tracker learns the round structure.
	void NormalFrame(GrassProbeDefer::Tracker& a_t)
	{
		SimEnv e;
		e.gate = { true, false, true, nullptr };
		auto steps = FullRound(0x21, 0x21);
		const auto r2 = FullRound(0x1, 0x5);
		steps.insert(steps.end(), r2.begin(), r2.end());
		RunAll(a_t, e, steps);
		a_t.PrepassEnd();
		a_t.EndFrame();
	}

	SimEnv DeferredEnv() { SimEnv e; e.gate = { true, true, true, nullptr }; return e; }

	void TestTracker()
	{
		namespace D = GrassProbeDefer;
		{   // frame 13174's order: both rounds held back and made exactly once, after their group-9 call, same accumulator and arguments
			D::Tracker t;
			NormalFrame(t);
			SimEnv e = DeferredEnv();
			RunAll(t, e, { { 24, &g_accB, 0x2B, 0x4000002B, 0x400, -1, false, false }, { 25, &g_accB, kGrassLo, kGrassHi, 0x400, -1, false, false } });   // a shadow view first
			auto steps = FullRound(0x21, 0x21);
			const auto r2 = FullRound(0x1, 0x5);
			steps.insert(steps.end(), r2.begin(), r2.end());
			RunAll(t, e, steps);
			t.PrepassEnd();
			Run(t, e, { 0, &g_accB, 1, 0x5C00002F, 0x41, -1, false, true });   // main pass
			const std::string order = MadeText(e.made);
			bool same = true;
			int  grass = 0;
			for (const auto& a : e.made) {
				if (a.site == 25 && a.acc == &g_accA) {
					same = same && a.first == kGrassLo && a.last == kGrassHi && a.fn == 0x1000u + 25 && a.group == -1 && a.flags == (grass == 0 ? 0x21u : 0x1u);
					++grass;
				}
			}
			Check(order == "24 25 24 26 27 25 17 24 26 27 25 17 0" && grass == 2 && same && e.deferred == 2 && e.planned == 2 && e.late == 0 && e.drops == 0 &&
			          t.State(0) == D::RoundState::kAfterG9 && t.State(1) == D::RoundState::kAfterG9 && !t.Off(),
				"grassprobe defer frame 13174", "calls made: " + order + "; both rounds replayed once after group 9 with their own flags");
		}
		{   // the first frame has no previous round structure: nothing is held back
			D::Tracker t;
			SimEnv     e = DeferredEnv();
			RunAll(t, e, FullRound(0x21, 0x21));
			Check(e.deferred == 0 && MadeText(e.made) == "24 25 26 27 17" && t.State(0) == D::RoundState::kNormal, "grassprobe defer needs a previous frame",
				"no group-9 call seen last frame: the grass call is made at its place");
		}
		{   // a round closed by group 4 without a group-9 call: made before the group-4 call (late), mode 2 off afterwards
			D::Tracker t;
			NormalFrame(t);
			SimEnv e = DeferredEnv();
			RunAll(t, e, { RangeCall(0x21), GrassCall(0x21), GroupCall(26, 1, 0x21), GroupCall(17, 4, 0x21) });
			Check(MadeText(e.made) == "24 26 25 17" && e.late == 1 && e.planned == 0 && e.drops == 0 && t.State(0) == D::RoundState::kLate && t.Off(),
				"grassprobe defer late replay", "calls made: " + MadeText(e.made) + "; the held call is not lost");
			t.EndFrame();
			NormalFrame(t);   // refills the group-9 history, so only the off switch can block the next deferral
			SimEnv e2 = DeferredEnv();
			auto   steps = FullRound(0x21, 0x21);
			const auto r2 = FullRound(0x1, 0x5);
			steps.insert(steps.end(), r2.begin(), r2.end());
			RunAll(t, e2, steps);
			Check(e2.deferred == 0 && MadeText(e2.made) == "24 25 26 27 17 24 25 26 27 17" && e2.planned == 0 && t.State(0) == D::RoundState::kNormal,
				"grassprobe defer off after a late replay", "after a normal frame with group 9 the next deferred frame still copies normally: " + MadeText(e2.made));
		}
		{   // another accumulator comes first: dropped once, logged by the caller, mode 2 off
			D::Tracker t;
			NormalFrame(t);
			SimEnv e = DeferredEnv();
			RunAll(t, e, { RangeCall(0x21), GrassCall(0x21), GroupCall(26, 1, 0x21), { 31, &g_accB, 1, 0x5C006074, 0, 14, true, true } });
			Check(MadeText(e.made) == "24 26 31" && e.drops == 1 && e.late == 0 && t.State(0) == D::RoundState::kDropped && t.Off(), "grassprobe defer other accumulator",
				"calls made: " + MadeText(e.made) + "; dropped before the other accumulator's call");
		}
		{   // a call nested inside the group-9 call passes on untouched; the replay still follows the group-9 call
			D::Tracker t;
			NormalFrame(t);
			SimEnv e = DeferredEnv();
			e.onMakeSite = 27;
			e.onMake = [&]() { Run(t, e, { 40, &g_accA, 0x2B, 0x4000002B, 0x21, -1, true, true }); };
			RunAll(t, e, FullRound(0x21, 0x21));
			Check(MadeText(e.made) == "24 26 27 40 25 17" && e.planned == 1 && e.late == 0 && e.drops == 0, "grassprobe defer nested call",
				"calls made: " + MadeText(e.made));
		}
		{   // the setting goes 2 -> 0 after round 0 was held back: round 0 is still made, round 1 is not touched
			D::Tracker t;
			NormalFrame(t);
			SimEnv e = DeferredEnv();
			RunAll(t, e, { RangeCall(0x21), GrassCall(0x21), GroupCall(26, 1, 0x21) });
			e.gate = {};   // GrassProbe = 0: not armed, no mode 2
			RunAll(t, e, { GroupCall(27, 9, 0x21), GroupCall(17, 4, 0x21) });
			RunAll(t, e, FullRound(0x1, 0x5));
			Check(MadeText(e.made) == "24 26 27 25 17 24 25 26 27 17" && e.planned == 1 && e.deferred == 1 && t.State(1) == D::RoundState::kNone,
				"grassprobe defer setting off mid-frame", "calls made: " + MadeText(e.made));
		}
		{   // the prepass ends with a call still held: the next call is outside the prepass, so it is dropped; nothing is left held
			D::Tracker t;
			NormalFrame(t);
			SimEnv e = DeferredEnv();
			RunAll(t, e, { RangeCall(0x21), GrassCall(0x21), GroupCall(26, 1, 0x21) });
			t.PrepassEnd();
			Run(t, e, { 30, &g_accA, 1, 0x5C006074, 0x41, -1, false, true });
			Check(MadeText(e.made) == "24 26 30" && e.drops == 1 && !t.StashActive() && t.EndFrame() == -1, "grassprobe defer prepass end",
				"calls made: " + MadeText(e.made) + "; dropped at the first call outside the prepass");
			D::Tracker t2;
			NormalFrame(t2);
			SimEnv e2 = DeferredEnv();
			RunAll(t2, e2, { RangeCall(0x21), GrassCall(0x21) });
			Check(t2.StashActive() && t2.EndFrame() == 0 && !t2.StashActive() && t2.Off(), "grassprobe defer frame end", "a call still held at the frame end is dropped there");
		}
		{   // the grass call no longer points at our thunk: copied normally, made at its place
			D::Tracker t;
			NormalFrame(t);
			SimEnv e = DeferredEnv();
			e.rewrittenSite = 25;
			RunAll(t, e, FullRound(0x21, 0x21));
			SimEnv e2 = DeferredEnv();
			D::Tracker t2;
			NormalFrame(t2);
			e2.gate.stateBlock = "state differs (round 0: cull)";
			RunAll(t2, e2, FullRound(0x21, 0x21));
			Check(e.deferred == 0 && MadeText(e.made) == "24 25 26 27 17" && t.State(0) == D::RoundState::kNormal && e2.deferred == 0 &&
			          MadeText(e2.made) == "24 25 26 27 17",
				"grassprobe defer blockers", "a rewritten call site or a failed state precheck keeps the call at its place");
		}
	}

	// The render-state precheck: per-round gate and invalidation on a context change (GrassProbeDefer.h Precheck).
	void TestPrecheck()
	{
		namespace D = GrassProbeDefer;
		int        worldA = 0;
		int        worldB = 0;
		int        cellA = 0;
		int        cellB = 0;
		const D::Context ext{ 2, true, &worldA };
		{   // only round 0 was kept on the normal comparison: round 1 was never checked and must stay blocked
			D::Precheck p;
			p.Record(ext, 5, { true, false }, { true, true }, { "none", "none" });   // passed without checked must not count
			const char* b1 = p.Block(1);
			Check(p.Block(0) == nullptr && b1 && std::string(b1).find("round 1 was not checked") != std::string::npos, "grassprobe precheck per round",
				std::string("round 0 allowed, round 1: ") + (b1 ? b1 : "allowed"));
			D::Precheck q;
			q.Record(ext, 6, { true, true }, { true, false }, { "none", "cull" });
			const char* c1 = q.Block(1);
			Check(q.Block(0) == nullptr && c1 && std::string(c1) == "state differs (round 1: cull) on normal comparison #6", "grassprobe precheck failed round",
				std::string("a checked round with a differing field stays blocked: ") + (c1 ? c1 : "allowed"));
			D::Precheck none;
			Check(none.Block(0) != nullptr && none.Block(1) != nullptr && q.Block(2) != nullptr, "grassprobe precheck failure", "no record or a round out of range blocks");
		}
		{   // context changes: the setting, an interior cell, exterior <-> interior, another worldspace; a walk across exterior cells keeps it
			D::Precheck p;
			p.Record(ext, 7, { true, true }, { true, true }, { "none", "none" });
			const bool sameWorld = !p.Invalidate({ 2, true, &worldA }) && p.Block(1) == nullptr;
			const bool modeHop = p.Invalidate({ 0, true, &worldA }) && p.Block(0) != nullptr && !p.Invalidate({ 2, true, &worldA }) && p.Block(0) != nullptr;
			p.Record(ext, 9, { true, true }, { true, true }, { "none", "none" });
			const bool otherWorld = p.Invalidate({ 2, true, &worldB }) && p.Block(0) != nullptr;
			p.Record(ext, 10, { true, true }, { true, true }, { "none", "none" });
			const bool toInterior = p.Invalidate({ 2, false, &cellA }) && p.Block(0) != nullptr;
			p.Record({ 2, false, &cellA }, 11, { true, true }, { true, true }, { "none", "none" });
			const bool otherCell = p.Invalidate({ 2, false, &cellB }) && p.Block(0) != nullptr;
			p.Record({ 2, false, &cellB }, 12, { true, true }, { true, true }, { "none", "none" });
			const bool backOut = p.Invalidate(ext) && p.Block(1) != nullptr;
			Check(sameWorld && modeHop && otherWorld && toInterior && otherCell && backOut, "grassprobe precheck invalidation",
				"same worldspace keeps it; 2 -> 0 -> 2, another worldspace, into an interior, another interior cell and back outside drop it until a new record");
		}
	}

	void PutRel(std::vector<std::uint8_t>& a_code, std::size_t a_at, std::uintptr_t a_base, std::uintptr_t a_target, std::uint8_t a_op)
	{
		const auto rel = static_cast<std::int32_t>(static_cast<std::intptr_t>(a_target) - static_cast<std::intptr_t>(a_base + a_at + 5));
		a_code[a_at] = a_op;
		std::memcpy(&a_code[a_at + 1], &rel, 4);
	}

	void TestJumpScan()
	{
		constexpr std::uintptr_t base = 0x140001000;
		constexpr std::uintptr_t target = 0x140123450;
		std::vector<std::uint8_t> code(48, 0x90);
		PutRel(code, 3, base, target, 0xE9);
		PutRel(code, 12, base, target, 0xE8);
		PutRel(code, 43, base, target, 0xE9);   // last full instruction
		const auto jumps = CallScan::FindJumpsTo(code, base, target);
		const auto calls = CallScan::FindCallsTo(code, base, target);
		Check(jumps.size() == 2 && jumps[0].offset == 3 && jumps[1].offset == 43 && calls.size() == 1 && calls[0].offset == 12, "grassprobe jump scan",
			"E9 at 3 and 43 (end), the E8 at 12 only as a call");
		Check(CallScan::FindJumpsTo({}, base, target).empty(), "grassprobe jump scan empty", "no code, no hit");
	}

	// The chain GrassProbe.cpp runs in game: a D24S8 depth target like the game's main depth, an identical-description
	// copy, a typeless staging texture, CopyResource twice, Map, decode with the stencil masked off.
	void TestWarpCopy()
	{
		ID3D11Device*        device = nullptr;
		ID3D11DeviceContext* ctx = nullptr;
		if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &ctx))) {
			Check(false, "grassprobe warp copy", "no WARP device");
			return;
		}
		D3D11_TEXTURE2D_DESC desc{};
		desc.Width = 64;
		desc.Height = 32;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.Format = DXGI_FORMAT_R24G8_TYPELESS;
		desc.SampleDesc.Count = 1;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
		ID3D11Texture2D*        src = nullptr;
		ID3D11Texture2D*        copy = nullptr;
		ID3D11Texture2D*        stage = nullptr;
		ID3D11DepthStencilView* dsv = nullptr;
		D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
		dsvDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
		dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		D3D11_TEXTURE2D_DESC stageDesc = desc;
		stageDesc.Usage = D3D11_USAGE_STAGING;
		stageDesc.BindFlags = 0;
		stageDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		bool        ok = SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &src)) && SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &copy)) &&
		                 SUCCEEDED(device->CreateTexture2D(&stageDesc, nullptr, &stage)) && SUCCEEDED(device->CreateDepthStencilView(src, &dsvDesc, &dsv));
		std::string detail = ok ? "" : "resource creation failed";
		if (ok) {
			ctx->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 0.25f, 7);
			ctx->OMSetRenderTargets(0, nullptr, dsv);   // still bound while copied, as in game
			ctx->CopyResource(copy, src);
			ctx->CopyResource(stage, copy);
			D3D11_MAPPED_SUBRESOURCE map{};
			ok = SUCCEEDED(ctx->Map(stage, 0, D3D11_MAP_READ, 0, &map));
			if (ok) {
				const auto*   texel = static_cast<const std::uint8_t*>(map.pData) + 5 * map.RowPitch + 9 * 4;
				std::uint32_t raw{};
				std::memcpy(&raw, texel, 4);
				const float d = M::DecodeDepth(M::CodecOf(desc.Format), texel);
				ok = std::fabs(d - 0.25f) < 1e-6f && (raw >> 24) == 7;
				detail = "depth " + std::to_string(d) + " stencil " + std::to_string(raw >> 24);
				ctx->Unmap(stage, 0);
			} else {
				detail = "Map failed";
			}
			ctx->OMSetRenderTargets(0, nullptr, nullptr);
		}
		Check(ok, "grassprobe warp copy", detail + " (R24G8 typeless: identical copy, staging, Map)");
		if (dsv) dsv->Release();
		if (stage) stage->Release();
		if (copy) copy->Release();
		if (src) src->Release();
		ctx->Release();
		device->Release();
	}

	// Probe v2's chain on the game's actual depth format (19, R32G8X24_TYPELESS, seen in game with view format 21):
	// a D32_FLOAT_S8X24 target written by a clear, copied straight into a staging texture while bound, Map, the first
	// float of the 8-byte texel read as the depth and byte 4 as the stencil. WARP is not the user's driver: this checks
	// the layout the code assumes (D3D11's 64-bit element), not what a given driver does.
	void TestWarpCopyF32S8()
	{
		ID3D11Device*        device = nullptr;
		ID3D11DeviceContext* ctx = nullptr;
		if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &ctx))) {
			Check(false, "grassprobe warp copy f32s8", "no WARP device");
			return;
		}
		D3D11_TEXTURE2D_DESC desc{};
		desc.Width = 64;
		desc.Height = 32;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
		desc.SampleDesc.Count = 1;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
		D3D11_TEXTURE2D_DESC stageDesc = desc;
		stageDesc.Usage = D3D11_USAGE_STAGING;
		stageDesc.BindFlags = 0;
		stageDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
		dsvDesc.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
		dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
		srvDesc.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MipLevels = 1;
		ID3D11Texture2D*          src = nullptr;
		ID3D11Texture2D*          stage = nullptr;
		ID3D11DepthStencilView*   dsv = nullptr;
		ID3D11ShaderResourceView* srv = nullptr;
		bool ok = SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &src)) && SUCCEEDED(device->CreateTexture2D(&stageDesc, nullptr, &stage)) &&
		          SUCCEEDED(device->CreateDepthStencilView(src, &dsvDesc, &dsv)) && SUCCEEDED(device->CreateShaderResourceView(src, &srvDesc, &srv));
		std::string detail = ok ? "" : "resource creation failed";
		if (ok) {
			ctx->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 0.25f, 7);
			ctx->OMSetRenderTargets(0, nullptr, dsv);
			ctx->CopyResource(stage, src);
			D3D11_MAPPED_SUBRESOURCE map{};
			ok = SUCCEEDED(ctx->Map(stage, 0, D3D11_MAP_READ, 0, &map));
			if (ok) {
				const auto codec = M::CodecOf(desc.Format);
				const auto* texel = static_cast<const std::uint8_t*>(map.pData) + 5 * map.RowPitch + 9 * M::TexelBytes(codec);
				const float d = M::DecodeDepth(codec, texel);
				ok = codec == M::Codec::kF32S8 && d == 0.25f && texel[4] == 7 && M::KeyToDepth(codec, M::RawKey(codec, texel)) == 0.25f && map.RowPitch >= 64u * 8u;
				detail = "depth " + std::to_string(d) + " stencil byte " + std::to_string(texel[4]) + " row pitch " + std::to_string(map.RowPitch);
				ctx->Unmap(stage, 0);
			} else {
				detail = "Map failed";
			}
			ctx->OMSetRenderTargets(0, nullptr, nullptr);
		}
		Check(ok, "grassprobe warp copy f32s8", detail + " (R32G8X24 typeless, D32S8 view, view 21 creatable, direct staging copy, Map)");
		if (srv) srv->Release();
		if (dsv) dsv->Release();
		if (stage) stage->Release();
		if (src) src->Release();
		ctx->Release();
		device->Release();
	}
}

int main()
{
	TestLinearDepth();
	TestBands();
	TestDecode();
	TestAccumulate();
	TestKeys();
	TestJumpScan();
	TestWarpCopy();
	TestKeyToDepth();
	TestRawStats();
	TestConvention();
	TestThresholds();
	TestDistance();
	TestGrassCounts();
	TestState();
	TestCadence();
	TestTracker();
	TestPrecheck();
	TestWarpCopyF32S8();
	std::printf("%s (%d failures)\n", g_fail == 0 ? "ALL PASS" : "FAILED", g_fail);
	return g_fail == 0 ? 0 : 1;
}
