#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <dxgiformat.h>

// Pure helpers of the grass depth probe (GrassProbe.cpp), kept free of D3D so the offline check can test them
// (tools/offline/GrassProbeCheck.cpp). Depth here is the raw value in the depth buffer; distance is view-space z
// in game units, from the projection as stored by the game (row vectors: clip.z = z * _33 + _43, clip.w = z).
namespace GrassProbeMath
{
	// How a texel of the copied depth texture is laid out (from its DXGI format; GrassProbe.cpp maps them).
	enum class Codec : std::uint8_t
	{
		kNone,
		kD24S8,    // R24G8_TYPELESS / D24_UNORM_S8_UINT: depth in the low 24 bits, stencil in the high 8
		kF32,      // R32_TYPELESS / D32_FLOAT
		kF32S8,    // R32G8X24_TYPELESS / D32_FLOAT_S8X24_UINT: float depth first, then stencil and padding
		kD16       // R16_TYPELESS / D16_UNORM
	};

	inline Codec CodecOf(DXGI_FORMAT a_format)
	{
		switch (a_format) {
		case DXGI_FORMAT_R24G8_TYPELESS:
		case DXGI_FORMAT_D24_UNORM_S8_UINT:
		case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
			return Codec::kD24S8;
		case DXGI_FORMAT_R32_TYPELESS:
		case DXGI_FORMAT_D32_FLOAT:
		case DXGI_FORMAT_R32_FLOAT:
			return Codec::kF32;
		case DXGI_FORMAT_R32G8X24_TYPELESS:
		case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
		case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
			return Codec::kF32S8;
		case DXGI_FORMAT_R16_TYPELESS:
		case DXGI_FORMAT_D16_UNORM:
		case DXGI_FORMAT_R16_UNORM:
			return Codec::kD16;
		default:
			return Codec::kNone;
		}
	}

	inline std::uint32_t TexelBytes(Codec a_codec)
	{
		switch (a_codec) {
		case Codec::kD24S8:
		case Codec::kF32:
			return 4;
		case Codec::kF32S8:
			return 8;
		case Codec::kD16:
			return 2;
		default:
			return 0;
		}
	}

	// Depth of one texel; the stencil bits are dropped, so a stencil write after the snapshot does not count as a change.
	inline float DecodeDepth(Codec a_codec, const std::uint8_t* a_texel)
	{
		switch (a_codec) {
		case Codec::kD24S8:
			{
				std::uint32_t v{};
				std::memcpy(&v, a_texel, sizeof(v));
				return static_cast<float>(v & 0xFFFFFFu) / 16777215.0f;
			}
		case Codec::kF32:
		case Codec::kF32S8:
			{
				float v{};
				std::memcpy(&v, a_texel, sizeof(v));
				return v;
			}
		case Codec::kD16:
			{
				std::uint16_t v{};
				std::memcpy(&v, a_texel, sizeof(v));
				return static_cast<float>(v) / 65535.0f;
			}
		default:
			return 0.0f;
		}
	}

	inline constexpr float kNoDistance = 1e9f;

	// Same formula as the raymarch shader's LinearDepth (RaymarchShaders.h): z = _43 / (d - _33).
	inline float LinearDepth(float a_depth, float a_p33, float a_p43)
	{
		const float denom = a_depth - a_p33;
		return std::fabs(denom) > 1e-7f ? a_p43 / denom : kNoDistance;
	}

	// Depth grows with distance when _43 < 0 (the game's standard projection); a reversed projection has _43 > 0.
	inline bool DepthGrows(float a_p43) { return a_p43 < 0.0f; }

	// The far-value thresholds (the cleared value: 1 for a growing depth, 0 for a reversed one). 0.9999999f is the float
	// 1 - 2^-23, so D24 0xFFFFFE and 0xFFFFFD also count: beyond any far plane the game uses.
	inline bool IsFarValue(float a_depth, bool a_grows) { return a_grows ? a_depth >= 0.9999999f : a_depth <= 1e-7f; }

	// Comparable key of one texel: the depth bits with the stencil dropped, ordered like the depth (D24 and D16 as
	// unsigned integers; a float depth's bit pattern orders like the value for values >= 0, and a negative one, which no
	// depth buffer holds, reads as 0). Two texels of the same codec are the same depth exactly when their keys match.
	inline std::uint32_t RawKey(Codec a_codec, const std::uint8_t* a_texel)
	{
		switch (a_codec) {
		case Codec::kD24S8:
			{
				std::uint32_t v{};
				std::memcpy(&v, a_texel, sizeof(v));
				return v & 0xFFFFFFu;
			}
		case Codec::kF32:
		case Codec::kF32S8:
			{
				std::uint32_t v{};
				std::memcpy(&v, a_texel, sizeof(v));
				return (v & 0x80000000u) ? 0u : v;
			}
		case Codec::kD16:
			{
				std::uint16_t v{};
				std::memcpy(&v, a_texel, sizeof(v));
				return v;
			}
		default:
			return 0;
		}
	}

	// Key on the D24 grid, for comparing two textures of different codecs (the coarser D24 step decides "same").
	inline std::uint32_t QuantKey24(float a_depth)
	{
		const float d = std::clamp(a_depth, 0.0f, 1.0f);
		return static_cast<std::uint32_t>(std::lround(static_cast<double>(d) * 16777215.0));
	}

	// Key a is nearer the eye than key b (keys of the same kind).
	inline bool NearerKey(std::uint32_t a_a, std::uint32_t a_b, bool a_grows) { return a_grows ? a_a < a_b : a_a > a_b; }

	// Distance bands: 0-600, 600-1024, 1024-2000, 2000-6000, beyond, the sky (the far value) and "bad" (negative, NaN
	// or no distance: what a wrong convention produces, kept apart instead of folded into 6000+). The 1024 edge is
	// Community Shaders' Terrain Blending zone (TerrainBlending.cpp: terrain within 1024 units of the eye is drawn
	// into its own depth during the prepass).
	inline constexpr std::array<float, 4> kBandEdges{ 600.0f, 1024.0f, 2000.0f, 6000.0f };
	inline constexpr int                  kBands = 7;
	inline constexpr int                  kFarBand = 4;
	inline constexpr int                  kSkyBand = 5;
	inline constexpr int                  kBadBand = 6;
	inline constexpr std::array<const char*, kBands> kBandNames{ "0-600", "600-1024", "1024-2000", "2000-6000", "6000+", "sky", "bad" };

	inline int Band(float a_distance)
	{
		if (!(a_distance >= 0.0f) || a_distance >= kNoDistance) {   // negative, NaN or no distance
			return kBadBand;
		}
		for (int i = 0; i < static_cast<int>(kBandEdges.size()); ++i) {
			if (a_distance < kBandEdges[i]) {
				return i;
			}
		}
		return kFarBand;
	}

	// A differing pixel whose two distances are within this share of the final distance: a shifted surface, not
	// a blade standing in front of the ground behind it (Terrain Blending would show here).
	inline constexpr float kTinyGapShare = 0.01f;

	struct Counts
	{
		std::array<std::uint32_t, kBands> samples{};
		std::array<std::uint32_t, kBands> differ{};    // drawn after the snapshot (grass, NoShadow, LOD, sky ...)
		std::array<std::uint32_t, kBands> tinyGap{};   // of those: distances within kTinyGapShare
		std::array<std::uint32_t, kBands> preFar{};    // the snapshot still has the cleared value here
		std::array<std::uint32_t, kBands> preNearer{}; // the snapshot is NEARER than the final depth (should be 0)

		std::uint64_t Total(const std::array<std::uint32_t, kBands>& a_v) const
		{
			std::uint64_t sum = 0;
			for (const auto v : a_v) sum += v;
			return sum;
		}
	};

	// One sample: a_pre / a_final are the snapshot's and the final depth (decoded, for distances), a_keyPre / a_keyFinal
	// their keys (RawKey, or QuantKey24 across codecs), which alone decide "differ" and "nearer": bit-exact, so an
	// untouched pixel never counts and any depth write after the snapshot does.
	inline void Accumulate(Counts& a_counts, float a_pre, float a_final, std::uint32_t a_keyPre, std::uint32_t a_keyFinal, float a_p33, float a_p43)
	{
		const bool grows = DepthGrows(a_p43);
		const bool finalFar = IsFarValue(a_final, grows);
		const float zFinal = LinearDepth(a_final, a_p33, a_p43);
		const int  band = finalFar ? kSkyBand : Band(zFinal);
		++a_counts.samples[band];
		if (IsFarValue(a_pre, grows)) {
			++a_counts.preFar[band];
		}
		if (a_keyPre != a_keyFinal) {
			++a_counts.differ[band];
			const float zPre = LinearDepth(a_pre, a_p33, a_p43);
			if (!finalFar && std::fabs(zPre - zFinal) <= kTinyGapShare * std::fabs(zFinal)) {
				++a_counts.tinyGap[band];
			}
		}
		if (NearerKey(a_keyPre, a_keyFinal, grows)) {
			++a_counts.preNearer[band];
		}
	}

	// Percent for the log: 0 when there is nothing to divide by.
	inline double Percent(std::uint64_t a_part, std::uint64_t a_whole)
	{
		return a_whole ? 100.0 * static_cast<double>(a_part) / static_cast<double>(a_whole) : 0.0;
	}

	// ---- probe v2 (prepass snapshots, DEBUG-LEDGER.md 2026-11-15 H-probe-rev, H-pregrass-5/6) ----

	// Depth of a key (RawKey of the same codec); the inverse of RawKey with the stencil gone.
	inline float KeyToDepth(Codec a_codec, std::uint32_t a_key)
	{
		switch (a_codec) {
		case Codec::kD24S8:
			return static_cast<float>(a_key & 0xFFFFFFu) / 16777215.0f;
		case Codec::kF32:
		case Codec::kF32S8:
			{
				float v{};
				std::memcpy(&v, &a_key, sizeof(v));
				return v;
			}
		case Codec::kD16:
			return static_cast<float>(a_key & 0xFFFFu) / 65535.0f;
		default:
			return 0.0f;
		}
	}

	// Raw values of one depth texture as stored, before any convention is applied: range, exact 0 and exact 1 (the two
	// possible cleared values), and ten buckets of width 0.1 over [0, 1]. Values outside [0, 1] (NaN included) are only
	// counted in `outside`.
	inline constexpr int kHistBuckets = 10;

	struct RawStats
	{
		std::uint32_t                         n{ 0 };
		float                                 min{ 0.0f };
		float                                 max{ 0.0f };
		std::uint32_t                         atZero{ 0 };   // <= 1e-7: the far value of a reversed depth
		std::uint32_t                         atOne{ 0 };    // >= 0.9999999: the far value of a growing depth
		std::uint32_t                         outside{ 0 };
		std::array<std::uint32_t, kHistBuckets> hist{};

		void Add(float a_raw)
		{
			if (!(a_raw >= 0.0f && a_raw <= 1.0f)) {
				++outside;
				return;
			}
			min = n == 0 ? a_raw : std::min(min, a_raw);
			max = n == 0 ? a_raw : std::max(max, a_raw);
			++n;
			if (a_raw <= 1e-7f) ++atZero;
			if (a_raw >= 0.9999999f) ++atOne;
			hist[static_cast<std::size_t>(std::min(static_cast<int>(a_raw * 10.0f), kHistBuckets - 1))] += 1;
		}
	};

	// Which way the stored depth runs, decided from the data before the projection is trusted.
	//   kWrites: a depth-tested draw only replaces a stored value with a nearer one, so the sign of the changes a draw
	//            made (snapshot right before -> right after it) is the stored encoding's "nearer" direction, whatever
	//            the projection constants say and however the CPU decodes the texel. Needs kMinWriteVotes changed
	//            samples and kWriteShare of them one way (a draw with the test off moves values both ways: no decision).
	//   kClear:  the cleared value survives where nothing was drawn (sky): only exact 1 seen -> grows, only exact 0
	//            seen -> reversed, each needing kClearShare of the samples.
	//   kProjection: the sign of _43 (DepthGrows), the convention v1 assumed.
	enum class Rule : std::uint8_t
	{
		kWrites,
		kClear,
		kProjection
	};

	inline constexpr std::uint64_t kMinWriteVotes = 256;
	inline constexpr double        kWriteShare = 0.99;
	inline constexpr double        kClearShare = 0.01;

	struct Convention
	{
		bool grows{ true };
		Rule rule{ Rule::kProjection };
	};

	// a_smaller / a_larger: changed samples whose key went down / up across a draw; a_atZero / a_atOne / a_samples:
	// from RawStats of the final depth.
	inline Convention DecideConvention(std::uint64_t a_smaller, std::uint64_t a_larger, std::uint64_t a_atZero, std::uint64_t a_atOne,
		std::uint64_t a_samples, float a_p43)
	{
		const std::uint64_t votes = a_smaller + a_larger;
		if (votes >= kMinWriteVotes) {
			if (static_cast<double>(a_smaller) >= kWriteShare * static_cast<double>(votes)) return { true, Rule::kWrites };
			if (static_cast<double>(a_larger) >= kWriteShare * static_cast<double>(votes)) return { false, Rule::kWrites };
		}
		if (a_samples > 0) {
			const double need = kClearShare * static_cast<double>(a_samples);
			if (static_cast<double>(a_atOne) >= need && a_atZero == 0) return { true, Rule::kClear };
			if (static_cast<double>(a_atZero) >= need && a_atOne == 0) return { false, Rule::kClear };
		}
		return { DepthGrows(a_p43), Rule::kProjection };
	}

	inline const char* RuleName(Rule a_rule)
	{
		return a_rule == Rule::kWrites ? "writes" : a_rule == Rule::kClear ? "clear" : "projection";
	}

	// View distance of a raw value once the convention is known. a_flip: the data runs the other way than the
	// projection (DecideConvention's grows != DepthGrows(_43)); the value is then read as 1 - raw, which for a
	// perspective projection is exactly the depth of the same distance under the flipped near/far mapping:
	// 1 - (_33 + _43 / z) is the reversed-Z value of z with the same planes. Double precision, because 1 - raw of a far
	// reversed value loses the digits that matter in float.
	inline float Distance(float a_raw, bool a_flip, float a_p33, float a_p43)
	{
		const double d = a_flip ? 1.0 - static_cast<double>(a_raw) : static_cast<double>(a_raw);
		const double denom = d - static_cast<double>(a_p33);
		return std::fabs(denom) > 1e-12 ? static_cast<float>(static_cast<double>(a_p43) / denom) : kNoDistance;
	}

	// Band of a final-depth sample under a decided convention (sky = the far value of that convention).
	inline int FinalBand(float a_raw, bool a_grows, bool a_flip, float a_p33, float a_p43)
	{
		return IsFarValue(a_raw, a_grows) ? kSkyBand : Band(Distance(a_raw, a_flip, a_p33, a_p43));
	}

	// One grass call seen through two snapshots (key right before it, right after it) and the final depth's key.
	// wrote: the call changed the sample; visible: the value it left is bit-equal to the final one; nearEqual: not
	// bit-equal, but the final distance is within kTinyGapShare of the grass's (the main pass drew the same blade again
	// with a slightly different value, which is not "covered"). Bands by the final depth and, *Own, by the distance of
	// the value the grass left. smaller / larger: direction of the change, the kWrites votes.
	struct GrassCounts
	{
		std::array<std::uint32_t, kBands> wrote{};
		std::array<std::uint32_t, kBands> visible{};
		std::array<std::uint32_t, kBands> nearEqual{};
		std::array<std::uint32_t, kBands> wroteOwn{};
		std::array<std::uint32_t, kBands> visibleOwn{};
		std::uint64_t                     smaller{ 0 };
		std::uint64_t                     larger{ 0 };
	};

	// Votes only (the band is not known until the convention is decided).
	inline void VoteWrite(GrassCounts& a_c, std::uint32_t a_keyBefore, std::uint32_t a_keyAfter)
	{
		if (a_keyAfter < a_keyBefore) ++a_c.smaller;
		else if (a_keyAfter > a_keyBefore) ++a_c.larger;
	}

	// Near-equal test of two distances (both must be real distances).
	inline bool NearEqual(float a_zAfter, float a_zFinal)
	{
		if (!(a_zFinal > 0.0f) || a_zFinal >= kNoDistance || !(a_zAfter > 0.0f) || a_zAfter >= kNoDistance) return false;
		return std::fabs(a_zAfter - a_zFinal) <= kTinyGapShare * a_zFinal;
	}

	// Band counts; returns true when the call wrote the sample. a_zAfter / a_zFinal: distances of the value the grass
	// left and of the final value.
	inline bool AddGrass(GrassCounts& a_c, int a_bandFinal, int a_bandOwn, std::uint32_t a_keyBefore, std::uint32_t a_keyAfter, std::uint32_t a_keyFinal,
		float a_zAfter, float a_zFinal)
	{
		if (a_bandFinal < 0 || a_bandFinal >= kBands || a_bandOwn < 0 || a_bandOwn >= kBands || a_keyBefore == a_keyAfter) {
			return false;
		}
		++a_c.wrote[a_bandFinal];
		++a_c.wroteOwn[a_bandOwn];
		if (a_keyAfter == a_keyFinal) {
			++a_c.visible[a_bandFinal];
			++a_c.visibleOwn[a_bandOwn];
		} else if (NearEqual(a_zAfter, a_zFinal)) {
			++a_c.nearEqual[a_bandFinal];
		}
		return true;
	}

	// Grass mask image values (also in the PGM comment line and the dump log line).
	inline constexpr std::uint8_t kMaskNoGrass = 0;
	inline constexpr std::uint8_t kMaskFinalFar = 32;       // the final depth is the far value (sky); no grass wrote
	inline constexpr std::uint8_t kMaskGrassCovered = 96;   // grass wrote, the final value is another, more than 1 % off
	inline constexpr std::uint8_t kMaskGrassNear = 160;     // grass wrote, the final value is within 1 % of the grass's
	inline constexpr std::uint8_t kMaskGrassVisible = 255;  // grass wrote and its value is still the final one (bit-equal)

	inline std::uint8_t MaskValue(bool a_wrote, bool a_visible, bool a_nearEqual, bool a_finalFar)
	{
		if (a_wrote) return a_visible ? kMaskGrassVisible : a_nearEqual ? kMaskGrassNear : kMaskGrassCovered;
		return a_finalFar ? kMaskFinalFar : kMaskNoGrass;
	}

	// The pixel the fog's [PROBE] compute shader reads for grid cell a_i of a_n over a_size pixels (RaymarchShaders.h
	// kProbeCS: int((t + 0.5) / n * size), 9 x 7 cells), so a CPU sample can be put on the same pixel.
	inline int ProbeGridPixel(int a_i, int a_n, float a_size)
	{
		if (a_n <= 0 || !(a_size > 0.0f)) return 0;
		return static_cast<int>((static_cast<float>(a_i) + 0.5f) / static_cast<float>(a_n) * a_size);
	}

	// Comparison cadence (a_index: 1-based count of issued comparisons, including the one being decided).
	// [Debug] GrassProbe = 2: every second (even) comparison defers the prepass grass calls.
	inline bool DeferredComparison(int a_mode, std::uint64_t a_index) { return a_mode == 2 && a_index > 0 && a_index % 2 == 0; }
	// Dumps and late [PROBE] requests: every 6th comparison in mode 1. In mode 2 every 6th is always even (deferred), so
	// there they fall on 6, 13, 18, 25, 30, 37 ... (index mod 12 = 6 or 1): deferred and normal frames alternate.
	inline constexpr std::uint64_t kDumpEvery = 6;
	inline constexpr std::uint64_t kDumpPairs = 8;
	inline bool CadenceDue(int a_mode, std::uint64_t a_index)
	{
		if (a_index == 0) return false;
		if (a_mode != 2) return a_index % kDumpEvery == 0;
		return a_index % (2 * kDumpEvery) == kDumpEvery || (a_index > 1 && a_index % (2 * kDumpEvery) == 1);
	}
	inline bool DumpDue(int a_mode, std::uint64_t a_index, bool a_exterior) { return a_exterior && CadenceDue(a_mode, a_index); }
	// Fog [PROBE] readback requested on every comparison until kFogProbesFirst of them were submitted (indoors the fog
	// does not draw, so the first comparisons may give none), then on the cadence above. a_submitted: [PROBE] readbacks
	// submitted at comparison calls so far.
	inline constexpr std::uint64_t kFogProbesFirst = 3;
	inline bool FogProbeDue(int a_mode, std::uint64_t a_index, std::uint64_t a_submitted)
	{
		return a_index > 0 && (a_submitted < kFogProbesFirst || CadenceDue(a_mode, a_index));
	}
	// Ring slot of the a_written-th dump pair (0-based count of pairs written before): the oldest is overwritten.
	inline std::uint64_t DumpSlot(std::uint64_t a_written) { return a_written % kDumpPairs; }

	// Render state a depth-only grass draw depends on, from the game's state cache (RendererShadowState, CommonLibSSE-NG
	// RendererShadowState.h offsets 0x88-0xb8): compared between the grass call's place and the replay point before
	// mode 2 may move the call. depthBias is not in the review's list but decals set it, so it is compared too.
	struct StateSet
	{
		bool          valid{ false };
		std::uint32_t depthMode{ 0 };
		std::uint32_t stencilMode{ 0 };
		std::uint32_t cull{ 0 };
		std::uint32_t depthBias{ 0 };
		std::uint32_t blend{ 0 };
		std::uint32_t alphaToCoverage{ 0 };
		bool          alphaTest{ false };
		float         alphaRef{ 0.0f };
	};

	inline constexpr int                                   kStateFields = 8;
	inline constexpr std::array<const char*, kStateFields> kStateNames{ "depthMode", "stencilMode", "cull", "depthBias", "blend", "alphaToCoverage",
		"alphaTest", "alphaRef" };
	inline constexpr std::uint32_t kStateInvalid = 1u << 31;   // one of the sets was not recorded

	// Bit i set: field kStateNames[i] differs (alphaRef bit-exact); kStateInvalid when either set is missing.
	inline std::uint32_t StateDiff(const StateSet& a_a, const StateSet& a_b)
	{
		if (!a_a.valid || !a_b.valid) return kStateInvalid;
		std::uint32_t ra{};
		std::uint32_t rb{};
		std::memcpy(&ra, &a_a.alphaRef, sizeof(ra));
		std::memcpy(&rb, &a_b.alphaRef, sizeof(rb));
		const bool diff[kStateFields]{ a_a.depthMode != a_b.depthMode, a_a.stencilMode != a_b.stencilMode, a_a.cull != a_b.cull,
			a_a.depthBias != a_b.depthBias, a_a.blend != a_b.blend, a_a.alphaToCoverage != a_b.alphaToCoverage, a_a.alphaTest != a_b.alphaTest, ra != rb };
		std::uint32_t mask = 0;
		for (int i = 0; i < kStateFields; ++i) {
			if (diff[i]) mask |= 1u << i;
		}
		return mask;
	}
}
