#pragma once

// Where fog collects, per height-grid cell (pure, tested offline): over lakes, rivers and marsh water, on marsh land,
// near shores, in valleys (cold air pools at the bottom); steam over hot springs; sea smoke over the sea when the air
// is cold. Sources as FogConditions.h (radiation fog favours swampy ground and valleys; steam fog over warm water).
//
// Output per cell: x = multiplier on radiation fog (scaled by FogConditions global), y = hot-spring steam (always),
// z = steam over open water (sea smoke, river smoke; scaled by FogConditions cold), w = pooled-fog depth (units) to a
// level lid: the terrain's closing (window max, then window min) fills hollows narrower than the window up to their
// rim, capped at the basin floor + poolMaxDepth, and lowered toward the floor by poolFill (NatGeo: valley fog is
// trapped in the bowl of the valley; photos 2026-11-08: lake and valley fog fill low ground with a near-level top).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

namespace PlacementRules
{
	enum class Kind : std::uint8_t
	{
		kLand = 0,
		kMarshLand,
		kLake,
		kRiver,
		kMarshWater,
		kHotSpring,
		kSea,
		kCount
	};

	// Pool density after the visibility floor. The pool's extinction per metre at the ground is
	// a_opacity x density x a / (1 - exp(-a)): the ground density of the tall layer's profile rho(h) = rho0 exp(-a h/H)
	// relative to its mean W / H (RaymarchShaders ProfileDensity; a held to >= 1e-3 as there, -> 1 for an even slab).
	// Visibility (5% contrast, Koschmieder) is 3 / extinction. Fog weathers multiply the opacity by up to 2.5, which took
	// the pools to 12-25 m; dense radiation fog is 50-200 m. a_minVisibility <= 0 switches the floor off.
	inline float LimitPoolDensity(float a_density, float a_opacity, float a_steepness, float a_minVisibility)
	{
		if (!(a_minVisibility > 0.0f) || !(a_opacity > 0.0f)) return a_density;
		const float a = std::max(a_steepness, 1e-3f);
		const float ground = a / -std::expm1(-a);
		return std::min(a_density, 3.0f / a_minVisibility / (a_opacity * ground));
	}

	inline const char* KindName(Kind a_k)
	{
		constexpr const char* names[] = { "land", "marsh land", "lake", "river", "marsh water", "hot spring", "sea" };
		const auto i = static_cast<std::size_t>(a_k);
		return i < std::size(names) ? names[i] : "?";
	}

	inline std::string LowerCopy(std::string_view a_s)
	{
		std::string out(a_s);
		for (auto& c : out) c = static_cast<char>((c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c);
		return out;
	}

	inline bool HasAny(const std::string& a_lower, std::initializer_list<const char*> a_words)
	{
		for (const char* w : a_words) {
			if (a_lower.find(w) != std::string::npos) return true;
		}
		return false;
	}

	// Water body from the cell's water type EditorID (vanilla, Water for ENB, Cathedral/Shades names) and its height.
	// Unnamed or default water at sea level is the sea; elsewhere a lake.
	inline Kind WaterKind(std::string_view a_waterEditorID, float a_waterZ, float a_seaLevel, float a_seaTolerance = 300.0f)
	{
		const std::string id = LowerCopy(a_waterEditorID);
		if (HasAny(id, { "volcanic", "hotspring", "hot spring", "geyser", "steam" })) return Kind::kHotSpring;
		if (HasAny(id, { "marsh", "murky", "morthal", "swamp", "bog", "fen" })) return Kind::kMarshWater;
		if (HasAny(id, { "river", "creek", "stream", "flow" })) return Kind::kRiver;
		if (HasAny(id, { "lake", "pond", "honrich", "geir", "ilinalta", "still" })) return Kind::kLake;
		if (std::isfinite(a_waterZ) && std::isfinite(a_seaLevel) && std::fabs(a_waterZ - a_seaLevel) <= a_seaTolerance) return Kind::kSea;
		return Kind::kLake;
	}

	// "mud" alone is not marsh: LRiverMud01 is the bed of every river and pond, and it is the base texture of whole
	// landscape quadrants (2048 x 2048 units) around them, dry banks included. With it the Falkreath mill pond's
	// quadrant took the marsh-land multiplier and showed as a square of denser fog with straight edges on the cell
	// quadrant's borders (in-game run 2026-11-13, LAND 00AC5F top-right quadrant, DEBUG-LEDGER.md). Marsh mud textures
	// still match by "marsh" or "swamp" (RP_LMarshMud01, CYRLSwampMud01).
	inline bool IsMarshLand(std::string_view a_landTextureEditorID)
	{
		return HasAny(LowerCopy(a_landTextureEditorID), { "marsh", "swamp", "bog" });
	}

	struct Params
	{
		float land{ 0.7f };           // dry ground (photos 2026-11-08: fog gathers over water, marsh and hollows, far less on dry land)
		float lake{ 1.8f };
		float river{ 1.4f };
		float marshWater{ 2.2f };
		float marshLand{ 2.0f };
		float hotSpring{ 1.5f };      // steam, added regardless of time
		float seaRadiation{ 1.2f };   // the sea as a radiation-fog surface
		float seaSteam{ 1.5f };       // sea smoke at full cold
		float freshSteam{ 1.0f };     // steam over lakes, rivers and marsh water at full cold ("river smoke")
		float shoreFactor{ 1.5f };    // land right at the water's edge ...
		float shoreDistance{ 700.0f };// ... fading to 1 over this many units (0 = no shore effect)
		bool  bankCap{ false };       // land at the water's edge held to that water's value, the limit fading out across the band (see Build); false = as before
		float valleyStrength{ 0.35f };// per valleyScale units below the local mean height
		float valleyScale{ 150.0f };  // units
		float valleyRadius{ 2048.0f };// units; the neighbourhood whose mean height a cell is compared with
		float poolRadius{ 1500.0f };  // units per closing pass (the fill reaches 2x this); 0 = no pools
		float poolMaxDepth{ 280.0f }; // units above the basin floor
		float poolFill{ 1.0f };       // 0..1: lid height between the basin floor (0) and the rim (1); from the conditions
		float minMultiplier{ 0.5f };  // limits of the terrain part (shore x valley); the kind multiplier is kept as set
		float maxMultiplier{ 3.0f };
	};

	// Sliding-window maximum (a_max) or minimum over a (2r+1)^2 square, separable, O(n) with monotonic deques. Cells
	// equal to a_absent are ignored; a window with no other value gives a_absent. Windows are clipped at the edge.
	inline std::vector<float> WindowExtreme(const std::vector<float>& a_in, int a_size, int a_r, bool a_max, float a_absent)
	{
		const std::size_t n = static_cast<std::size_t>(a_size) * static_cast<std::size_t>(a_size);
		auto better = [&](float a, float b) { return a_max ? a >= b : a <= b; };
		auto pass = [&](const std::vector<float>& a_src, bool a_rows) {
			std::vector<float> dst(n, a_absent);
			std::deque<int>    q;
			for (int line = 0; line < a_size; ++line) {
				auto at = [&](int a_i) -> float {
					return a_src[a_rows ? static_cast<std::size_t>(line) * a_size + a_i : static_cast<std::size_t>(a_i) * a_size + line];
				};
				q.clear();
				int next = 0;
				for (int i = 0; i < a_size; ++i) {
					for (; next < a_size && next <= i + a_r; ++next) {
						const float v = at(next);
						if (v == a_absent) continue;
						while (!q.empty() && better(v, at(q.back()))) q.pop_back();
						q.push_back(next);
					}
					while (!q.empty() && q.front() < i - a_r) q.pop_front();
					const float out = q.empty() ? a_absent : at(q.front());
					dst[a_rows ? static_cast<std::size_t>(line) * a_size + i : static_cast<std::size_t>(i) * a_size + line] = out;
				}
			}
			return dst;
		};
		return pass(pass(a_in, true), false);
	}

	// Eases the pool depths (w) of a fresh Build toward it from the previous ones instead of jumping: a pool that turns
	// from closed to open as the grid moves changes its formula and its top (GPT review 2026-11-08). a_prev holds the
	// last eased depths at the previous grid position; the grid has since moved by (a_dx, a_dy) cells (new = old shifted,
	// as Terrain shifts its heights), so each depth stays on its world spot; cells new to the grid take the target.
	// a_alpha 0..1 is the share of the way taken now. Writes the eased depths into a_out's w and back into a_prev, and
	// returns the largest remaining |target - eased| (units).
	inline float EasePools(std::vector<float>& a_prev, int a_size, int a_dx, int a_dy, std::vector<std::array<float, 4>>& a_out, float a_alpha)
	{
		const std::size_t n = static_cast<std::size_t>(a_size) * static_cast<std::size_t>(a_size);
		if (a_out.size() != n) return 0.0f;
		std::vector<float> next(n);
		float              remaining = 0.0f;
		const bool         have = a_prev.size() == n;
		const float        alpha = std::clamp(a_alpha, 0.0f, 1.0f);
		for (int y = 0; y < a_size; ++y) {
			for (int x = 0; x < a_size; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * a_size + x;
				const float       target = a_out[i][3];
				const int         ox = x + a_dx, oy = y + a_dy;
				const bool        old = have && ox >= 0 && oy >= 0 && ox < a_size && oy < a_size;
				const float       from = old ? a_prev[static_cast<std::size_t>(oy) * a_size + ox] : target;
				next[i] = from + (target - from) * alpha;
				remaining = std::max(remaining, std::fabs(target - next[i]));
				a_out[i][3] = next[i];
			}
		}
		a_prev.swap(next);
		return remaining;
	}

	// a_ground: size*size heights (water surface where there is water; values <= -1e29 unknown), a_kinds same layout.
	inline std::vector<std::array<float, 4>> Build(const std::vector<float>& a_ground, const std::vector<Kind>& a_kinds, int a_size, float a_cell,
		const Params& a_p)
	{
		const std::size_t n = static_cast<std::size_t>(a_size) * static_cast<std::size_t>(a_size);
		std::vector<std::array<float, 4>> out(n, { 1.0f, 0.0f, 0.0f, 0.0f });
		if (a_size <= 0 || a_ground.size() != n || a_kinds.size() != n) {
			return out;
		}
		auto known = [&](std::size_t i) { return a_ground[i] > -1.0e29f; };
		auto water = [&](Kind k) { return k == Kind::kLake || k == Kind::kRiver || k == Kind::kMarshWater || k == Kind::kHotSpring || k == Kind::kSea; };

		// Valley = below the mean of the known land heights within valleyRadius (a fixed world neighbourhood, so a
		// spot keeps its value as the grid moves; GPT review 2026-11-08: the whole-window mean followed the player).
		// Summed-area tables of height and count: O(n). A neighbourhood less than half known is weighted down.
		const std::size_t s1 = static_cast<std::size_t>(a_size) + 1;
		std::vector<double> sumH(s1 * s1, 0.0), sumC(s1 * s1, 0.0), sumM(s1 * s1, 0.0);   // sumM: marsh land among them
		for (int y = 0; y < a_size; ++y) {
			for (int x = 0; x < a_size; ++x) {
				const std::size_t i = static_cast<std::size_t>(y) * a_size + x;
				const bool use = known(i) && !water(a_kinds[i]);
				const std::size_t j = (static_cast<std::size_t>(y) + 1) * s1 + x + 1;
				sumH[j] = (use ? a_ground[i] : 0.0) + sumH[j - 1] + sumH[j - s1] - sumH[j - s1 - 1];
				sumC[j] = (use ? 1.0 : 0.0) + sumC[j - 1] + sumC[j - s1] - sumC[j - s1 - 1];
				sumM[j] = (use && a_kinds[i] == Kind::kMarshLand ? 1.0 : 0.0) + sumM[j - 1] + sumM[j - s1] - sumM[j - s1 - 1];
			}
		}
		const int r = std::max(1, static_cast<int>(std::lround(a_p.valleyRadius / std::max(a_cell, 1.0f))));
		const double area = static_cast<double>(2 * r + 1) * (2 * r + 1);
		auto boxSum = [&](const std::vector<double>& a_t, int a_x0, int a_y0, int a_x1, int a_y1) {
			return a_t[static_cast<std::size_t>(a_y1) * s1 + a_x1] - a_t[static_cast<std::size_t>(a_y0) * s1 + a_x1] -
			       a_t[static_cast<std::size_t>(a_y1) * s1 + a_x0] + a_t[static_cast<std::size_t>(a_y0) * s1 + a_x0];
		};
		const double scale = std::max(a_p.valleyScale, 1.0f);
		// Marsh land comes per landscape quadrant (the base texture of 2048 x 2048 units), so its multiplier is blended
		// over kMarshBlend units: dry land takes the share of marsh land around it. A hard step drew the quadrant's
		// straight borders in the fog.
		constexpr float kMarshBlend = 512.0f;
		const int rm = std::max(1, static_cast<int>(std::lround(kMarshBlend / std::max(a_cell, 1.0f))));

		// Distance (cells, 8-neighbour steps) from each cell to the nearest water cell, up to the shore band.
		// With bankCap the search also carries the multiplier of that water (bankWater): a cell first reached takes the
		// value of the cell it was reached from, and reached again at the same distance it takes the larger one. Every
		// cell at one distance is taken off the queue before any cell one step further, so a cell's value is final
		// before it is passed on: bankWater = the LARGEST multiplier among the water cells at the smallest distance,
		// whatever the scan order. Where two kinds of water meet, the bank between them is held to the larger, so the
		// cap never takes land below what either neighbour's rule allows. Water further off than the nearest is not
		// looked at (a lake two cells behind a river does not lift the river's bank).
		const bool shore = a_p.shoreDistance > 0.0f;
		const int band = shore ? std::max(1, static_cast<int>(std::ceil(a_p.shoreDistance / std::max(a_cell, 1.0f)))) : 0;
		const bool bankCap = a_p.bankCap && shore;
		auto waterMultiplier = [&](Kind k) {   // as the switch at the end of Build
			switch (k) {
			case Kind::kRiver:      return a_p.river;
			case Kind::kMarshWater: return a_p.marshWater;
			case Kind::kSea:        return a_p.seaRadiation;
			default:                return a_p.lake;   // lake, hot spring
			}
		};
		std::vector<int> dist(n, -1);
		std::vector<float> bankWater(bankCap ? n : 0, 0.0f);
		std::deque<std::size_t> queue;
		for (std::size_t i = 0; i < n; ++i) {
			if (known(i) && water(a_kinds[i])) {
				dist[i] = 0;
				if (bankCap) bankWater[i] = waterMultiplier(a_kinds[i]);
				queue.push_back(i);
			}
		}
		while (shore && !queue.empty()) {
			const std::size_t i = queue.front();
			queue.pop_front();
			if (dist[i] >= band) continue;
			const int x = static_cast<int>(i % a_size), y = static_cast<int>(i / a_size);
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					const int nx = x + dx, ny = y + dy;
					if ((dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= a_size || ny >= a_size) continue;
					const std::size_t j = static_cast<std::size_t>(ny) * a_size + nx;
					if (dist[j] < 0) {
						dist[j] = dist[i] + 1;
						if (bankCap) bankWater[j] = bankWater[i];
						queue.push_back(j);
					} else if (bankCap && dist[j] == dist[i] + 1) {
						bankWater[j] = std::max(bankWater[j], bankWater[i]);
					}
				}
			}
		}

		// Pooled fog: closing L = min(max(ground)) with the window r fills every hollow narrower than the window to its
		// rim (flat or monotonic slopes stay unfilled); each connected pool's floor B (its lowest ground) caps its lid at
		// B + poolMaxDepth, and poolFill lowers the lid toward B, so the top stays level as the fog weakens (GPT design review
		// 2026-11-08: a local-mean lid follows the terrain in a bowl, and scaling the depth tilts the top). Unknown
		// heights take no part; beyond the grid nothing is known, so a rim outside it is missed (fewer pools there).
		// Limit: valleys wider than 2 x poolRadius, or wider than the grid, are not filled (upgrade: priority-flood).
		std::vector<float> pool(n, 0.0f);
		const int poolR = a_p.poolRadius > 0.0f ? std::max(1, static_cast<int>(std::lround(a_p.poolRadius / std::max(a_cell, 1.0f)))) : 0;
		if (poolR > 0 && a_p.poolMaxDepth > 0.0f) {
			// Worked on a grid padded by r on every side: the max pass runs out to the pad (from the data inside), so
			// the min pass near the edge sees the low values just outside it. Clipping both passes at the edge made a
			// false pool r cells deep at the downhill edge of a uniform slope.
			constexpr float kLow = -1.0e30f, kHigh = 1.0e30f;
			const int pad = poolR, ps = a_size + 2 * pad;
			const std::size_t pn = static_cast<std::size_t>(ps) * static_cast<std::size_t>(ps);
			std::vector<float> forMax(pn, kLow);
			for (int y = 0; y < a_size; ++y) {
				for (int x = 0; x < a_size; ++x) {
					const std::size_t i = static_cast<std::size_t>(y) * a_size + x;
					const std::size_t j = static_cast<std::size_t>(y + pad) * ps + (x + pad);
					if (known(i)) forMax[j] = a_ground[i];
				}
			}
			std::vector<float> dil = WindowExtreme(forMax, ps, poolR, true, kLow);
			for (auto& v : dil) v = v == kLow ? kHigh : v;
			const std::vector<float> lidPad = WindowExtreme(dil, ps, poolR, false, kHigh);
			std::vector<float> lid(n, kHigh);
			for (int y = 0; y < a_size; ++y) {
				for (int x = 0; x < a_size; ++x) {
					lid[static_cast<std::size_t>(y) * a_size + x] = lidPad[static_cast<std::size_t>(y + pad) * ps + (x + pad)];
				}
			}
			// One floor and one lid per closed pool (4-neighbour cells the closing fills by more than half a unit, none
			// on the grid border or next to an unknown height): per-cell floors stepped the top of one asymmetric basin
			// as soon as the cap or the fill applied (GPT review 2026-11-08). The floor is the pool's 10th-percentile
			// ground, so one deep cell (a river channel) cannot drag the whole lid down, and the lid its lowest closing
			// value. An open pool reaches out of what is known: its true floor and lid are unknown and, taken from its
			// cells, followed the grid (a descending valley's pool sat at the grid's downhill end and moved with the
			// player; reviewer 2026-11-08), so each of its cells takes its own floor and the closed pool's formula from
			// there. That floor is the lowest ground of the SAME pool within kOpenFloorReach x poolMaxDepth (units, held to
			// 2 x poolRadius), each cell's ground first replaced by the lower quartile of its 3x3 pool neighbours (the
			// 3rd lowest of 9):
			//  - capping the per-cell closing depth instead (until 2026-11-12) put the lid at ground + poolMaxDepth on
			//    every cell: in a long valley the closing lid is the lowest pass within the window, far above the floor,
			//    so the fog lay as a slab up both sides to that height (a bright line over the ridge, PoolBoxRepro);
			//  - the same pool only: low ground across a ridge is not this pool's floor (GPT review 2026-11-12);
			//  - the lower quartile: one deep cell does not pull the floor down (GPT review 2026-11-12: a pool lost its
			//    fog when one border cell turned unknown). A channel one cell wide does, as it does the closed pool's
			//    percentile; the median was tried and read the floor of a V one cell up its sides (reviewer 2026-11-12:
			//    a top 154 units above floor + cap on the coarse grid);
			//  - the short reach: where the window is clipped by the grid edge the floor can read high, so in a falling
			//    valley the depth there changes as the grid moves, by the fall over the distance moved. With the reach
			//    at 4 x poolMaxDepth that band is 18 fine cells (under CSColumn's edge weight, 0.6 to 0) or 3 coarse
			//    cells (CoarsePoolAt fades the outer 2); with 2 x poolRadius it was most of the grid.
			// Limits: on sides gentler than 1 / kOpenFloorReach (14 degrees) the lid still climbs with the ground; a
			// valley falling more steeply than that along its length keeps almost no pooled fog; a pit cell holds up to
			// 2 x poolMaxDepth, as in a closed pool. Cost: one window pass per open pool over its bounding square, so
			// pools x n at worst (21 pools across a 129 grid: 5-6 ms on the worker thread against 2 ms; upgrade: one
			// labelled pass over the whole grid).
			constexpr float kOpenFloorReach = 4.0f;
			constexpr float kMinFill = 0.5f;
			auto inPool = [&](std::size_t a_i) { return known(a_i) && lid[a_i] < kHigh && lid[a_i] - a_ground[a_i] > kMinFill; };
			const float fill = std::clamp(a_p.poolFill, 0.0f, 1.0f);
			const int floorR = std::clamp(static_cast<int>(std::ceil(kOpenFloorReach * a_p.poolMaxDepth / std::max(a_cell, 1.0f))), 1, 2 * poolR);
			std::vector<int>         label(n, -1);
			std::vector<std::size_t> members;
			std::vector<float>       heights;
			std::deque<std::size_t>  q;
			for (std::size_t s0 = 0; s0 < n; ++s0) {
				if (label[s0] >= 0 || !inPool(s0)) continue;
				members.clear();
				q.push_back(s0);
				label[s0] = static_cast<int>(s0);
				float lidC = kHigh;
				bool  open = false;
				heights.clear();
				while (!q.empty()) {
					const std::size_t i = q.front();
					q.pop_front();
					members.push_back(i);
					heights.push_back(a_ground[i]);
					lidC = std::min(lidC, lid[i]);
					const int x = static_cast<int>(i % a_size), y = static_cast<int>(i / a_size);
					const int nx[4] = { x - 1, x + 1, x, x }, ny[4] = { y, y, y - 1, y + 1 };
					for (int d = 0; d < 4; ++d) {
						if (nx[d] < 0 || ny[d] < 0 || nx[d] >= a_size || ny[d] >= a_size) {
							open = true;   // on the grid border
							continue;
						}
						const std::size_t j = static_cast<std::size_t>(ny[d]) * a_size + nx[d];
						if (!known(j)) open = true;
						if (label[j] >= 0 || !inPool(j)) continue;
						label[j] = static_cast<int>(s0);
						q.push_back(j);
					}
				}
				if (open) {
					// On the pool's bounding square (WindowExtreme takes a square); cells of other pools and dry ground absent.
					int x0 = a_size, y0 = a_size, x1 = -1, y1 = -1;
					for (const std::size_t i : members) {
						const int x = static_cast<int>(i % a_size), y = static_cast<int>(i / a_size);
						x0 = std::min(x0, x);
						y0 = std::min(y0, y);
						x1 = std::max(x1, x);
						y1 = std::max(y1, y);
					}
					const int          side = std::max(x1 - x0, y1 - y0) + 1;
					std::vector<float> sub(static_cast<std::size_t>(side) * side, kHigh);
					for (const std::size_t i : members) {
						const int x = static_cast<int>(i % a_size), y = static_cast<int>(i / a_size);
						float     near[9];
						int       count = 0;
						for (int dy = -1; dy <= 1; ++dy) {
							for (int dx = -1; dx <= 1; ++dx) {
								const int nx2 = x + dx, ny2 = y + dy;
								if (nx2 < 0 || ny2 < 0 || nx2 >= a_size || ny2 >= a_size) continue;
								const std::size_t j = static_cast<std::size_t>(ny2) * a_size + nx2;
								if (label[j] == static_cast<int>(s0)) near[count++] = a_ground[j];
							}
						}
						const int pick = (count - 1) / 4;   // lower quartile; the cell itself is always in, so count >= 1
						std::nth_element(near, near + pick, near + count);
						sub[static_cast<std::size_t>(y - y0) * side + (x - x0)] = near[pick];
					}
					const std::vector<float> floorNear = WindowExtreme(sub, side, floorR, false, kHigh);
					for (const std::size_t i : members) {
						const int   x = static_cast<int>(i % a_size), y = static_cast<int>(i / a_size);
						const float floorI = floorNear[static_cast<std::size_t>(y - y0) * side + (x - x0)];
						const float topI = floorI + fill * std::clamp(lid[i] - floorI, 0.0f, a_p.poolMaxDepth);
						pool[i] = std::clamp(topI - a_ground[i], 0.0f, 2.0f * a_p.poolMaxDepth);   // a pit: as the closed pool
					}
					continue;
				}
				const std::size_t pick = heights.size() / 10;
				std::nth_element(heights.begin(), heights.begin() + static_cast<std::ptrdiff_t>(pick), heights.end());
				const float floorC = heights[pick];
				const float top = floorC + fill * std::clamp(lidC - floorC, 0.0f, a_p.poolMaxDepth);
				// A cell far below the floor (a pit, a channel) is held to 2 x poolMaxDepth; the cap above is on the lid,
				// so the ordinary cells keep one level top.
				for (const std::size_t i : members) pool[i] = std::clamp(top - a_ground[i], 0.0f, 2.0f * a_p.poolMaxDepth);
			}
		}

		for (std::size_t i = 0; i < n; ++i) {
			const Kind k = a_kinds[i];
			auto&      o = out[i];
			o[3] = pool[i];
			if (!known(i)) continue;
			float m = a_p.land;
			switch (k) {
			case Kind::kLake:       m = a_p.lake; o[2] = a_p.freshSteam; break;
			case Kind::kRiver:      m = a_p.river; o[2] = a_p.freshSteam; break;
			case Kind::kMarshWater: m = a_p.marshWater; o[2] = a_p.freshSteam; break;
			case Kind::kMarshLand:  break;   // blended below with the dry land around it
			case Kind::kHotSpring:  m = a_p.lake; o[1] = a_p.hotSpring; break;
			case Kind::kSea:        m = a_p.seaRadiation; o[2] = a_p.seaSteam; break;
			default: break;
			}
			float terrain = 1.0f;
			if (!water(k)) {
				{
					const int x = static_cast<int>(i % a_size), y = static_cast<int>(i / a_size);
					const int x0 = std::max(x - rm, 0), y0 = std::max(y - rm, 0);
					const int x1 = std::min(x + rm + 1, a_size), y1 = std::min(y + rm + 1, a_size);
					const double c = boxSum(sumC, x0, y0, x1, y1);
					if (c > 0.0) m = a_p.land + (a_p.marshLand - a_p.land) * static_cast<float>(boxSum(sumM, x0, y0, x1, y1) / c);
				}
				if (shore && dist[i] > 0) {
					const float t = std::clamp(static_cast<float>(dist[i] - 1) / static_cast<float>(band), 0.0f, 1.0f);
					terrain *= a_p.shoreFactor + (1.0f - a_p.shoreFactor) * t;
				}
				const int x = static_cast<int>(i % a_size), y = static_cast<int>(i / a_size);
				const int x0 = std::max(x - r, 0), y0 = std::max(y - r, 0);
				const int x1 = std::min(x + r + 1, a_size), y1 = std::min(y + r + 1, a_size);
				const double c = boxSum(sumC, x0, y0, x1, y1);
				if (c > 0.0) {
					const double z = (boxSum(sumH, x0, y0, x1, y1) / c - a_ground[i]) / scale;
					const double weight = std::min(1.0, 2.0 * c / area);
					terrain *= std::clamp(static_cast<float>(1.0 + a_p.valleyStrength * z * weight), 0.5f, 1.8f);
				}
				terrain = std::clamp(terrain, a_p.minMultiplier, a_p.maxMultiplier);
			}
			o[0] = std::max(m * terrain, 0.0f);
			// Bank cap (bankCap, off by default): shore x valley stack on the bank (0.7 x 1.3 x 1.8 = 1.64 on a 20-degree
			// bank) while the water beside it keeps its plain multiplier (river 1.1 to 1.4), so the fog stood thicker on the
			// bank than over the water (DEBUG-LEDGER.md 2026-11-14 22:xx). Land touching the water is held to
			// C = max(that water's multiplier, the cell's own land multiplier m); the limit is released linearly across
			// the band, t = 0 at the cell touching the water, 1 at the band's outermost cell (which is thus untouched, so
			// nothing new appears at the band's edge). Only ever lowers: a cell at or under C keeps its value bit for bit.
			// max(.., m): flat marsh land beside a river keeps its own marsh multiplier. Limits: a band one cell wide has
			// t = 0 (held to C, with the step to the land behind it); along the bank C steps where the nearest water
			// changes kind, as the water itself does. The cap is NOT "bank never above its water": the limit is released
			// from the first land cell on, so on a steep bank the land a few cells in still stands above the water (the
			// 20-degree test bank, river 1.1: 1.1 at the water, then about 1.15 / 1.18 / 1.20 / 1.21 at cells 2 to 5, a
			// ridge about 0.1 above the river some 320 units from it; printed by the offline check). Whether that ridge
			// shows in game is not known yet; holding C across the whole band and fading over the last cells would remove it.
			if (bankCap && !water(k) && dist[i] > 0) {
				const float c = std::max(bankWater[i], m);
				if (c < o[0]) {
					const float t = band > 1 ? std::clamp(static_cast<float>(dist[i] - 1) / static_cast<float>(band - 1), 0.0f, 1.0f) : 0.0f;
					o[0] = c * (1.0f - t) + o[0] * t;
				}
			}
		}
		return out;
	}

	// Diagnosis only (profiler / self-test log): does bank land exceed the water beside it? Read from a Build output
	// and the inputs it was built from. "Bank" = known dry land (kLand) within the shore band, the band as Build has it
	// (8-neighbour steps from the nearest known water cell, at most ceil(shoreDistance / cell)); marsh land in the band
	// is kept apart (its own multiplier would hide the dry bank) and unknown cells are skipped (Build leaves them at 1).
	// Each bank cell is compared with its nearest water cell; where several are equally near, with the one of the
	// largest x, whatever the scan order (the rule Build's bank cap has, so a bank the cap holds to a lake's value is
	// not counted as above the river on its other side). With BankCap on the count of cells "above" stays above 0 by
	// design (the limit fades across the band): read the drop in the bank land max and in the largest excess instead.
	// Water values are read from the grid (the largest x of each kind present; -1 = no such cell). Cost: one more band
	// search, O(n); it is a second copy of Build's (limit: a change to Build's band rule must be made here too).
	struct ShoreSummary
	{
		bool  valid{ false };
		int   band{ 0 };              // cells
		float bankLand{ -1.0f };      // largest x among bank cells (-1: none)
		int   bankCells{ 0 };
		int   bankAbove{ 0 };         // bank cells whose x is above their nearest water cell's x
		float bankExcess{ 0.0f };     // largest (bank x - nearest water x), 0 when none is above
		float bankMarsh{ -1.0f };     // largest x among marsh-land cells in the band (-1: none)
		float landBeyond{ -1.0f };    // largest x among known dry land outside the band (-1: none)
		std::array<float, static_cast<std::size_t>(Kind::kCount)> water = [] {   // by Kind; -1 = no such cell
			std::array<float, static_cast<std::size_t>(Kind::kCount)> a{};
			a.fill(-1.0f);
			return a;
		}();
		float steamY{ 0.0f };         // largest y (hot-spring steam) over water
		float steamZ{ 0.0f };         // largest z (steam at full cold) over water
	};

	inline ShoreSummary SummarizeShore(const std::vector<std::array<float, 4>>& a_out, const std::vector<float>& a_ground, const std::vector<Kind>& a_kinds,
		int a_size, float a_cell, const Params& a_p)
	{
		ShoreSummary      s;
		const std::size_t n = static_cast<std::size_t>(a_size) * static_cast<std::size_t>(a_size);
		if (a_size <= 0 || a_out.size() != n || a_ground.size() != n || a_kinds.size() != n) {
			return s;
		}
		auto known = [&](std::size_t i) { return a_ground[i] > -1.0e29f; };
		auto water = [&](Kind k) { return k == Kind::kLake || k == Kind::kRiver || k == Kind::kMarshWater || k == Kind::kHotSpring || k == Kind::kSea; };
		s.valid = true;
		s.band = a_p.shoreDistance > 0.0f ? std::max(1, static_cast<int>(std::ceil(a_p.shoreDistance / std::max(a_cell, 1.0f)))) : 0;
		std::vector<int>         dist(n, -1);
		std::vector<std::size_t> from(n, 0);   // the nearest water cell of each reached cell (the largest x among equally near ones)
		std::deque<std::size_t>  queue;
		for (std::size_t i = 0; i < n; ++i) {
			if (!known(i) || !water(a_kinds[i])) continue;
			dist[i] = 0;
			from[i] = i;
			queue.push_back(i);
			const auto k = static_cast<std::size_t>(a_kinds[i]);
			s.water[k] = std::max(s.water[k], a_out[i][0]);
			s.steamY = std::max(s.steamY, a_out[i][1]);
			s.steamZ = std::max(s.steamZ, a_out[i][2]);
		}
		while (s.band > 0 && !queue.empty()) {
			const std::size_t i = queue.front();
			queue.pop_front();
			if (dist[i] >= s.band) continue;
			const int x = static_cast<int>(i % a_size), y = static_cast<int>(i / a_size);
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					const int nx = x + dx, ny = y + dy;
					if ((dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= a_size || ny >= a_size) continue;
					const std::size_t j = static_cast<std::size_t>(ny) * a_size + nx;
					if (dist[j] < 0) {
						dist[j] = dist[i] + 1;
						from[j] = from[i];
						queue.push_back(j);
					} else if (dist[j] == dist[i] + 1 && a_out[from[i]][0] > a_out[from[j]][0]) {
						from[j] = from[i];   // equally near: the water of the larger value, as Build's bankWater
					}
				}
			}
		}
		for (std::size_t i = 0; i < n; ++i) {
			if (!known(i) || water(a_kinds[i])) continue;
			const float v = a_out[i][0];
			if (dist[i] <= 0) {
				if (a_kinds[i] == Kind::kLand) s.landBeyond = std::max(s.landBeyond, v);
				continue;
			}
			if (a_kinds[i] == Kind::kMarshLand) {
				s.bankMarsh = std::max(s.bankMarsh, v);
				continue;
			}
			s.bankLand = std::max(s.bankLand, v);
			++s.bankCells;
			const float over = v - a_out[from[i]][0];
			if (over > 0.0f) {
				++s.bankAbove;
				s.bankExcess = std::max(s.bankExcess, over);
			}
		}
		return s;
	}

	// One log line from a summary. a_cold and a_wind are the cold and wind factors of FogConditions at the moment: the
	// steam a water cell gets is (y + z x cold) x wind, and no cell has both y and z.
	inline std::string DescribeShore(const ShoreSummary& a_s, float a_cold, float a_wind)
	{
		if (!a_s.valid) return "shore: no summary yet";
		auto num = [](float a_v, int a_digits) {
			char buf[32];
			std::snprintf(buf, sizeof(buf), "%.*f", a_digits, static_cast<double>(a_v));
			return std::string(buf);
		};
		auto value = [&](float a_v) { return a_v < 0.0f ? std::string("-") : num(a_v, 3); };
		std::string out = "shore: band " + std::to_string(a_s.band) + " cells, bank land max " + value(a_s.bankLand) + " (" + std::to_string(a_s.bankCells) +
		                  " cells, " + std::to_string(a_s.bankAbove) + " above their nearest water, largest excess " + num(a_s.bankExcess, 3) + "), bank marsh land max " +
		                  value(a_s.bankMarsh) + ", land beyond the band max " + value(a_s.landBeyond) + " | water multiplier:";
		for (const Kind k : { Kind::kLake, Kind::kRiver, Kind::kMarshWater, Kind::kSea, Kind::kHotSpring }) {
			out += std::string(" ") + KindName(k) + " " + value(a_s.water[static_cast<std::size_t>(k)]);
		}
		out += " | steam over water: max z " + num(a_s.steamZ, 2) + ", max y " + num(a_s.steamY, 2) + ", now " +
		       num(std::max(a_s.steamY, a_s.steamZ * a_cold) * a_wind, 2) + " (conditions steam " + num(a_cold, 2) + ", wind " + num(a_wind, 2) + ")";
		return out;
	}
}
