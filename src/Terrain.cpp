#include "PCH.h"

#include "Terrain.h"

#include "EditorID.h"
#include "GridMath.h"
#include "PlacementRules.h"
#include "Settings.h"

namespace Terrain
{
	namespace
	{
		Grid               g_grid{};
		std::vector<float> g_heights;
		std::vector<std::uint8_t> g_kinds;   // PlacementRules::Kind per cell
		bool               g_kindsDirty{ true };
		std::uint32_t      g_generation{ 0 };   // bumped when the grid is rebuilt from scratch or reset
		Stats              g_stats{};
		bool               g_dirty{ true };
		bool               g_valid{ false };
		std::vector<std::size_t> g_fillOrder;   // cell indices nearest the centre first (GridMath::CentreFirstOrder)
		std::size_t        g_cursor{ 0 };        // positions in g_fillOrder
		std::size_t        g_retryCursor{ 0 };
		const void*        g_worldspace{ nullptr };
		float              g_probeZ{ 0.0f };
		// Coarse grid (pooled fog in wide valleys).
		Grid                     g_coarse{};
		std::vector<float>       g_coarseHeights;
		std::vector<std::size_t> g_coarseOrder;      // cell indices nearest the centre first
		std::size_t              g_coarseCursor{ 0 };
		std::uint32_t            g_coarseGeneration{ 0 };
		bool                     g_coarseDirty{ true };
		bool                     g_coarseValid{ false };
		// The far layer's land table.
		Grid                     g_land{};
		std::vector<float>       g_landHeights;
		std::vector<std::size_t> g_landOrder;        // vertex indices nearest the centre first
		std::size_t              g_landCursor{ 0 };
		std::uint32_t            g_landGeneration{ 0 };
		bool                     g_landDirty{ true };
		bool                     g_landValid{ false };
		int                g_pendingShiftX{ 0 };
		int                g_pendingShiftY{ 0 };
		bool               g_pendingReset{ true };

		float QueryHeight(RE::TES* a_tes, float a_x, float a_y, bool& a_ok, PlacementRules::Kind* a_kind = nullptr);

		// Self-test: after a recenter, ask a few cached cells again; a moved cache must agree with the game.
		void VerifyAfterShift(RE::TES* a_tes)
		{
			constexpr int kProbes = 8;
			const auto    count = g_heights.size();
			for (int i = 0; i < kProbes; ++i) {
				const std::size_t index = (count * static_cast<std::size_t>(2 * i + 1)) / (2 * kProbes);
				if (!IsKnown(g_heights[index])) {
					continue;
				}
				const int x = static_cast<int>(index % static_cast<std::size_t>(g_grid.size));
				const int y = static_cast<int>(index / static_cast<std::size_t>(g_grid.size));
				bool      ok = false;
				const float fresh = QueryHeight(a_tes, g_grid.originX + x * g_grid.cell, g_grid.originY + y * g_grid.cell, ok);
				if (!ok) {
					continue;
				}
				++g_stats.verifyChecks;
				g_stats.verifyMaxDiff = std::max(g_stats.verifyMaxDiff, std::fabs(fresh - g_heights[index]));
			}
		}

		// First fill and refills after a recenter may ask this many cells in one frame.
		constexpr int kFillBudgetPerFrame = 1024;

		float QueryHeight(RE::TES* a_tes, float a_x, float a_y, bool& a_ok, PlacementRules::Kind* a_kind)
		{
			if (a_kind) {
				*a_kind = PlacementRules::Kind::kLand;
			}
			// DTD asks with the player's height as z (SnowCoverage.cpp:353-357); a fixed z = 0 may miss land far
			// above or below it (suspected cause of the all-unknown grid seen 2026-11-07, not proven).
			RE::NiPoint3 probe{ a_x, a_y, g_probeZ };
			float        land = 0.0f;
			a_ok = a_tes->GetLandHeight(probe, land);
			if (!a_ok || !std::isfinite(land)) {
				a_ok = false;
				return kUnknown;
			}
			probe.z = land;
			float ground = land;
			// Raise to the water surface only where the cell itself says it has water.
			auto* cell = a_tes->GetCell(probe);
			float water = 0.0f;
			if (cell && cell->cellFlags.any(RE::TESObjectCELL::Flag::kHasWater) && cell->GetWaterHeight(probe, water) &&
				std::isfinite(water) && water > land) {
				ground = water;
				++g_stats.waterRaised;
				g_stats.waterRaiseMax = std::max(g_stats.waterRaiseMax, water - land);
				if (a_kind) {
					// Water body from the cell's water type (or the worldspace's), sea by the worldspace's sea level.
					const RE::TESWaterForm* type = nullptr;
					if (const auto* extra = cell->extraList.GetByType<RE::ExtraCellWaterType>()) {
						type = extra->water;
					}
					const auto* ws = cell->GetRuntimeData().worldSpace;
					// A child worldspace may take its water and LOD data from the parent (Solstheim, city spaces).
					auto parentOf = [](const RE::TESWorldSpace* a_ws, RE::TESWorldSpace::ParentUseFlag a_flag) {
						while (a_ws && a_ws->parentWorld && a_ws->parentUseFlags.any(a_flag)) a_ws = a_ws->parentWorld;
						return a_ws;
					};
					if (!type && ws) {
						type = parentOf(ws, RE::TESWorldSpace::ParentUseFlag::kUseWaterData)->worldWater;
					}
					float sea = ws ? ws->GetDefaultWaterHeight() : water;
					if (ws && !(std::fabs(sea) < 1.0e6f)) {
						sea = parentOf(ws, RE::TESWorldSpace::ParentUseFlag::kUseLODData)->lodWaterHeight;
					}
					*a_kind = PlacementRules::WaterKind(EditorIDOf(type), water, sea);
				}
			} else if (a_kind) {
				if (const auto* tex = a_tes->GetLandTexture(probe); tex && PlacementRules::IsMarshLand(EditorIDOf(tex))) {
					*a_kind = PlacementRules::Kind::kMarshLand;
				}
			}
			return ground;
		}

		// Asks up to a_budget cells whose value equals a_wanted (kUnknown or kFailed), walking g_fillOrder (nearest the
		// grid's centre first) from position a_cursor and round again. The first fill starts at the centre, and so does
		// the refill after a recenter (Update puts the unknown cursor back to 0): the land round the player is known
		// first whatever the grid's size (in row order a 193 grid reached the player's row in its 19th frame). The
		// failed cells' retries keep going round, so each gets its turn.
		int Fill(RE::TES* a_tes, float a_wanted, int a_budget, std::size_t& a_cursor)
		{
			const auto count = g_heights.size();
			int        asked = 0;
			if (count == 0 || a_budget <= 0 || g_fillOrder.size() != count) {
				return asked;
			}
			a_cursor %= count;
			for (std::size_t visited = 0; visited < count && asked < a_budget; ++visited) {
				const std::size_t index = g_fillOrder[a_cursor];
				a_cursor = (a_cursor + 1) % count;
				if (g_heights[index] != a_wanted) {
					continue;
				}
				const int x = static_cast<int>(index % static_cast<std::size_t>(g_grid.size));
				const int y = static_cast<int>(index / static_cast<std::size_t>(g_grid.size));
				bool      ok = false;
				PlacementRules::Kind kind = PlacementRules::Kind::kLand;
				const float h = QueryHeight(a_tes, g_grid.originX + x * g_grid.cell, g_grid.originY + y * g_grid.cell, ok, &kind);
				++asked;
				++g_stats.queries;
				if (ok) {
					g_heights[index] = h;
					g_kinds[index] = static_cast<std::uint8_t>(kind);
					++g_stats.kindCounts[std::min<std::size_t>(static_cast<std::size_t>(kind), 7)];
					g_dirty = true;
					g_kindsDirty = true;
				} else {
					g_heights[index] = kFailed;
					++g_stats.failures;
				}
			}
			return asked;
		}

		void FillPending(RE::TES* a_tes)
		{
			const auto start = std::chrono::steady_clock::now();
			int        asked = Fill(a_tes, kUnknown, kFillBudgetPerFrame, g_cursor);
			asked += Fill(a_tes, kFailed, Settings::heightRetriesPerFrame, g_retryCursor);
			if (asked > 0) {
				const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
				g_stats.fillMsTotal += ms;
				g_stats.fillMsMax = std::max(g_stats.fillMsMax, ms);
			}
			g_stats.unknownCells = static_cast<std::uint32_t>(
				std::count_if(g_heights.begin(), g_heights.end(), [](float a_h) { return !IsKnown(a_h); }));
		}
	}

	bool Update(const RE::NiPoint3& a_player, const void* a_worldspace)
	{
		auto* tes = RE::TES::GetSingleton();
		if (!tes || !a_worldspace) {
			return false;
		}
		g_probeZ = a_player.z;
		if (a_worldspace != g_worldspace) {
			// A door into another worldspace can land on overlapping coordinates; old heights would be wrong there.
			Reset();
			g_worldspace = a_worldspace;
			++g_stats.worldspaceResets;
		}

		const int size = Settings::heightResolution;
		const float cell = Settings::worldSize / static_cast<float>(size - 1);
		const float half = cell * static_cast<float>((size - 1) / 2);
		const float wantX = GridMath::SnapToCell(a_player.x, cell) - half;
		const float wantY = GridMath::SnapToCell(a_player.y, cell) - half;

		if (!g_valid || g_grid.size != size || g_grid.cell != cell) {
			g_grid = { wantX, wantY, cell, size };
			g_heights.assign(static_cast<std::size_t>(size) * size, kUnknown);
			g_kinds.assign(static_cast<std::size_t>(size) * size, 0);
			if (g_fillOrder.size() != g_heights.size()) {
				g_fillOrder = GridMath::CentreFirstOrder(size);   // here, not in FillPending: its time is the self-test's "height fill time"
			}
			g_kindsDirty = true;
			++g_generation;
			g_pendingReset = true;
			g_cursor = 0;
			g_retryCursor = 0;
			g_valid = true;
			g_dirty = true;
		} else {
			const float centreX = g_grid.originX + half;
			const float centreY = g_grid.originY + half;
			if (std::fabs(a_player.x - centreX) > Settings::recenterDistance ||
				std::fabs(a_player.y - centreY) > Settings::recenterDistance) {
				// Whole-cell moves keep every cached height on its world position.
				const int dx = static_cast<int>(std::lround((wantX - g_grid.originX) / cell));
				const int dy = static_cast<int>(std::lround((wantY - g_grid.originY) / cell));
				GridMath::Shift(g_heights, size, dx, dy, kUnknown);
				{
					// Kinds move with the heights (same shift through a float copy; 0..7 are exact in float).
					std::vector<float> kinds(g_kinds.begin(), g_kinds.end());
					GridMath::Shift(kinds, size, dx, dy, 0.0f);
					for (std::size_t i = 0; i < kinds.size(); ++i) g_kinds[i] = static_cast<std::uint8_t>(kinds[i]);
					g_kindsDirty = true;
				}
				g_grid.originX += dx * cell;
				g_grid.originY += dy * cell;
				g_pendingShiftX += dx;
				g_pendingShiftY += dy;
				g_stats.lastShiftX = static_cast<std::uint32_t>(std::abs(dx));
				g_stats.lastShiftY = static_cast<std::uint32_t>(std::abs(dy));
				++g_stats.recenters;
				g_cursor = 0;   // the new edge's cells nearest the centre first
				g_dirty = true;
				if (Settings::selfTest) {
					VerifyAfterShift(tes);
				}
			}
		}

		if (std::any_of(g_heights.begin(), g_heights.end(), [](float a_h) { return !IsKnown(a_h); })) {
			FillPending(tes);
		}
		return true;
	}

	void Reset()
	{
		g_landValid = false;
		g_landHeights.clear();
		g_landDirty = true;
		++g_landGeneration;
		g_coarseValid = false;
		g_coarseHeights.clear();
		g_coarseDirty = true;
		++g_coarseGeneration;
		g_valid = false;
		g_heights.clear();
		g_kinds.clear();
		g_dirty = true;
		g_kindsDirty = true;
		++g_generation;
		g_cursor = 0;
		g_retryCursor = 0;
		g_worldspace = nullptr;
	}

	bool ConsumeChange(int& a_dx, int& a_dy)
	{
		const bool reset = g_pendingReset;
		a_dx = g_pendingShiftX;
		a_dy = g_pendingShiftY;
		g_pendingShiftX = g_pendingShiftY = 0;
		g_pendingReset = false;
		return reset;
	}

	const Grid&            GetGrid() { return g_grid; }
	std::span<const float> Heights() { return g_heights; }
	std::span<const std::uint8_t> Kinds() { return g_kinds; }
	bool                   KindsDirty() { return g_kindsDirty; }
	void                   ClearKindsDirty() { g_kindsDirty = false; }
	std::uint32_t          Generation() { return g_generation; }

	void UpdateCoarse(const RE::NiPoint3& a_player)
	{
		auto* tes = RE::TES::GetSingleton();
		if (!tes || !g_valid) {
			return;
		}
		constexpr int   size = kCoarseSize;
		constexpr float cell = kCoarseCell;
		constexpr float half = cell * static_cast<float>((size - 1) / 2);
		constexpr int   kBudget = 48;            // queries a frame (first fill about 90 frames)
		constexpr float kRecenter = 2.0f * cell;
		const float wantX = GridMath::SnapToCell(a_player.x, cell) - half;
		const float wantY = GridMath::SnapToCell(a_player.y, cell) - half;
		if (!g_coarseValid) {
			g_coarse = { wantX, wantY, cell, size };
			g_coarseHeights.assign(static_cast<std::size_t>(size) * size, kUnknown);
			if (g_coarseOrder.size() != g_coarseHeights.size()) {
				g_coarseOrder.resize(g_coarseHeights.size());
				for (std::size_t i = 0; i < g_coarseOrder.size(); ++i) g_coarseOrder[i] = i;
				const int c = (size - 1) / 2;
				std::stable_sort(g_coarseOrder.begin(), g_coarseOrder.end(), [&](std::size_t a, std::size_t b) {
					const int ax = static_cast<int>(a % size) - c, ay = static_cast<int>(a / size) - c;
					const int bx = static_cast<int>(b % size) - c, by = static_cast<int>(b / size) - c;
					return ax * ax + ay * ay < bx * bx + by * by;
				});
			}
			g_coarseCursor = 0;
			g_coarseValid = true;
			g_coarseDirty = true;
			++g_coarseGeneration;
		} else if (std::fabs(a_player.x - (g_coarse.originX + half)) > kRecenter || std::fabs(a_player.y - (g_coarse.originY + half)) > kRecenter) {
			const int dx = static_cast<int>(std::lround((wantX - g_coarse.originX) / cell));
			const int dy = static_cast<int>(std::lround((wantY - g_coarse.originY) / cell));
			GridMath::Shift(g_coarseHeights, size, dx, dy, kUnknown);
			for (auto& h : g_coarseHeights) h = h == kFailed ? kUnknown : h;   // land may have loaded since (GPT review 2026-11-08)
			g_coarse.originX += dx * cell;
			g_coarse.originY += dy * cell;
			g_coarseCursor = 0;
			g_coarseDirty = true;
		}
		// Failed cells are land the game has not loaded (in game 2026-11-08: 2625 of 4225 failed, the known 1600 = the
		// loaded cells around the player). Asking them again every 5 s re-queried all of them for nothing; they are
		// asked again when the player enters another cell (new land loads around it) and, for land that loads late,
		// every 1800 frames (about 30 s).
		static std::uint32_t      frames = 0;
		static const RE::TESObjectCELL* lastCell = nullptr;
		const auto*               player = RE::PlayerCharacter::GetSingleton();
		const RE::TESObjectCELL*  parentCell = player ? player->GetParentCell() : nullptr;
		const bool                moved = parentCell != lastCell;
		lastCell = parentCell;
		if (moved || ++frames % 1800 == 0) {
			bool any = false;
			for (auto& h : g_coarseHeights) {
				if (h == kFailed) {
					h = kUnknown;
					any = true;
				}
			}
			if (any) g_coarseCursor = 0;
		}
		// Centre first: walk the distance order from the start each frame until the budget is spent on unknown cells.
		int asked = 0;
		for (std::size_t k = g_coarseCursor; k < g_coarseOrder.size() && asked < kBudget; ++k) {
			const std::size_t i = g_coarseOrder[k];
			if (g_coarseHeights[i] != kUnknown) {
				if (asked == 0) g_coarseCursor = k + 1;
				continue;
			}
			const int x = static_cast<int>(i % size), y = static_cast<int>(i / size);
			bool      ok = false;
			const float h = QueryHeight(tes, g_coarse.originX + x * cell, g_coarse.originY + y * cell, ok);
			g_coarseHeights[i] = ok ? h : kFailed;
			g_coarseDirty = true;
			++asked;
		}
	}

	void UpdateLand(const RE::NiPoint3& a_player)
	{
		auto* tes = RE::TES::GetSingleton();
		if (!tes || !g_valid) {
			return;
		}
		constexpr int   size = kLandSize;
		constexpr float cell = kLandCell;
		constexpr float half = cell * static_cast<float>((size - 1) / 2);
		// 37249 vertices at about 0.4 us a query (15 ms in all): 512 a frame keeps it near 0.2 ms, half of the fine grid's
		// first-fill budget, and fills the table in about 73 frames, the land near the player in the first few (centre
		// first). A recenter's new edge (4 rows, 772 vertices) takes 2 frames; vertices that failed are not asked again then.
		constexpr int   kBudget = 512;
		constexpr float kRecenter = 4.0f * cell;
		const float     wantX = GridMath::SnapToCell(a_player.x, cell) - half;
		const float     wantY = GridMath::SnapToCell(a_player.y, cell) - half;
		if (!g_landValid) {
			g_land = { wantX, wantY, cell, size };
			g_landHeights.assign(static_cast<std::size_t>(size) * size, kUnknown);
			if (g_landOrder.size() != g_landHeights.size()) {
				g_landOrder.resize(g_landHeights.size());
				for (std::size_t i = 0; i < g_landOrder.size(); ++i) g_landOrder[i] = i;
				const int c = (size - 1) / 2;
				std::stable_sort(g_landOrder.begin(), g_landOrder.end(), [&](std::size_t a, std::size_t b) {
					const int ax = static_cast<int>(a % size) - c, ay = static_cast<int>(a / size) - c;
					const int bx = static_cast<int>(b % size) - c, by = static_cast<int>(b / size) - c;
					return ax * ax + ay * ay < bx * bx + by * by;
				});
			}
			g_landCursor = 0;
			g_landValid = true;
			g_landDirty = true;
			++g_landGeneration;
		} else if (std::fabs(a_player.x - (g_land.originX + half)) > kRecenter || std::fabs(a_player.y - (g_land.originY + half)) > kRecenter) {
			const int dx = static_cast<int>(std::lround((wantX - g_land.originX) / cell));
			const int dy = static_cast<int>(std::lround((wantY - g_land.originY) / cell));
			// Failed vertices stay failed here: land loads when the player enters another cell, and that (and the timer)
			// asks them again below. Asking all of them again at every shift cost about 22 frames of the full budget every
			// 512 units of movement (reviewer 2026-11-13).
			GridMath::Shift(g_landHeights, size, dx, dy, kUnknown);
			g_land.originX += dx * cell;
			g_land.originY += dy * cell;
			g_landCursor = 0;
			g_landDirty = true;
		}
		// Failed vertices are land the game has not loaded: asked again when the player enters another cell and every 1800
		// frames (about 30 s), as the coarse grid's. Unknown and failed upload the same, so this does not dirty the table.
		static std::uint32_t            frames = 0;
		static const RE::TESObjectCELL* lastCell = nullptr;
		const auto*                     player = RE::PlayerCharacter::GetSingleton();
		const RE::TESObjectCELL*        parentCell = player ? player->GetParentCell() : nullptr;
		const bool                      moved = parentCell != lastCell;
		lastCell = parentCell;
		if (moved || ++frames % 1800 == 0) {
			bool any = false;
			for (auto& h : g_landHeights) {
				if (h == kFailed) {
					h = kUnknown;
					any = true;
				}
			}
			if (any) g_landCursor = 0;
		}
		// Centre first: walk the distance order from the cursor until the budget is spent on unknown vertices.
		int asked = 0;
		for (std::size_t k = g_landCursor; k < g_landOrder.size() && asked < kBudget; ++k) {
			const std::size_t i = g_landOrder[k];
			if (g_landHeights[i] != kUnknown) {
				if (asked == 0) g_landCursor = k + 1;
				continue;
			}
			const int x = static_cast<int>(i % size), y = static_cast<int>(i / size);
			bool      ok = false;
			const float h = QueryHeight(tes, g_land.originX + x * cell, g_land.originY + y * cell, ok);
			g_landHeights[i] = ok ? h : kFailed;
			g_landDirty = g_landDirty || ok;
			++asked;
		}
	}

	const Grid&            LandGrid() { return g_land; }
	std::span<const float> LandHeights() { return g_landHeights; }
	std::uint32_t          LandGeneration() { return g_landGeneration; }
	bool                   LandDirty() { return g_landDirty; }
	void                   ClearLandDirty() { g_landDirty = false; }

	namespace
	{
		// Bilinear height from one of the grids; false off it or next to a vertex not known.
		bool Bilinear(const Grid& a_grid, const std::vector<float>& a_heights, float a_x, float a_y, float& a_out)
		{
			const int size = a_grid.size;
			if (size < 2 || !(a_grid.cell > 0.0f) || a_heights.size() != static_cast<std::size_t>(size) * size) {
				return false;
			}
			const float gx = (a_x - a_grid.originX) / a_grid.cell;
			const float gy = (a_y - a_grid.originY) / a_grid.cell;
			if (!(gx >= 0.0f && gy >= 0.0f && gx <= static_cast<float>(size - 1) && gy <= static_cast<float>(size - 1))) {
				return false;
			}
			const int   x0 = std::min(static_cast<int>(gx), size - 2);
			const int   y0 = std::min(static_cast<int>(gy), size - 2);
			const float fx = gx - static_cast<float>(x0), fy = gy - static_cast<float>(y0);
			auto        at = [&](int a_x2, int a_y2) { return a_heights[static_cast<std::size_t>(a_y2) * size + a_x2]; };
			const float h00 = at(x0, y0), h10 = at(x0 + 1, y0), h01 = at(x0, y0 + 1), h11 = at(x0 + 1, y0 + 1);
			if (!IsKnown(h00) || !IsKnown(h10) || !IsKnown(h01) || !IsKnown(h11)) {
				return false;
			}
			a_out = (h00 * (1.0f - fx) + h10 * fx) * (1.0f - fy) + (h01 * (1.0f - fx) + h11 * fx) * fy;
			return true;
		}
	}

	bool GroundAt(float a_x, float a_y, float& a_out)
	{
		return HeightAt(a_x, a_y, a_out) || (g_valid && Bilinear(g_land, g_landHeights, a_x, a_y, a_out));
	}

	bool HeightAt(float a_x, float a_y, float& a_out)
	{
		const int size = g_grid.size;
		if (!g_valid || size < 2 || !(g_grid.cell > 0.0f) || g_heights.size() != static_cast<std::size_t>(size) * size) {
			return false;
		}
		const float gx = (a_x - g_grid.originX) / g_grid.cell;
		const float gy = (a_y - g_grid.originY) / g_grid.cell;
		if (!(gx >= 0.0f && gy >= 0.0f && gx <= static_cast<float>(size - 1) && gy <= static_cast<float>(size - 1))) {
			return false;
		}
		const int   x0 = std::min(static_cast<int>(gx), size - 2);
		const int   y0 = std::min(static_cast<int>(gy), size - 2);
		const float fx = gx - static_cast<float>(x0), fy = gy - static_cast<float>(y0);
		auto        at = [&](int a_x2, int a_y2) { return g_heights[static_cast<std::size_t>(a_y2) * size + a_x2]; };
		const float h00 = at(x0, y0), h10 = at(x0 + 1, y0), h01 = at(x0, y0 + 1), h11 = at(x0 + 1, y0 + 1);
		if (!IsKnown(h00) || !IsKnown(h10) || !IsKnown(h01) || !IsKnown(h11)) {
			return false;
		}
		a_out = (h00 * (1.0f - fx) + h10 * fx) * (1.0f - fy) + (h01 * (1.0f - fx) + h11 * fx) * fy;
		return true;
	}

	const Grid&            CoarseGrid() { return g_coarse; }
	std::span<const float> CoarseHeights() { return g_coarseHeights; }
	std::uint32_t          CoarseGeneration() { return g_coarseGeneration; }
	bool                   CoarseDirty() { return g_coarseDirty; }
	void                   ClearCoarseDirty() { g_coarseDirty = false; }
	const Stats&           GetStats() { return g_stats; }
	bool                   Dirty() { return g_dirty; }
	void                   ClearDirty() { g_dirty = false; }
}
