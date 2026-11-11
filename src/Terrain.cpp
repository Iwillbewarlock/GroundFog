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
		std::size_t        g_cursor{ 0 };
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

		// Asks up to a_budget cells whose value equals a_wanted (kUnknown or kFailed), walking from a_cursor.
		int Fill(RE::TES* a_tes, float a_wanted, int a_budget, std::size_t& a_cursor)
		{
			const auto count = g_heights.size();
			int        asked = 0;
			if (count == 0 || a_budget <= 0) {
				return asked;
			}
			for (std::size_t visited = 0; visited < count && asked < a_budget; ++visited) {
				const std::size_t index = a_cursor;
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

	const Grid&            CoarseGrid() { return g_coarse; }
	std::span<const float> CoarseHeights() { return g_coarseHeights; }
	std::uint32_t          CoarseGeneration() { return g_coarseGeneration; }
	bool                   CoarseDirty() { return g_coarseDirty; }
	void                   ClearCoarseDirty() { g_coarseDirty = false; }
	const Stats&           GetStats() { return g_stats; }
	bool                   Dirty() { return g_dirty; }
	void                   ClearDirty() { g_dirty = false; }
}
