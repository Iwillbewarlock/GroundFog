#pragma once

// Player-centred grid of ground heights (land, raised to the water surface where there is water),
// filled on the CPU with TES::GetLandHeight. Cells that could not be read stay "unknown" and are
// asked again on later frames; the shader hides fog over unknown cells.
namespace Terrain
{
	inline constexpr float kUnknown = -1.0e30f;  // never asked
	inline constexpr float kFailed = -2.0e30f;   // asked, GetLandHeight said no; retried slowly
	inline constexpr bool  IsKnown(float a_h) { return a_h > -1.0e29f; }

	struct Stats
	{
		std::uint64_t queries{ 0 };
		std::uint64_t failures{ 0 };
		std::uint64_t recenters{ 0 };
		double        fillMsTotal{ 0.0 };
		double        fillMsMax{ 0.0 };
		std::uint32_t unknownCells{ 0 };
		std::uint32_t lastShiftX{ 0 };  // |cells| of the last recenter, for the self-test
		std::uint32_t lastShiftY{ 0 };
		std::uint32_t verifyChecks{ 0 };      // cached cells asked again after a recenter (self-test)
		float         verifyMaxDiff{ 0.0f };  // largest |cached - fresh| among them
		std::uint32_t worldspaceResets{ 0 };
		std::uint64_t waterRaised{ 0 };       // queries where the water surface was above the land
		std::uint64_t kindCounts[8]{};        // queries per PlacementRules::Kind
		float         waterRaiseMax{ 0.0f };  // largest water - land among them (units)
	};

	// World position of grid cell (0, 0) and the spacing between cells, in game units.
	struct Grid
	{
		float originX{ 0.0f };
		float originY{ 0.0f };
		float cell{ 64.0f };
		int   size{ 0 };
	};

	// Call once per frame from the render thread. Returns false if there is no outdoor world.
	// The cache is cleared whenever a_worldspace differs from the last call.
	bool Update(const RE::NiPoint3& a_player, const void* a_worldspace);

	void Reset();

	// Grid moves since the last call, in height cells; returns true when the grid was rebuilt from scratch
	// (first fill, new worldspace, size change). The wind field follows these.
	bool ConsumeChange(int& a_dx, int& a_dy);

	const Grid&               GetGrid();
	std::span<const float>    Heights();  // size*size, row-major (y rows, x columns), kUnknown where unread
	// Same layout: PlacementRules::Kind of each cell (land, marsh land, lake, river, marsh water, hot spring, sea).
	std::span<const std::uint8_t> Kinds();
	bool                      KindsDirty();   // kinds or heights changed since the last ClearKindsDirty
	void                      ClearKindsDirty();
	std::uint32_t             Generation();   // changes when the grid is rebuilt from scratch or reset (new worldspace, size)
	const Stats&              GetStats();
	bool                      Dirty();    // heights changed since the last ClearDirty
	void                      ClearDirty();

	// A coarse grid around the player for pooled fog in valleys wider than the fine grid: kCoarseSize^2 heights,
	// kCoarseCell units apart (about 470 m across), filled centre-first a few cells a frame, moved by whole cells,
	// cleared with the fine grid (worldspace change, reset). Heights only (water surface over water), no kinds.
	inline constexpr int   kCoarseSize = 65;
	inline constexpr float kCoarseCell = 512.0f;
	void                   UpdateCoarse(const RE::NiPoint3& a_player);   // call after Update, same frame
	const Grid&            CoarseGrid();
	std::span<const float> CoarseHeights();
	std::uint32_t          CoarseGeneration();   // bumped when the coarse grid is rebuilt from scratch
	bool                   CoarseDirty();        // heights or position changed since the last ClearCoarseDirty
	void                   ClearCoarseDirty();

	// The far layer's land table (FogRenderer t6, RaymarchShaders.h FarGround): kLandSize^2 heights kLandCell apart, as
	// fine as the land itself (12288 units either side, past the land the game loads around the player), filled
	// centre-first with a per-frame query budget, moved by whole vertices, cleared with the fine grid (worldspace change,
	// reset). Heights only (water surface over water); a vertex GetLandHeight cannot read stays unknown and is asked again
	// as the coarse grid's are.
	inline constexpr int   kLandSize = 193;
	inline constexpr float kLandCell = 128.0f;
	void                   UpdateLand(const RE::NiPoint3& a_player);   // call after Update, same frame
	const Grid&            LandGrid();
	std::span<const float> LandHeights();
	std::uint32_t          LandGeneration();   // bumped when the land table is rebuilt from scratch
	bool                   LandDirty();        // a vertex became known or the table moved since the last ClearLandDirty
	void                   ClearLandDirty();

	// Ground at world (a_x, a_y) from the fine grid, bilinear; false off the grid or next to a vertex not known.
	bool HeightAt(float a_x, float a_y, float& a_out);
	// As HeightAt, and off the fine grid from the land table (12288 units either side, 128 apart; FogRenderer's
	// LocalLights, Codex review 2026-11-17: a light past the fine grid had been taken whole, then dropped on entering it).
	bool GroundAt(float a_x, float a_y, float& a_out);
}
