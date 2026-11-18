#pragma once

// Game side of the shared wind + fog grid: gathers the base wind (Sky, written by Dynamic Wind, only read
// here), moving actors and magic impacts, and steps WindSim once per drawn frame (twice when the frame is longer
// than 1/30 s, WindRules::StepsPerFrame). The grid covers the same square as the terrain height grid and moves
// with it.

namespace WindField
{
	struct Stats
	{
		std::uint64_t steps{ 0 };
		std::uint64_t shifts{ 0 };
		std::uint64_t resets{ 0 };
		std::uint64_t impacts{ 0 };    // projectile hits, explosions, shouts queued
		std::uint32_t lastActors{ 0 };
		std::uint32_t lastBlasts{ 0 };
		float         windSpeedMin{ 1e9f };
		float         windSpeedMax{ -1e9f };
		float         windAngleMin{ 1e9f };
		float         windAngleMax{ -1e9f };
		bool          created{ false };
		std::string   error;
	};

	// Hooks projectile impacts and explosions. Call once at load.
	void Install();

	// Spell cast events (shouts). Call at kDataLoaded, when the event source exists.
	void RegisterEvents();

	// One step. a_heights is the terrain height texture (walls come from it). Returns false if not ready.
	bool Update(ID3D11Device* a_device, ID3D11DeviceContext* a_ctx, ID3D11ShaderResourceView* a_heights);

	void Reset();

	// Builds the fog columns (H, W, ground) for the raymarch from the current field. Call after Update.
	void BuildColumns(ID3D11DeviceContext* a_ctx, ID3D11ShaderResourceView* a_heights, ID3D11ShaderResourceView* a_place = nullptr,
		std::array<float, 4> a_placeParams = { 1.0f, 0.0f, 1.0f, 0.0f }, std::array<float, 4> a_steamParams = { 0.0f, 0.0f, 0.0f, 0.0f },
		std::array<float, 4> a_placeGrid = { 0.0f, 0.0f, 0.0f, 0.0f }, std::array<float, 4> a_look = { 0.0f, 0.0f, 0.0f, 0.0f },
		ID3D11ShaderResourceView* a_coarse = nullptr, std::array<float, 4> a_coarseGrid = { 0.0f, 0.0f, 0.0f, 0.0f },
		std::array<float, 4> a_coarseParams = { 0.0f, 0.0f, 0.0f, 0.0f });
	ID3D11ShaderResourceView* ColumnSrv();
	ID3D11ShaderResourceView* UpperSrv();   // the tall layer (steam, pooled fog) next to ColumnSrv
	ID3D11ShaderResourceView* ReliefSrv();  // the column height factor - 1 (relief shading) next to ColumnSrv

	// Where the field lies in the world, and whether it has been stepped.
	bool                      Ready();
	float                     OriginX();
	float                     OriginY();
	float                     Extent();
	float                     GustX();  // base wind travel, units (the fog shape drifts by it)
	float                     GustY();
	float                     DriftSpeed();   // base wind this frame, units/s: how fast the fog shape travels
	std::string               DescribeInput();   // the game's wind changes per frame since the last call ("wind input:" line); resets the window
	float                     SimTime();   // seconds the field has run (the column noise clock)

	const Stats& GetStats();
}
