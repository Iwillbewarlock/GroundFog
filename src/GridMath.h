#pragma once

// Header-only grid helpers with no game dependency, shared by the plugin and the offline check.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace GridMath
{
	inline float SnapToCell(float a_value, float a_cell)
	{
		return std::floor(a_value / a_cell) * a_cell;
	}

	// new(x, y) = old(x + a_dx, y + a_dy); cells with no source get a_fill. O(size^2).
	inline void Shift(std::span<float> a_cells, int a_size, int a_dx, int a_dy, float a_fill)
	{
		if (a_size <= 0 || a_cells.size() != static_cast<std::size_t>(a_size) * static_cast<std::size_t>(a_size)) {
			return;
		}
		std::vector<float> moved(a_cells.size(), a_fill);
		for (int y = 0; y < a_size; ++y) {
			const int sy = y + a_dy;
			if (sy < 0 || sy >= a_size) {
				continue;
			}
			for (int x = 0; x < a_size; ++x) {
				const int sx = x + a_dx;
				if (sx < 0 || sx >= a_size) {
					continue;
				}
				moved[static_cast<std::size_t>(y) * a_size + x] = a_cells[static_cast<std::size_t>(sy) * a_size + sx];
			}
		}
		for (std::size_t i = 0; i < moved.size(); ++i) {
			a_cells[i] = moved[i];
		}
	}

	// Cell indices of an a_size x a_size grid (row-major), nearest the centre cell ((a_size - 1) / 2) first; cells at
	// the same distance keep their row order. A fill that asks the cells in this order knows the land round the grid's
	// centre (the player) first, whatever the grid's size. O(n log n) for n = a_size^2 cells.
	inline std::vector<std::size_t> CentreFirstOrder(int a_size)
	{
		std::vector<std::size_t> order(a_size > 0 ? static_cast<std::size_t>(a_size) * static_cast<std::size_t>(a_size) : 0);
		for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
		const int  c = (a_size - 1) / 2;
		const auto n = static_cast<std::size_t>(std::max(a_size, 1));
		std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
			const int ax = static_cast<int>(a % n) - c, ay = static_cast<int>(a / n) - c;
			const int bx = static_cast<int>(b % n) - c, by = static_cast<int>(b / n) - c;
			return ax * ax + ay * ay < bx * bx + by * by;
		});
		return order;
	}

	// Indices of the a_keep smallest values of a_dist2, smallest first; equal values keep their order (so an entry put
	// first with distance 0, the player, stays first). Values that are not finite are left out. O(n log n).
	inline std::vector<std::size_t> NearestFirst(std::span<const float> a_dist2, std::size_t a_keep)
	{
		std::vector<std::size_t> order;
		order.reserve(a_dist2.size());
		for (std::size_t i = 0; i < a_dist2.size(); ++i) {
			if (std::isfinite(a_dist2[i])) order.push_back(i);
		}
		std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return a_dist2[a] < a_dist2[b]; });
		if (order.size() > a_keep) order.resize(a_keep);
		return order;
	}

	// Cells from the coarse grid's centre that the first coarse pool build waits for (FogRenderer.cpp): the fog's reach
	// (the fade's outer circle, 0.475 x the field side) plus 1280 units (the coarse grid trails the player by up to 2 of
	// its cells, the field by RecenterDistance), in cells of a_cell, rounded up, at most a_half (the grid's half side).
	// 8192 -> 11, 12288 -> 14, 16384 -> 18 cells of 512.
	inline int CoarseGateCells(float a_worldSize, float a_cell, int a_half)
	{
		if (!(a_cell > 0.0f) || !(a_worldSize >= 0.0f)) return std::max(a_half, 0);
		return std::clamp(static_cast<int>(std::ceil((0.475f * a_worldSize + 1280.0f) / a_cell)), 0, std::max(a_half, 0));
	}

	// The far layer's land table (RaymarchShaders.h FarGround, t6): per vertex the ground z (a_unknown where the height
	// is not known, i.e. not above -1e29) and the share of the layer the land carries by its slope, 1 - smoothstep(tan 15
	// degrees, tan 30 degrees, |gradient|). The gradient is a central difference over the widest baseline up to
	// kLandSlopeReach vertices either side that the known land allows on each axis, one-sided at its edge, as
	// tools/offline/TerrainFogRepro.cpp builds it. a_rg: size * size * 2 floats (R ground, G share; 0 where unknown).
	// O(size^2 * kLandSlopeReach).
	inline constexpr int   kLandSlopeReach = 4;
	inline constexpr float kLandSlopeFull = 0.268f;   // tan 15 degrees: all of the layer up to here
	inline constexpr float kLandSlopeNone = 0.577f;   // tan 30 degrees: none from here
	inline void BuildLandTable(std::span<const float> a_heights, int a_size, float a_cell, float a_unknown, std::span<float> a_rg)
	{
		const std::size_t count = a_size > 0 ? static_cast<std::size_t>(a_size) * static_cast<std::size_t>(a_size) : 0;
		if (count == 0 || a_heights.size() != count || a_rg.size() != 2 * count || !(a_cell > 0.0f)) {
			return;
		}
		auto known = [&](int a_x, int a_y, double& a_z) {
			if (a_x < 0 || a_y < 0 || a_x >= a_size || a_y >= a_size) return false;
			const float h = a_heights[static_cast<std::size_t>(a_y) * a_size + a_x];
			if (!(h > -1.0e29f) || !std::isfinite(h)) return false;
			a_z = h;
			return true;
		};
		for (int y = 0; y < a_size; ++y) {
			for (int x = 0; x < a_size; ++x) {
				float* out = &a_rg[(static_cast<std::size_t>(y) * a_size + x) * 2];
				double z = 0.0;
				if (!known(x, y, z)) {
					out[0] = a_unknown;
					out[1] = 0.0f;
					continue;
				}
				auto axis = [&](int a_dx, int a_dy) {
					for (int r = kLandSlopeReach; r >= 1; --r) {
						double     za = z, zb = z;
						const bool ka = known(x - a_dx * r, y - a_dy * r, za), kb = known(x + a_dx * r, y + a_dy * r, zb);
						if (ka && kb) {
							// A vertex standing above the line between the two (a ridge, or just off its crest): the
							// difference across it misses the slope (a symmetric crest gave 0, so a 45-degree ridge
							// kept the whole layer along its top; reviewer 2026-11-13). The steeper side counts there.
							// A hollow keeps the difference across it: fog may lie on a valley floor.
							if (z > 0.5 * (za + zb) + 1.0e-3) return std::max(z - za, z - zb) / (r * a_cell);
							return (zb - za) / (2.0 * r * a_cell);
						}
						if (kb) return (zb - z) / (r * a_cell);
						if (ka) return (z - za) / (r * a_cell);
					}
					return 0.0;
				};
				const double grad = std::hypot(axis(1, 0), axis(0, 1));
				const double t = std::clamp((grad - kLandSlopeFull) / (kLandSlopeNone - kLandSlopeFull), 0.0, 1.0);
				out[0] = static_cast<float>(z);
				out[1] = static_cast<float>(1.0 - t * t * (3.0 - 2.0 * t));
			}
		}
	}
}
