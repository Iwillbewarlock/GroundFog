#pragma once

// Header-only grid helpers with no game dependency, shared by the plugin and the offline check.

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
}
