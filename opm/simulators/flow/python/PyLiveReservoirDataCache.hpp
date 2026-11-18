/*
  Copyright 2026 Equinor ASA.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef OPM_PY_LIVE_RESERVOIR_DATA_CACHE_HEADER_INCLUDED
#define OPM_PY_LIVE_RESERVOIR_DATA_CACHE_HEADER_INCLUDED

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace Opm::Pybind {

struct LiveReservoirCellPair
{
    std::size_t first{};
    std::size_t second{};

    bool operator==(const LiveReservoirCellPair& other) const
    {
        return first == other.first && second == other.second;
    }
};

struct LiveReservoirCellPairHash
{
    std::size_t operator()(const LiveReservoirCellPair& pair) const
    {
        // The key is scoped to one live simulator session. A lightweight integer mix keeps
        // lookups cheap without introducing a separate pair-hash dependency.
        return (pair.first * 1315423911u) ^ (pair.second + 0x9e3779b97f4a7c15ULL);
    }
};

struct PyLiveReservoirDataCache
{
    bool active_to_global_valid{false};
    std::vector<int> active_to_global{};
    // Indexed by Cartesian cell, nonzero for a cell of the simulator grid. Filled together
    // with active_to_global and valid under the same flag.
    std::vector<char> cartesian_active{};

    bool nnc_topology_valid{false};
    std::vector<LiveReservoirCellPair> nnc_pairs{};
    std::unordered_map<LiveReservoirCellPair,
                       std::vector<std::size_t>,
                       LiveReservoirCellPairHash> nnc_output_index{};

    bool water_flux_valid{false};
    std::size_t water_flux_step{0};
    std::vector<double> water_face_i{};
    std::vector<double> water_face_j{};
    std::vector<double> water_face_k{};
    std::vector<double> water_surface_face_i{};
    std::vector<double> water_surface_face_j{};
    std::vector<double> water_surface_face_k{};
    std::vector<double> nnc_water_reservoir_rate{};
    std::vector<double> nnc_water_surface_rate{};

    void reset()
    {
        active_to_global_valid = false;
        active_to_global.clear();
        cartesian_active.clear();

        nnc_topology_valid = false;
        nnc_pairs.clear();
        nnc_output_index.clear();

        resetWaterFlux();
    }

    void resetWaterFlux()
    {
        water_flux_valid = false;
        water_flux_step = 0;
        water_face_i.clear();
        water_face_j.clear();
        water_face_k.clear();
        water_surface_face_i.clear();
        water_surface_face_j.clear();
        water_surface_face_k.clear();
        nnc_water_reservoir_rate.clear();
        nnc_water_surface_rate.clear();
    }
};

} // namespace Opm::Pybind

#endif // OPM_PY_LIVE_RESERVOIR_DATA_CACHE_HEADER_INCLUDED
