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
#ifndef OPM_PY_LIVE_RESERVOIR_DATA_IMPL_HEADER_INCLUDED
#define OPM_PY_LIVE_RESERVOIR_DATA_IMPL_HEADER_INCLUDED

#include <dune/grid/common/rangegenerators.hh>

#include <opm/input/eclipse/EclipseState/Grid/FaceDir.hpp>
#include <opm/input/eclipse/EclipseState/Grid/NNC.hpp>
#include <opm/input/eclipse/Schedule/Well/Well.hpp>
#include <opm/material/common/MathToolbox.hpp>
#include <opm/models/common/multiphasebaseproperties.hh>
#include <opm/models/discretization/common/fvbaseproperties.hh>
#include <opm/models/utils/basicproperties.hh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace Opm::Pybind {

namespace LiveReservoirDataDetail {

inline void raisePythonError(PyObject* type, const std::string& message)
{
    PyErr_SetString(type, message.c_str());
    throw py::error_already_set();
}

template<class Value>
py::array_t<Value> makeArray(const std::size_t size)
{
    const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(size)};
    const std::vector<py::ssize_t> strides{static_cast<py::ssize_t>(sizeof(Value))};
    return py::array_t<Value>(shape, strides);
}

template<class Value>
py::array_t<Value> copyToArray(const std::vector<Value>& values)
{
    auto result = makeArray<Value>(values.size());
    auto result_mut = result.template mutable_unchecked<1>();
    for (std::size_t index = 0; index < values.size(); ++index) {
        result_mut(index) = values[index];
    }
    return result;
}

inline py::tuple tupleFromStrings(const std::vector<std::string>& values)
{
    py::tuple result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = py::cast(values[index]);
    }
    return result;
}

inline void validateCompletionConnxt(const std::string& well_name,
                                     const std::vector<long long>& connxt)
{
    const auto num_perf = connxt.size();
    std::vector<int> state(num_perf, 0);
    for (std::size_t index = 0; index < num_perf; ++index) {
        const auto parent = connxt[index];
        if (parent < 0 || parent > static_cast<long long>(num_perf)) {
            throw std::runtime_error(
                "Invalid live well completion topology for '" + well_name + "': parent index out of range.");
        }
    }

    for (std::size_t start = 0; start < num_perf; ++start) {
        std::size_t node = start;
        std::vector<std::size_t> path;
        while (state[node] == 0) {
            state[node] = 1;
            path.push_back(node);
            const auto parent = connxt[node];
            if (parent == 0) {
                break;
            }
            node = static_cast<std::size_t>(parent - 1);
        }
        if (state[node] == 1 && connxt[node] != 0) {
            throw std::runtime_error(
                "Invalid live well completion topology for '" + well_name + "': cycle detected.");
        }
        for (const auto visited : path) {
            state[visited] = 2;
        }
    }
}

inline std::vector<long long> linearCompletionConnxt(const std::size_t num_perf)
{
    std::vector<long long> connxt(num_perf, 0);
    for (std::size_t perf_index = 1; perf_index < num_perf; ++perf_index) {
        connxt[perf_index] = static_cast<long long>(perf_index);
    }
    return connxt;
}

template<class PerfData>
std::vector<long long> buildCompletionConnxt(const std::string& well_name,
                                             const Opm::Well& well,
                                             const PerfData& perf_data)
{
    auto connxt = linearCompletionConnxt(perf_data.size());
    if (!well.isMultiSegment()) {
        validateCompletionConnxt(well_name, connxt);
        return connxt;
    }

    const auto& connections = well.getConnections();
    const auto& segments = well.getSegments();
    std::unordered_map<int, std::size_t> segment_to_perf;
    std::vector<int> perf_segment(perf_data.size(), 0);
    for (std::size_t perf_index = 0; perf_index < perf_data.size(); ++perf_index) {
        const auto ecl_index = static_cast<std::size_t>(perf_data.ecl_index[perf_index]);
        const auto segment = connections[ecl_index].segment();
        perf_segment[perf_index] = segment;
        if (segment <= 0) {
            continue;
        }
        const auto [_, inserted] = segment_to_perf.emplace(segment, perf_index);
        if (!inserted) {
            throw std::runtime_error(
                "Cannot derive live completion tree for multi-segment well '" + well_name +
                "': multiple active completions share segment " + std::to_string(segment) + ".");
        }
    }

    std::fill(connxt.begin(), connxt.end(), 0);
    for (std::size_t perf_index = 0; perf_index < perf_data.size(); ++perf_index) {
        auto segment = perf_segment[perf_index];
        std::unordered_set<int> seen;
        while (segment > 0) {
            if (!seen.insert(segment).second) {
                throw std::runtime_error(
                    "Cannot derive live completion tree for multi-segment well '" + well_name +
                    "': segment outlet cycle detected.");
            }
            const auto segment_index = segments.segmentNumberToIndex(segment);
            const auto outlet_segment = segments[segment_index].outletSegment();
            if (outlet_segment <= 0) {
                break;
            }
            const auto parent_perf = segment_to_perf.find(outlet_segment);
            if (parent_perf != segment_to_perf.end()) {
                connxt[perf_index] = static_cast<long long>(parent_perf->second + 1);
                break;
            }
            segment = outlet_segment;
        }
    }

    validateCompletionConnxt(well_name, connxt);
    return connxt;
}

} // namespace LiveReservoirDataDetail

template<class TypeTag>
struct PyLiveReservoirData
{
    using Base = PyBaseSimulator<TypeTag>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;

    static bool hasCompletedConvergedTimestep(const Base& self)
    {
        return self.getFlowMain().getConvergedTimestepCount() > 0;
    }

    static bool isWaterReservoirFaceRateName(const std::string& name)
    {
        return name == "WATER_RESERVOIR_FACE_RATE_I+" ||
               name == "WATER_RESERVOIR_FACE_RATE_J+" ||
               name == "WATER_RESERVOIR_FACE_RATE_K+";
    }

    static void requireInitialized(const Base& self, const std::string& method)
    {
        if (!self.has_run_init_ || self.simulator_ == nullptr) {
            throw std::logic_error(method + "() called before step_init()");
        }
    }

    static void requireSerial(const Base& self, const std::string& method)
    {
        if (self.simulator_->vanguard().grid().comm().size() > 1) {
            LiveReservoirDataDetail::raisePythonError(
                PyExc_NotImplementedError,
                method + "() is only available for serial local OPM runs.");
        }
    }

    static void requireCompletedConvergedTimestep(const Base& self, const std::string& name)
    {
        if (!hasCompletedConvergedTimestep(self)) {
            LiveReservoirDataDetail::raisePythonError(
                PyExc_KeyError,
                name + " is only available after a completed converged timestep.");
        }
    }

    static py::dict getWaterReservoirFaceRateArrays(Base& self,
                                                    const bool want_i,
                                                    const bool want_j,
                                                    const bool want_k)
    {
        requireCompletedConvergedTimestep(self, "WATER_RESERVOIR_FACE_RATE_*");
        if (!FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)) {
            LiveReservoirDataDetail::raisePythonError(
                PyExc_KeyError, "Water phase is inactive for reservoir face-rate export.");
        }

        const int x_face_id = FaceDir::ToIntersectionIndex(FaceDir::DirEnum::XPlus);
        const int y_face_id = FaceDir::ToIntersectionIndex(FaceDir::DirEnum::YPlus);
        const int z_face_id = FaceDir::ToIntersectionIndex(FaceDir::DirEnum::ZPlus);

        using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;
        const auto& simulator = *self.simulator_;
        const auto num_dof = simulator.model().numGridDof();
        std::vector<double> flow_i(want_i ? num_dof : 0, 0.0);
        std::vector<double> flow_j(want_j ? num_dof : 0, 0.0);
        std::vector<double> flow_k(want_k ? num_dof : 0, 0.0);
        ElementContext elem_ctx(simulator);
        constexpr auto time_idx = 0u;

        for (const auto& elem : elements(simulator.gridView())) {
            elem_ctx.updateStencil(elem);
            elem_ctx.updateAllIntensiveQuantities();
            elem_ctx.updateExtensiveQuantities(time_idx);

            const auto& stencil = elem_ctx.stencil(time_idx);
            for (unsigned scvf_idx = 0; scvf_idx < elem_ctx.numInteriorFaces(time_idx); ++scvf_idx) {
                const auto& face = stencil.interiorFace(scvf_idx);
                const auto dir_id = face.dirId();
                if (!((want_i && dir_id == x_face_id) ||
                      (want_j && dir_id == y_face_id) ||
                      (want_k && dir_id == z_face_id))) {
                    continue;
                }

                const auto global_dof_idx = elem_ctx.globalSpaceIndex(face.interiorIndex(), time_idx);
                const auto& ext_quants = elem_ctx.extensiveQuantities(scvf_idx, time_idx);
                const auto alpha = getValue(ext_quants.extrusionFactor()) * face.area();
                const auto value = alpha * getValue(ext_quants.volumeFlux(FluidSystem::waterPhaseIdx));

                if (want_i && dir_id == x_face_id) {
                    flow_i[global_dof_idx] = value;
                }
                else if (want_j && dir_id == y_face_id) {
                    flow_j[global_dof_idx] = value;
                }
                else if (want_k && dir_id == z_face_id) {
                    flow_k[global_dof_idx] = value;
                }
            }
        }

        py::dict result;
        if (want_i) {
            result["WATER_RESERVOIR_FACE_RATE_I+"] = LiveReservoirDataDetail::copyToArray(flow_i);
        }
        if (want_j) {
            result["WATER_RESERVOIR_FACE_RATE_J+"] = LiveReservoirDataDetail::copyToArray(flow_j);
        }
        if (want_k) {
            result["WATER_RESERVOIR_FACE_RATE_K+"] = LiveReservoirDataDetail::copyToArray(flow_k);
        }
        return result;
    }

    static py::array_t<long long> getActiveToGlobalCells(Base& self)
    {
        requireInitialized(self, "get_active_to_global_cells");
        requireSerial(self, "get_active_to_global_cells");

        const auto num_dof = self.simulator_->model().numGridDof();
        const auto& ecl_state = self.simulator_->vanguard().eclState();
        const auto& input_grid = ecl_state.getInputGrid();
        const auto& input_actnum = input_grid.getACTNUM();
        const auto global_porv = ecl_state.fieldProps().porv(true);

        std::vector<int> active_to_global;
        active_to_global.reserve(num_dof);
        const auto nx = input_grid.getNX();
        const auto ny = input_grid.getNY();
        for (std::size_t global_cell = 0; global_cell < input_actnum.size(); ++global_cell) {
            if (input_actnum[global_cell] == 0) {
                continue;
            }
            const auto i = global_cell % nx;
            const auto j = (global_cell / nx) % ny;
            const auto k = global_cell / (nx * ny);
            if (input_grid.cellActiveAfterMINPV(i, j, k, global_porv[global_cell])) {
                active_to_global.push_back(static_cast<int>(global_cell));
            }
        }

        if (active_to_global.size() != static_cast<std::size_t>(num_dof)) {
            throw std::runtime_error("Reconstructed active-cell map does not match live simulator arrays.");
        }
        if (std::unordered_set<int>(active_to_global.begin(), active_to_global.end()).size()
            != active_to_global.size()) {
            throw std::runtime_error("Reconstructed active-cell map contains duplicate cells.");
        }

        auto values = LiveReservoirDataDetail::makeArray<long long>(active_to_global.size());
        auto values_mut = values.template mutable_unchecked<1>();
        for (std::size_t active_index = 0; active_index < active_to_global.size(); ++active_index) {
            values_mut(active_index) = static_cast<long long>(active_to_global[active_index]);
        }
        return values;
    }

    static py::tuple availableLiveCellArrays(Base& self)
    {
        requireInitialized(self, "available_live_cell_arrays");
        std::vector<std::string> names{"RPORV", "PRESSURE"};
        if (FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)) {
            names.push_back("SWAT");
            if (hasCompletedConvergedTimestep(self)) {
                names.push_back("WATER_RESERVOIR_FACE_RATE_I+");
                names.push_back("WATER_RESERVOIR_FACE_RATE_J+");
                names.push_back("WATER_RESERVOIR_FACE_RATE_K+");
            }
        }
        if (FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)) {
            names.push_back("SGAS");
        }
        if (FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)) {
            names.push_back("SOIL");
        }
        names.push_back("TEMP");
        return LiveReservoirDataDetail::tupleFromStrings(names);
    }

    static py::array_t<double> getLiveCellArray(Base& self, const std::string& name)
    {
        requireInitialized(self, "get_live_cell_array");
        if (name == "RPORV") {
            const auto porosity = self.getMaterialState().getPorosity();
            const auto volumes = self.getMaterialState().getCellVolumes();
            if (porosity.size() != volumes.size()) {
                throw std::runtime_error("Porosity and cell-volume array sizes differ for RPORV export.");
            }
            std::vector<double> rporv(porosity.size(), 0.0);
            for (std::size_t index = 0; index < rporv.size(); ++index) {
                rporv[index] = porosity[index] * volumes[index];
            }
            return LiveReservoirDataDetail::copyToArray(rporv);
        }
        if (name == "SWAT") {
            if (!FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)) {
                LiveReservoirDataDetail::raisePythonError(PyExc_KeyError, "Water phase is inactive.");
            }
            return LiveReservoirDataDetail::copyToArray(
                self.getFluidState().getFluidStateVariable("Sw"));
        }
        if (name == "SGAS") {
            if (!FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)) {
                LiveReservoirDataDetail::raisePythonError(PyExc_KeyError, "Gas phase is inactive.");
            }
            return LiveReservoirDataDetail::copyToArray(
                self.getFluidState().getFluidStateVariable("Sg"));
        }
        if (name == "SOIL") {
            if (!FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)) {
                LiveReservoirDataDetail::raisePythonError(PyExc_KeyError, "Oil phase is inactive.");
            }
            return LiveReservoirDataDetail::copyToArray(
                self.getFluidState().getFluidStateVariable("So"));
        }
        if (name == "TEMP") {
            return LiveReservoirDataDetail::copyToArray(
                self.getFluidState().getFluidStateVariable("T"));
        }
        if (name == "PRESSURE") {
            if (FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)) {
                return LiveReservoirDataDetail::copyToArray(
                    self.getFluidState().getFluidStateVariable("po"));
            }
            if (FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)) {
                return LiveReservoirDataDetail::copyToArray(
                    self.getFluidState().getFluidStateVariable("pw"));
            }
            if (FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)) {
                return LiveReservoirDataDetail::copyToArray(
                    self.getFluidState().getFluidStateVariable("pg"));
            }
        }
        if (isWaterReservoirFaceRateName(name)) {
            const auto arrays = getWaterReservoirFaceRateArrays(
                self,
                name == "WATER_RESERVOIR_FACE_RATE_I+",
                name == "WATER_RESERVOIR_FACE_RATE_J+",
                name == "WATER_RESERVOIR_FACE_RATE_K+");
            return arrays[py::str(name)].template cast<py::array_t<double>>();
        }
        LiveReservoirDataDetail::raisePythonError(
            PyExc_KeyError,
            "Unknown live cell array: " + name);
        return {};
    }

    static py::dict getLiveCellArrays(Base& self, const std::vector<std::string>& names)
    {
        py::dict result;
        bool want_i = false;
        bool want_j = false;
        bool want_k = false;
        for (const auto& name : names) {
            if (name == "WATER_RESERVOIR_FACE_RATE_I+") {
                want_i = true;
            }
            else if (name == "WATER_RESERVOIR_FACE_RATE_J+") {
                want_j = true;
            }
            else if (name == "WATER_RESERVOIR_FACE_RATE_K+") {
                want_k = true;
            }
            else {
                result[py::str(name)] = getLiveCellArray(self, name);
            }
        }
        if (want_i || want_j || want_k) {
            const auto arrays = getWaterReservoirFaceRateArrays(self, want_i, want_j, want_k);
            if (want_i) {
                result["WATER_RESERVOIR_FACE_RATE_I+"] = arrays["WATER_RESERVOIR_FACE_RATE_I+"];
            }
            if (want_j) {
                result["WATER_RESERVOIR_FACE_RATE_J+"] = arrays["WATER_RESERVOIR_FACE_RATE_J+"];
            }
            if (want_k) {
                result["WATER_RESERVOIR_FACE_RATE_K+"] = arrays["WATER_RESERVOIR_FACE_RATE_K+"];
            }
        }
        return result;
    }

    static py::dict getLiveWellTopology(Base& self)
    {
        requireInitialized(self, "get_live_well_topology");
        requireSerial(self, "get_live_well_topology");

        const auto& well_model = self.simulator_->problem().wellModel();
        const auto& live_wells = well_model.localNonshutWells();
        const auto& well_state = well_model.wellState();
        const auto num_wells = well_state.size();
        std::size_t num_perfs = 0;
        for (std::size_t well_index = 0; well_index < num_wells; ++well_index) {
            num_perfs += well_state.well(well_index).perf_data.size();
        }

        py::tuple well_names(num_wells);
        auto perf_well_index = LiveReservoirDataDetail::makeArray<long long>(num_perfs);
        auto perf_i = LiveReservoirDataDetail::makeArray<long long>(num_perfs);
        auto perf_j = LiveReservoirDataDetail::makeArray<long long>(num_perfs);
        auto perf_k = LiveReservoirDataDetail::makeArray<long long>(num_perfs);
        auto perf_connxt = LiveReservoirDataDetail::makeArray<long long>(num_perfs);
        auto perf_ecl_index = LiveReservoirDataDetail::makeArray<long long>(num_perfs);
        auto perf_well_index_mut = perf_well_index.template mutable_unchecked<1>();
        auto perf_i_mut = perf_i.template mutable_unchecked<1>();
        auto perf_j_mut = perf_j.template mutable_unchecked<1>();
        auto perf_k_mut = perf_k.template mutable_unchecked<1>();
        auto perf_connxt_mut = perf_connxt.template mutable_unchecked<1>();
        auto perf_ecl_index_mut = perf_ecl_index.template mutable_unchecked<1>();

        std::size_t perf_offset = 0;
        for (std::size_t well_index = 0; well_index < num_wells; ++well_index) {
            const auto& ws = well_state.well(well_index);
            well_names[well_index] = py::cast(well_state.name(well_index));

            decltype(live_wells.front().get()) live_well = nullptr;
            for (const auto& well_ptr : live_wells) {
                if (static_cast<std::size_t>(well_ptr->indexOfWell()) == well_index) {
                    live_well = well_ptr.get();
                    break;
                }
            }

            const auto& perf_data = ws.perf_data;
            auto connxt = LiveReservoirDataDetail::linearCompletionConnxt(perf_data.size());
            if (live_well != nullptr) {
                connxt = LiveReservoirDataDetail::buildCompletionConnxt(
                    well_state.name(well_index), live_well->wellEcl(), perf_data);
            }
            for (std::size_t perf_index = 0; perf_index < perf_data.size(); ++perf_index, ++perf_offset) {
                std::array<int, 3> coords{};
                self.simulator_->vanguard().cartesianCoordinate(perf_data.cell_index[perf_index], coords);
                perf_well_index_mut(perf_offset) = static_cast<long long>(well_index);
                perf_i_mut(perf_offset) = coords[0];
                perf_j_mut(perf_offset) = coords[1];
                perf_k_mut(perf_offset) = coords[2];
                perf_connxt_mut(perf_offset) = connxt[perf_index];
                perf_ecl_index_mut(perf_offset) = static_cast<long long>(perf_data.ecl_index[perf_index]);
            }
        }

        py::dict result;
        result["well_names"] = well_names;
        result["perf_well_index"] = perf_well_index;
        result["perf_i"] = perf_i;
        result["perf_j"] = perf_j;
        result["perf_k"] = perf_k;
        result["perf_connxt"] = perf_connxt;
        result["perf_ecl_index"] = perf_ecl_index;
        return result;
    }

    static py::dict getLiveWellRates(Base& self)
    {
        requireInitialized(self, "get_live_well_rates");
        requireSerial(self, "get_live_well_rates");
        if (!FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)) {
            LiveReservoirDataDetail::raisePythonError(
                PyExc_KeyError, "Live well water-rate export requires an active water phase.");
        }

        const auto& well_model = self.simulator_->problem().wellModel();
        const auto& live_wells = well_model.localNonshutWells();
        const auto& well_state = well_model.wellState();
        const int water_phase_idx = FluidSystem::waterPhaseIdx;
        const int water_active_idx =
            well_state.phaseUsageInfo().canonicalToActivePhaseIdx(water_phase_idx);
        using Scalar = typename FluidSystem::Scalar;

        const auto num_wells = well_state.size();
        std::size_t num_perfs = 0;
        for (std::size_t well_index = 0; well_index < num_wells; ++well_index) {
            num_perfs += well_state.well(well_index).perf_data.size();
        }

        auto well_is_open = LiveReservoirDataDetail::makeArray<bool>(num_wells);
        auto well_is_injector = LiveReservoirDataDetail::makeArray<bool>(num_wells);
        auto well_water_surface_rate = LiveReservoirDataDetail::makeArray<double>(num_wells);
        auto well_water_reservoir_rate = LiveReservoirDataDetail::makeArray<double>(num_wells);
        auto well_is_open_mut = well_is_open.template mutable_unchecked<1>();
        auto well_is_injector_mut = well_is_injector.template mutable_unchecked<1>();
        auto well_rate_mut = well_water_surface_rate.template mutable_unchecked<1>();
        auto well_reservoir_rate_mut = well_water_reservoir_rate.template mutable_unchecked<1>();

        auto perf_water_surface_rate = LiveReservoirDataDetail::makeArray<double>(num_perfs);
        auto perf_water_reservoir_rate = LiveReservoirDataDetail::makeArray<double>(num_perfs);
        auto perf_rate_mut = perf_water_surface_rate.template mutable_unchecked<1>();
        auto perf_reservoir_rate_mut = perf_water_reservoir_rate.template mutable_unchecked<1>();

        std::size_t perf_offset = 0;
        for (std::size_t well_index = 0; well_index < num_wells; ++well_index) {
            const auto& ws = well_state.well(well_index);
            well_is_open_mut(well_index) = ws.status == WellStatus::OPEN;
            well_is_injector_mut(well_index) = !ws.producer;
            well_rate_mut(well_index) = -ws.surface_rates[water_active_idx];
            well_reservoir_rate_mut(well_index) = -ws.reservoir_rates[water_active_idx];

            decltype(live_wells.front().get()) live_well = nullptr;
            for (const auto& well_ptr : live_wells) {
                if (static_cast<std::size_t>(well_ptr->indexOfWell()) == well_index) {
                    live_well = well_ptr.get();
                    break;
                }
            }

            const auto& perf_data = ws.perf_data;
            for (std::size_t perf_index = 0; perf_index < perf_data.size(); ++perf_index, ++perf_offset) {
                perf_rate_mut(perf_offset) =
                    -perf_data.phase_rates[perf_index * well_state.numPhases() + water_active_idx];

                std::vector<Scalar> surface_rates_perf(well_state.numPhases(), 0.0);
                std::vector<Scalar> voidage_rates_perf(well_state.numPhases(), 0.0);
                for (int phase = 0; phase < well_state.numPhases(); ++phase) {
                    surface_rates_perf[phase] =
                        perf_data.phase_rates[perf_index * well_state.numPhases() + phase];
                }
                if (live_well != nullptr) {
                    live_well->rateConverter().calcReservoirVoidageRates(
                        /*fipreg*/ 0,
                        live_well->pvtRegionIdx(),
                        surface_rates_perf,
                        voidage_rates_perf);
                }
                else if (ws.status == WellStatus::OPEN) {
                    throw std::runtime_error(
                        "Cannot compute live reservoir completion rates for open well '" + ws.name + "'.");
                }
                perf_reservoir_rate_mut(perf_offset) = -voidage_rates_perf[water_active_idx];
            }
        }

        py::dict result;
        result["well_is_open"] = well_is_open;
        result["well_is_injector"] = well_is_injector;
        result["well_water_surface_rate"] = well_water_surface_rate;
        result["well_water_reservoir_rate"] = well_water_reservoir_rate;
        result["perf_water_surface_rate"] = perf_water_surface_rate;
        result["perf_water_reservoir_rate"] = perf_water_reservoir_rate;
        return result;
    }

    static py::dict getLiveNncTopology(Base& self)
    {
        requireInitialized(self, "get_live_nnc_topology");
        requireSerial(self, "get_live_nnc_topology");

        const auto& output_nnc = self.simulator_->problem().eclWriter().getOutputNnc();
        if (output_nnc.size() > 1) {
            LiveReservoirDataDetail::raisePythonError(
                PyExc_NotImplementedError,
                "get_live_nnc_topology() currently exports only main-grid NNC topology.");
        }

        const auto num_nnc = output_nnc.empty() ? 0 : output_nnc.front().size();
        auto nnc1 = LiveReservoirDataDetail::makeArray<long long>(num_nnc);
        auto nnc2 = LiveReservoirDataDetail::makeArray<long long>(num_nnc);
        auto nnc1_mut = nnc1.template mutable_unchecked<1>();
        auto nnc2_mut = nnc2.template mutable_unchecked<1>();

        if (!output_nnc.empty()) {
            const auto& main_grid_nnc = output_nnc.front();
            for (std::size_t index = 0; index < main_grid_nnc.size(); ++index) {
                nnc1_mut(index) = static_cast<long long>(main_grid_nnc[index].cell1) + 1;
                nnc2_mut(index) = static_cast<long long>(main_grid_nnc[index].cell2) + 1;
            }
        }

        py::dict result;
        result["nnc1"] = nnc1;
        result["nnc2"] = nnc2;
        return result;
    }

    static py::dict getLiveNncRates(Base& self)
    {
        requireInitialized(self, "get_live_nnc_rates");
        requireSerial(self, "get_live_nnc_rates");
        if (!FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)) {
            LiveReservoirDataDetail::raisePythonError(
                PyExc_KeyError, "Live NNC water-rate export requires an active water phase.");
        }

        const auto& output_nnc = self.simulator_->problem().eclWriter().getOutputNnc();
        if (output_nnc.size() > 1) {
            LiveReservoirDataDetail::raisePythonError(
                PyExc_NotImplementedError,
                "get_live_nnc_rates() currently exports only main-grid NNC rates.");
        }

        const auto num_nnc = output_nnc.empty() ? 0 : output_nnc.front().size();
        auto water_reservoir_rate = LiveReservoirDataDetail::makeArray<double>(num_nnc);
        auto water_reservoir_rate_mut = water_reservoir_rate.template mutable_unchecked<1>();

        if (num_nnc > 0) {
            const auto& flows = self.simulator_->problem().eclWriter().outputModule().getFlows();
            if (!flows.hasFloresn()) {
                LiveReservoirDataDetail::raisePythonError(
                    PyExc_KeyError, "Live NNC reservoir water rates require active FLORESN output data.");
            }
            const auto& floresn = flows.getFloresn();
            const auto water = std::find_if(floresn.begin(), floresn.end(), [](const auto& values) {
                return values.name == "FLRWATN+";
            });
            if (water == floresn.end() || water->values.empty()) {
                LiveReservoirDataDetail::raisePythonError(
                    PyExc_KeyError, "Live NNC reservoir water rates require FLRWATN+ values.");
            }
            if (water->values.size() != num_nnc) {
                throw std::runtime_error("FLRWATN+ length does not match fixed live NNC topology.");
            }
            for (std::size_t index = 0; index < num_nnc; ++index) {
                water_reservoir_rate_mut(index) = water->values[index];
            }
        }

        py::dict result;
        result["water_reservoir_rate"] = water_reservoir_rate;
        return result;
    }
};

template<class TypeTag, class PyClass>
void bindLiveReservoirData(PyClass& cls)
{
    using Data = PyLiveReservoirData<TypeTag>;
    cls.def("get_active_to_global_cells", &Data::getActiveToGlobalCells,
            "Return zero-based global Cartesian cell indices for active live arrays.")
        .def("available_live_cell_arrays", &Data::availableLiveCellArrays,
             "Return names of cell arrays available from the current live simulator state.")
        .def("get_live_cell_array", &Data::getLiveCellArray, py::arg("name"),
             "Return one live cell array by name.")
        .def("get_live_cell_arrays", &Data::getLiveCellArrays, py::arg("names"),
             "Return several live cell arrays by name.")
        .def("get_live_well_topology", &Data::getLiveWellTopology,
             "Return fixed live well and perforation topology.")
        .def("get_live_well_rates", &Data::getLiveWellRates,
             "Return current live well and perforation water rates.")
        .def("get_live_nnc_topology", &Data::getLiveNncTopology,
             "Return fixed live NNC topology.")
        .def("get_live_nnc_rates", &Data::getLiveNncRates,
             "Return current live NNC reservoir water rates.");
}

} // namespace Opm::Pybind

#endif // OPM_PY_LIVE_RESERVOIR_DATA_IMPL_HEADER_INCLUDED
