// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#ifndef OPM_PYTIME_STEPPING_IMPL_HPP
#define OPM_PYTIME_STEPPING_IMPL_HPP

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace Opm::Pybind {

namespace PyTimeSteppingDetail {

inline void raiseNotImplementedError(const std::string& message)
{
    PyErr_SetString(PyExc_NotImplementedError, message.c_str());
    throw py::error_already_set();
}

} // namespace PyTimeSteppingDetail

template<class TypeTag>
struct PyTimeStepping
{
    using Base = PyBaseSimulator<TypeTag>;

    static void requireInitialized(const Base& self, const std::string& method)
    {
        if (!self.has_run_init_) {
            throw std::logic_error(method + " requires run initialization");
        }
    }

    static void requireRunning(const Base& self, const std::string& method)
    {
        requireInitialized(self, method);
        if (self.has_run_cleanup_) {
            throw std::logic_error(method + " cannot be called after run cleanup");
        }
    }

    static void requireSerial(const Base& self, const std::string& method)
    {
        const auto& comm = self.getFlowMain().getSimulatorPtr()->vanguard().grid().comm();
        if (comm.size() != 1) {
            PyTimeSteppingDetail::raiseNotImplementedError(
                method + " is only available for serial Python simulator runs");
        }
    }

    static bool advanceReportTimestep(Base& self)
    {
        requireRunning(self, "advance_report_timestep()");
        if (self.checkSimulationFinished()) {
            return false;
        }
        return static_cast<bool>(self.getFlowMain().executeReportTimestep());
    }

    static double lastReportTimestepSeconds(const Base& self)
    {
        requireInitialized(self, "last_report_timestep_seconds()");
        return self.getFlowMain().getLastReportTimestepSize();
    }

    static bool advanceConvergedTimestep(Base& self)
    {
        requireRunning(self, "advance_converged_timestep()");
        requireSerial(self, "advance_converged_timestep()");
        if (self.checkSimulationFinished()) {
            return false;
        }
        return static_cast<bool>(self.getFlowMain().executeConvergedTimestep());
    }

    static double lastConvergedTimestepSeconds(const Base& self)
    {
        requireInitialized(self, "last_converged_timestep_seconds()");
        if (self.getFlowMain().getConvergedTimestepCount() == 0) {
            throw std::logic_error(
                "last_converged_timestep_seconds() requires a completed converged timestep");
        }
        return self.getFlowMain().getLastConvergedTimestepSize();
    }
};

template<class TypeTag, class PyClass>
void bindTimeStepping(PyClass& cls)
{
    cls.def("advance_report_timestep",
            &PyTimeStepping<TypeTag>::advanceReportTimestep,
            "Advance one report timestep and return false when the simulation has finished.\n\n"
            "This is the report-step cadence used by the traditional step()/advance() API.");

    cls.def("last_report_timestep_seconds",
            &PyTimeStepping<TypeTag>::lastReportTimestepSeconds,
            "Return the duration in seconds of the last completed report timestep.");

    cls.def("advance_converged_timestep",
            &PyTimeStepping<TypeTag>::advanceConvergedTimestep,
            "Advance one converged internal timestep and return false when the simulation has finished.\n\n"
            "Rejected nonlinear retries stay inside OPM. A returned true value means one timestep has "
            "been accepted into the simulator time history, including controlled accept-on-unconverged "
            "policies when enabled by the deck and simulator settings.");

    cls.def("last_converged_timestep_seconds",
            &PyTimeStepping<TypeTag>::lastConvergedTimestepSeconds,
            "Return the duration in seconds of the last completed converged internal timestep.");
}

} // namespace Opm::Pybind

#endif // OPM_PYTIME_STEPPING_IMPL_HPP
