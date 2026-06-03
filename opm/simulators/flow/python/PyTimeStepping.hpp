// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#ifndef OPM_PYTIME_STEPPING_HPP
#define OPM_PYTIME_STEPPING_HPP

#include <opm/simulators/flow/python/PyBaseSimulator.hpp>

namespace Opm::Pybind {

template<class TypeTag>
struct PyTimeStepping;

template<class TypeTag, class PyClass>
void bindTimeStepping(PyClass& cls);

} // namespace Opm::Pybind

#include <opm/simulators/flow/python/PyTimeStepping_impl.hpp>

#endif // OPM_PYTIME_STEPPING_HPP
