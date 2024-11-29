#include "ExtrusionPair.h"
#include "EvaluatorPairConstantExtrusionForce.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

// Template specification for anisotropic extrusion pair potential. A specific
// template instance is needed since we expose the shape as just orientation in Python
// when the default behavior exposes setting and getting the shape through
// 'shape'.
template<>
inline void export_ExtrusionPair<EvaluatorPairConstantExtrusionForce>(pybind11::module& m,
                                                           const std::string& name)
    {
    pybind11::class_<ExtrusionPair<EvaluatorPairConstantExtrusionForce>,
                     ForceCompute,
                     std::shared_ptr<ExtrusionPair<EvaluatorPairConstantExtrusionForce>>>
        extrusionpair(m, name.c_str());
    extrusionpair
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("setParams", &ExtrusionPair<EvaluatorPairConstantExtrusionForce>::setParamsPython)
        .def("getParams", &ExtrusionPair<EvaluatorPairConstantExtrusionForce>::getParamsPython)
        .def("setOrientation", &ExtrusionPair<EvaluatorPairConstantExtrusionForce>::setShapePython)
        .def("getOrientation", &ExtrusionPair<EvaluatorPairConstantExtrusionForce>::getShapePython)
        .def("setRCut", &ExtrusionPair<EvaluatorPairConstantExtrusionForce>::setRCutPython)
        .def("getRCut", &ExtrusionPair<EvaluatorPairConstantExtrusionForce>::getRCut)
        .def_property("mode",
                      &ExtrusionPair<EvaluatorPairConstantExtrusionForce>::getShiftMode,
                      &ExtrusionPair<EvaluatorPairConstantExtrusionForce>::setShiftModePython)
        .def("getTypeShapesPy", &ExtrusionPair<EvaluatorPairConstantExtrusionForce>::getTypeShapesPy);
    }

void export_ExtrusionPairConstantExtrusionForce(pybind11::module& m)
    {
    export_ExtrusionPair<EvaluatorPairConstantExtrusionForce>(m, "ExtrusionPairConstantExtrusionForce");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd