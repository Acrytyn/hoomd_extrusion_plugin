#include "EvaluatorPairConstantExtrusionForce.h"
#include "ExtrusionPair.h"
#include <pybind11/pybind11.h>
#ifdef ENABLE_HIP
#include "ExtrusionPairGPU.h"
#endif


namespace hoomd
    {
namespace md
    {

// specify the python module. Note that the name must explicitly match the PROJECT() name provided
// in CMakeLists (with an underscore in front)
PYBIND11_MODULE(_extrusion_plugin, m)
    {
    detail::export_ExtrusionPair<EvaluatorPairConstantExtrusionForce>(m, "ExtrusionPairConstantExtrusionForce");
#ifdef ENABLE_HIP
    detail::export_ExtrusionPairGPU<EvaluatorPairConstantExtrusionForce>(m, "ExtrusionPairConstantExtrusionForceGPU");
#endif
    }

    } // end namespace md
    } // end namespace hoomd

