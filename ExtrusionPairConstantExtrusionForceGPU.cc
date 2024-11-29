#include "ExtrusionPairGPU.h"
#include "EvaluatorPairConstantExtrusionForce.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_ExtrusionPairConstantExtrusionForceGPU(pybind11::module& m)
    {
    export_ExtrusionPairGPU<EvaluatorPairConstantExtrusionForce>(m, "ExtrusionPairConstantExtrusionForceGPU");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
