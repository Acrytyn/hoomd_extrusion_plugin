#include "ExtrusionPairGPU.cuh"
#include "EvaluatorPairConstantExtrusionForce.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<EvaluatorPairConstantExtrusionForce>(
    const a_pair_args_t& pair_args,
    const EvaluatorPairConstantExtrusionForce::param_type* d_param,
    const EvaluatorPairConstantExtrusionForce::shape_type* d_shape_param);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
