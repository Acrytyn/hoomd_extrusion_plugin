#ifndef __PAIR_EVALUATOR_CONSTANTEXTRUSIONFORCE_H__
#define __PAIR_EVALUATOR_CONSTANTEXTRUSIONFORCE_H__

#ifndef __HIPCC__
#include <string>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include "hoomd/VectorMath.h"
#include <iostream>

/*! \file EvaluatorPairConstantExtrusionForce.h
    \brief Defines the dipole potential
*/

// need to declare these class methods with __device__ qualifiers when building
// in nvcc.  HOSTDEVICE is __host__ __device__ when included in nvcc and blank
// when included into the host compiler
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {
class EvaluatorPairConstantExtrusionForce
    {
    public:
    struct param_type
        {
        Scalar magForce;     //! The electrostatic energy scale.
        Scalar placeHolder; //! does nothing for now

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory
            allocation
        */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

        HOSTDEVICE param_type() : magForce(0), placeHolder(0) { }

#ifndef __HIPCC__

        param_type(pybind11::dict v, bool managed)
            {
            magForce = v["magForce"].cast<Scalar>();
            placeHolder = v["placeHolder"].cast<Scalar>();
            }

        pybind11::object toPython()
            {
            pybind11::dict v;
            v["magForce"] = magForce;
            v["placeHolder"] = placeHolder;
            return v;
            }

#endif
        }

#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    struct shape_type
        {
        vec3<Scalar> orientation; // direction of the center of mass of the ring

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
        */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

        HOSTDEVICE shape_type() : orientation {0, 0, 0} { }

#ifndef __HIPCC__

        shape_type(vec3<Scalar> orientation_, bool managed = false) : orientation(orientation_) { }

        shape_type(pybind11::object orientation_obj, bool managed)
            {
            auto orientation_ = (pybind11::tuple)orientation_obj;
            orientation = vec3<Scalar>(orientation_[0].cast<Scalar>(), orientation_[1].cast<Scalar>(), orientation_[2].cast<Scalar>());
            }

        pybind11::object toPython()
            {
            return pybind11::make_tuple(orientation.x, orientation.y, orientation.z);
            }
#endif // __HIPCC__

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const { }
#endif
        };

    //! Constructs the pair potential evaluator
    /*! \param _dr Displacement vector between particle centers of mass
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _quat_i Quaternion of i^{th} particle
        \param _quat_j Quaternion of j^{th} particle
        \param _magForce Magnitude of extrusion force
        \param _placeHolder Does nothing
        \param _params Per type pair parameters of this potential
    */
    HOSTDEVICE EvaluatorPairConstantExtrusionForce(Scalar3& _dr,
                                   Scalar4& _quat_i,
                                   Scalar4& _quat_j,
                                   Scalar _rcutsq,
                                   const param_type& _params)
        : dr(_dr), rcutsq(_rcutsq), q_i(0), q_j(0), quat_i(_quat_i), quat_j(_quat_j),
          orientation_i {0, 0, 0}, orientation_j {0, 0, 0}, magForce(_params.magForce), placeHolder(_params.placeHolder)
        {
        }

    //! Whether the pair potential uses shape.
    HOSTDEVICE static bool needsShape()
        {
        return true;
        }

    //! Whether the pair potential needs particle tags.
    HOSTDEVICE static bool needsTags()
        {
        return false;
        }

    //! whether pair potential requires charges
    HOSTDEVICE static bool needsCharge()
        {
        return false;
        }

    /// Whether the potential implements the energy_shift parameter
    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return false;
        }

    //! Accept the optional shape values
    /*! \param shape_i Shape of particle i
        \param shape_j Shape of particle j
    */
    HOSTDEVICE void setShape(const shape_type* shapei, const shape_type* shapej)
        {
        orientation_i = shapei->orientation;
        orientation_j = shapej->orientation;
        }

    //! Accept the optional tags
    /*! \param tag_i Tag of particle i
        \param tag_j Tag of particle j
    */
    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) { }

    //! Accept the optional charge values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    HOSTDEVICE void setCharge(Scalar qi, Scalar qj)
        {
        q_i = qi;
        q_j = qj;
        }

    
    //! Evaluate the force and energy
    /*! \param force Output parameter to write the computed force.
        \param pair_eng Output parameter to write the computed pair energy.
        \param energy_shift If true, the potential must be shifted so that
            V(r) is continuous at the cutoff.
        \param torque_i The torque exerted on the i^th particle.
        \param torque_j The torque exerted on the j^th particle.
        \return True if they are evaluated or false if they are not because
            we are beyond the cutoff.
    */
    HOSTDEVICE bool evaluate(Scalar3& force,
                             Scalar& pair_eng,
                             bool energy_shift,
                             Scalar3& torque_i,
                             Scalar3& torque_j)
        {
        vec3<Scalar> rvec(dr);
        Scalar rsq = dot(rvec, rvec);

        if (rsq > rcutsq)
            return false;

        // convert dipole vector in the body frame of each particle to space
        // frame
        vec3<Scalar> direction_i = rotate(quat<Scalar>(quat_i), orientation_i);
        vec3<Scalar> direction_j = rotate(quat<Scalar>(quat_j), orientation_j);

        direction_i = normalize(direction_i);
        //direction_j = normalize(direction_j);
        // #ifndef __HIPCC__  // This ensures that this part only runs on the host
        // std::cout << "direction_i: (" 
        //         << direction_i.x << ", "
        //         << direction_i.y << ", "
        //         << direction_i.z << " , "
        //         << magForce      << " , "
        //         << rsq           << ")" << std::endl;
        // #endif

        //  #ifndef __HIPCC__  // This ensures that this part only runs on the host
        // std::cout << "quat_i: (" 
        //         << quat_i.x << ", "
        //         << quat_i.y << ", "
        //         << quat_i.z << ", "
        //         << quat_i.w << magForce << ")"<< std::endl;
        // #endif
        vec3<Scalar> f;
        vec3<Scalar> t_i;
        vec3<Scalar> t_j;
        Scalar e = Scalar(0.0);

        f += magForce * direction_i;

        t_i += vec3<double>(0, 0, 0);
        t_j += vec3<double>(0, 0, 0);

        force = vec_to_scalar3(f);
        torque_i = vec_to_scalar3(t_i);
        torque_j = vec_to_scalar3(t_j);
        pair_eng = e;
        return true;
        }

    DEVICE Scalar evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    //! Get the name of the potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return "dipole";
        }
    static std::string getShapeParamName()
        {
        return "Orientation";
        }
    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar3 dr;             //!< Stored vector pointing between particle centers of mass
    Scalar rcutsq;          //!< Stored rcutsq from the constructor
    Scalar q_i, q_j;        //!< Stored particle charges
    Scalar4 quat_i, quat_j; //!< Stored quaternion of ith and jth particle from constructor
    vec3<Scalar> orientation_i;      /// Magnetic moment for ith particle
    vec3<Scalar> orientation_j;      /// Magnetic moment for jth particle
    Scalar magForce;
    Scalar placeHolder;
    // const param_type &params;   //!< The pair potential parameters
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_DIPOLE_H__

