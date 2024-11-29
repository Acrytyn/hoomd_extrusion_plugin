r"""Extrusion pair forces. Derived from Anisotropic pair forces

Anisotropic pair force classes apply a force, torque, and virial on every
particle in the simulation state commensurate with the potential energy:

.. math::

    U_\mathrm{pair,total} = \frac{1}{2} \sum_{i=0}^\mathrm{N_particles-1}
                      \sum_{j \ne i, (i,j) \notin \mathrm{exclusions}}
                      U_\mathrm{pair}(r_{ij}, \mathbf{q}_i, \mathbf{q}_j)

`AnisotropicPair` applies cuttoffs, exclusions, and assigns per particle
energies and virials in the same manner as `hoomd.md.pair.Pair`

`AnisotropicPair` does not support the ``'xplor'`` shifting mode or the ``r_on``
parameter.
"""

from collections.abc import Sequence
import json
from numbers import Number

from hoomd.md.pair.pair import Pair
from hoomd.logging import log
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyIf, to_type_converter
from hoomd.extrusion_plugin import _extrusion_plugin

class ExtrusionPair(Pair):
    r"""Base class anisotropic pair force.

    `AnisotropicPair` is the base class for all anisotropic pair forces.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.

    Args:
        nlist (hoomd.md.nlist.NeighborList) : The neighbor list.
        default_r_cut (`float`, optional) : The default cutoff for the
            potential, defaults to ``None`` which means no cutoff
            :math:`[\mathrm{length}]`.
        mode (`str`, optional) : the energy shifting mode, defaults to "none".
    """

    _accepted_modes = ("none", "shift")
    #we need to redefine this for our external plugin so that hoomd does not look
    #into _md (line 63 in pair.py)
    _ext_module = _extrusion_plugin
    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, 0.0, mode)

    def _return_type_shapes(self):
        type_shapes = self._cpp_obj.getTypeShapesPy()
        ret = [json.loads(json_string) for json_string in type_shapes]
        return ret
    

class ConstantExtrusionForce(ExtrusionPair):
    r"""
    
    ExtrusionPair is the base class for extrusion forces

    ConstantForce class simply computes normal component to the plane of the rings and
    applies corresponding force to the bead in the vicinity of the ring.

    """

    _cpp_class_name = "ExtrusionPairConstantExtrusionForce"

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(magForce=float, placeHolder=float, len_keys=2))
        orientation = TypeParameter('orientation', 'particle_types',
                           TypeParameterDict((float, float, float), len_keys=1))
        self._extend_typeparam((params, orientation))

