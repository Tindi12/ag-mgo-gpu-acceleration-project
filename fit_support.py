"""
Created:                        June 6th 2024
Substantial revision:           October 18th 2024
Minor updates:                  March 2025


This script contains many useful functions for Ag/MgO
    supports gasphase NPs
    checks for sanity of structures
    modifies structures
    classifies structures

    and many more!

    It is the most useful script I have ever written.       ##April 2026:   what the actual fuck??
    Some updates in the Active_Learning module. Check the "deprecated" function below


Provides basic utilities for NP work
BY LAW, this MUST not itself import any custom module except a more foundational utilities script,
e.g. ~/utilities/utilities.py
"""

#!/usr/bin/env python
from ase.build.surface import fcc111, fcc100, fcc110, fcc211
import statistics
import functools
from ase.io import read, write, iread
import numpy as np
from tqdm import tqdm
from ase import Atoms, Atom
import math
from io import StringIO
from ascii_colors import ASCIIColors
from sys import argv, exit, stdout
from typing import Tuple, Literal, List, Union
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.constraints import FixAtoms
import warnings
import matplotlib.pyplot as plt
from copy import deepcopy
from os import system
from ase.geometry import get_duplicate_atoms

LATERAL_SPACING = 15  # 14.5  # spacing between between each nanoparticles repeated image. Note, though that
# this is a little lower than the spacing, due to the ceil function being used
Z_SPACING = 15  # 13.5  # spacing in z direction
ADSORPTION_HEIGHT = 2.5  # adsorption height
LAYER_HEIGHT = 1.5  # heigth of first MgO layer for constraining in a 2-layer model
SECONDLAYER_HEIGHT = 3  # height of second-to-the-highest MgO layer
LAYERS = "two"  # Number of layers for the unit support
INTERFACE_SCALING = 1.3  # scaling factor for interfacial NP atoms' covalent radii
NANOPARTICLE_ELEMENT = "Ag"  # what kind of atom is the NP made of?
LOW_INDEX_FACET_BUILDERS = {
    (1, 1, 1): (fcc111, (4, 4)),
    (1, 0, 0): (fcc100, (4, 4)),
    (1, 1, 0): (fcc110, (4, 4)),
    (2, 1, 1): (fcc211, (6, 4)),
}
# AG_MGO_INTERFACE = (1,0,0)
Mg_O_HALF_LATTICE = 2.13  # half of MgO's lattice constant
TARGET = 10  # target for reducing vacuum and lateral spacings; needed to speed up DFT
MgO_SIDEWAYS_LAYER_TOLERANCE = 0.4 #A
LAYER_TOLERANCE = (
    1.2  # A. Every atom within this z of each other is in the same 'layer'
)
MgO_SIDEWAYS_LAYER_TOLERANCE = 0.4 #A
MgO_Z_LAYER_TOLERANCE = 1 #untested



def deprecated(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"""{function.__name__} is less functional than that in
            the Active_Learning suite. Please use that instead; or use ~/utilities/check_sanity.py""",
            DeprecationWarning,
            stacklevel=2,
        )
        return function(*args, **kwargs)

    return wrapper


def calc_vacuum(atoms) -> float:
    """
    calculate the amount of vacuum in the system, considering ALL atoms' z-position
    """
    z = atoms.positions[:, 2]
    minz = min(z)
    maxz = max(z)
    cell_height = atoms.cell.lengths()[2]
    vacuum = cell_height + minz - maxz

    return vacuum, minz, maxz


def set_vacuum(atoms, set_point: float = 15) -> Atoms:
    """
    set vacuum to a specified amount
    """
    current_vacuum, minz, maxz = calc_vacuum(atoms)
    current_cell_height = atoms.cell[2][2]
    atoms.cell[2][2] = current_cell_height + set_point - current_vacuum

    return atoms


def classify_into_layers(
    atoms: Atoms, tol: Optional[float] = None ,
    interested_system: Literal["NP", "Support"] = "NP",
    coordinate: Literal["x", "y", "z"] = "z",
) -> Tuple[Atoms, np.ndarray]:
    """
    Classify an atoms object into the layers each atom belongs to

            !!!! Warnings !!!!

    if interested_system == 'NP':
        atoms object SHOULD BE JUST THE NANOPARTICLE

    if interested_system == 'Support':
        atoms object SHOULD BE JUST THE SUPPORT

    returns:
        atoms:      same atoms object
        layers:     array of layer index, starting from 1
            i.e. layers=[1,5] means atom 0 is in the 1st layer and atom 1 is in the 5th
    """
    warnings.warn(
    f"""
    You requested {interested_system};
    I hope the supplied atoms object is of ONLY THAT SYSTEM!!""",
    stacklevel=2
    )
    #sanity
    valid_systems = {"NP", "Support"}
    if interested_system not in valid_systems:
        raise ValueError(
            f"interested_system must be one of {sorted(valid_systems)}; received {interested_system}"
        )
    axis_map = {"x": 0, "y": 1, "z": 2}
    if coordinate not in axis_map:
        raise ValueError(
            f"coordinate must be one of {sorted(axis_map)}; received {coordinate}"
        )
    if tol is None:
        if interested_system == "NP":
            tol = LAYER_TOLERANCE
        elif coordinate == "z":
            tol = MgO_Z_LAYER_TOLERANCE
        else:
            tol= MgO_SIDEWAYS_LAYER_TOLERANCE

    axis = axis_map[coordinate]
    zs = atoms.positions[:, axis]
    sorting_mask = np.argsort(zs)
    zs_sorted = zs[sorting_mask]

    layers_sorted = np.zeros(zs_sorted.shape, dtype=int)
    current_layer = 1  # interfacelayer = 1
    ref = zs_sorted[0]  # lowest z
    layers_sorted[0] = current_layer

    for i in range(1, len(zs_sorted)):
        if abs(zs_sorted[i] - ref) > tol:
            current_layer += (
                1  # if we are outside of the tolerance, we move to next layer
            )
            ref = zs_sorted[i]
        layers_sorted[i] = current_layer  # else we remain on current layer

    # unsort back to original atom order
    layers = np.empty(layers_sorted.shape, dtype=int)
    layers[sorting_mask] = layers_sorted

    return atoms, layers


def create_unit_support(layers: str = LAYERS) -> Atoms:
    """Creates a unit cell of the MgO support
    Either 2 layers (lower fixed), or 4 layers (lower three fixed)

    layers must be either 'two' or 'four'

    This uses PBE-D3BJ's lattice constant for MgO; works whether D3 is on Mg or not
    (i.e. 4.26 to 4.27A lattice constant
    Done in GPAW
    """

    # 2 layers, the bottom layer fixed
    two_layers = """Mg O
1.0000000000000000
    4.2602238819630642    0.0000000000000000    0.0000000000000000
    0.0000000000000003    4.2602238819630642    0.0000000000000000
    0.0000000000000000    0.0000000000000000   24.9107835868707248
Mg  O
  4   4
Selective dynamics
Cartesian
0.0000000000000021  0.0000000000000009 17.7806716458891927   F   F   F
2.1301119409815343  2.1301119409815330 17.7806716458891927   F   F   F
0.0000000000000026  2.1301119409815330 19.9107835868707248   T   T   T
2.1301119409815348  0.0000000000000011 19.9107835868707248   T   T   T
0.0000000000000023  2.1301119409815330 17.7806716458891927   F   F   F
2.1301119409815343  0.0000000000000009 17.7806716458891927   F   F   F
0.0000000000000025  0.0000000000000011 19.9107835868707248   T   T   T
2.1301119409815348  2.1301119409815334 19.9107835868707248   T   T   T
 """
    # 4 layers with the lower 3 layers fixed
    four_layers = """Mg O
1.0000000000000000
    4.2602238819630642    0.0000000000000000    0.0000000000000000
    0.0000000000000003    4.2602238819630642    0.0000000000000000
    0.0000000000000000    0.0000000000000000   24.9107835868707248
Mg  O
  8   8
Selective dynamics
Cartesian
0.0000000000000014  0.0000000000000006 13.5204477639261285   F   F   F
2.1301119409815334  2.1301119409815326 13.5204477639261285   F   F   F
0.0000000000000019  2.1301119409815326 15.6505597049076606   F   F   F
2.1301119409815339  0.0000000000000008 15.6505597049076606   F   F   F
0.0000000000000021  0.0000000000000009 17.7806716458891927   F   F   F
2.1301119409815343  2.1301119409815330 17.7806716458891927   F   F   F
0.0000000000000026  2.1301119409815330 19.9107835868707248   T   T   T
2.1301119409815348  0.0000000000000011 19.9107835868707248   T   T   T
0.0000000000000016  2.1301119409815326 13.5204477639261285   F   F   F
2.1301119409815334  0.0000000000000006 13.5204477639261285   F   F   F
0.0000000000000018  0.0000000000000008 15.6505597049076606   F   F   F
2.1301119409815339  2.1301119409815330 15.6505597049076606   F   F   F
0.0000000000000023  2.1301119409815330 17.7806716458891927   F   F   F
2.1301119409815343  0.0000000000000009 17.7806716458891927   F   F   F
0.0000000000000025  0.0000000000000011 19.9107835868707248   T   T   T
2.1301119409815348  2.1301119409815334 19.9107835868707248   T   T   T
 """

    if layers not in ("two", "four", "2", "4"):
        warnings.warn(
            f"""You have requested {layers} MgO layers
        but that option is invalid.
        Defaulting to two layers with the lower fixed""",
            category=UserWarning,
        )
        layers = "two"

    basic = StringIO(two_layers if layers == "two" else four_layers)
    unit_mgo = read(basic, format="vasp")

    return unit_mgo


def divider(
    atoms: Atoms, elements: Union[List[str], str] = NANOPARTICLE_ELEMENT
) -> Union[None, Tuple[Atoms]]:
    """
    Divide the structure into adsorbate and surface
    Preserve the constraints of the support; assume the adsorbate has none

    Inputs:
                Supported_adsorbate
                elements         (what element is the adsorbate?),
                                can also be a list of elements, e.g. ["C", "O"]
                                default = 'Ag'
    Returns:    adsorbate
                Bare_Surface
    """
    if not isinstance(atoms, Atoms):
        return None
    if not isinstance(elements, list):
        elements = [elements]

    elements = [i.capitalize() for i in elements]

    constrained_indices = (
        atoms.constraints[0].get_indices() if atoms.constraints else list()
    )
    silver_indices = [
        index for index, atom in enumerate(atoms) if atom.symbol in elements
    ]
    support_indices = [
        index for index, atom in enumerate(atoms) if not atom.symbol in elements
    ]

    silvers, support = atoms[silver_indices], atoms[support_indices]
    silvers.set_cell(atoms.get_cell())
    support.set_cell(atoms.get_cell())
    silvers.pbc, support.pbc = True, True

    old_to_new_support_indices = dict()
    for new_index, old_index in enumerate(support_indices):
        old_to_new_support_indices[old_index] = new_index

    new_constrained_support_indices = [
        old_to_new_support_indices[old_i]
        for old_i in constrained_indices
        if old_i in old_to_new_support_indices
    ]

    if new_constrained_support_indices:
        support.set_constraint(FixAtoms(indices=new_constrained_support_indices))

    return silvers, support


def concatenate(
    nanoparticle: Atoms, support: Atoms, adsorption_height: float = ADSORPTION_HEIGHT
) -> Atoms:
    """
    Join an NP to a Surface
    Inputs:     NP
                Bare_Surface
                Adsorption height (Optional)
    Returns:    Supported_NP
    """
    max_support_z = max(support.positions[:, 2])
    min_np_z = min(nanoparticle.positions[:, 2])

    support2 = support.copy()
    nanoparticle2 = nanoparticle.copy()

    nanoparticle2.translate(
        displacement=(0, 0, adsorption_height - min_np_z + max_support_z)
    )
    support2.extend(nanoparticle2)

    return support2


def centralize(atoms: Atoms, element: str = NANOPARTICLE_ELEMENT) -> Atoms:
    """
    Centralize the NP to make viewing easier
    Inputs:     Supported_NP
                element the NP is made of
    Returns:    Supported_NP (with NP centralized)
    """
    atoms2=atoms.copy()
    silvers, support = divider(atoms2, elements=element)

    center_x = support.cell.lengths()[0] / 2
    center_y = support.cell.lengths()[1] / 2

    silvers_com_x = silvers.get_center_of_mass()[0]
    silvers_com_y = silvers.get_center_of_mass()[1]

    x_move = center_x - silvers_com_x
    y_move = center_y - silvers_com_y
    silvers.translate(displacement=(x_move, y_move, 0))

    support.extend(silvers)
    support.pbc = atoms2.pbc
    support.set_constraint(atoms.constraints)

    support.pbc = True

    return support


def calculate_current_lateral_spacing(
    atoms: Atoms, element: str = NANOPARTICLE_ELEMENT
) -> Tuple[float, float]:
    """
    Calculate the current lateral spacing of NPs
    Inputs:     Supported_NP
                element the NP is made of
    Returns:    X- and Y-spacing of periodic NP images
    """
    silvers, support = divider(atoms, elements=element)

    min_x = min(silvers.positions[:, 0])
    min_y = min(silvers.positions[:, 1])
    max_x = max(silvers.positions[:, 0])
    max_y = max(silvers.positions[:, 1])

    x_spacing = atoms.cell.lengths()[0] - (max_x - min_x)
    y_spacing = atoms.cell.lengths()[1] - (max_y - min_y)

    return x_spacing, y_spacing


def constrain_lower_MgO(atoms: Atoms, layer_height: float = LAYER_HEIGHT) -> Atoms:
    """
    Translate system to bottom of cell, then constrain the lower MgO layer
    Inputs:     Supported_NP
    Returns:    Supported_NP (at bottom of cell, with lower MgO constrained)

    Note that all of this works even when there is no NP (i.e. just the substrate)
    """
    minz = min(atoms.positions[:, 2])
    atoms.translate(displacement=(0, 0, -minz))
    fixed = FixAtoms([atom.index for atom in atoms if atom.z < layer_height])
    atoms.set_constraint(fixed)

    return atoms


def relax_second_layer(
    atoms: Atoms, secondlayer_height: float = SECONDLAYER_HEIGHT
) -> Atoms:
    """
    Given four MgO layers, relax the second layer
    Inputs:     Supported_NP
    Returns:    Supported_NP (at bottom of cell, with lower MgO constrained)
    """
    ##first, just in case: constrain and translate to bottom of cell
    atoms = constrain_lower_MgO(atoms)
    ##remove the constraint on the 2nd layer
    relaxed = None
    atoms.set_constraint(relaxed)
    fixed = FixAtoms([atom.index for atom in atoms if atom.z < secondlayer_height])
    atoms.set_constraint(fixed)

    return atoms


def four_to_two_MgO(
    atoms: Atoms, secondlayer_height: float = SECONDLAYER_HEIGHT
) -> Atoms:
    """
    Delete the two lower MgO layers, turning it from 4 layers to 2
    Inputs:     Supported_NP with 4 MgO layers
    Returns:    Supported_NP with 2 MgO layers instead of 4

    Note that all of this works even when there is no NP (i.e. just the substrate)
    Note also, that we keep the amount of vacuum unchanged (i.e making the cell shorter)
    """
    ##first, just in case: constrain and translate to bottom of cell
    atoms = constrain_lower_MgO(atoms)
    vacuum = atoms.cell[2][2] - max(atoms.positions[:, 2])
    ##now, delete lower 2 layers
    mask = [atom.z >= secondlayer_height for atom in atoms]
    atoms = atoms[mask]
    # again, translate to bottom of cell
    atoms = constrain_lower_MgO(atoms)
    # reduce vacuum by the height lost
    max_z = max(atoms.positions[:, 2])
    atoms.cell[2][2] = vacuum + max_z

    return atoms


def reduce_vacuum(
    atoms: Atoms,
    x_target: float = TARGET,
    y_target: float = TARGET,
    z_target: float = TARGET,
    support_half_dist: float = Mg_O_HALF_LATTICE,
    mgo_tol: float = MgO_SIDEWAYS_LAYER_TOLERANCE,
    centralize_np: bool = False,
    check: bool = True,
    verbose: bool = False,
) -> Tuple[
    Atoms,
    float,
    float,
    float,
    float,
    float,
    float,
    bool,
]:
    """
    Given an atoms object of a supported nanoparticle,
    remove MgO to reduce the lateral spacing
    Make sure to keep the MgO crystal structure the same

            !!!WARNING!!!
    This should not have any adsorbates

    Requires:
        atoms:              supported nanoparticle
        x_target:           target x lateral spacing
        y_target:           target y lateral spacing
        z_target:           target vacuum
        support_half_dist:  the distance that defines one minimum unit of the support.
                            For MgO, that is the Mg-O bond length
        mgo_tol:            tolerance for classifying support into lateral 'layers'
        centralize_np:         whether to centralize the NP before deleting support units.
        check:              whether to check for overlapping atoms at the end. Strongly Recommended but expensive
        verbose:            to print out debugging info

                        !!! Warning !!!
    To use for other supports, change the mgo_tol and support_half_dist
    their defaults have been chosen for MgO

    Returns:
        agmgo:              supported nanoparticle, with spacings reduced
        x_space:            initial x lateral spacing
        y_space:            initial y lateral spacing
        init_vacuum:        initial vacuum
        x_space_final:      resulting x lateral spacing
        y_space_final:      resulting y lateral spacing
        z_target:           resulting vacuum
        overlapping:        if atoms are overlapping
    """
    mg_o_dist = support_half_dist
    mgo_lattice = 2 * mg_o_dist
    # outline limitations
    warning_message = f"""
            Pay close attention to the following:
            -------------------------------------

    1. Assuming Mg-O distance = {mg_o_dist} A
    2. To use for other supports, change mgo_tol and support_half_dist; defaults have been chosen for MgO
    3. Default MgO lateral-layer tolerance of {MgO_SIDEWAYS_LAYER_TOLERANCE} not yet been tested extensively.
    4. Assumes NP is approximately centralized. Else, will increase the chances of a Traceback
       asking you to increase the lateral spacings
    5. May not achieve the target lateral spacings. Check what it returns
    6. Assumes an orthogonal cell.
    7. Returned atom object may have atom indices changed,
    i.e. atom 4 may now be atom 8. As a result of the atoms.extend() function used near the end
    """
    if verbose: #True:
        ASCIIColors.print(
            warning_message,
            color=ASCIIColors.color_red,
            style=ASCIIColors.style_bold,
            background=ASCIIColors.color_black,
            end="\n\n",
            flush=True,
            file=stdout,
        )
    # z-target
    atoms2=atoms.copy()
    if centralize_np:
        warnings.warn("centralizing, which means NP-Support geometry changes")
        atoms2 = centralize(atoms2)
    init_vacuum, *_ = calc_vacuum(atoms2)
    atoms = set_vacuum(atoms2, set_point=z_target)
    # prepare
    ag, mgo = divider(atoms)
    ag_x = ag.positions[:, 0]
    ag_y = ag.positions[:, 1]
    mgo_x = mgo.positions[:, 0]
    mgo_y = mgo.positions[:, 1]
    mgo_x_extent = np.max(mgo_x) - np.min(mgo_x)
    mgo_y_extent = np.max(mgo_y) - np.min(mgo_y)
    x_space, y_space = calculate_current_lateral_spacing(atoms)
    if verbose:
        print(f"Initial vacuum:\t{init_vacuum:.1f} A")
        print(f"Initial x and y lateral spacings:\t{x_space:.1f} A and {y_space:.1f} A")
    cell_x, cell_y, cell_z = atoms.cell.lengths()
    cell = atoms.cell
    cell_x_padding = cell_x - mgo_x_extent
    cell_y_padding = cell_y - mgo_y_extent
    if (cell_x_padding < -0.1) or (cell_y_padding < -0.1):
        warnings.warn("what the heck is going on here?? I refuse to proceed with this. Will return None allthrough")
        return [None] * 8

    x_shave_dist, y_shave_dist = x_space - x_target, y_space - y_target
    if x_shave_dist <= 0 or y_shave_dist <= 0:
        print(
            "Requested target exceeds the current lateral spacing. Will return with only vacuum changed"
        )
        return (atoms, x_space, y_space, init_vacuum, x_space, y_space, z_target)

    x_shave = int((x_shave_dist // mgo_lattice))
    y_shave = int((y_shave_dist // mgo_lattice))
    buffer = mg_o_dist - 1.5  # don't want Ag at the very edge of the new cell
    if verbose:
        print(
            f"Will delete {x_shave} half-units from each extreme of the x axis; {y_shave} for y"
        )
        print(f"Buffer for Ag:\t{buffer} A")
    # check if any ag will fall outside
    ag_x_greater = np.any(ag_x >= cell_x - (x_shave_dist / 2) - buffer)
    ag_x_less = np.any(ag_x <= (x_shave_dist / 2) + buffer)
    ag_y_greater = np.any(ag_y >= cell_y - (y_shave_dist / 2) - buffer)
    ag_y_less = np.any(ag_y <= (y_shave_dist / 2) + buffer)
    if any([ag_x_greater, ag_x_less, ag_y_greater, ag_y_less]):
#        pass
        raise RuntimeError(
            "Reduced cell will place some Ag atoms outside the cell. Increase the x or y target"
        )

    # shave off x
    if x_shave > 0:
        mgo, mgo_x_layers = classify_into_layers(
            atoms=mgo, tol=mgo_tol, interested_system="Support", coordinate="x"
        )
        max_mgo_xlayers = np.max(mgo_x_layers)
        # shave x_shave half-units from each side
        delete_x_indices = list(range(1, x_shave + 1, 1)) + list(
            range(max_mgo_xlayers, max_mgo_xlayers - x_shave, -1)
        )
        delete_x_mask = np.where(np.isin(mgo_x_layers, delete_x_indices))[0]
        del mgo[delete_x_mask]

    if y_shave > 0:
        ##shave off y
        mgo, mgo_y_layers = classify_into_layers(
            atoms=mgo, tol=mgo_tol, interested_system="Support", coordinate="y"
        )
        max_mgo_ylayers = np.max(mgo_y_layers)
        # shave y_shave half-units from each side
        delete_y_indices = list(range(1, y_shave + 1, 1)) + list(
            range(max_mgo_ylayers, max_mgo_ylayers - y_shave, -1)
        )
        delete_y_mask = np.where(np.isin(mgo_y_layers, delete_y_indices))[0]
        del mgo[delete_y_mask]

    mgo_x = mgo.positions[:, 0]
    mgo_y = mgo.positions[:, 1]
    mgo_x_extent = np.max(mgo_x) - np.min(mgo_x)
    mgo_y_extent = np.max(mgo_y) - np.min(mgo_y)

    new_cell_y = mgo_y_extent + cell_y_padding
    new_cell_x = mgo_x_extent + cell_x_padding
    new_cell_y = [cell[1][0], new_cell_y, cell[1][2]]
    new_cell_x = [new_cell_x, cell[0][1], cell[0][2]]
    translate_x = np.min(mgo_x)
    translate_y = np.min(mgo_y)

    agmgo = mgo.copy()
    agmgo.extend(ag)
    agmgo.set_cell([new_cell_x, new_cell_y, cell[2]])
    agmgo.translate(displacement=(-translate_x, -translate_y, 0))
    x_space_final, y_space_final = calculate_current_lateral_spacing(agmgo)
    agmgo.pbc = atoms.pbc
    
    if verbose:
        print(f"Final lateral spacings:\t{x_space_final:.1f} A, {y_space_final:.1f} A")
    overlapping=False
    if True:#check:
        ##sanity check, since I don't yet fully trust this function
        if verbose:
            print("checking for overlapping atoms")
        sanity = agmgo * (2,2,1)
        duplicates = get_duplicate_atoms(sanity, delete=False)
        if len(duplicates) > 0:
            overlapping=True
#            raise ValueError("duplicate atoms found:\n{duplicates}")

        ##check MgO stoichiometry
        if verbose:
            print("checking stoichiometry of the support")
        symbols=sanity.symbols
        mg=symbols.count("Mg")
        support_o=symbols.count("O")#-len(adsorbate_indices)
        if mg != support_o:
            raise ValueError("{mg} Mg atoms; {support_o} O atoms in support. Not stoichiometric!")

    return (
        agmgo,
        x_space,
        y_space,
        init_vacuum,
        x_space_final,
        y_space_final,
        z_target,
        overlapping,
    )


def scaler(
    image: Atoms,
    element: str = NANOPARTICLE_ELEMENT,
    adsorption_height: float = ADSORPTION_HEIGHT,
    z_spacing: float = Z_SPACING,
    lateral_spacing: float = LATERAL_SPACING,
    layers: str = LAYERS,
    unit_support: Union[Atoms, None] = None,
) -> Atoms:
    """
    Return an NP supported upon MgO of a good size
    Inputs:
        Image:              NP or Supported NP
        element:            What element the NP is made of. default = 'Ag'
        adsorption_height:  Desired adsorption height (Optional)
        z_spacing:          Desired Z-spacing of periodic images (Optional)
        lateral_spacing:    Desired X- and Y-spacing of periodic NP images (Optional)
        layers:             How many MgO layers. Must be 'two' or 'four'
        unit_support:       Atoms object of the unit support.
                            Defaults to that given here (GPAW-D3(BJ) for 2 or 4 layers)

    Returns:
                            Supported_NP
    """
    if not unit_support:
        unit_support = create_unit_support(layers=layers)
    unit_cell = unit_support.cell
    unit_cell_x = unit_cell[0, 0]
    unit_cell_y = unit_cell[1, 1]
    unit_cell_z = unit_cell[2, 2]
    unit_cell_max_z = max(unit_support.positions[:, 2])

    silvers, support = divider(image, elements=element)
    silvers = Atoms(silvers)
    silvers.center(vacuum=10)

    min_x = min(silvers.positions[:, 0])
    min_y = min(silvers.positions[:, 1])
    max_x = max(silvers.positions[:, 0])
    max_y = max(silvers.positions[:, 1])
    min_z = min(silvers.positions[:, 2])

    x_diameter = max_x - min_x
    y_diameter = max_y - min_y

    required_x = x_diameter + lateral_spacing
    required_y = y_diameter + lateral_spacing

    ratio_x = math.ceil(required_x / unit_cell_x)
    ratio_y = math.ceil(required_y / unit_cell_y)

    adsorption_height = adsorption_height + unit_cell_max_z
    new_support = unit_support * (ratio_x, ratio_y, 1)
    silvers_displacement = adsorption_height - min_z
    silvers.translate(displacement=(0, 0, silvers_displacement))

    new_support.cell[2, 2] = (
        10  # deliberately set too low so that the logic within the if statement (below) will play out fine
    )
    new_support.extend(silvers)

    cell_bottom = 0
    min_z = min(new_support.positions[:, 2])
    new_cell_displacement = cell_bottom - min_z
    new_support.translate(displacement=(0, 0, new_cell_displacement))

    max_height = max(new_support.positions[:, 2])
    cell_top = new_support.cell[2, 2]
    distance = cell_top - max_height

    if distance < z_spacing:
        new_support.cell[2, 2] += lateral_spacing - cell_top + max_height

    #    new_support = centralize(new_support)
    new_support.info.update(image.info)

    return new_support


@deprecated
def check_exploded(atoms: Atoms, element: str = NANOPARTICLE_ELEMENT) -> float:
    """
    Indicate that the system might have exploded
    Inputs:     NP or Supported_NP
                element the NP is made of. default to Ag
    Returns:    Max X- or Y-spacing between any two atoms in the system

    Example usage:
    it_exploded = check_exploded(atoms) > 70 #angstrom
    """
    silvers, surface = divider(atoms, elements=element)
    min_x, max_x = min(silvers.positions[:, 0]), max(silvers.positions[:, 0])
    min_y, max_y = min(silvers.positions[:, 1]), max(silvers.positions[:, 1])

    return max(max_x - min_x, max_y - min_y)


def check_inversion(atoms: Atoms, element: str = NANOPARTICLE_ELEMENT) -> bool:
    """
    Check if the system got inverted, which I have noticed
    sometimes happens, long after the calculation has exploded

    Inputs:     NP or Supported_NP
                element the NP is made of. default to Ag
    Returns:    True if system is inverted, else False
    """
    silvers, surface = divider(atoms, elements=element)
    silvers_min_z = min(silvers.positions[:, 2])
    surface_max_z = max(surface.positions[:, 2])

    return silvers_min_z < surface_max_z


@deprecated
def check_for_uncoordinated_atoms(
    atoms: Atoms, element: str = NANOPARTICLE_ELEMENT
) -> bool:
    """
    Check to see if any atom/atom-groups has flown off the NP
    Inputs:     NP or Supported_NP
                element the NP is made of. default to Ag
    Returns:    True if so, else False
    """
    silvers, surface = divider(atoms, elements=element)
    nl = NeighborList(
        natural_cutoffs(silvers, mult=1.07), self_interaction=False, bothways=True
    )
    nl.update(silvers)
    bonds = nl.get_connectivity_matrix(sparse=False).sum(axis=0)

    return 0 in bonds


def classify_traj(
    traj: List[Atoms],
    element: str = NANOPARTICLE_ELEMENT,
    coverage: float = 0.8,
    buffer: float = 0.05,
    return_lattices: bool = False,
    show: bool = False,
    plot_name: str = "nameless",
) -> List:
    """
    Divides a list of structures into their sublists, which are:
        Bulk NP
        Supported NP
        Supported ML
        Gasphase NP
        Pristine Support (Slab)
        Bulk Support
        Unknown/Miscellaneous systems

    Also plots a pie chart of these structures' counts
    also gives you the lattice constant of the support
    (assuming it is MgO) and of the Bulk NP (somewhat)

    Inputs:
        traj: List[Atoms]       List of atoms objects to classify
        element (str)           element the NP is made of. defaults to Ag
        coverage: float         What fraction of the cell would bulk Ag fill?
                                This is required to distinguish
                                between gasphase and bulk Ag
                                Default of 0.75 seems to work well
        buffer: float           Buffer used in logic for calculating lattice constants.
                                Default of 0.05 A should be universally
                                valid for a fixed, crytalline layer
        return_lattices: bool   Whether or not to calculate the bulk Ag's
                                'pseudo' lattice constants  ('pseudo',
                                since it might not be crystalline)
                                and the support's lattice constants

    Returns:
        in the following order:
            supported_NP, supported_ML, pristine_support (slab), bulk_support, bulk_NP, gasphase_NP, miscellaneous,
            NP_pseudolattice, support_lattice

        note that miscellaneous and-or support_lattice_constant
        and-or NP_pseudolattice may be empty lists

    Example usage:
    images = read('traj.traj', ':')
    agnp_mgo, agml_mgo, pristine_mgo, bulk_mgo, bulk_ag, gas_ag, misc, ag_plattices,\
            mgo_lattices = classify_traj(images,
            return_lattices = True, show=True)
    """

    gas_ag, pristine_mgo, bulk_mgo, bulk_ag, agnp_mgo, agml_mgo, miscellaneous = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    mgo_lattices, ag_pseudolattices = [], []
    element = element.capitalize()

    if return_lattices:  ##get the lattice constant
        warnings.warn(
            f"""You have requested the lattice constants of the support
        The calculations assume that:
            1. MgO is the support
            2. The lowest layer is constrained

        You have also asked for {element}'s pseudo-lattice constant
        Note that since the {element} might not be perfectly crystalline,
        the values returned are only a sort of 'average' lattice constant
        Also note that the formula used ONLY applies to FCC metals
            """,
            category=UserWarning,
        )

    warnings.warn(
        f"""An extent/coverage of {coverage} may result in
    imperfect discrimination between gasphase and bulk {element}.
    Check results carefully!""",
        category=UserWarning,
    )

    warnings.warn(
        """The method for distinguishing supported ML
    from supported NP is far from idiot-proof.\n
    This is the same method used to distinguish between slab and bulk support""",
        category=UserWarning,
    )

    for atoms in tqdm(traj, total=len(traj), desc="discriminating"):

        ag = Atoms([atom for atom in atoms if atom.symbol == element])
        mgo = Atoms([atom for atom in atoms if atom.symbol != element])

        if len(mgo) == 0:  # gas or bulk Ag
            x_positions, y_positions, z_positions = atoms.positions.T
            x_extent = max(x_positions) - min(x_positions)
            y_extent = max(y_positions) - min(y_positions)
            z_extent = max(z_positions) - min(z_positions)

            x_cell, y_cell, z_cell = atoms.cell.lengths()

            if (
                (x_extent >= coverage * x_cell)
                and (y_extent >= coverage * y_cell)
                and (z_extent >= coverage * z_cell)
            ):  # bulk Ag virtually fills the cell

                atoms.info["Class"] = "Bulk_NP"
                bulk_ag.append(atoms)

                if return_lattices:  ##get the pseudolattice constants
                    volume = atoms.get_volume()
                    ag_pseudolattices.append((4 * volume / len(atoms)) ** (1 / 3))

            else:
                atoms.info["Class"] = "Gas_NP"
                gas_ag.append(atoms)

        elif len(ag) == 0:  # pristine or bulk mgo
            cell_x, cell_y, cell_z = atoms.cell.lengths()
            mgo_x, mgo_y, mgo_z = (
                mgo.positions[:, 0],
                mgo.positions[:, 1],
                mgo.positions[:, 2],
            )
            mgo_x_extent = np.max(mgo_x) - np.min(mgo_x)
            mgo_y_extent = np.max(mgo_y) - np.min(mgo_y)
            mgo_z_extent = np.max(mgo_z) - np.min(mgo_z)

            if (
                mgo_x_extent >= (0.93 * cell_x)
                and mgo_y_extent >= (0.93 * cell_y)
                and mgo_z_extent >= (0.93 * cell_z)
            ):
                atoms.info["Class"] = "Bulk Support"
                bulk_mgo.append(atoms)
            else:
                atoms.info["Class"] = "Pristine Support"
                pristine_mgo.append(atoms)

            if return_lattices:  ##get the lattice constants
                new_atoms = deepcopy(atoms)
                new_atoms = constrain_lower_MgO(new_atoms)
                lower_layer = [atom.z < LAYER_HEIGHT for atom in new_atoms]
                new_atoms = new_atoms[lower_layer]

                x_slice_anchor = min(new_atoms.positions[:, 0])
                x_slice = [
                    (x_slice_anchor - buffer) < atom.x < (x_slice_anchor + buffer)
                    for atom in new_atoms
                ]

                x_slice_atoms = new_atoms[x_slice]
                y_s = new_atoms.positions[:, 1]

                y_length = max(y_s) - min(y_s)
                number = len(x_slice_atoms) / 2
                mg_o = y_length / (number - 0.5)
                mgo_lattices.append(mg_o)

        elif len(ag) != 0:  # Ag/MgO
            # Let's try to (roughly) discriminate between Ag NPs and Ag MLs
            # if it's an ML, its X and Y extents should be about as large as the cell's
            cell_x, cell_y = atoms.cell.lengths()[:2]
            ag_x, ag_y = ag.positions[:, 0], ag.positions[:, 1]
            ag_x_extent = np.max(ag_x) - np.min(ag_x)
            ag_y_extent = np.max(ag_y) - np.min(ag_y)

            if ag_x_extent >= (coverage * cell_x) and ag_y_extent >= (
                coverage * cell_y
            ):
                atoms.info["Class"] = "Supported ML"
                agml_mgo.append(atoms)
            else:
                atoms.info["Class"] = "Supported NP"
                agnp_mgo.append(atoms)

        else:  # unknown
            atoms.info["Class"] = "Unknown"
            miscellaneous.append(atoms)

    if miscellaneous:
        warnings.warn(
            f"{len(miscellaneous)} images have been missed!", category=UserWarning
        )

    systems = {
        "Ag NP/MgO": agnp_mgo,
        "Ag ML/MgO": agml_mgo,
        "MgO(100)": pristine_mgo,
        "MgO": bulk_mgo,
        "Ag": bulk_ag,
        "Ag NP": gas_ag,
        "Unknown": miscellaneous,
    }
    piechart = {name: len(system_) for name, system_ in systems.items()}
    plt.pie(
        list(piechart.values()),
        labels=list(piechart.keys()),
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.axis("equal")
    plt.title("Types of systems in supplied trajectory")
    plt.savefig(f"{plot_name}.png", dpi=120)
    if show:
        plt.show()


    return (
        agnp_mgo,
        agml_mgo,
        pristine_mgo,
        bulk_mgo,
        bulk_ag,
        gas_ag,
        miscellaneous,
        ag_pseudolattices,
        mgo_lattices,
    )


@deprecated
def remove_uncoordinated_atoms(
    atoms: Atoms, element: str = NANOPARTICLE_ELEMENT, coord_cutoff: int = 4
) -> Atoms:
    """
    Remove uncoordinated atoms or atom-groups
    Inputs:     NP or Supported_NP
                element the NP is made of. defaults to Ag
                Min. no. of bonds for an atom/atom-group to be considered uncoordinated
                default = 4
    Returns:    NP or Supported_NP with uncoordinated atom/atom-group removed
    """
    silvers, surface = divider(atoms, elements=element)
    nl = NeighborList(
        natural_cutoffs(silvers, mult=1.07), self_interaction=False, bothways=True
    )
    nl.update(silvers)
    bonds = nl.get_connectivity_matrix(sparse=False).sum(axis=0)
    sakujo = []
    for index in range(len(silvers)):
        if bonds[index] < coord_cutoff:
            sakujo.append(index)
    del silvers[sakujo]

    if len(sakujo) > 0:
        warnings.warn(
            f"""{len(sakujo)} atoms deleted from this NP.
        I hope you know what you are doing!""",
            category=UserWarning,
        )

    if len(silvers) == 0:
        print(f"No {element} atoms left! Something is very wrong")
        print(f"Error from remove_uncoordinated_atoms() in fit_support.py")
        exit(1)

    surface.extend(silvers)
    surface.set_cell(atoms.get_cell())
    surface.set_constraint(atoms.constraints)
    surface.pbc = True

    return surface


def tem_rotate(atoms2, viewpoint: Literal["x", "y", "z", "x-y"] = "z") -> Atoms:
    """
    For generating TEM images, the default is looking from the z

    viewpoint:
        z           default (z pointing out of screen), leave atoms object unchanged
        x           rotate so that y points out of screen
        y           rotate so that x points out of screen
        x-y          rotate inbetween x and y
    """
    atoms = atoms2.copy()
    if viewpoint == "z":
        return atoms
    elif viewpoint == "x":
        atoms.rotate("x", -90, rotate_cell=True)
    elif viewpoint == "y":
        warnings.warn(
            f"""Why on earth would you want to 
        have the {viewpoint} viewpoint??"""
        )
        atoms.rotate("y", -90, rotate_cell=True)
    elif viewpoint == "x-y":
        atoms.rotate("x", -90, rotate_cell=True)
        atoms.rotate("y", -45, rotate_cell=True)

    return atoms


if __name__ == "__main__":
    if len(argv) < 7:
        print(
            f"run as {argv[0]} gas-phase-np.traj <adsorption_height> <z_spacing> <lateral_spacing> <layers: 'two' or 'four' <element NP is made of>"
        )
        exit(1)

    auto_backup()

    data = iread(argv[1], ":")
    adsorption_height, z_spacing, lateral_spacing, num_layers, element = (
        float(argv[2]),
        float(argv[3]),
        float(argv[4]),
        argv[5],
        argv[6],
    )
    data = [
        scaler(
            image,
            elements=element,
            adsorption_height=adsorption_height,
            z_spacing=z_spacing,
            lateral_spacing=lateral_spacing,
            layers=num_layers,
            unit_support=None,
        )
        for image in tqdm(data)
    ]

    write("supported-NPs.traj", data)
    print("Processing completed.\nFile saved in 'supported-NPs.traj'")
