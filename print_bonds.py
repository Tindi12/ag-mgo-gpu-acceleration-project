#!/usr/bin/env python
from ase.neighborlist import NeighborList, natural_cutoffs
from argparse import ArgumentParser
import numpy as np
from ase.geometry.analysis import Analysis
from itertools import combinations
from ase.io import read
from typing import Tuple, List, Union
from ase import Atoms


"""We print all bond lengths of the system"""
"""TODO: Print bond angles also"""
SCALING_FACTOR = 1.0

def setup_neighborlist(atoms: Atoms, scaling_factor: float = SCALING_FACTOR,
        radial_cutoff: float = None):
    """
    Setup neighborlist object,
    given the Atoms object and the scaling factor for covalent radii

    if radial_cutoff is given, then we will consider only neighbors within
    the lower of that cutoff and the covalent/ionic bond length
    """
    if not radial_cutoff:
        cutoffs = natural_cutoffs(atoms, mult = scaling_factor)
    else:
        cutoffs = np.minimum(radial_cutoff / 2, np.array(natural_cutoffs(atoms)))

    neighbors = NeighborList(cutoffs = cutoffs, self_interaction = False, bothways = True)
    neighbors.update(atoms)
    return neighbors


def setup_analyzer(atoms: Atoms, neighborlist: NeighborList = None) -> Analysis:
    """
    Setup the analyzer object, given an optional NeighborList object
    If not given a NeighborList object, use defaults
    """
    analyzer = Analysis(atoms, nl = neighborlist)
    return analyzer


def get_bond_lengths(
        bond_type: Tuple[str, str], analyzer: Analysis
        ) -> Union[None, Tuple[str, str, List[float]]]:
    """
    Does:   Finds if any of such a bond type exists in this system
            If True, then returns the bond length(s)

    Parameters:
    bond_type (Tuple[str, str]):    The atom-types of the bond
    analyzer (Analysis):            ase.geometry.analysis.Analysis object
    """
    atom_1, atom_2 = bond_type

    try:
        bonds = analyzer.get_bonds(atom_1, atom_2, unique = True)
        return (atom_1, atom_2, analyzer.get_values(bonds))
    except IndexError:
        return None


def unscramble(all_bonds_and_types: List[List]) -> None:
    """
    Print results in a neat form
    """
    for type_and_bonds in all_bonds_and_types:
        atom1, atom2 = type_and_bonds[0], type_and_bonds[1]
        print(f"{atom1}-{atom2}")

        for bond_length in type_and_bonds[2][0][:]:
            print(f"{bond_length:.3f} Å")

        print()



if __name__ == "__main__":

    parser = ArgumentParser(description = "This script prints all bond lengths of the system")
    parser.add_argument("--input", "-i", type = str, required = True, help = """Input file containing Atoms object. Required.
            See https://wiki.fysik.dtu.dk/ase/ase/io/io.html for supported file formats""")
    parser.add_argument(
            "--scaling_factor", "-s", type = float, default = SCALING_FACTOR,
            help = f"Factor with which to scale ASE's default covalent radii. Default: {SCALING_FACTOR}"
            )

    args = parser.parse_args()

    try:
        atoms = read(args.input)
    except IOError as IE:
        print(f"Cannot read file:\n{IE}")

    atom_types = np.unique(atoms.symbols)
    possible_bond_types = list(combinations(atom_types, 2))

    neighbors = setup_neighborlist(atoms, args.scaling_factor)
    analyzer = setup_analyzer(atoms, neighbors)

    all_bonds_and_types = []
    for types in possible_bond_types:
        result = get_bond_lengths(types,analyzer)
        if result:
            all_bonds_and_types.append(result)

    #fancier method for doing the same, says ChatGPT:
        #all_bonds_and_types = [result for types in possible_bond_types if (result := get_bond_lengths(types, analyzer))]

    print("Here are your bond lengths:\n")
    unscramble(all_bonds_and_types)


