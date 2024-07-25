import sys
import numpy as np
from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import Draw
from openbabel import openbabel
from xyz_to_smiles import read_xyz_file, geom_to_smi_and_bonds
from xyz_adj_matrix import construct_adjacency_matrix
import matplotlib.pyplot as plt
from xyz_find_rings_in_adj_matrix import *

def is_conjugated_ring(ring):
    conjugated_elements_3 = {'C', 'B', 'N', 'Si'}
    conjugated_elements_2 = {'O', 'S'}
    
    for atom_index in ring['atom_indices']:
        atom_symbol = ring['atom_symbols'][ring['atom_indices'].index(atom_index)]
        adjacency_symbols = ring['adjacency_symbols'][atom_index]
        
        if atom_symbol in conjugated_elements_3 and len(adjacency_symbols) == 3:
            continue
        elif atom_symbol in conjugated_elements_2 and len(adjacency_symbols) == 2:
            continue
        else:
            return False
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python xyz_find_conjugated_non_conjugated.py <path_to_xyz_file>")
        sys.exit(1)
    
    xyz_file_path = sys.argv[1]
    atoms, coords = read_xyz_file(xyz_file_path)
    bonds, smiles, rdkit_mol, bonds_symbol = geom_to_smi_and_bonds(atoms, coords)
    num_atoms = len(atoms)
    adjacency_matrix = construct_adjacency_matrix(bonds, num_atoms)
    assert len(adjacency_matrix) == num_atoms
    print(adjacency_matrix.shape)
    print(adjacency_matrix)
    rings_info = find_ring_atoms_in_adj_matrix(rdkit_mol, adjacency_matrix)
    for ring in rings_info:
        print(ring)
    conjugated_rings = []
    for ring in rings_info:
        if is_conjugated_ring(ring):
            conjugated_rings.append(ring)
            print("Conjugated ring found:")
        else:
            print("Non-conjugated ring found:")
        print(ring)
    
    print(f"Total conjugated rings: {len(conjugated_rings)}")

# python xyz_find_conjugated_non_conjugated.py example/compas.xyz