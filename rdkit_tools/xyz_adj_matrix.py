import sys
import numpy as np
from typing import List, Tuple
from rdkit import Chem
from openbabel import openbabel
from xyz_to_smiles import read_xyz_file, geom_to_smi_and_bonds

def construct_adjacency_matrix(bonds: List[List[int]], num_atoms: int) -> np.ndarray:
    """
    Constructs the adjacency matrix for a molecule from bond information.
    Args:
    bonds: List[List[int]] - A list of lists where each sublist represents a bond as [atom1, atom2, bond_type].
    num_atoms: int - The total number of atoms in the molecule.
    
    Returns:
    np.ndarray - The adjacency matrix of the molecule as a NumPy array with dtype float32.
    """
    adjacency_matrix = [[0] * num_atoms for _ in range(num_atoms)]
    for bond in bonds:
        atom1, atom2, bond_type = bond
        adjacency_matrix[atom1][atom2] = bond_type
        adjacency_matrix[atom2][atom1] = bond_type  # Since the matrix is symmetric

    return np.array(adjacency_matrix, dtype=np.float32)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python xyz_adj_matrix.py <path_to_xyz_file>")
        sys.exit(1)
    
    xyz_file_path = sys.argv[1]
    atoms, coords = read_xyz_file(xyz_file_path)
    bonds, smiles, rdkit_mol, bonds_symbol = geom_to_smi_and_bonds(atoms, coords)
    num_atoms = len(atoms)
    adjacency_matrix = construct_adjacency_matrix(bonds, num_atoms)
    assert len(adjacency_matrix) == num_atoms
    print(adjacency_matrix.shape)
    print(adjacency_matrix)


# python xyz_adj_matrix.py rdkit_tools/example/compas.xyz