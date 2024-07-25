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


def find_ring_atoms_in_adj_matrix(mol: Chem.Mol, adjacency_matrix: np.ndarray) -> np.ndarray:
    """
    Identify the rows and columns in the adjacency matrix corresponding to atoms that are part of rings.

    :param mol: An RDKit molecule object.
    :param adjacency_matrix: The adjacency matrix of the molecule.
    :return: A list of dictionaries, each containing information about a ring in the molecule.
    """
    if mol is None:
        raise ValueError("Invalid mol provided.")
    # Get ring information
    ring_info = mol.GetRingInfo().AtomRings()
    
    rings_info = []
    
    # Iterate over each ring
    for ring in ring_info:
        ring_dict = {'atom_indices': [], 'atom_symbols': [], 'adjacency_rows': [], 'adjacency_symbols': {}}
        for i in ring:
            ring_dict['atom_indices'].append(i)
            ring_dict['atom_symbols'].append(mol.GetAtomWithIdx(i).GetSymbol())
            ring_dict['adjacency_rows'].append(adjacency_matrix[i])
            symbol_row = [mol.GetAtomWithIdx(int(j)).GetSymbol() for j in np.nonzero(adjacency_matrix[i])[0]]
            ring_dict['adjacency_symbols'][i] = symbol_row
        
        rings_info.append(ring_dict)
    
    return rings_info

def visualize_molecule_with_atom_numbers(mol: Chem.Mol, file_name: str):
    """
    Visualize the molecule and display atom numbers, then save the image to a file.

    :param mol: An RDKit molecule object.
    :param file_name: The name of the file to save the image.
    """
    if mol is None:
        raise ValueError("Invalid mol provided.")
    
    # Add atom indices as atom labels
    for atom in mol.GetAtoms():
        atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))
    
    # Draw the molecule
    img = Draw.MolToImage(mol, size=(600, 600), kekulize=True)
    
    # Save the image
    img.save(file_name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python xyz_find_rings_in_adj_matrix.py <path_to_xyz_file>")
        sys.exit(1)
    
    xyz_file_path = sys.argv[1]
    atoms, coords = read_xyz_file(xyz_file_path)
    bonds, smiles, rdkit_mol, bonds_symbol = geom_to_smi_and_bonds(atoms, coords)
    print(smiles)
    num_atoms = len(atoms)
    adjacency_matrix = construct_adjacency_matrix(bonds, num_atoms)
    assert len(adjacency_matrix) == num_atoms
    print(adjacency_matrix.shape)
    print(adjacency_matrix)
    rings_info = find_ring_atoms_in_adj_matrix(rdkit_mol, adjacency_matrix)
    for ring in rings_info:
        print(ring)

    visualize_molecule_with_atom_numbers(rdkit_mol, "rdkit_tools/example/molecule_with_atom_numbers.png")