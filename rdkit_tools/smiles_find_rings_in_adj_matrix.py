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


def smiles_to_mol_bonds(smi: str) -> Tuple[List[List[int]], str]:
    """
    Converts a SMILES string to bond information and RDKit molecule object.

    Args:
    smi (str): SMILES string of the molecule.

    Returns:
    Tuple[List[List[int]], str, Chem.Mol, List[List[str]], int]: 
    A tuple containing the bond information, SMILES string, RDKit molecule object, bond symbols, and number of atoms.
    """
    rdkit_mol = Chem.MolFromSmiles(smi)
    if rdkit_mol is None:
        raise ValueError("Invalid SMILES string generated.")

    # Add explicit hydrogens to the RDKit molecule
    rdkit_mol = Chem.AddHs(rdkit_mol)

    bonds = []
    bonds_symbol = []
    for bond in rdkit_mol.GetBonds():
        idx_0 = bond.GetBeginAtomIdx()
        idx_1 = bond.GetEndAtomIdx()
        atom_0 = rdkit_mol.GetAtomWithIdx(idx_0).GetSymbol()
        atom_1 = rdkit_mol.GetAtomWithIdx(idx_1).GetSymbol()
        bond_type = bond.GetBondType()
        bonds.append([idx_0, idx_1, int(bond_type)])
        bonds_symbol.append([atom_0, atom_1, int(bond_type)])

    return bonds, smi, rdkit_mol, bonds_symbol, rdkit_mol.GetNumAtoms()

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python smiles_find_rings_in_adj_matrix.py <smiles_string>")
        sys.exit(1)
    
    smiles_input = sys.argv[1]
    bonds, smiles, rdkit_mol, bonds_symbol, num_atoms = smiles_to_mol_bonds(smiles_input)
    print(smiles)
    adjacency_matrix = construct_adjacency_matrix(bonds, num_atoms)
    assert len(adjacency_matrix) == num_atoms
    print(adjacency_matrix.shape)
    print(adjacency_matrix)
    rings_info = find_ring_atoms_in_adj_matrix(rdkit_mol, adjacency_matrix)
    for ring in rings_info:
        print(ring)
