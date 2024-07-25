import sys
import numpy as np
import random
from typing import List, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
from rdkit.Chem import Draw
from openbabel import openbabel
from xyz_to_smiles import read_xyz_file, geom_to_smi_and_bonds
from xyz_adj_matrix import construct_adjacency_matrix
from xyz_find_rings_in_adj_matrix import find_ring_atoms_in_adj_matrix, visualize_molecule_with_atom_numbers
from xyz_find_conjugated_non_conjugated import is_conjugated_ring

def xyz2smi(atoms: list[str], coords: list[list[float]]) -> str:
    """
    Convert atoms and coordinates to a SMILES string.

    Parameters:
    atoms (list[str]): List of atomic symbols.
    coords (list[list[float]]): List of 3D coordinates for each atom.

    Returns:
    str: The SMILES string representation of the molecule.
    """
    # Create a new molecule object
    mol = openbabel.OBMol()
    for j in range(len(coords)):
        atom = mol.NewAtom()
        atom.SetAtomicNum(openbabel.GetAtomicNum(atoms[j]))
        x, y, z = map(float, coords[j])
        atom.SetVector(x, y, z)
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    mol.AddHydrogens()
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat('smi')
    smi = obConversion.WriteString(mol)

    return smi.split('\t\n')[0]


def identify_hydrogens_to_remove(rings_info: List[Dict]) -> List[int]:
    """
    Identify hydrogens to remove from non-conjugated rings.

    :param rings_info: List of dictionaries containing ring information.
    :return: List of hydrogen atom indices to remove.
    """
    hydrogens_to_remove = []

    for ring in rings_info:
        if not is_conjugated_ring(ring):
            for idx, symbol in zip(ring['atom_indices'], ring['atom_symbols']):
                connected_indices = np.nonzero(ring['adjacency_rows'][ring['atom_indices'].index(idx)])[0]
                connected_symbols = ring['adjacency_symbols'][idx]

                if symbol in {'C', 'B', 'N', 'Si'} and len(connected_symbols) == 4:
                    hydrogen_indices = [connected_indices[i] for i, sym in enumerate(connected_symbols) if sym == 'H']
                    if hydrogen_indices:
                        hydrogen_idx = random.choice(hydrogen_indices)
                        hydrogens_to_remove.append(hydrogen_idx)

                elif symbol in {'O', 'S'} and len(connected_symbols) == 3:
                    hydrogen_indices = [connected_indices[i] for i, sym in enumerate(connected_symbols) if sym == 'H']
                    if hydrogen_indices:
                        hydrogen_idx = random.choice(hydrogen_indices)
                        hydrogens_to_remove.append(hydrogen_idx)

    return hydrogens_to_remove

def remove_hydrogens_from_atoms_coords(atoms: List[str], coords: List[Tuple[float, float, float]], hydrogens_to_remove: List[int]) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    """
    Remove specified hydrogens from the atoms and coords lists.

    :param atoms: List of atom symbols.
    :param coords: List of atom coordinates.
    :param hydrogens_to_remove: List of hydrogen atom indices to remove.
    :return: Tuple of modified atoms and coords lists.
    """
    atoms = [atom for i, atom in enumerate(atoms) if i not in hydrogens_to_remove]
    coords = [coord for i, coord in enumerate(coords) if i not in hydrogens_to_remove]
    assert len(atoms) == len(coords)
    return atoms, np.array(coords)

def save_xyz_file(atoms: List[str], coords: List[Tuple[float, float, float]], file_name: str):
    """
    Save atoms and coordinates to an XYZ file.

    :param atoms: List of atom symbols.
    :param coords: List of atom coordinates.
    :param file_name: The name of the file to save the XYZ data.
    """
    with open(file_name, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write("Atoms and coordinates\n")
        for atom, coord in zip(atoms, coords):
            f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

def visualize_molecules(mol: Chem.Mol, file_name: str):
    """
    Visualize the molecule and display atom numbers, then save the image to a file.

    :param mol: An RDKit molecule object.
    :param file_name: The name of the file to save the image.
    """
    if mol is None:
        raise ValueError("Invalid mol provided.")
    
    # Draw the molecule
    img = Draw.MolToImage(mol, size=(600, 600), kekulize=True)
    
    # Save the image
    img.save(file_name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python xyz_convert_non_conjugated_to_conjugated.py <path_to_xyz_file>")
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
    non_conjugated_rings = []
    for ring in rings_info:
        if is_conjugated_ring(ring):
            conjugated_rings.append(ring)
            print("Conjugated ring found:")
        else:
            non_conjugated_rings.append(ring)
            print("Non-conjugated ring found:")
        print(ring)
    
    # Identify hydrogens to remove
    hydrogens_to_remove = identify_hydrogens_to_remove(non_conjugated_rings)
    print(f"Hydrogens to remove: {hydrogens_to_remove}")

    atoms, coords = remove_hydrogens_from_atoms_coords(atoms, coords, hydrogens_to_remove)

    smi = xyz2smi(atoms, coords)
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    sdf_filename = 'rdkit_tools/example/conjugated_molecule.sdf'
    writer = Chem.SDWriter(sdf_filename)
    writer.write(mol)
    writer.close()
    print(f"分子结构已保存为 {sdf_filename}")