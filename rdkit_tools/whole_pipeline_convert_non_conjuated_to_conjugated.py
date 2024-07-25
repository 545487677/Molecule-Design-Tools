import sys
import numpy as np
import random
from typing import List, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
from rdkit.Chem import Draw
from openbabel import openbabel

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

def visualize_molecule(mol: Chem.Mol, file_name: str):
    """
    Visualize the molecule and display atom numbers, then save the image to a file.

    :param mol: An RDKit molecule object.
    :param file_name: The name of the file to save the image.
    """
    if mol is None:
        raise ValueError("Invalid mol provided.")
    
    # Add atom indices as atom labels
    # for atom in mol.GetAtoms():
    #     atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))
    
    # Draw the molecule
    img = Draw.MolToImage(mol, size=(600, 600), kekulize=True)
    
    # Save the image
    img.save(file_name)

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

def read_xyz_file(file_path: str) -> Tuple[List[str], List[List[float]]]:
    """
    Reads an XYZ file and extracts atomic symbols and their coordinates.

    Args:
    file_path (str): The path to the XYZ file.

    Returns:
    Tuple[List[str], List[List[float]]]: A tuple containing a list of atomic symbols
                                         and a list of their corresponding coordinates.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    atoms = []
    coords = []

    for line in lines[2:]:  # Skip the header lines
        parts = line.split()
        if len(parts) == 4:
            atoms.append(parts[0])
            coords.append([float(part) for part in parts[1:4]])

    return atoms, coords

def geom_to_smi_and_bonds(atoms: List[str], coords: List[List[float]]) -> Tuple[List[List[int]], str]:
    """
    Converts a list of atomic symbols and their coordinates to SMILES format and extracts bond orders.

    Args:
    atoms (List[str]): List of atomic symbols.
    coords (List[List[float]]): List of coordinates for each atom.

    Returns:
    Tuple[List[List[int]], str]: A tuple containing the bond order information and the corresponding SMILES string.
    """
    mol = openbabel.OBMol()
    for atom, coord in zip(atoms, coords):
        ob_atom = mol.NewAtom()
        ob_atom.SetAtomicNum(openbabel.GetAtomicNum(atom))
        ob_atom.SetVector(float(coord[0]), float(coord[1]), float(coord[2]))

    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    mol.AddHydrogens()
    ob_conversion = openbabel.OBConversion()
    ob_conversion.SetOutFormat('smi')

    # obtain the bonds infor 
    bonds = []; bonds_symbol = []
    for bond in openbabel.OBMolBondIter(mol):
        bond_order = bond.GetBondOrder()
        atom_0 = bond.GetBeginAtom().GetIdx() - 1
        atom_1 = bond.GetEndAtom().GetIdx() - 1
        bonds.append([atom_0, atom_1, bond_order])
        bonds_symbol.append([atoms[atom_0], atoms[atom_1], bond_order])
    smi = ob_conversion.WriteString(mol).strip()
    rdkit_mol = Chem.RWMol()
    for ii in range(len(atoms)):
        rdkit_mol.AddAtom(Chem.Atom(atoms[ii]))
    for bond in bonds:
        if bond[-1] == 1:
            rdkit_mol.AddBond(bond[0], bond[1], Chem.BondType.SINGLE)
        elif bond[-1] == 2:
            rdkit_mol.AddBond(bond[0], bond[1], Chem.BondType.DOUBLE)
        elif bond[-1] == 3:
            rdkit_mol.AddBond(bond[0], bond[1], Chem.BondType.TRIPLE)
        else:
            rdkit_mol.AddBond(bond[0], bond[1], Chem.BondType.AROMATIC)

    AllChem.Compute2DCoords(rdkit_mol)
    Chem.SanitizeMol(rdkit_mol)
    return bonds, smi, rdkit_mol, bonds_symbol

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
        print("Usage: python whole_pipeline_convert_non_conjuated_to_conjugated.py <path_to_xyz_file>")
        sys.exit(1)
    
    xyz_file_path = sys.argv[1]
    atoms, coords = read_xyz_file(xyz_file_path)
    bonds, smiles, rdkit_mol, bonds_symbol = geom_to_smi_and_bonds(atoms, coords)
    # visualize_molecule(rdkit_mol, "rdkit_tools/example/origin_mol.png")

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