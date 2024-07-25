import sys
from typing import List, Tuple
from rdkit import Chem
from openbabel import openbabel

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
    smi = ob_conversion.WriteString(mol).strip()
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

    return bonds, smi, rdkit_mol, bonds_symbol

# Usage example
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python xyz_to_smiles.py <path_to_xyz_file>")
        sys.exit(1)
    
    xyz_file_path = sys.argv[1]
    atoms, coords = read_xyz_file(xyz_file_path)
    bonds, smiles, rdkit_mol, bonds_symbol = geom_to_smi_and_bonds(atoms, coords)
    print("SMILES:", smiles)
    print("Atoms:", atoms)
    print("Bonds:", bonds)