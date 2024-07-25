import sys
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_3d_mol(smiles: str) -> Chem.Mol:
    """
    Generates a molecule object with 3D coordinates from a SMILES string.

    Args:
    smiles (str): SMILES string of the molecule.

    Returns:
    Chem.Mol: RDKit molecule object with 3D coordinates.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return mol

def save_to_xyz(mol: Chem.Mol, file_path: str) -> None:
    """
    Writes molecule data to an XYZ file.

    Args:
    mol (Chem.Mol): RDKit molecule object.
    file_path (str): Path to save the XYZ file.
    """
    num_atoms = mol.GetNumAtoms()
    with open(file_path, 'w') as file:
        file.write(f"{num_atoms}\n")
        file.write("XYZ coordinates from SMILES\n")
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            file.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

def save_to_sdf(mol: Chem.Mol, file_path: str) -> None:
    """
    Writes molecule data to an SDF file.

    Args:
    mol (Chem.Mol): RDKit molecule object.
    file_path (str): Path to save the SDF file.
    """
    writer = Chem.SDWriter(file_path)
    writer.write(mol)
    writer.close()

# Usage example
if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python smiles_to_xyz_sdf.py <smiles_string> <output_path> [xyz|sdf]")
        sys.exit(1)

    smiles_input = sys.argv[1]
    output_path = sys.argv[2]
    output_format = sys.argv[3] if len(sys.argv) == 4 else 'xyz'

    mol = smiles_to_3d_mol(smiles_input)

    if output_format.lower() == 'sdf':
        save_to_sdf(mol, output_path)
        print(f"SDF file generated at {output_path}")
    else:
        save_to_xyz(mol, output_path)
        print(f"XYZ file generated at {output_path}")


# python script.py "YourSmilesString" "output.xyz"  # For XYZ output
# python script.py "YourSmilesString" "output.sdf" sdf  # For SDF output