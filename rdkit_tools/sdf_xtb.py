import sys
import subprocess
from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

def read_sdf_file(filename: str) -> Tuple[List[str], List[List[float]]]:
    """
    Read an SDF file and extract atoms and coordinates.

    Parameters:
    filename (str): Path to the SDF file.

    Returns:
    Tuple[List[str], List[List[float]]]: A tuple containing a list of atomic symbols and a list of coordinates.
    """
    atoms = []
    coords = []
    supplier = Chem.SDMolSupplier(filename)
    for mol in supplier:
        if mol is None:
            continue
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            atoms.append(atom.GetSymbol())
            coords.append([pos.x, pos.y, pos.z])
        break  # Only read the first molecule
    return atoms, coords

def write_xyz_file(filename: str, atoms: List[str], coords: List[List[float]]) -> None:
    """
    Write atoms and coordinates to an XYZ file.

    Parameters:
    filename (str): Path to the XYZ file.
    atoms (List[str]): List of atomic symbols.
    coords (List[List[float]]): List of coordinates.
    """
    with open(filename, 'w') as file:
        file.write(f"{len(atoms)}\n\n")
        for atom, coord in zip(atoms, coords):
            file.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

def read_xyz_file(filename: str) -> Tuple[List[str], List[List[float]]]:
    """
    Read an XYZ file and extract atoms and coordinates.

    Parameters:
    filename (str): Path to the XYZ file.

    Returns:
    Tuple[List[str], List[List[float]]]: A tuple containing a list of atomic symbols and a list of coordinates.
    """
    atoms = []
    coords = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[2:]:  # Skip the first two lines
            parts = line.split()
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return atoms, coords

def run_xtb_optimization(input_xyz: str, output_xyz: str) -> None:
    """
    Run XTB optimization on an XYZ file.

    Parameters:
    input_xyz (str): Path to the input XYZ file.
    output_xyz (str): Path to the output XYZ file.
    """
    subprocess.run(['xtb', input_xyz, '--opt', '--xyz', output_xyz])

def sdf_to_xyz(sdf_filename: str, xyz_filename: str) -> None:
    """
    Convert an SDF file to an XYZ file.

    Parameters:
    sdf_filename (str): Path to the SDF file.
    xyz_filename (str): Path to the XYZ file.
    """
    atoms, coords = read_sdf_file(sdf_filename)
    write_xyz_file(xyz_filename, atoms, coords)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python optimize_molecule.py <path_to_xyz_or_sdf_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    if file_path.endswith('.xyz'):
        input_xyz = file_path
    elif file_path.endswith('.sdf'):
        input_xyz = 'input.xyz'
        sdf_to_xyz(file_path, input_xyz)
    else:
        print("Unsupported file format. Please provide an XYZ or SDF file.")
        sys.exit(1)

    output_xyz = 'rdkit_tools/example/optimized.xyz'

    # Run XTB optimization
    run_xtb_optimization(input_xyz, output_xyz)

    # Read the optimized XYZ file
    atoms, coords = read_xyz_file(output_xyz)

    # Process the optimized molecule
    # Further processing can be added here, such as converting to SMILES, adding hydrogens, etc.

    print(f"Optimized XYZ file saved as {output_xyz}")