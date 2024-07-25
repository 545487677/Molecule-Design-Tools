from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG

def smiles_to_svg(smiles: str, output_path: str, width: int = 300, height: int = 300) -> None:
    """
    Converts a SMILES string to an SVG file.

    Args:
        smiles (str): The SMILES string representing the molecule.
        output_path (str): The path to save the SVG output.
        width (int, optional): The width of the image. Defaults to 300.
        height (int, optional): The height of the image. Defaults to 300.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    # Generate SVG
    drawer = MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg_content = drawer.GetDrawingText()

    # Save SVG to a file
    with open(output_path, 'w') as file:
        file.write(svg_content)

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python smiles_visualize.py <smiles_string> <output_svg_path>")
        sys.exit(1)

    smiles_input = sys.argv[1]
    output_svg_path = sys.argv[2]

    try:
        smiles_to_svg(smiles_input, output_svg_path)
        print(f"SVG file has been saved to {output_svg_path}")
    except Exception as e:
        print("An error occurred:", str(e))