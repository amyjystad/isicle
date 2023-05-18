import argparse
import subprocess
from isicle.utils import read_string, write_string
import sys
from isicle import __version__


def inchi2smi(inchi):
    """Converts InChI string to canonical SMILES string."""

    try:
        res = subprocess.check_output(
            'echo "%s" | obabel -iinchi -ocan' % inchi,
            stderr=subprocess.STDOUT,
            shell=True,
        ).decode("ascii")
    except:
        return None

    res = [x.strip() for x in res.split("\n") if x != ""]

    if "molecule converted" in res[-1]:
        return res[-2]

    return None


def smi2inchi(smi):
    """Converts SMILES string to InChI string."""
    try:
        res = subprocess.check_output(
            'obabel -:"%s" -oinchi' % smi, stderr=subprocess.STDOUT, shell=True
        ).decode("ascii")
    except:
        return None

    res = [x.strip() for x in res.split("\n") if x != ""]

    if "molecule converted" in res[-1]:
        return res[-2]

    return None


def canonicalize(smiles):
    """Converts SMILES string to canonical SMILES string."""

    try:
        res = subprocess.check_output(
            'echo "%s" | obabel -ismi -ocan' % smiles,
            stderr=subprocess.STDOUT,
            shell=True,
        ).decode("ascii")
    except:
        return None

    res = [x.strip() for x in res.split("\n") if x != ""]

    if "molecule converted" in res[-1]:
        return res[-2]

    return None


def desalt(smiles):
    """Desalts a canonical SMILES string."""

    try:
        res = subprocess.check_output(
            'echo "%s" | obabel -ican -r -ocan' % smiles,
            stderr=subprocess.STDOUT,
            shell=True,
        ).decode("ascii")
    except:
        return None

    res = [x.strip() for x in res.split("\n") if x != ""]

    if "molecule converted" in res[-1]:
        return res[-2]

    return None


def neutralize2(smiles):
    """Neutralizes an canonical SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def _InitializeNeutralisationReactions():
        patts = (
            # Imidazoles
            ("[n+;H]", "n"),
            # Amines
            ("[N+;!H0]", "N"),
            # Carboxylic acids and alcohols
            ("[$([O-]);!$([O-][#7])]", "O"),
            # Thiols
            ("[S-;X1]", "S"),
            # Sulfonamides
            ("[$([N-;X2]S(=O)=O)]", "N"),
            # Enamines
            ("[$([N-;X2][C,N]=C)]", "N"),
            # Tetrazoles
            ("[n-]", "[nH]"),
            # Sulfoxides
            ("[$([S-]=O)]", "S"),
            # Amides
            ("[$([N-]C=O)]", "N"),
        )
        return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]

    reactions = _InitializeNeutralisationReactions()
    mol = Chem.MolFromSmiles(smiles)
    replaced = False
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return canonicalize(Chem.MolToSmiles(mol))
    else:
        return smiles


def neutralize(smiles):
    """Neutralizes an canonical SMILES string (alternate)."""

    def neutralize_inchi(inchi):
        """Neutralizes an InChI string."""
        if "q" in inchi:
            layers = inchi.split("/")
            new = layers[0]
            for i in range(1, len(layers)):
                if "q" not in layers[i]:
                    new += "/" + layers[i]
            return new
        return inchi

    inchi = smi2inchi(smiles)
    inchi = neutralize_inchi(inchi)
    return inchi2smi(inchi)


def tautomerize(smiles):
    """Determines major tautomer of canonical SMILES string."""

    res = subprocess.check_output(
        'cxcalc majortautomer -f smiles "%s"' % smiles,
        stderr=subprocess.STDOUT,
        shell=True,
    ).decode("ascii")

    res = [x.strip() for x in res.split("\n") if x != ""]

    if len(res) > 1:
        return None
    else:
        return canonicalize(res[0])


def smiles2formula(smiles):
    """Determines formula from canonical SMILES string."""

    res = subprocess.check_output(
        'cxcalc formula "%s"' % smiles, stderr=subprocess.STDOUT, shell=True
    ).decode("ascii")

    res = [x.strip() for x in res.split("\n") if x != ""]
    return res[-1].split()[-1].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process InChI/SMILES string")
    parser.add_argument(
        "infile", help="path to input inchi (.inchi) or smiles (.smi) file"
    )
    parser.add_argument("outfile", help="path to output smiles (.smi) file")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
        help="print version and exit",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inchi", action="store_true", help="inchi to smiles mode")
    mode.add_argument("--desalt", action="store_true", help="desalt mode")
    mode.add_argument("--neutralize", action="store_true", help="neutralize mode")
    mode.add_argument("--tautomerize", action="store_true", help="tautomerize mode")
    mode.add_argument("--formula", action="store_true", help="formula mode")
    mode.add_argument(
        "--canonicalize", action="store_true", help="canonicalization mode"
    )

    args = parser.parse_args()

    s = read_string(args.infile)

    if args.inchi is True:
        smiles = inchi2smi(s)
    elif args.desalt is True:
        smiles = desalt(s)
    elif args.neutralize is True:
        smiles = neutralize2(s)
    elif args.tautomerize is True:
        smiles = tautomerize(s)
    elif args.formula is True:
        smiles = smiles2formula(s)
    elif args.canonicalize is True:
        smiles = canonicalize(s)

    if smiles is not None:
        write_string(smiles, args.outfile)
    else:
        sys.exit(1)
