"""Simple, rule-based J-coupling estimates for predicted NMR spectra.

Nothing here is a quantum-chemical calculation. The numbers are textbook
typical values plus a Karplus curve, good enough to say "this CH2 is a
quartet of about 7 Hz" next to a predicted shift — not to fit a measured
spectrum.

Two estimates are offered:

* ``predict_hh_couplings`` — proton-proton couplings for a 1H spectrum:
  geminal 2J, vicinal 3J (Karplus on ring bonds, rotational average on
  free single bonds, cis/trans for alkenes, ortho for arenes) and the
  aromatic meta 4J. Protons on O, N and S are treated as exchanging and
  left uncoupled, the way they usually appear.
* ``predict_ch_couplings`` — the one-bond 13C-1H coupling of each carbon,
  i.e. what a proton-coupled 13C spectrum shows (s/d/t/q splitting).

Both work on an RDKit molecule with explicit hydrogens. A 3D conformer is
used for the Karplus dihedrals when there is one; without it ring bonds fall
back to the averaged value.

The module is pure Python + RDKit and has no Qt dependency, so it is shared
verbatim between the CASCADE and the nmrshiftdb2 predictor plugins.
"""

from __future__ import annotations

import math
from math import comb

#: Couplings smaller than this (Hz) are not resolved in a routine spectrum
#: and are dropped from the multiplet.
MIN_RESOLVED_J = 1.0

#: Partner groups whose couplings differ by less than this (Hz) are merged
#: when naming a multiplet: a dd with 7.1 and 6.9 Hz looks like a triplet.
MERGE_TOLERANCE_J = 1.0

# Typical values (Hz) — Pretsch / Silverstein tables.
J_GEMINAL_SP3 = 12.0  # |2J| H-C(sp3)-H, sign dropped for first-order shapes
J_GEMINAL_SP2 = 2.0  # =CH2
J_VICINAL_FREE = 7.0  # 3J across a freely rotating single bond
J_ALKENE_CIS = 10.0
J_ALKENE_TRANS = 17.0
J_AROMATIC_ORTHO = 7.5
J_AROMATIC_META = 1.5
J_ALDEHYDE_VICINAL = 2.5  # 3J H-C(=O)-C-H

# Karplus coefficients, J(phi) = A cos^2(phi) + B cos(phi) + C.
# The generic H-C-C-H set (Haasnoot/Altona without the electronegativity
# terms); it gives ~1.4 Hz at 90 deg and ~10.3 Hz at 180 deg.
KARPLUS_A = 7.76
KARPLUS_B = -1.10
KARPLUS_C = 1.40

#: Atoms whose protons exchange and are shown uncoupled.
_EXCHANGEABLE = {"O", "N", "S"}

_MULT_LETTERS = {1: "s", 2: "d", 3: "t", 4: "q", 5: "quint", 6: "sext", 7: "sept"}


def karplus(phi_deg: float) -> float:
    """Vicinal H-C-C-H coupling (Hz) for a dihedral angle in degrees."""
    c = math.cos(math.radians(phi_deg))
    return KARPLUS_A * c * c + KARPLUS_B * c + KARPLUS_C


# ---------------------------------------------------------------------------
# Molecule helpers
# ---------------------------------------------------------------------------


def _has_3d(mol) -> bool:
    try:
        return mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()
    except Exception:
        return False


def _dihedral(mol, a, b, c, d):
    """Dihedral a-b-c-d in degrees from the first conformer, or None."""
    if not _has_3d(mol):
        return None
    try:
        from rdkit.Chem import rdMolTransforms

        return rdMolTransforms.GetDihedralDeg(mol.GetConformer(), a, b, c, d)
    except Exception:
        return None


def symmetry_classes(mol) -> list:
    """Equivalence class of every atom (equal = equivalent).

    Topological symmetry, except that the two protons of a terminal =CH2 are
    kept apart: one is cis and one trans to the substituent, which the
    canonical ranking cannot see, and merging them would average a 10 Hz
    and a 17 Hz coupling into a meaningless 13.5 Hz.
    """
    from rdkit import Chem

    classes = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    offset = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        parent = _parent(atom)
        if (
            parent is not None
            and not parent.GetIsAromatic()
            and _hybridization(parent) == "sp2"
            and sum(1 for n in parent.GetNeighbors() if n.GetAtomicNum() == 1) == 2
        ):
            classes[atom.GetIdx()] = offset + atom.GetIdx()
    return classes


def _is_exchangeable_h(atom) -> bool:
    if atom.GetAtomicNum() != 1:
        return False
    nbrs = atom.GetNeighbors()
    return bool(nbrs) and nbrs[0].GetSymbol() in _EXCHANGEABLE


def _parent(atom):
    nbrs = atom.GetNeighbors()
    return nbrs[0] if len(nbrs) == 1 else None


def _hybridization(atom) -> str:
    """'sp', 'sp2' or 'sp3', from bond orders rather than RDKit's perception.

    RDKit marks the carbon next to a carbonyl or an amide nitrogen as sp2 in
    some cases; for coupling purposes only real multiple bonds count.
    """
    if atom.GetIsAromatic():
        return "sp2"
    triple = double = 0
    for bond in atom.GetBonds():
        order = bond.GetBondTypeAsDouble()
        if order >= 2.9:
            triple += 1
        elif order >= 1.9:
            double += 1
    if triple or double >= 2:
        return "sp"
    if double:
        return "sp2"
    return "sp3"


def _is_carbonyl_carbon(atom) -> bool:
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetSymbol() == "O" and bond.GetBondTypeAsDouble() >= 1.9:
            return True
    return False


# ---------------------------------------------------------------------------
# 1H-1H couplings
# ---------------------------------------------------------------------------


def _vicinal_j(mol, h1, c1, c2, h2):
    """3J for H1-C1-C2-H2, or None when the pair should not couple."""
    bond = mol.GetBondBetweenAtoms(c1.GetIdx(), c2.GetIdx())
    if bond is None:
        return None
    order = bond.GetBondTypeAsDouble()

    # Aromatic ortho protons.
    if bond.GetIsAromatic():
        return J_AROMATIC_ORTHO, "3J ortho"

    # Alkene: cis or trans from the geometry (or the stereo label).
    if order >= 1.9:
        phi = _dihedral(mol, h1.GetIdx(), c1.GetIdx(), c2.GetIdx(), h2.GetIdx())
        if phi is not None:
            if abs(phi) > 90.0:
                return J_ALKENE_TRANS, "3J trans"
            return J_ALKENE_CIS, "3J cis"
        # No 3D: fall back to the mean of cis and trans.
        return (J_ALKENE_CIS + J_ALKENE_TRANS) / 2.0, "3J alkene"

    if order >= 1.4:  # any other non-single bond (should not happen)
        return None

    # Aldehyde proton to the alpha CH.
    if _is_carbonyl_carbon(c1) or _is_carbonyl_carbon(c2):
        return J_ALDEHYDE_VICINAL, "3J CHO"

    # Single bond. In a ring the conformation is locked enough for Karplus;
    # an acyclic bond rotates and averages to ~7 Hz.
    if bond.IsInRing():
        phi = _dihedral(mol, h1.GetIdx(), c1.GetIdx(), c2.GetIdx(), h2.GetIdx())
        if phi is not None:
            return round(karplus(phi), 1), "3J Karplus"
    return J_VICINAL_FREE, "3J"


def predict_hh_couplings(mol, min_j: float = MIN_RESOLVED_J) -> list:
    """Estimate 1H-1H couplings.

    Returns a list of dicts ``{"h1", "h2", "j", "kind"}`` (atom indices of
    ``mol``, ``h1 < h2``, ``j`` in Hz, always positive). ``mol`` must carry
    explicit hydrogens.
    """
    couplings = []
    hydrogens = [
        a for a in mol.GetAtoms() if a.GetAtomicNum() == 1 and not _is_exchangeable_h(a)
    ]
    h_set = {a.GetIdx() for a in hydrogens}
    seen = set()

    for h1 in hydrogens:
        c1 = _parent(h1)
        if c1 is None:
            continue
        # 2J: another H on the same carbon.
        for h2 in c1.GetNeighbors():
            j_idx = h2.GetIdx()
            if j_idx == h1.GetIdx() or j_idx not in h_set:
                continue
            key = tuple(sorted((h1.GetIdx(), j_idx)))
            if key in seen:
                continue
            seen.add(key)
            hyb = _hybridization(c1)
            if hyb == "sp3":
                j, kind = J_GEMINAL_SP3, "2J gem"
            elif hyb == "sp2":
                j, kind = J_GEMINAL_SP2, "2J gem"
            else:
                continue
            if j >= min_j:
                couplings.append({"h1": key[0], "h2": key[1], "j": j, "kind": kind})

        for c2 in c1.GetNeighbors():
            if c2.GetIdx() == h1.GetIdx() or c2.GetAtomicNum() == 1:
                continue
            # 3J: H on the neighbouring heavy atom.
            for h2 in c2.GetNeighbors():
                j_idx = h2.GetIdx()
                if j_idx not in h_set:
                    continue
                key = tuple(sorted((h1.GetIdx(), j_idx)))
                if key in seen:
                    continue
                seen.add(key)
                result = _vicinal_j(mol, h1, c1, c2, h2)
                if result is None:
                    continue
                j, kind = result
                if j >= min_j:
                    couplings.append({"h1": key[0], "h2": key[1], "j": j, "kind": kind})

            # 4J aromatic meta: H1-C1:C2:C3-H2 with all three carbons in one ring.
            if not c1.GetIsAromatic() or not c2.GetIsAromatic():
                continue
            ring_info = mol.GetRingInfo()
            for c3 in c2.GetNeighbors():
                if c3.GetIdx() == c1.GetIdx() or not c3.GetIsAromatic():
                    continue
                if not _share_ring(ring_info, c1.GetIdx(), c3.GetIdx()):
                    continue
                for h2 in c3.GetNeighbors():
                    j_idx = h2.GetIdx()
                    if j_idx not in h_set:
                        continue
                    key = tuple(sorted((h1.GetIdx(), j_idx)))
                    if key in seen:
                        continue
                    seen.add(key)
                    if J_AROMATIC_META >= min_j:
                        couplings.append(
                            {"h1": key[0], "h2": key[1], "j": J_AROMATIC_META, "kind": "4J meta"}
                        )
    return couplings


def _share_ring(ring_info, a, b) -> bool:
    return any(a in ring and b in ring for ring in ring_info.AtomRings())


# ---------------------------------------------------------------------------
# Multiplets
# ---------------------------------------------------------------------------


def multiplicity_name(pattern) -> str:
    """First-order name for ``[(J, n_partners), ...]`` sorted by J descending.

    ``[]`` -> "s", ``[(7, 3)]`` -> "q", ``[(10, 1), (2, 1)]`` -> "dd",
    ``[(17, 1), (7, 2)]`` -> "dt". Anything needing more than three letters,
    or more than six equivalent partners in one set, is "m".
    """
    if not pattern:
        return "s"
    parts = []
    for _j, n in pattern:
        letter = _MULT_LETTERS.get(n + 1)
        if letter is None:
            return "m"
        parts.append(letter)
    if len(parts) == 1:
        return parts[0]
    if len(parts) > 3 or any(len(p) > 1 for p in parts):
        return "m"
    return "".join(parts)


def merge_pattern(partners, tolerance: float = MERGE_TOLERANCE_J) -> list:
    """Collapse ``[(J, n), ...]`` sets whose J agree within ``tolerance``.

    The merged J is the partner-weighted mean. Returns the pattern sorted by
    J, largest first.
    """
    merged = []
    for j, n in sorted(partners, key=lambda p: -p[0]):
        if merged and abs(merged[-1][0] - j) < tolerance:
            j0, n0 = merged[-1]
            merged[-1] = ((j0 * n0 + j * n) / (n0 + n), n0 + n)
        else:
            merged.append((j, n))
    return [(round(j, 1), n) for j, n in merged]


def hh_multiplets(mol, couplings, groups=None) -> dict:
    """First-order multiplet of every coupled hydrogen.

    ``groups`` maps atom index -> equivalence key; protons sharing a key do
    not split each other (defaults to ``symmetry_classes``). Couplings are
    averaged over each group, as fast rotation and ring flipping do in the
    real spectrum, so equivalent protons always get the same multiplet.
    Returns ``{h_idx: {"mult", "pattern", "j_text"}}`` for every non-
    exchangeable hydrogen; uncoupled ones come back as singlets.
    """
    if groups is None:
        classes = symmetry_classes(mol)
        groups = {i: classes[i] for i in range(mol.GetNumAtoms())}

    def key_of(idx):
        key = groups.get(idx)
        return ("atom", idx) if key is None else key

    hydrogens = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 1 and not _is_exchangeable_h(a)
    ]
    size = {}
    for idx in hydrogens:
        size[key_of(idx)] = size.get(key_of(idx), 0) + 1

    # (group, partner group) -> [J of every coupling between the two groups]
    between = {}
    for c in couplings:
        ga, gb = key_of(c["h1"]), key_of(c["h2"])
        if ga == gb:
            continue  # equivalent protons do not split each other
        between.setdefault((ga, gb), []).append(c["j"])
        between.setdefault((gb, ga), []).append(c["j"])

    patterns = {}
    for (ga, _gb), js in between.items():
        # Each member of ga couples to len(js)/|ga| members of gb on average.
        n = max(1, int(round(len(js) / float(size.get(ga, 1)))))
        patterns.setdefault(ga, []).append((sum(js) / len(js), n))

    result = {}
    for idx in hydrogens:
        pattern = merge_pattern(patterns.get(key_of(idx), []))
        result[idx] = {
            "mult": multiplicity_name(pattern),
            "pattern": pattern,
            "j_text": format_j(pattern),
        }
    return result


def format_j(pattern) -> str:
    """'J = 7.5, 1.5 Hz' style text, empty for a singlet."""
    if not pattern:
        return ""
    return ", ".join(f"{j:.1f}" for j, _n in pattern)


def multiplet_lines(center_ppm: float, pattern, spectrometer_mhz: float) -> list:
    """Stick positions/intensities of a first-order multiplet.

    Each ``(J, n)`` set splits every line into ``n + 1`` lines with binomial
    intensities. The total intensity is normalised to 1, so the caller can
    scale it by the number of nuclei. Returns ``[(ppm, intensity), ...]``.
    """
    lines = [(0.0, 1.0)]  # offset in Hz, intensity
    for j, n in pattern:
        weights = [comb(n, k) for k in range(n + 1)]
        total = float(sum(weights))
        new = []
        for off, inten in lines:
            for k, w in enumerate(weights):
                new.append((off + (k - n / 2.0) * j, inten * w / total))
        lines = new
    mhz = float(spectrometer_mhz) if spectrometer_mhz else 400.0
    # Merge coincident lines so the stick plot stays light.
    merged = {}
    for off, inten in lines:
        key = round(off, 3)
        merged[key] = merged.get(key, 0.0) + inten
    return [(center_ppm + off / mhz, inten) for off, inten in sorted(merged.items())]


# ---------------------------------------------------------------------------
# 13C-1H one-bond couplings
# ---------------------------------------------------------------------------

#: 1J(CH) base values by hybridization (CH4, C2H4/benzene, C2H2).
J_CH_BASE = {"sp3": 125.0, "sp2": 157.0, "sp": 249.0}

#: Increments (Hz) per substituent on an sp3 carbon — Malinowski's
#: additivity rule, e.g. CH3OH 141, CH3Cl 150, CH2Cl2 178.
J_CH_SP3_INCREMENT = {
    "F": 24.0,
    "Cl": 25.0,
    "Br": 27.0,
    "I": 26.0,
    "O": 16.0,
    "N": 8.0,
    "S": 13.0,
}

#: 1J(CH) of a formyl C-H (aldehydes ~172, formates/formamides ~190-226).
J_CH_ALDEHYDE = 172.0
J_CH_AROMATIC = 159.0

_CH_MULT_LETTERS = {0: "s", 1: "d", 2: "t", 3: "q"}
_CH_DEPT = {0: "C", 1: "CH", 2: "CH2", 3: "CH3"}


def one_bond_ch(atom) -> float:
    """Estimated 1J(13C-1H) in Hz for a carbon atom (whether or not it has H)."""
    hyb = _hybridization(atom)
    if hyb == "sp2":
        if atom.GetIsAromatic():
            base = J_CH_AROMATIC
        elif _is_carbonyl_carbon(atom):
            # H-C(=O)-X: an extra heteroatom (formate, formamide) adds more.
            hetero = sum(
                1
                for n in atom.GetNeighbors()
                if n.GetSymbol() in ("O", "N")
                and atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()
                < 1.9
            )
            base = J_CH_ALDEHYDE + 20.0 * hetero
        else:
            base = J_CH_BASE["sp2"]
            # Enol ethers / enamines: heteroatom on the alkene carbon.
            if any(n.GetSymbol() in ("O", "N") for n in atom.GetNeighbors()):
                base += 20.0
        return round(base, 0)
    if hyb == "sp":
        return J_CH_BASE["sp"]
    j = J_CH_BASE["sp3"]
    for n in atom.GetNeighbors():
        j += J_CH_SP3_INCREMENT.get(n.GetSymbol(), 0.0)
    return round(j, 0)


def predict_ch_couplings(mol) -> dict:
    """One-bond C-H coupling of every carbon.

    Returns ``{c_idx: {"n_h", "mult", "dept", "j_ch", "pattern", "j_text"}}``;
    ``mult`` is the proton-coupled 13C multiplicity (s/d/t/q) and ``j_ch``
    is ``None`` for a quaternary carbon. Hydrogens may be explicit or
    implicit.
    """
    result = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        n_h = atom.GetTotalNumHs(includeNeighbors=True)
        j = one_bond_ch(atom) if n_h else None
        pattern = [(j, n_h)] if n_h else []
        result[atom.GetIdx()] = {
            "n_h": n_h,
            "mult": _CH_MULT_LETTERS.get(n_h, "m"),
            "dept": _CH_DEPT.get(n_h, f"CH{n_h}"),
            "j_ch": j,
            "pattern": pattern,
            "j_text": f"{j:.0f}" if j else "",
        }
    return result


def ensure_3d(mol, seed: int = 0xC0FFEE):
    """Return ``mol`` if it has 3D coordinates, else an embedded copy.

    The copy is only used for dihedrals; atom order is unchanged. Returns
    the original molecule (without 3D) when embedding fails, in which case
    the Karplus terms fall back to averaged values.
    """
    if _has_3d(mol):
        return mol
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        copy = Chem.Mol(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(copy, params) != 0:
            return mol
        try:
            AllChem.MMFFOptimizeMolecule(copy, maxIters=500)
        except Exception:
            pass
        return copy
    except Exception:
        return mol
