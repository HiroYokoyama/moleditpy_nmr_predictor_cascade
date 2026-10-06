"""Simple J-coupling (multiplicity) estimates for predicted NMR spectra.

=============================================================================
SHARED MODULE — the same file lives in two plugins:

    moleditpy_nmr_predictor_cascade/nmr_predictor_cascade/coupling.py
    moleditpy_nmr_predicator_nmrshiftdb2/nmr_predicator_nmrshiftdb2/coupling.py

Edit one, copy it to the other, and bump COUPLING_MODULE_VERSION. Both test
suites carry the same tests/test_coupling.py. The module depends on RDKit
only (no Qt, no numpy), so it can be imported anywhere.
=============================================================================

What it does
------------
Given an RDKit molecule with explicit hydrogens and a list of predicted
shifts, it estimates how each signal is split:

* 1H: proton-proton couplings -> first-order multiplets (s, d, t, q, dd, dt,
  ..., m) with their J values in Hz.
* 13C: one-bond carbon-proton couplings -> the multiplicity of a proton-
  COUPLED 13C spectrum (s/d/t/q, i.e. C/CH/CH2/CH3) and 1J(CH) in Hz.

What it is NOT
--------------
A calculation. The J values are textbook typical values (and a Karplus curve
for ring bonds). They are good for "this CH2 is a quartet of about 7 Hz",
not for fitting a measured spectrum. Second-order effects (roofing, AB
systems, magnetic non-equivalence) are ignored.

Public API (everything else is a helper)
----------------------------------------
annotate_predictions(mol_h, items, nucleus)
    The one call a plugin needs. Adds "mult", "pattern" and "j_text" to
    every prediction dict in ``items`` (and "dept", "j_ch" for 13C).
spectrum_sticks(items, spectrometer_mhz, show_multiplets)
    Stick positions/heights for plotting, with or without the splitting.
predict_hh_couplings(mol_h) / hh_multiplets(mol_h, couplings)
predict_ch_couplings(mol_h)
multiplet_lines(center_ppm, pattern, spectrometer_mhz)
karplus(phi_deg)

Data shapes
-----------
pattern    list of (J_Hz, n_partners), largest J first.
           [(7.0, 3)] is a quartet; [(17.0, 1), (7.0, 3)] a dq; [] a singlet.
item       a prediction dict with at least "idx" (atom index in mol_h),
           "atom" ("H" or "C") and "ppm".

The 1H rules
------------
=====================  ===========================================  =======
coupling               when                                          J (Hz)
=====================  ===========================================  =======
2J geminal             two H on one sp3 carbon                       12
2J geminal             =CH2                                          2
3J vicinal             H-C-C-H across a freely rotating single bond  7
3J vicinal             same, but the bond is in a ring               Karplus
3J aldehyde            H-C(=O)-C-H                                   2.5
3J alkene              H-C=C-H, dihedral < 90 deg (cis)              10
3J alkene              H-C=C-H, dihedral > 90 deg (trans)            17
3J aromatic ortho      H-C:C-H                                       7.5
4J aromatic meta       H-C:C:C-H in one ring                         1.5
=====================  ===========================================  =======

* Protons on O, N and S exchange and are shown uncoupled (singlets).
* Equivalent protons (same symmetry class) do not split each other, and
  couplings are averaged over each class — this is what fast rotation and
  ring flipping do, and it guarantees equivalent rows show the same
  multiplet.
* Couplings below MIN_RESOLVED_J are dropped; sets within MERGE_TOLERANCE_J
  of each other are merged (a dd of 7.1 and 6.9 Hz is reported as a t).
* Without a 3D conformer, ensure_3d() embeds one (RDKit ETKDG + MMFF) just
  for the Karplus dihedrals and the alkene cis/trans test.

The 13C rule
------------
1J(CH) by hybridisation: sp3 125, sp2 157, aromatic 159, sp 249, formyl 172
Hz. On sp3 carbons each heteroatom substituent adds Malinowski's increment
(O +16, N +8, S +13, F +24, Cl +25, Br +27, I +26): CH3OH 141, CH2Cl2 175.
"""

from __future__ import annotations

import math

COUPLING_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Constants (Hz). Typical values from the Pretsch / Silverstein tables.
# ---------------------------------------------------------------------------

#: Smaller couplings are not resolved in a routine spectrum and are dropped.
MIN_RESOLVED_J = 1.0

#: Coupling sets closer than this are merged when naming a multiplet.
MERGE_TOLERANCE_J = 1.0

J_GEMINAL_SP3 = 12.0  # |2J|; the sign does not change a first-order pattern
J_GEMINAL_SP2 = 2.0
J_VICINAL_FREE = 7.0
J_ALDEHYDE_VICINAL = 2.5
J_ALKENE_CIS = 10.0
J_ALKENE_TRANS = 17.0
J_AROMATIC_ORTHO = 7.5
J_AROMATIC_META = 1.5

# Karplus curve J(phi) = A cos^2(phi) + B cos(phi) + C, the generic H-C-C-H
# parameter set: about 1.4 Hz at 90 deg, 10.3 Hz at 180 deg, 8.1 Hz at 0 deg.
KARPLUS_A = 7.76
KARPLUS_B = -1.10
KARPLUS_C = 1.40

# 13C-1H one-bond couplings.
J_CH_SP3 = 125.0
J_CH_SP2 = 157.0
J_CH_AROMATIC = 159.0
J_CH_SP = 249.0
J_CH_FORMYL = 172.0
J_CH_HETERO_SP2 = 20.0  # extra for O/N on an sp2 carbon (enol ether, formate)
J_CH_SP3_INCREMENT = {"O": 16.0, "N": 8.0, "S": 13.0, "F": 24.0, "Cl": 25.0, "Br": 27.0, "I": 26.0}

#: Protons on these elements exchange and are shown uncoupled.
EXCHANGEABLE_PARENTS = {"O", "N", "S"}

#: Multiplet letter for n+1 lines.
_LETTER = {1: "s", 2: "d", 3: "t", 4: "q", 5: "quint", 6: "sext", 7: "sept"}
_DEPT = {0: "C", 1: "CH", 2: "CH2", 3: "CH3"}


# ===========================================================================
# 1. The one-call API used by the plugins
# ===========================================================================


def annotate_predictions(mol_h, items, nucleus):
    """Add coupling information to predicted shifts, in place.

    ``mol_h``   RDKit molecule with explicit hydrogens; ``item["idx"]`` are
                indices into it. A 3D conformer is used if present.
    ``items``   list of prediction dicts (see "Data shapes" above).
    ``nucleus`` "1H" or "13C".

    Every item gets ``mult`` (e.g. "dq"), ``pattern`` and ``j_text`` (e.g.
    "17.0, 7.0"); 13C items also get ``dept`` ("CH3") and ``j_ch`` (Hz or
    None). An item this module cannot handle is left with empty values, so
    a failure here never hides the shifts themselves.
    """
    try:
        if nucleus == "1H":
            geom = ensure_3d(mol_h)
            info = hh_multiplets(geom, predict_hh_couplings(geom))
        else:
            info = predict_ch_couplings(mol_h)
    except Exception:
        info = {}

    for item in items:
        found = info.get(item.get("idx"), {})
        item["mult"] = found.get("mult", "")
        item["pattern"] = found.get("pattern", [])
        item["j_text"] = found.get("j_text", "")
        if nucleus != "1H":
            item["dept"] = found.get("dept", "")
            item["j_ch"] = found.get("j_ch")
    return items


def spectrum_sticks(items, spectrometer_mhz, show_multiplets=True):
    """Stick spectrum for plotting: ``[(ppm, intensity), ...]``.

    Each nucleus contributes intensity 1. Without multiplets the nuclei at
    one shift stack into a single stick (the old behaviour); with them each
    signal is split by its ``pattern`` at the given spectrometer frequency
    (MHz of the observed nucleus — 400 for 1H on a 400 MHz magnet, 100 for
    13C on the same magnet).
    """
    sticks = {}
    for item in items:
        if show_multiplets and item.get("pattern"):
            lines = multiplet_lines(item["ppm"], item["pattern"], spectrometer_mhz)
        else:
            lines = [(item["ppm"], 1.0)]
        for ppm, height in lines:
            key = round(ppm, 5)
            sticks[key] = sticks.get(key, 0.0) + height
    return sorted(sticks.items())


# ===========================================================================
# 2. 1H-1H couplings
# ===========================================================================


def karplus(phi_deg):
    """Vicinal H-C-C-H coupling (Hz) for a dihedral angle in degrees."""
    c = math.cos(math.radians(phi_deg))
    return KARPLUS_A * c * c + KARPLUS_B * c + KARPLUS_C


def predict_hh_couplings(mol_h, min_j=MIN_RESOLVED_J):
    """Every resolved H-H coupling: ``[{"h1", "h2", "j", "kind"}, ...]``.

    ``h1 < h2`` are atom indices, ``j`` is positive (Hz) and ``kind`` a
    label such as "3J Karplus". Exchangeable protons are skipped.
    """
    hydrogens = [a for a in mol_h.GetAtoms() if _is_coupling_proton(a)]
    proton_ids = {a.GetIdx() for a in hydrogens}
    found = {}

    def add(a, b, j, kind):
        key = (min(a, b), max(a, b))
        if key not in found and j >= min_j:
            found[key] = {"h1": key[0], "h2": key[1], "j": j, "kind": kind}

    for h1 in hydrogens:
        c1 = h1.GetNeighbors()[0]

        # 2J — another proton on the same carbon.
        for h2 in _protons_on(c1, proton_ids):
            if h2.GetIdx() != h1.GetIdx():
                gem = _geminal_j(c1)
                if gem is not None:
                    add(h1.GetIdx(), h2.GetIdx(), gem, "2J gem")

        for c2 in c1.GetNeighbors():
            if c2.GetAtomicNum() == 1:
                continue
            # 3J — a proton on the next heavy atom.
            for h2 in _protons_on(c2, proton_ids):
                vic = _vicinal_j(mol_h, h1, c1, c2, h2)
                if vic is not None:
                    add(h1.GetIdx(), h2.GetIdx(), *vic)
            # 4J — aromatic meta: H-C:C:C-H with all three carbons in one ring.
            if c1.GetIsAromatic() and c2.GetIsAromatic():
                for c3 in c2.GetNeighbors():
                    if (
                        c3.GetIdx() != c1.GetIdx()
                        and c3.GetIsAromatic()
                        and _in_same_ring(mol_h, c1.GetIdx(), c3.GetIdx())
                    ):
                        for h2 in _protons_on(c3, proton_ids):
                            add(h1.GetIdx(), h2.GetIdx(), J_AROMATIC_META, "4J meta")

    return list(found.values())


def _geminal_j(carbon):
    hyb = _hybridization(carbon)
    if hyb == "sp3":
        return J_GEMINAL_SP3
    if hyb == "sp2":
        return J_GEMINAL_SP2
    return None


def _vicinal_j(mol_h, h1, c1, c2, h2):
    """``(J, kind)`` for H1-C1-C2-H2, or None when the pair is not coupled."""
    bond = mol_h.GetBondBetweenAtoms(c1.GetIdx(), c2.GetIdx())
    if bond is None:
        return None

    if bond.GetIsAromatic():
        return J_AROMATIC_ORTHO, "3J ortho"

    order = bond.GetBondTypeAsDouble()
    if order >= 1.9:  # C=C: decide cis/trans from the geometry
        phi = _dihedral(mol_h, h1, c1, c2, h2)
        if phi is None:
            return (J_ALKENE_CIS + J_ALKENE_TRANS) / 2.0, "3J alkene"
        if abs(phi) > 90.0:
            return J_ALKENE_TRANS, "3J trans"
        return J_ALKENE_CIS, "3J cis"
    if order > 1.1:  # anything else that is not a plain single bond
        return None

    if _is_carbonyl_carbon(c1) or _is_carbonyl_carbon(c2):
        return J_ALDEHYDE_VICINAL, "3J CHO"

    # A ring bond is held in one conformation, so the Karplus curve applies;
    # an open-chain bond rotates and averages out to about 7 Hz.
    if bond.IsInRing():
        phi = _dihedral(mol_h, h1, c1, c2, h2)
        if phi is not None:
            return round(karplus(phi), 1), "3J Karplus"
    return J_VICINAL_FREE, "3J"


# ===========================================================================
# 3. From couplings to multiplets
# ===========================================================================


def symmetry_classes(mol_h):
    """Equivalence class of every atom; equal numbers = equivalent atoms.

    Topological symmetry (RDKit canonical ranking without tie-breaking),
    with one correction: the two protons of a terminal =CH2 are kept apart.
    One is cis and one trans to the substituent, which the ranking cannot
    see, and merging them would average 10 and 17 Hz into a fake 13.5 Hz.
    """
    from rdkit import Chem

    classes = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))
    offset = mol_h.GetNumAtoms()
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() != 1 or not atom.GetNeighbors():
            continue
        parent = atom.GetNeighbors()[0]
        if (
            not parent.GetIsAromatic()
            and _hybridization(parent) == "sp2"
            and _count_h_neighbours(parent) == 2
        ):
            classes[atom.GetIdx()] = offset + atom.GetIdx()  # a class of its own
    return classes


def hh_multiplets(mol_h, couplings, classes=None):
    """First-order multiplet of every non-exchangeable proton.

    Returns ``{h_idx: {"mult", "pattern", "j_text"}}``. ``classes`` defaults
    to ``symmetry_classes(mol_h)``.

    How: for each pair of equivalence classes (A, B), every coupling
    between their members is collected. The J of the A-B set is the mean,
    and the number of B partners one A proton sees is
    (number of A-B couplings) / (size of A). That is the fast-exchange
    average and gives every member of A the same multiplet.
    """
    if classes is None:
        classes = symmetry_classes(mol_h)

    protons = [a.GetIdx() for a in mol_h.GetAtoms() if _is_coupling_proton(a)]
    class_size = {}
    for idx in protons:
        class_size[classes[idx]] = class_size.get(classes[idx], 0) + 1

    between = {}  # (class A, class B) -> [J, ...]
    for c in couplings:
        a, b = classes[c["h1"]], classes[c["h2"]]
        if a == b:
            continue  # equivalent protons do not split each other
        between.setdefault((a, b), []).append(c["j"])
        between.setdefault((b, a), []).append(c["j"])

    sets_of_class = {}  # class A -> [(mean J, partners), ...]
    for (a, _b), js in between.items():
        partners = max(1, int(round(len(js) / float(class_size.get(a, 1)))))
        sets_of_class.setdefault(a, []).append((sum(js) / len(js), partners))

    result = {}
    for idx in protons:
        pattern = merge_pattern(sets_of_class.get(classes[idx], []))
        result[idx] = {
            "mult": multiplicity_name(pattern),
            "pattern": pattern,
            "j_text": format_j(pattern),
        }
    return result


def merge_pattern(sets, tolerance=MERGE_TOLERANCE_J):
    """Merge ``(J, n)`` sets whose J agree within ``tolerance``.

    The merged J is the partner-weighted mean; the result is sorted by J,
    largest first, with J rounded to 0.1 Hz.
    """
    merged = []
    for j, n in sorted(sets, key=lambda s: -s[0]):
        if merged and abs(merged[-1][0] - j) < tolerance:
            j0, n0 = merged[-1]
            merged[-1] = ((j0 * n0 + j * n) / (n0 + n), n0 + n)
        else:
            merged.append((j, n))
    return [(round(j, 1), n) for j, n in merged]


def multiplicity_name(pattern):
    """First-order name of a pattern.

    []                    -> "s"
    [(7.0, 3)]            -> "q"
    [(10.0, 1), (2.0, 1)] -> "dd"
    [(17.0, 1), (7.0, 3)] -> "dq"
    Four or more sets, a set of more than six partners, or a long name
    (quint/sext/sept) inside a combination -> "m".
    """
    if not pattern:
        return "s"
    letters = []
    for _j, n in pattern:
        letter = _LETTER.get(n + 1)
        if letter is None:
            return "m"
        letters.append(letter)
    if len(letters) == 1:
        return letters[0]
    if len(letters) > 3 or any(len(x) > 1 for x in letters):
        return "m"
    return "".join(letters)


def format_j(pattern):
    """"17.0, 7.0" — the J values of a pattern, empty for a singlet."""
    return ", ".join(f"{j:.1f}" for j, _n in pattern)


def multiplet_lines(center_ppm, pattern, spectrometer_mhz):
    """Lines of a first-order multiplet: ``[(ppm, intensity), ...]``.

    Every (J, n) set splits each line into n + 1 lines with binomial
    (Pascal's triangle) intensities. Total intensity is 1.
    """
    mhz = float(spectrometer_mhz) if spectrometer_mhz else 400.0
    lines = [(0.0, 1.0)]  # (offset in Hz, intensity)
    for j, n in pattern:
        weights = [math.comb(n, k) for k in range(n + 1)]
        total = float(sum(weights))
        lines = [
            (offset + (k - n / 2.0) * j, height * w / total)
            for offset, height in lines
            for k, w in enumerate(weights)
        ]
    merged = {}
    for offset, height in lines:
        key = round(offset, 3)
        merged[key] = merged.get(key, 0.0) + height
    return [(center_ppm + offset / mhz, height) for offset, height in sorted(merged.items())]


# ===========================================================================
# 4. 13C-1H one-bond couplings
# ===========================================================================


def predict_ch_couplings(mol_h):
    """One-bond C-H coupling of every carbon.

    Returns ``{c_idx: {"n_h", "mult", "dept", "j_ch", "pattern", "j_text"}}``.
    ``mult`` is the proton-coupled multiplicity (s/d/t/q), ``dept`` the
    carbon type (C/CH/CH2/CH3), ``j_ch`` None for a quaternary carbon.
    Works with explicit or implicit hydrogens.
    """
    result = {}
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        n_h = atom.GetTotalNumHs(includeNeighbors=True)
        j = one_bond_ch(atom) if n_h else None
        pattern = [(j, n_h)] if n_h else []
        result[atom.GetIdx()] = {
            "n_h": n_h,
            "mult": _LETTER.get(n_h + 1, "m"),
            "dept": _DEPT.get(n_h, f"CH{n_h}"),
            "j_ch": j,
            "pattern": pattern,
            "j_text": f"{j:.0f}" if j else "",
        }
    return result


def one_bond_ch(carbon):
    """Estimated 1J(13C-1H) in Hz for a carbon atom."""
    hyb = _hybridization(carbon)
    if hyb == "sp":
        return J_CH_SP
    if hyb == "sp2":
        if carbon.GetIsAromatic():
            return J_CH_AROMATIC
        hetero = sum(1 for n in _single_bonded(carbon) if n.GetSymbol() in ("O", "N"))
        if _is_carbonyl_carbon(carbon):  # aldehyde 172, formate/formamide ~192
            return J_CH_FORMYL + J_CH_HETERO_SP2 * hetero
        return J_CH_SP2 + (J_CH_HETERO_SP2 if hetero else 0.0)
    j = J_CH_SP3
    for n in carbon.GetNeighbors():
        j += J_CH_SP3_INCREMENT.get(n.GetSymbol(), 0.0)
    return j


# ===========================================================================
# 5. Helpers
# ===========================================================================


def ensure_3d(mol_h, seed=0xC0FFEE):
    """``mol_h`` itself if it has 3D coordinates, else an embedded copy.

    The copy (same atom order) only feeds the dihedral angles. If embedding
    fails the 2D molecule is returned and the angle-dependent rules fall
    back to their averaged values.
    """
    if _has_3d(mol_h):
        return mol_h
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        copy = Chem.Mol(mol_h)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(copy, params) != 0:
            return mol_h
        try:
            AllChem.MMFFOptimizeMolecule(copy, maxIters=500)
        except Exception:
            pass  # an unoptimised embedding is still fine for dihedrals
        return copy
    except Exception:
        return mol_h


def _has_3d(mol):
    try:
        return mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()
    except Exception:
        return False


def _dihedral(mol, h1, c1, c2, h2):
    """Dihedral H1-C1-C2-H2 in degrees from the first conformer, or None."""
    if not _has_3d(mol):
        return None
    try:
        from rdkit.Chem import rdMolTransforms

        return rdMolTransforms.GetDihedralDeg(
            mol.GetConformer(), h1.GetIdx(), c1.GetIdx(), c2.GetIdx(), h2.GetIdx()
        )
    except Exception:
        return None


def _is_coupling_proton(atom):
    """A hydrogen bonded to one atom that is not O, N or S."""
    if atom.GetAtomicNum() != 1:
        return False
    nbrs = atom.GetNeighbors()
    return len(nbrs) == 1 and nbrs[0].GetSymbol() not in EXCHANGEABLE_PARENTS


def _protons_on(atom, proton_ids):
    return [n for n in atom.GetNeighbors() if n.GetIdx() in proton_ids]


def _count_h_neighbours(atom):
    return sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)


def _single_bonded(atom):
    mol = atom.GetOwningMol()
    return [
        n
        for n in atom.GetNeighbors()
        if mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() < 1.5
    ]


def _hybridization(atom):
    """'sp', 'sp2' or 'sp3', counted from multiple bonds.

    RDKit's own hybridisation calls some atoms next to a carbonyl or an
    amide nitrogen sp2; for couplings only real multiple bonds count.
    """
    if atom.GetIsAromatic():
        return "sp2"
    double = triple = 0
    for bond in atom.GetBonds():
        order = bond.GetBondTypeAsDouble()
        if order >= 2.9:
            triple += 1
        elif order >= 1.9:
            double += 1
    if triple or double >= 2:
        return "sp"
    return "sp2" if double else "sp3"


def _is_carbonyl_carbon(atom):
    return any(
        bond.GetOtherAtom(atom).GetSymbol() == "O" and bond.GetBondTypeAsDouble() >= 1.9
        for bond in atom.GetBonds()
    )


def _in_same_ring(mol, a, b):
    return any(a in ring and b in ring for ring in mol.GetRingInfo().AtomRings())
