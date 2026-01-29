# unified_smiles_gui_final_physchem_pubchem_fix.py
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import tempfile
import subprocess
import warnings
from datetime import datetime

import pandas as pd
import requests
import pubchempy as pcp

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QSpinBox, QComboBox, QFormLayout, QGroupBox, QCheckBox,
    QDoubleSpinBox, QTextEdit, QListWidget, QListWidgetItem, QPlainTextEdit,
    QSplitter
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

HARTREE_TO_EV = 27.211386245988


# =========================================================
# Shared helpers
# =========================================================
def safe_decode(b: bytes) -> str:
    if b is None:
        return ""
    if isinstance(b, str):
        return b
    # xTB output is usually UTF-8; Windows default cp949 is what caused your crash
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return b.decode(enc, errors="replace")
        except Exception:
            continue
    return b.decode(errors="replace")

def run_cmd_capture_bytes(cmd, cwd=None, timeout=900):
    """
    Robust subprocess capture that avoids Windows cp949 text decoding issues by:
    - capturing stdout/stderr as bytes
    - decoding with safe_decode()
    """
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        shell=False
    )
    try:
        out_b, err_b = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
        out_b, err_b = p.communicate()
        raise RuntimeError("Subprocess timeout.\n" + safe_decode(out_b) + "\n" + safe_decode(err_b))

    out = safe_decode(out_b)
    err = safe_decode(err_b)
    return p.returncode, out, err


# =========================================================
# RDKit drawing helpers (PNG bytes, no Pillow required)
# =========================================================
def _compute_2d(mol: Chem.Mol):
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass

def rdkit_png_from_mols(mols, legends=None, subImgSize=(320, 240), molsPerRow=3) -> bytes:
    if not mols:
        return b""
    ms = []
    for m in mols:
        if m is None:
            continue
        m2 = Chem.Mol(m)
        _compute_2d(m2)
        ms.append(m2)
    if not ms:
        return b""
    if legends is None:
        legends = ["" for _ in range(len(ms))]
    try:
        return Draw.MolsToGridImage(
            ms,
            molsPerRow=min(molsPerRow, len(ms)),
            subImgSize=subImgSize,
            legends=legends,
            returnPNG=True
        )
    except Exception:
        try:
            return Draw.MolsToGridImage([ms[0]], molsPerRow=1, subImgSize=subImgSize, legends=[""], returnPNG=True)
        except Exception:
            return b""

def pixmap_from_png_bytes(png: bytes) -> QPixmap:
    pm = QPixmap()
    if png:
        pm.loadFromData(png, "PNG")
    return pm

class AspectRatioImageLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self._pix = None
        self.setAlignment(Qt.AlignCenter)

    def setPixmapKeep(self, pix: QPixmap):
        self._pix = pix
        if self._pix is not None and not self._pix.isNull():
            super().setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            super().setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pix is not None and not self._pix.isNull():
            super().setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# =========================================================
# SMILES normalization (Fix: sulfonate shorthand + dummy)
# =========================================================
def normalize_smiles_keep_dummy(smiles: str) -> str:
    # keep polymer dummy atoms as [*]
    s = (smiles or "").strip().replace(" ", "")
    token = "__DUMMY__"
    s = s.replace("[*]", token)
    s = s.replace("*", "[*]")
    s = s.replace(token, "[*]")
    return s

def normalize_sulfonate_shorthand(smiles: str) -> str:
    """
    Fix RDKit parse errors for patterns like C[SO3-], CSO3-, SO3-.
    - RDKit-safe sulfonate: S(=O)(=O)[O-]
    """
    s = (smiles or "").strip()

    # Common shorthand variants
    s = s.replace("CSO3-", "CS(=O)(=O)[O-]")
    s = s.replace("C[SO3-]", "CS(=O)(=O)[O-]")
    s = s.replace("[SO3-]", "S(=O)(=O)[O-]")
    s = s.replace("SO3-", "S(=O)(=O)[O-]")

    return s

def preprocess_input_smiles(smiles: str) -> str:
    s = normalize_sulfonate_shorthand(smiles)
    s = normalize_smiles_keep_dummy(s)
    return s

def has_dummy_atoms(smiles: str) -> bool:
    return "[*]" in preprocess_input_smiles(smiles)

def safe_sanitize(mol: Chem.Mol) -> Chem.Mol:
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
            return mol
        except Exception:
            return mol


# =========================================================
# Polymer oligomer builder (same logic)
# =========================================================
def mol_from_smiles(smiles: str):
    s = preprocess_input_smiles(smiles)
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.AddHs(mol)

def rdkit_3d_xyz(mol, xyz_path: str):
    m3 = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    ok = AllChem.EmbedMolecule(m3, params)
    if ok != 0:
        ok = AllChem.EmbedMolecule(m3, randomSeed=42)
        if ok != 0:
            raise RuntimeError("RDKit 3D embedding failed.")

    try:
        AllChem.MMFFOptimizeMolecule(m3)
    except Exception:
        AllChem.UFFOptimizeMolecule(m3)

    conf = m3.GetConformer()
    with open(xyz_path, "w", encoding="utf-8") as f:
        f.write(f"{m3.GetNumAtoms()}\n")
        f.write("generated by RDKit\n")
        for i, atom in enumerate(m3.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol()} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}\n")

def _dummy_indices(mol: Chem.Mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]

def _dummy_neighbor(mol: Chem.Mol, dummy_idx: int) -> int:
    a = mol.GetAtomWithIdx(dummy_idx)
    nbs = [n.GetIdx() for n in a.GetNeighbors()]
    if len(nbs) != 1:
        raise ValueError("Each dummy atom ([*]) must have exactly one neighbor in the repeat unit.")
    return nbs[0]

def build_oligomer_from_repeat_unit(smiles_with_two_dummies: str, n: int = 3, cap: str = "H") -> Chem.Mol:
    if n < 1:
        raise ValueError("n must be >= 1")

    s = preprocess_input_smiles(smiles_with_two_dummies)
    unit0 = Chem.MolFromSmiles(s)
    if unit0 is None:
        raise ValueError("Invalid repeat-unit SMILES (RDKit parse failed).")

    d0 = _dummy_indices(unit0)
    if len(d0) != 2:
        raise ValueError("Polymer mode requires exactly two dummy atoms ([*]) in the repeat unit.")

    head_dummy_u, tail_dummy_u = sorted(d0)
    chain = Chem.Mol(unit0)

    for _ in range(n - 1):
        old_n = chain.GetNumAtoms()
        combined = Chem.CombineMols(chain, unit0)
        rw = Chem.RWMol(combined)

        old_dummies = [i for i in range(old_n) if rw.GetAtomWithIdx(i).GetAtomicNum() == 0]
        if len(old_dummies) != 2:
            raise RuntimeError("Internal error: chain should have exactly two terminal dummies before growth.")
        tail_dummy_old = max(old_dummies)
        tail_nb_old = _dummy_neighbor(rw, tail_dummy_old)

        new_dummies = [i for i in range(old_n, rw.GetNumAtoms()) if rw.GetAtomWithIdx(i).GetAtomicNum() == 0]
        if len(new_dummies) != 2:
            raise RuntimeError("Internal error: unit should contribute exactly two dummies.")
        head_dummy_new = head_dummy_u + old_n
        if head_dummy_new not in new_dummies:
            head_dummy_new = min(new_dummies)
        head_nb_new = _dummy_neighbor(rw, head_dummy_new)

        rw.AddBond(tail_nb_old, head_nb_new, Chem.rdchem.BondType.SINGLE)

        for idx in sorted([head_dummy_new, tail_dummy_old], reverse=True):
            rw.RemoveAtom(idx)

        chain = safe_sanitize(rw.GetMol())

    rw = Chem.RWMol(chain)
    dummies = _dummy_indices(rw)
    if len(dummies) != 2:
        raise RuntimeError("Internal error: expected two terminal dummies before capping.")

    cap_key = (cap or "H").strip().upper()

    if cap_key == "H":
        for idx in sorted(dummies, reverse=True):
            rw.RemoveAtom(idx)
        mol = safe_sanitize(rw.GetMol())
        return Chem.AddHs(mol)

    if cap_key in ["ME", "CH3", "METHYL"]:
        for d_idx in sorted(dummies, reverse=True):
            nb = _dummy_neighbor(rw, d_idx)
            c_idx = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(nb, c_idx, Chem.rdchem.BondType.SINGLE)
            rw.RemoveAtom(d_idx)
        mol = safe_sanitize(rw.GetMol())
        return Chem.AddHs(mol)

    raise ValueError("cap must be 'H' or 'Me'.")

def mol_weight_gmol(mol: Chem.Mol) -> float:
    return float(Descriptors.MolWt(mol))

def suggest_n_from_target_dp(target_dp: int, n_min=1, n_max=5000) -> int:
    n = int(target_dp)
    return max(n_min, min(n_max, n))

def suggest_n_from_target_mn(repeat_unit_smiles: str, target_mn: float, cap: str = "H", n_max: int = 500):
    if target_mn <= 0:
        raise ValueError("Target Mn must be > 0.")

    m1 = build_oligomer_from_repeat_unit(repeat_unit_smiles, n=1, cap=cap)
    m2 = build_oligomer_from_repeat_unit(repeat_unit_smiles, n=2, cap=cap)

    mw1 = mol_weight_gmol(m1)
    mw2 = mol_weight_gmol(m2)
    delta = mw2 - mw1
    if delta <= 0:
        raise RuntimeError("Failed to estimate repeat-unit increment (ΔM <= 0).")

    n_est = int(round((target_mn - mw1) / delta + 1))
    n_est = max(1, min(n_max, n_est))

    candidates = sorted(set([max(1, min(n_max, k)) for k in range(n_est - 2, n_est + 3)]))
    best = None
    best_err = None
    best_mw = None

    for n in candidates:
        moln = build_oligomer_from_repeat_unit(repeat_unit_smiles, n=n, cap=cap)
        mwn = mol_weight_gmol(moln)
        err = abs(mwn - target_mn)
        if best is None or err < best_err:
            best = n
            best_err = err
            best_mw = mwn

    return best, best_mw, mw1, delta


# =========================================================
# PubChem (PUG-View): extract physchem from
# (Chemical and Physical Properties + Physical Description)
# =========================================================
def get_pubchem_cid_from_smiles(smiles: str):
    if has_dummy_atoms(smiles):
        return None
    try:
        comps = pcp.get_compounds(preprocess_input_smiles(smiles), namespace="smiles")
        if not comps:
            return None
        return comps[0].cid
    except Exception:
        return None

def pubchem_pugview_fetch(cid: int, heading: str | None = None) -> dict:
    if heading:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?heading={requests.utils.quote(heading)}"
    else:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
    headers = {"User-Agent": "Mozilla/5.0 (PubChemPhysChemExtractor/1.0)"}
    r = requests.get(url, timeout=50, headers=headers)
    r.raise_for_status()
    return r.json()

def _pubchem_value_to_text(val):
    if val is None:
        return None

    unit = None
    if isinstance(val, dict):
        unit = val.get("Unit")
        if "StringWithMarkup" in val and isinstance(val["StringWithMarkup"], list):
            ss = []
            for swm in val["StringWithMarkup"]:
                if isinstance(swm, dict) and swm.get("String") is not None:
                    ss.append(str(swm["String"]))
            text = " ".join([s for s in ss if s])
        elif "String" in val:
            text = str(val["String"])
        elif "Number" in val:
            nums = val["Number"]
            if isinstance(nums, list):
                text = "; ".join(str(x) for x in nums)
            else:
                text = str(nums)
        elif "Boolean" in val:
            b = val["Boolean"]
            if isinstance(b, list):
                text = "; ".join(str(x) for x in b)
            else:
                text = str(b)
        else:
            text = None
    else:
        text = str(val)

    if text is None or text == "":
        return None
    if unit:
        return f"{text} {unit}"
    return text

def pubchem_flatten_pugview_json(data: dict, top_section_whitelist=None, max_list_items_per_field=8) -> dict:
    rec = (data or {}).get("Record", {})
    sections = rec.get("Section", [])
    out = {}

    whitelist = set(top_section_whitelist) if top_section_whitelist else None

    def add_value(key, val):
        if val is None:
            return
        if key not in out:
            out[key] = val
            return
        if out[key] == val:
            return
        out[key] = f"{out[key]} ; {val}"

    def in_whitelist(path_titles):
        if whitelist is None:
            return True
        if not path_titles:
            return False
        return path_titles[0] in whitelist

    def walk_section(sec, path_titles):
        title = sec.get("TOCHeading") or sec.get("Title") or sec.get("Heading") or ""
        new_path = path_titles + ([title] if title else [])

        if not in_whitelist(new_path):
            return

        info_list = sec.get("Information", [])
        if isinstance(info_list, list) and info_list:
            for info in info_list:
                if not isinstance(info, dict):
                    continue
                name = info.get("Name") or info.get("TOCHeading") or info.get("Description") or "Value"
                val_raw = info.get("Value")
                val = _pubchem_value_to_text(val_raw)
                if val is None:
                    continue

                key = "PubChem|" + " > ".join([p for p in new_path if p]) + "|" + str(name)
                add_value(key, val)

                if isinstance(val_raw, dict) and "StringWithMarkup" in val_raw:
                    swm = val_raw.get("StringWithMarkup", [])
                    if isinstance(swm, list) and len(swm) > max_list_items_per_field:
                        add_value(key + "|note", f"truncated_to_{max_list_items_per_field}_items")

                if info.get("URL"):
                    add_value(key + "|URL", str(info["URL"]))

        subs = sec.get("Section", [])
        if isinstance(subs, list):
            for s2 in subs:
                if isinstance(s2, dict):
                    walk_section(s2, new_path)

    for sec in sections:
        if isinstance(sec, dict):
            walk_section(sec, [])

    return out

def pubchem_extract_physchem_limited(cid: int) -> dict:
    """
    Extract ONLY:
    - Chemical and Physical Properties
    - Physical Description
    Robust strategy:
    1) try full record + whitelist
    2) fallback: fetch each heading separately if needed
    """
    wanted = ("Chemical and Physical Properties", "Physical Description")
    out = {}

    # Try full record (fewer requests)
    try:
        full = pubchem_pugview_fetch(cid, heading=None)
        out.update(pubchem_flatten_pugview_json(full, top_section_whitelist=wanted))
    except Exception as e:
        out["PubChem|_fetch_full_error"] = str(e)

    # If too empty, fallback per-heading
    if len(out) < 10:
        for h in wanted:
            try:
                part = pubchem_pugview_fetch(cid, heading=h)
                out_part = pubchem_flatten_pugview_json(part, top_section_whitelist=None)
                # keep first seen if collision
                for k, v in out_part.items():
                    out.setdefault(k, v)
            except Exception:
                continue

    return out


# =========================================================
# xTB parsing (FIX: no cp949 decoding crash)
# =========================================================
def _extract_nested_numeric_list(obj, must_contain=("orbital", "ener")):
    if isinstance(obj, dict):
        for k, v in obj.items():
            k_low = k.lower()
            if all(s in k_low for s in must_contain):
                if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                    return v
            got = _extract_nested_numeric_list(v, must_contain)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for it in obj:
            got = _extract_nested_numeric_list(it, must_contain)
            if got is not None:
                return got
    return None

def run_xtb_and_parse(mol_for_calc: Chem.Mol, charge: int = 0, uhf: int = 0):
    with tempfile.TemporaryDirectory() as td:
        rdkit_3d_xyz(mol_for_calc, os.path.join(td, "mol.xyz"))

        with open(os.path.join(td, "xtb.inp"), "w", encoding="utf-8") as f:
            f.write("$write\njson=true\n$end\n")

        cmd = ["xtb", "mol.xyz", "--gfn", "2", "--chrg", str(charge), "--uhf", str(uhf),
               "--input", "xtb.inp", "--json"]

        try:
            rc, out, err = run_cmd_capture_bytes(cmd, cwd=td, timeout=900)
        except FileNotFoundError:
            raise RuntimeError("xtb 실행 파일을 찾을 수 없습니다. (conda env에서 xtb 설치/활성화 확인)")

        out_text = (out or "") + "\n" + (err or "")
        if rc != 0:
            raise RuntimeError(out_text)

        xtb_json = None
        for fn in ["xtbout.json", "xtb.json"]:
            fp = os.path.join(td, fn)
            if os.path.exists(fp):
                xtb_json = fp
                break
        if xtb_json is None:
            raise RuntimeError("xTB JSON not found (xtbout.json).")

        with open(xtb_json, "r", encoding="utf-8", errors="replace") as f:
            j = json.load(f)

        orbital_energies = _extract_nested_numeric_list(j, must_contain=("orbital", "ener"))
        orbital_occs = _extract_nested_numeric_list(j, must_contain=("occup",))

        homo_ev = lumo_ev = gap_ev = None
        if orbital_energies and orbital_occs and len(orbital_energies) == len(orbital_occs):
            occ_idx = [i for i, o in enumerate(orbital_occs) if o is not None and float(o) > 1e-6]
            if occ_idx:
                homo_i = max(occ_idx)
                lumo_i = homo_i + 1 if homo_i + 1 < len(orbital_energies) else None
                homo_ev = float(orbital_energies[homo_i]) * HARTREE_TO_EV
                if lumo_i is not None:
                    lumo_ev = float(orbital_energies[lumo_i]) * HARTREE_TO_EV
                    gap_ev = lumo_ev - homo_ev

        dipole_debye = None
        m = re.search(r"total\s+dipole\s+moment\s*[:=]\s*([0-9\.\-]+)\s*D", out_text, flags=re.I)
        if m:
            try:
                dipole_debye = float(m.group(1))
            except Exception:
                dipole_debye = None

        return {
            "HOMO (eV, xTB)": homo_ev,
            "LUMO (eV, xTB)": lumo_ev,
            "Bandgap (eV, xTB)": gap_ev,
            "Dipole (D, xTB)": dipole_debye,
        }


# =========================================================
# pGrAdd thermo (FIX: warning/absence safe)
# =========================================================
def pgradd_thermo(smiles: str, T: float = 298.15):
    """
    Some environments throw:
    'GroupLibrary.load(): No property sets defined.'
    -> handle gracefully and return Nones + note.
    """
    try:
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            from pgradd.GroupAdd.Library import GroupLibrary
            lib = GroupLibrary.Load("BensonGA")

            warned = any("No property sets defined" in str(w.message) for w in wrec)
            if warned:
                return {
                    "H (kcal/mol)": None, "S (cal/mol/K)": None, "G (kcal/mol)": None, "Cp (cal/mol/K)": None,
                    "pGrAdd note": "No property sets defined (pgradd install/config issue)."
                }

            desc = lib.GetDescriptors(smiles)
            therm = lib.Estimate(desc, "thermochem")
    except Exception as e:
        return {
            "H (kcal/mol)": None, "S (cal/mol/K)": None, "G (kcal/mol)": None, "Cp (cal/mol/K)": None,
            "pGrAdd error": str(e)
        }

    out = {}
    try:
        out["H (kcal/mol)"] = float(therm.get_H(T=T, units="kcal/mol"))
    except Exception:
        out["H (kcal/mol)"] = None
    try:
        out["S (cal/mol/K)"] = float(therm.get_S(T=T, units="cal/(mol*K)"))
    except Exception:
        out["S (cal/mol/K)"] = None
    try:
        out["G (kcal/mol)"] = float(therm.get_G(T=T, units="kcal/mol"))
    except Exception:
        out["G (kcal/mol)"] = None

    cp_val = None
    for name in ["get_Cp", "get_cp"]:
        if hasattr(therm, name):
            try:
                cp_val = float(getattr(therm, name)(T=T, units="cal/(mol*K)"))
                break
            except Exception:
                pass
    out["Cp (cal/mol/K)"] = cp_val
    return out


# =========================================================
# PhysChem compute_all (PubChem limited sections, density 포함 가능)
# =========================================================
def compute_all(
    smiles_input: str,
    polymer_nmer: int = 3,
    polymer_endcap: str = "H",
    charge: int = 0,
    uhf: int = 0,
    T: float = 298.15,
    use_pubchem: bool = True,
    run_xtb: bool = True,
    run_pgradd: bool = True,
):
    s = preprocess_input_smiles(smiles_input)
    is_poly = "[*]" in s

    mol_input = Chem.MolFromSmiles(s)

    if is_poly:
        unit = Chem.MolFromSmiles(s)
        if unit is None:
            raise ValueError("Repeat-unit SMILES parse failed.")
        if len(_dummy_indices(unit)) != 2:
            raise ValueError("Polymer 모드: [*] dummy가 정확히 2개인 repeat unit만 지원합니다.")
        mol_calc = build_oligomer_from_repeat_unit(s, n=int(polymer_nmer), cap=polymer_endcap)
        smiles_for_thermo = Chem.MolToSmiles(Chem.RemoveHs(mol_calc), canonical=True)
    else:
        mol_calc = mol_from_smiles(s)
        if mol_calc is None:
            raise ValueError("Invalid SMILES (RDKit parse failed).")
        smiles_for_thermo = s

    result = {
        "SMILES_input": s,
        "is_polymer_smiles": is_poly,
        "nmer_used": int(polymer_nmer) if is_poly else 1,
        "endcap": polymer_endcap if is_poly else "",
        "SMILES_used_for_calc": Chem.MolToSmiles(Chem.RemoveHs(mol_calc), canonical=True),
        "MolWt_used_for_calc (g/mol)": mol_weight_gmol(mol_calc),
    }

    # PubChem (small molecule only): limit sections as requested
    if use_pubchem and (not is_poly):
        cid = get_pubchem_cid_from_smiles(s)
        result["PubChem CID"] = cid
        if cid is not None:
            try:
                flat = pubchem_extract_physchem_limited(cid)
                # prefix to avoid collisions
                for k, v in flat.items():
                    result[k] = v
            except Exception as e:
                result["PubChem error"] = str(e)
        else:
            result["PubChem note"] = "CID not found."
    else:
        result["PubChem CID"] = None
        result["PubChem note"] = "Skipped (polymer dummy detected or PubChem disabled)."

    # xTB
    if run_xtb:
        try:
            result.update(run_xtb_and_parse(mol_calc, charge=int(charge), uhf=int(uhf)))
        except Exception as e:
            result.update({
                "HOMO (eV, xTB)": None,
                "LUMO (eV, xTB)": None,
                "Bandgap (eV, xTB)": None,
                "Dipole (D, xTB)": None,
                "xTB error": str(e),
            })
    else:
        result.update({
            "HOMO (eV, xTB)": None,
            "LUMO (eV, xTB)": None,
            "Bandgap (eV, xTB)": None,
            "Dipole (D, xTB)": None,
            "xTB note": "Skipped (run_xtb=False)."
        })

    # RDKit proxy
    try:
        result["Polarizability proxy: MolMR (RDKit)"] = float(Crippen.MolMR(Chem.RemoveHs(mol_calc)))
    except Exception:
        result["Polarizability proxy: MolMR (RDKit)"] = None

    # Thermo (pGrAdd)
    if run_pgradd:
        result.update(pgradd_thermo(smiles_for_thermo, T=float(T)))
    else:
        result.update({
            "H (kcal/mol)": None, "S (cal/mol/K)": None, "G (kcal/mol)": None, "Cp (cal/mol/K)": None,
            "pGrAdd note": "Skipped (run_pgradd=False)."
        })

    return result, mol_input, mol_calc


# =========================================================
# HSP/Polarity engine (same as your integrated version)
# =========================================================
def hsp_validate_smiles(smiles: str):
    if not smiles or len(smiles.strip()) == 0:
        return False, "SMILES를 입력하세요"

    invalid_chars = re.findall(r"[^a-zA-Z0-9\[\]\(\)=\-\+\.\@\#\%/\\\*\:]", smiles)
    if invalid_chars:
        return False, f"유효하지 않은 문자가 포함되어 있습니다: {invalid_chars}"

    paren_count = 0
    bracket_count = 0
    for char in smiles:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        if paren_count < 0 or bracket_count < 0:
            return False, "괄호 또는 대괄호가 불균형입니다"

    if paren_count != 0 or bracket_count != 0:
        return False, "괄호 또는 대괄호가 불균형입니다"

    components = smiles.split('.')
    has_valid = any(len(comp.strip()) > 1 for comp in components)
    if not has_valid:
        return False, "유효한 분자 구조가 포함되어 있지 않습니다"

    return True, None

def hansen_hvk_group_contribution(smiles: str):
    try:
        s = preprocess_input_smiles(smiles)
        s = s.replace("[*]", "C")  # for matching
        components = [c.strip() for c in s.split('.') if c.strip()]
        if not components:
            return None

        GC = {
            'CH3': (180.0, 0.0, 0.0, 33.5),
            'CH2': (133.0, 0.0, 0.0, 16.1),
            'CH':  ( 80.0, 0.0, 0.0,  9.3),
            'C':   (  0.0, 0.0, 0.0,  3.6),
            'C_ar': (110.0, 50.0, 0.0, 13.4),
            'S_ar': (130.0, 120.0, 0.0, 25.0),
            'O_ether': (100.0, 400.0, 3000.0, 10.0),
            'OH': (210.0, 500.0, 20000.0, 9.7),
            'SO2': (590.0, 1460.0, 11300.0, 32.5),
        }

        patt = {
            'SO2': Chem.MolFromSmarts('S(=O)(=O)'),
            'OH': Chem.MolFromSmarts('[OX2H]'),
            'O_ether': Chem.MolFromSmarts('[O;X2;!$([O]=S);!$([O]=P)]'),
            'S_ar': Chem.MolFromSmarts('[s]'),
            'C_ar': Chem.MolFromSmarts('[c]'),
            'CH3': Chem.MolFromSmarts('[CH3]'),
            'CH2': Chem.MolFromSmarts('[CH2]'),
            'CH': Chem.MolFromSmarts('[CH]'),
            'C': Chem.MolFromSmarts('[C;H0;X4]'),
        }

        sum_Fd = 0.0
        sum_Fp2 = 0.0
        sum_Eh = 0.0
        sum_V = 0.0

        for comp in components:
            mol = Chem.MolFromSmiles(comp)
            if mol is None:
                continue

            counts = {}
            for key, p in patt.items():
                if p is None:
                    continue
                matches = mol.GetSubstructMatches(p)
                if not matches:
                    continue
                if key in ['C_ar', 'S_ar', 'CH3', 'CH2', 'CH', 'C']:
                    atom_ids = set(m[0] for m in matches)
                    counts[key] = len(atom_ids)
                else:
                    counts[key] = len(matches)

            for g, n in counts.items():
                if g not in GC:
                    continue
                Fd_i, Fp_i, Eh_i, V_i = GC[g]
                sum_Fd += n * Fd_i
                sum_Fp2 += n * (Fp_i ** 2)
                sum_Eh += n * Eh_i
                sum_V += n * V_i

        if sum_V <= 0:
            return None

        delta_d = sum_Fd / sum_V
        delta_p = (math.sqrt(sum_Fp2) / sum_V) if sum_Fp2 > 0 else 0.0
        delta_h = math.sqrt(sum_Eh / sum_V) if sum_Eh > 0 else 0.0

        return {
            'hansen_dispersive': delta_d,
            'hansen_polar': delta_p,
            'hansen_hydrogen_bonding': delta_h,
            'hansen_molar_volume': sum_V
        }
    except Exception:
        return None

def generate_hsp_results(smiles: str):
    s = preprocess_input_smiles(smiles)
    carbon_count = s.count('C') + s.count('c')
    nitrogen_count = s.count('N') + s.count('n')
    oxygen_count = s.count('O') + s.count('o')
    sulfur_count = s.count('S') + s.count('s')

    total_mw = (carbon_count * 12 + nitrogen_count * 14 + oxygen_count * 16 + sulfur_count * 32 + (len(s) * 0.5))

    has_oxygen_nitrogen = oxygen_count > 0 or nitrogen_count > 0
    has_ion = '[' in s and ('-' in s or '+' in s)

    tpsa = (oxygen_count * 20.23 + nitrogen_count * 12.03) * 1.2
    logp = carbon_count * 0.5 - (oxygen_count + nitrogen_count) * 0.4 - 1.5

    polarity = 'Very High' if has_ion else ('High' if (has_oxygen_nitrogen or has_ion) else 'Low')

    dipole_estimate = ((oxygen_count * 1.7)**2 + (nitrogen_count * 1.0)**2 + (sulfur_count * 1.2)**2)**0.5 * 0.8
    polar_bonds = max(1, int((oxygen_count + nitrogen_count) * 1.5))
    rot_bonds = max(0, int(len(s) / 8))
    rings = int(max(0, len(re.findall(r"[0-9]", s)) / 2))

    hansen_gc = hansen_hvk_group_contribution(s)
    if hansen_gc is not None:
        hansen_d = hansen_gc['hansen_dispersive']
        hansen_p = hansen_gc['hansen_polar']
        hansen_h = hansen_gc['hansen_hydrogen_bonding']
    else:
        hansen_d = carbon_count * 1.5
        hansen_p = (oxygen_count * 8 + nitrogen_count * 6) if has_oxygen_nitrogen else 0.0
        hansen_h = (oxygen_count + nitrogen_count) * 10 if has_oxygen_nitrogen else 0.0

    return {
        'molecular_formula': f"C{carbon_count}N{nitrogen_count}O{oxygen_count}S{sulfur_count}...",
        'molecular_weight': f"{total_mw:.2f}",
        'tpsa': f"{tpsa:.2f}",
        'logp': f"{logp:.2f}",
        'dipole_moment': f"{dipole_estimate:.2f}",
        'polarity': polarity,
        'polar_bonds': polar_bonds,
        'rot_bonds': rot_bonds,
        'rings': rings,
        'hansen_dispersive': f"{hansen_d:.2f}",
        'hansen_polar': f"{hansen_p:.2f}",
        'hansen_hydrogen_bonding': f"{hansen_h:.2f}",
    }

def rdkit_mols_from_component_smiles_for_display(smiles: str):
    s = preprocess_input_smiles(smiles)
    components = [c.strip() for c in s.split('.') if c.strip()]
    mols = []
    for comp in components:
        comp_clean = comp.replace('[*]', 'C')
        try:
            mol = Chem.MolFromSmiles(comp_clean)
        except Exception:
            mol = None
        if mol is not None:
            mols.append(mol)
    return mols


# =========================================================
# Unified GUI (Batch -> HSP + PhysChem -> merged CSV)
# =========================================================
class UnifiedSMILESGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified SMILES GUI (PhysChem + HSP) | PubChem physchem limited | xTB cp949 fix")
        self.resize(1750, 960)

        self.hsp_batch = []
        self.physchem_batch = []
        self.df_hsp = None
        self.df_physchem = None
        self.df_merged = None

        left_col = QVBoxLayout()

        batch_box = QGroupBox("Batch input")
        batch_form = QGridLayout()

        self.batch_smiles = QTextEdit()
        self.batch_smiles.setPlaceholderText("One SMILES per line")
        self.batch_smiles.setText(
            "CC(C)C\n"
            "c1ccccc1\n"
            "CC1=C2OCCOC2=C(C)S1.CCC(C3=CC=CC=C3)C.C[SO3-]\n"
            "CCCCCC1=C(-*)SC(-*)=C1"
        )

        self.batch_memo = QTextEdit()
        self.batch_memo.setPlaceholderText("One memo per line (optional)")
        self.batch_memo.setText("프로판\n벤젠\nPEDOT:PSS (sulfonate)\nP3HT (repeat unit)")

        batch_form.addWidget(QLabel("SMILES list"), 0, 0)
        batch_form.addWidget(self.batch_smiles, 1, 0)
        batch_form.addWidget(QLabel("Memo list"), 2, 0)
        batch_form.addWidget(self.batch_memo, 3, 0)
        batch_box.setLayout(batch_form)

        opt_box = QGroupBox("Options")
        opt_form = QFormLayout()

        self.charge_spin = QSpinBox(); self.charge_spin.setRange(-10, 10)
        self.uhf_spin = QSpinBox(); self.uhf_spin.setRange(0, 20)

        self.temp_spin = QSpinBox()
        self.temp_spin.setRange(50, 2000)
        self.temp_spin.setValue(298)

        self.use_pubchem_chk = QCheckBox("Use PubChem (Chemical&Physical Properties + Physical Description only)")
        self.use_pubchem_chk.setChecked(True)

        self.run_xtb_chk = QCheckBox("Run xTB (slow)")
        self.run_xtb_chk.setChecked(True)

        self.run_pgradd_chk = QCheckBox("Run pGrAdd thermo (optional)")
        self.run_pgradd_chk.setChecked(True)

        self.poly_cap_combo = QComboBox()
        self.poly_cap_combo.addItems(["H", "Me"])

        self.poly_n_spin = QSpinBox()
        self.poly_n_spin.setRange(1, 5000)
        self.poly_n_spin.setValue(3)

        self.target_mode_combo = QComboBox()
        self.target_mode_combo.addItems(["None", "Target DP", "Target Mn (g/mol)"])

        self.target_dp_spin = QSpinBox()
        self.target_dp_spin.setRange(1, 5000)
        self.target_dp_spin.setValue(50)

        self.target_mn_spin = QDoubleSpinBox()
        self.target_mn_spin.setDecimals(2)
        self.target_mn_spin.setRange(1.0, 1.0e9)
        self.target_mn_spin.setValue(10000.0)

        self.suggest_note = QLabel("Polymer: repeat unit에 [*] dummy 2개일 때 Target DP/Mn → n-mer 자동 적용(배치에서도).")
        self.suggest_note.setWordWrap(True)

        opt_form.addRow("xTB charge", self.charge_spin)
        opt_form.addRow("xTB UHF", self.uhf_spin)
        opt_form.addRow("Thermo T (K)", self.temp_spin)
        opt_form.addRow("", self.use_pubchem_chk)
        opt_form.addRow("", self.run_xtb_chk)
        opt_form.addRow("", self.run_pgradd_chk)
        opt_form.addRow("Polymer end-cap", self.poly_cap_combo)
        opt_form.addRow("Default polymer n-mer", self.poly_n_spin)
        opt_form.addRow("Polymer n-mode", self.target_mode_combo)
        opt_form.addRow("Target DP", self.target_dp_spin)
        opt_form.addRow("Target Mn (g/mol)", self.target_mn_spin)
        opt_form.addRow("", self.suggest_note)
        opt_box.setLayout(opt_form)

        btn_box = QGroupBox("Actions")
        btn_row = QHBoxLayout()

        self.btn_run_both = QPushButton("Analyze batch (HSP + PhysChem)")
        self.btn_run_both.clicked.connect(self.on_run_batch_both)

        self.btn_export_merged = QPushButton("Export integrated CSV (merge by Index)")
        self.btn_export_merged.clicked.connect(self.on_export_integrated_csv)

        self.btn_export_images = QPushButton("Export structures PNGs")
        self.btn_export_images.clicked.connect(self.on_export_structure_pngs)

        self.btn_clear = QPushButton("Clear results")
        self.btn_clear.clicked.connect(self.on_clear_results)

        btn_row.addWidget(self.btn_run_both)
        btn_row.addWidget(self.btn_export_merged)
        btn_row.addWidget(self.btn_export_images)
        btn_row.addWidget(self.btn_clear)
        btn_box.setLayout(btn_row)

        left_col.addWidget(batch_box)
        left_col.addWidget(opt_box)
        left_col.addWidget(btn_box)

        left_widget = QWidget()
        left_widget.setLayout(left_col)

        mid = QVBoxLayout()
        self.list_results = QListWidget()
        self.list_results.currentRowChanged.connect(self.on_select_result)
        mid.addWidget(QLabel("Results (select to preview)"))
        mid.addWidget(self.list_results)
        mid_widget = QWidget()
        mid_widget.setLayout(mid)

        right = QVBoxLayout()
        self.img_preview = AspectRatioImageLabel("Preview")
        self.img_preview.setMinimumSize(520, 360)

        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)

        self.table_preview = QTableWidget(0, 2)
        self.table_preview.setHorizontalHeaderLabels(["Key", "Value"])
        self.table_preview.horizontalHeader().setStretchLastSection(True)

        right.addWidget(QLabel("2D structure preview"))
        right.addWidget(self.img_preview)
        right.addWidget(QLabel("Text details"))
        right.addWidget(self.detail, 2)
        right.addWidget(QLabel("Merged row preview (일부만 표시; PubChem는 CSV에서 전체 확인 권장)"))
        right.addWidget(self.table_preview, 2)

        right_widget = QWidget()
        right_widget.setLayout(right)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left_widget)
        split.addWidget(mid_widget)
        split.addWidget(right_widget)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setStretchFactor(2, 3)

        main = QVBoxLayout()
        main.addWidget(split)
        self.setLayout(main)

        self.target_mode_combo.currentIndexChanged.connect(self._on_target_mode_changed)
        self._on_target_mode_changed()

    def _on_target_mode_changed(self):
        mode = self.target_mode_combo.currentText()
        self.target_dp_spin.setEnabled(mode == "Target DP")
        self.target_mn_spin.setEnabled(mode == "Target Mn (g/mol)")

    def _parse_batch_inputs(self):
        smiles_lines = [s.strip() for s in self.batch_smiles.toPlainText().splitlines() if s.strip()]
        memo_lines = [m.strip() for m in self.batch_memo.toPlainText().splitlines()]
        return smiles_lines, memo_lines

    def _polymer_n_for_smiles(self, smiles: str):
        s = preprocess_input_smiles(smiles)
        if "[*]" not in s:
            return None, None

        unit = Chem.MolFromSmiles(s)
        if unit is None or len(_dummy_indices(unit)) != 2:
            raise ValueError("Polymer SMILES는 repeat unit에 [*] dummy가 정확히 2개 있어야 합니다.")

        cap = self.poly_cap_combo.currentText()
        mode = self.target_mode_combo.currentText()

        if mode == "Target DP":
            n = suggest_n_from_target_dp(self.target_dp_spin.value(), n_max=self.poly_n_spin.maximum())
            return int(n), cap

        if mode == "Target Mn (g/mol)":
            target_mn = float(self.target_mn_spin.value())
            n, _, _, _ = suggest_n_from_target_mn(s, target_mn=target_mn, cap=cap, n_max=500)
            return int(n), cap

        return int(self.poly_n_spin.value()), cap

    def _set_preview_image(self, smiles: str):
        mols = rdkit_mols_from_component_smiles_for_display(smiles)
        if not mols:
            self.img_preview.setText("Cannot draw structure")
            self.img_preview._pix = None
            return
        legends = [f"Component {i+1}" for i in range(len(mols))]
        png = rdkit_png_from_mols(mols, legends=legends, subImgSize=(420, 340), molsPerRow=3)
        self.img_preview.setPixmapKeep(pixmap_from_png_bytes(png))

    def _fill_kv_table(self, d: dict):
        items = list(d.items())
        self.table_preview.setRowCount(len(items))
        for r, (k, v) in enumerate(items):
            self.table_preview.setItem(r, 0, QTableWidgetItem(str(k)))
            self.table_preview.setItem(r, 1, QTableWidgetItem("" if v is None else str(v)))

    def on_run_batch_both(self):
        smiles_list, memo_list = self._parse_batch_inputs()
        if not smiles_list:
            QMessageBox.critical(self, "Input error", "SMILES list is empty.")
            return

        self.hsp_batch = []
        self.physchem_batch = []
        self.list_results.clear()
        self.df_hsp = None
        self.df_physchem = None
        self.df_merged = None

        use_pubchem = self.use_pubchem_chk.isChecked()
        run_xtb_flag = self.run_xtb_chk.isChecked()
        run_pgradd_flag = self.run_pgradd_chk.isChecked()
        charge = int(self.charge_spin.value())
        uhf = int(self.uhf_spin.value())
        T = float(self.temp_spin.value())

        n_ok = 0
        errors = 0

        for idx, smi_raw in enumerate(smiles_list, start=1):
            memo = memo_list[idx - 1] if (idx - 1) < len(memo_list) else ""
            smi = preprocess_input_smiles(smi_raw)
            row_key = {"Index": idx, "Memo": memo, "SMILES_raw": smi_raw, "SMILES_preprocessed": smi}

            # HSP
            valid, err = hsp_validate_smiles(smi_raw)
            if valid:
                hsp = generate_hsp_results(smi_raw)
                hsp_row = dict(row_key)
                hsp_row.update({f"HSP_{k}": v for k, v in hsp.items()})
                hsp_row["HSP_status"] = "OK"
            else:
                hsp_row = dict(row_key)
                hsp_row["HSP_status"] = f"ERROR: {err}"
                errors += 1

            # PhysChem
            try:
                if "[*]" in smi:
                    nmer, cap = self._polymer_n_for_smiles(smi)
                else:
                    nmer, cap = 1, ""

                phys, _, _ = compute_all(
                    smiles_input=smi,
                    polymer_nmer=int(nmer) if nmer else int(self.poly_n_spin.value()),
                    polymer_endcap=(cap if cap else self.poly_cap_combo.currentText()),
                    charge=charge,
                    uhf=uhf,
                    T=T,
                    use_pubchem=use_pubchem,
                    run_xtb=run_xtb_flag,
                    run_pgradd=run_pgradd_flag,
                )

                phys_row = dict(row_key)
                phys_row.update({f"PhysChem_{k}": v for k, v in phys.items()})
                phys_row["PhysChem_status"] = "OK"
            except Exception as e:
                phys_row = dict(row_key)
                phys_row["PhysChem_status"] = f"ERROR: {str(e)}"
                errors += 1

            self.hsp_batch.append(hsp_row)
            self.physchem_batch.append(phys_row)

            label = f"#{idx}"
            if memo:
                label += f" - {memo}"
            label += f" | HSP: {hsp_row.get('HSP_status','')}"
            label += f" | PhysChem: {phys_row.get('PhysChem_status','')}"
            self.list_results.addItem(QListWidgetItem(label))

            if (hsp_row.get("HSP_status") == "OK") and (phys_row.get("PhysChem_status") == "OK"):
                n_ok += 1

        self.df_hsp = pd.DataFrame(self.hsp_batch)
        self.df_physchem = pd.DataFrame(self.physchem_batch)
        self.df_merged = pd.merge(self.df_physchem, self.df_hsp, on=["Index", "Memo", "SMILES_raw", "SMILES_preprocessed"], how="outer")

        if self.list_results.count() > 0:
            self.list_results.setCurrentRow(0)

        QMessageBox.information(self, "Done", f"Batch finished. OK(both): {n_ok}/{len(smiles_list)} | Errors: {errors}")

    def on_select_result(self, row: int):
        if row < 0:
            return
        if self.df_merged is None or row >= len(self.df_merged):
            return

        rec = self.df_merged.iloc[row].to_dict()
        smi = rec.get("SMILES_preprocessed", "") or rec.get("SMILES_raw", "")

        if smi:
            self._set_preview_image(smi)

        txt_lines = [
            f"Index: {rec.get('Index')}",
            f"Memo: {rec.get('Memo')}",
            f"SMILES_raw: {rec.get('SMILES_raw')}",
            f"SMILES_preprocessed: {rec.get('SMILES_preprocessed')}",
            "",
            f"HSP_status: {rec.get('HSP_status')}",
            f"PhysChem_status: {rec.get('PhysChem_status')}",
            "",
            "PubChem key는 'PhysChem_PubChem|...' 형태로 CSV에 다 들어갑니다."
        ]
        self.detail.setPlainText("\n".join(txt_lines))

        # preview subset + density/boiling/melting related keys
        kv = {}
        base_keys = [
            "Index", "Memo", "SMILES_raw", "SMILES_preprocessed", "HSP_status", "PhysChem_status",
            "HSP_molecular_weight", "HSP_tpsa", "HSP_logp", "HSP_polarity",
            "PhysChem_MolWt_used_for_calc (g/mol)", "PhysChem_HOMO (eV, xTB)",
            "PhysChem_LUMO (eV, xTB)", "PhysChem_Bandgap (eV, xTB)",
            "PhysChem_Dipole (D, xTB)", "PhysChem_PubChem CID"
        ]
        for k in base_keys:
            if k in rec:
                kv[k] = rec[k]

        for k in rec.keys():
            if not str(k).startswith("PhysChem_PubChem|"):
                continue
            kl = str(k).lower()
            if ("density" in kl) or ("boiling" in kl) or ("melting" in kl) or ("solubility" in kl) or ("vapor" in kl):
                kv[k] = rec[k]

        self._fill_kv_table(kv)

    def on_export_integrated_csv(self):
        if self.df_merged is None or self.df_merged.empty:
            QMessageBox.information(self, "Info", "먼저 Analyze batch를 실행해서 결과를 만든 뒤 Export 하세요.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"integrated_pubchem_physchem_limited_{ts}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save integrated CSV", default, "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.df_merged.to_csv(path, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "Saved", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def on_export_structure_pngs(self):
        if self.df_merged is None or self.df_merged.empty:
            QMessageBox.information(self, "Info", "먼저 Analyze batch를 실행하세요.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select folder to save PNGs")
        if not folder:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(folder, f"structures_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        for _, rec in self.df_merged.iterrows():
            idx = int(rec.get("Index", 0))
            memo = str(rec.get("Memo", "") or f"mol_{idx}")
            smi = str(rec.get("SMILES_preprocessed", "") or rec.get("SMILES_raw", "") or "")
            safe_memo = "".join(c for c in memo if c.isalnum() or c in "._- ").rstrip()
            fp = os.path.join(out_dir, f"{idx:03d}_{safe_memo}.png")

            if not smi:
                continue
            try:
                mols = rdkit_mols_from_component_smiles_for_display(smi)
                if not mols:
                    continue
                png = rdkit_png_from_mols(
                    mols,
                    legends=[f"Component {i+1}" for i in range(len(mols))],
                    subImgSize=(360, 300),
                    molsPerRow=3
                )
                if png:
                    with open(fp, "wb") as f:
                        f.write(png)
            except Exception:
                pass

        QMessageBox.information(self, "Saved", f"Saved PNGs:\n{out_dir}")

    def on_clear_results(self):
        self.hsp_batch = []
        self.physchem_batch = []
        self.df_hsp = None
        self.df_physchem = None
        self.df_merged = None
        self.list_results.clear()
        self.detail.setPlainText("")
        self.table_preview.setRowCount(0)
        self.img_preview.setText("Preview")
        self.img_preview._pix = None


if __name__ == "__main__":
    app = QApplication([])
    w = UnifiedSMILESGUI()
    w.show()
    app.exec()
