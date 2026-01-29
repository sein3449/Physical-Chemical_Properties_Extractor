# GUI_PhysChemProp_SMILES.py
import os
import re
import json
import tempfile
import subprocess
from datetime import datetime

import pandas as pd
import requests
import pubchempy as pcp

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Crippen

HARTREE_TO_EV = 27.211386245988

# =========================================================
# 0) Streamlit bootstrap guard (python 파일.py 실행도 허용)
# =========================================================
def _ensure_streamlit_bootstrap():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    if ctx is None and __name__ == "__main__":
        try:
            from streamlit.web import bootstrap
            bootstrap.run(__file__, False, [], {})
        except Exception:
            from streamlit.web import cli as stcli
            import sys
            sys.argv = ["streamlit", "run", __file__]
            raise SystemExit(stcli.main())
        raise SystemExit


# =========================================================
# 1) SMILES 전처리 / polymer 판별
# =========================================================
def normalize_smiles(smiles: str) -> str:
    """
    - 공백 제거
    - '*' -> '[*]'로 변환(이미 '[*]'면 그대로)
    """
    s = (smiles or "").strip().replace(" ", "")
    token = "__DUMMY__"
    s = s.replace("[*]", token)
    s = s.replace("*", "[*]")
    s = s.replace(token, "[*]")
    return s

def has_dummy_atoms(smiles: str) -> bool:
    return "[*]" in normalize_smiles(smiles)

def mol_from_smiles(smiles: str):
    s = normalize_smiles(smiles)
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.AddHs(mol)

def safe_sanitize(mol: Chem.Mol) -> Chem.Mol:
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        # Kekulize가 깨지는 경우를 피하기 위한 fallback
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
            return mol
        except Exception:
            return mol


# =========================================================
# 2) Polymer repeat unit (2 dummy) -> oligomer 생성
# =========================================================
def _dummy_indices(mol: Chem.Mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]

def _dummy_neighbor(mol: Chem.Mol, dummy_idx: int) -> int:
    a = mol.GetAtomWithIdx(dummy_idx)
    nbs = [n.GetIdx() for n in a.GetNeighbors()]
    if len(nbs) != 1:
        raise ValueError("Dummy atom must have exactly one neighbor.")
    return nbs[0]

def build_oligomer_from_repeat_unit(smiles_with_two_dummies: str, n: int = 3, cap: str = "H") -> Chem.Mol:
    """
    repeat unit SMILES에 dummy atom [*]가 정확히 2개 있다고 가정.
    - n-mer로 연결: (tail of chain dummy) -- (head of new unit dummy)
    - 내부 dummy 2개는 제거
    - 마지막에 양 끝 dummy를 cap 처리
      cap="H": dummy만 제거 (RDKit이 implicit H로 처리)
      cap="Me": 양 끝에 -CH3 부착 후 dummy 제거
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    unit0 = Chem.MolFromSmiles(normalize_smiles(smiles_with_two_dummies))
    if unit0 is None:
        raise ValueError("Invalid repeat-unit SMILES for RDKit.")

    d0 = _dummy_indices(unit0)
    if len(d0) != 2:
        raise ValueError("Repeat unit must contain exactly two dummy atoms ([*]).")

    # head/tail을 “인덱스 작은 쪽=head, 큰 쪽=tail”로 정의(일관성)
    head_dummy_u, tail_dummy_u = sorted(d0)

    chain = Chem.Mol(unit0)

    for _ in range(n - 1):
        # 현재 chain은 루프마다 dummy 2개만 남아있게 설계
        old_n = chain.GetNumAtoms()
        combined = Chem.CombineMols(chain, unit0)
        rw = Chem.RWMol(combined)

        # old part dummies (should be 2)
        old_dummies = [i for i in range(old_n) if rw.GetAtomWithIdx(i).GetAtomicNum() == 0]
        if len(old_dummies) != 2:
            raise RuntimeError("Internal error: chain should have exactly two dummies at each step.")
        tail_dummy_old = max(old_dummies)
        tail_nb_old = _dummy_neighbor(rw, tail_dummy_old)

        # new part dummies (exactly 2)
        new_dummies = [i for i in range(old_n, rw.GetNumAtoms()) if rw.GetAtomWithIdx(i).GetAtomicNum() == 0]
        if len(new_dummies) != 2:
            raise RuntimeError("Internal error: unit should contribute exactly two dummies.")
        # new head dummy corresponds to unit's head_dummy_u (offset 적용)
        head_dummy_new = head_dummy_u + old_n
        if head_dummy_new not in new_dummies:
            # fallback: 그냥 작은 인덱스를 head로
            head_dummy_new = min(new_dummies)
        head_nb_new = _dummy_neighbor(rw, head_dummy_new)

        # connect neighbors (polymerization bond)
        rw.AddBond(tail_nb_old, head_nb_new, Chem.rdchem.BondType.SINGLE)

        # remove the two used dummy atoms (remove higher index first)
        for idx in sorted([head_dummy_new, tail_dummy_old], reverse=True):
            rw.RemoveAtom(idx)

        chain = safe_sanitize(rw.GetMol())

    # 끝단 cap 처리
    rw = Chem.RWMol(chain)
    dummies = _dummy_indices(rw)
    if len(dummies) != 2:
        # n=1이라도 dummy 2개여야 정상
        raise RuntimeError("Internal error: oligomer should still have two terminal dummies before capping.")

    if cap.upper() == "H":
        for idx in sorted(dummies, reverse=True):
            rw.RemoveAtom(idx)
        mol = safe_sanitize(rw.GetMol())
        return Chem.AddHs(mol)

    if cap.upper() in ["ME", "CH3", "METHYL"]:
        # dummy idx는 제거하면서 index shift가 생기므로 큰 idx부터 처리
        for d_idx in sorted(dummies, reverse=True):
            nb = _dummy_neighbor(rw, d_idx)
            c_idx = rw.AddAtom(Chem.Atom(6))  # carbon
            rw.AddBond(nb, c_idx, Chem.rdchem.BondType.SINGLE)
            rw.RemoveAtom(d_idx)
        mol = safe_sanitize(rw.GetMol())
        return Chem.AddHs(mol)

    raise ValueError("cap must be 'H' or 'Me'.")


def rdkit_mol_image(mol, size=(420, 280)):
    m2 = Chem.Mol(mol)
    try:
        AllChem.Compute2DCoords(m2)
    except Exception:
        pass
    return Draw.MolToImage(m2, size=size)


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


# =========================================================
# 3) PubChem (실험 물성) - polymer/dummy면 스킵 + 에러 캐치
# =========================================================
def get_pubchem_cid_from_smiles(smiles: str):
    # dummy 포함이면 PubChem 표준화 실패 가능성이 매우 크므로 스킵 [web:193]
    if has_dummy_atoms(smiles):
        return None

    try:
        comps = pcp.get_compounds(smiles, namespace="smiles")
        if not comps:
            return None
        return comps[0].cid
    except Exception:
        return None


def _find_values_in_pugview(node, target_headings):
    found = {}

    def walk(x):
        if isinstance(x, dict):
            toc = x.get("TOCHeading")
            if toc in target_headings:
                texts = []
                info = x.get("Information", [])
                if isinstance(info, list):
                    for it in info:
                        val = it.get("Value", {})
                        if isinstance(val, dict):
                            if "StringWithMarkup" in val and isinstance(val["StringWithMarkup"], list):
                                for swm in val["StringWithMarkup"]:
                                    s = swm.get("String")
                                    if s:
                                        texts.append(s)
                            if "String" in val and isinstance(val["String"], str):
                                texts.append(val["String"])
                if texts:
                    found[toc] = texts
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(node)
    return found


def pubchem_experimental_properties(cid: int):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    targets = ["Boiling Point", "Melting Point", "Solubility", "Density", "Surface Tension"]
    found = _find_values_in_pugview(data, targets)

    def first_or_none(k):
        return found.get(k, [None])[0]

    return {
        "끓는점 (PubChem)": first_or_none("Boiling Point"),
        "녹는점 (PubChem)": first_or_none("Melting Point"),
        "용해도 (PubChem)": first_or_none("Solubility"),
        "밀도 (PubChem)": first_or_none("Density"),
        "표면장력 (PubChem)": first_or_none("Surface Tension"),
    }


# =========================================================
# 4) xTB (HOMO/LUMO/Bandgap, dipole) via CLI + JSON
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
        xyz_path = os.path.join(td, "mol.xyz")
        rdkit_3d_xyz(mol_for_calc, xyz_path)

        inp_path = os.path.join(td, "xtb.inp")
        with open(inp_path, "w", encoding="utf-8") as f:
            f.write("$write\njson=true\n$end\n")

        cmd = [
            "xtb", "mol.xyz",
            "--gfn", "2",
            "--chrg", str(charge),
            "--uhf", str(uhf),
            "--input", "xtb.inp",
            "--json",
        ]
        p = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=900)
        out_text = (p.stdout or "") + "\n" + (p.stderr or "")

        if p.returncode != 0:
            raise RuntimeError(f"xTB failed.\n{out_text}")

        xtb_json = None
        for fn in ["xtbout.json", "xtb.json"]:
            fp = os.path.join(td, fn)
            if os.path.exists(fp):
                xtb_json = fp
                break
        if xtb_json is None:
            raise RuntimeError(f"xTB succeeded but JSON not found.\n{out_text}")

        with open(xtb_json, "r", encoding="utf-8") as f:
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
            dipole_debye = float(m.group(1))

        return {
            "HOMO (eV, xTB)": homo_ev,
            "LUMO (eV, xTB)": lumo_ev,
            "Bandgap (eV, xTB)": gap_ev,
            "쌍극자 모멘트 (D, xTB)": dipole_debye,
        }


# =========================================================
# 5) pGrAdd thermo (H,S,G,Cp) - 실패 가능성 높아서 try/except
# =========================================================
def pgradd_thermo(smiles: str, T: float = 298.15):
    try:
        from pgradd.GroupAdd.Library import GroupLibrary
    except Exception:
        return {"H (kcal/mol)": None, "S (cal/mol/K)": None, "G (kcal/mol)": None, "Cp (cal/mol/K)": None}

    try:
        lib = GroupLibrary.Load("BensonGA")
        desc = lib.GetDescriptors(smiles)
        therm = lib.Estimate(desc, "thermochem")
    except Exception:
        return {"H (kcal/mol)": None, "S (cal/mol/K)": None, "G (kcal/mol)": None, "Cp (cal/mol/K)": None}

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
# 6) Streamlit app
# =========================================================
def app():
    import streamlit as st

    st.set_page_config(page_title="SMILES PhysChem", layout="wide")
    st.title("SMILES → 구조 + 물리화학 특성 계산기")

    st.caption("주의: [*]가 들어간 polymer/repeat-unit SMILES는 PubChem 조회가 실패할 수 있어 자동으로 스킵됩니다.")

    smiles_in = st.text_input("SMILES 입력", value="CCO")
    smiles_in = normalize_smiles(smiles_in)

    colA, colB = st.columns([1, 2], gap="large")

    is_poly = has_dummy_atoms(smiles_in)

    with colA:
        st.write("입력 SMILES(정규화):")
        st.code(smiles_in, language="text")

        if smiles_in:
            mol_show = Chem.MolFromSmiles(smiles_in)
            if mol_show is None:
                st.error("유효하지 않은 SMILES입니다. (RDKit 파싱 실패)")
            else:
                st.image(rdkit_mol_image(mol_show), caption="입력 구조 (dummy 포함 가능)")

    with colB:
        st.subheader("계산 옵션")

        if is_poly:
            st.info("dummy([*])가 감지되어 oligomer로 변환 후 xTB/thermo 계산을 수행합니다.")
            nmer = st.slider("Oligomer 길이 (n-mer)", min_value=1, max_value=8, value=3, step=1)
            cap = st.selectbox("End-cap", ["H", "Me"], index=0)
        else:
            nmer = 1
            cap = "H"

        charge = st.number_input("xTB charge", value=0, step=1)
        uhf = st.number_input("xTB UHF (unpaired electrons)", value=0, step=1)
        T = st.number_input("Thermo temperature (K) (pGrAdd)", min_value=50.0, max_value=2000.0,
                            value=298.15, step=1.0)

        if st.button("한꺼번에 계산", type="primary"):
            result = {
                "SMILES_input": smiles_in,
                "is_polymer_smiles": is_poly,
                "nmer_used": nmer,
                "endcap": cap,
            }

            # 6-1) 계산용 molecule 준비 (poly면 oligomer)
            mol_calc = None
            smiles_for_thermo = None  # pGrAdd에는 SMILES로
            try:
                if is_poly:
                    unit = Chem.MolFromSmiles(smiles_in)
                    if unit is None:
                        raise ValueError("Repeat-unit SMILES parse failed.")
                    d = _dummy_indices(unit)
                    if len(d) != 2:
                        raise ValueError("Polymer mode requires exactly two dummy atoms ([*]).")
                    mol_calc = build_oligomer_from_repeat_unit(smiles_in, n=nmer, cap=cap)
                    smiles_for_thermo = Chem.MolToSmiles(Chem.RemoveHs(mol_calc), canonical=True)
                else:
                    mol_calc = mol_from_smiles(smiles_in)
                    if mol_calc is None:
                        raise ValueError("SMILES parse failed.")
                    smiles_for_thermo = smiles_in
                result["SMILES_used_for_calc"] = Chem.MolToSmiles(Chem.RemoveHs(mol_calc), canonical=True)
            except Exception as e:
                st.error(f"계산용 구조 생성 실패: {e}")
                st.stop()

            # 계산용 구조 표시
            st.image(rdkit_mol_image(Chem.RemoveHs(mol_calc)), caption="계산에 사용된 구조(oligomer/capped)")

            # 6-2) PubChem (small molecule only)
            if not is_poly:
                cid = get_pubchem_cid_from_smiles(smiles_in)
                result["PubChem CID"] = cid
                if cid is not None:
                    try:
                        result.update(pubchem_experimental_properties(cid))
                    except Exception as e:
                        result.update({
                            "끓는점 (PubChem)": None,
                            "녹는점 (PubChem)": None,
                            "용해도 (PubChem)": None,
                            "밀도 (PubChem)": None,
                            "표면장력 (PubChem)": None,
                            "PubChem error": str(e),
                        })
                else:
                    result.update({
                        "끓는점 (PubChem)": None,
                        "녹는점 (PubChem)": None,
                        "용해도 (PubChem)": None,
                        "밀도 (PubChem)": None,
                        "표면장력 (PubChem)": None,
                        "PubChem CID": None,
                    })
            else:
                # polymer SMILES는 PubChem 조회 스킵
                result.update({
                    "PubChem CID": None,
                    "끓는점 (PubChem)": None,
                    "녹는점 (PubChem)": None,
                    "용해도 (PubChem)": None,
                    "밀도 (PubChem)": None,
                    "표면장력 (PubChem)": None,
                    "PubChem note": "Skipped because dummy atoms ([*]) detected.",
                })

            # 6-3) xTB
            try:
                result.update(run_xtb_and_parse(mol_calc, charge=int(charge), uhf=int(uhf)))
            except Exception as e:
                result.update({
                    "HOMO (eV, xTB)": None,
                    "LUMO (eV, xTB)": None,
                    "Bandgap (eV, xTB)": None,
                    "쌍극자 모멘트 (D, xTB)": None,
                    "xTB error": str(e),
                })

            # 6-4) Polarizability proxy (RDKit MolMR)
            try:
                m_noh = Chem.RemoveHs(mol_calc)
                result["분극률 proxy: MolMR (RDKit)"] = float(Crippen.MolMR(m_noh))
            except Exception:
                result["분극률 proxy: MolMR (RDKit)"] = None

            # 6-5) Thermo (pGrAdd) - 큰 oligomer는 실패할 수도 있음
            try:
                result.update(pgradd_thermo(smiles_for_thermo, T=float(T)))
            except Exception as e:
                result.update({
                    "H (kcal/mol)": None,
                    "S (cal/mol/K)": None,
                    "G (kcal/mol)": None,
                    "Cp (cal/mol/K)": None,
                    "pGrAdd error": str(e),
                })

            df = pd.DataFrame([result])
            st.dataframe(df, use_container_width=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = re.sub(r"[^0-9A-Za-z_\-]+", "_", smiles_in)[:40]
            default_name = f"physchem_{safe}_{ts}.csv"
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

            st.download_button("CSV 다운로드", data=csv_bytes, file_name=default_name, mime="text/csv")

            if st.button("현재 폴더에 CSV 저장"):
                with open(default_name, "wb") as f:
                    f.write(csv_bytes)
                st.success(f"저장 완료: {default_name}")


# ---- main ----
_ensure_streamlit_bootstrap()
app()
