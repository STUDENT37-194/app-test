# -*- coding: utf-8 -*-
"""
Version Cloud Run :
- Reçoit le HTML du mail Looker via API POST
- Extrait la table Looker
- Analyse PJI / PSR / Programme
- Charge RefPSR.csv local
- Retourne JSON
"""

import numpy as np
import pandas as pd
import io
import re
from typing import List, Tuple, Dict

from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# CONFIG
# =========================
REF_PSR_CSV = "RefPSR.csv"
RAYON_BOULE_MM = 20
SEUIL_PSR_PROXIMITE = 2

app = FastAPI()

# =========================
# UTILITAIRES PARSING LOOKER
# =========================
def normalize_cols(cols: List[str]) -> List[str]:
    return [str(c).strip().lower().replace(" ", "_") for c in cols]


def pick_looker_table(html: str) -> pd.DataFrame:
    dfs = pd.read_html(io.StringIO(html))
    target = None
    for d in dfs:
        norm = normalize_cols(d.columns)
        if {"label_robot", "brique", "alerte_description"}.issubset(set(norm)) \
        or {"label", "brique", "alerte_description"}.issubset(set(norm)):
            target = d
            break

    if target is None:
        target = max(dfs, key=lambda x: x.shape[0])

    target.columns = normalize_cols(target.columns)

    if "label_robot" in target.columns and "label" not in target.columns:
        target = target.rename(columns={"label_robot": "label"})

    return target


TRIPLE_RE = re.compile(r'(\d{5,})\s*/\s*([A-Za-z0-9 _-]+)\s*/\s*(\d{2,})')


def extract_all_triples_looker(desc: str) -> List[dict]:
    if not isinstance(desc, str):
        return []
    triples = []
    for pji_str, psr_str, prog_str in TRIPLE_RE.findall(desc):
        pji_num = pd.to_numeric(pji_str, errors="coerce")
        psr_num = pd.to_numeric(psr_str, errors="coerce")
        prog_num = pd.to_numeric(prog_str, errors="coerce")
        triples.append({
            "pji_str": pji_str,
            "psr_id_str": psr_str.strip(),
            "pji": pji_num,
            "psr_id": psr_num,
            "programme": int(prog_num) if pd.notna(prog_num) else None,
        })
    return triples


def build_df_final_from_looker_table(df_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    skipped = []

    for col_needed in ["label", "brique", "alerte_description"]:
        if col_needed not in df_table.columns:
            df_table[col_needed] = None

    for _, r in df_table.iterrows():
        label = r.get("label")
        brique = r.get("brique")
        desc = r.get("alerte_description", "")

        triples = extract_all_triples_looker(desc)
        if triples:
            for t in triples:
                rows.append({
                    "label": label,
                    "brique": brique,
                    "alerte_description": desc,
                    "pji_str": t["pji_str"],
                    "psr_id_str": t["psr_id_str"],
                    "pji": t["pji"],
                    "psr_id": t["psr_id"],
                    "programme": t["programme"],
                })
        else:
            skipped.append({
                "label": label,
                "brique": brique,
                "alerte_description": desc
            })

    return pd.DataFrame(rows), pd.DataFrame(skipped)


# =========================
# CHARGEMENT COORDONNÉES
# =========================
def load_ref_psr(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=';', encoding='latin1')

    for col in ["X_Linx", "Y_Linx", "Z_Linx"]:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Spotname"] = pd.to_numeric(df["Spotname"], errors="coerce")

    for opt_col in ["Prog", "Timername"]:
        if opt_col not in df.columns:
            df[opt_col] = np.nan

    return df


# =========================
# ANALYSES
# =========================
def calculer_distance_3d(x1, y1, z1, x2, y2, z2) -> float:
    return float(np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2))


def verifier_proximite_spatiale(df_coords_calcul: pd.DataFrame,
                                rayon_mm: float,
                                seuil_psr: int):
    psr_list = df_coords_calcul.to_dict("records")
    psr_proches = set()
    distances = {}

    if len(psr_list) < 2:
        return False, [], {}

    for i in range(len(psr_list)):
        for j in range(i + 1, len(psr_list)):
            d = calculer_distance_3d(
                psr_list[i]['X_Linx'], psr_list[i]['Y_Linx'], psr_list[i]['Z_Linx'],
                psr_list[j]['X_Linx'], psr_list[j]['Y_Linx'], psr_list[j]['Z_Linx']
            )
            paire = f"{int(psr_list[i]['Spot Name'])} <-> {int(psr_list[j]['Spot Name'])}"
            distances[paire] = round(d, 2)
            if d <= rayon_mm:
                psr_proches.add(psr_list[i]['Spot Name'])
                psr_proches.add(psr_list[j]['Spot Name'])

    alerte = len(psr_proches) >= seuil_psr
    return alerte, sorted(list(psr_proches)), distances


def verifier_sequences_consecutives_detail(prog_nos: List[int]):
    if len(prog_nos) < 2:
        return False, []

    prog_nos_sorted = sorted(prog_nos)
    groupes_consecutifs = []
    groupe_courant = [prog_nos_sorted[0]]

    for i in range(1, len(prog_nos_sorted)):
        if prog_nos_sorted[i] - prog_nos_sorted[i-1] == 1:
            groupe_courant.append(prog_nos_sorted[i])
        else:
            if len(groupe_courant) >= 2:
                groupes_consecutifs.append(groupe_courant)
            groupe_courant = [prog_nos_sorted[i]]

    if len(groupe_courant) >= 2:
        groupes_consecutifs.append(groupe_courant)

    return len(groupes_consecutifs) > 0, groupes_consecutifs


# =========================
# PIPELINE
# =========================
def analyser_derive_process(df_final: pd.DataFrame, df_coords: pd.DataFrame):
    df_dashboard = df_final[df_final["brique"].astype(str).str.contains("derive", case=False, na=False)].copy()
    if df_dashboard.empty:
        return "PAS DE CONTROLE US", None

    df_dashboard["pji"] = pd.to_numeric(df_dashboard["pji"], errors="coerce")
    df_dashboard["psr_id"] = pd.to_numeric(df_dashboard["psr_id"], errors="coerce")
    df_dashboard["programme"] = pd.to_numeric(df_dashboard["programme"], errors="coerce")

    alertes = []

    for pji, groupe in df_dashboard.groupby("pji"):
        pji_strs = groupe["pji_str"].dropna().astype(str).unique().tolist()
        pji_label = pji_strs[0] if pji_strs else str(int(pji))

        psr_non_vides = []
        for psr in groupe["psr_id"].dropna().unique().tolist():
            try:
                psr_non_vides.append(float(psr))
            except:
                pass

        match_par_prog = pd.merge(
            groupe,
            df_coords[["Prog", "Timername", "Spotname", "X_Linx", "Y_Linx", "Z_Linx"]],
            left_on=["programme", "label"],
            right_on=["Prog", "Timername"],
            how="left"
        )

        spots_via_prog = [
            float(s) for s in match_par_prog["Spotname"].dropna().unique().tolist()
        ]

        psr_non_vides = sorted(list(set(psr_non_vides) | set(spots_via_prog)))

        if not psr_non_vides:
            continue

        df_coords_filtre = (
            df_coords[df_coords["Spotname"].isin(psr_non_vides)]
            .drop_duplicates(subset="Spotname")
            .dropna(subset=["X_Linx", "Y_Linx", "Z_Linx"])
        )

        if len(df_coords_filtre) >= 2:
            df_calcul = df_coords_filtre.rename(columns={"Spotname": "Spot Name"})
            alerte_geo, psr_proches, distances = verifier_proximite_spatiale(
                df_calcul, RAYON_BOULE_MM, SEUIL_PSR_PROXIMITE
            )
            if alerte_geo:
                alertes.append({
                    "PJI": pji_label,
                    "Type": "Proximité spatiale",
                    "PSR_proches": psr_proches,
                    "Distances": distances
                })
        else:
            prog_nos = sorted(groupe["programme"].dropna().astype(int).unique().tolist())
            if len(prog_nos) < 2:
                continue
            alerte_seq, groups = verifier_sequences_consecutives_detail(prog_nos)
            if alerte_seq:
                alertes.append({
                    "PJI": pji_label,
                    "Type": "Séquences consécutives",
                    "Sequences": groups
                })

    if alertes:
        return "CONTROLE US INDISPENSABLE", alertes

    return "PAS DE CONTROLE US", None


# =========================
# API FASTAPI
# =========================
class RequestModel(BaseModel):
    html: str


@app.post("/analyse")
def analyse(data: RequestModel):
    try:
        df_table = pick_looker_table(data.html)
        df_final, _ = build_df_final_from_looker_table(df_table)
        df_coords = load_ref_psr(REF_PSR_CSV)

        decision, details = analyser_derive_process(df_final, df_coords)

        return {
            "decision": decision,
            "details": details
        }

    except Exception as e:
        return {"error": str(e)}