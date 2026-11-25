"""Step 01: national household and Indigenous-person files from the 2018 census microdata (CNPV 2018).

Reads the 33 departmental files of the census (zip with CSV) from data/raw/CNPV_2018 and builds, for persons in private households
(units used as dwellings or mixed units; persons in special accommodation facilities, LEA, are excluded):
  households.parquet   one row per household: location, territory, indicators of material deprivation, household
                       size, number of Indigenous members and characteristics of the household head
  indigenous.parquet   one row per Indigenous person: household key, people (pueblo), sex and age group
  provenance.json      SHA-256 of every departmental file read and row counts

Indicators (household level, assigned to every member): no electricity; unimproved water source for cooking
(WHO/UNICEF JMP: unprotected wells and surface water; a stricter variant also counts rainwater, water trucks and water
carriers); unimproved sanitation (and a variant with latrines); cooking in a room also used for sleeping; critical
overcrowding (> 3 persons per sleeping room); dirt floor; walls of cane, waste or none. Codes as in the census data
dictionary. Households that do not prepare food are flagged: their water source for cooking is missing, and they are not
deprived of cooking space.

Usage (from the repository root):  python src/01_build_national.py
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "CNPV_2018"
OUT = ROOT / "data" / "processed"
KEY = ["COD_ENCUESTAS", "U_VIVIENDA"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_from(zf, kind):
    name = next(n for n in zf.namelist() if f"_{kind}_" in n.upper() and n.upper().endswith(".CSV"))
    return name


def department(path):
    outer = zipfile.ZipFile(path)
    inner_name = next(n for n in outer.namelist() if n.endswith("_CSV.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))

    def load(kind, cols):
        with inner.open(read_csv_from(inner, kind)) as f:
            return pd.read_csv(f, dtype=str, usecols=cols)

    viv = load("1VIV", KEY + ["U_DPTO", "U_MPIO", "UA_CLASE", "UVA1_TIPOTER", "UVA2_CODTER", "UVA_USO_UNIDAD", "VA_EE",
                              "V_TIPO_SERSA", "V_MAT_PISO", "V_MAT_PARED", "V_TIPO_VIV"])
    hog = load("2HOG", KEY + ["H_NROHOG", "H_NRO_DORMIT", "H_DONDE_PREPALIM", "H_AGUA_COCIN", "HA_TOT_PER"])
    per = load("5PER", KEY + ["P_NROHOG", "P_SEXO", "P_EDADR", "P_PARENTESCOR", "PA1_GRP_ETNIC", "PA11_COD_ETNIA",
                              "P_NIVEL_ANOSR"])
    hk = KEY + ["H_NROHOG"]
    per = per.rename(columns={"P_NROHOG": "H_NROHOG"})
    counts = {"persons_file": int(len(per))}
    # special accommodation facilities (LEA): their residents are not private households and have no household number
    lea = viv.loc[viv["UVA_USO_UNIDAD"] == "4", KEY].drop_duplicates().assign(_lea=True)
    is_lea = per[KEY].merge(lea, on=KEY, how="left")["_lea"].fillna(False).astype(bool).set_axis(per.index)
    counts["persons_lea"] = int(is_lea.sum())
    counts["indigenous_lea"] = int((is_lea & (per["PA1_GRP_ETNIC"] == "1")).sum())
    per = per[~is_lea]
    viv = viv[viv["UVA_USO_UNIDAD"] != "4"]
    counts["persons_private"] = int(len(per))
    counts["persons_private_blank_household_number"] = int(per["H_NROHOG"].isna().sum())
    # some households have a blank household number in both files; keep them (groupby would drop missing keys)
    per["H_NROHOG"] = per["H_NROHOG"].fillna("0")
    hog["H_NROHOG"] = hog["H_NROHOG"].fillna("0")
    per["indig"] = (per["PA1_GRP_ETNIC"] == "1").astype("int16")
    per["noinfo"] = (per["PA1_GRP_ETNIC"].isna() | (per["PA1_GRP_ETNIC"] == "9")).astype("int16")
    per["otheth"] = per["PA1_GRP_ETNIC"].isin(["2", "3", "4", "5"]).astype("int16")
    size = per.groupby(hk).agg(n_persons=("indig", "size"), n_indig=("indig", "sum"), n_noinfo=("noinfo", "sum"),
                               n_otheth=("otheth", "sum")).reset_index()
    head = per[per["P_PARENTESCOR"] == "1"].drop_duplicates(hk)[hk + ["P_SEXO", "P_EDADR", "PA1_GRP_ETNIC",
                                                                      "P_NIVEL_ANOSR"]]
    head = head.rename(columns={"P_SEXO": "head_sex", "P_EDADR": "head_age", "PA1_GRP_ETNIC": "head_ethnic",
                                "P_NIVEL_ANOSR": "head_educ"})
    hv = hog.merge(viv, on=KEY, how="inner")
    counts["household_records"] = int(len(hog))
    counts["household_records_without_dwelling"] = int(len(hog) - len(hv))
    h = hv.merge(size, on=hk, how="inner").merge(head, on=hk, how="left")
    counts["households_without_persons"] = int(len(hv) - len(h))
    counts["persons_not_linked"] = int(len(per) - h["n_persons"].sum())

    na = lambda s, bad: s.where(~s.isin(bad))
    out = pd.DataFrame({"dpto": h["U_DPTO"].str.zfill(2), "mpio": h["U_DPTO"].str.zfill(2) + h["U_MPIO"].str.zfill(3),
                        "clase": h["UA_CLASE"].astype("int8"),
                        "tipoter": pd.to_numeric(h["UVA1_TIPOTER"], errors="coerce").fillna(0).astype("int8"),
                        "codter": h["UVA2_CODTER"].fillna("")})
    out["elec"] = (h["VA_EE"] == "2").astype(float).where(h["VA_EE"].notna())
    w = na(h["H_AGUA_COCIN"], ["12", "99"])
    out["water"] = w.isin(["5", "7"]).astype(float).where(w.notna())
    out["water_strict"] = w.isin(["5", "6", "7", "9", "10"]).astype(float).where(w.notna())
    out["water_na_noprep"] = (h["H_AGUA_COCIN"] == "12").astype("int8")
    out["cook_na_noprep"] = (h["H_DONDE_PREPALIM"] == "6").astype("int8")
    ss = na(h["V_TIPO_SERSA"], ["9"])
    out["sanit"] = ss.isin(["3", "5", "6"]).astype(float).where(ss.notna())
    out["sanit_latrine"] = ss.isin(["3", "4", "5", "6"]).astype(float).where(ss.notna())
    c = na(h["H_DONDE_PREPALIM"], ["9"])  # 6 (food not prepared in the dwelling) is not deprived
    out["cook"] = (c == "2").astype(float).where(c.notna())
    rooms = pd.to_numeric(na(h["H_NRO_DORMIT"], ["99"]), errors="coerce")
    pers = pd.to_numeric(h["HA_TOT_PER"], errors="coerce")
    out["crowd"] = (pers / rooms > 3).astype(float).where(rooms.gt(0) & pers.notna())
    out["floor"] = (h["V_MAT_PISO"] == "6").astype(float).where(h["V_MAT_PISO"].notna())
    out["walls"] = h["V_MAT_PARED"].isin(["7", "8", "9"]).astype(float).where(h["V_MAT_PARED"].notna())
    out["traditional"] = (h["V_TIPO_VIV"] == "4").astype("int8")
    for k in ["elec", "water", "water_strict", "sanit", "sanit_latrine", "cook", "crowd", "floor", "walls"]:
        out[k] = out[k].astype("float32")
    out["n_persons"] = h["n_persons"].astype("int16")
    out["n_indig"] = h["n_indig"].astype("int16")
    out["n_noinfo"] = h["n_noinfo"].astype("int16")
    out["n_otheth"] = h["n_otheth"].astype("int16")
    out["head_ethnic"] = pd.to_numeric(h["head_ethnic"], errors="coerce").fillna(0).astype("int8")
    out["head_sex"] = pd.to_numeric(h["head_sex"], errors="coerce").fillna(0).astype("int8")
    out["head_age"] = pd.to_numeric(h["head_age"], errors="coerce").fillna(0).astype("int8")
    out["head_educ"] = pd.to_numeric(h["head_educ"], errors="coerce").fillna(0).astype("int8")
    out["hid"] = (h["COD_ENCUESTAS"] + "-" + h["U_VIVIENDA"] + "-" + h["H_NROHOG"]).values
    ind = per[per["indig"] == 1]
    ip = pd.DataFrame({"hid": (ind["COD_ENCUESTAS"] + "-" + ind["U_VIVIENDA"] + "-" + ind["H_NROHOG"]).values,
                       "pueblo": ind["PA11_COD_ETNIA"].fillna("").values,
                       "sex": pd.to_numeric(ind["P_SEXO"], errors="coerce").fillna(0).astype("int8").values,
                       "age": pd.to_numeric(ind["P_EDADR"], errors="coerce").fillna(0).astype("int8").values})
    ip = ip[ip["hid"].isin(set(out["hid"]))]
    counts.update({"households": int(len(out)), "persons_in_households": int(out["n_persons"].sum()),
                   "indigenous_in_households": int(len(ip))})
    return out, ip, counts


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    files = sorted(SRC.glob("*.zip"))
    assert len(files) == 33, len(files)
    hh, ip, prov = [], [], {"source_folder": str(SRC), "files": {}}
    for f in files:
        o, p, counts = department(f)
        # households are unique within a department; the department code makes the key national
        o["hid"] = f.stem[:2] + "-" + o["hid"]
        p["hid"] = f.stem[:2] + "-" + p["hid"]
        hh.append(o)
        ip.append(p)
        prov["files"][f.name] = {"sha256": sha256(f), **counts}
        print(f.name, counts, flush=True)
    hh = pd.concat(hh, ignore_index=True)
    ip = pd.concat(ip, ignore_index=True)
    assert not hh["hid"].duplicated().any()
    hh.to_parquet(OUT / "households.parquet", index=False)
    ip.to_parquet(OUT / "indigenous.parquet", index=False)
    keys = next(iter(prov["files"].values())).keys() - {"sha256"}
    prov["totals"] = {k: int(sum(v[k] for v in prov["files"].values())) for k in sorted(keys)}
    (OUT / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    print(prov["totals"])


if __name__ == "__main__":
    main()
