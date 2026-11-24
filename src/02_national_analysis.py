"""Step 02: material deprivation of Indigenous peoples in Colombia by territory (CNPV 2018).

Inputs: data/processed/households.parquet, indigenous.parquet (step 01) and the census data dictionary (DDI).

1. Prevalence of the five deprivations, deprivation score and multiple deprivation (>= 2 of 5), for Indigenous and
   non-Indigenous persons, by type of territory: municipal head; populated centre; rural dispersed area outside a
   resguardo; Indigenous resguardo. The census asks the question on ethnic territories only outside municipal heads.
2. Inequality of the deprivation score among Indigenous persons: Theil T and GE(2), split exactly into between- and
   within-group components for several partitions (department, municipality, municipality x area, type of territory,
   detailed territory, people, people x type of territory). Specifications: main; with housing materials; latrine as
   unimproved sanitation; without overcrowding.
3. Ethnic gap within places: household-level linear models of the deprivation score on an Indigenous household head,
   weighted by household size, with fixed effects for municipality x area and municipality x territory, and household
   controls (education, sex and age group of the head, household size). Standard errors clustered by municipality.
4. Resguardos and their surroundings: mean score of Indigenous persons in each resguardo and in rural areas outside
   resguardos, compared with non-Indigenous persons in rural areas of the same municipality.

Census data: no sampling error; groups with fewer than 30 persons are not reported.

Outputs (results/): prevalence.csv, decomposition.csv, gap_models.csv, resguardos.csv,
                                  surroundings.csv, summary.json

Usage (from the repository root):  python src/02_national_analysis.py
"""
import json
import re
import warnings
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results"
SPECS = {"main": ["elec", "water", "sanit", "cook", "crowd"],
         "materials": ["elec", "water", "sanit", "cook", "crowd", "floor", "walls"],
         "latrine": ["elec", "water", "sanit_latrine", "cook", "crowd"],
         "no_crowding": ["elec", "water", "sanit", "cook"]}
ITEMS = ["elec", "water", "sanit", "cook", "crowd", "floor", "walls"]
TERR = {1: "Municipal head", 2: "Populated centre", 3: "Rural dispersed", 4: "Resguardo"}
MIN_CELL = 30
warnings.filterwarnings("ignore", category=RuntimeWarning)


def ddi_labels():
    t = ET.parse(ROOT / "data" / "raw" / "cnpv2018.ddi.xml").getroot()
    ns = t.tag.split("}")[0] + "}"
    out = {"resguardo": {}, "pueblo": {}}
    for v in t.iter(ns + "var"):
        n = v.get("name")
        cats = [((c.findtext(ns + "catValu") or "").strip(), (c.findtext(ns + "labl") or "").strip())
                for c in v.findall(ns + "catgry")]
        if n == "UVA2_CODTER" and not out["resguardo"]:
            for _, lab in cats:
                m = re.match(r"(\d+)_(.*?)_\d{5}_", lab)
                if m:
                    out["resguardo"].setdefault(m.group(1), m.group(2).title())
        elif n == "PA11_COD_ETNIA" and not out["pueblo"]:
            out["pueblo"] = dict(cats)
    return out


def ge_decomp(y, g, w=None, alpha=1):
    """Generalised entropy GE(alpha) (alpha = 1: Theil T; 2: half the squared CV), exact between/within split."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    mu = np.average(y, weights=w)
    s = y / mu
    if alpha == 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            total = np.average(np.where(s > 0, s * np.log(s), 0.0), weights=w)
    else:
        total = 0.5 * (np.average(s ** 2, weights=w) - 1)
    df = pd.DataFrame({"yw": y * w, "w": w, "g": np.asarray(g)}).groupby("g")[["yw", "w"]].sum()
    share, rel = df["w"] / df["w"].sum(), (df["yw"] / df["w"]) / mu
    if alpha == 1:
        between = float(np.sum(share * np.where(rel > 0, rel * np.log(rel), 0.0)))
    else:
        between = float(0.5 * (np.sum(share * rel ** 2) - 1))
    return float(total), between


def demean(a, codes, w):
    """Weighted within transformation of the columns of a by the group codes."""
    a = np.asarray(a, float)
    one = a.ndim == 1
    a = a[:, None] if one else a
    sw = np.bincount(codes, weights=w)
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        m = np.bincount(codes, weights=w * a[:, j]) / sw
        out[:, j] = a[:, j] - m[codes]
    return out[:, 0] if one else out


def wls_cluster(y, X, w, clusters):
    XtW = X.T * w
    A = np.linalg.inv(XtW @ X)
    b = A @ (XtW @ y)
    e = y - X @ b
    u = X * (w * e)[:, None]
    cl = pd.factorize(clusters)[0]
    S = np.zeros((X.shape[1], X.shape[1]))
    G = cl.max() + 1
    sums = np.zeros((G, X.shape[1]))
    np.add.at(sums, cl, u)
    S = sums.T @ sums
    n, k = X.shape
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * A @ S @ A
    return b, np.sqrt(np.diag(V)), G


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    lab = ddi_labels()
    hh = pd.read_parquet(DATA / "households.parquet")
    ip = pd.read_parquet(DATA / "indigenous.parquet")
    hh["terr"] = np.where(hh["clase"] == 1, 1, np.where(hh["tipoter"] == 1, 4, np.where(hh["clase"] == 2, 2, 3)))
    hh["area"] = np.where(hh["clase"] == 1, "head", "rural")
    for spec, items in SPECS.items():
        hh[f"score_{spec}"] = hh[items].mean(axis=1, skipna=True).astype("float32")
    hh["multi"] = (hh[SPECS["main"]].sum(axis=1, skipna=True) >= 2).astype("float32")
    summary_missing = int(hh["score_main"].isna().sum())
    hh["n_non"] = hh["n_persons"] - hh["n_indig"]
    summary = {"households": int(len(hh)), "persons": int(hh["n_persons"].sum()), "indigenous": int(len(ip)),
               "households_without_any_indicator": summary_missing}

    # 1. prevalence by type of territory (person-weighted)
    rows = []
    for pop, wcol in (("Indigenous", "n_indig"), ("Non-Indigenous", "n_non")):
        for t in [0, 1, 2, 3, 4]:
            d = hh if t == 0 else hh[hh["terr"] == t]
            w = d[wcol].astype(float)
            if w.sum() < MIN_CELL:
                continue
            r = {"population": pop, "territory": "All" if t == 0 else TERR[t], "persons": int(w.sum())}
            for k in ITEMS:
                ok = d[k].notna()
                r[k] = float(np.average(d.loc[ok, k], weights=w[ok]))
            ok = d["score_main"].notna()
            r["score"] = float(np.average(d.loc[ok, "score_main"], weights=w[ok]))
            r["multi"] = float(np.average(d.loc[ok, "multi"], weights=w[ok]))
            rows.append(r)
    prev = pd.DataFrame(rows)
    prev.to_csv(OUT / "prevalence.csv", index=False)

    # 2. decomposition among Indigenous persons (one row per person)
    p = ip.merge(hh[["hid", "dpto", "mpio", "area", "terr", "codter"] + [f"score_{s}" for s in SPECS]], on="hid",
                 how="inner")
    p["pueblo_lab"] = p["pueblo"].map(lab["pueblo"]).fillna("Not reported")
    p["territory"] = p["terr"].map(TERR)
    p["mpio_area"] = p["mpio"] + "|" + p["area"]
    p["detail"] = np.where(p["terr"] == 4, "R" + p["codter"], p["mpio"] + "|" + p["terr"].astype(str))
    p["pueblo_terr"] = p["pueblo_lab"] + "|" + p["territory"]
    summary["indigenous_by_territory"] = p["territory"].value_counts().to_dict()
    summary["n_resguardos_with_indigenous"] = int(p.loc[p["terr"] == 4, "codter"].nunique())
    summary["n_pueblos"] = int(p["pueblo_lab"].nunique())
    dec = []
    for spec in SPECS:
        y = p[f"score_{spec}"].to_numpy(float)
        ok = ~np.isnan(y)
        for part in ["dpto", "mpio", "mpio_area", "territory", "detail", "pueblo_lab", "pueblo_terr"]:
            g = p[part].to_numpy()[ok]
            T, bT = ge_decomp(y[ok], g, alpha=1)
            G2, bG = ge_decomp(y[ok], g, alpha=2)
            dec.append({"spec": spec, "partition": part, "n_groups": int(pd.Series(g).nunique()), "n": int(ok.sum()),
                        "mean_score": float(y[ok].mean()), "theil_T": T, "between_share_T": bT / T, "ge2": G2,
                        "between_share_ge2": bG / G2})
    dec = pd.DataFrame(dec)
    dec.to_csv(OUT / "decomposition.csv", index=False)

    # resguardos: mean score of their Indigenous population
    rg = (p[p["terr"] == 4].groupby("codter")
          .agg(persons=("score_main", "size"), score=("score_main", "mean"), mpio=("mpio", lambda s: s.mode().iat[0]),
               pueblo=("pueblo_lab", lambda s: s.mode().iat[0])).reset_index())
    rg["name"] = rg["codter"].map(lab["resguardo"])
    rg = rg[rg["persons"] >= MIN_CELL].sort_values("persons", ascending=False)
    rg.to_csv(OUT / "resguardos.csv", index=False)
    pb = (p.groupby("pueblo_lab")
          .agg(persons=("score_main", "size"), score=("score_main", "mean"),
               share_resguardo=("terr", lambda s: float((s == 4).mean())),
               share_head=("terr", lambda s: float((s == 1).mean())))
          .reset_index().sort_values("persons", ascending=False))
    pb[pb["persons"] >= MIN_CELL].to_csv(OUT / "pueblos.csv", index=False)
    # resguardos of Cesar (only the part of each resguardo inside the department), illustrative case
    rc = (p[(p["terr"] == 4) & (p["dpto"] == "20")].groupby("codter")
          .agg(persons=("score_main", "size"), score=("score_main", "mean"),
               pueblo=("pueblo_lab", lambda s: s.mode().iat[0])).reset_index())
    rc["name"] = rc["codter"].map(lab["resguardo"])
    rc[rc["persons"] >= MIN_CELL].sort_values("persons", ascending=False).to_csv(OUT / "resguardos_cesar.csv", index=False)
    summary["resguardos_reported"] = int(len(rg))
    summary["resguardo_score_quantiles"] = rg["score"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(4).to_dict()

    # 3. ethnic gap within places (households with a head of known ethnicity)
    h = hh[hh["head_ethnic"].between(1, 6) & hh["score_main"].notna()].copy()
    h["indig_head"] = (h["head_ethnic"] == 1).astype(float)
    w = h["n_persons"].to_numpy(float)
    y = h["score_main"].to_numpy(float)
    ctrl = pd.get_dummies(pd.DataFrame({
        "educ": h["head_educ"].where(h["head_educ"].isin(range(1, 11)), 99).astype(str),
        "sex": h["head_sex"].astype(str),
        "age": h["head_age"].astype(str),
        "size": np.minimum(h["n_persons"], 10).astype(str)}), drop_first=True).to_numpy(float)
    fe = {"none": None, "mpio_area": pd.factorize(h["mpio"] + "|" + h["area"])[0],
          "mpio_terr": pd.factorize(h["mpio"] + "|" + h["terr"].astype(str))[0]}
    models = []
    for name, codes, use_ctrl in (("raw", None, False), ("mpio_area", fe["mpio_area"], False),
                                  ("mpio_terr", fe["mpio_terr"], False), ("mpio_terr_controls", fe["mpio_terr"], True)):
        X = h[["indig_head"]].to_numpy(float)
        if use_ctrl:
            X = np.hstack([X, ctrl])
        if codes is None:
            Xd, yd = np.hstack([np.ones((len(X), 1)), X]), y
            j = 1
        else:
            Xd, yd = demean(X, codes, w), demean(y, codes, w)
            j = 0
        b, se, G = wls_cluster(yd, Xd, w, h["mpio"].to_numpy())
        models.append({"model": name, "gap": float(b[j]), "se": float(se[j]), "clusters": int(G), "households": int(len(h))})
        print(models[-1], flush=True)
    # within-place gap by type of territory (interactions of the Indigenous head with the territory)
    inter = np.column_stack([h["indig_head"].to_numpy(float) * (h["terr"].to_numpy() == t) for t in (1, 2, 3, 4)])
    Xd = demean(np.hstack([inter, ctrl]), fe["mpio_terr"], w)
    b, se, G = wls_cluster(demean(y, fe["mpio_terr"], w), Xd, w, h["mpio"].to_numpy())
    for j, t in enumerate((1, 2, 3, 4)):
        models.append({"model": f"mpio_terr_controls | {TERR[t]}", "gap": float(b[j]), "se": float(se[j]),
                       "clusters": int(G), "households": int(((h["terr"] == t) & (h["indig_head"] == 1)).sum())})
    gm = pd.DataFrame(models)
    gm["share_of_raw_gap"] = gm["gap"] / gm.loc[gm["model"] == "raw", "gap"].iat[0]
    gm.to_csv(OUT / "gap_models.csv", index=False)

    # 4. resguardos and rural areas outside them against non-Indigenous rural persons of the same municipality
    rur = hh[(hh["clase"] != 1) & hh["score_main"].notna()]
    non = (rur[rur["terr"] != 4].assign(wy=lambda d: d["score_main"] * d["n_non"])
           .groupby("mpio")[["wy", "n_non"]].sum())
    non = non[non["n_non"] >= MIN_CELL]
    non["non_rural_score"] = non["wy"] / non["n_non"]
    sur = []
    for t, label in ((4, "Resguardo"), (3, "Rural dispersed"), (2, "Populated centre")):
        g = (p[(p["terr"] == t) & p["score_main"].notna()].groupby("mpio")["score_main"].agg(["size", "mean"]).rename(
            columns={"size": "indig_persons", "mean": "indig_score"}))
        g = g[g["indig_persons"] >= MIN_CELL].join(non["non_rural_score"], how="inner")
        g["territory"] = label
        sur.append(g.reset_index())
    sur = pd.concat(sur, ignore_index=True)
    sur.to_csv(OUT / "surroundings.csv", index=False)
    summary["surroundings"] = {
        t: {"municipalities": int(len(g)), "corr": float(np.corrcoef(g["indig_score"], g["non_rural_score"])[0, 1]),
            "mean_gap_weighted": float(np.average(g["indig_score"] - g["non_rural_score"], weights=g["indig_persons"]))}
        for t, g in sur.groupby("territory")}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print(prev.round(3).to_string())
    print(dec[dec.spec == "main"][["partition", "n_groups", "mean_score", "theil_T", "between_share_T",
                                    "between_share_ge2"]].round(3).to_string())
    print(gm.round(4).to_string())


if __name__ == "__main__":
    main()
