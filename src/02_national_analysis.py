"""Step 02: material deprivation of Indigenous peoples in Colombia by territory (CNPV 2018).

Inputs: data/processed/households.parquet, indigenous.parquet, provenance.json (step 01) and the census data
dictionary (DDI). Population: persons in private households (special accommodation facilities excluded in step 01).

Comparison groups: Indigenous persons (census self-identification) and non-Indigenous persons with a reported ethnic
self-identification; persons whose ethnicity was not reported are left out of the comparisons.

Deprivation score: share of the five household deprivations (no electricity; unimproved water source for cooking,
WHO/UNICEF JMP; unimproved sanitation; cooking in a room also used for sleeping; critical overcrowding) among the
observed indicators, for households with at least three observed indicators. Multiple deprivation: deprived in at
least 40% of the observed indicators (two of five when all are observed).

1. Construction of the analytical population and missing indicators by population and type of territory.
2. Prevalence of the deprivations, score and multiple deprivation by population and type of territory: municipal head;
   populated centre; rural dispersed area outside a resguardo; Indigenous resguardo (asked only outside municipal heads).
3. Inequality of the score among Indigenous persons: Theil T and GE(2), split exactly into between- and within-group
   components (absolute values and shares) for several partitions, including a joint municipality x people partition.
   Specifications: main; with housing materials; latrine as unimproved sanitation; without overcrowding; strict water
   (rainwater, water trucks and carriers also unimproved); three basic services; households with all five indicators.
4. Ethnic disparity within places: household-level linear models of the score on an Indigenous household head,
   weighted by household size, with fixed effects for municipality x area and municipality x territory, and household
   controls (education, sex and age group of the head, household size); interactions with the type of territory and
   Wald tests of their equality. Standard errors clustered by municipal unit. Robustness: alternative scores, complete
   information, individual self-identification, unweighted households, strata with both groups, heads without any
   ethnic affiliation as comparator, and fixed effects for each resguardo.
5. Resguardos and their surroundings: mean score of Indigenous persons in each resguardo and in rural areas outside
   resguardos, against non-Indigenous persons in rural areas outside resguardos of the same municipality.

Census data: no sampling error for descriptive results; groups with fewer than 30 persons are not reported.

Outputs (results/): population_flow.csv, missing.csv, prevalence.csv, decomposition.csv,
  gap_models.csv, gap_contrasts.csv, robustness.csv, support.csv, resguardos.csv, resguardos_cesar.csv, pueblos.csv,
  surroundings.csv, summary.json

Usage (from the repository root):  python src/02_national_analysis.py
"""
import json
import re
import sys
import warnings
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
         "no_crowding": ["elec", "water", "sanit", "cook"],
         "water_strict": ["elec", "water_strict", "sanit", "cook", "crowd"],
         "basic3": ["elec", "water", "sanit"]}
MAIN = SPECS["main"]
ITEMS = ["elec", "water", "water_strict", "sanit", "cook", "crowd", "floor", "walls"]
TERR = {1: "Municipal head", 2: "Populated centre", 3: "Rural dispersed", 4: "Resguardo"}
MIN_CELL = 30
MIN_ITEMS = 3
FOREIGN = {"900", "910", "920", "930", "940", "950", "960", "970"}
UNKNOWN_PEOPLE = {"999", ""}
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
    """Generalised entropy GE(alpha) (alpha = 1: Theil T; 2: half the squared CV), exact between/within split.

    Convention for Theil T: s log s = 0 when s = 0, and a group with mean zero contributes zero to both components.
    Returns (total, between); within = total - between.
    """
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


def wls_cluster(y, X, w, clusters, vcov=False):
    """Weighted least squares with cluster-robust (CR1) standard errors; optionally the full covariance matrix."""
    XtW = X.T * w
    A = np.linalg.inv(XtW @ X)
    b = A @ (XtW @ y)
    e = y - X @ b
    u = X * (w * e)[:, None]
    cl = pd.factorize(clusters)[0]
    G = cl.max() + 1
    sums = np.zeros((G, X.shape[1]))
    np.add.at(sums, cl, u)
    S = sums.T @ sums
    n, k = X.shape
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * A @ S @ A
    return (b, np.sqrt(np.diag(V)), G, V) if vcov else (b, np.sqrt(np.diag(V)), G)


def score(df, items):
    n = df[items].notna().sum(axis=1)
    return df[items].mean(axis=1, skipna=True).where(n >= MIN_ITEMS).astype("float32")


def controls(h):
    return pd.get_dummies(pd.DataFrame({
        "educ": h["head_educ"].where(h["head_educ"].isin(range(1, 11)), 99).astype(str),
        "sex": h["head_sex"].astype(str),
        "age": h["head_age"].astype(str),
        "size": np.minimum(h["n_persons"], 10).astype(str)}), drop_first=True).to_numpy(float)


def fit(y, ind, w, mpio, codes=None, ctrl=None, vcov=False):
    """Coefficient of ind (one column, or several interaction columns) with optional fixed effects and controls."""
    X = ind if ind.ndim == 2 else ind[:, None]
    k = X.shape[1]
    if ctrl is not None:
        X = np.hstack([X, ctrl])
    if codes is None:
        X = np.hstack([X, np.ones((len(X), 1))])
        yd = y
    else:
        X, yd = demean(X, codes, w), demean(y, codes, w)
    res = wls_cluster(yd, X, w, mpio, vcov=vcov)
    if vcov:
        b, se, G, V = res
        return b[:k], se[:k], G, V[:k, :k]
    b, se, G = res
    return b[:k], se[:k], G


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    lab = ddi_labels()
    prov = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))["totals"]
    hh = pd.read_parquet(DATA / "households.parquet")
    ip = pd.read_parquet(DATA / "indigenous.parquet")
    hh["terr"] = np.where(hh["clase"] == 1, 1, np.where(hh["tipoter"] == 1, 4, np.where(hh["clase"] == 2, 2, 3)))
    hh["area"] = np.where(hh["clase"] == 1, "head", "rural")
    hh["n_items"] = hh[MAIN].notna().sum(axis=1)
    hh["complete"] = hh["n_items"] == len(MAIN)
    for spec, items in SPECS.items():
        hh[f"score_{spec}"] = score(hh, items)
    hh["score_complete"] = hh["score_main"].where(hh["complete"])
    hh["multi"] = ((hh[MAIN].sum(axis=1, skipna=True) / hh["n_items"]) >= 0.4).astype("float32").where(
        hh["score_main"].notna())
    hh["n_non"] = hh["n_persons"] - hh["n_indig"] - hh["n_noinfo"]  # non-Indigenous with reported ethnicity
    scored = hh["score_main"].notna()

    # 1. analytical population and missing indicators
    flow = [("Persons in the census files", prov["persons_file"], None),
            ("Persons in special accommodation facilities (excluded)", prov["persons_lea"], prov["indigenous_lea"]),
            ("Persons in private households", prov["persons_private"], int(len(ip))),
            ("Private households", prov["households"], None),
            ("Persons in private households not linked to a household record", prov["persons_not_linked"], None),
            ("Persons in households with fewer than three observed indicators (excluded)",
             int(hh.loc[~scored, "n_persons"].sum()), int(hh.loc[~scored, "n_indig"].sum())),
            ("Persons with ethnicity not reported (excluded from comparisons)",
             int(hh.loc[scored, "n_noinfo"].sum()), None),
            ("Analytical population: Indigenous persons", int(hh.loc[scored, "n_indig"].sum()),
             int(hh.loc[scored, "n_indig"].sum())),
            ("Analytical population: non-Indigenous persons", int(hh.loc[scored, "n_non"].sum()), None)]
    summary = {"persons_file": prov["persons_file"], "persons_lea": prov["persons_lea"],
               "indigenous_lea": prov["indigenous_lea"], "households": int(len(hh)),
               "persons": int(hh["n_persons"].sum()), "indigenous": int(len(ip)),
               "households_unscored": int((~scored).sum()), "persons_unscored": int(hh.loc[~scored, "n_persons"].sum()),
               "indigenous_unscored": int(hh.loc[~scored, "n_indig"].sum()),
               "persons_noinfo": int(hh["n_noinfo"].sum()),
               "indigenous_scored": int(hh.loc[scored, "n_indig"].sum()),
               "non_indigenous_scored": int(hh.loc[scored, "n_non"].sum())}
    miss = []
    for pop, wcol in (("Indigenous", "n_indig"), ("Non-Indigenous", "n_non")):
        for t in [0, 1, 2, 3, 4]:
            d = hh if t == 0 else hh[hh["terr"] == t]
            w = d[wcol].astype(float)
            r = {"population": pop, "territory": "All" if t == 0 else TERR[t], "persons": int(w.sum())}
            for k in MAIN:
                r[f"missing_{k}"] = float((w * d[k].isna()).sum() / w.sum())
            r["water_not_applicable"] = float((w * d["water_na_noprep"]).sum() / w.sum())
            r["all_five_observed"] = float((w * d["complete"]).sum() / w.sum())
            r["unscored"] = float((w * d["score_main"].isna()).sum() / w.sum())
            miss.append(r)
    pd.DataFrame(miss).to_csv(OUT / "missing.csv", index=False)

    # 2. prevalence by type of territory (person-weighted, scored households)
    rows = []
    for pop, wcol in (("Indigenous", "n_indig"), ("Non-Indigenous", "n_non")):
        for t in [0, 1, 2, 3, 4]:
            d = hh[scored] if t == 0 else hh[scored & (hh["terr"] == t)]
            w = d[wcol].astype(float)
            if w.sum() < MIN_CELL:
                continue
            r = {"population": pop, "territory": "All" if t == 0 else TERR[t], "persons": int(w.sum())}
            for k in ITEMS:
                ok = d[k].notna()
                r[k] = float(np.average(d.loc[ok, k], weights=w[ok]))
                r[f"n_{k}"] = int(w[ok].sum())
            r["score"] = float(np.average(d["score_main"], weights=w))
            r["multi"] = float(np.average(d["multi"], weights=w))
            rows.append(r)
    prev = pd.DataFrame(rows)
    prev.to_csv(OUT / "prevalence.csv", index=False)

    # 3. decomposition among Indigenous persons (one row per person)
    scols = [f"score_{s}" for s in SPECS] + ["score_complete"]
    p = ip.merge(hh[["hid", "dpto", "mpio", "area", "terr", "codter"] + scols], on="hid", how="inner")
    p = p[p["score_main"].notna()].copy()
    p["identified"] = ~p["pueblo"].isin(FOREIGN | UNKNOWN_PEOPLE)
    p["pueblo_lab"] = p["pueblo"].map(lab["pueblo"]).fillna("Not reported")
    p["territory"] = p["terr"].map(TERR)
    p["mpio_area"] = p["mpio"] + "|" + p["area"]
    p["detail"] = np.where(p["terr"] == 4, "R" + p["codter"], p["mpio"] + "|" + p["terr"].astype(str))
    p["pueblo_terr"] = p["pueblo_lab"] + "|" + p["territory"]
    p["mpio_pueblo"] = p["mpio"] + "|" + p["pueblo_lab"]
    summary["indigenous_by_territory"] = p["territory"].value_counts().to_dict()
    summary["n_resguardos_with_indigenous"] = int(p.loc[p["terr"] == 4, "codter"].nunique())
    summary["n_people_categories"] = int(p["pueblo"].nunique())
    summary["n_native_peoples"] = int(p.loc[p["identified"], "pueblo"].nunique())
    summary["n_foreign_categories"] = int(p.loc[p["pueblo"].isin(FOREIGN), "pueblo"].nunique())
    summary["persons_people_unknown"] = int(p["pueblo"].isin(UNKNOWN_PEOPLE).sum())
    summary["persons_people_foreign"] = int(p["pueblo"].isin(FOREIGN).sum())
    summary["persons_complete"] = int(p["score_complete"].notna().sum())
    parts = ["dpto", "mpio", "mpio_area", "territory", "detail", "pueblo_lab", "pueblo_terr", "mpio_pueblo"]
    dec = []
    for spec in list(SPECS) + ["complete", "identified"]:
        col = "score_main" if spec == "identified" else f"score_{spec}"
        sub = p[p["identified"]] if spec == "identified" else p
        y = sub[col].to_numpy(float)
        ok = ~np.isnan(y)
        for part in parts:
            g = sub[part].to_numpy()[ok]
            T, bT = ge_decomp(y[ok], g, alpha=1)
            G2, bG = ge_decomp(y[ok], g, alpha=2)
            dec.append({"spec": spec, "partition": part, "n_groups": int(pd.Series(g).nunique()), "n": int(ok.sum()),
                        "mean_score": float(y[ok].mean()), "theil_T": T, "between_T": bT, "within_T": T - bT,
                        "between_share_T": bT / T, "ge2": G2, "between_share_ge2": bG / G2})
    dec = pd.DataFrame(dec)
    dec.to_csv(OUT / "decomposition.csv", index=False)
    m = dec[dec.spec == "main"].set_index("partition")["between_share_T"]
    summary["joint_mpio_pueblo"] = {"between_share": float(m["mpio_pueblo"]),
                                    "added_by_people_given_mpio": float(m["mpio_pueblo"] - m["mpio"]),
                                    "added_by_mpio_given_people": float(m["mpio_pueblo"] - m["pueblo_lab"]),
                                    "shapley_mpio": float(0.5 * (m["mpio"] + m["mpio_pueblo"] - m["pueblo_lab"])),
                                    "shapley_people": float(0.5 * (m["pueblo_lab"] + m["mpio_pueblo"] - m["mpio"]))}

    # resguardos, peoples and the resguardos of Cesar
    rg = (p[p["terr"] == 4].groupby("codter")
          .agg(persons=("score_main", "size"), score=("score_main", "mean"), mpio=("mpio", lambda s: s.mode().iat[0]),
               pueblo=("pueblo_lab", lambda s: s.mode().iat[0])).reset_index())
    rg["name"] = rg["codter"].map(lab["resguardo"])
    rg = rg[rg["persons"] >= MIN_CELL].sort_values("persons", ascending=False)
    rg.to_csv(OUT / "resguardos.csv", index=False)
    pb = (p.groupby(["pueblo", "pueblo_lab"])
          .agg(persons=("score_main", "size"), score=("score_main", "mean"),
               share_resguardo=("terr", lambda s: float((s == 4).mean())),
               share_head=("terr", lambda s: float((s == 1).mean())),
               n_mpio=("mpio", "nunique"))
          .reset_index().sort_values("persons", ascending=False))
    pb["identified"] = ~pb["pueblo"].isin(FOREIGN | UNKNOWN_PEOPLE)
    pb[pb["persons"] >= MIN_CELL].to_csv(OUT / "pueblos.csv", index=False)
    rc = (p[(p["terr"] == 4) & (p["dpto"] == "20")].groupby("codter")
          .agg(persons=("score_main", "size"), score=("score_main", "mean"),
               pueblo=("pueblo_lab", lambda s: s.mode().iat[0])).reset_index())
    rc["name"] = rc["codter"].map(lab["resguardo"])
    rc[rc["persons"] >= MIN_CELL].sort_values("persons", ascending=False).to_csv(OUT / "resguardos_cesar.csv", index=False)
    summary["resguardos_reported"] = int(len(rg))
    summary["resguardo_score_quantiles"] = rg["score"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(4).to_dict()
    del p

    # 4. ethnic disparity within places (households with a head of reported ethnicity)
    h = hh[hh["head_ethnic"].between(1, 6) & scored].copy()
    h["indig_head"] = (h["head_ethnic"] == 1).astype(float)
    h["mt"] = h["mpio"] + "|" + h["terr"].astype(str)
    mpio = h["mpio"].to_numpy()
    w = h["n_persons"].to_numpy(float)
    y = h["score_main"].to_numpy(float)
    ind = h["indig_head"].to_numpy(float)
    ctrl = controls(h)
    fe_ma = pd.factorize(h["mpio"] + "|" + h["area"])[0]
    fe_mt = pd.factorize(h["mt"])[0]
    models = []
    for name, codes, c in (("raw", None, None), ("mpio_area", fe_ma, None), ("mpio_terr", fe_mt, None),
                           ("mpio_terr_controls", fe_mt, ctrl)):
        b, se, G = fit(y, ind, w, mpio, codes, c)
        models.append({"model": name, "gap": float(b[0]), "se": float(se[0]), "clusters": int(G),
                       "households": int(len(h)), "persons": int(w.sum()),
                       "strata": 1 if codes is None else int(codes.max() + 1)})
        print(models[-1], flush=True)
    inter = np.column_stack([ind * (h["terr"].to_numpy() == t) for t in (1, 2, 3, 4)])
    b, se, G, V = fit(y, inter, w, mpio, fe_mt, ctrl, vcov=True)
    for j, t in enumerate((1, 2, 3, 4)):
        sel = (h["terr"] == t).to_numpy()
        models.append({"model": f"mpio_terr_controls | {TERR[t]}", "gap": float(b[j]), "se": float(se[j]),
                       "clusters": int(G), "households": int((sel & (ind == 1)).sum()),
                       "persons": int(w[sel & (ind == 1)].sum()), "strata": int(pd.Series(fe_mt[sel]).nunique())})
    gm = pd.DataFrame(models)
    gm["share_of_raw_gap"] = gm["gap"] / gm.loc[gm["model"] == "raw", "gap"].iat[0]
    gm["lo"], gm["hi"] = gm["gap"] - 1.96 * gm["se"], gm["gap"] + 1.96 * gm["se"]
    gm.to_csv(OUT / "gap_models.csv", index=False)
    from scipy import stats
    con = []
    for i, j in ((0, 2), (0, 3), (2, 3), (0, 1), (1, 2), (1, 3)):
        d = b[i] - b[j]
        s = float(np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j]))
        con.append({"contrast": f"{TERR[i + 1]} - {TERR[j + 1]}", "difference": float(d), "se": s,
                    "z": float(d / s), "p": float(2 * stats.norm.sf(abs(d / s)))})
    R = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]], float)
    rb = R @ b
    wald = float(rb @ np.linalg.inv(R @ V @ R.T) @ rb)
    con.append({"contrast": "Joint equality of the four coefficients (Wald, 3 df)", "difference": np.nan, "se": np.nan,
                "z": wald, "p": float(stats.chi2.sf(wald, 3))})
    pd.DataFrame(con).to_csv(OUT / "gap_contrasts.csv", index=False)

    # common support: strata (municipality x type of territory) with both groups
    h["p_ind"] = h["n_persons"] * h["indig_head"]
    st = h.groupby("mt").agg(terr=("terr", "first"), hh_ind=("indig_head", "sum"), hh_all=("indig_head", "size"),
                             p_ind=("p_ind", "sum"), p_all=("n_persons", "sum"))
    st["hh_non"] = st["hh_all"] - st["hh_ind"]
    st["share_ind"] = st["p_ind"] / st["p_all"]
    st["both"] = (st["hh_ind"] >= 1) & (st["hh_non"] >= 1)
    st["support"] = (st["hh_ind"] >= 10) & (st["hh_non"] >= 10)
    sup = []
    for t in (0, 1, 2, 3, 4):
        d = st if t == 0 else st[st["terr"] == t]
        di = d[d["hh_ind"] >= 1]
        sup.append({"territory": "All" if t == 0 else TERR[t], "strata_with_indigenous_heads": int(len(di)),
                    "strata_with_both": int(di["both"].sum()), "strata_with_10_of_each": int(di["support"].sum()),
                    "indigenous_head_persons": float(di["p_ind"].sum()),
                    "share_in_strata_with_both": float(di.loc[di["both"], "p_ind"].sum() / di["p_ind"].sum()),
                    "share_in_strata_with_10_of_each": float(di.loc[di["support"], "p_ind"].sum() / di["p_ind"].sum()),
                    "indigenous_share_p10": float(np.percentile(di["share_ind"], 10)),
                    "indigenous_share_median": float(di["share_ind"].median()),
                    "indigenous_share_p90": float(np.percentile(di["share_ind"], 90))})
    pd.DataFrame(sup).to_csv(OUT / "support.csv", index=False)
    support_strata = set(st.index[st["support"]])

    # robustness of the within-place disparity
    def row(spec, label, yv, indv, wv, mp, codes, c, terr_v):
        k = np.isfinite(yv)  # alternative scores can be missing where the main score is not
        if not k.all():
            yv, indv, wv, mp, codes, c, terr_v = yv[k], indv[k], wv[k], mp[k], codes[k], c[k], terr_v[k]
        r0 = fit(yv, indv, wv, mp)
        r1 = fit(yv, indv, wv, mp, codes, c)
        it = np.column_stack([indv * (terr_v == t) for t in (1, 2, 3, 4)])
        r2 = fit(yv, it, wv, mp, codes, c)
        out = {"spec": spec, "label": label, "units": int(len(yv)), "raw_gap": float(r0[0][0]),
               "gap": float(r1[0][0]), "se": float(r1[1][0]), "share_of_raw_gap": float(r1[0][0] / r0[0][0])}
        for j, t in enumerate((1, 2, 3, 4)):
            out[f"gap_{t}"], out[f"se_{t}"] = float(r2[0][j]), float(r2[1][j])
        print(out, flush=True)
        return out

    terr_v = h["terr"].to_numpy()
    rob = [row("main", "Main score", y, ind, w, mpio, fe_mt, ctrl, terr_v)]
    for spec, label in (("basic3", "Three basic services"), ("water_strict", "Strict water classification"),
                        ("no_crowding", "Without overcrowding")):
        rob.append(row(spec, label, h[f"score_{spec}"].to_numpy(float), ind, w, mpio, fe_mt, ctrl, terr_v))
    k = h["complete"].to_numpy()
    rob.append(row("complete", "Households with all five indicators", y[k], ind[k], w[k], mpio[k],
                   pd.factorize(h["mt"][k])[0], ctrl[k], terr_v[k]))
    rob.append(row("unweighted", "Households unweighted", y, ind, np.ones_like(w), mpio, fe_mt, ctrl, terr_v))
    k = h["mt"].isin(support_strata).to_numpy()
    rob.append(row("support", "Strata with at least 10 households of each group", y[k], ind[k], w[k], mpio[k],
                   pd.factorize(h["mt"][k])[0], ctrl[k], terr_v[k]))
    k = h["head_ethnic"].isin([1, 6]).to_numpy()
    rob.append(row("no_ethnic_comparator", "Comparator: heads without ethnic affiliation", y[k], ind[k], w[k], mpio[k],
                   pd.factorize(h["mt"][k])[0], ctrl[k], terr_v[k]))
    det = np.where(h["terr"] == 4, "R" + h["codter"] + "|" + h["mpio"], h["mt"])
    rob.append(row("resguardo_fe", "Fixed effects for each resguardo", y, ind, w, mpio, pd.factorize(det)[0], ctrl,
                   terr_v))
    # individual self-identification: each household enters once for its Indigenous members and once for its
    # non-Indigenous members with reported ethnicity, weighted by their number; controls refer to the household head
    a = (h["n_indig"] > 0).to_numpy()
    c2 = (h["n_non"] > 0).to_numpy()
    yi = np.concatenate([y[a], y[c2]])
    ii = np.concatenate([np.ones(a.sum()), np.zeros(c2.sum())])
    wi = np.concatenate([h["n_indig"].to_numpy(float)[a], h["n_non"].to_numpy(float)[c2]])
    mi = np.concatenate([mpio[a], mpio[c2]])
    fi = pd.factorize(np.concatenate([h["mt"].to_numpy()[a], h["mt"].to_numpy()[c2]]))[0]
    rob.append(row("individual", "Individual self-identification", yi, ii, wi, mi, fi,
                   np.vstack([ctrl[a], ctrl[c2]]), np.concatenate([terr_v[a], terr_v[c2]])))
    del yi, ii, wi, mi, fi
    pd.DataFrame(rob).to_csv(OUT / "robustness.csv", index=False)
    summary["households_mixed_head_members"] = {
        "indigenous_persons_with_non_indigenous_head": int(h.loc[h["indig_head"] == 0, "n_indig"].sum()),
        "non_indigenous_persons_with_indigenous_head": int(h.loc[h["indig_head"] == 1, "n_non"].sum())}

    # 5. resguardos and rural areas outside them against non-Indigenous rural persons of the same municipality
    p = ip.merge(hh[["hid", "mpio", "terr", "score_main"]], on="hid", how="inner")
    p = p[p["score_main"].notna()]
    rur = hh[(hh["clase"] != 1) & scored]
    non = (rur[rur["terr"] != 4].assign(wy=lambda d: d["score_main"] * d["n_non"])
           .groupby("mpio")[["wy", "n_non"]].sum())
    non = non[non["n_non"] >= MIN_CELL]
    non["non_rural_score"] = non["wy"] / non["n_non"]
    sur = []
    for t, label in ((4, "Resguardo"), (3, "Rural dispersed"), (2, "Populated centre")):
        g = (p[p["terr"] == t].groupby("mpio")["score_main"].agg(["size", "mean"]).rename(
            columns={"size": "indig_persons", "mean": "indig_score"}))
        g = g[g["indig_persons"] >= MIN_CELL].join(non[["non_rural_score", "n_non"]], how="inner")
        g = g.rename(columns={"n_non": "non_rural_persons"})
        g["territory"] = label
        sur.append(g.reset_index())
    sur = pd.concat(sur, ignore_index=True)
    sur.to_csv(OUT / "surroundings.csv", index=False)
    summary["surroundings"] = {
        t: {"municipalities": int(len(g)), "corr": float(np.corrcoef(g["indig_score"], g["non_rural_score"])[0, 1]),
            "mean_gap_weighted": float(np.average(g["indig_score"] - g["non_rural_score"], weights=g["indig_persons"])),
            "mean_gap_unweighted": float((g["indig_score"] - g["non_rural_score"]).mean()),
            "share_indigenous_higher": float((g["indig_score"] > g["non_rural_score"]).mean()),
            "n_indigenous_higher": int((g["indig_score"] > g["non_rural_score"]).sum()),
            "indigenous_persons": int(g["indig_persons"].sum())}
        for t, g in sur.groupby("territory")}
    pd.DataFrame(flow, columns=["stage", "persons", "indigenous_persons"]).to_csv(OUT / "population_flow.csv",
                                                                                index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print(prev.round(3).to_string())
    print(dec[dec.spec == "main"][["partition", "n_groups", "theil_T", "between_share_T"]].round(3).to_string())
    print(gm.round(4).to_string())


if __name__ == "__main__":
    main()
