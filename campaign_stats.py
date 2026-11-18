"""Evidence engine for CampaignIQ.

Replaces the single-number DiD with: matched control, bootstrap CI, a placebo
(pre-trend) test, minimum-sample rules, segment-level lift, type-level pooling,
coupon cost, and a Repeat / Retest / Drop verdict.
"""
import numpy as np
import pandas as pd

MIN_RECIPIENTS = 50      # below this a campaign is flagged "insufficient data"
MIN_CELL = 30            # minimum treated AND control households per segment cell
N_BOOT = 300


# ---------- data prep ----------
def build_matrix(tx):
    """Dense household x day spend matrix (+ leading zero column for cumsum)."""
    hh_ids = np.sort(tx["HOUSEHOLD_KEY"].unique())
    pos = {h: i for i, h in enumerate(hh_ids)}
    max_day = int(tx["DAY"].max())
    M = np.zeros((len(hh_ids), max_day), dtype=np.float64)
    g = tx.groupby(["HOUSEHOLD_KEY", "DAY"], as_index=False)["SALES_VALUE"].sum()
    M[g["HOUSEHOLD_KEY"].map(pos).to_numpy(), g["DAY"].to_numpy() - 1] = g["SALES_VALUE"].to_numpy()
    C = np.concatenate([np.zeros((len(hh_ids), 1)), np.cumsum(M, axis=1)], axis=1)
    return hh_ids, C, max_day


def wmean(C, a, b):
    """Per-household mean daily spend over days a..b inclusive (1-indexed)."""
    return (C[:, b] - C[:, a - 1]) / (b - a + 1)


# ---------- estimator ----------
def _point(preT, dT, preC, dC, nbins=5):
    """Stratified DiD: reweight controls to the treated pre-spend distribution."""
    edges = np.unique(np.quantile(preT, np.linspace(0, 1, nbins + 1)))
    if len(edges) < 3:
        return dT.mean() - dC.mean()
    bt = np.clip(np.searchsorted(edges, preT, side="right") - 1, 0, len(edges) - 2)
    bc = np.clip(np.searchsorted(edges, preC, side="right") - 1, 0, len(edges) - 2)
    est, wsum = 0.0, 0.0
    for k in range(len(edges) - 1):
        mt, mc = bt == k, bc == k
        if mt.sum() == 0 or mc.sum() == 0:
            continue
        w = mt.mean()
        est += w * (dT[mt].mean() - dC[mc].mean())
        wsum += w
    return est / wsum if wsum else np.nan


def did_with_ci(preT, durT, preC, durC, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    dT, dC = durT - preT, durC - preC
    est = _point(preT, dT, preC, dC)
    boots = np.empty(n_boot)
    nT, nC = len(preT), len(preC)
    for i in range(n_boot):
        it, ic = rng.integers(0, nT, nT), rng.integers(0, nC, nC)
        boots[i] = _point(preT[it], dT[it], preC[ic], dC[ic])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return est, lo, hi


# ---------- per-campaign analysis ----------
def overlap_households(cid, a, b, campaigns, sent):
    ids = campaigns.loc[(campaigns["CAMPAIGN"] != cid) & (campaigns["START_DAY"] <= b)
                        & (campaigns["END_DAY"] >= a), "CAMPAIGN"]
    return set(sent.loc[sent["CAMPAIGN"].isin(ids), "HOUSEHOLD_KEY"])


def coupon_cost(tx_coupon, redemptions, cid):
    """Retailer-funded coupon discount on the exact household-days where this campaign's coupons were redeemed."""
    r = redemptions.loc[redemptions["CAMPAIGN"] == cid, ["HOUSEHOLD_KEY", "DAY"]].drop_duplicates()
    if r.empty:
        return 0.0
    m = r.merge(tx_coupon, on=["HOUSEHOLD_KEY", "DAY"], how="left")
    return float(-m["COUPON_DISC"].fillna(0).sum())


def analyze_campaign(row, hh_ids, C, max_day, campaigns, sent, hh_demo=None, demo_field=None,
                     tx_coupon=None, redemptions=None):
    cid, start = int(row.CAMPAIGN), int(row.START_DAY)

    # Exclude campaigns whose full duration extends beyond transaction data.
    if int(row.END_DAY) > max_day:
        return None
   
    end = int(row.END_DAY)
    truncated = False
    
    dur = end - start + 1
    pre1_a, pre1_b = start - dur, start - 1
    pre2_a, pre2_b = start - 2 * dur, start - dur - 1
    if pre1_a < 1:
        return None

    recips = set(sent.loc[sent["CAMPAIGN"] == cid, "HOUSEHOLD_KEY"])
    excl = recips | overlap_households(cid, max(pre2_a, 1), end, campaigns, sent)
    inT = np.isin(hh_ids, list(recips))
    inC = ~np.isin(hh_ids, list(excl))

    pre = wmean(C, pre1_a, pre1_b)
    during = wmean(C, start, end)
    active = pre > 0                                   # condition on PRE spend only (not the outcome)
    T, Ct = inT & active, inC & active
    out = dict(CAMPAIGN=cid, TYPE=row.DESCRIPTION, DAYS=dur, END_TRUNCATED=truncated,
               N_TREATED=int(T.sum()), N_CONTROL=int(Ct.sum()))
    if T.sum() < 5 or Ct.sum() < 5:
        return out

    est, lo, hi = did_with_ci(pre[T], during[T], pre[Ct], during[Ct])
    base = pre[T].mean()
    out.update(LIFT_PER_HH_DAY=est, CI_LO=lo, CI_HI=hi, LIFT_PCT=est / base if base else np.nan,
               INCR_SALES=est * dur * T.sum())

    # placebo: pretend the campaign started one window earlier -> should be ~0 if trends are parallel
    if pre2_a >= 1:
        pre2 = wmean(C, pre2_a, pre2_b)
        a2 = pre2 > 0
        T2, C2 = inT & a2, inC & a2
        p, plo, phi = did_with_ci(pre2[T2], pre[T2], pre2[C2], pre[C2], n_boot=150, seed=1)
        se1, se2 = (hi - lo) / 3.92, (phi - plo) / 3.92
        adj, adj_se = est - p, float(np.hypot(se1, se2))
        out.update(PLACEBO=p, PLACEBO_LO=plo, PLACEBO_HI=phi, PRETREND_OK=bool(plo <= 0 <= phi),
                   ADJ_LIFT=adj, ADJ_LO=adj - 1.96 * adj_se, ADJ_HI=adj + 1.96 * adj_se)
    else:
        out.update(PLACEBO=np.nan, PLACEBO_LO=np.nan, PLACEBO_HI=np.nan, PRETREND_OK=np.nan,
                   ADJ_LIFT=np.nan, ADJ_LO=np.nan, ADJ_HI=np.nan)

    if tx_coupon is not None and redemptions is not None:
        out["COUPON_COST"] = coupon_cost(tx_coupon, redemptions, cid)
        out["INCR_SALES_PER_COUPON_$"] = out["INCR_SALES"] / out["COUPON_COST"] if out["COUPON_COST"] > 0 else np.nan

    # segment lift (same estimator, controls drawn from the same demographic segment)
    seg_rows = []
    if hh_demo is not None and demo_field is not None:
        seg = pd.Series(hh_ids).map(hh_demo.set_index("HOUSEHOLD_KEY")[demo_field]).to_numpy()
        for s in pd.unique(seg[~pd.isna(seg)]):
            m = seg == s
            nt, nc = int((T & m).sum()), int((Ct & m).sum())
            if nt < MIN_CELL or nc < MIN_CELL:
                seg_rows.append(dict(CAMPAIGN=cid, SEGMENT=s, N_TREATED=nt, N_CONTROL=nc,
                                     LIFT_PER_HH_DAY=np.nan, CI_LO=np.nan, CI_HI=np.nan))
                continue
            e, l, h = did_with_ci(pre[T & m], during[T & m], pre[Ct & m], during[Ct & m], n_boot=200, seed=2)
            seg_rows.append(dict(CAMPAIGN=cid, SEGMENT=s, N_TREATED=nt, N_CONTROL=nc,
                                 LIFT_PER_HH_DAY=e, CI_LO=l, CI_HI=h))
    out["_segments"] = seg_rows
    return out


def verdict(r):
    if r.get("N_TREATED", 0) < MIN_RECIPIENTS or pd.isna(r.get("LIFT_PER_HH_DAY")):
        return "Insufficient data"
    lo, hi = r["CI_LO"], r["CI_HI"]
    note = ""
    if r.get("PRETREND_OK") is False and pd.notna(r.get("ADJ_LO")):
        lo, hi, note = r["ADJ_LO"], r["ADJ_HI"], " (trend-adjusted)"   # targeted HHs were already diverging pre-campaign
    if lo > 0:
        return "Repeat" + note
    if hi < 0:
        return "Drop" + note
    return "Retest" + note


def pool_by_type(res):
    """Inverse-variance pooled lift by campaign type: answers 'which traits work?' with more power."""
    d = res.dropna(subset=["LIFT_PER_HH_DAY"]).copy()
    d = d[d["N_TREATED"] >= MIN_RECIPIENTS]
    d["SE"] = (d["CI_HI"] - d["CI_LO"]) / (2 * 1.96)
    d = d[d["SE"] > 0]
    d["W"] = 1 / d["SE"] ** 2
    g = d.groupby("TYPE").apply(lambda x: pd.Series({
        "CAMPAIGNS": len(x),
        "POOLED_LIFT": np.average(x["LIFT_PER_HH_DAY"], weights=x["W"]),
        "POOLED_SE": 1 / np.sqrt(x["W"].sum()),
    }), include_groups=False)
    g["CI_LO"], g["CI_HI"] = g["POOLED_LIFT"] - 1.96 * g["POOLED_SE"], g["POOLED_LIFT"] + 1.96 * g["POOLED_SE"]
    return g.reset_index()


def run_all(tx, campaigns, sent, hh_demo=None, demo_field=None, redemptions=None):
    hh_ids, C, max_day = build_matrix(tx)
    cols = ["HOUSEHOLD_KEY", "DAY", "COUPON_DISC"]
    tx_coupon = tx[cols].groupby(["HOUSEHOLD_KEY", "DAY"], as_index=False)["COUPON_DISC"].sum() \
        if "COUPON_DISC" in tx.columns else None
    rows, segs = [], []
    for r in campaigns.itertuples(index=False):
        o = analyze_campaign(r, hh_ids, C, max_day, campaigns, sent, hh_demo, demo_field, tx_coupon, redemptions)
        if o is None:
            continue
        segs += o.pop("_segments", [])
        o["VERDICT"] = verdict(o)
        rows.append(o)
    return pd.DataFrame(rows), pd.DataFrame(segs)


# ---------- helpers used by the UI ----------
def campaign_groups(row, hh_ids, C, max_day, campaigns, sent):
    """Treated / control masks for one campaign (same definitions as analyze_campaign)."""
    cid, start = int(row.CAMPAIGN), int(row.START_DAY)
    end = min(int(row.END_DAY), max_day)
    dur = end - start + 1
    if start - dur < 1:
        return None
    recips = set(sent.loc[sent["CAMPAIGN"] == cid, "HOUSEHOLD_KEY"])
    ov = overlap_households(cid, max(start - 2 * dur, 1), end, campaigns, sent)
    inT = np.isin(hh_ids, list(recips))
    inC = ~np.isin(hh_ids, list(recips | ov))
    pre = wmean(C, start - dur, start - 1)
    active = pre > 0
    n_overlap_campaigns = int(((campaigns["CAMPAIGN"] != cid) & (campaigns["START_DAY"] <= end)
                               & (campaigns["END_DAY"] >= max(start - 2 * dur, 1))).sum())
    never_sent = ~np.isin(hh_ids, sent["HOUSEHOLD_KEY"].unique())
    return dict(cid=cid, start=start, end=end, dur=dur, T=inT & active, Ct=inC & active,
                n_overlap_campaigns=n_overlap_campaigns,
                control_never_sent_share=float(never_sent[inC & active].mean()) if (inC & active).any() else np.nan)


def trend_index(C, g, max_day, block=7):
    """Weekly spend of treated vs control, indexed to each group's own pre-campaign mean (=100)."""
    s, e, d = g["start"], g["end"], g["dur"]
    base_t = wmean(C, s - d, s - 1)[g["T"]].mean()
    base_c = wmean(C, s - d, s - 1)[g["Ct"]].mean()
    lo, hi = max(1, s - 2 * d), min(max_day, e + d)
    rows = []
    for a in range(lo, hi + 1, block):
        b = min(a + block - 1, hi)
        if b - a + 1 < 3:
            continue
        w = wmean(C, a, b)
        rows.append(dict(DAY=a, Treated=100 * w[g["T"]].mean() / base_t if base_t else np.nan,
                         Control=100 * w[g["Ct"]].mean() / base_c if base_c else np.nan))
    return pd.DataFrame(rows)


def period_compare(C, g, max_day):
    """Average daily spend before / during / after for treated AND control."""
    s, e, d = g["start"], g["end"], g["dur"]
    periods = {"Before": (s - d, s - 1), "During": (s, e)}
    if e < max_day:
        periods["After"] = (e + 1, min(max_day, e + d))
    rows = []
    for name, (a, b) in periods.items():
        w = wmean(C, a, b)
        rows.append(dict(Period=name, Group="Recipients", Spend=w[g["T"]].mean()))
        rows.append(dict(Period=name, Group="Comparison households", Spend=w[g["Ct"]].mean()))
    return pd.DataFrame(rows)


def decision_table(res):
    """Use consistent estimates for campaign rankings and sales estimates."""
    d = res.copy()

    use_adj = (d["PRETREND_OK"] == False) & d["ADJ_LIFT"].notna()

    d["BASIS"] = np.where(use_adj, "Trend-adjusted", "Raw DiD")
    d["USED_LIFT"] = np.where(
        use_adj, d["ADJ_LIFT"], d["LIFT_PER_HH_DAY"]
    )
    d["USED_LO"] = np.where(use_adj, d["ADJ_LO"], d["CI_LO"])
    d["USED_HI"] = np.where(use_adj, d["ADJ_HI"], d["CI_HI"])

    # Use the same lift estimate displayed in the scorecard.
    d["INCR_SALES"] = (
        d["USED_LIFT"] * d["DAYS"] * d["N_TREATED"]
    )

    # Recalculate the coupon comparison using the consistent sales estimate.
    if "COUPON_COST" in d.columns:
        d["INCR_SALES_PER_COUPON_$"] = np.where(
            d["COUPON_COST"] > 0,
            d["INCR_SALES"] / d["COUPON_COST"],
            np.nan
        )

    return d


def segment_pooled(seg, res):
    """Pool campaign-level segment lift by campaign TYPE x SEGMENT (inverse-variance)."""
    ok = res.loc[res["N_TREATED"] >= MIN_RECIPIENTS, ["CAMPAIGN", "TYPE"]]
    d = seg.merge(ok, on="CAMPAIGN").dropna(subset=["LIFT_PER_HH_DAY"]).copy()
    d["SE"] = (d["CI_HI"] - d["CI_LO"]) / 3.92
    d = d[d["SE"] > 0]
    if d.empty:
        return pd.DataFrame(columns=["TYPE", "SEGMENT", "CAMPAIGNS", "HOUSEHOLDS", "POOLED_LIFT", "CI_LO", "CI_HI"])
    d["W"] = 1 / d["SE"] ** 2
    rows = []
    for (t, s), x in d.groupby(["TYPE", "SEGMENT"]):
        lift = np.average(x["LIFT_PER_HH_DAY"], weights=x["W"])
        se = 1 / np.sqrt(x["W"].sum())
        rows.append(dict(TYPE=t, SEGMENT=s, CAMPAIGNS=len(x), HOUSEHOLDS=int(x["N_TREATED"].sum()),
                         POOLED_LIFT=lift, CI_LO=lift - 1.96 * se, CI_HI=lift + 1.96 * se))
    return pd.DataFrame(rows)
