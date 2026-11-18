import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

import streamlit as st
from dotenv import load_dotenv

import campaign_stats as cs

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", APP_DIR / "data" / "raw"))
KAGGLE_SLUG = "frtgnn/dunnhumby-the-complete-journey"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

st.set_page_config(
    page_title="CampaignIQ | Retail Campaign Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background: rgba(128,128,128,.07); padding: 14px; border-radius: 10px;}
    .small-note {font-size: .85rem; color: #777;}
</style>
""", unsafe_allow_html=True)

REQUIRED_FILES = [
    "transaction_data.csv", "hh_demographic.csv", "product.csv",
    "campaign_desc.csv", "campaign_table.csv", "coupon.csv",
    "coupon_redempt.csv"
]
OPTIONAL_FILES = ["causal_data.csv"]


def locate_csv(filename: str) -> Optional[Path]:
    # First, try the original CSV filename.
    direct = DATA_DIR / filename
    if direct.exists():
        return direct

    # Next, try a compressed version.
    compressed = DATA_DIR / f"{filename}.gz"
    if compressed.exists():
        return compressed

    # Finally, check other folders under the data directory.
    if DATA_DIR.exists():
        matches = list(DATA_DIR.rglob(filename))
        if matches:
            return matches[0]

        compressed_matches = list(
            DATA_DIR.rglob(f"{filename}.gz")
        )
        if compressed_matches:
            return compressed_matches[0]

    return None

def list_missing_files():
    return [f for f in REQUIRED_FILES if locate_csv(f) is None]


@st.cache_data(show_spinner=False)
def read_csv_cached(path_string: str, mtime: float) -> pd.DataFrame:
    # mtime in cache key makes updated files reload.
    return pd.read_csv(path_string, low_memory=False)


def load_table(filename: str, required: bool = True) -> Optional[pd.DataFrame]:
    path = locate_csv(filename)
    if path is None:
        if required:
            raise FileNotFoundError(f"Could not find {filename} under {DATA_DIR}")
        return None
    df = read_csv_cached(str(path), path.stat().st_mtime)
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def clean_id(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")


@st.cache_data(show_spinner="Loading and preparing the Dunnhumby tables…")
def load_core_data(data_dir_string: str, _refresh_token: int = 0):
    # Use local helper paths, while keeping the cache dependent on refresh token.
    global DATA_DIR
    DATA_DIR = Path(data_dir_string)

    tx = load_table("transaction_data.csv")
    hh = load_table("hh_demographic.csv")
    products = load_table("product.csv")
    campaigns = load_table("campaign_desc.csv")
    sent = load_table("campaign_table.csv")
    coupons = load_table("coupon.csv")
    redemptions = load_table("coupon_redempt.csv")

    for df, cols in [
        (tx, ["HOUSEHOLD_KEY", "PRODUCT_ID", "STORE_ID", "DAY", "WEEK_NO"]),
        (hh, ["HOUSEHOLD_KEY"]),
        (products, ["PRODUCT_ID"]),
        (campaigns, ["CAMPAIGN", "START_DAY", "END_DAY"]),
        (sent, ["HOUSEHOLD_KEY", "CAMPAIGN"]),
        (coupons, ["CAMPAIGN", "PRODUCT_ID"]),
        (redemptions, ["HOUSEHOLD_KEY", "CAMPAIGN", "DAY"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    tx["SALES_VALUE"] = pd.to_numeric(tx["SALES_VALUE"], errors="coerce").fillna(0)
    tx["DAY"] = pd.to_numeric(tx["DAY"], errors="coerce")
    tx["WEEK_NO"] = pd.to_numeric(tx["WEEK_NO"], errors="coerce")
    tx = tx.dropna(subset=["HOUSEHOLD_KEY", "DAY"])
    tx["HOUSEHOLD_KEY"] = tx["HOUSEHOLD_KEY"].astype(int)
    tx["DAY"] = tx["DAY"].astype(int)
    tx["PRODUCT_ID"] = pd.to_numeric(tx["PRODUCT_ID"], errors="coerce")
    tx["WEEK_NO"] = pd.to_numeric(tx["WEEK_NO"], errors="coerce")

    campaigns = campaigns.dropna(subset=["CAMPAIGN", "START_DAY", "END_DAY"]).copy()
    campaigns["CAMPAIGN"] = campaigns["CAMPAIGN"].astype(int)
    campaigns["START_DAY"] = campaigns["START_DAY"].astype(int)
    campaigns["END_DAY"] = campaigns["END_DAY"].astype(int)
    campaigns["DURATION_DAYS"] = campaigns["END_DAY"] - campaigns["START_DAY"] + 1

    sent = sent.dropna(subset=["HOUSEHOLD_KEY", "CAMPAIGN"]).copy()
    sent["HOUSEHOLD_KEY"] = sent["HOUSEHOLD_KEY"].astype(int)
    sent["CAMPAIGN"] = sent["CAMPAIGN"].astype(int)

    if "CAMPAIGN" in coupons.columns:
        coupons = coupons.dropna(subset=["CAMPAIGN"]).copy()
        coupons["CAMPAIGN"] = coupons["CAMPAIGN"].astype(int)
    if "CAMPAIGN" in redemptions.columns:
        redemptions = redemptions.dropna(subset=["CAMPAIGN"]).copy()
        redemptions["CAMPAIGN"] = redemptions["CAMPAIGN"].astype(int)
        redemptions["HOUSEHOLD_KEY"] = pd.to_numeric(redemptions["HOUSEHOLD_KEY"], errors="coerce")
        redemptions = redemptions.dropna(subset=["HOUSEHOLD_KEY"])
        redemptions["HOUSEHOLD_KEY"] = redemptions["HOUSEHOLD_KEY"].astype(int)

    # Household-day sales, including only observed household-days. Later period
    # averages divide by the full number of calendar-index days in the window.
    hh_day = tx.groupby(["HOUSEHOLD_KEY", "DAY"], as_index=False)["SALES_VALUE"].sum()

    return tx, hh, products, campaigns, sent, coupons, redemptions, hh_day


def campaign_overlap_recipients(campaign_id, start_day, end_day, campaigns, sent):
    """Households sent another campaign whose scheduled dates overlap this window."""
    overlapping_ids = campaigns.loc[
        (campaigns["CAMPAIGN"] != campaign_id)
        & (campaigns["START_DAY"] <= end_day)
        & (campaigns["END_DAY"] >= start_day),
        "CAMPAIGN"
    ].tolist()
    if not overlapping_ids:
        return set()
    return set(sent.loc[sent["CAMPAIGN"].isin(overlapping_ids), "HOUSEHOLD_KEY"].unique())


def period_spend(hh_day, households, start_day, end_day):
    days = max(1, end_day - start_day + 1)
    part = hh_day[
        (hh_day["DAY"] >= start_day) & (hh_day["DAY"] <= end_day)
        & (hh_day["HOUSEHOLD_KEY"].isin(households))
    ]
    totals = part.groupby("HOUSEHOLD_KEY")["SALES_VALUE"].sum()
    out = pd.DataFrame({"HOUSEHOLD_KEY": list(households)})
    out["PER_DAY_SPEND"] = out["HOUSEHOLD_KEY"].map(totals).fillna(0) / days
    return out


@st.cache_data(show_spinner="Calculating campaign metrics and comparison estimates…")
def build_campaign_metrics(_tx, _hh, _products, campaigns, sent, coupons, redemptions, hh_day, max_day):
    # DataFrames are passed with underscore names to avoid hashing huge inputs.
    records = []
    all_households = set(hh_day["HOUSEHOLD_KEY"].unique())
    all_sent = sent.groupby("CAMPAIGN")["HOUSEHOLD_KEY"].apply(set).to_dict()

    coupon_campaign_counts = {}
    coupon_category_counts = {}
    if "CAMPAIGN" in coupons.columns:
        coupon_campaign_counts = coupons.groupby("CAMPAIGN").size().to_dict()
        if "PRODUCT_ID" in coupons.columns and "PRODUCT_ID" in _products.columns:
            cprod = coupons[["CAMPAIGN", "PRODUCT_ID"]].dropna().drop_duplicates().merge(
                _products[["PRODUCT_ID"] + [c for c in ["DEPARTMENT", "COMMODITY_DESC"] if c in _products.columns]].drop_duplicates("PRODUCT_ID"),
                on="PRODUCT_ID", how="left"
            )
            category_col = "COMMODITY_DESC" if "COMMODITY_DESC" in cprod.columns else "DEPARTMENT"
            if category_col in cprod.columns:
                coupon_category_counts = cprod.groupby("CAMPAIGN")[category_col].nunique().to_dict()

    redemptions_count = redemptions.groupby("CAMPAIGN").size().to_dict() if "CAMPAIGN" in redemptions.columns else {}
    redemption_hh_count = (
        redemptions.groupby("CAMPAIGN")["HOUSEHOLD_KEY"].nunique().to_dict()
        if "CAMPAIGN" in redemptions.columns and "HOUSEHOLD_KEY" in redemptions.columns else {}
    )

    for row in campaigns.itertuples(index=False):
        cid = int(row.CAMPAIGN)
        start = int(row.START_DAY)
        end = min(int(row.END_DAY), int(max_day))   # data stops at max_day; do not divide by unobserved days
        duration = max(1, end - start + 1)
        recipients = all_sent.get(cid, set())
        n_recipients = len(recipients)

        during = period_spend(hh_day, recipients, start, end)
        pre_start = max(1, start - duration)
        pre_end = start - 1
        before = period_spend(hh_day, recipients, pre_start, pre_end) if pre_end >= pre_start else pd.DataFrame(columns=["HOUSEHOLD_KEY", "PER_DAY_SPEND"])
        pre_map = before.set_index("HOUSEHOLD_KEY")["PER_DAY_SPEND"] if len(before) else pd.Series(dtype=float)
        during_map = during.set_index("HOUSEHOLD_KEY")["PER_DAY_SPEND"] if len(during) else pd.Series(dtype=float)
        treated_change = float(during_map.mean() - pre_map.reindex(during_map.index).fillna(0).mean()) if len(during_map) else np.nan

        # Comparison group excludes recipients of the selected campaign and households
        # sent another campaign overlapping the pre/during observation window.
        overlap_recipients = campaign_overlap_recipients(cid, pre_start, end, campaigns, sent)
        control_hh = all_households - recipients - overlap_recipients
        control_during = period_spend(hh_day, control_hh, start, end)
        control_before = period_spend(hh_day, control_hh, pre_start, pre_end)
        cb = control_before.set_index("HOUSEHOLD_KEY")["PER_DAY_SPEND"]
        cd = control_during.set_index("HOUSEHOLD_KEY")["PER_DAY_SPEND"]
        common_control = cd.index.intersection(cb.index)
        control_change = float(cd.loc[common_control].mean() - cb.loc[common_control].mean()) if len(common_control) else np.nan

        did = treated_change - control_change if pd.notna(treated_change) and pd.notna(control_change) else np.nan
        rcount = int(redemptions_count.get(cid, 0))
        redeemed_hh = int(redemption_hh_count.get(cid, 0))
        records.append({
            "CAMPAIGN": cid,
            "TYPE": getattr(row, "DESCRIPTION", "Unknown"),
            "START_DAY": start,
            "END_DAY": end,
            "DURATION_DAYS": duration,
            "RECIPIENT_HOUSEHOLDS": n_recipients,
            "AVG_DAILY_SPEND_DURING": float(during["PER_DAY_SPEND"].mean()) if len(during) else np.nan,
            "AVG_DAILY_SPEND_BEFORE": float(pre_map.mean()) if len(pre_map) else np.nan,
            "TREATED_CHANGE_PER_HH_DAY": treated_change,
            "CONTROL_CHANGE_PER_HH_DAY": control_change,
            "DID_ESTIMATE_PER_HH_DAY": did,  # naive (no CI / pre-trend check)
            "COUPON_ROWS": int(coupon_campaign_counts.get(cid, 0)),
            "COUPON_CATEGORY_COUNT": int(coupon_category_counts.get(cid, 0)),
            "REDEMPTION_EVENTS": rcount,
            "REDEEMING_HOUSEHOLDS": redeemed_hh,
            "REDEEMING_HH_RATE": redeemed_hh / n_recipients if n_recipients else np.nan,
            "CONTROL_HOUSEHOLDS": len(control_hh),
        })
    return pd.DataFrame(records)


def analyze_store_promo_context(tx, recipient_households, start_day, end_day, causal_path):
    """Stream the large product-store-week promotion table and match campaign-period sales keys."""
    campaign_tx = tx[
        tx["HOUSEHOLD_KEY"].isin(recipient_households)
        & tx["DAY"].between(start_day, end_day)
    ].copy()
    key_cols = ["PRODUCT_ID", "STORE_ID", "WEEK_NO"]
    if not all(c in campaign_tx.columns for c in key_cols):
        raise ValueError("Transaction data is missing PRODUCT_ID, STORE_ID, or WEEK_NO.")
    keys = campaign_tx[key_cols].dropna().drop_duplicates()
    if keys.empty:
        return pd.DataFrame(), pd.DataFrame()

    matched = []
    usecols = ["PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"]
    for chunk in pd.read_csv(
        causal_path, usecols=usecols, chunksize=750_000,
        dtype={"display": "string", "mailer": "string"}, low_memory=False
    ):
        for c in key_cols:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        chunk = chunk.dropna(subset=key_cols)
        hit = chunk.merge(keys, on=key_cols, how="inner")
        if len(hit):
            matched.append(hit)
    if not matched:
        return pd.DataFrame(), pd.DataFrame()

    promo = pd.concat(matched, ignore_index=True).drop_duplicates(key_cols)
    promo["ON_DISPLAY"] = promo["display"].fillna("0").astype(str).str.strip().ne("0")
    promo["IN_MAILER"] = promo["mailer"].fillna("0").astype(str).str.strip().ne("0")

    sales = campaign_tx.groupby(key_cols, as_index=False)["SALES_VALUE"].sum()
    joined = sales.merge(promo[key_cols + ["display", "mailer", "ON_DISPLAY", "IN_MAILER"]], on=key_cols, how="left")
    joined["ON_DISPLAY"] = joined["ON_DISPLAY"].fillna(False)
    joined["IN_MAILER"] = joined["IN_MAILER"].fillna(False)
    summary = pd.DataFrame([
        {"Promotion condition": "On display", "Product-store-week combinations": int(joined["ON_DISPLAY"].sum()), "Net sales in matched combinations": float(joined.loc[joined["ON_DISPLAY"], "SALES_VALUE"].sum())},
        {"Promotion condition": "Not on display", "Product-store-week combinations": int((~joined["ON_DISPLAY"]).sum()), "Net sales in matched combinations": float(joined.loc[~joined["ON_DISPLAY"], "SALES_VALUE"].sum())},
        {"Promotion condition": "In mailer", "Product-store-week combinations": int(joined["IN_MAILER"].sum()), "Net sales in matched combinations": float(joined.loc[joined["IN_MAILER"], "SALES_VALUE"].sum())},
        {"Promotion condition": "Not in mailer", "Product-store-week combinations": int((~joined["IN_MAILER"]).sum()), "Net sales in matched combinations": float(joined.loc[~joined["IN_MAILER"], "SALES_VALUE"].sum())},
    ])
    return summary, joined


def make_ai_client(api_key):
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the Gemini client. Install the google-genai package "
            "and verify your API key. Details: " + str(exc)
        )


# ---------------- Evidence layer (matched DiD, CIs, placebo, segments) ----------------
INCOME_BANDS = {
    "Under 15K": "Low (<$35K)", "15-24K": "Low (<$35K)", "25-34K": "Low (<$35K)",
    "35-49K": "Mid ($35-74K)", "50-74K": "Mid ($35-74K)",
    "75-99K": "High ($75K+)", "100-124K": "High ($75K+)", "125-149K": "High ($75K+)",
    "150-174K": "High ($75K+)", "175-199K": "High ($75K+)", "200-249K": "High ($75K+)", "250K+": "High ($75K+)",
}
AGE_BANDS = {"19-24": "Under 35", "25-34": "Under 35", "35-44": "35-54", "45-54": "35-54", "55-64": "55+", "65+": "55+"}
SEGMENT_LABELS = {
    "INCOME_BAND": "Income (3 bands)", "AGE_BAND": "Age (3 bands)", "KIDS": "Children in household",
    "HOMEOWNER_DESC": "Home ownership", "HOUSEHOLD_SIZE_DESC": "Household size",
}


def add_segment_bands(hh):
    """Coarser groupings so segment cells have enough households to be meaningful."""
    hh = hh.copy()
    if "INCOME_DESC" in hh.columns:
        hh["INCOME_BAND"] = hh["INCOME_DESC"].map(INCOME_BANDS)
    if "AGE_DESC" in hh.columns:
        hh["AGE_BAND"] = hh["AGE_DESC"].map(AGE_BANDS)
    if "KID_CATEGORY_DESC" in hh.columns:
        hh["KIDS"] = np.where(hh["KID_CATEGORY_DESC"].astype(str).str.contains("None", case=False), "No kids / unknown", "Has kids")
    return hh


@st.cache_data(show_spinner="Building spend matrix…")
def get_matrix(_tx, token=0):
    return cs.build_matrix(_tx)


@st.cache_data(show_spinner="Running matched difference-in-differences with bootstrap intervals and placebo tests…")
def get_evidence(_tx, campaigns, sent, _hh, redemptions, demo_field, token=0):
    res, seg = cs.run_all(_tx, campaigns, sent, _hh, demo_field, redemptions)
    return cs.decision_table(res), seg


def short_verdict(v):
    return str(v).split(" (")[0]


VERDICT_COLORS = {"Repeat": "#2e9d5b", "Retest": "#e0a21b", "Drop": "#d64545", "Insufficient data": "#9aa0a6"}

# Client-facing labels; underlying classification values stay unchanged.
VERDICT_LABELS = {
    "Repeat": "Repeat candidate",
    "Retest": "Retest recommended",
    "Drop": "Do not prioritize",
    "Insufficient data": "Insufficient sample"
}


def rule_based_brief(dec, type_pool):
    """Plain-English CMO brief built straight from the evidence table (works without an AI key)."""
    d = dec.copy()
    d["V"] = d["VERDICT"].map(short_verdict)
    rep = d[d["V"] == "Repeat"].sort_values("USED_LO", ascending=False)
    lines = []
    lines.append(f"**Bottom line.** Of {len(d)} campaigns, {len(rep)} show a lift that stays above zero after correcting for "
                 f"how targeted households were already trending; {int((d['V']=='Retest').sum())} need a retest "
                 f"(the effect cannot be told apart from zero), {int((d['V']=='Drop').sum())} look like drops, and "
                 f"{int((d['V']=='Insufficient data').sum())} were sent to too few households to judge.")
    if len(rep):
        top = ", ".join(f"#{int(r.CAMPAIGN)} (Type {str(r.TYPE).replace('Type','')}, +${r.USED_LIFT:.2f}/household/day)" for r in rep.head(3).itertuples())
        lines.append(f"**Strongest candidates to repeat:** {top}. Ranked by the conservative end of the interval.")
    if len(type_pool):
        best = type_pool.sort_values("POOLED_LIFT", ascending=False).iloc[0]
        lines.append(f"**Campaign type signal:** {best['TYPE']} has the highest pooled lift (${best['POOLED_LIFT']:.2f}/household/day, "
                     f"95% CI ${best['CI_LO']:.2f} to ${best['CI_HI']:.2f}); pooled figures are not adjusted for pre-campaign trends, so treat them as upper bounds.")
    fail = d[d["PRETREND_OK"] == False]  # noqa: E712
    if len(fail):
        lines.append(f"**Why caution:** {len(fail)} campaigns reached households whose spending was already rising before launch, "
                     "so raw before/after numbers overstate impact. Verdicts for those use a trend-adjusted estimate.")
    lines.append("**What this cannot prove:** campaigns were targeted, not randomised. Weather, paydays, competitor activity and "
                 "store promotions are not measured, and overlapping campaigns are only handled by excluding exposed households from the comparison group.")
    return "\n\n".join(lines)


def ask_gemini(api_key, model, question, context, history=None):
    client = make_ai_client(api_key)
    hist = ""
    if history:
        hist = "\nPREVIOUS CONVERSATION (for follow-ups):\n" + "\n".join(f"{r.upper()}: {t}" for r, t in history[-6:])
    prompt = f"""
You are CampaignIQ, a careful retail marketing analytics advisor working for the grocer's CMO,
who must decide which of 30 direct-mail campaigns to run again.
Use ONLY the supplied evidence. Do not invent metrics, customer facts, campaign attributes or causal conclusions.
Distinguish descriptive association from causal evidence. Lift figures come from an observational matched
difference-in-differences with bootstrap 95% intervals and a placebo (pre-trend) test. If an interval includes zero,
say the evidence cannot separate the campaign from noise. If a campaign is 'Insufficient data', do not rank it.
Segment results are unadjusted for pre-trend and cells are suppressed when too small; say so when you cite them.
Never claim to have controlled for factors that were not measured (weather, paydays, competitors).

EVIDENCE:
{context}
{hist}

USER QUESTION:
{question}

Respond with:
1. Short answer (decision-oriented, 2-3 sentences)
2. Evidence from the supplied metrics (cite campaign numbers and intervals)
3. What the evidence cannot establish
4. Recommended next action
"""
    response = client.models.generate_content(model=model, contents=prompt)
    return getattr(response, "text", str(response))


# ---------------- Sidebar and data setup ----------------
st.sidebar.title("🛒 CampaignIQ")
st.sidebar.caption("AI-powered retail campaign intelligence")
st.sidebar.markdown("---")

with st.sidebar.expander("Data setup", expanded=False):
    st.write(f"**Data folder:** `{DATA_DIR}`")
    st.caption("Place the eight Dunnhumby CSV files in this folder, or download them below.")
    if st.button("Download dataset from Kaggle", use_container_width=True):
        with st.spinner("Downloading the dataset with KaggleHub. This can take several minutes…"):
            try:
                import kagglehub
                downloaded_path = Path(kagglehub.dataset_download(KAGGLE_SLUG))
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                copied = []
                for filename in REQUIRED_FILES + OPTIONAL_FILES:
                    matches = list(downloaded_path.rglob(filename))
                    if matches:
                        shutil.copy2(matches[0], DATA_DIR / filename)
                        copied.append(filename)
                st.success(f"Downloaded and copied {len(copied)} CSV files into {DATA_DIR}")
                st.write("Files copied:", ", ".join(copied) if copied else "No expected CSV files found")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Download failed: {exc}")
                st.caption("You may need to accept Kaggle's dataset terms or configure Kaggle authentication.")

st.sidebar.markdown("---")
st.sidebar.subheader("Gemini AI")
api_key = st.sidebar.text_input(
    "Gemini API key",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password",
    help="Use a newly rotated key. Do not hardcode API keys in app.py."
)
model_name = st.sidebar.text_input("Gemini model", value=DEFAULT_MODEL)
st.sidebar.caption("AI uses aggregate campaign metrics, not raw household transaction rows.")

missing = list_missing_files()
if missing:
    st.title("🛒 CampaignIQ")
    st.subheader("Retail Campaign Intelligence")
    st.warning("The required Dunnhumby CSV files were not found.")
    st.write("Expected files:")
    for f in REQUIRED_FILES:
        st.write(("❌ " if f in missing else "✅ ") + f)
    st.markdown(f"**Current data folder:** `{DATA_DIR}`")
    st.markdown("""
**How to get started**
1. Run `pip install -r requirements.txt`.
2. Click **Download dataset from Kaggle** in the sidebar, or download the CSV archive manually.
3. Put the CSV files in `data/raw/` (or set `DATA_DIR` to their folder).
4. Restart Streamlit.
""")
    st.stop()

refresh_token = st.session_state.get("refresh_token", 0)
try:
    tx, hh, products, campaigns, sent, coupons, redemptions, hh_day = load_core_data(str(DATA_DIR), refresh_token)
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

max_day = int(tx["DAY"].max())
hh = add_segment_bands(hh)
metrics_legacy = build_campaign_metrics(tx, hh, products, campaigns, sent, coupons, redemptions, hh_day, max_day)
hh_ids, CMAT, _ = get_matrix(tx, refresh_token)
dec, _seg0 = get_evidence(tx, campaigns, sent, hh, redemptions, None, refresh_token)
type_pool = cs.pool_by_type(dec)

EVID_COLS = ["CAMPAIGN", "N_TREATED", "N_CONTROL", "LIFT_PER_HH_DAY", "CI_LO", "CI_HI", "PLACEBO", "PRETREND_OK",
             "BASIS", "USED_LIFT", "USED_LO", "USED_HI", "LIFT_PCT", "INCR_SALES", "COUPON_COST",
             "INCR_SALES_PER_COUPON_$", "VERDICT", "END_TRUNCATED"]
metrics = metrics_legacy.merge(dec[[c for c in EVID_COLS if c in dec.columns]], on="CAMPAIGN", how="left")
metrics["VERDICT_SHORT"] = metrics["VERDICT"].fillna("Insufficient data").map(short_verdict)

# ---------------- Header ----------------
st.title("🛒 CampaignIQ")
st.markdown("### Which of the 30 promotional campaigns deserve a rerun?")
st.info(
    "Data Coverage Note: The dataset contains 30 promotional campaigns, "
    "but only 29 can be fully evaluated. Campaign 24 is excluded from "
    "performance analysis because it ends on Day 719, while transaction "
    "data ends on Day 711."
)
st.caption(
    "**Client:** the grocer's Chief Marketing Officer. **Pain point:** direct-mail and coupon campaigns cost money, but raw sales during a campaign "
    "mix real lift with seasonality, overlapping promotions and the fact that campaigns are sent to households already spending more. "
    "CampaignIQ separates the two and ends with a Repeat / Retest / Drop call for every campaign."
)
st.info("The dataset uses sequential DAY and WEEK_NO indices, not real calendar dates. All spend comparisons use those indices.")

total_sales = float(tx["SALES_VALUE"].sum())
households = int(tx["HOUSEHOLD_KEY"].nunique())
n_campaigns = int(campaigns["CAMPAIGN"].nunique())
total_transactions = int(tx["BASKET_ID"].nunique()) if "BASKET_ID" in tx.columns else len(tx)
demographic_coverage = hh["HOUSEHOLD_KEY"].nunique() / max(1, households) if "HOUSEHOLD_KEY" in hh.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Net sales in file", f"${total_sales:,.0f}")
c2.metric("Households", f"{households:,}")
c3.metric("Campaigns", f"{n_campaigns}")
c4.metric("Households with demographics", f"{demographic_coverage:.1%}")
st.caption(f"{len(tx):,} transaction line items · {total_transactions:,} baskets · Demographics are available for a subset of households.")

tab_decide, tab_overview, tab_campaigns, tab_demo, tab_impact, tab_plan, tab_ai = st.tabs([
    "✅ Decision Scorecard",
    "📊 Executive Overview",
    "🎯 Campaign Intelligence",
    "👥 Demographics",
    "📈 Spending & Impact",
    "🧭 Next-Campaign Planner",
    "🤖 AI Marketing Advisor",
])


def fmt_money(x, nd=2):
    return "N/A" if pd.isna(x) else f"${x:,.{nd}f}"


# ================= Decision Scorecard (primary task) =================
with tab_decide:
    st.subheader("Repeat, retest or drop: the evidence for each campaign")
    counts = metrics.loc[
    metrics["USED_LIFT"].notna(), "VERDICT_SHORT"
    ].value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(VERDICT_LABELS["Repeat"], int(counts.get("Repeat", 0)))
    k2.metric(VERDICT_LABELS["Retest"], int(counts.get("Retest", 0)))
    k3.metric(VERDICT_LABELS["Drop"], int(counts.get("Drop", 0)))
    k4.metric(
        VERDICT_LABELS["Insufficient data"],
        int(counts.get("Insufficient data", 0))
    )

    show_small = st.checkbox("Show campaigns with fewer than 50 recipient households", value=False)
    fp = metrics.dropna(subset=["USED_LIFT"]).copy()
    if not show_small:
        fp = fp[fp["N_TREATED"] >= cs.MIN_RECIPIENTS]
    fp = fp.sort_values("USED_LO")
    fp["LABEL"] = fp.apply(lambda r: f"Campaign {int(r.CAMPAIGN)} · {str(r.TYPE).replace('Type', 'Type ')}", axis=1)

    # Use client-friendly verdict labels in the chart legend.
    fp["VERDICT_DISPLAY"] = (
        fp["VERDICT_SHORT"]
        .map(VERDICT_LABELS)
        .fillna(fp["VERDICT_SHORT"])
    )

    DISPLAY_VERDICT_COLORS = {
        VERDICT_LABELS[key]: color
        for key, color in VERDICT_COLORS.items()
    }

    fp["ERR_PLUS"] = fp["USED_HI"] - fp["USED_LIFT"]
    fp["ERR_MINUS"] = fp["USED_LIFT"] - fp["USED_LO"]
    fig = px.scatter(
        fp, x="USED_LIFT", y="LABEL", color="VERDICT_DISPLAY", color_discrete_map=DISPLAY_VERDICT_COLORS,
        error_x="ERR_PLUS", error_x_minus="ERR_MINUS", size="N_TREATED", size_max=16,
        hover_data={"N_TREATED": True, "BASIS": True, "PLACEBO": ":.2f", "USED_LO": ":.2f", "USED_HI": ":.2f", "LABEL": False},
        title="Estimated lift in daily spend per recipient household, with 95% interval",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title="Lift vs comparison households (USD per household per day)", yaxis_title="", legend_title="Screening recommendation",
                      height=max(420, 26 * len(fp) + 160))
    fig.update_yaxes(categoryorder="array", categoryarray=fp["LABEL"].tolist())
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A bar that crosses the dashed zero line means the campaign cannot be told apart from noise. Dot size = households sent. "
               "Where targeted households were already diverging before launch, the trend-adjusted estimate is plotted.")

    st.markdown("**Scorecard**")

    # Keep only campaigns with a valid calculated result.
    score = metrics.loc[
        metrics["USED_LIFT"].notna()
    ].sort_values("USED_LO", ascending=False, na_position="last")[[
        "CAMPAIGN", "TYPE", "N_TREATED", "USED_LIFT", "USED_LO", "USED_HI", "BASIS", "PLACEBO", "PRETREND_OK",
        "INCR_SALES", "COUPON_COST", "INCR_SALES_PER_COUPON_$", "VERDICT"]].copy()
    
    score.columns = [
        "Campaign",
        "Type",
        "Households",
        "Estimated difference ($/hh/day)",
        "95% CI lower",
        "95% CI upper",
        "Estimate basis",
        "Pre-campaign gap (placebo)",
        "Pre-trend check",
        "Estimated sales difference ($)",
        "Coupon-cost proxy ($)",
        "Sales difference / coupon-cost proxy",
        "Screening verdict"
    ]

    # Display client-friendly recommendations without changing the underlying classifications.
    score["Screening verdict"] = (
        score["Screening verdict"]
        .apply(short_verdict)
        .map(VERDICT_LABELS)
        .fillna("Insufficient sample")
    )

    st.dataframe(score.round(2), use_container_width=True, hide_index=True)

    st.caption(
        "Campaign 24 is excluded from the analysis because its scheduled "
        "end date (Day 719) exceeds the available transaction data (Day 711)."
)


    with st.expander("How to read this, and what it cannot tell you", expanded=False):
        st.markdown(
            "- **Comparison group:** households not sent this campaign and not sent any campaign overlapping the analysis window, reweighted to match recipients' pre-campaign spending level.\n"
            "- **Interval:** bootstrap over households (95%).\n"
            "- **Placebo test:** the same calculation one window earlier. If it is not near zero, recipients were already pulling away from the comparison group, "
            "so the raw before/after lift is biased upward and the verdict uses a trend-adjusted figure (wider interval).\n"
            "- **Minimum sample:** campaigns sent to fewer than 50 households are not ranked.\n"
            "- **Coupon cost proxy:** retailer-funded coupon discount on household-days when that campaign's coupons were redeemed. Incremental *sales*, not margin.\n"
            "- **Not proven:** campaigns were targeted, not randomised. Weather, paydays and competitor actions are not measured."
        )

    st.markdown("**CMO brief (generated from the table above, no AI key needed)**")
    st.markdown(rule_based_brief(dec, type_pool))

# ================= Executive Overview =================
with tab_overview:
    st.subheader("Campaign performance at a glance")
    rank_options = ["USED_LO", "USED_LIFT", "DID_ESTIMATE_PER_HH_DAY", "AVG_DAILY_SPEND_DURING", "REDEEMING_HH_RATE", "RECIPIENT_HOUSEHOLDS"]
    rank_metric = st.selectbox(
        "Rank campaigns by", rank_options,
        format_func=lambda x: {
            "USED_LO": "Conservative lift (lower end of 95% interval)",
            "USED_LIFT": "Estimated lift (matched DiD, trend-adjusted where needed)",
            "DID_ESTIMATE_PER_HH_DAY": "Naive incremental daily spend (no interval, no pre-trend check)",
            "AVG_DAILY_SPEND_DURING": "Average daily spend during campaign",
            "REDEEMING_HH_RATE": "Redeeming-household rate",
            "RECIPIENT_HOUSEHOLDS": "Number of recipient households",
        }[x]
    )
    hide_small = st.checkbox("Hide campaigns sent to fewer than 50 households", value=True, key="hide_small_overview")
    pool = metrics.copy()
    if hide_small:
        pool = pool[pool["RECIPIENT_HOUSEHOLDS"] >= cs.MIN_RECIPIENTS]
    ranked = pool.sort_values(rank_metric, ascending=False, na_position="last").head(12)
    if rank_metric == "REDEEMING_HH_RATE":
        ytitle = "Redeeming-household rate"
    elif rank_metric in ["USED_LO", "USED_LIFT", "DID_ESTIMATE_PER_HH_DAY", "AVG_DAILY_SPEND_DURING"]:
        ytitle = "USD per household per day"
    else:
        ytitle = "Households"
    ranked = ranked.assign(CAMPAIGN_LABEL=ranked["CAMPAIGN"].astype(int).astype(str))
    fig = px.bar(
        ranked, x="CAMPAIGN_LABEL", y=rank_metric, color="TYPE",
        hover_data=["DURATION_DAYS", "RECIPIENT_HOUSEHOLDS", "REDEMPTION_EVENTS"],
        title="Top 12 campaigns by selected measure"
    )
    fig.update_layout(xaxis_title="Campaign", yaxis_title=ytitle, legend_title="Campaign type")
    st.plotly_chart(fig, use_container_width=True)

    
    ranking_explanations = {
       
        "USED_LO": (
            "Campaigns are ranked by the lower bound of their estimated "
            "95% interval. This favors campaigns with stronger evidence "
            "of positive spending differences, but does not prove causation."
    ),
        "USED_LIFT": (
        "Campaigns are ranked by their estimated spending difference "
        "per household per day, using trend-adjusted estimates when "
        "the pre-trend check fails."
    ),
        "DID_ESTIMATE_PER_HH_DAY": (
        "Campaigns are ranked using the original difference-in-differences "
        "estimate, without trend adjustment or uncertainty screening. "
        "Interpret these results cautiously."
    ),
        "AVG_DAILY_SPEND_DURING": (
        "Campaigns are ranked by average daily spending during their "
        "campaign window. Higher spending does not necessarily mean "
        "that the campaign caused an increase."
    ),
        "REDEEMING_HH_RATE": (
        "Campaigns are ranked by the percentage of recipient households "
        "that redeemed at least one coupon."
    ),
    "RECIPIENT_HOUSEHOLDS": (
        "Campaigns are ranked by the number of recipient households. "
        "A larger audience does not necessarily mean better performance."
    ),
    }

    st.caption(ranking_explanations.get(
        rank_metric,
            "Rankings are exploratory and do not establish causal impact."
    ))

    st.subheader("Campaign performance table")
    display_cols = [
        "CAMPAIGN", "TYPE", "DURATION_DAYS", "RECIPIENT_HOUSEHOLDS",
        "AVG_DAILY_SPEND_DURING", "TREATED_CHANGE_PER_HH_DAY",
        "CONTROL_CHANGE_PER_HH_DAY", "DID_ESTIMATE_PER_HH_DAY", "USED_LIFT", "USED_LO", "USED_HI",
        "COUPON_ROWS", "REDEMPTION_EVENTS", "REDEEMING_HH_RATE", "VERDICT"
    ]
    st.dataframe(metrics[display_cols].sort_values("CAMPAIGN"), use_container_width=True, hide_index=True)

# ================= Campaign Intelligence =================
with tab_campaigns:
    st.subheader("Campaign profiles and traits")
    campaign_ids = sorted(metrics["CAMPAIGN"].dropna().astype(int).unique().tolist())
    selected_campaign = st.selectbox("Select a campaign", campaign_ids, key="campaign_profile")
    m = metrics.loc[metrics["CAMPAIGN"] == selected_campaign].iloc[0]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Campaign type", str(m["TYPE"]))
    p2.metric("Duration", f"{int(m['DURATION_DAYS'])} days")
    p3.metric("Households sent", f"{int(m['RECIPIENT_HOUSEHOLDS']):,}")
    p4.metric("Coupon redemptions", f"{int(m['REDEMPTION_EVENTS']):,}")

    st.markdown("**Measured campaign characteristics**")
    profile_cols = st.columns(3)
    profile_cols[0].metric("Coupon-product rows", f"{int(m['COUPON_ROWS']):,}")
    profile_cols[1].metric("Distinct coupon product categories", f"{int(m['COUPON_CATEGORY_COUNT']):,}")
    profile_cols[2].metric("Redeeming-household rate", "N/A" if pd.isna(m["REDEEMING_HH_RATE"]) else f"{m['REDEEMING_HH_RATE']:.1%}")

    st.markdown("**Did these traits go with better results? Evidence for this campaign**")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Lift per household-day", fmt_money(m["USED_LIFT"]),
              help="Matched difference-in-differences; trend-adjusted when the placebo test fails.")
    e2.metric("95% interval", "N/A" if pd.isna(m["USED_LO"]) else f"{fmt_money(m['USED_LO'])} to {fmt_money(m['USED_HI'])}")
    e3.metric("Incremental sales per coupon $", "N/A" if pd.isna(m.get("INCR_SALES_PER_COUPON_$")) else f"{m['INCR_SALES_PER_COUPON_$']:.1f}x")
    e4.metric("Verdict", short_verdict(m["VERDICT"]) if pd.notna(m["VERDICT"]) else "Insufficient data")

    st.markdown("**Across all campaigns: which trait lines up with lift?**")
    trait_options = {
        "DURATION_DAYS": "Duration (days)", "RECIPIENT_HOUSEHOLDS": "Households sent",
        "COUPON_ROWS": "Coupon-product rows", "COUPON_CATEGORY_COUNT": "Distinct coupon categories",
        "REDEEMING_HH_RATE": "Redeeming-household rate",
    }
    trait = st.selectbox("Campaign trait", list(trait_options), format_func=lambda x: trait_options[x], key="trait_pick")
    tp = metrics[(metrics["N_TREATED"] >= cs.MIN_RECIPIENTS)].dropna(subset=["USED_LIFT", trait]).copy()
    if len(tp):
        tp["ERR_PLUS"] = tp["USED_HI"] - tp["USED_LIFT"]
        tp["ERR_MINUS"] = tp["USED_LIFT"] - tp["USED_LO"]
        fig = px.scatter(tp, x=trait, y="USED_LIFT", color="TYPE", size="N_TREATED", error_y="ERR_PLUS", error_y_minus="ERR_MINUS",
                         hover_data=["CAMPAIGN", "VERDICT"], title=f"Lift vs {trait_options[trait].lower()} (campaigns with at least 50 households)")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_title=trait_options[trait], yaxis_title="Lift (USD per household per day)")
        st.plotly_chart(fig, use_container_width=True)
        if len(tp) >= 5:
            r = float(np.corrcoef(tp[trait].astype(float), tp["USED_LIFT"].astype(float))[0, 1])
            st.caption(f"Correlation across {len(tp)} campaigns: {r:+.2f}. With this few campaigns, treat it as a lead to investigate, not a rule.")
    st.markdown("**Pooled lift by campaign type** (inverse-variance weighted; pooling gives far tighter estimates than single campaigns)")
    tpool = type_pool.copy()
    if len(tpool):
        tpool["ERR"] = tpool["CI_HI"] - tpool["POOLED_LIFT"]
        fig = px.bar(tpool, x="TYPE", y="POOLED_LIFT", error_y="ERR", color="TYPE", text="CAMPAIGNS",
                     title="Pooled lift per household-day by campaign type (95% interval)")
        fig.update_layout(xaxis_title="Campaign type", yaxis_title="USD per household per day", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bar labels show the number of campaigns pooled. Pooled figures use the raw DiD, so they are upper bounds where pre-trends differ.")

    st.markdown("**Products and categories associated with this campaign's coupons**")
    if "CAMPAIGN" in coupons.columns and "PRODUCT_ID" in coupons.columns:
        cproducts = coupons.loc[coupons["CAMPAIGN"] == selected_campaign, ["PRODUCT_ID"]].dropna().drop_duplicates()
        category_cols = [c for c in ["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND"] if c in products.columns]
        cproducts = cproducts.merge(products[category_cols].drop_duplicates("PRODUCT_ID"), on="PRODUCT_ID", how="left")
        if len(cproducts):
            cat_col = "COMMODITY_DESC" if "COMMODITY_DESC" in cproducts.columns else ("DEPARTMENT" if "DEPARTMENT" in cproducts.columns else None)
            if cat_col:
                cat_counts = cproducts[cat_col].fillna("Unknown").value_counts().head(15).rename_axis("Category").reset_index(name="Coupon-linked products")
                fig = px.bar(cat_counts, x="Coupon-linked products", y="Category", orientation="h", title="Top coupon-linked product categories")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cproducts.head(100), use_container_width=True, hide_index=True)
        else:
            st.info("No product mappings were found for this campaign in coupon.csv.")
    else:
        st.info("Coupon table does not contain the expected CAMPAIGN and PRODUCT_ID columns.")

    st.markdown("**In-store display and mailer context (optional; large file)**")
    causal_path = locate_csv("causal_data.csv")
    if causal_path is None:
        st.info("causal_data.csv was not found. The app can still analyze campaign coupons, redemptions, products, and household spending.")
    else:
        st.caption("This table records product × store × week promotional placement. It is context for whether a campaign window coincided with heavy in-store promotion, "
                   "not proof that a specific direct-mail campaign caused a sale.")
        if st.button("Analyze display and mailer context for this campaign", key="promo_context_btn"):
            recipients_for_promo = set(sent.loc[sent["CAMPAIGN"] == selected_campaign, "HOUSEHOLD_KEY"].unique())
            with st.spinner("Streaming the large causal_data.csv file and matching campaign-period product/store/week records…"):
                try:
                    promo_summary, promo_joined = analyze_store_promo_context(
                        tx, recipients_for_promo, int(m["START_DAY"]), int(m["END_DAY"]), causal_path
                    )
                    st.session_state["promo_summary"] = promo_summary
                    st.session_state["promo_joined"] = promo_joined
                    st.session_state["promo_campaign_id"] = selected_campaign
                except Exception as exc:
                    st.error(f"Could not analyze promotion context: {exc}")
        if st.session_state.get("promo_campaign_id") == selected_campaign and "promo_summary" in st.session_state:
            ps = st.session_state["promo_summary"]
            pj = st.session_state["promo_joined"]
            if len(ps):
                st.dataframe(ps, use_container_width=True, hide_index=True)
                share_disp = float(ps.loc[ps["Promotion condition"] == "On display", "Net sales in matched combinations"].iloc[0]) / max(1e-9, float(pj["SALES_VALUE"].sum()))
                share_mail = float(ps.loc[ps["Promotion condition"] == "In mailer", "Net sales in matched combinations"].iloc[0]) / max(1e-9, float(pj["SALES_VALUE"].sum()))
                st.caption(f"Matched {len(pj):,} product-store-week combinations. {share_disp:.0%} of recipient sales in the window were on in-store display and "
                           f"{share_mail:.0%} were in the store mailer: a high share means in-store promotion is a competing explanation for any lift.")
            else:
                st.info("No matching product-store-week promotional records were found for this campaign window.")

# ================= Demographics =================
with tab_demo:
    st.subheader("Which campaigns work for which customer segments?")
    seg_fields = [f for f in SEGMENT_LABELS if f in hh.columns]
    if "HOUSEHOLD_KEY" not in hh.columns or not seg_fields:
        st.error("No usable demographic columns were found.")
    else:
        demo_field = st.selectbox("Choose a demographic characteristic", seg_fields, format_func=lambda x: SEGMENT_LABELS[x], key="demo_field_pick")
        dec_seg, seg = get_evidence(tx, campaigns, sent, hh, redemptions, demo_field, refresh_token)
        st.markdown("**1. Lift by segment: did the campaign change spending for this group?**")
        st.caption(f"Each cell compares recipients in the segment with comparison households in the same segment. Cells with fewer than {cs.MIN_CELL} recipients or comparison households are blanked.")
        sl = seg.dropna(subset=["LIFT_PER_HH_DAY"]).copy()
        if len(sl):
            sl["Campaign"] = sl["CAMPAIGN"].astype(int).map(lambda x: f"Campaign {x:02d}")
            piv = sl.pivot(index="Campaign", columns="SEGMENT", values="LIFT_PER_HH_DAY").sort_index()
            fig = px.imshow(piv, aspect="auto", color_continuous_scale="RdBu", color_continuous_midpoint=0, text_auto=".1f",
                            labels=dict(x=SEGMENT_LABELS[demo_field], y="Campaign", color="Lift $/hh/day"),
                            title=f"Lift in daily spend by campaign and {SEGMENT_LABELS[demo_field].lower()}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Blue = spending rose more than comparable households; red = less. Segment lifts are not adjusted for pre-campaign trends, so read them as upper bounds. Blank = too few households.")
            with st.expander("Households behind each cell (recipients / comparison)"):
                nn = seg.copy()
                nn["Campaign"] = nn["CAMPAIGN"].astype(int).map(lambda x: f"Campaign {x:02d}")
                nn["n"] = nn["N_TREATED"].astype(str) + " / " + nn["N_CONTROL"].astype(str)
                st.dataframe(nn.pivot(index="Campaign", columns="SEGMENT", values="n").sort_index(), use_container_width=True)
        else:
            st.warning("No segment had enough recipients and comparison households for this characteristic. Try a coarser characteristic.")

        st.markdown("**2. Representation: who did each campaign actually reach?**")
        base = hh[["HOUSEHOLD_KEY", demo_field]].dropna()
        base_share = base[demo_field].value_counts(normalize=True)
        rep_rows = []
        for cr in campaigns.itertuples(index=False):
            rec = sent.loc[sent["CAMPAIGN"] == int(cr.CAMPAIGN), "HOUSEHOLD_KEY"].unique()
            rec_demo = base[base["HOUSEHOLD_KEY"].isin(rec)]
            if len(rec_demo) < 20:
                continue
            shares = rec_demo[demo_field].value_counts(normalize=True)
            for sgm, bs in base_share.items():
                rep_rows.append({"Campaign": f"Campaign {int(cr.CAMPAIGN):02d}", "Segment": sgm, "Index": 100 * shares.get(sgm, 0) / bs})
        rep = pd.DataFrame(rep_rows)
        if len(rep):
            rp = rep.pivot(index="Campaign", columns="Segment", values="Index").sort_index()
            fig = px.imshow(rp, aspect="auto", color_continuous_scale="PuOr", color_continuous_midpoint=100, text_auto=".0f",
                            labels=dict(x=SEGMENT_LABELS[demo_field], y="Campaign", color="Index"),
                            title="Targeting index: 100 = same share as the whole customer base")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Above 100 means the campaign over-reached that group; below 100 under-reached it. Only households with a demographic record are counted, and campaigns with under 20 such households are omitted.")
        cov = pd.DataFrame({
            "Campaign": campaigns["CAMPAIGN"].astype(int),
            "Recipients": [int(sent.loc[sent["CAMPAIGN"] == int(c), "HOUSEHOLD_KEY"].nunique()) for c in campaigns["CAMPAIGN"]],
            "Recipients with demographics": [int(sent.loc[sent["CAMPAIGN"] == int(c), "HOUSEHOLD_KEY"].isin(hh["HOUSEHOLD_KEY"]).sum()) for c in campaigns["CAMPAIGN"]],
        }).sort_values("Campaign")
        cov["Coverage"] = (cov["Recipients with demographics"] / cov["Recipients"].replace(0, np.nan)).round(2)
        with st.expander("Demographic coverage per campaign"):
            st.dataframe(cov, use_container_width=True, hide_index=True)
        st.metric("Households with a demographic record", f"{hh['HOUSEHOLD_KEY'].nunique():,} of {households:,}")
        st.caption("Demographic records cover only a subset of households. Segment findings describe that subset only.")

        with st.expander("3. Spending level by segment (descriptive only: mostly reflects income and household size)"):
            selected_demo = hh[["HOUSEHOLD_KEY", demo_field]].dropna().copy()
            selected_demo[demo_field] = selected_demo[demo_field].astype(str)
            heat_records = []
            for cr in campaigns.itertuples(index=False):
                cid = int(cr.CAMPAIGN)
                recips = set(sent.loc[sent["CAMPAIGN"] == cid, "HOUSEHOLD_KEY"].unique())
                start, end = int(cr.START_DAY), min(int(cr.END_DAY), max_day)
                per_hh = period_spend(hh_day, recips, start, end).merge(selected_demo, on="HOUSEHOLD_KEY", how="inner")
                if len(per_hh):
                    for segment, value in per_hh.groupby(demo_field)["PER_DAY_SPEND"].mean().items():
                        heat_records.append({"Campaign": f"Campaign {cid}", "Segment": segment, "Avg daily spend": value})
            heat = pd.DataFrame(heat_records)
            if len(heat):
                pivot = heat.pivot(index="Campaign", columns="Segment", values="Avg daily spend")
                fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues",
                                labels=dict(x=SEGMENT_LABELS[demo_field], y="Campaign", color="Avg daily spend"),
                                title=f"Average daily recipient spending by campaign and {SEGMENT_LABELS[demo_field].lower()}")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Spending level, not campaign effect: dark columns usually mean high-spending segments, whatever the campaign.")

# ================= Spending & Impact =================
with tab_impact:
    st.subheader("Was the increase due to the campaign, or something else?")
    impact_campaign = st.selectbox("Choose campaign for impact analysis", sorted(metrics["CAMPAIGN"].unique()), key="impact_campaign")
    im = metrics.loc[metrics["CAMPAIGN"] == impact_campaign].iloc[0]
    crow = next(campaigns.loc[campaigns["CAMPAIGN"] == impact_campaign].itertuples(index=False))
    g = cs.campaign_groups(crow, hh_ids, CMAT, max_day, campaigns, sent)
    start, end = int(im["START_DAY"]), int(min(im["END_DAY"], max_day))
    duration = end - start + 1
    recips = set(sent.loc[sent["CAMPAIGN"] == impact_campaign, "HOUSEHOLD_KEY"].unique())
    if g is None:
        st.info("This campaign starts too early in the data to build a before-period and a placebo period.")
    else:
        pc = cs.period_compare(CMAT, g, max_day)
        fig = px.bar(pc, x="Period", y="Spend", color="Group", barmode="group",
                     title=f"Campaign {impact_campaign}: average daily spend, recipients vs comparison households")
        fig.update_layout(yaxis_title="Average daily spend per household (USD)")
        st.plotly_chart(fig, use_container_width=True)

        ti = cs.trend_index(CMAT, g, max_day)
        if len(ti):
            long = ti.melt(id_vars="DAY", var_name="Group", value_name="Spend index")
            fig = px.line(long, x="DAY", y="Spend index", color="Group", markers=True,
                          title="Weekly spend indexed to each group's own pre-campaign level (100)")
            fig.add_vrect(x0=start, x1=end, fillcolor="gray", opacity=0.12, line_width=0, annotation_text="campaign window")
            fig.add_hline(y=100, line_dash="dot", line_color="gray")
            fig.update_layout(xaxis_title="Day index", yaxis_title="Spend index (pre-campaign = 100)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("If the two lines already move apart before the shaded window, recipients were on a different trend regardless of the campaign.")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Raw lift (DiD)", fmt_money(im["LIFT_PER_HH_DAY"]) + ("" if pd.isna(im["LIFT_PER_HH_DAY"]) else "/day"),
              help="Matched comparison; excludes recipients of overlapping campaigns.")
    d2.metric("Pre-campaign gap (placebo)", fmt_money(im["PLACEBO"]) + ("" if pd.isna(im["PLACEBO"]) else "/day"),
              help="Same calculation one window earlier. Should be near zero.")
    d3.metric("Lift used for the verdict", fmt_money(im["USED_LIFT"]) + ("" if pd.isna(im["USED_LIFT"]) else "/day"))
    d4.metric("Verdict", short_verdict(im["VERDICT"]) if pd.notna(im["VERDICT"]) else "Insufficient data")

    st.markdown("**Other explanations we checked**")
    checks = []
    if g is not None:
        checks.append(("Overlapping campaigns in the window", f"{g['n_overlap_campaigns']} campaigns overlap; their recipients are excluded from the comparison group",
                       "ok" if g["n_overlap_campaigns"] <= 6 else "warn"))
        b_rec = pc.loc[(pc["Period"] == "Before") & (pc["Group"] == "Recipients"), "Spend"].iloc[0]
        b_ctl = pc.loc[(pc["Period"] == "Before") & (pc["Group"] == "Comparison households"), "Spend"].iloc[0]
        checks.append(("Baseline spending of comparison group", f"Comparison households spent {b_ctl / b_rec:.0%} of recipients' level before launch; they are reweighted to match recipients' spend bands, but remain a different kind of household",
                       "ok" if b_rec and b_ctl / b_rec >= 0.7 else "warn"))
        checks.append(("Comparison group makeup", f"{g['control_never_sent_share']:.0%} of comparison households were never sent any campaign",
                       "ok" if g["control_never_sent_share"] < 0.5 else "warn"))
    checks.append(("Pre-campaign trend (placebo)", "Parallel before launch" if im["PRETREND_OK"] is True or im["PRETREND_OK"] == True
                   else ("Recipients were already diverging; estimate is trend-adjusted" if im["PRETREND_OK"] == False else "Not enough pre-period data"),  # noqa: E712
                   "ok" if im["PRETREND_OK"] == True else "warn"))  # noqa: E712
    checks.append(("Sample size", f"{int(im['N_TREATED']) if pd.notna(im['N_TREATED']) else 0} recipient households with pre-period spending",
                   "ok" if pd.notna(im["N_TREATED"]) and im["N_TREATED"] >= cs.MIN_RECIPIENTS else "warn"))
    checks.append(("Data window", "Campaign end falls after the last observed day; window clipped" if bool(im.get("END_TRUNCATED", False)) else "Full campaign window observed",
                   "warn" if bool(im.get("END_TRUNCATED", False)) else "ok"))
    checks.append(("Weather, paydays, competitors", "Not measured in this dataset, so not controlled for", "warn"))
    st.dataframe(pd.DataFrame([{"Check": a, "Result": b, "Status": "✅" if c == "ok" else "⚠️"} for a, b, c in checks]),
                 use_container_width=True, hide_index=True)

    during_data = period_spend(hh_day, recips, start, end)
    before_data = period_spend(hh_day, recips, max(1, start - duration), start - 1)
    st.markdown("**Scatter plot: prior spending versus spending during campaign**")
    scatter = before_data.merge(during_data, on="HOUSEHOLD_KEY", suffixes=("_BEFORE", "_DURING"))
    if len(scatter):
        scatter = scatter.merge(hh, on="HOUSEHOLD_KEY", how="left")
        color_col = "INCOME_DESC" if "INCOME_DESC" in scatter.columns else None
        fig = px.scatter(
            scatter, x="PER_DAY_SPEND_BEFORE", y="PER_DAY_SPEND_DURING",
            color=color_col, hover_data=["HOUSEHOLD_KEY"],
            opacity=.55, title="Each point represents one recipient household"
        )
        maxv = max(float(scatter["PER_DAY_SPEND_BEFORE"].max()), float(scatter["PER_DAY_SPEND_DURING"].max()), 1)
        fig.add_trace(go.Scatter(x=[0, maxv], y=[0, maxv], mode="lines", name="No change reference", line=dict(dash="dash")))
        fig.update_layout(xaxis_title="Average daily spend before", yaxis_title="Average daily spend during")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough household data to create the scatter plot.")

    st.warning(
        "This is an observational estimate, not a randomized experiment. Campaigns were targeted, so recipients can differ from comparison households in ways we cannot see. "
        "The comparison is matched on prior spending, uses bootstrap intervals and a placebo test, but weather, paydays and competitor actions remain unmeasured."
    )

# ================= Phase 2: Next-Campaign Planner =================
with tab_plan:
    st.subheader("Phase 2: which campaign type should reach which customers next?")
    st.write("Turns the evidence into a targeting plan: for each customer segment, which campaign type has shown the strongest lift, and where is there no proven winner yet.")
    plan_fields = [f for f in SEGMENT_LABELS if f in hh.columns]
    if plan_fields:
        plan_field = st.selectbox("Segment by", plan_fields, format_func=lambda x: SEGMENT_LABELS[x], key="plan_field_pick")
        dec_p, seg_p = get_evidence(tx, campaigns, sent, hh, redemptions, plan_field, refresh_token)
        sp = cs.segment_pooled(seg_p, dec_p)
        if len(sp):
            sp["Cell"] = sp["POOLED_LIFT"].round(2)
            pv = sp.pivot(index="SEGMENT", columns="TYPE", values="POOLED_LIFT")
            fig = px.imshow(pv, aspect="auto", color_continuous_scale="RdBu", color_continuous_midpoint=0, text_auto=".2f",
                            labels=dict(x="Campaign type", y=SEGMENT_LABELS[plan_field], color="Pooled lift"),
                            title="Pooled lift per household-day by segment and campaign type")
            st.plotly_chart(fig, use_container_width=True)
            rec_rows = []
            for sgm in sorted(sp["SEGMENT"].unique()):
                x = sp[sp["SEGMENT"] == sgm].sort_values("POOLED_LIFT", ascending=False)
                best = x.iloc[0]
                tested = set(x["TYPE"])
                untested = sorted({"TypeA", "TypeB", "TypeC"} - tested)
                rec_rows.append({
                    SEGMENT_LABELS[plan_field]: sgm,
                    "Best-performing type": best["TYPE"],
                    "Pooled lift $/hh/day": round(best["POOLED_LIFT"], 2),
                    "95% interval": f"{best['CI_LO']:.2f} to {best['CI_HI']:.2f}",
                    "Households behind it": int(best["HOUSEHOLDS"]),
                    "Confidence": "Evidence-backed" if best["CI_LO"] > 0 else "Directional only: run a test",
                    "Not enough data for": ", ".join(untested) if untested else "-",
                })
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
            st.caption("Pooled over campaigns with at least 50 recipients; cells with too few households are dropped. Segment lifts are not adjusted for pre-campaign trends, "
                       "so treat 'evidence-backed' as 'best available', and confirm with a holdout test.")
            st.markdown("**Suggested holdout test:** for the top-ranked type per segment, hold out a random 10-20% of eligible households next time. "
                        "A randomised holdout removes the targeting bias that limits every estimate in this app.")
        else:
            st.info("Not enough households per segment to pool results. Try a coarser characteristic.")

# ================= AI Advisor =================
with tab_ai:
    st.subheader("Ask Gemini to interpret the campaign evidence")
    st.write("Gemini receives the aggregated scorecard (lift, 95% intervals, placebo results, verdicts), pooled results by campaign type and segment, and the method notes. "
             "It does not receive raw transaction rows or personal household records.")
    ai_campaign = st.selectbox("Campaign context", ["All campaigns"] + [f"Campaign {x}" for x in sorted(metrics["CAMPAIGN"].unique())], key="ai_campaign")
    suggested_questions = [
        "Which campaign should the grocer consider repeating, and what are the risks?",
        "Which campaigns look strongest, and what evidence supports that assessment?",
        "Which campaign types and customer segments deserve a test next?",
        "Explain the difference between high sales and incremental campaign impact.",
        "What additional data would help separate campaign effects from seasonality or other external factors?",
    ]
    question_choice = st.selectbox("Choose a suggested question", suggested_questions)
    custom_question = st.text_area("Or ask your own question", placeholder="For example: Which campaigns should we repeat for larger households, and what evidence supports that?")

    def build_ai_context(selection):
        cols = ["CAMPAIGN", "TYPE", "DURATION_DAYS", "RECIPIENT_HOUSEHOLDS", "N_TREATED", "N_CONTROL", "LIFT_PER_HH_DAY", "CI_LO", "CI_HI",
                "PLACEBO", "PRETREND_OK", "USED_LIFT", "USED_LO", "USED_HI", "BASIS", "INCR_SALES", "COUPON_COST",
                "REDEEMING_HH_RATE", "COUPON_CATEGORY_COUNT", "VERDICT"]
        sel = metrics.copy() if selection == "All campaigns" else metrics.loc[metrics["CAMPAIGN"] == int(selection.split()[-1])].copy()
        recs = sel[[c for c in cols if c in sel.columns]].round(3).replace({np.nan: None}).to_dict(orient="records")
        seg_ctx = []
        for fld in ["INCOME_BAND", "AGE_BAND"]:
            if fld in hh.columns:
                d_f, s_f = get_evidence(tx, campaigns, sent, hh, redemptions, fld, refresh_token)
                sp_f = cs.segment_pooled(s_f, d_f)
                if len(sp_f):
                    seg_ctx.append({fld: sp_f.round(3).to_dict(orient="records")})
        return (
            "Per-campaign evidence (lift in USD per recipient household per day vs matched comparison households; USED_* is the figure behind the verdict): "
            + json.dumps(recs, default=str)
            + "\nPooled lift by campaign type (raw DiD, upper bound): " + json.dumps(type_pool.round(3).to_dict(orient="records"), default=str)
            + "\nPooled lift by campaign type and segment (unadjusted for pre-trend): " + json.dumps(seg_ctx, default=str)
            + f"\nDataset: {len(tx)} line items, {households} households, {n_campaigns} campaigns; demographics for {hh['HOUSEHOLD_KEY'].nunique()} households. "
              "DAY is a sequential index, not a calendar date. Minimum 50 recipient households to rank a campaign."
        )

    if "ai_history" not in st.session_state:
        st.session_state["ai_history"] = []

    b1, b2 = st.columns(2)
    run_brief = b1.button("Write the CMO brief", type="primary", use_container_width=True)
    run_q = b2.button("Answer the question above", use_container_width=True)
    if run_brief or run_q:
        question = ("Write a one-page brief for the CMO: which campaigns to repeat, retest or drop, which campaign types and segments look best, and the main risks."
                    if run_brief else (custom_question.strip() or question_choice))
        if not api_key:
            st.warning("No Gemini key found, so here is the evidence-based brief generated directly from the scorecard.")
            st.markdown(rule_based_brief(dec, type_pool))
        else:
            with st.spinner("Gemini is analyzing the aggregate evidence…"):
                try:
                    answer = ask_gemini(api_key, model_name, question, build_ai_context(ai_campaign), st.session_state["ai_history"])
                    st.session_state["ai_history"] += [("user", question), ("assistant", answer)]
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")
    for role, text in st.session_state["ai_history"][-6:]:
        with st.chat_message(role):
            st.markdown(text)
    if st.session_state["ai_history"] and st.button("Clear conversation"):
        st.session_state["ai_history"] = []
        st.rerun()
    st.caption("AI recommendations are decision support, not proof of causal impact. Validate results and business assumptions before acting.")

with st.sidebar:
    st.markdown("---")
    st.caption("CampaignIQ · Built with Streamlit, Pandas, Plotly and Gemini")
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.session_state["refresh_token"] = refresh_token + 1
        st.rerun()
