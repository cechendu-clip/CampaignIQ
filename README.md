# CampaignIQ — AI-Powered Retail Campaign Intelligence

CampaignIQ is an interactive Streamlit application that helps a grocery retailer evaluate promotional campaigns, compare household spending, identify potential rerun candidates, and explore future household-targeting strategies.

The application uses dunnhumby's *The Complete Journey* dataset and integrates Google Gemini for AI-assisted interpretation of campaign results.

## App Deployment URL

**Live application:** Deployment pending — URL will be added after publishing to Streamlit Community Cloud.

## Project Overview

**Client:** Raymond Reddington, Chief Marketing Officer, Larsen's Market.

**Business Decision:** Of the 30 promotional campaigns conducted over two years, which should be considered for funding again next quarter, and which household groups should be targeted?

**Mandate:** Develop an interactive Campaign Scorecard that ranks previous campaigns using observed purchase behavior and comparison-household analysis.

CampaignIQ contains 30 campaigns, of which 29 can be fully analyzed. Campaign 24 is excluded from performance evaluation because its scheduled end date (Day 719) exceeds the available transaction data (Day 711).

## Features

1. **Decision Scorecard:** Classifies campaigns into repeat candidates, retest recommendations, lower-priority campaigns, and insufficient-sample cases.
2. **Executive Overview:** Provides interactive campaign rankings and performance comparisons.
3. **Campaign Intelligence:** Examines campaign duration, audience size, coupon characteristics, redemption metrics, and estimated spending differences.
4. **Demographics:** Explores campaign performance across available household demographic segments.
5. **Spending & Impact:** Compares spending before, during, and after campaigns, including recipient and comparison-household trends.
6. **Next-Campaign Planner:** Identifies potential campaign-type and household-segment combinations for further testing.
7. **AI Marketing Advisor:** Uses Google Gemini to interpret aggregated campaign results, explain uncertainty, and discuss potential risks.

## Technology Stack

- Python 3.13+
- Streamlit
- Pandas and NumPy
- Plotly
- Google Gemini API
- `uv` for dependency and environment management

## Local Setup Instructions

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CampaignIQ
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual GitHub repository URL after publishing.

### 2. Install uv

Install `uv` using the instructions at https://docs.astral.sh/uv/getting-started/installation/

### 3. Install project dependencies

```bash
uv sync
```

This creates the project's virtual environment and installs dependencies from `pyproject.toml` and `uv.lock`.

### 4. Configure Gemini (optional)

Copy `.env.example` to `.env` and provide a valid Gemini API key.

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

The main analytics dashboard can be used without a Gemini API key. A valid key is required to generate live AI responses.

Never commit `.env` or expose API credentials.

### 5. Run CampaignIQ

```bash
uv run streamlit run app.py
```

Open the local URL provided by Streamlit, usually:

http://localhost:8501

## Dataset

CampaignIQ uses dunnhumby's *The Complete Journey* dataset, containing approximately:

- 2.6 million transaction records
- 2,500 loyalty-card households
- 30 promotional campaigns
- Household demographics, products, coupons, and redemption information

The repository includes the seven required application datasets in `data/raw/`:

```text
data/raw/
├── transaction_data.csv.gz
├── hh_demographic.csv
├── product.csv
├── campaign_desc.csv
├── campaign_table.csv
├── coupon.csv
└── coupon_redempt.csv
```

The transaction dataset is compressed to reduce repository size. CampaignIQ reads the compressed file directly.

`causal_data.csv` is excluded because of its large size and is not required for the current application.

Original dataset source:

https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey

Dataset redistribution is subject to the source's applicable terms and license.

## Analytical Limitations

- Campaign comparisons are observational, not randomized experiments.
- Estimated spending differences should not be interpreted as proven causal impact or verified return on investment.
- Confidence intervals and pre-trend checks help assess uncertainty but cannot eliminate unmeasured confounding.
- Demographic records are available for approximately 32% of households.
- Campaign exposure does not prove that recipients read or acted on an offer.
- Overlapping promotions, seasonality, competitor activity, and other unmeasured factors may influence observed spending.
- The dataset uses sequential DAY and WEEK_NO indices rather than calendar dates.
- The Next-Campaign Planner provides targeting hypotheses that should be validated using randomized holdout testing.
- Gemini-generated interpretations are decision-support material and must be checked against the underlying calculations.

## Security

Gemini API credentials must be stored in local environment settings or secure deployment secrets.

The repository excludes `.env`, local virtual environments, temporary files, and unnecessary large datasets.

Any API key previously exposed should be revoked and replaced before deployment.

## Project Status

The application has been tested locally across all seven dashboard sections, including a successful live Gemini response.

Deployment and final public-access verification are pending.

## Team

Team 8 — Visual Analytics

- Clip Echendu
- Mekedes Mekete
- Smriti Bhemireddy

## Acknowledgment

Dataset: dunnhumby, *The Complete Journey*.

Built with Streamlit, Python, Plotly, and Google Gemini.