# Austin Cost of Living Tracker

**A full-stack data intelligence application that collects, processes, analyzes, and visualizes cost-of-living data for Austin, TX — with an ML prediction model and AI-powered Q&A chatbot.**

> Built as a portfolio project demonstrating end-to-end data engineering, machine learning, and AI/LLM integration skills.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-yellow)

---

## What This Project Does

This application answers one question: **"How expensive is it to live in Austin, and where is it heading?"**

It does this through five connected layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE OVERVIEW                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│   [Data Sources]                                              │
│       │                                                       │
│       ▼                                                       │
│   ┌──────────────────┐                                        │
│   │  LAYER 1: PIPELINE│  Python scripts scrape/pull data      │
│   │  (Bronze → Silver │  from public APIs & websites.         │
│   │   → Gold tables)  │  Store in PostgreSQL with             │
│   └────────┬─────────┘  medallion architecture.               │
│            │                                                   │
│            ▼                                                   │
│   ┌──────────────────┐                                        │
│   │  LAYER 2: ML      │  Scikit-learn model predicts          │
│   │  (Prediction)     │  future rent/cost trends by           │
│   │                   │  neighborhood.                        │
│   └────────┬─────────┘                                        │
│            │                                                   │
│            ▼                                                   │
│   ┌──────────────────┐                                        │
│   │  LAYER 3: API     │  FastAPI serves predictions           │
│   │  (REST Endpoints) │  and data to the frontend.            │
│   └────────┬─────────┘                                        │
│            │                                                   │
│            ▼                                                   │
│   ┌──────────────────┐                                        │
│   │  LAYER 4: FRONTEND│  Streamlit dashboard +                │
│   │  (Dashboard + AI) │  LangChain chatbot that answers       │
│   │                   │  natural-language questions            │
│   └──────────────────┘  about your data.                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

- **Automated Data Collection** — Pulls rent, grocery, gas, and utility cost data from public sources into a PostgreSQL database
- **Medallion Architecture** — Raw data (Bronze) → Cleaned data (Silver) → Aggregated analytics-ready data (Gold)
- **ML Rent Prediction** — Forecasts neighborhood rent trends using time-series features and regression models
- **REST API** — FastAPI endpoints serve predictions and aggregated data
- **Interactive Dashboard** — Streamlit app with charts, filters, and neighborhood comparisons
- **AI Chatbot** — Ask questions like *"What's the cheapest neighborhood for a 1BR?"* and get answers powered by LangChain + your own data

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Pipeline | Python, requests, BeautifulSoup | Collect data from public sources |
| Storage | PostgreSQL | Structured storage with bronze/silver/gold schema |
| Processing | Pandas, NumPy | Clean, validate, transform data |
| ML Model | scikit-learn | Rent prediction (Linear Regression → Gradient Boosting) |
| API | FastAPI | Serve predictions and data via REST endpoints |
| Frontend | Streamlit | Interactive dashboard and chatbot UI |
| AI/Chat | LangChain, OpenAI API | Natural-language Q&A over your database |
| Deployment | Render / HuggingFace Spaces | Live demo accessible via URL |

---

## Project Structure

```
austin-cost-tracker/
│
├── README.md
├── requirements.txt
├── .env.example              # Template for API keys and DB credentials
├── .gitignore
│
├── data/
│   ├── raw/                  # Downloaded CSVs, JSON responses (bronze)
│   └── processed/            # Cleaned datasets (silver)
│
├── pipeline/
│   ├── __init__.py
│   ├── scraper.py            # Data collection scripts
│   ├── clean.py              # Bronze → Silver transformation
│   ├── aggregate.py          # Silver → Gold aggregation
│   ├── load_db.py            # Load data into PostgreSQL
│   └── validate.py           # Data quality checks
│
├── database/
│   ├── schema.sql            # CREATE TABLE statements (bronze, silver, gold)
│   └── seed.sql              # Optional: sample data for testing
│
├── model/
│   ├── __init__.py
│   ├── train.py              # Train ML model
│   ├── predict.py            # Generate predictions
│   └── model.pkl             # Saved trained model
│
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI app
│   └── routes.py             # API endpoints
│
├── app/
│   ├── streamlit_app.py      # Main Streamlit dashboard
│   ├── pages/
│   │   ├── dashboard.py      # Charts and visualizations
│   │   └── chatbot.py        # LangChain chatbot page
│   └── components/
│       └── charts.py         # Reusable chart functions
│
├── chatbot/
│   ├── __init__.py
│   ├── agent.py              # LangChain SQL agent setup
│   └── prompts.py            # System prompts for the chatbot
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
└── tests/
    ├── test_pipeline.py
    └── test_api.py
```

---

## Database Schema (Medallion Architecture)

### Bronze Layer (Raw Data)
```sql
CREATE TABLE bronze_rent (
    id SERIAL PRIMARY KEY,
    source VARCHAR(100),           -- 'zillow', 'apartments_com', 'census'
    neighborhood VARCHAR(200),
    bedrooms INTEGER,
    price DECIMAL(10,2),
    date_scraped DATE,
    raw_json JSONB,                -- Store full raw response
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE bronze_groceries (
    id SERIAL PRIMARY KEY,
    source VARCHAR(100),
    item_name VARCHAR(200),
    price DECIMAL(10,2),
    store VARCHAR(200),
    zip_code VARCHAR(10),
    date_scraped DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE bronze_utilities (
    id SERIAL PRIMARY KEY,
    source VARCHAR(100),
    utility_type VARCHAR(50),      -- 'electric', 'water', 'internet'
    avg_monthly_cost DECIMAL(10,2),
    zip_code VARCHAR(10),
    date_scraped DATE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Silver Layer (Cleaned & Validated)
```sql
CREATE TABLE silver_rent (
    id SERIAL PRIMARY KEY,
    neighborhood VARCHAR(200) NOT NULL,
    bedrooms INTEGER NOT NULL,
    price_cleaned DECIMAL(10,2) NOT NULL,
    price_per_sqft DECIMAL(10,2),
    month DATE NOT NULL,
    source VARCHAR(100),
    is_valid BOOLEAN DEFAULT TRUE,
    cleaned_at TIMESTAMP DEFAULT NOW()
);
```

### Gold Layer (Aggregated & Analytics-Ready)
```sql
CREATE TABLE gold_monthly_rent_avg (
    id SERIAL PRIMARY KEY,
    neighborhood VARCHAR(200),
    bedrooms INTEGER,
    avg_rent DECIMAL(10,2),
    median_rent DECIMAL(10,2),
    min_rent DECIMAL(10,2),
    max_rent DECIMAL(10,2),
    sample_count INTEGER,
    month DATE,
    computed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE gold_cost_of_living_index (
    id SERIAL PRIMARY KEY,
    neighborhood VARCHAR(200),
    month DATE,
    rent_index DECIMAL(5,2),
    grocery_index DECIMAL(5,2),
    utility_index DECIMAL(5,2),
    overall_index DECIMAL(5,2),   -- Weighted composite
    computed_at TIMESTAMP DEFAULT NOW()
);
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/rent/current` | Current average rent by neighborhood |
| GET | `/api/v1/rent/history?neighborhood=&months=12` | Historical rent trend |
| GET | `/api/v1/rent/predict?neighborhood=&bedrooms=1` | ML-predicted rent for next 3 months |
| GET | `/api/v1/cost-index?neighborhood=` | Overall cost of living index |
| GET | `/api/v1/neighborhoods` | List all tracked neighborhoods |
| POST | `/api/v1/chat` | Send a natural-language question, get AI answer |
| GET | `/api/v1/health` | API health check |

---

## Data Sources

All data comes from **free, publicly available sources**:

| Source | What It Provides | How to Access |
|--------|-----------------|---------------|
| [Zillow ZORI](https://www.zillow.com/research/data/) | Monthly rent index by zip code | Direct CSV download (free) |
| [Census ACS](https://data.census.gov/) | Median rent, income, housing costs | API (free key) |
| [Austin Open Data](https://data.austintexas.gov/) | City-level housing and utility data | Socrata API (free) |
| [BLS CPI](https://www.bls.gov/cpi/) | Consumer price index for groceries, utilities | API (free key) |
| [Numbeo](https://www.numbeo.com/cost-of-living/) | Cost of living comparisons | Web scraping (BeautifulSoup) |

> **Note:** Start with Zillow ZORI CSV + Census API. These two alone give you enough data to build the full pipeline. Add others incrementally.

---

## 2-Week Build Plan

### Week 1: Pipeline + Database + ML Model (Days 1–7)

**Day 1-2: Setup + Data Collection**
- [ ] Create GitHub repo with this project structure
- [ ] Set up PostgreSQL locally (or use Supabase free tier)
- [ ] Write `schema.sql` with bronze/silver/gold tables
- [ ] Download Zillow ZORI rent data CSV
- [ ] Write `scraper.py` to pull Census API data
- [ ] Load raw data into bronze tables via `load_db.py`

**Day 3-4: Data Cleaning Pipeline**
- [ ] Write `clean.py` — handle nulls, standardize neighborhood names, validate prices
- [ ] Write `validate.py` — data quality checks (no negative prices, no future dates, etc.)
- [ ] Write `aggregate.py` — compute monthly averages, medians, cost index by neighborhood
- [ ] Populate silver and gold tables
- [ ] Create `01_data_exploration.ipynb` — EDA with charts showing rent trends

**Day 5-6: ML Model**
- [ ] Create `02_feature_engineering.ipynb` — build features (month, neighborhood encoded, lagged prices, rolling averages)
- [ ] Create `03_model_training.ipynb` — train Linear Regression first, then Gradient Boosting
- [ ] Evaluate with RMSE and R-squared; pick best model
- [ ] Write `train.py` and `predict.py` — save model as `model.pkl`
- [ ] Test: given a neighborhood + bedrooms, predict next-month rent

**Day 7: FastAPI**
- [ ] Write `api/main.py` with all endpoints listed above
- [ ] Connect API to PostgreSQL (use SQLAlchemy or psycopg2)
- [ ] Connect `/predict` endpoint to your saved ML model
- [ ] Test all endpoints with curl or Postman
- [ ] Write `test_api.py` with basic tests

---

### Week 2: Frontend + Chatbot + Deploy (Days 8–14)

**Day 8-9: Streamlit Dashboard**
- [ ] Build `streamlit_app.py` with:
  - Neighborhood selector dropdown
  - Line chart: rent trends over time
  - Bar chart: compare neighborhoods side-by-side
  - KPI cards: average rent, cost index, predicted trend
- [ ] Connect Streamlit to your FastAPI (or directly to PostgreSQL)

**Day 10-11: LangChain Chatbot**
- [ ] Install langchain, openai
- [ ] Write `agent.py` — create a SQL agent that can query your gold tables
- [ ] Write `prompts.py` — system prompt telling the LLM it's a cost-of-living expert
- [ ] Build `chatbot.py` Streamlit page with chat input/output
- [ ] Test questions like:
  - "What's the average rent in East Austin?"
  - "Which neighborhood has the cheapest 2BR?"
  - "How has rent changed in the last 6 months?"

**Day 12-13: Polish + Deploy**
- [ ] Add error handling to pipeline scripts
- [ ] Write clear docstrings in every module
- [ ] Take screenshots for README
- [ ] Deploy Streamlit app to HuggingFace Spaces or Streamlit Cloud (free)
- [ ] Deploy FastAPI to Render (free tier)
- [ ] Update README with live demo links and screenshots

**Day 14: Documentation + LinkedIn**
- [ ] Final README polish — add architecture diagram screenshot, demo GIF
- [ ] Write a short LinkedIn post: "I built an Austin Cost of Living Tracker..."
- [ ] Add project to your resume with 2-3 strong bullet points
- [ ] Push everything to GitHub

---

## How to Run Locally

### Prerequisites
- Python 3.10+
- PostgreSQL 16+
- OpenAI API key (for chatbot — get free credits at platform.openai.com)

### Setup
```bash
# Clone the repo
git clone https://github.com/SumanAch10/austin-cost-tracker.git
cd austin-cost-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your PostgreSQL credentials and OpenAI API key

# Create database tables
psql -U your_user -d austin_cost -f database/schema.sql

# Run the data pipeline
python pipeline/scraper.py        # Collect data
python pipeline/clean.py          # Clean and validate
python pipeline/aggregate.py      # Build gold tables
python pipeline/load_db.py        # Load into PostgreSQL

# Train the ML model
python model/train.py

# Start the API
uvicorn api.main:app --reload

# Start the Streamlit dashboard (in a new terminal)
streamlit run app/streamlit_app.py
```

---

## What I Learned Building This

- **Data Engineering:** Designing schemas, building ETL pipelines, medallion architecture (bronze/silver/gold), data validation and quality monitoring
- **Machine Learning:** Feature engineering with time-series data, model selection and evaluation, serving predictions via API
- **AI/LLM Integration:** LangChain SQL agents, RAG concepts, prompt engineering, building chat interfaces over structured data
- **Full-Stack Development:** FastAPI REST endpoints, Streamlit interactive dashboards, connecting frontend to backend to database
- **Software Engineering:** Project structure, environment management, testing, documentation, deployment

---

## Future Improvements

- [ ] Add Apache Airflow for scheduled pipeline runs
- [ ] Add more data sources (gas prices, grocery baskets, commute costs)
- [ ] Build a Power BI report connected to the same PostgreSQL database
- [ ] Add user authentication to the Streamlit app
- [ ] Implement caching for API responses
- [ ] Add CI/CD with GitHub Actions

---

## Author

**Suman Acharya**
- CS @ Texas State University | GPA: 4.0
- [LinkedIn](https://www.linkedin.com/in/suman-acharya-b0844627b/)
- [GitHub](https://github.com/SumanAch10)

---

## License

This project is open source under the [MIT License](LICENSE).
# AustinCstLivingTracker
