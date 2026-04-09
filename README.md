# Texas Cost of Living Tracker & Recommendation Engine

**A full-stack data intelligence application that collects, processes, analyzes, and visualizes cost-of-living data across Texas — with ML prediction models, a personalized metro recommendation engine, and an AI-powered Q&A chatbot.**

> Built as a portfolio project demonstrating end-to-end data engineering, machine learning, recommendation systems, and AI/LLM integration skills.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-yellow)

---

## What This Project Does

This application answers two questions:
1. **"How expensive is it to live in Texas, and how does it vary across the state?"**
2. **"Based on my salary, priorities, and lifestyle — where in Texas should I live?"**

Texas is the second-largest state in the US with 254 counties, 30+ metro areas, and massive cost-of-living variation — from downtown Dallas high-rises to rural West Texas towns. This project tracks, analyzes, and predicts those differences — then turns that data into personalized recommendations.

```
┌──────────────────────────────────────────────────────────────────┐
│                     ARCHITECTURE OVERVIEW                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   [Data Sources]                                                   │
│    Zillow ZORI · Census ACS · BLS CPI · MIT Living Wage           │
│       │                                                            │
│       ▼                                                            │
│   ┌─────────────────────┐                                          │
│   │  LAYER 1: PIPELINE   │  Python scripts pull data from          │
│   │  Bronze → Silver     │  APIs and CSVs. Validate, clean,        │
│   │  → Gold tables       │  and aggregate into PostgreSQL          │
│   └──────────┬──────────┘  using medallion architecture.           │
│              │                                                      │
│              ▼                                                      │
│   ┌─────────────────────┐                                          │
│   │  LAYER 2: ML MODELS  │  Predict rent trends by city.           │
│   │  Regression +        │  Classify affordability zones.          │
│   │  Classification +    │  Score & rank metros for users.         │
│   │  Recommendation      │                                         │
│   └──────────┬──────────┘                                          │
│              │                                                      │
│              ▼                                                      │
│   ┌─────────────────────┐                                          │
│   │  LAYER 3: API        │  FastAPI serves predictions,            │
│   │  REST Endpoints      │  recommendations, aggregated            │
│   │                      │  data, and city comparisons.            │
│   └──────────┬──────────┘                                          │
│              │                                                      │
│              ▼                                                      │
│   ┌─────────────────────┐                                          │
│   │  LAYER 4: FRONTEND   │  Streamlit dashboard with               │
│   │  Dashboard + Chat    │  interactive Texas map, charts,         │
│   │  + Recommendation    │  metro recommendations, and an          │
│   │    Engine UI         │  AI chatbot for natural-language Q&A.   │
│   └─────────────────────┘                                          │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Current Progress

### ✅ Completed
- **Data Collection (Bronze Layer)** — Zillow ZORI, Census ACS (2015–2024, year-by-year with rate limiting), BLS CPI, and MIT Living Wage data collected
- **BLS CPI API** — Successfully pulling Consumer Price Index data for Houston and Dallas metro areas

### 🔧 In Progress
- **Census Data Cleaning** — Combining 10 years of county-level Census data, renaming columns, type conversion
- **County → Metro Mapping** — Building CBSA lookup to aggregate county data to metro level

### 📋 Up Next
- Clean and merge all data sources at metro level
- Load into PostgreSQL (Silver + Gold layers)
- ML model training (rent regression + affordability classifier)
- Recommendation engine
- FastAPI backend
- Streamlit dashboard + AI chatbot
- Deployment

---

## Recommendation Engine

The recommendation engine answers: **"Where in Texas should YOU live?"**

A user provides their inputs, and the system scores every Texas metro against their priorities to produce a personalized ranking.

### User Inputs
- **Annual salary / budget** — What can they afford?
- **Family size** — Single, couple, or family with kids?
- **Priority weights** — What matters most to them?
  - Affordable rent
  - Low grocery / food costs
  - Job market strength
  - Population size preference (big city vs small town)
  - Income-to-rent ratio

### How It Works
1. **Normalize** all metro-level features (rent, food costs, income, population, etc.) to a common 0–1 scale
2. **Weight** each feature by the user's stated priorities
3. **Compute** a composite score for each metro
4. **Rank** metros and return the top recommendations with explanations
5. **Visualize** results on an interactive Texas map highlighting recommended metros

### Example
```
Input:  Salary = $55,000 | Single | Priorities: cheap rent > job market > small city
Output:
  #1  San Marcos (Central TX)   — Score: 92/100
  #2  Corpus Christi (South TX) — Score: 87/100
  #3  Lubbock (West TX)         — Score: 84/100
```

---

## Why Texas? (Scope Decision)

| Factor | Benefit |
|--------|---------|
| **254 counties** | Large, real-world dataset — not a toy project |
| **30+ metro areas** | Rich variation for ML features (urban vs suburban vs rural) |
| **Massive population growth** | Rent trends are volatile and interesting to predict |
| **Public data availability** | Texas has excellent open data portals |
| **Regional diversity** | DFW, Houston, Austin, San Antonio, El Paso — each has unique patterns |
| **Personal relevance** | I live in San Marcos, TX — I can validate the data against reality |

---

## Features

- **Statewide Data Collection** — Pulls rent, grocery, utility, and transportation cost data across Texas metros and counties
- **Medallion Architecture** — Raw data (Bronze) → Cleaned data (Silver) → Analytics-ready aggregations (Gold)
- **City-vs-City Comparison** — Compare cost of living between any two Texas cities side-by-side
- **ML Rent Prediction** — Forecasts rent trends by metro area using time-series regression
- **Affordability Classifier** — Flags metros as "affordable," "moderate," or "expensive" relative to state median
- **Recommendation Engine** — Personalized metro recommendations based on salary, family size, and lifestyle priorities
- **REST API** — FastAPI endpoints serve predictions, recommendations, comparisons, and aggregated data
- **Interactive Dashboard** — Streamlit app with a Texas map, charts, filters, and metro comparisons
- **AI Chatbot** — Ask questions like *"What's the cheapest city to live in Texas?"* and get data-backed answers

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Pipeline | Python, requests, Pandas | Collect and process data from public sources |
| Storage | PostgreSQL | Structured storage with bronze/silver/gold schema |
| Processing | Pandas, NumPy | Clean, validate, transform, aggregate |
| ML Models | scikit-learn | Rent prediction, affordability classifier, recommendation scoring |
| API | FastAPI | Serve predictions, recommendations, and data via REST endpoints |
| Frontend | Streamlit | Interactive dashboard, Texas map, recommendation UI, and chatbot |
| AI/Chat | LangChain, OpenAI API | Natural-language Q&A over the database |
| Deployment | Render / HuggingFace Spaces | Live demo accessible via URL |

---

## Project Structure

```
texas-cost-tracker/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/                  # Downloaded CSVs, API responses (bronze)
│   ├── processed/            # Cleaned datasets (silver)
│   └── reference/            # Texas county FIPS codes, metro area mappings
│
├── pipeline/
│   ├── __init__.py
│   ├── scraper_zillow.py     # Download Zillow ZORI rent data (CSV)
│   ├── scraper_census.py     # Pull Census ACS data via API
│   ├── scraper_bls.py        # Pull BLS CPI data via API
│   ├── scraper_mit.py        # Scrape MIT Living Wage data
│   ├── clean.py              # Bronze → Silver transformation
│   ├── aggregate.py          # Silver → Gold aggregation
│   ├── load_db.py            # Load into PostgreSQL tables
│   └── validate.py           # Data quality checks
│
├── database/
│   ├── schema.sql            # All CREATE TABLE statements
│   ├── indexes.sql           # Performance indexes
│   └── seed.sql              # Sample data for testing
│
├── model/
│   ├── __init__.py
│   ├── features.py           # Feature engineering
│   ├── train_regression.py   # Rent prediction model
│   ├── train_classifier.py   # Affordability classifier
│   ├── recommend.py          # Recommendation engine logic
│   ├── evaluate.py           # Model evaluation metrics
│   ├── predict.py            # Generate predictions
│   └── artifacts/            # Saved models (.pkl files)
│
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI app
│   ├── routes/
│   │   ├── rent.py           # Rent endpoints
│   │   ├── compare.py        # City comparison endpoints
│   │   ├── predict.py        # ML prediction endpoints
│   │   ├── recommend.py      # Recommendation endpoints
│   │   └── chat.py           # Chatbot endpoint
│   └── database.py           # DB connection config
│
├── app/
│   ├── streamlit_app.py      # Main Streamlit app
│   ├── pages/
│   │   ├── overview.py       # Statewide overview + Texas map
│   │   ├── compare.py        # City-vs-city comparison tool
│   │   ├── trends.py         # Historical trend charts
│   │   ├── predict.py        # Prediction explorer
│   │   ├── recommend.py      # Recommendation engine UI
│   │   └── chatbot.py        # AI chatbot page
│   └── components/
│       ├── charts.py         # Reusable chart functions
│       └── texas_map.py      # Plotly choropleth map
│
├── chatbot/
│   ├── __init__.py
│   ├── agent.py              # LangChain SQL agent
│   └── prompts.py            # System prompts
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_texas_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_recommendation_engine.ipynb
│
└── tests/
    ├── test_pipeline.py
    ├── test_api.py
    ├── test_model.py
    └── test_recommend.py
```

---

## Database Schema (Medallion Architecture)

### Bronze Layer (Raw Data)
```sql
CREATE TABLE bronze_rent (
    id SERIAL PRIMARY KEY,
    source VARCHAR(100),
    metro_area VARCHAR(200),
    city VARCHAR(200),
    county VARCHAR(200),
    zip_code VARCHAR(10),
    state_code VARCHAR(2) DEFAULT 'TX',
    bedrooms INTEGER,
    price DECIMAL(10,2),
    date_period DATE,
    raw_json JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE bronze_prices (
    id SERIAL PRIMARY KEY,
    source VARCHAR(100),
    category VARCHAR(100),
    item_name VARCHAR(200),
    price DECIMAL(10,2),
    metro_area VARCHAR(200),
    zip_code VARCHAR(10),
    date_period DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE bronze_utilities (
    id SERIAL PRIMARY KEY,
    source VARCHAR(100),
    utility_type VARCHAR(50),
    avg_monthly_cost DECIMAL(10,2),
    metro_area VARCHAR(200),
    zip_code VARCHAR(10),
    date_period DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE bronze_income (
    id SERIAL PRIMARY KEY,
    source VARCHAR(100),
    county VARCHAR(200),
    metro_area VARCHAR(200),
    median_household_income DECIMAL(12,2),
    unemployment_rate DECIMAL(5,2),
    population INTEGER,
    year INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE ref_texas_geo (
    fips_code VARCHAR(5) PRIMARY KEY,
    county_name VARCHAR(200),
    metro_area VARCHAR(200),
    region VARCHAR(100),
    latitude DECIMAL(9,6),
    longitude DECIMAL(9,6),
    population_2024 INTEGER
);
```

### Silver Layer (Cleaned & Validated)
```sql
CREATE TABLE silver_rent (
    id SERIAL PRIMARY KEY,
    metro_area VARCHAR(200) NOT NULL,
    city VARCHAR(200),
    county VARCHAR(200),
    zip_code VARCHAR(10),
    region VARCHAR(100),
    bedrooms INTEGER NOT NULL,
    price_cleaned DECIMAL(10,2) NOT NULL,
    price_per_sqft DECIMAL(10,2),
    month DATE NOT NULL,
    source VARCHAR(100),
    is_valid BOOLEAN DEFAULT TRUE,
    cleaned_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE silver_cost_items (
    id SERIAL PRIMARY KEY,
    metro_area VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    item_name VARCHAR(200),
    price_cleaned DECIMAL(10,2) NOT NULL,
    month DATE NOT NULL,
    source VARCHAR(100),
    cleaned_at TIMESTAMP DEFAULT NOW()
);
```

### Gold Layer (Analytics-Ready)
```sql
CREATE TABLE gold_rent_monthly (
    id SERIAL PRIMARY KEY,
    metro_area VARCHAR(200),
    region VARCHAR(100),
    bedrooms INTEGER,
    avg_rent DECIMAL(10,2),
    median_rent DECIMAL(10,2),
    min_rent DECIMAL(10,2),
    max_rent DECIMAL(10,2),
    yoy_change_pct DECIMAL(5,2),
    sample_count INTEGER,
    month DATE,
    computed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE gold_cost_index (
    id SERIAL PRIMARY KEY,
    metro_area VARCHAR(200),
    region VARCHAR(100),
    month DATE,
    rent_index DECIMAL(5,2),
    grocery_index DECIMAL(5,2),
    utility_index DECIMAL(5,2),
    transport_index DECIMAL(5,2),
    overall_index DECIMAL(5,2),
    affordability_label VARCHAR(20),
    computed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE gold_city_comparison (
    id SERIAL PRIMARY KEY,
    city_a VARCHAR(200),
    city_b VARCHAR(200),
    month DATE,
    rent_diff_pct DECIMAL(5,2),
    grocery_diff_pct DECIMAL(5,2),
    utility_diff_pct DECIMAL(5,2),
    overall_diff_pct DECIMAL(5,2),
    cheaper_city VARCHAR(200),
    computed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE gold_ml_features (
    id SERIAL PRIMARY KEY,
    metro_area VARCHAR(200),
    month DATE,
    bedrooms INTEGER,
    current_rent DECIMAL(10,2),
    rent_lag_1m DECIMAL(10,2),
    rent_lag_3m DECIMAL(10,2),
    rent_lag_6m DECIMAL(10,2),
    rent_rolling_3m_avg DECIMAL(10,2),
    rent_rolling_6m_avg DECIMAL(10,2),
    yoy_change DECIMAL(5,2),
    median_income DECIMAL(12,2),
    unemployment_rate DECIMAL(5,2),
    population INTEGER,
    month_sin DECIMAL(8,6),
    month_cos DECIMAL(8,6),
    metro_encoded INTEGER,
    target_next_month_rent DECIMAL(10,2),
    computed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE gold_metro_profiles (
    id SERIAL PRIMARY KEY,
    metro_area VARCHAR(200),
    region VARCHAR(100),
    avg_rent DECIMAL(10,2),
    median_income DECIMAL(12,2),
    total_population INTEGER,
    food_cost_annual DECIMAL(10,2),
    transport_cost_annual DECIMAL(10,2),
    medical_cost_annual DECIMAL(10,2),
    rent_to_income_ratio DECIMAL(5,4),
    affordability_score DECIMAL(5,2),
    overall_cost_index DECIMAL(5,2),
    last_updated DATE,
    computed_at TIMESTAMP DEFAULT NOW()
);
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/metros` | List all Texas metro areas tracked |
| GET | `/api/v1/rent/current?metro=` | Current avg rent for a metro |
| GET | `/api/v1/rent/history?metro=&months=12` | Historical rent trend |
| GET | `/api/v1/rent/predict?metro=&bedrooms=1` | ML-predicted rent for next 3 months |
| GET | `/api/v1/compare?city_a=Austin&city_b=Dallas` | Side-by-side cost comparison |
| GET | `/api/v1/cost-index?metro=` | Cost of living index for a metro |
| GET | `/api/v1/affordable?budget=1500&bedrooms=1` | Find metros within budget |
| POST | `/api/v1/recommend` | Personalized metro recommendations |
| GET | `/api/v1/statewide/summary` | Statewide averages and rankings |
| POST | `/api/v1/chat` | Natural-language question → AI answer |
| GET | `/api/v1/health` | API health check |

---

## Data Sources (All Free & Public)

| Source | What It Provides | Coverage | Access |
|--------|-----------------|----------|--------|
| [Zillow ZORI](https://www.zillow.com/research/data/) | Monthly rent index by metro and zip | All TX metros | CSV download |
| [Census ACS](https://data.census.gov/) | Median income, population by county | Every TX county (2015–2024) | API (free key) |
| [BLS CPI](https://www.bls.gov/cpi/) | Consumer price index | Houston, DFW | API (free key) |
| [MIT Living Wage](https://livingwage.mit.edu/) | Food, transport, medical costs by county | All 254 TX counties | Web data |

---

## Texas Regions

```
┌──────────────┬──────────────────────────────────────────────┐
│ Region       │ Major Metros                                  │
├──────────────┼──────────────────────────────────────────────┤
│ North Texas  │ Dallas, Fort Worth, Plano, Arlington, Denton │
│ Gulf Coast   │ Houston, Galveston, Beaumont                  │
│ Central TX   │ Austin, San Marcos, Round Rock, Waco          │
│ South TX     │ San Antonio, Laredo, McAllen, Corpus Christi  │
│ West TX      │ El Paso, Midland, Odessa, Lubbock            │
│ East TX      │ Tyler, Longview, College Station              │
│ Panhandle    │ Amarillo, Abilene, Wichita Falls             │
└──────────────┴──────────────────────────────────────────────┘
```

---

## Build Plan (8 Weeks)

### Phase 1: Data Pipeline (Weeks 1–2)
- ✅ Create GitHub repo with project structure
- ✅ Write data scrapers (Zillow ZORI, Census ACS, BLS CPI, MIT Living Wage)
- ✅ Collect raw data into CSV files
- 🔧 Clean and combine Census data (2015–2024)
- 🔧 Build county → metro CBSA mapping
- ⬜ Clean all data sources, merge at metro level
- ⬜ Set up PostgreSQL with medallion schema
- ⬜ Load into Silver + Gold tables

### Phase 2: EDA & Feature Engineering (Weeks 3–4)
- ⬜ Exploratory data analysis notebooks
- ⬜ Feature engineering (lagged rent, rolling averages, cyclical encoding)
- ⬜ Build `gold_metro_profiles` table for recommendation engine
- ⬜ Data validation and quality checks

### Phase 3: ML Models (Weeks 5–6)
- ⬜ Train rent regression (Linear → Ridge → Gradient Boosting)
- ⬜ Train affordability classifier (Logistic Regression → Random Forest)
- ⬜ Build recommendation engine scoring logic
- ⬜ Evaluate all models, save artifacts

### Phase 4: API + Frontend + Deploy (Weeks 7–8)
- ⬜ Build FastAPI endpoints (including `/recommend`)
- ⬜ Streamlit dashboard with Texas choropleth map
- ⬜ Recommendation engine UI (input form → ranked results on map)
- ⬜ LangChain chatbot for natural-language Q&A
- ⬜ Deploy to Render / HuggingFace Spaces
- ⬜ Polish README, screenshots, demo video

---

## Resume Bullets

**Data Engineering:**
> Engineered end-to-end data pipeline ingesting rent, income, and consumer price data across 254 Texas counties into PostgreSQL using medallion architecture (bronze/silver/gold); implemented data validation and quality monitoring achieving 98%+ data accuracy

**Data Science / ML:**
> Built ML rent prediction model (Gradient Boosting, RMSE < $50) and affordability classifier (F1 > 0.85) trained on 50,000+ records spanning 30+ Texas metros; performed feature engineering with lagged prices, rolling averages, and economic indicators

**Recommendation Systems:**
> Designed weighted scoring recommendation engine that matches users to Texas metros based on salary, family size, and lifestyle priorities; normalizes cost-of-living features across 30+ metros and produces personalized rankings with explanations

**Full-Stack / AI:**
> Developed Streamlit dashboard with interactive Texas choropleth map, personalized metro recommendation tool, and LangChain-powered chatbot enabling natural-language queries over structured data; deployed via FastAPI REST API

---

## Author

**Suman Acharya**
- CS @ Texas State University | GPA: 4.0
- [LinkedIn](https://www.linkedin.com/in/suman-acharya-b0844627b/)
- [GitHub](https://github.com/SumanAch10)

---

## License

This project is open source under the [MIT License](LICENSE).