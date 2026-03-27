# DataWrangler Pro

A Streamlit-based data preparation studio for the 5COSC038C Data Wrangling & Visualization module.

## Live App
**https://datawrangler-progit-ktvnbhshzvfekirot4vrfp.streamlit.app/**

## Features
- Upload CSV, Excel, JSON datasets
- Interactive data profiling and quality scoring
- 8 cleaning feature sets (missing values, duplicates, type conversion, categorical tools, outlier handling, normalization, column ops, validation)
- Dynamic visualization builder (8 chart types)
- Export cleaned data + transformation reports + Python recipes

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sample Data
Two sample datasets in `sample_data/`:
- `sales_data.csv` - 1200 rows, sales transactions with intentional data quality issues
- `student_records.csv` - 1100 rows, academic records with intentional data quality issues

## How to Use

### 1. Upload & Overview
- Click **Browse files** and upload a CSV, Excel, or JSON file
- Review the auto-generated profile — data types, missing values, duplicates, quality score
- Use the **Reset Session** button to start over with a new file

### 2. Cleaning Studio
- Work through the expanders top to bottom
- Every operation shows a confirmation toast and is logged in the **Transform Log** at the bottom
- Use **↩ Undo Last Step** to revert any operation

| Expander | What it does |
|---|---|
| Missing Values | Drop or fill nulls per column |
| Duplicate Detection | Preview and remove duplicate rows |
| Type Conversion | Convert columns to int/float/string/category/datetime |
| Categorical Tools | Standardize casing, remap values, group rare categories |
| Numeric Cleaning | Detect and cap/remove outliers via IQR or z-score |
| Normalization | Min-max or z-score scale numeric columns |
| Column Operations | Rename, drop, create new columns, or bin numerics |
| Data Validation | Set rules and export violations |

### 3. Visualization Builder
- Use the **sidebar filters** to subset your data before plotting
- Select a chart type, configure axes, and click **Generate Chart**
- Click **Save Chart** to keep it in the gallery below
- Download filtered/aggregated data as CSV

### 4. Export & Report
- Download the cleaned dataset as CSV, Excel, or JSON
- Download the full **Transformation Report** (CSV or JSON)
- Copy the **Recipe Script** — a standalone Python file that reproduces all your cleaning steps using plain pandas
- Click **Start New Session** to clear everything

---

## Project Structure
```
app/
├── app.py
├── requirements.txt
├── pages/
│   ├── a_upload.py
│   ├── b_cleaning.py
│   ├── c_visualization.py
│   └── d_export.py
├── utils/
│   ├── profiler.py
│   ├── cleaners.py
│   └── validators.py
└── sample_data/
```
