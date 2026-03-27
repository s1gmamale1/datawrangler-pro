# DataWrangler Pro

A Streamlit-based data preparation studio for the 5COSC038C Data Wrangling & Visualization module.

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
