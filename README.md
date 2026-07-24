# 📊 Sales Forecasting Dashboard

An end-to-end Machine Learning & Data Analytics project built to analyze retail store performance, engineer time-series features, evaluate predictive sales models, and present interactive forecasts via a **Streamlit** Web Application.

---

## 📁 Repository Structure
```
Sales-Forecasting-Dashboard/
│
├── data/
│   ├── train.csv                 # Raw sales transactions data
│   ├── store.csv                 # Store metadata
│   ├── processed_sales.csv       # Cleaned & merged dataset (Day 2)
│   ├── final_sales_dataset.csv   # Feature-engineered dataset (Day 4)
│   └── feature_importance.csv    # Random Forest feature importance scores
│
├── dashboard/
│   └── app.py                    # Streamlit interactive web application
│
├── models/
│   └── sales_forecasting_model.pkl # Trained Random Forest model binary
│
├── notebooks/
│   └── Sales_Forecasting.ipynb   # Complete step-by-step Jupyter Notebook
│
├── images/                       # Screenshots and visual assets
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Key Features & Highlights

- **Data Exploration & Cleaning**: Merged transaction records with store metadata, resolved missing values, handled categorical features, and formatted datetime metrics.
- **Time-Series Feature Engineering**: Built historical sales lag features (`Lag_1`, `Lag_7`), 7-day rolling statistics (`Rolling_Mean_7`, `Rolling_STD_7`), weekend indicators, and month boundary flags.
- **Machine Learning Benchmarking**: Evaluated baseline **Linear Regression** against **Random Forest Regressor** using MAE, RMSE, and $R^2$ metrics.
- **Interactive Streamlit Dashboard**: Filter sales by store, visualize historical trends, analyze promotion impacts, inspect feature importance, and compare actual vs. predicted sales in real-time.

---

## 📈 Model Performance Comparison

| Model | MAE | RMSE | $R^2$ Score |
|---|---|---|---|
| **Linear Regression** | *Baseline* | *Baseline* | *Baseline* |
| **Random Forest Regressor** | **Lowest** | **Lowest** | **Highest (~0.90+)** |

*The trained Random Forest model was serialized with `joblib` and selected for live forecasting inside the Streamlit application.*

---

## 💻 How to Run the Project Locally

### 1. Clone the Repository
```bash
git clone https://github.com/ruhile/Sales-Forecasting-Dashboard.git
cd Sales-Forecasting-Dashboard
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
*The dashboard will open automatically in your browser at `http://localhost:8501`.*

---

## 📝 Complete Project Lifecycle Summary

### Day 1: Project Setup & Data Exploration
- Established folder architecture.
- Inspected shapes, columns, datatypes, missingness, and descriptive statistics.

### Day 2: Data Cleaning & Preparation
- Left-joined `train.csv` and `store.csv` on `Store`.
- Imputed missing competition metrics and promo indicators.
- Created core date features (`Year`, `Month`, `Day`, `Week`, `Quarter`, `DayName`).
- Exported [`processed_sales.csv`](file:///C:/Users/acer/Desktop/python/week17/Sales-Forecasting-Dashboard/data/processed_sales.csv).

### Day 3: Exploratory Data Analysis (EDA)
- Analyzed daily, monthly, and yearly sales distributions.
- Investigated store performance, promotion lift, state holiday effects, and correlation matrices.

### Day 4: Feature Engineering
- Created 1-day and 7-day lag features alongside 7-day rolling mean & std.
- Encoded categorical variables using `LabelEncoder`.
- Exported ML-ready [`final_sales_dataset.csv`](file:///C:/Users/acer/Desktop/python/week17/Sales-Forecasting-Dashboard/data/final_sales_dataset.csv).

### Day 5: Model Building & Forecasting
- Trained Linear Regression & Random Forest models.
- Extracted Top 15 Feature Importances.
- Saved best model to [`models/sales_forecasting_model.pkl`](file:///C:/Users/acer/Desktop/python/week17/Sales-Forecasting-Dashboard/models/sales_forecasting_model.pkl).

### Day 6: Streamlit Dashboard Deployment
- Developed [`dashboard/app.py`](file:///C:/Users/acer/Desktop/python/week17/Sales-Forecasting-Dashboard/dashboard/app.py) with dynamic KPIs, store filtering, Plotly charts, feature importances, and actual vs. predicted sales views.

---

## 🤝 Acknowledgments & Repository Link
- GitHub Repository: [https://github.com/ruhile/Sales-Forecasting-Dashboard](https://github.com/ruhile/Sales-Forecasting-Dashboard)
