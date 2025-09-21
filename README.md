#  AgroVision
**IoT-style Data + Machine Learning Powered Precision Farming Platform**

AgroVision is a decision-support system that provides **irrigation recommendations** using simulated IoT datasets and machine learning models.

##  Features
- IoT-style data simulation (soil moisture, temp, humidity, rainfall, wind)
- ML models (baseline Linear Regression; room for RF/XGBoost)
- Decision-support interface (trends, scenario analysis, recommendations)
- Export to CSV
- Lightweight & local (Python 3.10+)

##  Quickstart
\`\`\`bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/simulate_data.py
python3 scripts/train_baseline.py
python -m streamlit run app/main.py
\`\`\`

##  Screenshots
![Latest Trends](docs/screenshots/screenshot1.png)
![Recommendation](docs/screenshots/screenshot2.png)

## 🛠️ Tech Stack
Python · Pandas · NumPy · scikit-learn · Streamlit · Matplotlib · Joblib

##  Roadmap
- [x] Data simulation & baseline model
- [ ] Advanced models (XGBoost/RandomForest)
- [ ] Complete interface & export
- [ ] SRS, test cases, YouTube demo
