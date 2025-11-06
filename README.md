🌾 AgroVision – Smart Irrigation Recommendation Platform

IoT-driven Precision Farming powered by Machine Learning

AgroVision is a lightweight, local decision-support system that uses IoT-style environmental data and machine learning models to provide irrigation recommendations for smarter farming decisions.

✨ Features

🌦️ Simulated IoT data (soil moisture, temperature, humidity, rainfall, wind)

🤖 ML models — baseline Linear Regression (with room for RF/XGBoost)

📊 Interactive dashboard (environmental trends, scenario analysis, recommendations)

💾 Export results to CSV

⚙️ Lightweight & runs locally on Python 3.10+

🚀 Quickstart
# 1️⃣ Create & activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Generate sample IoT data
python3 scripts/simulate_data.py

# 4️⃣ Train the baseline ML model
python3 scripts/train_baseline.py

# 5️⃣ Launch AgroVision dashboard
streamlit run app/main.py

🖼️ Screenshots
📊 Dashboard – Environmental Trends

💧 Dashboard – ML-based Irrigation Recommendation



🛠️ Tech Stack

Languages & Frameworks: Python, Streamlit
Libraries: Pandas · NumPy · scikit-learn · Plotly · Joblib
Version Control: Git · GitHub

🗺️ Roadmap

✅ IoT data simulation & baseline model

🔄 Advanced ML models (XGBoost, RandomForest)

🧩 Complete Streamlit interface with export options

🎥 Publish demo video on YouTube

👩‍💻 Author

Navya M
📍 Master’s in Computer & Information Science, Harrisburg University
📅 © 2025 AgroVision | Precision Farming Prototype