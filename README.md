# AgroVision
IoT-style data + ML decision-support for irrigation recommendations.

## Quickstart
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/simulate_data.py
python3 scripts/train_baseline.py
python -m streamlit run app/main.py
