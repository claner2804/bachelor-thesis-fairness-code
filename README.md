# Bachelor Thesis Fairness

Starterprojekt für die Bachelorarbeit **"Fairness in Machine Learning: Eine visuelle Simulation von Bias-Effekten in KI-Entscheidungen"**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Start

Notebook:
```bash
jupyter notebook
```

Streamlit-App:
```bash
streamlit run app/streamlit_app.py
```

## Projektidee
Pipeline: **Daten -> Bias-Injektion -> Modell -> Vorhersage -> Fairness-Metriken -> Visualisierung**
