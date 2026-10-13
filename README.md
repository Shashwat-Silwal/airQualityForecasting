# Tomorrow's Air

Streamlit app over the airQualityForecasting notebook: will tomorrow's PM2.5 exceed 15 µg/m³?

    pip install -r requirements.txt
    streamlit run app.py

- `model/` – tuned XGBoost classifier + the 19 feature names 
- `data/history.csv.gz` – used for explore the data tab
- `data/derivation_ratios.json` – used to fill in PM10 features

