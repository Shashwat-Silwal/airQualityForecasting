# Tomorrow's Air

Streamlit app over the airQualityForecasting notebook: will tomorrow's PM2.5 exceed 15 µg/m³?

    pip install -r requirements.txt
    streamlit run app.py

- `model/` – tuned XGBoost classifier + the 19 feature names (from the notebook / inference.py)
- `data/history.csv.gz` – slimmed LAWA dataset (used for "Try a real day" and "Explore the data")
- `data/derivation_ratios.json` – seasonal PM10:PM2.5 ratios used to fill in PM10 features
- `build_data.py` – regenerates both data files from forecasting_dataset.xls
