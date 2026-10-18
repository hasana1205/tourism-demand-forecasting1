# Tourism Demand Forecasting Website

A Streamlit college-project starter that combines:
- **Streamlit**: website and dashboard UI
- **Open-Meteo Weather API**: live weather for selected destinations (no API key required)
- **Python chatbot logic**: answers common questions using the loaded tourism data
- **Scikit-learn**: Random Forest or Linear Regression forecasting
- **Pandas**: data cleaning and processing
- **Plotly**: interactive graphs

## Run on your computer

1. Install Python 3.10 or newer.
2. Extract this ZIP file.
3. Open a terminal/command prompt in the extracted folder.
4. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the website:

   ```bash
   streamlit run app.py
   ```

6. Streamlit will show a local address (usually `http://localhost:8501`). Open it in your browser.

## Use real tourism data

Upload a CSV using the sidebar. It must contain these columns:

```csv
Year,Month,Tourist_Arrivals
2022,1,12500
2022,2,13100
2022,3,14800
```

Use at least 12 valid monthly rows; more years are better. The included data is **simulated demo data**, not official tourist statistics.

## Publish a shareable website

To get a public link, create a GitHub repository, upload `app.py` and `requirements.txt`, then deploy the repository with Streamlit Community Cloud. Select `app.py` as the main file. The weather tab requires internet access at runtime.

## Notes

- Forecasts are demonstrations and should not be treated as official predictions.
- The chatbot uses simple rule-based Python logic; it is not a generative AI chatbot.
- Weather condition is displayed as the API's numeric weather code.
