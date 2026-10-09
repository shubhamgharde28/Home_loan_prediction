# Home Loan Prediction

A Streamlit demo for the home-loan classification notebook. It trains the
notebook's tuned random-forest classifier from `train.csv` and includes an
overview dashboard, all of the notebook's graphs, an interactive prediction
form, and searchable/downloadable training and test data.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app needs both `train.csv` and `test.csv` in the project root. The training
sample powers the model and approval charts; the test sample powers the
notebook's test-data feature charts.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, `train.csv`, and `test.csv`
   to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with
   GitHub and create an app from that repository.
3. Set the app's main file path to `app.py` and deploy.

Only publish the CSV if you have permission to share its contents. Do not
commit real applicant or other sensitive financial data to a public repository.

This is an educational sample model, not a real lending or underwriting
system. Its predictions should not be used to make financial decisions.
