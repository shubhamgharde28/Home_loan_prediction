# Home Loan Prediction

A Streamlit demo for the home-loan classification notebook. It trains the
notebook's tuned random-forest classifier from `train.csv` and provides a form
for trying applicant details.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app needs `train.csv` in the project root. The separate `test.csv` file is
not needed for interactive predictions.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, and `train.csv` to a GitHub
   repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with
   GitHub and create an app from that repository.
3. Set the app's main file path to `app.py` and deploy.

Only publish the CSV if you have permission to share its contents. Do not
commit real applicant or other sensitive financial data to a public repository.

This is an educational sample model, not a real lending or underwriting
system. Its predictions should not be used to make financial decisions.
