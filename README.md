# Yellow Card Predictions

A machine learning-powered platform for predicting yellow cards in professional football using historical match data, referee behavior, player statistics, and betting odds.

---

## Overview

Yellow Card Predictions analyzes historical football matches to estimate the probability of bookings before a match is played.

The project combines machine learning with referee tendencies, player booking history, betting odds, and match context to generate predictions and identify potential betting value.

---

## Features

- Yellow card probability predictions
- Referee tendency analysis
- Player booking-risk profiles
- Historical backtesting
- Betting value detection
- Interactive Streamlit dashboard
- Support for multiple European leagues

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Streamlit
- Plotly
- Requests
- python-dotenv

---

## Project Structure

```text
yellow-card-predictions/
│
├── data/
│   ├── Historical datasets
│   ├── Prediction outputs
│   ├── Referee profiles
│   ├── Player profiles
│   └── Backtesting results
│
├── src/
│   ├── dashboard.py
│   ├── train_model.py
│   ├── predict_upcoming.py
│   ├── backtest.py
│   ├── model.py
│   ├── player_risk.py
│   ├── odds.py
│   └── ...
│
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/saicanoe/yellow-card-predictions.git
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

**Windows**

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch the dashboard:

```bash
streamlit run src/dashboard.py
```

Train the model:

```bash
python src/train_model.py
```

Generate predictions:

```bash
python src/predict_upcoming.py
```

Run historical backtesting:

```bash
python src/backtest.py
```

---

## Future Improvements

- Additional football leagues
- Expected card probability calibration
- Team-level prediction models
- Automated data updates
- Improved betting strategy evaluation

---

## Disclaimer

This project was created for educational and analytical purposes.

Predictions are probabilistic estimates and should not be interpreted as guaranteed outcomes.