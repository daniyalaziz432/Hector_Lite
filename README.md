# HECTOR — Crypto Intelligence Dashboard

HECTOR is a crypto dashboard I built to pull real market data, run it through
some machine learning, and actually show whether the signals it generates are
any good — not just pretty charts.

Point it at any of ~25 major coins and it'll fetch live price data, compute
30+ technical indicators, train a small model ensemble to predict short-term
direction, size bets with a Kelly-based approach, and then backtest the whole
thing honestly — fees, slippage, and all. There's also a risk tab (VaR, CVaR,
stress scenarios), a portfolio optimizer (HRP + efficient frontier), some
quant research tools (CUSUM, entropy, fractional differentiation), and a
sentiment tab pulling from Reddit and crypto news.

Everything runs on free, no-key-required APIs — yfinance, CoinGecko,
alternative.me's Fear & Greed index, Reddit's public JSON endpoint, and
CryptoCompare.

## Heads up

This is a research and learning tool, not financial advice. The backtests are
realistic but backtests are still backtests — nothing here guarantees future
performance. Don't trade real money based solely on what this dashboard tells
you.

## Getting it running

```bash
pip install -r requirements.txt
streamlit run hector.py
```

That's it — no API keys, no config files, no secrets to set up.

## What's inside

- **Market Overview** — live candles, RSI/MACD, fear & greed, structural
  break detection
- **Signals** — triple-barrier labeling and the current long/short/flat call
- **Models** — how the ensemble is actually performing (F1, AUC, confusion
  matrix, feature importance)
- **Backtest** — equity curve, drawdown, Sharpe/Sortino/PSR/DSR, walk-forward
  validation
- **Risk** — VaR, CVaR, volatility, stress scenarios
- **Portfolio** — hierarchical risk parity weights across whatever coins you
  pick to compare
- **Research** — entropy, fractional diff, CUSUM — the more experimental
  quant stuff
- **Sentiment** — Reddit mood and recent headlines
- **Export** — download the data, labels, signals, or a JSON run summary

## Why it's a single file

It's built to drop straight into a Hugging Face Space with zero fuss — one
`hector.py`, one `requirements.txt`, no extra setup. The code itself is
organized into clear classes (data fetching, feature engineering, modeling,
backtesting, etc.) so it reads more like a small library than a script, even
though it lives in one place.

## Credit

Built by Daniyal Aziz.
