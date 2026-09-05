import streamlit as st

st.set_page_config(page_title="HECTOR — Crypto Intelligence", page_icon="⬡",
                    layout="wide", initial_sidebar_state="expanded")

import re
import json
import time
import sqlite3
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, confusion_matrix, roc_curve)

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("HECTOR")


# ───────────────────────────── CONFIG ─────────────────────────────

class Config:
    DB_PATH = "hector.db"
    HTTP_TIMEOUT = 12
    RETRIES = 3
    BACKOFF = 1.5
    CACHE_OHLCV = 180
    CACHE_INFO = 300
    CACHE_GLOBAL = 360
    MAX_LABELS = 100
    CV_SPLITS = 3
    PURGE_EMBARGO = 2
    KELLY_FRACTION = 0.25
    COMMISSION = 0.001
    SLIPPAGE = 0.0005
    CUSUM_THR = 0.02
    ENTROPY_WINDOW = 60


COINS: Dict[str, Tuple[str, str]] = {
    "Bitcoin (BTC)":      ("bitcoin", "BTC-USD"),
    "Ethereum (ETH)":     ("ethereum", "ETH-USD"),
    "BNB":                ("binancecoin", "BNB-USD"),
    "Solana (SOL)":       ("solana", "SOL-USD"),
    "XRP":                ("ripple", "XRP-USD"),
    "Dogecoin (DOGE)":    ("dogecoin", "DOGE-USD"),
    "Cardano (ADA)":      ("cardano", "ADA-USD"),
    "Avalanche (AVAX)":   ("avalanche-2", "AVAX-USD"),
    "TRON (TRX)":         ("tron", "TRX-USD"),
    "Polkadot (DOT)":     ("polkadot", "DOT-USD"),
    "Chainlink (LINK)":   ("chainlink", "LINK-USD"),
    "Polygon (MATIC)":    ("matic-network", "MATIC-USD"),
    "Litecoin (LTC)":     ("litecoin", "LTC-USD"),
    "Shiba Inu (SHIB)":   ("shiba-inu", "SHIB-USD"),
    "Bitcoin Cash (BCH)": ("bitcoin-cash", "BCH-USD"),
    "Uniswap (UNI)":      ("uniswap", "UNI-USD"),
    "NEAR Protocol":      ("near", "NEAR-USD"),
    "Cosmos (ATOM)":      ("cosmos", "ATOM-USD"),
    "Aave":               ("aave", "AAVE-USD"),
    "Algorand (ALGO)":    ("algorand", "ALGO-USD"),
    "Stellar (XLM)":      ("stellar", "XLM-USD"),
    "Monero (XMR)":       ("monero", "XMR-USD"),
    "Hedera (HBAR)":      ("hedera-hashgraph", "HBAR-USD"),
    "Toncoin (TON)":      ("the-open-network", "TON-USD"),
    "Sui (SUI)":          ("sui", "SUI-USD"),
}

FEATURE_COLS = [
    "r1", "r5", "r10", "r20",
    "vol_5", "vol_20", "ewma_vol",
    "rsi", "macd", "macd_hist", "bb_pct", "bb_bw",
    "atr_pct", "stoch_k", "mfi", "willr", "cci",
    "trend_up", "above_vwap", "vol_ratio",
    "ofi", "cvd_norm", "frac_d35",
]

PALETTE = dict(
    bg="#0a0b0d", panel="#080d16", card="#12141a", border="#1e2130",
    orange="#ff8c00", green="#00d48a", red="#ff3d5a", blue="#0088ff",
    yellow="#ffd700", purple="#9b6dff", cyan="#00cfff", muted="#4a5270",
    text="#e8eaf0", sec="#8892aa",
)
PCONF = dict(displayModeBar=False)


# ───────────────────────────── DATABASE ─────────────────────────────

class Database:

    def __init__(self, path: str = Config.DB_PATH):
        self.path = path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init(self) -> None:
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT, ts TEXT,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (symbol, ts)
            );
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, symbol TEXT, model_name TEXT,
                cv_f1 REAL, auc REAL, accuracy REAL, precision_ REAL, recall_ REAL
            );
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, symbol TEXT, signal INTEGER, prob REAL, bet_size REAL, model TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ohlcv ON ohlcv(symbol, ts);
        """)
        conn.commit()
        conn.close()

    def upsert_ohlcv(self, symbol: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        try:
            conn = self._conn()
            rows = [
                (symbol, str(idx), float(r["Open"]), float(r["High"]),
                 float(r["Low"]), float(r["Close"]), float(r["Volume"]))
                for idx, r in df.iterrows()
            ]
            conn.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?)", rows)
            conn.commit()
            conn.close()
        except Exception as exc:
            log.debug("upsert_ohlcv: %s", exc)

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        try:
            conn = self._conn()
            df = pd.read_sql("SELECT * FROM ohlcv WHERE symbol=? ORDER BY ts",
                              conn, params=(symbol,))
            conn.close()
            if df.empty:
                return pd.DataFrame()
            df["ts"] = pd.to_datetime(df["ts"])
            if df["ts"].dt.tz is not None:
                df["ts"] = df["ts"].dt.tz_localize(None)
            df = df.set_index("ts").rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume"})
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            return pd.DataFrame()

    def log_metric(self, symbol: str, name: str, m: dict) -> None:
        try:
            conn = self._conn()
            conn.execute(
                "INSERT INTO model_metrics(ts,symbol,model_name,cv_f1,auc,accuracy,precision_,recall_)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (datetime.utcnow().isoformat(), symbol, name,
                 m.get("cv_f1", 0), m.get("auc", 0), m.get("accuracy", 0),
                 m.get("precision", 0), m.get("recall", 0)))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def log_signal(self, symbol: str, sig: dict) -> None:
        try:
            conn = self._conn()
            conn.execute(
                "INSERT INTO signal_history(ts,symbol,signal,prob,bet_size,model) VALUES(?,?,?,?,?,?)",
                (datetime.utcnow().isoformat(), symbol, sig.get("signal", 0),
                 sig.get("prob", 0.5), sig.get("bet_size", 0), sig.get("model", "")))
            conn.commit()
            conn.close()
        except Exception:
            pass


DB = Database()


# ───────────────────────────── HTTP CLIENT ─────────────────────────────

class HttpClient:
    _session = requests.Session()
    _session.headers.update({
        "Accept": "application/json, text/xml, */*",
        "User-Agent": "Mozilla/5.0 HECTOR/OOP (+github.com)",
    })

    @classmethod
    def get(cls, url: str, params: dict = None, timeout: int = None,
            as_text: bool = False) -> Optional[Any]:
        to = timeout or Config.HTTP_TIMEOUT
        for attempt in range(Config.RETRIES):
            try:
                r = cls._session.get(url, params=params, timeout=to)
                if r.status_code == 429:
                    time.sleep(min(20 * (attempt + 1), 60))
                    continue
                if r.status_code == 200:
                    return r.text if as_text else r.json()
            except Exception as exc:
                log.debug("GET %s attempt %d: %s", url, attempt + 1, exc)
                if attempt < Config.RETRIES - 1:
                    time.sleep(Config.BACKOFF ** attempt)
        return None


# ───────────────────────────── SANITIZER ─────────────────────────────

class Sanitizer:

    @staticmethod
    def array(arr: np.ndarray, clip: float = 1e6) -> np.ndarray:
        arr = np.where(np.isfinite(arr), arr, 0.0)
        return np.clip(arr, -clip, clip)

    @staticmethod
    def frame(df: pd.DataFrame, clip: float = 1e6) -> np.ndarray:
        raw = df.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)
        return np.clip(raw, -clip, clip)

    @staticmethod
    def winsorize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.replace([np.inf, -np.inf], np.nan).copy()
        for col in out.columns:
            q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
            iqr = q3 - q1
            out[col] = out[col].clip(lower=q1 - 10 * iqr, upper=q3 + 10 * iqr)
        return out.fillna(out.median()).fillna(0)


# ───────────────────────────── DATA FEED ─────────────────────────────

class DataFeed:

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_OHLCV, show_spinner=False)
    def ohlcv(yf_symbol: str, period: str = "3mo", interval: str = "1h") -> pd.DataFrame:
        import yfinance as yf
        try:
            raw = yf.download(yf_symbol, period=period, interval=interval,
                               progress=False, auto_adjust=True, threads=False)
            if raw is None or raw.empty:
                return DB.load_ohlcv(yf_symbol)
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            df = raw[["Open", "High", "Low", "Close", "Volume"]].ffill().bfill().dropna()
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            DB.upsert_ohlcv(yf_symbol, df)
            return df
        except Exception as exc:
            log.error("ohlcv %s: %s", yf_symbol, exc)
            return DB.load_ohlcv(yf_symbol)

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_OHLCV, show_spinner=False)
    def multi_ohlcv(symbols: tuple, period: str = "3mo") -> Dict[str, pd.DataFrame]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        out: Dict[str, pd.DataFrame] = {}

        def _one(sym):
            return sym, DataFeed.ohlcv(sym, period=period, interval="1d")

        with ThreadPoolExecutor(max_workers=min(5, len(symbols))) as pool:
            futs = {pool.submit(_one, s): s for s in symbols}
            for fut in as_completed(futs):
                try:
                    sym, df = fut.result()
                    if not df.empty:
                        out[sym] = df
                except Exception:
                    pass
        return out

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_INFO, show_spinner=False)
    def coin_info(cg_id: str) -> dict:
        empty = dict(name=cg_id, symbol=cg_id.upper()[:6], price=0, market_cap=0,
                     volume_24h=0, change_24h=0, change_7d=0, high_24h=0, low_24h=0)
        data = HttpClient.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}",
                               params={"localization": "false", "tickers": "false",
                                       "market_data": "true", "community_data": "false",
                                       "developer_data": "false"})
        if not isinstance(data, dict):
            return empty
        md = data.get("market_data") or {}

        def _u(f):
            return (md.get(f) or {}).get("usd") or 0

        return dict(
            name=data.get("name", ""), symbol=(data.get("symbol") or "").upper(),
            price=_u("current_price"), market_cap=_u("market_cap"),
            volume_24h=_u("total_volume"),
            change_24h=md.get("price_change_percentage_24h") or 0,
            change_7d=md.get("price_change_percentage_7d") or 0,
            high_24h=_u("high_24h"), low_24h=_u("low_24h"),
        )

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_GLOBAL, show_spinner=False)
    def global_market() -> dict:
        data = HttpClient.get("https://api.coingecko.com/api/v3/global")
        if not isinstance(data, dict):
            return {}
        d = data.get("data") or {}
        return dict(
            total_mcap=(d.get("total_market_cap") or {}).get("usd", 0),
            btc_dom=(d.get("market_cap_percentage") or {}).get("btc", 0),
            eth_dom=(d.get("market_cap_percentage") or {}).get("eth", 0),
        )

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_GLOBAL, show_spinner=False)
    def fear_greed() -> dict:
        default = {"current": 50, "label": "NEUTRAL", "history": [50] * 30}
        data = HttpClient.get("https://api.alternative.me/fng/", params={"limit": 30})
        if not isinstance(data, dict):
            return default
        items = data.get("data") or []
        if not items:
            return default
        lm = {"Extreme Fear": "EXTREME FEAR", "Fear": "FEAR", "Neutral": "NEUTRAL",
              "Greed": "GREED", "Extreme Greed": "EXTREME GREED"}
        raw = items[0].get("value_classification", "Neutral")
        return {"current": int(items[0].get("value", 50)),
                "label": lm.get(raw, raw.upper()),
                "history": [int(x.get("value", 50)) for x in reversed(items)]}

    @staticmethod
    @st.cache_data(ttl=900, show_spinner=False)
    def news() -> List[dict]:
        data = HttpClient.get("https://min-api.cryptocompare.com/data/v2/news/",
                               params={"lang": "EN", "sortOrder": "latest"})
        if isinstance(data, dict) and isinstance(data.get("Data"), list):
            pos_w = ["bull", "surge", "rally", "gain", "rise", "buy", "moon"]
            neg_w = ["bear", "crash", "drop", "fall", "dump", "sell", "fear"]
            out = []
            for n in data["Data"][:12]:
                title = n.get("title", "")
                tl = title.lower()
                pc = sum(1 for w in pos_w if w in tl)
                nc = sum(1 for w in neg_w if w in tl)
                sent = "bullish" if pc > nc else "bearish" if nc > pc else "neutral"
                out.append({"title": title, "sentiment": sent, "url": n.get("url", "")})
            if out:
                return out
        return []

    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)
    def reddit_sentiment(subreddit: str = "CryptoCurrency") -> dict:
        default = {"score": 0.0, "label": "Neutral", "count": 0}
        data = HttpClient.get(f"https://www.reddit.com/r/{subreddit}/hot.json",
                               params={"limit": 50})
        if not isinstance(data, dict):
            return default
        posts = (data.get("data") or {}).get("children") or []
        texts = [p.get("data", {}).get("title", "") for p in posts]
        if not texts:
            return default
        pos_w = ["bull", "moon", "pump", "buy", "long", "ath", "surge", "green"]
        neg_w = ["bear", "dump", "crash", "sell", "short", "fear", "red", "loss"]
        scores = []
        for t in texts:
            tl = t.lower()
            p = sum(1 for w in pos_w if w in tl)
            n = sum(1 for w in neg_w if w in tl)
            scores.append(float(np.clip((p - n) / max(p + n, 1), -1, 1)) if (p + n) else 0.0)
        avg = float(np.mean(scores))
        label = "Bullish" if avg > 0.05 else "Bearish" if avg < -0.05 else "Neutral"
        return {"score": round(avg, 3), "label": label, "count": len(texts)}


# ───────────────────────────── FEATURE ENGINEER ─────────────────────────────

class FeatureEngineer:

    @staticmethod
    def validate(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        if df.empty:
            return df, {}
        n = len(df)
        dupes = df.index.duplicated(keep="last").sum()
        df = df[~df.index.duplicated(keep="last")].copy()
        ret = df["Close"].squeeze().pct_change()
        mu_, sd_ = float(ret.mean()), float(ret.std())
        mask = (ret.abs() > mu_ + 5 * sd_) & (sd_ > 0)
        df["Close"] = df["Close"].where(~mask, other=df["Close"].shift(1))
        df = df.ffill(limit=3).dropna(subset=["Close"])
        return df, {"duplicates": int(dupes), "outliers": int(mask.sum()),
                     "rows_before": n, "rows_after": len(df)}

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_OHLCV, show_spinner=False)
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 30:
            return df
        df = df.copy()
        c, h, lo, v = df["Close"].squeeze(), df["High"].squeeze(), df["Low"].squeeze(), df["Volume"].squeeze()

        df["returns"] = c.pct_change().replace([np.inf, -np.inf], np.nan)
        df["log_returns"] = np.log((c.replace(0, np.nan) / c.shift(1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan))

        df["ema9"] = c.ewm(span=9, adjust=False).mean()
        df["ema21"] = c.ewm(span=21, adjust=False).mean()
        df["sma50"] = c.rolling(50).mean()

        bb_mid, bb_std = c.rolling(20).mean(), c.rolling(20).std()
        df["bb_up"] = bb_mid + 2 * bb_std
        df["bb_dn"] = bb_mid - 2 * bb_std
        df["bb_pct"] = (c - df["bb_dn"]) / (df["bb_up"] - df["bb_dn"] + 1e-10)
        df["bb_bw"] = (df["bb_up"] - df["bb_dn"]) / (bb_mid + 1e-10)

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-10))

        ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        lo14, hi14 = lo.rolling(14).min(), h.rolling(14).max()
        df["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14 + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / (c + 1e-10) * 100

        df["obv"] = (v * np.sign(c.diff().fillna(0))).cumsum()
        tp = (h + lo + c) / 3
        df["vwap"] = (tp * v).cumsum() / (v.cumsum() + 1e-10)

        df["vol_5"] = df["returns"].rolling(5).std() * np.sqrt(365 * 24)
        df["vol_20"] = df["returns"].rolling(20).std() * np.sqrt(365 * 24)
        df["ewma_vol"] = df["returns"].ewm(span=30, adjust=False).std() * np.sqrt(365 * 24)

        df["trend_up"] = (df["ema9"] > df["ema21"]).astype(int)
        df["above_vwap"] = (c > df["vwap"]).astype(int)

        mf_raw = tp * v
        mf_pos = mf_raw.where(tp > tp.shift(1), 0)
        mf_neg = mf_raw.where(tp < tp.shift(1), 0)
        mfr = mf_pos.rolling(14).sum() / (mf_neg.rolling(14).sum() + 1e-10)
        df["mfi"] = 100 - 100 / (1 + mfr)
        df["willr"] = -100 * (hi14 - c) / (hi14 - lo14 + 1e-10)
        mdev = (tp - tp.rolling(20).mean()).abs().rolling(20).mean()
        df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * mdev + 1e-10)

        buy_vol = v.where(c >= df["Open"].squeeze(), 0)
        sell_vol = v.where(c < df["Open"].squeeze(), 0)
        df["cvd"] = (buy_vol - sell_vol).cumsum()
        df["cvd_ma"] = df["cvd"].ewm(span=20).mean()
        df["ofi"] = (buy_vol - sell_vol) / (v + 1e-10)

        vol_ma = v.rolling(20).mean()
        df["vol_ratio"] = (v / (vol_ma + 1e-10)).replace([np.inf, -np.inf], 1.0)

        return df.ffill().bfill().dropna(subset=["returns"])

    @staticmethod
    def regime(returns: pd.Series) -> str:
        r = returns.dropna()
        if r.empty:
            return "sideways"
        rm = r.rolling(20, min_periods=1).mean().iloc[-1]
        if pd.isna(rm):
            return "sideways"
        return "bull" if rm > 0.001 else "bear" if rm < -0.001 else "sideways"


# ───────────────────────────── LABELING ─────────────────────────────

class Labeler:

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_OHLCV, show_spinner=False)
    def triple_barrier(df: pd.DataFrame, pt: float = 1.5, sl: float = 1.0,
                        max_hold: int = 24) -> pd.DataFrame:
        if df.empty or "returns" not in df.columns:
            return pd.DataFrame()
        c = df["Close"].squeeze()
        vol = df["returns"].rolling(20).std().bfill()
        n = len(c)
        step = max(1, n // Config.MAX_LABELS)
        rows = []
        for i in range(20, n - max_hold, step):
            price, v_ = float(c.iloc[i]), float(vol.iloc[i])
            upper, lower = price * (1 + pt * v_), price * (1 - sl * v_)
            end = min(i + max_hold, n - 1)
            label = 0
            for j in range(i, end + 1):
                p = float(c.iloc[j])
                if p >= upper:
                    label = 1
                    break
                if p <= lower:
                    label = -1
                    break
            ret = float(c.iloc[end] / price) - 1
            if label == 0:
                label = 1 if ret > 0 else -1 if ret < 0 else 0
            rows.append(dict(idx=i, time=c.index[i], label=label, ret=ret,
                              vol_entry=v_, pt=upper, sl=lower))
        return pd.DataFrame(rows)


# ───────────────────────────── FEATURE BUILDER ─────────────────────────────

class FeatureBuilder:

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_OHLCV, show_spinner=False)
    def build(df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or labels_df.empty:
            return pd.DataFrame()
        c, v_s = df["Close"].squeeze(), df["Volume"].squeeze()
        cvd_ = df["cvd"] if "cvd" in df.columns else pd.Series(0, index=df.index)
        cvd_max = float(cvd_.abs().max()) + 1e-10
        rows = []

        for _, row in labels_df.iterrows():
            i = int(row["idx"])
            if i < 20 or i >= len(df):
                continue
            lp = np.log(c.iloc[max(0, i - 20):i + 1].values + 1e-10)
            w = [1.0, -0.35, -0.35 * 0.65 / 2, -0.35 * 0.65 * 0.3 / 6, -0.35 * 0.65 * 0.3 * 0.025 / 24]
            frac = float(sum(a * lp[-(k + 1)] for k, a in enumerate(w) if k < len(lp)))
            va5 = float(v_s.iloc[max(0, i - 5):i].mean()) + 1e-10

            def _g(col, default=0.0):
                try:
                    return float(df[col].iloc[i]) if col in df.columns else default
                except Exception:
                    return default

            rows.append(dict(
                label=int(row["label"]), ret=float(row["ret"]),
                r1=float(c.iloc[i] / c.iloc[i - 1] - 1) if i >= 1 else 0.0,
                r5=float(c.iloc[i] / c.iloc[i - 5] - 1) if i >= 5 else 0.0,
                r10=float(c.iloc[i] / c.iloc[i - 10] - 1) if i >= 10 else 0.0,
                r20=float(c.iloc[i] / c.iloc[i - 20] - 1) if i >= 20 else 0.0,
                vol_5=_g("vol_5"), vol_20=_g("vol_20"), ewma_vol=_g("ewma_vol"),
                rsi=_g("rsi", 50), macd=_g("macd"), macd_hist=_g("macd_hist"),
                bb_pct=_g("bb_pct", 0.5), bb_bw=_g("bb_bw"),
                atr_pct=_g("atr_pct"), stoch_k=_g("stoch_k", 50),
                mfi=_g("mfi", 50), willr=_g("willr", -50), cci=_g("cci"),
                trend_up=_g("trend_up"), above_vwap=_g("above_vwap"),
                vol_ratio=float(v_s.iloc[i]) / va5, ofi=_g("ofi"),
                cvd_norm=float(cvd_.iloc[i]) / cvd_max, frac_d35=frac,
            ))

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).replace([np.inf, -np.inf], 0.0).fillna(0.0)


# ───────────────────────────── MODEL ENGINE ─────────────────────────────

class ModelEngine:

    @staticmethod
    def _purged_splits(n: int, n_splits: int = Config.CV_SPLITS, embargo: int = Config.PURGE_EMBARGO):
        fold = max(1, n // n_splits)
        for k in range(n_splits):
            ts, te = k * fold, (k + 1) * fold if k < n_splits - 1 else n
            tr = list(range(0, max(0, ts - embargo))) + list(range(min(n, te + embargo), n))
            va = list(range(ts, te))
            if tr and va:
                yield tr, va

    @staticmethod
    def _remove_outliers(feat_df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
        if feat_df.empty or len(feat_df) < 30:
            return feat_df
        try:
            X = Sanitizer.winsorize(feat_df[FEATURE_COLS]).values
            iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            mask = iso.fit_predict(X) == 1
            return feat_df[mask].copy()
        except Exception:
            return feat_df

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_OHLCV, show_spinner=False)
    def train(feat_df: pd.DataFrame, symbol: str = "BTC-USD") -> dict:
        df = feat_df[feat_df["label"] != 0].copy()
        if len(df) < 40:
            return {}
        df = ModelEngine._remove_outliers(df)
        if len(df) < 30:
            return {}

        avail = [c for c in FEATURE_COLS if c in df.columns]
        if len(avail) < 5:
            return {}
        X = Sanitizer.winsorize(df[avail]).values.astype(float)
        y = (df["label"] == 1).astype(int).values
        sc = StandardScaler()
        Xs = sc.fit_transform(X)

        n = len(Xs)
        split = int(n * 0.70)
        X_tr, X_te, y_tr, y_te = Xs[:split], Xs[split:], y[:split], y[split:]
        cv_splits = list(ModelEngine._purged_splits(split))

        models_def = {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5,
                                                      random_state=42, n_jobs=-1, class_weight="balanced"),
            "Logistic": LogisticRegression(max_iter=500, random_state=42, class_weight="balanced"),
        }

        results, voting_ests, voting_wts = {}, [], []
        for name, model in models_def.items():
            try:
                cv_f1 = []
                for tr_idx, va_idx in cv_splits:
                    try:
                        model.fit(Xs[tr_idx], y[tr_idx])
                        cv_f1.append(f1_score(y[va_idx], model.predict(Xs[va_idx]), zero_division=0))
                    except Exception:
                        cv_f1.append(0.0)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else np.full(len(y_te), 0.5)
                imp = getattr(model, "feature_importances_", None)
                if imp is None:
                    coef = getattr(model, "coef_", None)
                    imp = np.abs(coef[0]) if coef is not None else np.zeros(len(avail))
                cv_f1_mean = float(np.mean(cv_f1)) if cv_f1 else 0.0
                metrics = dict(
                    model=model, scaler=sc, cv_f1=cv_f1_mean,
                    cv_std=float(np.std(cv_f1)) if cv_f1 else 0.0,
                    accuracy=float((y_pred == y_te).mean()),
                    f1=float(f1_score(y_te, y_pred, zero_division=0)),
                    precision=float(precision_score(y_te, y_pred, zero_division=0)),
                    recall=float(recall_score(y_te, y_pred, zero_division=0)),
                    auc=float(roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5),
                    confusion=confusion_matrix(y_te, y_pred).tolist(),
                    importance=list(imp), y_te=y_te.tolist(), y_pred=y_pred.tolist(), y_prob=y_prob.tolist(),
                )
                results[name] = metrics
                DB.log_metric(symbol, name, metrics)
                voting_ests.append((name, model))
                voting_wts.append(cv_f1_mean)
            except Exception as exc:
                log.warning("model %s failed: %s", name, exc)

        if len(voting_ests) >= 2:
            try:
                norm_wts = [w / (sum(voting_wts) + 1e-10) for w in voting_wts]
                voter = VotingClassifier(estimators=voting_ests, voting="soft", weights=norm_wts)
                voter.fit(X_tr, y_tr)
                yv_pred, yv_prob = voter.predict(X_te), voter.predict_proba(X_te)[:, 1]
                results["Ensemble"] = dict(
                    model=voter, scaler=sc, cv_f1=float(np.average(voting_wts, weights=voting_wts)),
                    cv_std=0.0, accuracy=float((yv_pred == y_te).mean()),
                    f1=float(f1_score(y_te, yv_pred, zero_division=0)),
                    precision=float(precision_score(y_te, yv_pred, zero_division=0)),
                    recall=float(recall_score(y_te, yv_pred, zero_division=0)),
                    auc=float(roc_auc_score(y_te, yv_prob) if len(np.unique(y_te)) > 1 else 0.5),
                    confusion=confusion_matrix(y_te, yv_pred).tolist(),
                    importance=[0.0] * len(avail),
                    y_te=y_te.tolist(), y_pred=yv_pred.tolist(), y_prob=yv_prob.tolist(),
                )
            except Exception as exc:
                log.warning("ensemble failed: %s", exc)

        return results


# ───────────────────────────── SIGNAL ENGINE ─────────────────────────────

class SignalEngine:

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_OHLCV, show_spinner=False)
    def generate(df: pd.DataFrame, labels_df: pd.DataFrame, feat_df: pd.DataFrame,
                 _model_results: dict, confidence_thr: float = 0.55,
                 regime: str = "bull") -> pd.DataFrame:
        mr_all = _model_results
        if not mr_all or feat_df.empty or "label" not in feat_df.columns:
            return pd.DataFrame()

        best_name = "Ensemble" if "Ensemble" in mr_all else max(mr_all, key=lambda k: mr_all[k]["cv_f1"])
        mr, mdl, sc = mr_all[best_name], mr_all[best_name]["model"], mr_all[best_name]["scaler"]

        X_raw = Sanitizer.winsorize(feat_df[FEATURE_COLS])
        X = X_raw.values.astype(float)
        if X.shape[1] != sc.n_features_in_:
            return pd.DataFrame()
        Xs = sc.transform(X)
        probs = mdl.predict_proba(Xs)[:, 1] if hasattr(mdl, "predict_proba") else np.full(len(X), 0.5)

        wins = float(mr.get("precision", 0.55))
        b = wins / max(1 - wins, 0.001)
        kelly = float(np.clip(Config.KELLY_FRACTION * (b * wins - (1 - wins)) / max(b, 0.001), 0.0, 1.0))

        raw_bets = 2 * probs - 1
        smoothed = pd.Series(raw_bets).ewm(span=5, adjust=False).mean().values
        thr = 2 * confidence_thr - 1
        signals = np.where(smoothed > thr, 1, np.where(smoothed < -thr, -1, 0))
        atr_now = float(df["atr_pct"].iloc[-1]) if "atr_pct" in df.columns else 0.01

        out = pd.DataFrame({
            "time": labels_df["time"].values[:len(feat_df)],
            "label": feat_df["label"].values, "prob": probs,
            "bet_size": smoothed * kelly, "signal": signals,
            "model": best_name, "regime": regime,
            "tp_pct": atr_now * 2.0, "sl_pct": atr_now * 1.0,
        })
        if not out.empty:
            last = out.iloc[-1]
            DB.log_signal(df.index.name or "BTC-USD", {
                "signal": int(last["signal"]), "prob": float(last["prob"]),
                "bet_size": float(last["bet_size"]), "model": best_name})
        return out


# ───────────────────────────── BACKTESTER ─────────────────────────────

class Backtester:

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_OHLCV, show_spinner=False)
    def run(price: pd.Series, signals_df: pd.DataFrame, commission: float = None) -> dict:
        commission = commission or Config.COMMISSION
        if signals_df.empty or len(signals_df) < 5:
            return {}
        try:
            sig = signals_df.set_index("time")["signal"]
            if sig.index.tz is not None:
                sig.index = sig.index.tz_localize(None)
            if price.index.tz is not None:
                price = price.tz_localize(None)
            sig = sig.reindex(price.index, method="ffill").fillna(0)
            pos = sig.shift(1).fillna(0)
            raw_ret = price.pct_change().fillna(0)
            gross = raw_ret * pos
            trades = pos.diff().abs().fillna(0)
            taker = np.where(trades.abs() > 0.5, 1.5, 0.5)
            fee_c = trades * commission * taker
            vol_n = price.pct_change().abs().rolling(20).mean().fillna(0.001)
            slip_c = trades * Config.SLIPPAGE * (1 + vol_n * 10)
            net = gross - fee_c - slip_c

            cum, bh = (1 + net).cumprod(), (1 + raw_ret).cumprod()
            peak = cum.expanding().max()
            dd = (cum - peak) / (peak.abs() + 1e-10)

            ann_f = 365 * 24
            ann_ret = float(net.mean() * ann_f)
            ann_vol = float(net.std() * np.sqrt(ann_f)) + 1e-10
            sharpe = ann_ret / ann_vol
            downvol = float(net[net < 0].std() * np.sqrt(ann_f)) + 1e-10
            sortino = ann_ret / downvol
            max_dd = float(dd.min())
            calmar = abs(ann_ret / (max_dd + 1e-10))
            wins_ = float((net[pos != 0] > 0).mean()) if (pos != 0).any() else 0.5
            gp, gl = float(net[net > 0].sum()), float(abs(net[net < 0].sum()))
            pf = gp / (gl + 1e-10)
            sk, ku, T = float(net.skew()), float(net.kurtosis()), len(net)

            psr_d = np.sqrt(max(1e-10, (1 - sk * sharpe + (ku - 1) / 4 * sharpe ** 2) / max(T - 1, 1)))
            psr = float(norm.cdf(sharpe / psr_d))
            n_trials = 4
            e_max = (np.sqrt(2 * np.log(n_trials)) - (np.log(np.log(n_trials + 1e-10)) + np.log(4 * np.pi))
                     / (2 * np.sqrt(2 * np.log(n_trials)) + 1e-10))
            dsr = float(norm.cdf((sharpe - e_max) / (psr_d + 1e-10)))

            r_arr = net.dropna().values
            boot = []
            for _ in range(300):
                s = np.random.choice(r_arr, size=len(r_arr), replace=True)
                sv = s.std() * np.sqrt(ann_f) + 1e-10
                boot.append(s.mean() * ann_f / sv)
            sharpe_ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

            return dict(equity=cum, bh=bh, drawdown=dd, returns=net, ann_ret=ann_ret,
                        ann_vol=ann_vol, sharpe=sharpe, sortino=sortino, calmar=calmar,
                        max_dd=max_dd, win_rate=wins_, profit_factor=pf,
                        n_trades=int(trades.astype(bool).sum()), psr=psr, dsr=dsr,
                        sharpe_ci=sharpe_ci, total_cost=float((fee_c + slip_c).sum()))
        except Exception as exc:
            log.error("backtest: %s", exc)
            return {}

    @staticmethod
    def walk_forward(df: pd.DataFrame, signals_df: pd.DataFrame, n_folds: int = 5) -> List[dict]:
        if signals_df.empty or df.empty:
            return []
        price, n = df["Close"].squeeze(), len(df)
        fold_size, results = n // n_folds, []
        for k in range(1, n_folds):
            end_idx = (k + 1) * fold_size
            if end_idx > n:
                break
            p_fold = price.iloc[:end_idx]
            s_fold = signals_df[signals_df["time"] <= p_fold.index[-1]]
            bt = Backtester.run(p_fold, s_fold)
            if bt:
                bt["fold"] = k
                results.append(bt)
        return results


# ───────────────────────────── RISK ENGINE ─────────────────────────────

class RiskEngine:

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_OHLCV, show_spinner=False)
    def compute(returns: pd.Series) -> dict:
        r = returns.dropna()
        if len(r) < 10:
            return {}
        var95, var99 = float(np.percentile(r, 5)) * 100, float(np.percentile(r, 1)) * 100
        cvar95 = float(r[r <= np.percentile(r, 5)].mean()) * 100
        ann_vol = float(r.std() * np.sqrt(365 * 24)) * 100
        sk, ku = float(r.skew()), float(r.kurtosis())
        sr = r.mean() / (r.std() + 1e-10) * np.sqrt(365 * 24)
        denom = np.sqrt(max(1e-10, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
        psr = float(norm.cdf(sr * np.sqrt(len(r)) / denom))
        cum = (1 + r).cumprod()
        dd = (cum - cum.expanding().max()) / (cum.expanding().max().abs() + 1e-10)
        rs = (r.rolling(30).mean() / (r.rolling(30).std() + 1e-10)) * np.sqrt(365 * 24)
        stress = {
            "Flash Crash −30%": float(np.percentile(r, 1) * 30 * 100),
            "Bear Market −60%": float(np.percentile(r, 1) * 60 * 100),
            "Vol Spike ×4": float(var95 * 4),
            "Corr Breakdown": float(cvar95 * 2),
            "Liquidity Shock": float(var99 * 3),
        }
        return dict(var95=var95, cvar95=cvar95, var99=var99, ann_vol=ann_vol, psr=psr,
                    skew=sk, kurt=ku, max_dd=float(dd.min() * 100), roll_sharpe=rs,
                    drawdown=dd, returns=r, stress=stress)


# ───────────────────────────── PORTFOLIO ENGINE ─────────────────────────────

class PortfolioEngine:

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_OHLCV, show_spinner=False)
    def hrp_weights(ret_df: pd.DataFrame) -> pd.Series:
        try:
            cov = ret_df.cov()
            dist = np.sqrt((1 - ret_df.corr().clip(-1, 1)) / 2).fillna(0)
            link = linkage(squareform(dist.values), method="ward")
            n = len(ret_df.columns)

            def _ser(node: int) -> list:
                if node < n:
                    return [node]
                L, R = int(link[node - n, 0]), int(link[node - n, 1])
                return _ser(L) + _ser(R)

            ordered = [ret_df.columns[i] for i in _ser(n + n - 2)]
            wts, clusters = pd.Series(1.0, index=ordered), [list(ordered)]
            while clusters:
                nc = []
                for cl in clusters:
                    if len(cl) <= 1:
                        continue
                    mid = len(cl) // 2
                    left, right = cl[:mid], cl[mid:]

                    def _cv(cols):
                        sub = cov.loc[cols, cols].values
                        iv = np.linalg.pinv(sub)
                        w = iv.sum(axis=1) / (iv.sum() + 1e-12)
                        return float(w @ sub @ w)

                    vl, vr = _cv(left), _cv(right)
                    alpha = 1 - vl / (vl + vr + 1e-12)
                    for a in left:
                        wts[a] *= alpha
                    for a in right:
                        wts[a] *= (1 - alpha)
                    nc += [left, right]
                clusters = nc
            total = wts.sum()
            return wts / (total if total > 0 else 1)
        except Exception:
            n = len(ret_df.columns)
            return pd.Series(1.0 / n, index=ret_df.columns)

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_OHLCV, show_spinner=False)
    def efficient_frontier(ret_df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
        mu, cov, na = ret_df.mean().values, ret_df.cov().values, len(ret_df.columns)
        rows = []
        for target in np.linspace(mu.min(), mu.max(), n):
            cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                    {"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
            res = minimize(lambda w: float(w @ cov @ w), np.ones(na) / na, method="SLSQP",
                            bounds=[(0, 1)] * na, constraints=cons, options={"ftol": 1e-9, "maxiter": 150})
            if res.success:
                v = np.sqrt(res.fun)
                rows.append({"vol": v * 100, "ret": target * 100, "sharpe": target / (v + 1e-10)})
        return pd.DataFrame(rows)


# ───────────────────────────── RESEARCH TOOLS ─────────────────────────────

class ResearchTools:

    @staticmethod
    def cusum_series(prices: pd.Series) -> Tuple[list, list]:
        r = prices.pct_change().fillna(0)
        sp_arr, sn_arr = [0.0], [0.0]
        thr = Config.CUSUM_THR
        for val in r.values[1:]:
            sp_arr.append(max(0.0, sp_arr[-1] + val))
            sn_arr.append(min(0.0, sn_arr[-1] + val))
            if sp_arr[-1] > thr or sn_arr[-1] < -thr:
                sp_arr[-1] = sn_arr[-1] = 0.0
        return sp_arr, sn_arr

    @staticmethod
    def entropy_series(price: pd.Series) -> pd.Series:
        r = price.pct_change().dropna()
        enc = (r > r.median()).astype(int)
        win, ent = Config.ENTROPY_WINDOW, []
        for i in range(win, len(enc)):
            ch = enc.iloc[i - win:i]
            vc = ch.value_counts(normalize=True)
            ent.append(-sum(p * np.log2(p + 1e-10) for p in vc))
        return pd.Series(ent, index=price.index[win + 1:win + 1 + len(ent)])

    @staticmethod
    def frac_diff_series(price: pd.Series) -> pd.Series:
        lp = np.log(price.replace(0, np.nan).dropna() + 1e-10)
        w = [1.0, -0.35, -0.35 * 0.65 / 2, -0.35 * 0.65 * 0.3 / 6]
        fv = [float(sum(a * b for a, b in zip(w[::-1], lp.iloc[i - len(w):i].values)))
              for i in range(len(w), len(lp))]
        return pd.Series(fv, index=lp.index[len(w):])

    @staticmethod
    def volume_profile(df: pd.DataFrame, n_bins: int = 30) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        try:
            lo_, hi_ = float(df["Low"].min()), float(df["High"].max())
            bins = np.linspace(lo_, hi_, n_bins + 1)
            mid, vols = (bins[:-1] + bins[1:]) / 2, np.zeros(n_bins)
            for _, row in df.iterrows():
                rl, rh = min(float(row["Low"]), float(row["High"])), max(float(row["Low"]), float(row["High"]))
                for i, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
                    ov = max(0, min(b1, rh) - max(b0, rl))
                    sp = max(rh - rl, 1e-10)
                    vols[i] += float(row["Volume"]) * (ov / sp)
            vp = pd.DataFrame({"price": mid, "volume": vols})
            vp["is_hvn"] = vp["volume"] >= vp["volume"].quantile(0.80)
            return vp
        except Exception:
            return pd.DataFrame()


# ───────────────────────────── CHART FACTORY ─────────────────────────────

class ChartFactory:
    C = PALETTE

    @staticmethod
    def layout(title: str = "", h: int = 350, legend: bool = True) -> dict:
        C = ChartFactory.C
        ax = dict(gridcolor=C["border"], linecolor=C["border"], tickfont=dict(size=8),
                  zerolinecolor=C["border"], showgrid=True)
        return dict(
            paper_bgcolor=C["card"], plot_bgcolor=C["panel"],
            font=dict(family="IBM Plex Mono, monospace", color=C["sec"], size=9),
            title=dict(text=f"<b style='color:{C['orange']}'>{title}</b>", font=dict(size=10)) if title else None,
            margin=dict(l=44, r=14, t=32 if title else 10, b=28),
            xaxis=ax.copy(), yaxis=ax.copy(),
            legend=dict(bgcolor="rgba(18,20,26,.85)", bordercolor=C["border"], borderwidth=1,
                        font=dict(size=8)) if legend else dict(visible=False),
            height=h, hovermode="x unified",
            hoverlabel=dict(bgcolor=C["card"], bordercolor=C["orange"], font=dict(family="IBM Plex Mono", size=9)),
        )

    @staticmethod
    def candle(df: pd.DataFrame, title: str, show_bb: bool, show_ema: bool, show_vwap: bool,
               signals_df: pd.DataFrame = None) -> go.Figure:
        C = ChartFactory.C
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                             row_heights=[0.60, 0.20, 0.20])
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing=dict(fillcolor="rgba(0,212,138,.20)", line=dict(color=C["green"], width=1)),
            decreasing=dict(fillcolor="rgba(255,61,90,.20)", line=dict(color=C["red"], width=1)),
            name="OHLC", showlegend=False), row=1, col=1)

        if show_bb and "bb_up" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_up"], line=dict(color=C["blue"], width=.8, dash="dot"),
                                      name="BB+", hoverinfo="skip", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_dn"], line=dict(color=C["blue"], width=.8, dash="dot"),
                                      name="BB−", fill="tonexty", fillcolor="rgba(0,136,255,.04)",
                                      hoverinfo="skip", showlegend=False), row=1, col=1)
        if show_ema and "ema9" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["ema9"], line=dict(color=C["orange"], width=1.2), name="EMA9"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["ema21"], line=dict(color=C["yellow"], width=1.2), name="EMA21"), row=1, col=1)
        if show_vwap and "vwap" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["vwap"], line=dict(color=C["purple"], width=1.0, dash="dash"), name="VWAP"), row=1, col=1)

        if signals_df is not None and not signals_df.empty and "time" in signals_df.columns:
            c_price = df["Close"].squeeze()
            for t in signals_df[signals_df["signal"] == 1]["time"]:
                if t in c_price.index:
                    fig.add_annotation(x=t, y=float(c_price.loc[t]) * 0.985, text="▲", showarrow=False,
                                        font=dict(color=C["green"], size=11))
            for t in signals_df[signals_df["signal"] == -1]["time"]:
                if t in c_price.index:
                    fig.add_annotation(x=t, y=float(c_price.loc[t]) * 1.015, text="▼", showarrow=False,
                                        font=dict(color=C["red"], size=11))

        vol_colors = ["rgba(0,212,138,.55)" if float(cc) >= float(oo) else "rgba(255,61,90,.55)"
                      for cc, oo in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume", showlegend=False), row=2, col=1)

        if "rsi" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], line=dict(color=C["cyan"], width=1.2), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_color=C["red"], line_dash="dot", line_width=1, row=3, col=1)
            fig.add_hline(y=30, line_color=C["green"], line_dash="dot", line_width=1, row=3, col=1)

        ly = ChartFactory.layout(title, h=580)
        ly["xaxis"] = ChartFactory._xa()
        ly["xaxis2"] = ChartFactory._xa()
        ly["xaxis3"] = ChartFactory._xa()
        ly["yaxis3"] = dict(**ChartFactory.layout()["yaxis"], range=[0, 100])
        ly["showlegend"] = True
        fig.update_layout(**ly)
        return fig

    @staticmethod
    def _xa() -> dict:
        ax = ChartFactory.layout()["xaxis"].copy()
        ax["rangeslider"] = dict(visible=False)
        return ax

    @staticmethod
    def macd(df: pd.DataFrame) -> go.Figure:
        C = ChartFactory.C
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.03, row_heights=[.5, .5])
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], line=dict(color=C["blue"], width=1.4), name="MACD"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], line=dict(color=C["orange"], width=1.4), name="Signal"), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"],
                              marker_color=[C["green"] if v >= 0 else C["red"] for v in df["macd_hist"]],
                              showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["stoch_k"], line=dict(color=C["cyan"], width=1.2), name="Stoch%K"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["stoch_d"], line=dict(color=C["purple"], width=1.2), name="Stoch%D"), row=2, col=1)
        ly = ChartFactory.layout("MACD & Stochastic", h=340)
        ly["xaxis"], ly["xaxis2"] = ChartFactory._xa(), ChartFactory._xa()
        ly["yaxis2"] = dict(**ChartFactory.layout()["yaxis"], range=[0, 100])
        ly["showlegend"] = True
        fig.update_layout(**ly)
        return fig

    @staticmethod
    def equity(bt: dict) -> go.Figure:
        C = ChartFactory.C
        if not bt:
            return go.Figure()
        eq, bh = bt["equity"], bt["bh"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=(eq - 1) * 100, line=dict(color=C["orange"], width=2),
                                  name="HECTOR", fill="tozeroy", fillcolor="rgba(255,140,0,.06)"))
        fig.add_trace(go.Scatter(x=bh.index, y=(bh - 1) * 100, line=dict(color=C["blue"], width=1.5, dash="dash"), name="Buy & Hold"))
        fig.update_layout(**ChartFactory.layout("Equity Curve (%)", h=320))
        return fig

    @staticmethod
    def drawdown(bt: dict) -> go.Figure:
        C = ChartFactory.C
        if not bt:
            return go.Figure()
        dd = bt["drawdown"]
        fig = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100, fill="tozeroy",
                                    fillcolor="rgba(255,61,90,.20)", line=dict(color=C["red"], width=1)))
        fig.update_layout(**ChartFactory.layout("Drawdown (%)", h=220, legend=False))
        return fig

    @staticmethod
    def var_dist(risk: dict) -> go.Figure:
        C = ChartFactory.C
        if not risk:
            return go.Figure()
        r = risk["returns"] * 100
        fig = go.Figure(go.Histogram(x=r, nbinsx=60, marker_color=C["orange"], opacity=.70, name="Returns"))
        fig.add_vline(x=risk["var95"], line_color=C["red"], line_dash="dash", line_width=2)
        fig.add_vline(x=risk["cvar95"], line_color=C["purple"], line_dash="dot", line_width=1)
        fig.update_layout(**ChartFactory.layout("VaR / CVaR Distribution", h=280, legend=False))
        return fig

    @staticmethod
    def fear_gauge(fg: dict) -> go.Figure:
        C = ChartFactory.C
        val, lbl = fg.get("current", 50), fg.get("label", "Neutral")
        col = C["green"] if val > 60 else C["red"] if val < 40 else C["yellow"]
        fig = go.Figure(go.Indicator(mode="gauge+number", value=val,
            gauge=dict(axis=dict(range=[0, 100], tickfont=dict(size=8)), bar=dict(color=col, thickness=.25)),
            title=dict(text=f"Fear & Greed — {lbl}", font=dict(size=10, color=C["sec"])),
            number=dict(font=dict(color=col))))
        fig.update_layout(**ChartFactory.layout(h=220, legend=False))
        return fig

    @staticmethod
    def corr_matrix(ret_df: pd.DataFrame) -> go.Figure:
        C = ChartFactory.C
        corr = ret_df.corr()
        labs = [s.replace("-USD", "") for s in corr.columns]
        fig = go.Figure(go.Heatmap(z=corr.values, x=labs, y=labs,
            colorscale=[[0, C["red"]], [.5, C["panel"]], [1, C["green"]]], zmid=0,
            text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=8)))
        fig.update_layout(**ChartFactory.layout("Correlation Matrix", h=340, legend=False))
        return fig

    @staticmethod
    def hrp_bar(wts: pd.Series) -> go.Figure:
        C = ChartFactory.C
        s = wts.sort_values(ascending=True)
        fig = go.Figure(go.Bar(x=s.values * 100, y=[sym.replace("-USD", "") for sym in s.index],
                                orientation="h", marker=dict(color=C["orange"], opacity=.80)))
        fig.update_layout(**ChartFactory.layout("HRP Portfolio Weights (%)", h=max(240, len(s) * 26), legend=False))
        return fig

    @staticmethod
    def frontier(front: pd.DataFrame) -> go.Figure:
        C = ChartFactory.C
        if front.empty:
            return go.Figure()
        fig = go.Figure(go.Scatter(x=front["vol"], y=front["ret"], mode="markers+lines",
            marker=dict(color=front["sharpe"], colorscale="Plasma", size=6, showscale=True),
            line=dict(color=C["orange"], width=1), name="Efficient Frontier"))
        best = front.loc[front["sharpe"].idxmax()]
        fig.add_trace(go.Scatter(x=[best["vol"]], y=[best["ret"]], mode="markers",
                                  marker=dict(color=C["orange"], size=14, symbol="star"), name="Max Sharpe"))
        fig.update_layout(**ChartFactory.layout("Mean-Variance Frontier", h=300))
        return fig

    @staticmethod
    def cusum(prices: pd.Series) -> go.Figure:
        C = ChartFactory.C
        sp_arr, sn_arr = ResearchTools.cusum_series(prices)
        idx = prices.index
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=sp_arr[1:], fill="tozeroy", fillcolor="rgba(0,212,138,.12)",
                                  line=dict(color=C["green"], width=1), name="S+ CUSUM"))
        fig.add_trace(go.Scatter(x=idx, y=sn_arr[1:], fill="tozeroy", fillcolor="rgba(255,61,90,.12)",
                                  line=dict(color=C["red"], width=1), name="S− CUSUM"))
        fig.add_hline(y=Config.CUSUM_THR, line_color=C["yellow"], line_dash="dash", line_width=1)
        fig.add_hline(y=-Config.CUSUM_THR, line_color=C["yellow"], line_dash="dash", line_width=1)
        fig.update_layout(**ChartFactory.layout("CUSUM Structural Break Detection", h=260))
        return fig

    @staticmethod
    def entropy(price: pd.Series) -> go.Figure:
        C = ChartFactory.C
        ent_s = ResearchTools.entropy_series(price)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.5, .5], vertical_spacing=.03)
        fig.add_trace(go.Scatter(x=price.index, y=price.values, line=dict(color="#3a4a60", width=1), name="Price"), row=1, col=1)
        if not ent_s.empty:
            fig.add_trace(go.Scatter(x=ent_s.index, y=ent_s.values, fill="tozeroy",
                                      fillcolor="rgba(155,109,255,.12)", line=dict(color=C["purple"], width=1.5),
                                      name="Shannon Entropy"), row=2, col=1)
        ly = ChartFactory.layout("Market Entropy — Inefficiency Detector", h=320)
        ly["xaxis2"] = ChartFactory._xa()
        ly["showlegend"] = True
        fig.update_layout(**ly)
        return fig

    @staticmethod
    def frac_diff(price: pd.Series) -> go.Figure:
        C = ChartFactory.C
        fd_s = ResearchTools.frac_diff_series(price)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.5, .5], vertical_spacing=.03)
        fig.add_trace(go.Scatter(x=price.index, y=price.values, line=dict(color=C["blue"], width=1.2), name="Price (d=0)"), row=1, col=1)
        if not fd_s.empty:
            fig.add_trace(go.Scatter(x=fd_s.index, y=fd_s.values, line=dict(color=C["orange"], width=1.2), name="FracDiff d=0.35"), row=2, col=1)
        ly = ChartFactory.layout("Fractional Differentiation", h=300)
        ly["xaxis2"] = ChartFactory._xa()
        ly["showlegend"] = True
        fig.update_layout(**ly)
        return fig

    @staticmethod
    def news(items: list) -> go.Figure:
        C = ChartFactory.C
        if not items:
            return go.Figure()
        labels = [(n["title"][:42] + "..." if len(n["title"]) > 42 else n["title"]) for n in items[:10]]
        vals = [1 if n.get("sentiment") == "bullish" else -1 if n.get("sentiment") == "bearish" else 0 for n in items[:10]]
        cols = [C["green"] if v > 0 else C["red"] if v < 0 else C["muted"] for v in vals]
        fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=cols, opacity=.80))
        fig.update_layout(**ChartFactory.layout("News Sentiment", h=300, legend=False))
        return fig

    @staticmethod
    def model_radar(mr: dict) -> go.Figure:
        C = ChartFactory.C
        metrics = ["accuracy", "f1", "precision", "recall", "auc"]
        palette = [C["orange"], C["green"], C["blue"], C["purple"], C["cyan"]]
        fig = go.Figure()
        for i, (name, m) in enumerate(mr.items()):
            vals = [m.get(k, 0) for k in metrics] + [m.get(metrics[0], 0)]
            fig.add_trace(go.Scatterpolar(r=vals, theta=metrics + [metrics[0]], fill="toself", name=name,
                                           line=dict(color=palette[i % len(palette)], width=1.5), opacity=0.75))
        ly = ChartFactory.layout("Model Comparison", h=320)
        ly["polar"] = dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="rgba(255,255,255,.05)", color=C["muted"], tickfont=dict(size=7)),
                            angularaxis=dict(gridcolor="rgba(255,255,255,.05)", color=C["muted"], tickfont=dict(size=9)))
        ly["showlegend"] = True
        fig.update_layout(**ly)
        return fig

    @staticmethod
    def feat_importance(mr: dict) -> go.Figure:
        C = ChartFactory.C
        if not mr:
            return go.Figure()
        best = "Ensemble" if "Ensemble" in mr else max(mr, key=lambda k: mr[k]["cv_f1"])
        imp = mr[best]["importance"]
        if len(imp) != len(FEATURE_COLS):
            return go.Figure()
        s = pd.Series(np.abs(imp), index=FEATURE_COLS).sort_values()
        cols = [C["orange"] if v >= s.median() else C["blue"] for v in s.values]
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker=dict(color=cols, opacity=.85)))
        fig.update_layout(**ChartFactory.layout(f"Feature Importance ({best})", h=340, legend=False))
        return fig

    @staticmethod
    def confusion(mr: dict) -> go.Figure:
        C = ChartFactory.C
        if not mr:
            return go.Figure()
        best = max(mr, key=lambda k: mr[k]["cv_f1"])
        cm = np.array(mr[best]["confusion"])
        if cm.shape != (2, 2):
            return go.Figure()
        lbl = [["TN", "FP"], ["FN", "TP"]]
        text = [[f"{lbl[i][j]}\n{cm[i, j]}" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=cm, text=text, texttemplate="%{text}", textfont=dict(size=12, color=C["text"]),
            colorscale=[[0, C["panel"]], [1, "rgba(255,140,0,.7)"]], showscale=False,
            x=["Pred Long", "Pred Short"], y=["Actual Long", "Actual Short"]))
        fig.update_layout(**ChartFactory.layout(f"Confusion Matrix ({best})", h=280, legend=False))
        return fig

    @staticmethod
    def roc(mr: dict) -> go.Figure:
        C = ChartFactory.C
        palette = [C["orange"], C["green"], C["blue"], C["purple"], C["cyan"]]
        fig = go.Figure()
        for i, (name, m) in enumerate(mr.items()):
            if not m.get("y_prob"):
                continue
            fpr, tpr, _ = roc_curve(m["y_te"], m["y_prob"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={m['auc']:.2f})",
                                      line=dict(color=palette[i % len(palette)], width=1.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=C["muted"], dash="dash", width=1), name="Random"))
        fig.update_layout(**ChartFactory.layout("ROC Curves", h=300))
        return fig

    @staticmethod
    def triple_barrier(df: pd.DataFrame, labels_df: pd.DataFrame) -> go.Figure:
        C = ChartFactory.C
        c = df["Close"].squeeze()
        fig = go.Figure(go.Scatter(x=c.index, y=c.values, line=dict(color="#3a4a60", width=1), name="Price"))
        if not labels_df.empty:
            for sig_val, color, sym in [(1, C["green"], "triangle-up"), (-1, C["red"], "triangle-down")]:
                sub = labels_df[labels_df["label"] == sig_val]
                valid = [t for t in sub["time"] if t in c.index]
                if valid:
                    fig.add_trace(go.Scatter(x=valid, y=c.loc[valid].values, mode="markers",
                        marker=dict(symbol=sym, size=9, color=color), name=("Long" if sig_val == 1 else "Short")))
        fig.update_layout(**ChartFactory.layout("Triple Barrier Labels", h=320))
        return fig

    @staticmethod
    def vol_profile(df: pd.DataFrame) -> go.Figure:
        C = ChartFactory.C
        vp = ResearchTools.volume_profile(df)
        if vp.empty:
            return go.Figure()
        curr = float(df["Close"].iloc[-1])
        fig = make_subplots(rows=1, cols=2, column_widths=[.75, .25], shared_yaxes=True)
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(), line=dict(color=C["muted"], width=1), name="Price"), row=1, col=1)
        fig.add_trace(go.Bar(y=vp["price"], x=vp["volume"], orientation="h",
            marker_color=[C["orange"] if r["is_hvn"] else C["blue"] for _, r in vp.iterrows()],
            opacity=.75, name="Vol Profile"), row=1, col=2)
        fig.add_hline(y=curr, line_color=C["cyan"], line_width=1.5, row="all")
        ly = ChartFactory.layout("Volume Profile — HVN Support/Resistance", h=360)
        ly["showlegend"] = True
        fig.update_layout(**ly)
        return fig


# ───────────────────────────── UI COMPONENTS ─────────────────────────────

class UI:
    C = PALETTE

    @staticmethod
    def inject_css() -> None:
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');
        html,body,[class*="css"]{font-family:'IBM Plex Mono',monospace;background:#0a0b0d!important;color:#e8eaf0!important;}
        .main,.block-container{background:#0a0b0d!important;}
        section[data-testid="stSidebar"]{background:#060810!important;border-right:1px solid #1e2130;}
        .bb-hdr{background:linear-gradient(90deg,#000 0%,#0d0f16 100%);border-bottom:2px solid #ff8c00;
                padding:10px 20px;margin:-1rem -1rem 1rem -1rem;display:flex;align-items:center;justify-content:space-between;}
        .bb-logo{font-size:20px;font-weight:700;color:#ff8c00;letter-spacing:4px;}
        .bb-sub{font-size:8px;color:#4a5270;letter-spacing:2px;text-transform:uppercase;}
        .bb-live{font-size:10px;color:#00d48a;}
        .kpi-card{background:#12141a;border:1px solid #1e2130;border-top:2px solid #ff8c00;
                  border-radius:2px;padding:10px 12px;margin-bottom:5px;}
        .kpi-lbl{font-size:8px;color:#4a5270;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;}
        .kpi-val{font-size:15px;font-weight:700;color:#e8eaf0;}
        .kpi-pos{font-size:9px;color:#00d48a;font-weight:600;}
        .kpi-neg{font-size:9px;color:#ff3d5a;font-weight:600;}
        .sec-hdr{font-size:8px;color:#ff8c00;letter-spacing:2.5px;text-transform:uppercase;
                 border-bottom:1px solid #1e2130;padding-bottom:4px;margin:14px 0 8px 0;}
        .stat-row{display:flex;justify-content:space-between;padding:3px 0;
                  border-bottom:1px solid #0e1420;font-size:9px;}
        .stat-k{color:#4a5270;text-transform:uppercase;letter-spacing:.4px;}
        .stat-v{color:#c0cce0;}
        .signal-lg{background:rgba(0,212,138,.12);border:2px solid #00d48a;color:#00d48a;
                   text-align:center;padding:12px;font-size:22px;font-weight:700;border-radius:3px;}
        .signal-sh{background:rgba(255,61,90,.12);border:2px solid #ff3d5a;color:#ff3d5a;
                   text-align:center;padding:12px;font-size:22px;font-weight:700;border-radius:3px;}
        .signal-fl{background:rgba(74,82,112,.2);border:2px solid #2a2f42;color:#4a5270;
                   text-align:center;padding:12px;font-size:22px;font-weight:700;border-radius:3px;}
        .tag-long{background:rgba(0,212,138,.15);border:1px solid #00d48a;color:#00d48a;font-size:8px;padding:1px 5px;border-radius:2px;}
        .tag-short{background:rgba(255,61,90,.15);border:1px solid #ff3d5a;color:#ff3d5a;font-size:8px;padding:1px 5px;border-radius:2px;}
        .tag-neu{background:rgba(74,82,112,.3);border:1px solid #2a2f42;color:#4a5270;font-size:8px;padding:1px 5px;border-radius:2px;}
        .stButton>button{background:#ff8c00;color:#000;border:none;border-radius:2px;
          font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:.8px;padding:5px 14px;width:100%;font-weight:700;}
        .stButton>button:hover{background:#ffa333;}
        div[data-testid="stMetric"]{background:#12141a;border:1px solid #1e2130;border-radius:2px;padding:8px;}
        .stTabs [data-baseweb="tab"]{background:#12141a;color:#4a5270;border-radius:0;border-bottom:2px solid transparent;
          font-family:'IBM Plex Mono',monospace;font-size:8px;letter-spacing:1.2px;text-transform:uppercase;}
        .stTabs [aria-selected="true"]{background:#12141a;color:#ff8c00;border-bottom:2px solid #ff8c00;}
        .alert-green{background:rgba(0,212,138,.1);border:1px solid #00d48a;color:#00d48a;padding:6px 10px;font-size:9px;border-radius:2px;margin:4px 0;}
        .alert-red{background:rgba(255,61,90,.1);border:1px solid #ff3d5a;color:#ff3d5a;padding:6px 10px;font-size:9px;border-radius:2px;margin:4px 0;}
        .alert-yellow{background:rgba(255,215,0,.1);border:1px solid #ffd700;color:#ffd700;padding:6px 10px;font-size:9px;border-radius:2px;margin:4px 0;}
        div.block-container{padding-top:.3rem;}
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def stat_block(rows: list) -> None:
        html = ""
        for item in rows:
            k, v = item[0], item[1]
            col = item[2] if len(item) > 2 else UI.C["text"]
            html += (f'<div class="stat-row"><span class="stat-k">{k}</span>'
                     f'<span class="stat-v" style="color:{col}">{v}</span></div>')
        st.markdown(html, unsafe_allow_html=True)

    @staticmethod
    def kpi_banner(df: pd.DataFrame, info: dict, gm: dict, fg: dict) -> None:
        price = info.get("price") or (float(df["Close"].iloc[-1]) if not df.empty else 0)
        chg24 = info.get("change_24h", 0)
        sym = info.get("symbol", "BTC")
        rsi_n = float(df["rsi"].iloc[-1]) if "rsi" in df.columns and not df.empty else 50

        chg_html = (f'<div class="kpi-pos">▲ {chg24:+.2f}%</div>' if chg24 >= 0
                    else f'<div class="kpi-neg">▼ {chg24:+.2f}%</div>')
        rsi_badge = ('<span class="tag-short">OVERBOUGHT</span>' if rsi_n > 70
                     else '<span class="tag-long">OVERSOLD</span>' if rsi_n < 30
                     else '<span class="tag-neu">NEUTRAL</span>')

        def _fmt_price(p):
            if p < 0.01:
                return f"${p:.6f}"
            if p < 1:
                return f"${p:.4f}"
            return f"${p:,.2f}"

        def _fmt_large(v):
            if v >= 1e12:
                return f"${v/1e12:.2f}T"
            if v >= 1e9:
                return f"${v/1e9:.2f}B"
            if v >= 1e6:
                return f"${v/1e6:.0f}M"
            return f"${v:,.0f}"

        cols = st.columns(7)
        tiles = [
            (f"{sym}/USD", _fmt_price(price) if price else "—", chg_html),
            ("24H High", _fmt_price(info.get("high_24h", 0)) if info.get("high_24h") else "—", ""),
            ("24H Low", _fmt_price(info.get("low_24h", 0)) if info.get("low_24h") else "—", ""),
            ("Market Cap", _fmt_large(info.get("market_cap", 0)) if info.get("market_cap") else "—", ""),
            ("Volume 24h", _fmt_large(info.get("volume_24h", 0)) if info.get("volume_24h") else "—", ""),
            ("BTC Dom", f"{gm.get('btc_dom', 0):.1f}%" if gm.get("btc_dom") else "—", ""),
            ("RSI (14)", f"{rsi_n:.1f}", rsi_badge),
        ]
        for col, (label, value, extra) in zip(cols, tiles):
            with col:
                st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">{label}</div>'
                            f'<div class="kpi-val">{value}</div>{extra}</div>', unsafe_allow_html=True)

    @staticmethod
    def check_alerts(df: pd.DataFrame, info: dict) -> list:
        alerts = []
        if df.empty:
            return alerts
        rsi = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50
        if rsi > 75:
            alerts.append(("red", f"RSI {rsi:.1f} — EXTREME OVERBOUGHT"))
        elif rsi > 70:
            alerts.append(("yellow", f"RSI {rsi:.1f} — Overbought zone"))
        elif rsi < 25:
            alerts.append(("red", f"RSI {rsi:.1f} — EXTREME OVERSOLD"))
        elif rsi < 30:
            alerts.append(("green", f"RSI {rsi:.1f} — Oversold opportunity"))
        if "ema9" in df.columns and "ema21" in df.columns and len(df) > 2:
            prev_x = df["ema9"].iloc[-2] - df["ema21"].iloc[-2]
            curr_x = df["ema9"].iloc[-1] - df["ema21"].iloc[-1]
            if prev_x < 0 < curr_x:
                alerts.append(("green", "EMA 9/21 Golden Cross detected"))
            elif prev_x > 0 > curr_x:
                alerts.append(("red", "EMA 9/21 Death Cross detected"))
        chg = info.get("change_24h", 0)
        if abs(chg) > 10:
            alerts.append(("red" if chg < 0 else "yellow", f"Large 24H move: {chg:+.1f}%"))
        return alerts

    @staticmethod
    def render_alerts(alerts: list) -> None:
        cls_map = {"red": "alert-red", "green": "alert-green", "yellow": "alert-yellow"}
        for level, msg in alerts:
            st.markdown(f'<div class="{cls_map.get(level, "alert-yellow")}">⚠  {msg}</div>', unsafe_allow_html=True)

    @staticmethod
    def dl_csv(df: pd.DataFrame, filename: str, label: str) -> None:
        st.download_button(label=label, data=df.to_csv().encode(), file_name=filename, mime="text/csv")


# ───────────────────────────── SIDEBAR ─────────────────────────────

class Sidebar:

    @staticmethod
    def render() -> dict:
        st.sidebar.markdown("""
        <div style="padding:8px 0 6px;border-bottom:1px solid #1e2130;margin-bottom:10px;">
          <div style="font-size:16px;font-weight:700;color:#ff8c00;letter-spacing:3px;">⬡ HECTOR</div>
          <div style="font-size:7px;color:#4a5270;letter-spacing:2px;text-transform:uppercase;">
            OOP Edition · Lean Intelligence Stack
          </div>
        </div>""", unsafe_allow_html=True)

        st.sidebar.markdown('<div class="sec-hdr">Asset</div>', unsafe_allow_html=True)
        coin_label = st.sidebar.selectbox("Asset", list(COINS.keys()), index=0, label_visibility="collapsed")
        cg_id, yf_sym = COINS[coin_label]

        st.sidebar.markdown('<div class="sec-hdr">Portfolio Assets</div>', unsafe_allow_html=True)
        comp_keys = st.sidebar.multiselect("Compare", [k for k in COINS if k != coin_label],
                                            default=list(COINS.keys())[1:4], label_visibility="collapsed")
        comp_symbols = [COINS[k][1] for k in comp_keys]

        st.sidebar.markdown('<div class="sec-hdr">Time Frame</div>', unsafe_allow_html=True)
        c1, c2 = st.sidebar.columns(2)
        with c1:
            period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1,
                                   format_func={"1mo": "1M", "3mo": "3M", "6mo": "6M", "1y": "1Y"}.get,
                                   label_visibility="collapsed")
        with c2:
            interval = st.selectbox("Interval", ["1h", "1d"], format_func={"1h": "1H", "1d": "1D"}.get,
                                     label_visibility="collapsed")

        st.sidebar.markdown('<div class="sec-hdr">Chart Overlays</div>', unsafe_allow_html=True)
        show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
        show_ema = st.sidebar.checkbox("EMA 9 / 21", value=True)
        show_vwap = st.sidebar.checkbox("VWAP", value=True)

        st.sidebar.markdown('<div class="sec-hdr">ML Parameters</div>', unsafe_allow_html=True)
        pt_mult = st.sidebar.slider("Profit Barrier ×σ", 0.5, 3.0, 1.5, 0.1)
        sl_mult = st.sidebar.slider("Stop Barrier ×σ", 0.5, 3.0, 1.0, 0.1)
        max_hold = st.sidebar.slider("Max Hold (bars)", 5, 48, 20, 1)
        conf_thr = st.sidebar.slider("Confidence Thr", 0.50, 0.80, 0.55, 0.01)

        st.sidebar.markdown('<div class="sec-hdr">Costs</div>', unsafe_allow_html=True)
        commission = st.sidebar.slider("Commission %", 0.0, 0.5, 0.1, 0.01) / 100

        st.sidebar.markdown('<div class="sec-hdr">Controls</div>', unsafe_allow_html=True)
        run_ml = st.sidebar.checkbox("Run ML Pipeline", value=True)
        if st.sidebar.button("🔄  REFRESH DATA"):
            st.cache_data.clear()
            st.rerun()

        st.sidebar.markdown("""
        ---
        <div style="text-align:center;padding-top:6px;">
          <span style="font-size:8px;color:#2a3040;letter-spacing:1px;">
            Created by <strong style="color:#ff8c00;">Daniyal Aziz</strong>
          </span>
        </div>""", unsafe_allow_html=True)

        return dict(coin_label=coin_label, cg_id=cg_id, yf_sym=yf_sym, comp_symbols=comp_symbols,
                     period=period, interval=interval, show_bb=show_bb, show_ema=show_ema,
                     show_vwap=show_vwap, pt_mult=pt_mult, sl_mult=sl_mult, max_hold=max_hold,
                     conf_thr=conf_thr, commission=commission, run_ml=run_ml)


# ───────────────────────────── APPLICATION ─────────────────────────────

class HectorApp:

    def __init__(self):
        UI.inject_css()

    def _header(self) -> None:
        now_str = datetime.utcnow().strftime("%b %d, %Y · %I:%M %p UTC")
        st.markdown(
            f'<div class="bb-hdr"><div><div class="bb-logo">⬡ HECTOR</div>'
            f'<div class="bb-sub">OOP Edition · Multi-Layer Intelligence · {len(COINS)} Assets · Free Data</div></div>'
            f'<div class="bb-live">🟢 LIVE &nbsp;|&nbsp; {now_str}</div></div>', unsafe_allow_html=True)

    def run(self) -> None:
        self._header()
        cfg = Sidebar.render()

        with st.spinner("Loading market data…"):
            df_raw = DataFeed.ohlcv(cfg["yf_sym"], cfg["period"], cfg["interval"])
            info = DataFeed.coin_info(cfg["cg_id"])
            gm = DataFeed.global_market()
            fg = DataFeed.fear_greed()

        if df_raw.empty:
            st.error(f"No data returned for **{cfg['yf_sym']}**. Try a different asset or refresh.")
            return

        df_raw, val_report = FeatureEngineer.validate(df_raw)
        df = FeatureEngineer.transform(df_raw)
        regime = FeatureEngineer.regime(df["returns"]) if "returns" in df.columns else "sideways"

        labels_df, feat_df, mr, signals_df, bt = pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame(), {}
        if cfg["run_ml"]:
            with st.spinner("Running ML pipeline…"):
                labels_df = Labeler.triple_barrier(df, cfg["pt_mult"], cfg["sl_mult"], cfg["max_hold"])
                feat_df = FeatureBuilder.build(df, labels_df) if not labels_df.empty else pd.DataFrame()
                mr = ModelEngine.train(feat_df, cfg["yf_sym"]) if len(feat_df) >= 40 else {}
                signals_df = SignalEngine.generate(df, labels_df, feat_df, mr, cfg["conf_thr"], regime) if mr else pd.DataFrame()
                bt = Backtester.run(df["Close"].squeeze(), signals_df, cfg["commission"]) if not signals_df.empty else {}

        tabs = st.tabs(["Market Overview", "Signals", "Models", "Backtest", "Risk",
                        "Portfolio", "Research", "Sentiment", "Export"])

        with tabs[0]:
            self._tab_overview(df, df_raw, info, gm, fg, val_report, regime, signals_df, cfg)
        with tabs[1]:
            self._tab_signals(df, df_raw, labels_df, signals_df, regime, cfg)
        with tabs[2]:
            self._tab_models(mr, feat_df, cfg)
        with tabs[3]:
            self._tab_backtest(df, signals_df, bt, cfg)
        with tabs[4]:
            self._tab_risk(df)
        with tabs[5]:
            self._tab_portfolio(cfg)
        with tabs[6]:
            self._tab_research(df)
        with tabs[7]:
            self._tab_sentiment()
        with tabs[8]:
            self._tab_export(df, labels_df, signals_df, bt, cfg)

        st.markdown(
            f'<div style="text-align:right;padding:6px 4px 2px;border-top:1px solid {PALETTE["border"]};margin-top:16px;">'
            f'<span style="font-size:7px;color:#2a3040;letter-spacing:1px;">'
            f'Created by <strong style="color:{PALETTE["orange"]}">Daniyal Aziz</strong></span></div>',
            unsafe_allow_html=True)

    def _tab_overview(self, df, df_raw, info, gm, fg, val_report, regime, signals_df, cfg):
        UI.kpi_banner(df, info, gm, fg)
        alerts = UI.check_alerts(df, info)
        if alerts:
            UI.render_alerts(alerts)
        st.markdown('<div class="sec-hdr">Live Price Action</div>', unsafe_allow_html=True)
        st.plotly_chart(ChartFactory.candle(df, f"{info.get('symbol','BTC')}/USD · {cfg['period'].upper()} {cfg['interval'].upper()}",
                                             cfg["show_bb"], cfg["show_ema"], cfg["show_vwap"],
                                             signals_df if not signals_df.empty else None),
                         use_container_width=True, config=PCONF)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(ChartFactory.macd(df), use_container_width=True, config=PCONF)
        with c2:
            st.plotly_chart(ChartFactory.fear_gauge(fg), use_container_width=True, config=PCONF)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(ChartFactory.cusum(df["Close"].squeeze()), use_container_width=True, config=PCONF)
        with c4:
            UI.stat_block([
                ("Rows fetched", str(val_report.get("rows_before", len(df_raw))), PALETTE["green"]),
                ("After cleaning", str(len(df)), PALETTE["green"]),
                ("Outliers capped", str(val_report.get("outliers", 0)), PALETTE["yellow"]),
                ("Duplicates rm'd", str(val_report.get("duplicates", 0)), PALETTE["yellow"]),
                ("Date range", f"{df.index[0].date()} → {df.index[-1].date()}" if not df.empty else "—", PALETTE["text"]),
                ("Regime", regime.upper(), PALETTE["green"] if regime == "bull" else PALETTE["red"] if regime == "bear" else PALETTE["yellow"]),
            ])

    def _tab_signals(self, df, df_raw, labels_df, signals_df, regime, cfg):
        st.markdown('<div class="sec-hdr">Signal Generator</div>', unsafe_allow_html=True)
        if not cfg["run_ml"]:
            st.info("Enable 'Run ML Pipeline' in the sidebar.")
            return
        if labels_df.empty:
            st.warning("Not enough data for labeling. Try 6 months or more.")
            return
        longs, shorts = int((labels_df["label"] == 1).sum()), int((labels_df["label"] == -1).sum())
        neuts, total = int((labels_df["label"] == 0).sum()), max(len(labels_df), 1)
        last_l = int(labels_df["label"].iloc[-1])
        sig_cls = {1: "signal-lg", -1: "signal-sh", 0: "signal-fl"}.get(last_l, "signal-fl")
        sig_text = {1: "▲ LONG", -1: "▼ SHORT", 0: "— FLAT"}.get(last_l, "— FLAT")

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.markdown(f'<div class="{sig_cls}">{sig_text}</div>', unsafe_allow_html=True)
            if not signals_df.empty:
                ls = signals_df.iloc[-1]
                st.markdown(f'<div style="text-align:center;font-size:11px;color:{PALETTE["sec"]};margin-top:6px;">'
                            f'Prob: {float(ls["prob"])*100:.1f}% · Kelly: {float(ls["bet_size"]):.3f}</div>', unsafe_allow_html=True)
        with c2:
            UI.stat_block([
                ("Total Labels", str(total), PALETTE["text"]),
                ("Long (1)", f"{longs} ({longs/total*100:.1f}%)", PALETTE["green"]),
                ("Short (−1)", f"{shorts} ({shorts/total*100:.1f}%)", PALETTE["red"]),
                ("Neutral (0)", f"{neuts} ({neuts/total*100:.1f}%)", PALETTE["muted"]),
                ("Regime", regime.upper(), PALETTE["green"] if regime == "bull" else PALETTE["red"] if regime == "bear" else PALETTE["yellow"]),
            ])
        with c3:
            st.plotly_chart(ChartFactory.triple_barrier(df, labels_df), use_container_width=True, config=PCONF)

        st.markdown('<div class="sec-hdr">Recent Signals</div>', unsafe_allow_html=True)
        disp = labels_df.tail(15).copy()
        disp["Direction"] = disp["label"].map({1: "▲ LONG", -1: "▼ SHORT", 0: "— FLAT"})
        disp["Return %"] = (disp["ret"] * 100).round(3)
        st.dataframe(disp[["time", "Direction", "Return %"]], use_container_width=True, hide_index=True)
        st.plotly_chart(ChartFactory.vol_profile(df_raw), use_container_width=True, config=PCONF)

    def _tab_models(self, mr, feat_df, cfg):
        st.markdown('<div class="sec-hdr">Multi-Model Ensemble</div>', unsafe_allow_html=True)
        if not cfg["run_ml"]:
            st.info("Enable 'Run ML Pipeline' in the sidebar.")
            return
        if not mr:
            st.warning("Insufficient data. Try a longer period (6mo+ recommended).")
            return
        m_cols = st.columns(min(len(mr), 4))
        for col, (name, metrics) in zip(m_cols, mr.items()):
            with col:
                UI.stat_block([
                    (name, "", PALETTE["orange"]),
                    ("CV F1", f"{metrics['cv_f1']:.3f}±{metrics['cv_std']:.3f}", PALETTE["green"]),
                    ("Accuracy", f"{metrics['accuracy']*100:.1f}%", PALETTE["text"]),
                    ("AUC-ROC", f"{metrics['auc']:.3f}", PALETTE["yellow"]),
                ])
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ChartFactory.model_radar(mr), use_container_width=True, config=PCONF)
        with c2:
            st.plotly_chart(ChartFactory.confusion(mr), use_container_width=True, config=PCONF)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(ChartFactory.feat_importance(mr), use_container_width=True, config=PCONF)
        with c4:
            st.plotly_chart(ChartFactory.roc(mr), use_container_width=True, config=PCONF)
        st.markdown('<div class="sec-hdr">Feature Matrix</div>', unsafe_allow_html=True)
        cols_show = [c for c in FEATURE_COLS + ["label"] if c in feat_df.columns]
        if cols_show:
            st.dataframe(feat_df[cols_show].head(12).round(4), use_container_width=True, hide_index=True)

    def _tab_backtest(self, df, signals_df, bt, cfg):
        st.markdown('<div class="sec-hdr">Backtest & Validation</div>', unsafe_allow_html=True)
        if not cfg["run_ml"]:
            st.info("Enable 'Run ML Pipeline' in the sidebar.")
            return
        if not bt:
            st.info("Not enough trading signals. Try a longer period or lower the confidence threshold.")
            return
        bc = st.columns(6)
        kv = [
            ("Total Return", f"{(bt['equity'].iloc[-1]-1)*100:+.1f}%", PALETTE["green"] if bt["equity"].iloc[-1] > 1 else PALETTE["red"]),
            ("Ann Return", f"{bt['ann_ret']*100:+.1f}%", PALETTE["green"] if bt["ann_ret"] > 0 else PALETTE["red"]),
            ("Sharpe", f"{bt['sharpe']:.3f}", PALETTE["orange"]),
            ("Max Drawdown", f"{bt['max_dd']*100:.1f}%", PALETTE["red"]),
            ("Win Rate", f"{bt['win_rate']*100:.1f}%", PALETTE["green"]),
            ("PSR", f"{bt['psr']:.3f}", PALETTE["green"] if bt["psr"] > 0.9 else PALETTE["yellow"]),
        ]
        for col, (lbl, val, color) in zip(bc, kv):
            with col:
                st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">{lbl}</div>'
                            f'<div class="kpi-val" style="font-size:14px;color:{color}">{val}</div></div>', unsafe_allow_html=True)
        st.plotly_chart(ChartFactory.equity(bt), use_container_width=True, config=PCONF)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ChartFactory.drawdown(bt), use_container_width=True, config=PCONF)
        with c2:
            UI.stat_block([
                ("Sharpe 95% CI", f"[{bt['sharpe_ci'][0]:.3f}, {bt['sharpe_ci'][1]:.3f}]", PALETTE["cyan"]),
                ("Sortino", f"{bt['sortino']:.3f}", PALETTE["text"]),
                ("Calmar Ratio", f"{bt['calmar']:.3f}", PALETTE["text"]),
                ("Profit Factor", f"{bt['profit_factor']:.2f}", PALETTE["text"]),
                ("# Trades", str(bt["n_trades"]), PALETTE["text"]),
                ("DSR", f"{bt['dsr']:.3f}", PALETTE["green"] if bt["dsr"] > 0.9 else PALETTE["yellow"]),
            ])
        st.markdown('<div class="sec-hdr">Walk-Forward Validation</div>', unsafe_allow_html=True)
        wf = Backtester.walk_forward(df, signals_df, n_folds=5)
        if wf:
            wf_df = pd.DataFrame([{"Fold": r["fold"], "Sharpe": f"{r['sharpe']:.3f}",
                                    "Ann Ret": f"{r['ann_ret']*100:.2f}%", "Max DD": f"{r['max_dd']*100:.2f}%"} for r in wf])
            st.dataframe(wf_df, use_container_width=True, hide_index=True)

    def _tab_risk(self, df):
        st.markdown('<div class="sec-hdr">Risk Analysis</div>', unsafe_allow_html=True)
        risk = RiskEngine.compute(df["returns"]) if "returns" in df.columns else {}
        if not risk:
            st.warning("Insufficient data for risk analysis.")
            return
        kr = st.columns(4)
        for col_, (lbl_, val_, clr_) in zip(kr, [
            ("VaR 95%", f"{risk['var95']:.3f}%", PALETTE["red"]),
            ("CVaR 95%", f"{risk['cvar95']:.3f}%", PALETTE["red"]),
            ("Ann Vol", f"{risk['ann_vol']:.2f}%", PALETTE["yellow"]),
            ("Max DD", f"{risk['max_dd']:.2f}%", PALETTE["red"]),
        ]):
            with col_:
                st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">{lbl_}</div>'
                            f'<div class="kpi-val" style="color:{clr_}">{val_}</div></div>', unsafe_allow_html=True)
        st.plotly_chart(ChartFactory.var_dist(risk), use_container_width=True, config=PCONF)
        st.markdown('<div class="sec-hdr">Stress Scenarios</div>', unsafe_allow_html=True)
        UI.stat_block([(k, f"{v:.2f}%", PALETTE["red"]) for k, v in risk["stress"].items()])

    def _tab_portfolio(self, cfg):
        st.markdown('<div class="sec-hdr">Portfolio Optimisation</div>', unsafe_allow_html=True)
        syms_port = [cfg["yf_sym"]] + cfg["comp_symbols"]
        if len(syms_port) < 2:
            st.info("Select at least 2 comparison assets in the sidebar.")
            return
        with st.spinner("Fetching multi-asset data & optimising…"):
            multi = DataFeed.multi_ohlcv(tuple(syms_port), cfg["period"])
        if len(multi) < 2:
            st.warning("Could not fetch enough asset data.")
            return
        ret_df = pd.DataFrame({s: d["Close"].squeeze().pct_change() for s, d in multi.items()}).dropna()
        wts = PortfolioEngine.hrp_weights(ret_df)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ChartFactory.hrp_bar(wts), use_container_width=True, config=PCONF)
        with c2:
            st.plotly_chart(ChartFactory.corr_matrix(ret_df), use_container_width=True, config=PCONF)
        with st.spinner("Computing efficient frontier…"):
            front = PortfolioEngine.efficient_frontier(ret_df, n=15)
        st.plotly_chart(ChartFactory.frontier(front), use_container_width=True, config=PCONF)
        wt_tab = pd.DataFrame({
            "Asset": [s.replace("-USD", "") for s in wts.index],
            "Weight": [f"{w * 100:.2f}%" for w in wts.values],
            "Period Ret": [f"{ret_df[s].sum() * 100:.1f}%" for s in wts.index],
        })
        st.dataframe(wt_tab, use_container_width=True, hide_index=True)

    def _tab_research(self, df):
        st.markdown('<div class="sec-hdr">Quantitative Research</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ChartFactory.entropy(df["Close"].squeeze()), use_container_width=True, config=PCONF)
        with c2:
            st.plotly_chart(ChartFactory.frac_diff(df["Close"].squeeze()), use_container_width=True, config=PCONF)
        st.plotly_chart(ChartFactory.cusum(df["Close"].squeeze()), use_container_width=True, config=PCONF)

    def _tab_sentiment(self):
        st.markdown('<div class="sec-hdr">Market Sentiment</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sec-hdr">Reddit Sentiment</div>', unsafe_allow_html=True)
            with st.spinner("Fetching Reddit…"):
                red = DataFeed.reddit_sentiment("CryptoCurrency")
            score = red["score"]
            col_ = PALETTE["green"] if score > 0.05 else PALETTE["red"] if score < -0.05 else PALETTE["yellow"]
            st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">SENTIMENT SCORE</div>'
                        f'<div class="kpi-val" style="color:{col_}">{score:+.3f} — {red["label"].upper()}</div>'
                        f'<div style="font-size:8px;color:{PALETTE["sec"]};">{red["count"]} posts analysed</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="sec-hdr">Crypto News</div>', unsafe_allow_html=True)
            with st.spinner("Fetching news…"):
                news = DataFeed.news()
            if news:
                st.plotly_chart(ChartFactory.news(news), use_container_width=True, config=PCONF)
            else:
                st.info("News unavailable right now.")

    def _tab_export(self, df, labels_df, signals_df, bt, cfg):
        st.markdown('<div class="sec-hdr">Export</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.markdown("**OHLCV + Indicators**")
            UI.dl_csv(df.reset_index(), "hector_indicators.csv", "Download CSV")
        with e2:
            if not labels_df.empty:
                st.markdown("**Triple-Barrier Labels**")
                UI.dl_csv(labels_df, "hector_labels.csv", "Download Labels")
        with e3:
            if not signals_df.empty:
                st.markdown("**Signal History**")
                UI.dl_csv(signals_df, "hector_signals.csv", "Download Signals")
        with e4:
            st.markdown("**Run Summary (JSON)**")
            summary = {
                "asset": cfg["yf_sym"], "period": cfg["period"], "interval": cfg["interval"],
                "generated_at": datetime.utcnow().isoformat(),
                "sharpe": bt.get("sharpe"), "ann_ret": bt.get("ann_ret"), "n_trades": bt.get("n_trades"),
            }
            st.download_button("Download JSON", json.dumps(summary, indent=2),
                                "hector_summary.json", mime="application/json")


if __name__ == "__main__":
    HectorApp().run()
