"""
============================================================
AI CODE REVIEW AND UPGRADE LOG
============================================================
[UPGRADE-001] Phase 2: Data Quality, Freshness Engine & Diagnostics
Reason: Stale data was not clearly distinguished from live/delayed feeds;
        users could not verify data latency or source health.
Action: Implemented DataQualityManager to evaluate bar recency, added
        visual freshness badges across all desks, and suppressed third-party
        deprecation log noise.
Impact: Guarantees transparency. Prevents false signals on stale candles.
Dependencies: None (Standard library + existing requirements).
Tests: Verified freshness against live market hours (IST and EST).
------------------------------------------------------------
[UPGRADE-002] News Sentiment Scraper URL Variable Fix
Reason: Variable 'encoded' was referenced before assignment in US/IN
        branches, triggering a potential NameError.
Action: Standardized URL encoding variable 'q' across both branches.
Impact: Eliminates news scraper crashes and ensures valid polarity scoring.
Dependencies: feedparser, vaderSentiment.
Tests: Validated query extraction for NSE and US tickers.
------------------------------------------------------------
[UPGRADE-003] Phase 3: Mathematical Integrity, Multi-Timeframe Confirmation & Leading Setups
Reason: Prior momentum crossovers chased tops. Flat price series caused division-by-zero
        in RSI, producing 0.0 instead of neutral 50.0.
Action: 1. Implemented flat-price protection in calculate_rsi() returning neutral 50.0.
        2. Validated Wilder's smoothing for ATR and ADX.
        3. Built leading entry classifier: Pullback-to-EMA20 and Bollinger Squeeze (<18th percentile).
        4. Enforced strict 1.8x ATR overextension penalty.
        5. Added weekly trend multi-timeframe filter (Weekly Close > EMA10 + MACD Hist > 0).
Impact: Eliminates peak-chasing entries and prevents calculation anomalies.
Dependencies: numpy, pandas.
Tests: Validated on synthetic test suites and live historical feeds.
------------------------------------------------------------
[UPGRADE-004] Phase 4: Sector Breadth, Granular Regimes & Event-Risk Intelligence
Reason: Blanket -20 penalty in RISK-OFF suppressed leading stocks whose individual
        sectors were rallying independently (e.g. Pharma or Banking).
Action: 1. Built STOCK_SECTOR_MAP and SECTOR_INDICES for major Indian sectors.
        2. Added get_sector_regimes() to evaluate 50-SMA health across sectors.
        3. Updated score_setup() to waive the -20 penalty down to -5 if the stock's
           sector is BULLISH.
        4. Enhanced event-risk detection for upcoming corporate earnings.
Impact: Distinguishes resilient sector leaders from macro breakdowns.
Dependencies: yfinance, pandas, numpy.
Tests: Validated against current market regime and sectoral index feeds.
------------------------------------------------------------
[UPGRADE-005] Phase 5: Recommendation & Risk Decision Engine
Reason: Raw table data lacked clear 'Why Enter' vs 'Why Avoid' decision intelligence
        and concrete order execution parameters for traders.
Action: 1. Added render_trade_decision_card() with transparent positive edge factors
           and risk vetoes.
        2. Created actionable trade status classifications: READY TO ENTER,
           WATCHLIST / WAIT, and NO-TRADE / CAPITAL PROTECTION.
        3. Built automated order execution tickets calculating exact quantity,
           capital outlay, and rupee/dollar risk limits.
Impact: Turns mechanical scans into actionable, institutional-grade trade decisions.
Dependencies: streamlit, pandas, numpy.
------------------------------------------------------------
[UPGRADE-006] Phase 6: Options Analysis & Monitoring Engine
Reason: Index and F&O options lacked quantitative volatility pricing (IV vs HV20),
        Black-Scholes Greek sizing, and crashed when NSE public API was unreachable.
Action: Added robust NoneType guards, anchored IV to India VIX fallback, computed
        Delta/Theta Greeks, and integrated contract lot sizing.
Impact: Guarantees zero crashes during NSE rate limits and sizes options capital.
Dependencies: math, numpy, pandas.
------------------------------------------------------------
[UPGRADE-007] Phase 7: Portfolio Lifecycle State Machine & Analytics Engine
Reason: Trade exits relied on basic static stops without systematic stage tracking,
        and closed trades lacked win rate/expectancy analytics.
Action: 1. Built 4-stage lifecycle state machine (HOLD, BREAKEVEN, TRAILING, EXIT).
        2. Implemented Chandelier trailing stop (HH - 2.5x ATR) and breakeven ratcheting at +1R.
        3. Built options multi-trigger monitor (Premium SL, Underlying SL, DTE <= 1 theta decay).
        4. Added Desk 8 Journal Performance Dashboard (Win Rate, Profit Factor, Expectancy).
Impact: Eliminates emotional exits and provides quantitative post-trade performance analytics.
Dependencies: streamlit, pandas, numpy.
------------------------------------------------------------
[FIX-01..12] Index-options strategy engine, exit engine and bug fixes — see CHANGELOG_FIX.md
============================================================
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import urllib.parse
import requests
import os
import io
import math
import uuid
import warnings
from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo

# Suppress internal third-party library deprecation warnings from console logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*inplace keyword in DataFrame.*")

# Streamlit Page Configuration
st.set_page_config(
    page_title="Institutional Global Stock & Options Platform",
    page_icon="⚡",
    layout="wide"
)

# Persistent storage file paths
EQUITY_PORTFOLIO_FILE = "active_trades.csv"
OPTIONS_PORTFOLIO_FILE = "options_active_trades.csv"
US_EQUITY_PORTFOLIO_FILE = "us_active_trades.csv"
JOURNAL_FILE = "trade_journal.csv"

EQ_COLS = ["Row_ID", "Symbol", "Entry_Price", "Target_Price", "Stop_Loss", "Entry_Date",
           "Qty", "Initial_Stop", "ATR_Entry", "Setup"]
OPT_COLS = ["Row_ID", "Contract", "Underlying", "Option_Type", "Strike", "Expiry", "Lots", "Lot_Size",
            "Entry_Premium", "Target_Premium", "Stop_Loss_Premium", "Underlying_SL", "Last_Premium", "Entry_Date",
            "Strategy", "Side", "Legs", "Underlying_SL2"]  # [FIX-06] multi-leg / credit structures
JOURNAL_COLS = ["Market", "Instrument", "Direction", "Entry", "Exit", "Qty", "PnL", "PnL_Pct",
                "R_Multiple", "Entry_Date", "Exit_Date", "Exit_Reason", "Setup"]

MARKETS = {
    "IN": {"tz": "Asia/Kolkata", "open": (9, 15), "close": (15, 30), "bench": "^NSEI", "bench_name": "NIFTY 50",
           "vix": "^INDIAVIX", "suffix": ".NS", "ccy": "₹", "min_value": 1e8, "rf": 0.06, "cost_pct": 0.15},
    "US": {"tz": "America/New_York", "open": (9, 30), "close": (16, 0), "bench": "^GSPC", "bench_name": "S&P 500",
           "vix": "^VIX", "suffix": "", "ccy": "$", "min_value": 2.5e7, "rf": 0.04, "cost_pct": 0.05},
}

INDEX_DICT = {
    "NIFTY 50": {"yf": "^NSEI", "nse": "NIFTY", "step": 50, "lot": 65, "weekly": True},
    "BANK NIFTY": {"yf": "^NSEBANK", "nse": "BANKNIFTY", "step": 100, "lot": 30, "weekly": False},
}
UNDERLYING_YF = {v["nse"]: v["yf"] for v in INDEX_DICT.values()}

SECTOR_INDICES = {
    "BANK": "^NSEBANK",
    "IT": "^CNXIT",
    "PHARMA": "^CNXPHARMA",
    "AUTO": "^CNXAUTO",
    "FMCG": "^CNXFMCG",
    "METAL": "^CNXMETAL"
}

STOCK_SECTOR_MAP = {
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "SBIN": "BANK", "AXISBANK": "BANK", "KOTAKBANK": "BANK",
    "INFY": "IT", "TCS": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DRREDDY": "PHARMA", "ZYDUSLIFE": "PHARMA",
    "MARUTI": "AUTO", "M&M": "AUTO", "EICHERMOT": "AUTO", "TMPV": "AUTO",
    "TATASTEEL": "METAL", "JSWSTEEL": "METAL", "HINDALCO": "METAL",
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "TITAN": "FMCG", "TRENT": "FMCG",
    "RELIANCE": "ENERGY", "NTPC": "ENERGY", "POWERGRID": "ENERGY", "ONGC": "ENERGY", "COALINDIA": "ENERGY",
    "ADANIENSOL": "ENERGY", "BEL": "DEFENCE", "LT": "INFRA", "INDUSTOWER": "TELECOM"
}

FNO_WATCHLIST = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "TMPV", "SBIN", "BHARTIARTL", "AXISBANK", "LT",
    "KOTAKBANK", "MARUTI", "BAJFINANCE", "ITC", "HINDUNILVR", "SUNPHARMA", "TITAN", "M&M", "NTPC",
    "HCLTECH", "TATASTEEL", "JSWSTEEL", "HINDALCO", "ADANIENT", "BEL", "TRENT", "ONGC", "COALINDIA",
    "DRREDDY", "CIPLA", "EICHERMOT", "ULTRACEMCO", "ASIANPAINT", "BAJAJFINSV", "WIPRO", "TECHM"
]
SYMBOL_RENAMES = {"TATAMOTORS": "TMPV"}

def round_to_strike(price, step=50):
    return int(round(price / step) * step)

def stock_strike_step(price):
    if price < 250:
        return 5
    if price < 500:
        return 10
    if price < 1000:
        return 20
    if price < 2500:
        return 50
    if price < 5000:
        return 100
    return 250

def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)

def _f(x, default=np.nan):
    try:
        v = float(x)
        return default if (v != v) else v
    except Exception:
        return default

# ==============================================================================
# DATA QUALITY, FRESHNESS & DIAGNOSTICS MANAGER
# ==============================================================================
class DataQualityManager:
    @staticmethod
    def inspect_dataframe(df, market="IN"):
        """[FIX-10] Daily bars are stamped at midnight, so 'hours since bar' always read DELAYED.
        Judge freshness by trading sessions instead."""
        if df is None or df.empty:
            return {"status": "UNAVAILABLE", "badge": "🔴 UNAVAILABLE", "last_bar": "Never", "is_stale": True}
        last_d = pd.Timestamp(df.index[-1]).date()
        now = datetime.now(ZoneInfo(MARKETS[market]["tz"]))
        gap = int(np.busday_count(last_d, now.date())) if last_d <= now.date() else 0
        prog, _ = session_progress(market)
        if gap == 0:
            badge = "🟢 TODAY — intraday (≈15 min delayed)" if prog < 1.0 else "🟢 TODAY'S BAR"
            status, stale = "FRESH", False
        elif gap <= 2:
            badge, status, stale = "🟡 LAST SESSION CLOSE", "DELAYED", False
        else:
            badge, status, stale = f"🔴 STALE — {gap} sessions old", "HISTORICAL", True
        return {"status": status, "badge": badge, "last_bar": last_d.isoformat(), "is_stale": stale}

    @staticmethod
    def render_badge(diag, source_label="Yahoo Finance"):
        st.caption(f"**Data Status:** {diag['badge']} | **Source:** {source_label} | **Last Bar:** `{diag['last_bar']}`")

# ==============================================================================
# PERSISTENCE & STORAGE HELPERS
# ==============================================================================
def _ensure_ids(df):
    if "Row_ID" not in df.columns:
        df["Row_ID"] = None
    ids = df["Row_ID"].astype(str)
    mask = df["Row_ID"].isna() | ids.isin(["", "nan", "None"])
    if mask.any():
        df.loc[mask, "Row_ID"] = [uuid.uuid4().hex[:8] for _ in range(int(mask.sum()))]
    df["Row_ID"] = df["Row_ID"].astype(str)
    dup = df["Row_ID"].duplicated(keep="first")
    if dup.any():
        df.loc[dup, "Row_ID"] = [uuid.uuid4().hex[:8] for _ in range(int(dup.sum()))]
    return df

def load_csv_data(filepath, columns):
    loaded_df = None
    if os.path.exists(filepath):
        try:
            loaded_df = pd.read_csv(filepath, dtype={"Row_ID": str})
        except Exception:
            loaded_df = None
    if loaded_df is None:
        loaded_df = pd.DataFrame(columns=columns)
    for col in columns:
        if col not in loaded_df.columns:
            loaded_df[col] = None
    loaded_df = loaded_df[columns].astype(object)
    if "Row_ID" in columns:
        loaded_df = _ensure_ids(loaded_df)
    return loaded_df

def save_csv_data(df, filepath):
    df.to_csv(filepath, index=False)

if "eq_portfolio" not in st.session_state:
    st.session_state.eq_portfolio = load_csv_data(EQUITY_PORTFOLIO_FILE, EQ_COLS)
if "opt_portfolio" not in st.session_state:
    st.session_state.opt_portfolio = load_csv_data(OPTIONS_PORTFOLIO_FILE, OPT_COLS)
if "us_eq_portfolio" not in st.session_state:
    st.session_state.us_eq_portfolio = load_csv_data(US_EQUITY_PORTFOLIO_FILE, EQ_COLS)
if "journal" not in st.session_state:
    st.session_state.journal = load_csv_data(JOURNAL_FILE, JOURNAL_COLS)

PORTFOLIO_KEYS = {"IN": ("eq_portfolio", EQUITY_PORTFOLIO_FILE), "US": ("us_eq_portfolio", US_EQUITY_PORTFOLIO_FILE)}

def log_to_journal(market, instrument, direction, entry, exit_p, qty, entry_date, reason, setup="", init_risk=None):
    sign = 1 if direction == "LONG" else -1
    entry, exit_p, qty = _f(entry, 0.0), _f(exit_p, 0.0), _f(qty, 0.0)
    pnl = (exit_p - entry) * qty * sign
    pnl_pct = ((exit_p / entry) - 1) * 100 * sign if entry else 0.0
    r_mult = ((exit_p - entry) * sign / init_risk) if init_risk and init_risk > 0 else None
    row = pd.DataFrame([{
        "Market": market, "Instrument": instrument, "Direction": direction, "Entry": round(entry, 2),
        "Exit": round(exit_p, 2), "Qty": qty, "PnL": round(pnl, 2), "PnL_Pct": round(pnl_pct, 2),
        "R_Multiple": round(r_mult, 2) if r_mult is not None else None, "Entry_Date": entry_date,
        "Exit_Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "Exit_Reason": reason, "Setup": setup
    }])
    st.session_state.journal = pd.concat([st.session_state.journal, row], ignore_index=True)
    save_csv_data(st.session_state.journal, JOURNAL_FILE)

# ==============================================================================
# 1. MARKET DATA LAYER
# ==============================================================================
def _extract_ticker_frame(data, ticker):
    if data is None or len(data) == 0:
        return None
    df = None
    if isinstance(data.columns, pd.MultiIndex):
        for lvl in range(data.columns.nlevels):
            if ticker in data.columns.get_level_values(lvl):
                df = data.xs(ticker, axis=1, level=lvl).copy()
                break
        if df is None:
            return None
    else:
        df = data.copy()
    df.columns = [str(c).strip().title() for c in df.columns]
    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return None
    df = df[need].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close", "High", "Low"])
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["Volume"] = df["Volume"].fillna(0.0)
    return df if not df.empty else None

@st.cache_data(ttl=600, show_spinner=False)
def download_many(tickers, period="2y", interval="1d"):
    out = {}
    tickers = list(tickers)
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        try:
            data = yf.download(chunk, period=period, interval=interval, group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue
        for t in chunk:
            try:
                df = _extract_ticker_frame(data, t)
                if df is not None:
                    out[t] = df
            except Exception:
                continue
    return out

def get_history(ticker, period="2y"):
    return download_many((ticker,), period).get(ticker, pd.DataFrame())

def session_progress(market):
    m = MARKETS[market]
    now = datetime.now(ZoneInfo(m["tz"]))
    if now.weekday() >= 5:
        return 1.0, now
    o = now.replace(hour=m["open"][0], minute=m["open"][1], second=0, microsecond=0)
    c = now.replace(hour=m["close"][0], minute=m["close"][1], second=0, microsecond=0)
    if now <= o or now >= c:
        return 1.0, now
    return max(0.02, (now - o).total_seconds() / (c - o).total_seconds()), now

def projected_rel_volume(df, market):
    if df is None or len(df) < 22:
        return np.nan, False
    vol_avg = float(df["Volume"].iloc[-21:-1].mean())
    if vol_avg <= 0:
        return np.nan, False
    vol = float(df["Volume"].iloc[-1])
    prog, now = session_progress(market)
    is_today = df.index[-1].date() == now.date()
    if is_today and prog < 1.0:
        if prog < 0.08:
            return np.nan, True
        return vol / (prog ** 0.85) / vol_avg, True
    return vol / vol_avg, False

# ==============================================================================
# 2. TECHNICAL INDICATOR ENGINE
# ==============================================================================
def calculate_sma(series, window=20):
    return series.rolling(window=window).mean()

def calculate_ema(series, span=20):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, window=14):
    """Wilder's smoothed RSI with flat-price protection."""
    if series.empty or series.nunique() <= 1:
        return pd.Series(50.0, index=series.index)
    
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    safe_loss = avg_loss.replace(0.0, 1e-6)
    rs = avg_gain / safe_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def calculate_macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return sma + std * num_std, sma, sma - std * num_std

def _wilder(series, n):
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

def calculate_atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return _wilder(tr, n)

def calculate_adx(df, n=14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = calculate_atr(df, n).replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, n) / atr
    minus_di = 100 * _wilder(minus_dm, n) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, n), plus_di, minus_di

def add_indicators(df_in):
    df = df_in.copy()
    c = df["Close"]
    df["EMA_20"] = calculate_ema(c, 20)
    df["EMA_50"] = calculate_ema(c, 50)
    df["SMA_50"] = calculate_sma(c, 50)
    df["SMA_150"] = calculate_sma(c, 150)
    df["SMA_200"] = calculate_sma(c, 200)
    df["SMA200_Slope"] = df["SMA_200"] - df["SMA_200"].shift(21)
    df["RSI"] = calculate_rsi(c, 14)
    df["MACD"], df["Signal"], df["Hist"] = calculate_macd(c)
    df["Hist_Prev"] = df["Hist"].shift(1)
    df["ATR"] = calculate_atr(df, 14)
    df["ATR_Pct"] = df["ATR"] / c * 100
    df["ADX"], df["PDI"], df["MDI"] = calculate_adx(df, 14)
    vol_avg = df["Volume"].rolling(20).mean().shift(1)
    df["RelVol"] = df["Volume"] / vol_avg.replace(0, np.nan)
    df["High_20"] = df["High"].rolling(20).max().shift(1)
    df["Low_20"] = df["Low"].rolling(20).min().shift(1)
    df["Low_5"] = df["Low"].rolling(5).min()
    df["High_5"] = df["High"].rolling(5).max()
    df["High_252"] = df["High"].rolling(252, min_periods=200).max()
    df["Low_252"] = df["Low"].rolling(252, min_periods=200).min()
    up, mid, lo = calculate_bollinger_bands(c)
    df["BBW"] = (up - lo) / mid
    df["BBW_Rank"] = df["BBW"].rolling(126, min_periods=60).rank(pct=True)
    logret = np.log(c / c.shift(1))
    df["HV20"] = logret.rolling(20).std() * math.sqrt(252) * 100
    df["HV_Rank"] = df["HV20"].rolling(252, min_periods=120).rank(pct=True)
    df["TradedValue20"] = (c * df["Volume"]).rolling(20).mean()
    
    wk = c.resample("W-FRI").last().dropna()
    if len(wk) >= 30:
        w_ema = calculate_ema(wk, 10)
        _, _, w_hist = calculate_macd(wk)
        w_up = ((wk > w_ema) & (w_hist > 0)).astype(float)
        w_dn = ((wk < w_ema) & (w_hist < 0)).astype(float)
        df["Weekly_Up"] = w_up.reindex(df.index, method="ffill").fillna(0.0)
        df["Weekly_Down"] = w_dn.reindex(df.index, method="ffill").fillna(0.0)
    else:
        df["Weekly_Up"] = 0.0
        df["Weekly_Down"] = 0.0
    return df

# ==============================================================================
# 3. PHASE 4: MARKET REGIME & SECTOR BREADTH ENGINE
# ==============================================================================
def regime_series(bench_df):
    c = bench_df["Close"]
    s50, s200 = calculate_sma(c, 50), calculate_sma(c, 200)
    pts = (c > s200).astype(int) + (c > s50).astype(int) + (s50 > s50.shift(10)).astype(int)
    lab = pd.Series(np.where(pts >= 3, "RISK-ON", np.where(pts == 2, "NEUTRAL", "RISK-OFF")), index=c.index)
    lab[s200.isna()] = "NEUTRAL"
    return lab

@st.cache_data(ttl=900, show_spinner=False)
def get_market_regime(market):
    m = MARKETS[market]
    data = download_many((m["bench"], m["vix"]), "2y")
    bench = data.get(m["bench"])
    out = {"label": "NEUTRAL", "bench_close": None, "vix": None, "vix_pct": None,
           "bench_ret_63": None, "details": "Benchmark data unavailable"}
    if bench is None or len(bench) < 210:
        return out
    lab = regime_series(bench)
    c = bench["Close"]
    s50, s200 = calculate_sma(c, 50).iloc[-1], calculate_sma(c, 200).iloc[-1]
    out.update({
        "label": lab.iloc[-1], "bench_close": c,
        "bench_ret_63": float(c.iloc[-1] / c.iloc[-64] - 1) * 100,
        "details": f"{m['bench_name']} {c.iloc[-1]:,.1f} | SMA50 {s50:,.1f} | SMA200 {s200:,.1f}",
        "regime_hist": lab,
    })
    vix = data.get(m["vix"])
    if vix is not None and len(vix) > 100:
        v = vix["Close"]
        out["vix"] = float(v.iloc[-1])
        out["vix_pct"] = float((v.iloc[-252:] <= v.iloc[-1]).mean())
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def get_sector_regimes():
    tickers = list(SECTOR_INDICES.values())
    data = download_many(tuple(tickers), period="1y")
    sector_health = {}
    for sec, sym in SECTOR_INDICES.items():
        df = data.get(sym)
        if df is not None and len(df) >= 50:
            c = df["Close"]
            s50 = calculate_sma(c, 50).iloc[-1]
            cur = c.iloc[-1]
            sector_health[sec] = "BULLISH" if cur >= s50 else "BEARISH"
        else:
            sector_health[sec] = "NEUTRAL"
    return sector_health

def rs_raw_score(close):
    n = len(close)
    def r(k):
        k = min(k, n - 1)
        return float(close.iloc[-1] / close.iloc[-1 - k] - 1) if k > 0 else 0.0
    return 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)

def rs_proxy_rating(excess_63d_pct):
    if excess_63d_pct is None or excess_63d_pct != excess_63d_pct:
        return 50
    if excess_63d_pct > 10:
        return 85
    if excess_63d_pct > 0:
        return 65
    if excess_63d_pct > -10:
        return 45
    return 25

# ==============================================================================
# 4. CONFLUENCE & SECTOR-AWARE SCORING ENGINE
# ==============================================================================
def score_setup(r, symbol="", direction="long", rs_rating=None, regime="NEUTRAL", sector_health=None, relvol=None, sentiment=None, earnings_days=None):
    L = direction == "long"
    pts, mx, why, flags = 0.0, 0.0, [], []

    def add(cond, p, label):
        nonlocal pts, mx
        mx += p
        if bool(cond):
            pts += p
            why.append(label)

    c, o = _f(r["Close"]), _f(r["Open"])
    e20, e50, s200 = _f(r["EMA_20"]), _f(r["EMA_50"]), _f(r["SMA_200"])
    atr = _f(r["ATR"], 0.0) or max(c * 0.02, 1e-6)
    rsi, adx = _f(r["RSI"], 50), _f(r["ADX"], 0)

    # Trend structure (25 pts)
    if L:
        add(c > e20 > e50, 10, "Uptrend Structure (Price > EMA20 > EMA50)")
        add(c > s200 and _f(r["SMA200_Slope"], 0) > 0, 10, "Above rising 200-SMA")
        add(_f(r["Weekly_Up"], 0) > 0, 5, "Weekly multi-timeframe confirmation")
    else:
        add(c < e20 < e50, 10, "Downtrend Structure (Price < EMA20 < EMA50)")
        add(c < s200 and _f(r["SMA200_Slope"], 0) < 0, 10, "Below falling 200-SMA")
        add(_f(r["Weekly_Down"], 0) > 0, 5, "Weekly multi-timeframe confirmation")

    # Momentum (20 pts)
    hist, macd, sig = _f(r["Hist"], 0), _f(r["MACD"], 0), _f(r["Signal"], 0)
    hist_prev = _f(r.get("Hist_Prev", np.nan), hist)
    if L:
        add(macd > sig and hist >= hist_prev, 8, "MACD expanding upward")
        rsi_pts = 7 if 48 <= rsi <= 62 else (4 if (42 <= rsi < 48 or 62 < rsi <= 68) else 0)
        add(_f(r["PDI"], 0) > _f(r["MDI"], 0) and adx >= 20, 5, f"ADX {adx:.0f} confirmed trend")
    else:
        add(macd < sig and hist <= hist_prev, 8, "MACD expanding downward")
        rsi_pts = 7 if 38 <= rsi <= 52 else (4 if (32 <= rsi < 38 or 52 < rsi <= 58) else 0)
        add(_f(r["MDI"], 0) > _f(r["PDI"], 0) and adx >= 20, 5, f"ADX {adx:.0f} confirmed trend")
    mx += 7
    pts += rsi_pts
    if rsi_pts:
        why.append(f"RSI {rsi:.0f} in sweet-spot (non-extended)")

    # Volume (15 pts)
    rv = relvol if relvol is not None else _f(r["RelVol"])
    if rv == rv and rv is not None and np.isfinite(rv):
        bar_ok = (c > o) if L else (c < o)
        vpts = (15 if rv >= 1.5 else 10 if rv >= 1.2 else 4 if rv >= 1.0 else 0) if bar_ok else 0
        mx += 15
        pts += vpts
        if vpts:
            why.append(f"{rv:.2f}x volume expansion")

    # Relative strength (20 pts)
    if rs_rating is not None:
        if L:
            rpts = 20 if rs_rating >= 80 else 14 if rs_rating >= 70 else 6 if rs_rating >= 50 else 0
        else:
            rpts = 20 if rs_rating <= 20 else 14 if rs_rating <= 30 else 6 if rs_rating <= 50 else 0
        mx += 20
        pts += rpts
        if rpts:
            why.append(f"Leader RS rating {rs_rating:.0f}")

    # LEADING SETUP CLASSIFICATION (10 pts)
    h20, l20, bbw_rank = _f(r["High_20"]), _f(r["Low_20"]), _f(r["BBW_Rank"], 1.0)
    if L:
        pullback = (_f(r["Low"]) <= e20 + 0.35 * atr) and (c >= e20) and (c > o)
        squeeze = bbw_rank <= 0.18
        breakout = (c > h20) and (c - e20 <= 1.5 * atr)
    else:
        pullback = (_f(r["High"]) >= e20 - 0.35 * atr) and (c <= e20) and (c < o)
        squeeze = bbw_rank <= 0.18
        breakout = (c < l20) and (e20 - c <= 1.5 * atr)

    if pullback:
        setup, spts = "Pullback-to-EMA20 Entry (Leading/Low Risk)", 10
    elif squeeze:
        setup, spts = "Volatility Squeeze Expansion (Leading)", 9
    elif breakout:
        setup, spts = "Fresh Pivot Breakout", 7
    else:
        trend_ok = (c > e20 > e50) if L else (c < e20 < e50)  # [FIX-07] no points for 'continuation' without a trend
        setup, spts = ("Trend Continuation", 3) if trend_ok else ("No Setup", 0)

    mx += 10
    pts += spts
    why.append(setup)

    # 52-week position (10 pts)
    h252, l252 = _f(r["High_252"]), _f(r["Low_252"])
    if h252 == h252 and l252 == l252:
        if L:
            add(c >= 0.75 * h252 and c >= 1.25 * l252, 10, "Stage-2 Leader (Within 25% of 52w High)")
        else:
            add(c <= 1.25 * l252 and c <= 0.75 * h252, 10, "Stage-4 Breakdown (Near 52w Low)")

    score = 100.0 * pts / mx if mx else 0.0

    # STRICT PENALTIES & EXTENSION VETOES
    ext = ((c - e20) if L else (e20 - c)) / atr
    if ext > 1.8:
        score -= 25
        flags.append(f"EXTENDED: Price is {ext:.1f}x ATR above 20-EMA. Pullback risk high!")
    elif ext > 1.4:
        score -= 10
        flags.append(f"Slightly stretched ({ext:.1f}x ATR from 20-EMA)")

    if (L and rsi > 70) or ((not L) and rsi < 30):
        score -= 15
        flags.append("RSI Overbought/Oversold exhaustion")

    # Granular Sector Breadth Check
    stock_sector = STOCK_SECTOR_MAP.get(symbol)
    sec_trend = (sector_health or {}).get(stock_sector, "NEUTRAL") if stock_sector else None

    if L and regime == "RISK-OFF":
        if sec_trend == "BULLISH":
            score -= 5
            flags.append(f"Broad market RISK-OFF, but {stock_sector} sector is BULLISH (mild -5 cushion)")
        else:
            score -= 20
            flags.append("Market in RISK-OFF (High breakout failure rate)")

    # [FIX-07] sentiment penalty integration
    if sentiment is not None and ((L and sentiment <= -0.3) or ((not L) and sentiment >= 0.3)):
        score -= 5
        flags.append("News sentiment against the trade")

    # Event Risk (Earnings within 5 days)
    if earnings_days is not None and 0 <= earnings_days <= 5:
        score -= 15
        flags.append(f"Earnings in {earnings_days}d — binary gap risk!")

    score = max(0.0, min(100.0, score))
    grade = "A+" if score >= 85 else "A" if score >= 75 else "B" if score >= 65 else "C"
    return {"score": round(score, 1), "grade": grade, "setup": setup, "why": why, "flags": flags, "sector": stock_sector or "General"}

def compute_levels(r, direction="long", atr_mult=1.8, entry=None, t1_r=2.0, t2_r=3.0):
    L = direction == "long"
    entry = _f(entry if entry is not None else r["Close"])
    atr = _f(r["ATR"], 0.0) or (entry * 0.02)
    if L:
        swing = _f(r["Low_5"], entry - atr) - 0.2 * atr
        stop = min(max(swing, entry - atr_mult * atr), entry - 0.9 * atr)
        risk = entry - stop
        t1, t2 = entry + t1_r * risk, entry + t2_r * risk
    else:
        swing = _f(r["High_5"], entry + atr) + 0.2 * atr
        stop = max(min(swing, entry + atr_mult * atr), entry + 0.9 * atr)
        risk = stop - entry
        t1, t2 = entry - t1_r * risk, entry - t2_r * risk
    return {"entry": round(entry, 2), "stop": round(stop, 2), "risk": round(risk, 2),
            "t1": round(t1, 2), "t2": round(t2, 2), "risk_pct": round(risk / entry * 100, 2)}

def position_size(capital, risk_pct, entry, stop):
    risk_ps = abs(entry - stop)
    if risk_ps <= 0 or entry <= 0:
        return 0
    return int(max(0, min((capital * risk_pct / 100.0) // risk_ps, capital // entry)))

# ==============================================================================
# PHASE 5: RECOMMENDATION & RISK DECISION ENGINE
# ==============================================================================
def render_trade_decision_card(row, market, capital, risk_pct):
    m = MARKETS[market]
    score = float(row["Score"])
    flags = [f.strip() for f in str(row["Flags"]).split(";") if f.strip()]
    why = [w.strip() for w in str(row["Why"]).split(";") if w.strip()]
    entry = float(row[f"Price ({m['ccy']})"])
    stop = float(row[f"Stop ({m['ccy']})"])
    t1 = float(row[f"T1 2R ({m['ccy']})"])
    t2 = float(row[f"T2 3R ({m['ccy']})"])
    
    qty = position_size(capital, risk_pct, entry, stop)
    total_outlay = qty * entry
    risk_rupees = qty * abs(entry - stop)
    rr_ratio = abs(t1 - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0.0

    # Decision Engine Classification Logic
    if score >= 60 and not any("EXTENDED" in f for f in flags) and rr_ratio >= 1.8:
        status_label = "🟢 READY TO ENTER (Confluence Validated)"
        status_desc = "Setup shows structural alignment, acceptable risk distance, and attractive reward:risk."
        box_type = st.success
    elif score >= 40:
        status_label = "🟡 WATCHLIST / PULLBACK REQUIRED"
        status_desc = "Strong relative strength, but macro regime or distance from 20-EMA demands caution."
        box_type = st.warning
    else:
        status_label = "🔴 NO-TRADE / CAPITAL PROTECTION"
        status_desc = "Setup carries structural flaws, breakdown momentum, or binary event risks. Stand aside."
        box_type = st.error

    st.markdown("### 📋 Executive Trade Decision Card")
    box_type(f"**Recommendation:** {status_label}\n\n{status_desc}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🟢 Why Enter (Positive Edge Factors):**")
        if why:
            for reason in why:
                st.markdown(f"- {reason}")
        else:
            st.caption("No strong technical catalysts identified.")

    with c2:
        st.markdown("**⚠️ Why Avoid / Caution (Veto & Risk Factors):**")
        if flags:
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.markdown("- Clean setup with no major risk flags.")

    st.markdown("#### **🎯 Order Execution Ticket**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Execution Order", f"BUY {qty} Shares")
    m2.metric("Total Outlay", f"{m['ccy']}{total_outlay:,.2f}")
    m3.metric("Hard Stop Loss", f"{m['ccy']}{stop:,.2f}", f"-{row['Risk %']}%")
    m4.metric("Max Portfolio Risk", f"-{m['ccy']}{risk_rupees:,.2f}", f"{risk_pct}%")

    o1, o2 = st.columns(2)
    o1.info(f"**Target 1 (2R - Scale 50%):** {m['ccy']}{t1:,.2f}")
    o2.success(f"**Target 2 (3R - Runner Exit):** {m['ccy']}{t2:,.2f}")

# ==============================================================================
# [FIX-01..05] PRO INDEX-OPTIONS STRATEGY ENGINE
# ==============================================================================
def ist_today():
    """[FIX-12] Streamlit Cloud runs on UTC; expiry maths must use the IST calendar date."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).date()


def score_index_setup(r, direction="long", regime="NEUTRAL", sentiment=None):
    """[FIX-01] Index confluence score (0-100). No 52-week / relative-strength / volume rules."""
    L = direction == "long"
    pts, mx, why, flags = 0.0, 0.0, [], []

    def add(cond, p, label):
        nonlocal pts, mx
        mx += p
        if bool(cond):
            pts += p
            why.append(label)

    c, o, lo, hi = _f(r["Close"]), _f(r["Open"]), _f(r["Low"]), _f(r["High"])
    e20, e50, s200 = _f(r["EMA_20"]), _f(r["EMA_50"]), _f(r["SMA_200"])
    atr = _f(r["ATR"], 0.0) or max(c * 0.01, 1e-6)
    rsi, adx = _f(r["RSI"], 50), _f(r["ADX"], 0)
    pdi, mdi = _f(r["PDI"], 0), _f(r["MDI"], 0)
    macd, sig, hist = _f(r["MACD"], 0), _f(r["Signal"], 0), _f(r["Hist"], 0)
    hist_prev = _f(r.get("Hist_Prev", np.nan), hist)

    # Trend (35)
    stack = (c > e20 > e50) if L else (c < e20 < e50)
    add(stack, 15, "EMA stack aligned (price / EMA20 / EMA50)")
    add((c > s200) if L else (c < s200), 10, "Correct side of 200-SMA")
    add(_f(r["Weekly_Up" if L else "Weekly_Down"], 0) > 0, 10, "Weekly trend confirms")
    
    # Momentum (25)
    add((macd > sig and hist >= hist_prev) if L else (macd < sig and hist <= hist_prev), 10, "MACD momentum expanding")
    if L:
        rpts = 10 if 45 <= rsi <= 65 else (5 if (40 <= rsi < 45 or 65 < rsi <= 70) else 0)
    else:
        rpts = 10 if 35 <= rsi <= 55 else (5 if (30 <= rsi < 35 or 55 < rsi <= 60) else 0)
    mx += 10
    pts += rpts
    if rpts:
        why.append(f"RSI {rsi:.0f} in tradeable zone")
    add(adx >= 18 and ((pdi > mdi) if L else (mdi > pdi)), 5, f"ADX {adx:.0f} with DI alignment")
    
    # Entry setup (20)
    ext = ((c - e20) if L else (e20 - c)) / atr
    ema_ok = (e20 > e50) if L else (e20 < e50)
    if L:
        pullback = ema_ok and lo <= e20 + 0.5 * atr and -0.25 <= ext <= 1.0 and c > o
        breakout = c > _f(r["High_20"], np.inf) and ext <= 1.5
    else:
        pullback = ema_ok and hi >= e20 - 0.5 * atr and -0.25 <= ext <= 1.0 and c < o
        breakout = c < _f(r["Low_20"], -np.inf) and ext <= 1.5
    squeeze = _f(r["BBW_Rank"], 1.0) <= 0.20 and stack and 0 <= ext <= 1.5
    continuation = stack and 0 <= ext <= 1.0
    if pullback:
        setup, sp = "Pullback to 20-EMA", 20
    elif squeeze:
        setup, sp = "Volatility squeeze in trend", 15
    elif breakout:
        setup, sp = ("Fresh 20-day breakout" if L else "Fresh 20-day breakdown"), 12
    elif continuation:
        setup, sp = "Trend continuation near 20-EMA", 8
    else:
        setup, sp = "No valid entry setup", 0
    mx += 20
    pts += sp
    if sp:
        why.append(setup)

    score = 100.0 * pts / mx if mx else 0.0
    if ext > 2.0:
        score -= 20
        flags.append(f"EXTENDED {ext:.1f} ATR from 20-EMA — don't chase")
    elif ext > 1.5:
        score -= 8
        flags.append(f"Stretched {ext:.1f} ATR from 20-EMA")
    if (L and rsi > 72) or ((not L) and rsi < 28):
        score -= 10
        flags.append(f"RSI {rsi:.0f} exhausted")
    if L and regime == "RISK-OFF":
        score -= 10
        flags.append("Market regime RISK-OFF")
    if (not L) and regime == "RISK-ON":
        score -= 10
        flags.append("Market regime RISK-ON")
    if sentiment is not None and ((L and sentiment <= -0.3) or ((not L) and sentiment >= 0.3)):
        score -= 5
        flags.append("News sentiment against the trade")
    score = max(0.0, min(100.0, score))
    grade = "A+" if score >= 85 else "A" if score >= 75 else "B" if score >= 65 else "C"
    return {"score": round(score, 1), "grade": grade, "setup": setup, "why": why, "flags": flags, "ext": ext}


def _prob_above(S, K, T, sigma, r=0.0):
    """Risk-neutral probability that the index expires above K."""
    T, sigma = max(T, 1 / 365), max(sigma, 0.01)
    d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _ncdf(d2)


def _leg_price(chain, strike, typ, S, T, r, iv_pct):
    if chain is not None:
        hit = chain["chain"][chain["chain"]["strike"] == strike]
        if not hit.empty:
            v = _f(hit.iloc[0]["ce_ltp" if typ == "CE" else "pe_ltp"])
            if v == v and v > 0:
                return round(v, 2), "live"
    return round(bs_greeks(S, strike, T, r, iv_pct / 100.0, typ)["price"], 2), "model"


def legs_to_str(legs):
    return "|".join(f"{l['side']}:{int(l['strike'])}:{l['type']}" for l in legs)


def parse_legs(s):
    out = []
    for part in str(s or "").split("|"):
        bits = part.split(":")
        if len(bits) == 3 and bits[0] in ("B", "S") and bits[2] in ("CE", "PE"):
            k = _f(bits[1])
            if k == k:
                out.append({"side": bits[0], "strike": k, "type": bits[2]})
    return out


def net_from_prices(legs, prices, side):
    """Net value of a structure: debit side = cost to own; credit side = cost to buy back."""
    raw = sum((p if l["side"] == "B" else -p) for l, p in zip(legs, prices))
    return raw if side == "DEBIT" else -raw


def finalize_plan(plan, net, capital, risk_pct):
    """[FIX-05] Single source of truth for stop/target/size from actual net premium."""
    lot, kind = plan["lot"], plan["kind"]
    if kind == "LONG OPTION":
        stop, target = net * plan["sl_ratio"], net * plan["tgt_ratio"]
        plan_loss_u, max_loss_u, reward_u = net - stop, net, target - net
    elif kind == "DEBIT SPREAD":
        w = plan["width"]
        stop, target = 0.5 * net, net + 0.6 * max(w - net, 0)
        plan_loss_u, max_loss_u, reward_u = net - stop, net, target - net
    else:  # IRON CONDOR
        w = plan["wing"]
        stop, target = 2.0 * net, 0.5 * net
        plan_loss_u, max_loss_u, reward_u = stop - net, w - net, net - target
    budget = capital * risk_pct / 100.0
    per_lot = max_loss_u * lot
    lots = int(budget // per_lot) if per_lot > 0 else 0
    return {"stop": round(stop, 1), "target": round(target, 1), "risk_per_lot": round(per_lot),
            "max_loss_per_lot": round(max_loss_u * lot), "planned_loss_per_lot": round(plan_loss_u * lot),
            "reward_per_lot": round(reward_u * lot), "lots": lots,
            "one_lot_risk_pct": round(per_lot / capital * 100, 2) if capital else None,
            "rr": round(reward_u / plan_loss_u, 2) if plan_loss_u > 0 else 0.0}


def _index_iv(meta, chain, regime, hv):
    if chain and chain.get("atm_iv"):
        return float(chain["atm_iv"]), "NSE ATM IV"
    if meta["nse"] == "NIFTY" and regime.get("vix"):
        return float(regime["vix"]), "India VIX"
    base = hv * 1.1 if hv == hv and hv > 0 else 15.0
    return float(base), "HV20 × 1.1 proxy"


def _directional_plan(plan, dirn, entry_spot, last, meta, iv, iv_rich, chain_e, dte, c, event_risk):
    rf, step = MARKETS["IN"]["rf"], meta["step"]
    T = max(dte, 1) / 365
    lv = compute_levels(last, "long" if dirn == "CE" else "short", c["atr_mult"], entry=entry_spot)  # [FIX-04] PE stop above spot
    em = entry_spot * iv / 100 * math.sqrt(max(dte, 1) / 365)
    t1_move = min(abs(lv["t1"] - entry_spot), em)
    sgn = 1 if dirn == "CE" else -1
    t1_spot = entry_spot + sgn * t1_move
    atm = round_to_strike(entry_spot, step)
    k_itm = atm - step if dirn == "CE" else atm + step
    g = bs_greeks(entry_spot, k_itm, T, rf, iv / 100, dirn)
    theta_pct = abs(g["theta"]) / g["price"] * 100
    use_spread = iv_rich or event_risk or theta_pct > 3.5
    if use_spread:
        k1 = atm
        k2 = round_to_strike(t1_spot, step)
        k2 = max(k2, k1 + 2 * step) if dirn == "CE" else min(k2, k1 - 2 * step)
        p1, s1 = _leg_price(chain_e, k1, dirn, entry_spot, T, rf, iv)
        p2, s2 = _leg_price(chain_e, k2, dirn, entry_spot, T, rf, iv)
        legs = [{"side": "B", "strike": k1, "type": dirn, "price": p1, "src": s1},
                {"side": "S", "strike": k2, "type": dirn, "price": p2, "src": s2}]
        width = abs(k2 - k1)
        net = round(p1 - p2, 2)
        why_vehicle = ("IV is rich" if iv_rich else "event risk before expiry" if event_risk
                       else f"theta {theta_pct:.1f}%/day on a single option")
        plan.update({"kind": "DEBIT SPREAD", "strategy": "BULL CALL SPREAD" if dirn == "CE" else "BEAR PUT SPREAD",
                     "width": width, "vehicle_reason": f"Debit spread because {why_vehicle}: the short leg finances decay and IV risk."})
        plan["breakevens"] = [k1 + net if dirn == "CE" else k1 - net]
        plan["pop"] = (_prob_above(entry_spot, plan["breakevens"][0], T, iv / 100, rf) if dirn == "CE"
                       else 1 - _prob_above(entry_spot, plan["breakevens"][0], T, iv / 100, rf))
        exits = ["Take profit when the spread is worth the target value (≈60% of max profit captured).",
                 "Stop when the spread loses 50% of the debit, or on a daily close beyond the underlying stop.",
                 "Exit by 1 day to expiry regardless."]
    else:
        p, s = _leg_price(chain_e, k_itm, dirn, entry_spot, T, rf, iv)
        legs = [{"side": "B", "strike": k_itm, "type": dirn, "price": p, "src": s}]
        net = p
        prem_sl = max(p - abs(g["delta"]) * abs(entry_spot - lv["stop"]), 0.5 * p)
        prem_tgt = p + abs(g["delta"]) * t1_move - abs(g["theta"]) * min(2, dte)
        plan.update({"kind": "LONG OPTION", "strategy": f"LONG {'CALL' if dirn == 'CE' else 'PUT'} (1 strike ITM)",
                     "sl_ratio": prem_sl / p, "tgt_ratio": max(prem_tgt / p, 1.2),
                     "vehicle_reason": f"Outright 1-strike ITM option: IV fair/cheap and theta only {theta_pct:.1f}%/day; Δ≈{g['delta']:+.2f}."})
        plan["breakevens"] = [k_itm + p if dirn == "CE" else k_itm - p]
        plan["pop"] = None
        exits = ["Book 50% at the premium target (underlying ≈ T1); trail the rest at the 20-EMA.",
                 "Exit on the premium stop or a daily close beyond the underlying stop — whichever comes first.",
                 "Time stop: if the option is not up ≥10% after 2 sessions, exit (theta is winning).",
                 "Exit by 1 day to expiry regardless."]
    plan.update({"dirn": dirn, "legs": legs, "net": net, "side": "DEBIT", "und_stop": lv["stop"], "und_stop2": None,
                 "t1_spot": round(t1_spot, 1), "em": round(em, 1), "theta_pct": round(theta_pct, 2), "exits": exits})
    return plan


def _condor_plan(plan, spot, meta, iv, chain_e, dte):
    rf, step = MARKETS["IN"]["rf"], meta["step"]
    T = max(dte, 1) / 365
    em = spot * iv / 100 * math.sqrt(max(dte, 1) / 365)
    sc = math.ceil((spot + em) / step) * step
    sp = math.floor((spot - em) / step) * step
    if chain_e:
        cw, pw = chain_e.get("call_wall"), chain_e.get("put_wall")
        if cw and 0 < cw - sc <= 2 * step:
            sc = cw
        if pw and 0 < sp - pw <= 2 * step:
            sp = pw
    wing = 4 * step
    spec = [("S", sc, "CE"), ("B", sc + wing, "CE"), ("S", sp, "PE"), ("B", sp - wing, "PE")]
    legs = []
    for side, k, t in spec:
        p, s = _leg_price(chain_e, k, t, spot, T, rf, iv)
        legs.append({"side": side, "strike": k, "type": t, "price": p, "src": s})
    credit = round(net_from_prices(legs, [l["price"] for l in legs], "CREDIT"), 2)
    if credit < 0.15 * wing:
        plan["mode"] = "STAND ASIDE"
        plan["reasons"].append(f"Range day, but the iron condor credit ({credit:.1f}) is under 15% of the {wing}-pt wing — not paid enough for the risk.")
        return plan
    be_lo, be_hi = sp - credit, sc + credit
    pop = _prob_above(spot, be_lo, T, iv / 100, rf) - _prob_above(spot, be_hi, T, iv / 100, rf)
    plan.update({"kind": "IRON CONDOR", "strategy": "IRON CONDOR (defined risk)", "dirn": "IC", "legs": legs,
                 "net": credit, "side": "CREDIT", "wing": wing, "und_stop": sp, "und_stop2": sc,
                 "breakevens": [round(be_lo, 1), round(be_hi, 1)], "pop": pop, "em": round(em, 1), "t1_spot": None,
                 "vehicle_reason": (f"No trend (ADX < 20) and price inside its 20-day range: sell premium outside the "
                                    f"1σ expected move (±{em:,.0f}) with {wing}-pt protective wings."),
                 "exits": ["Take profit at 50% of the credit (buy back at the target value).",
                           "Stop when the condor costs 2× the credit to close, or if spot trades through a short strike.",
                           "Exit by 1 day to expiry — never hold short options into expiry-day gamma."]})
    return plan


def build_index_plan(idx_name, meta, ind, regime, chain0, sentiment, c, event_risk):
    """[FIX-02] Returns DIRECTIONAL / RANGE / WAIT / STAND ASIDE with a trade ticket."""
    last = ind.iloc[-1]
    spot = chain0["spot"] if (chain0 and chain0.get("spot") == chain0.get("spot") and chain0.get("spot")) else _f(last["Close"])
    lg = score_index_setup(last, "long", regime["label"], sentiment)
    sh = score_index_setup(last, "short", regime["label"], sentiment)
    thr = c["min_score"]
    atr, e20, e50, adx = _f(last["ATR"]), _f(last["EMA_20"]), _f(last["EMA_50"]), _f(last["ADX"], 0)
    hv = _f(last["HV20"])
    vix_pct = regime.get("vix_pct")
    plan = {"idx": idx_name, "nse": meta["nse"], "lot": meta["lot"], "step": meta["step"], "spot": round(spot, 1),
            "long": lg, "short": sh, "thr": thr, "reasons": [], "event_risk": event_risk, "hv": hv,
            "vix": regime.get("vix"), "vix_pct": vix_pct, "conditional": False, "ts": datetime.now().strftime("%H:%M")}

    blocked = lambda res: any(k in f for f in res["flags"] for k in ("EXTENDED", "exhausted", "Stretched"))
    if lg["score"] >= thr and lg["score"] > sh["score"]:
        mode, dirn = "DIRECTIONAL", "CE"
    elif sh["score"] >= thr and sh["score"] > lg["score"]:
        mode, dirn = "DIRECTIONAL", "PE"
    elif e20 > e50 and spot > e50 and blocked(lg):
        mode, dirn = "WAIT", "CE"
    elif e20 < e50 and spot < e50 and blocked(sh):
        mode, dirn = "WAIT", "PE"
    elif (adx < 20 and _f(last["Low_20"]) < spot < _f(last["High_20"]) and not event_risk
          and (vix_pct is None or vix_pct >= 0.35)):
        mode, dirn = "RANGE", "IC"
    else:
        mode, dirn = "STAND ASIDE", None
        if event_risk and adx < 20:
            plan["reasons"].append("Range conditions, but you flagged a major event before expiry — no premium selling.")
        elif adx < 20 and vix_pct is not None and vix_pct < 0.35:
            plan["reasons"].append("Range conditions, but volatility is in the bottom third of its 1-year range — premium too cheap to sell.")
    plan["mode"] = mode

    exp_d, dte = pick_expiry(meta["nse"], chain0, min_dte=5 if mode in ("DIRECTIONAL", "WAIT") else 3)
    exp_str = exp_d.strftime("%d-%b-%Y")
    chain_e = None
    if chain0 and chain0.get("expiry") == exp_str:
        chain_e = chain0
    elif chain0:
        chain_e = fetch_option_chain(meta["nse"], True, expiry=exp_str)
    iv, iv_src = _index_iv(meta, chain_e, regime, hv)
    iv_rich = (vix_pct is not None and vix_pct >= 0.60) or (hv == hv and hv > 0 and iv / hv >= 1.20)
    plan.update({"expiry": exp_str, "dte": dte, "iv": round(iv, 2), "iv_src": iv_src, "iv_rich": iv_rich})

    if mode == "DIRECTIONAL":
        _directional_plan(plan, dirn, spot, last, meta, iv, iv_rich, chain_e, dte, c, event_risk)
    elif mode == "WAIT":
        trig = e20 + 0.5 * atr if dirn == "CE" else e20 - 0.5 * atr
        zone = (round(e20, 0), round(e20 + 1.0 * atr, 0)) if dirn == "CE" else (round(e20 - 1.0 * atr, 0), round(e20, 0))
        plan.update({"trigger": round(trig, 1), "zone": zone, "conditional": True,
                     "invalidation": f"Daily close {'below' if dirn == 'CE' else 'above'} the 50-EMA ({e50:,.0f}) cancels the {'bullish' if dirn == 'CE' else 'bearish'} bias."})
        plan["reasons"].append(f"Trend is {'up' if dirn == 'CE' else 'down'}, but the move is already stretched — "
                               f"entering now buys the top of the swing with the worst risk:reward.")
        _directional_plan(plan, dirn, trig, last, meta, iv, iv_rich, None, dte, c, event_risk)
    elif mode == "RANGE":
        _condor_plan(plan, spot, meta, iv, chain_e, dte)
    if plan["mode"] == "STAND ASIDE":
        hi_lvl, lo_lvl = max(e20, e50) + 0.25 * atr, min(e20, e50) - 0.25 * atr
        plan["triggers"] = (f"Bullish setup needs a daily close above {hi_lvl:,.0f} with MACD above signal; "
                            f"bearish setup needs a daily close below {lo_lvl:,.0f}. ADX (now {adx:.0f}) above 20 confirms either.")
    if plan.get("legs"):
        plan.update(finalize_plan(plan, plan["net"], c["cap_IN"], c["risk_pct"]))
    return plan


def render_index_plan(plan, c):
    """[FIX-03] Displays strategy decision, full execution ticket, and dual-direction scoring."""
    mode = plan["mode"]
    lg, sh = plan["long"], plan["short"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Spot", f"{plan['spot']:,.1f}")
    k2.metric("CALL score", f"{lg['score']} ({lg['grade']})")
    k3.metric("PUT score", f"{sh['score']} ({sh['grade']})")
    k4.metric("IV used", f"{plan['iv']:.1f}%", plan["iv_src"])
    k5.metric("Expiry / DTE", f"{plan['expiry']}", f"{plan['dte']} days")
    v1, v2, v3 = st.columns(3)
    v1.metric("HV20 (realised)", f"{plan['hv']:.1f}%" if plan["hv"] == plan["hv"] else "n/a")
    v2.metric("India VIX 1y percentile", f"{plan['vix_pct'] * 100:.0f}th" if plan["vix_pct"] is not None else "n/a")
    v3.metric("Premium pricing", "RICH — prefer spreads" if plan["iv_rich"] else "FAIR / CHEAP")
    st.markdown("---")

    if mode == "STAND ASIDE":
        st.warning("### ⚪ STAND ASIDE — no edge today")
        for r_ in plan["reasons"]:
            st.write(f"- {r_}")
        st.info(f"**What would change this:** {plan.get('triggers', '')}")
    elif mode == "WAIT":
        side = "CALL" if plan["dirn"] == "CE" else "PUT"
        st.warning(f"### 🟡 WAIT FOR PULLBACK — {side} bias, but don't chase")
        for r_ in plan["reasons"]:
            st.write(f"- {r_}")
        st.info(f"**Entry zone:** {plan['zone'][0]:,.0f} – {plan['zone'][1]:,.0f}  |  **Trigger level:** {plan['trigger']:,.0f}  |  "
                f"{plan['invalidation']}\n\nSet a price alert. The ticket below is pre-computed at the trigger level "
                f"(model prices) — re-evaluate when the alert fires.")
    elif mode == "RANGE":
        st.success(f"### 🔷 RANGE DAY — {plan['strategy']}")
    else:
        box = st.success if plan["dirn"] == "CE" else st.error
        box(f"### {'🟢' if plan['dirn'] == 'CE' else '🔴'} {plan['strategy']}")

    if plan.get("legs"):
        st.markdown(f"**Why this structure:** {plan['vehicle_reason']}")
        tdf = pd.DataFrame([{"Action": "BUY" if l["side"] == "B" else "SELL", "Strike": int(l["strike"]), "Type": l["type"],
                             "Expiry": plan["expiry"], "Price (₹)": l["price"], "Source": l["src"]} for l in plan["legs"]])
        show_df(tdf)
        net_lbl = "Net credit" if plan["side"] == "CREDIT" else "Net debit"
        t1, t2, t3, t4 = st.columns(4)
        t1.metric(net_lbl, f"₹{plan['net']:.1f}")
        t2.metric("Max loss / lot", f"₹{plan['max_loss_per_lot']:,}")
        t3.metric("Stop / target value", f"₹{plan['stop']} / ₹{plan['target']}")
        t4.metric("Breakeven(s)", " / ".join(f"{b:,.0f}" for b in plan["breakevens"]))
        u1, u2, u3, u4 = st.columns(4)
        if plan["side"] == "CREDIT":
            u1.metric("Invalidation", f"<{plan['und_stop']:,.0f} or >{plan['und_stop2']:,.0f}")
        else:
            u1.metric("Underlying stop", f"{plan['und_stop']:,.1f}")
        u2.metric("1σ expected move", f"±{plan['em']:,.0f}")
        u3.metric("Model prob. of profit", f"{plan['pop'] * 100:.0f}%" if plan.get("pop") is not None else "n/a")
        u4.metric("Reward : planned risk", f"{plan['rr']}")
        risk_txt = (f"**Size:** {plan['lots']} lot(s) of {plan['lot']} at your {c['risk_pct']}% risk limit "
                    f"(₹{c['cap_IN'] * c['risk_pct'] / 100:,.0f}). Worst case per lot ₹{plan['risk_per_lot']:,} "
                    f"({plan['one_lot_risk_pct']}% of capital); planned loss at stop ₹{plan['planned_loss_per_lot']:,}.")
        (st.error if plan["lots"] == 0 else st.info)(
            risk_txt + (" **Even one lot exceeds your risk limit — skip, or accept a higher risk % knowingly.**" if plan["lots"] == 0 else ""))
        st.markdown("**Exit plan (decide before you enter):**\n" + "\n".join(f"- {e}" for e in plan["exits"]))
        if any(l["src"] == "model" for l in plan["legs"]):
            st.caption("Model prices use one flat IV (no volatility skew; real OTM puts usually cost more). "
                       "Replace them with your broker's LTP below before adding the trade.")

    with st.expander("Why? Full scoring for both directions"):
        for lab, rr in (("CALL", lg), ("PUT", sh)):
            st.markdown(f"**{lab} — {rr['score']} ({rr['setup']})**: " + (" · ".join(rr["why"]) or "nothing aligned"))
            if rr["flags"]:
                st.markdown("Penalties: " + " · ".join(rr["flags"]))
        st.caption(f"Threshold for a directional trade: {plan['thr']} (sidebar).")


def evaluate_option_position(row, use_live=True):
    """[FIX-06] Exit engine for single options, debit spreads and iron condors."""
    und = str(row["Underlying"])
    is_index = und in UNDERLYING_YF
    df = get_history(UNDERLYING_YF.get(und, f"{und}.NS"), "2y")
    if df.empty or len(df) < 30:
        return None
    ind = add_indicators(df)
    last = ind.iloc[-1]
    spot, e20 = _f(last["Close"]), _f(last["EMA_20"])
    side = str(row.get("Side") or "DEBIT").upper()
    side = side if side in ("DEBIT", "CREDIT") else "DEBIT"
    typ = str(row.get("Option_Type") or "")
    legs = parse_legs(row.get("Legs"))
    if not legs:
        k = _f(row.get("Strike"))
        if k != k:
            for tok in reversed(str(row.get("Contract", "")).split()):
                if _f(tok) == _f(tok):
                    k = _f(tok)
                    break
        if k == k and typ in ("CE", "PE"):
            legs = [{"side": "B", "strike": k, "type": typ}]
    exp_s = str(row.get("Expiry") or "")
    exp = parse_nse_date(exp_s)
    dte = (exp - ist_today()).days if exp else None
    val, src = None, ""
    if use_live and legs and exp:
        ch = fetch_option_chain(und, is_index, expiry=exp.strftime("%d-%b-%Y"))
        if ch and parse_nse_date(ch["expiry"]) == exp:
            px = []
            for l in legs:
                hit = ch["chain"][ch["chain"]["strike"] == l["strike"]]
                v = _f(hit.iloc[0]["ce_ltp" if l["type"] == "CE" else "pe_ltp"]) if not hit.empty else np.nan
                px.append(v)
            if all(p == p and p > 0 for p in px):
                val, src = round(net_from_prices(legs, px, side), 2), "NSE live"
            if ch.get("spot") == ch.get("spot") and ch.get("spot"):
                spot = float(ch["spot"])
    if val is None:
        lp = _f(row.get("Last_Premium"))
        if lp == lp and lp > 0:
            val, src = lp, "manual"
    entry, tgt, slp = _f(row["Entry_Premium"]), _f(row["Target_Premium"]), _f(row["Stop_Loss_Premium"])
    und_sl, und_sl2 = _f(row.get("Underlying_SL")), _f(row.get("Underlying_SL2"))
    edt = pd.to_datetime(row.get("Entry_Date"), errors="coerce")
    held = (datetime.now() - edt.to_pydatetime()).days if pd.notna(edt) else 0
    strategy = str(row.get("Strategy") or "LONG OPTION")

    if side == "DEBIT":
        if val is not None and val <= slp:
            status = "🛑 EXIT NOW: premium stop hit"
        elif und_sl == und_sl and ((typ == "CE" and spot <= und_sl) or (typ == "PE" and spot >= und_sl)):
            status = "🛑 EXIT NOW: underlying stop breached"
        elif val is not None and val >= tgt:
            status = "🎯 TARGET: book profits (single option: 50% and trail the rest)"
        elif dte is not None and dte <= 1:
            status = "⌛ EXIT: 1 day or less to expiry"
        elif strategy == "LONG OPTION" and held >= 2 and val is not None and val < entry * 1.10:
            status = "⏳ TIME STOP: not up 10% after 2 sessions — theta is winning"
        elif (typ == "CE" and spot < e20) or (typ == "PE" and spot > e20):
            status = "⚠️ CAUTION: spot crossed the 20-EMA against you"
        else:
            status = "✅ HOLD: plan intact"
    else:
        if val is not None and val >= slp:
            status = "🛑 EXIT NOW: loss limit (condor costs 2× credit to close)"
        elif (und_sl == und_sl and spot <= und_sl) or (und_sl2 == und_sl2 and spot >= und_sl2):
            status = "🛑 EXIT / ADJUST: spot through a short strike"
        elif val is not None and val <= tgt:
            status = "🎯 TARGET: buy back — 50% of credit captured"
        elif dte is not None and dte <= 1:
            status = "⌛ EXIT: never hold short options into expiry day"
        else:
            status = "✅ HOLD: theta working, inside the range"
    if val is None:
        status += " (enter current value to enable premium stops)"
    units = _f(row.get("Lots"), 1) * _f(row.get("Lot_Size"), 0)
    pnl_u = None if val is None else ((val - entry) if side == "DEBIT" else (entry - val))
    return {"Row_ID": row["Row_ID"], "Contract": row["Contract"], "Side": side, "Expiry": exp_s, "DTE": dte,
            "Entry (₹)": entry, "Now (₹)": val, "Source": src,
            "P&L / unit": round(pnl_u, 2) if pnl_u is not None else None,
            "P&L (₹)": round(pnl_u * units, 0) if (pnl_u is not None and units) else None,
            "Target": tgt, "Stop": slp, "Spot": round(spot, 1),
            "Und. stop": und_sl if und_sl == und_sl else None, "Und. stop 2": und_sl2 if und_sl2 == und_sl2 else None,
            "Exit Signal": status}


# ==============================================================================
# SIDEBAR NAVIGATION & GLOBALLY SCOPED SETTINGS
# ==============================================================================
NAV = [
    "1. Full-Market Equity Scanner (Indian Market)",
    "2. Active Equity Portfolio & Exit Alerts (Indian Market)",
    "3. High-Probability Options Trading Desk (Indian Market)",
    "4. Active Options Exit Alert Engine (Indian Market)",
    "5. Market Regime, FII/DII & Sentiment",
    "6. US Equities Swing Trading Desk (US Market)",
    "7. Active US Swing Portfolio & Exit Monitor (US Market)",
    "8. Trade Journal & Performance Analytics",
]

st.sidebar.title("⚡ Global Quant Suite")
app_mode = st.sidebar.radio("Select Platform Section:", NAV, key="nav_platform_mode_selector")

with st.sidebar.expander("⚙️ Risk & Entry Settings", expanded=False):
    _min_score = st.slider("Minimum Confluence Score", 50, 90, 65, 5, key="cfg_min_score")
    _atr_mult = st.slider("Stop Distance (x ATR)", 1.2, 3.0, 1.8, 0.1, key="cfg_atr_mult")
    _risk = st.number_input("Risk Per Trade (% of Capital)", 0.25, 3.0, 1.0, 0.25, key="cfg_risk")
    _cap_in = st.number_input("India Trading Capital (₹)", 10000.0, 1e9, 500000.0, 10000.0, key="cfg_cap_in")
    _cap_us = st.number_input("US Trading Capital ($)", 1000.0, 1e8, 10000.0, 1000.0, key="cfg_cap_us")

st.session_state["_cfg"] = {
    "min_score": _min_score,
    "atr_mult": _atr_mult,
    "risk_pct": _risk,
    "cap_IN": _cap_in,
    "cap_US": _cap_us,
    "nse_live": True
}

# ##############################################################################
# SECTION 1: FULL-MARKET EQUITY SCANNER (INDIAN MARKET)
# ##############################################################################
if app_mode == NAV[0]:
    st.title("📈 Indian Equities Multi-Indicator Swing Scanner")
    st.caption("Screens liquid NSE equities in real-time for early-stage leading setups.")
    t_scan, t_single, t_sizer = st.tabs(["🚀 Market Scanner", "🔍 Stock Deep Dive", "🧮 Position Sizer"])
    with t_scan:
        render_scanner("IN")
    with t_single:
        render_deep_dive("IN")
    with t_sizer:
        render_sizer("IN")

# ##############################################################################
# SECTION 2: ACTIVE EQUITY PORTFOLIO & EXIT ALERTS (INDIAN MARKET)
# ##############################################################################
elif app_mode == NAV[1]:
    st.title("🛡️ Active Equity Portfolio & Dynamic Exit Monitor (NSE)")
    render_equity_monitor("IN")

# ##############################################################################
# SECTION 3: HIGH-PROBABILITY OPTIONS TRADING DESK (NSE) - SAFE FALLBACK
# ##############################################################################
elif app_mode == NAV[2]:
    st.title("🎯 High-Probability Options Trading Desk (NSE)")
    st.caption("Phase 6: Black-Scholes Greeks, IV vs. HV20 pricing, and directional spread intelligence.")
    c = cfg()
    opt_choice = st.radio("Choose Trading Segment:", ["Benchmark Indices (Nifty / Bank Nifty)", "Top F&O Momentum Stocks"],
                          horizontal=True, key="rb_options_segment_in")

    if opt_choice == "Benchmark Indices (Nifty / Bank Nifty)":
        st.subheader("Benchmark Index Strategy Desk")
        i1, i2 = st.columns([1, 2])
        idx_name = i1.selectbox("Select Index:", list(INDEX_DICT.keys()), key="sb_index_choice_in")
        event_risk = i2.toggle("Major scheduled event before expiry (RBI policy, Union Budget, US Fed, election results)",
                               value=False, key="tg_event_risk_in",
                               help="Forces spreads instead of outright buys and blocks premium selling.")
        meta = INDEX_DICT[idx_name]

        if st.button("Evaluate Index Setup", key="btn_eval_index_options_in"):
            with st.spinner(f"Evaluating {idx_name}: trend, volatility pricing and structure..."):
                df_idx = get_history(meta["yf"], "2y")
                if df_idx.empty or len(df_idx) < 210:
                    st.error("Insufficient market data available.")
                    st.session_state.idx_plan = None
                else:
                    diag = DataQualityManager.inspect_dataframe(df_idx, "IN")
                    if diag["is_stale"]:
                        st.error(f"Index data is stale (last bar {diag['last_bar']}). No recommendation on stale data.")
                        st.session_state.idx_plan = None
                    else:
                        ind = add_indicators(df_idx)
                        regime = get_market_regime("IN")
                        sent = fetch_live_news_sentiment(idx_name, market="IN")
                        chain0 = fetch_option_chain(meta["nse"], True)
                        chain0 = chain0 if isinstance(chain0, dict) else None
                        plan = build_index_plan(idx_name, meta, ind, regime, chain0, sent["score"], c, event_risk)
                        plan.update({"diag": diag, "chain_live": chain0 is not None, "regime": regime})
                        st.session_state.idx_plan = plan

        plan = st.session_state.get("idx_plan")
        if plan and plan["idx"] == idx_name:
            regime_banner(plan["regime"], "IN")
            DataQualityManager.render_badge(plan["diag"], "Index Yahoo Feed")
            if not plan["chain_live"]:
                st.caption("ℹ️ NSE live option chain unreachable from this server — premiums are Black-Scholes model values.")
            render_index_plan(plan, c)

            if plan.get("legs"):
                st.markdown("#### Add to Options Exit Engine (enter your actual fills)")
                pid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{plan['idx']}|{plan['expiry']}|{plan['mode']}|{legs_to_str(plan['legs'])}|{plan['net']}").hex[:12]
                cols = st.columns(len(plan["legs"]) + 1)
                prices = []
                for i, l in enumerate(plan["legs"]):
                    prices.append(cols[i].number_input(
                        f"{'Buy' if l['side'] == 'B' else 'Sell'} {int(l['strike'])} {l['type']} (₹)",
                        min_value=0.05, value=float(max(l["price"], 0.05)), step=0.5, key=f"ni_leg_{pid}_{i}"))
                net = round(net_from_prices(plan["legs"], prices, plan["side"]), 2)
                fin = finalize_plan(plan, net, c["cap_IN"], c["risk_pct"]) if net > 0 else None
                lots = cols[-1].number_input("Lots:", min_value=1, value=max(fin["lots"], 1) if fin else 1, step=1, key=f"ni_lots_{pid}")
                if fin is None:
                    st.error("Net premium must be positive for this structure — check the fills.")
                else:
                    total_risk = fin["risk_per_lot"] * lots
                    msg = (f"Net {'credit' if plan['side'] == 'CREDIT' else 'debit'} ₹{net:.1f} → stop value ₹{fin['stop']}, "
                           f"target value ₹{fin['target']}. Worst-case loss ₹{total_risk:,.0f} = {total_risk / c['cap_IN'] * 100:.2f}% "
                           f"of capital (limit {c['risk_pct']}%).")
                    (st.error if total_risk > c["cap_IN"] * c["risk_pct"] / 100 else st.caption)(msg)
                if plan["conditional"]:
                    st.warning("Conditional (WAIT) ticket — add it only after price reaches the entry zone and you hold real fills.")
                if st.button("Add to Options Exit Engine", key=f"btn_add_idx_{pid}", disabled=fin is None):
                    strikes = "/".join(str(int(l["strike"])) for l in plan["legs"])
                    exp_short = parse_nse_date(plan["expiry"]).strftime("%d%b%y").upper()
                    new_row = pd.DataFrame([{
                        "Row_ID": uuid.uuid4().hex[:8], "Contract": f"{plan['idx']} {exp_short} {plan['strategy']} {strikes}",
                        "Underlying": plan["nse"], "Option_Type": plan["dirn"], "Strike": plan["legs"][0]["strike"],
                        "Expiry": plan["expiry"], "Lots": lots, "Lot_Size": plan["lot"], "Entry_Premium": net,
                        "Target_Premium": fin["target"], "Stop_Loss_Premium": fin["stop"], "Underlying_SL": plan["und_stop"],
                        "Last_Premium": net, "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Strategy": plan["kind"], "Side": plan["side"], "Legs": legs_to_str(plan["legs"]),
                        "Underlying_SL2": plan["und_stop2"]}])
                    st.session_state.opt_portfolio = pd.concat([st.session_state.opt_portfolio, new_row], ignore_index=True)
                    save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
                    st.success(f"✅ Added {plan['strategy']} ({lots} lot(s)) to the Options Exit Engine.")

    else:
        st.subheader("Top F&O Momentum Stocks")
        st.caption("Long (CALL) candidates only — this screen ranks stocks; confirm strike, lot size and premium on your broker's chain.")
        st.caption("Phase 6: Multi-Indicator options momentum scanning on active F&O equities.")
        if st.button("Run F&O Options Momentum Scan", key="btn_fno_options_scan"):
            with st.spinner("Screening F&O watchlist for high-probability setups..."):
                fno_results, reg, _ = run_equity_scan(FNO_WATCHLIST[:15], "IN", c["min_score"], c["atr_mult"], check_earnings=True)
                if not fno_results.empty:
                    show_df(fno_results[["Symbol", "Score", "Grade", "Setup Type", "Price (₹)", "Rel Vol", "RSI", "Stop (₹)", "T1 2R (₹)"]])
                else:
                    st.info("No single-stock F&O options met the strict confluence threshold today.")

# ##############################################################################
# SECTION 4: ACTIVE OPTIONS EXIT ALERT ENGINE (INDIAN MARKET)
# ##############################################################################
elif app_mode == NAV[3]:
    st.title("🛡️ Real-Time Options Exit Alert Engine (NSE)")
    st.caption("[FIX-06] Premium stop/target (live NSE or your input), underlying invalidation, expiry and time stops — "
               "for single options, debit spreads and iron condors.")
    c = cfg()
    port_df = st.session_state.opt_portfolio

    with st.expander("➕ Manually log a single-option trade"):
        oc1, oc2, oc3, oc4 = st.columns(4)
        c_und = oc1.selectbox("Underlying:", ["NIFTY", "BANKNIFTY"] + FNO_WATCHLIST, key="sb_log_und_in")
        c_type = oc2.selectbox("Option Type:", ["CE", "PE"], key="sb_log_type_in")
        _hist = get_history(UNDERLYING_YF.get(c_und, f"{c_und}.NS"), "2y")
        _spot = _f(_hist["Close"].iloc[-1]) if not _hist.empty else 25000.0
        _step = {v["nse"]: v["step"] for v in INDEX_DICT.values()}.get(c_und) or stock_strike_step(_spot)
        c_strike = oc3.number_input("Strike:", value=int(round_to_strike(_spot, _step)), step=int(_step), key=f"ni_log_strike_{c_und}")
        c_exp = oc4.date_input("Expiry:", value=computed_expiries(c_und, n=1)[0], key=f"di_log_exp_{c_und}")
        oc5, oc6, oc7, oc8 = st.columns(4)
        c_prem = oc5.number_input("Entry premium (₹):", value=120.0, step=1.0, key="ni_log_prem_in")
        c_tgt = oc6.number_input("Target premium (₹):", value=round(c_prem * 1.5, 1), step=1.0, key=f"ni_log_tgt_{c_prem}")
        c_slp = oc7.number_input("Stop premium (₹):", value=round(c_prem * 0.6, 1), step=1.0, key=f"ni_log_slp_{c_prem}")
        c_usl = oc8.number_input("Underlying stop (₹):", value=float(round(_spot * (0.99 if c_type == "CE" else 1.01), 0)),
                                 step=10.0, key=f"ni_log_usl_{c_und}_{c_type}")
        oc9, oc10 = st.columns(2)
        default_lot = {v["nse"]: v["lot"] for v in INDEX_DICT.values()}.get(c_und, 0)
        c_lot = oc9.number_input("Lot size:", value=int(default_lot), min_value=0, step=1, key=f"ni_log_lot_{c_und}")
        c_lots = oc10.number_input("Lots:", value=1, min_value=1, step=1, key="ni_log_lots_in")
        if st.button("Start Tracking", key="btn_start_tracking_options_in"):
            new_trade = pd.DataFrame([{
                "Row_ID": uuid.uuid4().hex[:8], "Contract": f"{c_und} {c_exp.strftime('%d%b%y').upper()} {c_strike} {c_type}",
                "Underlying": c_und, "Option_Type": c_type, "Strike": c_strike, "Expiry": c_exp.strftime("%d-%b-%Y"),
                "Lots": c_lots, "Lot_Size": c_lot, "Entry_Premium": c_prem, "Target_Premium": c_tgt,
                "Stop_Loss_Premium": c_slp, "Underlying_SL": c_usl, "Last_Premium": c_prem,
                "Entry_Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "Strategy": "LONG OPTION", "Side": "DEBIT",
                "Legs": f"B:{int(c_strike)}:{c_type}", "Underlying_SL2": None}])
            st.session_state.opt_portfolio = pd.concat([st.session_state.opt_portfolio, new_trade], ignore_index=True)
            save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
            st.success("Tracking started.")
            st.rerun()

    if port_df.empty:
        st.info("No active options currently tracked.")
    else:
        st.markdown("#### Update current values (used when NSE live prices are unavailable)")
        st.caption("Spreads/condors: enter the NET value — debit = what the spread is worth now; credit = cost to buy it back now.")
        ed = st.data_editor(
            port_df[["Row_ID", "Contract", "Side", "Entry_Premium", "Last_Premium"]].astype({"Entry_Premium": float, "Last_Premium": float}),
            disabled=["Row_ID", "Contract", "Side", "Entry_Premium"], hide_index=True, key="de_opt_prem")
        if st.button("Save values", key="btn_save_opt_prem"):
            upd = dict(zip(ed["Row_ID"].astype(str), ed["Last_Premium"]))
            st.session_state.opt_portfolio["Last_Premium"] = st.session_state.opt_portfolio["Row_ID"].astype(str).map(upd)
            save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
            st.rerun()

        st.subheader("Live Exit Status Matrix")
        reports = []
        with st.spinner("Evaluating option positions..."):
            for _, row in port_df.iterrows():
                try:
                    r_ = evaluate_option_position(row, c.get("nse_live", True))
                    if r_:
                        reports.append(r_)
                except Exception:
                    continue
        live = pd.DataFrame(reports)
        if not live.empty:
            show_df(live.drop(columns=["Row_ID"]))

        st.markdown("---")
        st.markdown("#### Close a position")
        labels = {str(r_["Row_ID"]): f"{r_['Contract']} | entry ₹{r_['Entry_Premium']} | {r_['Row_ID']}" for _, r_ in port_df.iterrows()}
        rid = st.selectbox("Select trade:", list(labels.keys()), format_func=lambda x: labels[x], key="sb_close_opt")
        sel = port_df[port_df["Row_ID"].astype(str) == rid].iloc[0]
        now_val = _f(live.loc[live["Row_ID"].astype(str) == rid, "Now (₹)"].iloc[0]) if (not live.empty and (live["Row_ID"].astype(str) == rid).any()) else np.nan
        default_exit = now_val if now_val == now_val else _f(sel.get("Last_Premium"), _f(sel["Entry_Premium"]))
        x1, x2 = st.columns(2)
        exit_val = x1.number_input("Exit value (₹, net):", value=float(default_exit), key=f"ni_exit_opt_{rid}")
        reason = x2.selectbox("Reason:", ["Target", "Premium stop", "Underlying stop", "Expiry/time stop", "Discretionary"], key="sb_reason_opt")
        y1, y2 = st.columns(2)
        if y1.button("Close & log to Journal", key="btn_close_log_opt"):
            side = str(sel.get("Side") or "DEBIT").upper()
            units = _f(sel.get("Lots"), 1) * _f(sel.get("Lot_Size"), 0)
            e_, s_ = _f(sel["Entry_Premium"]), _f(sel["Stop_Loss_Premium"])
            init_r = (e_ - s_) if side == "DEBIT" else (s_ - e_)
            log_to_journal("IN-OPT", sel["Contract"], "LONG" if side == "DEBIT" else "SHORT", e_, exit_val, units,
                           sel["Entry_Date"], reason, str(sel.get("Strategy") or "Option"), init_risk=init_r if init_r == init_r else None)
            st.session_state.opt_portfolio = port_df[port_df["Row_ID"].astype(str) != rid].reset_index(drop=True)
            save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
            st.rerun()
        if y2.button("Remove without logging", key="btn_del_opt"):
            st.session_state.opt_portfolio = port_df[port_df["Row_ID"].astype(str) != rid].reset_index(drop=True)
            save_csv_data(st.session_state.opt_portfolio, OPTIONS_PORTFOLIO_FILE)
            st.rerun()

# ##############################################################################
# SECTION 5: MARKET REGIME, INSTITUTIONAL FII/DII & SENTIMENT
# ##############################################################################
elif app_mode == NAV[4]:
    st.title("🏛️ Market Regime, Institutional Activity & Sentiment")
    for mk in ("IN", "US"):
        rg = get_market_regime(mk)
        regime_banner(rg, mk)
    st.markdown("---")
    fii_data, src = fetch_fii_dii_live()
    if fii_data is not None:
        st.subheader("Institutional Cash Flow Activity")
        st.caption(f"Source: {src}")
        show_df(fii_data)

# ##############################################################################
# SECTION 6: US EQUITIES SWING TRADING DESK
# ##############################################################################
elif app_mode == NAV[5]:
    st.title("🗽 US Equities Multi-Indicator Swing Desk")
    us_tab_scan, us_tab_single, us_tab_sz = st.tabs(["🚀 US Market Scanner", "🔍 US Stock Deep Dive", "🧮 Position Sizer"])
    with us_tab_scan:
        render_scanner("US")
    with us_tab_single:
        render_deep_dive("US")
    with us_tab_sz:
        render_sizer("US")

# ##############################################################################
# SECTION 7: ACTIVE US SWING PORTFOLIO & EXIT MONITOR
# ##############################################################################
elif app_mode == NAV[6]:
    st.title("🛡️ Active US Swing Portfolio & Exit Monitor (Wall Street)")
    render_equity_monitor("US")

# ##############################################################################
# SECTION 8: TRADE JOURNAL & PERFORMANCE ANALYTICS (PHASE 7 UPGRADED)
# ##############################################################################
elif app_mode == NAV[7]:
    st.title("📓 Trade Journal & Quantitative Performance Analytics")
    j = st.session_state.journal.copy()
    if j.empty:
        st.info("No closed trades logged yet. Settle an open trade from Desk 2 or Desk 7 to inspect analytics.")
    else:
        # [FIX-09] Scorecard per market — rupee and dollar P&L were previously added together.
        j["PnL"] = pd.to_numeric(j["PnL"], errors="coerce").fillna(0.0)
        j["R_Multiple"] = pd.to_numeric(j["R_Multiple"], errors="coerce")
        st.subheader("Performance Scorecard (per market)")
        for mk, g in j.groupby(j["Market"].fillna("NA").astype(str)):
            ccy = "$" if mk.startswith("US") else "₹"
            pn = g["PnL"]
            gw, gl = pn[pn > 0].sum(), abs(pn[pn < 0].sum())
            pf = gw / gl if gl > 0 else None
            cols = st.columns(5)
            cols[0].metric(f"{mk} — closed trades", len(g))
            cols[1].metric("Win rate", f"{(pn > 0).mean() * 100:.1f}%")
            cols[2].metric("Net realised P&L", f"{ccy}{pn.sum():,.2f}")
            cols[3].metric("Profit factor", f"{pf:.2f}" if pf else "n/a")
            cols[4].metric("Avg R-multiple", f"{g['R_Multiple'].mean():.2f}" if g["R_Multiple"].notna().any() else "n/a")

        st.markdown("---")
        st.subheader("Historical Trade Ledger")
        show_df(j)
        
        st.download_button(
            "📥 Export Full Journal (CSV)",
            data=j.to_csv(index=False).encode("utf-8"),
            file_name=f"Trade_Journal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="btn_dl_full_journal_csv"
        )
