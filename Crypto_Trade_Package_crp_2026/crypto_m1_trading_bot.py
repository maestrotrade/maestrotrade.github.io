#!/usr/bin/env python3
"""
NDAX CRYPTO TRUE M1 SCALPER - V4.0
========================================
Adapted from the user's MT4 V32 currency EA, optimized for NDAX spot crypto.

Production V3 adds:
- Correct NDAX completed-candle handling (NDAX/CCXT candle timestamp behaves as period-end)
- 60-second heartbeat so the console never appears "stuck"
- Optional authenticated NDAX account connection
- Live market BUY
- Exchange-native STOP-MARKET protective SELL
- Local fee-aware 1.5:1 take-profit exit while the native stop remains on exchange
- M1 entry + M5 trend filter, cooldown, fixed CAD risk and max CAD capital per trade
- Persistent state and recovery after restart
- Paper/live modes controlled from .env

SAFETY DESIGN:
- LIVE_TRADING=false by default.
- To place real orders BOTH must be set:
    LIVE_TRADING=true
    LIVE_ARM=YES_REAL_ORDERS
- Withdrawals are never used.
- Spot-long only. No shorts, no leverage.
- After a live BUY fills, the bot immediately attempts to place an exchange-native
  stop-market SELL. If that protective stop cannot be created, the bot attempts
  an emergency market SELL of the newly bought asset.
"""

import os
import json
import time
import math
import traceback
from pathlib import Path
from datetime import datetime, timezone

import ccxt
import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============================================================
# CONFIG
# ============================================================

BOT_VERSION = "NDAX_CRYPTO_M1_SHARED_V4_2"

SYMBOLS = [
    s.strip()
    for s in os.getenv("NDAX_SYMBOLS", "BTC/CAD").split(",")
    if s.strip()
]

TIMEFRAME = "1m"
TIMEFRAME_MS = 60 * 1000
OHLCV_LIMIT = 240

TREND_TIMEFRAME = "5m"
TREND_OHLCV_LIMIT = 140

POLL_SECONDS = int(os.getenv("M1_POLL_SECONDS", "5"))
HEARTBEAT_SECONDS = int(os.getenv("M1_HEARTBEAT_SECONDS", "60"))
CANDLE_CLOSE_GRACE_SECONDS = int(os.getenv("M1_CANDLE_CLOSE_GRACE_SECONDS", "2"))

LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"
LIVE_ARM = os.getenv("LIVE_ARM", "").strip()
LIVE_ALLOWED = LIVE_TRADING and LIVE_ARM == "YES_REAL_ORDERS"

# Risk and allocation
RISK_CAD = float(os.getenv("M1_RISK_CAD", "5.00"))
RR_RATIO = float(os.getenv("M1_RR_RATIO", "1.5"))
PAPER_CAD_BALANCE = float(os.getenv("M1_PAPER_CAD_BALANCE", "500.00"))
MAX_POSITION_CAD = float(os.getenv("M1_MAX_POSITION_CAD", "150.00"))
MAX_OPEN_POSITIONS = int(os.getenv("M1_MAX_OPEN_POSITIONS", "1"))

# Production account-level safeguards
MIN_CAD_RESERVE = float(os.getenv("M1_MIN_CAD_RESERVE", "25.00"))
MAX_TRADES_PER_UTC_DAY = int(os.getenv("M1_MAX_TRADES_PER_UTC_DAY", "4"))
MAX_DAILY_LOSS_CAD = float(os.getenv("M1_MAX_DAILY_LOSS_CAD", "15.00"))
NDAX_ACCOUNT_ID_ENV = os.getenv("NDAX_ACCOUNT_ID", "").strip()


# Crypto-adapted V32 filters
MIN_ADX = float(os.getenv("M1_MIN_ADX", "18.0"))
BUY_RSI_MIN = float(os.getenv("M1_BUY_RSI_MIN", "53.0"))
BUY_RSI_MAX = float(os.getenv("M1_BUY_RSI_MAX", "68.0"))
SELL_RSI_MIN = float(os.getenv("M1_SELL_RSI_MIN", "32.0"))
SELL_RSI_MAX = float(os.getenv("M1_SELL_RSI_MAX", "47.0"))

MIN_EMA_SEPARATION_ATR = float(os.getenv("M1_MIN_EMA_SEPARATION_ATR", "0.05"))
MIN_BAND_WIDTH_ATR = float(os.getenv("M1_MIN_BAND_WIDTH_ATR", "1.00"))
VOL_RATIO_THRESH = float(os.getenv("M1_VOL_RATIO_THRESH", "0.00"))
MAX_SIGNAL_BODY_ATR = float(os.getenv("M1_MAX_SIGNAL_BODY_ATR", "1.50"))

ATR_STOP_MULTIPLIER = float(os.getenv("M1_ATR_STOP_MULTIPLIER", "1.25"))
MIN_STOP_PCT = float(os.getenv("M1_MIN_STOP_PCT", "0.18")) / 100.0
MAX_STOP_ATR = float(os.getenv("M1_MAX_STOP_ATR", "5.0"))

MAX_SPREAD_BPS_DEFAULT = float(
    os.getenv("M1_MAX_SPREAD_BPS_DEFAULT", "50.0")
)

# Pair-specific spread ceilings in basis points.
PAIR_MAX_SPREAD_BPS = {
    "BTC/CAD": float(os.getenv("M1_MAX_SPREAD_BPS_BTC_CAD", "45.0")),
    "ETH/CAD": float(os.getenv("M1_MAX_SPREAD_BPS_ETH_CAD", "55.0")),
    "SOL/CAD": float(os.getenv("M1_MAX_SPREAD_BPS_SOL_CAD", "85.0")),
    "LINK/CAD": float(os.getenv("M1_MAX_SPREAD_BPS_LINK_CAD", "80.0")),
}

def max_spread_bps_for(symbol):
    return PAIR_MAX_SPREAD_BPS.get(symbol, MAX_SPREAD_BPS_DEFAULT)

# Estimated taker fee used for sizing only.
ESTIMATED_TAKER_FEE_PCT = float(
    os.getenv("M1_ESTIMATED_TAKER_FEE_PCT", "0.20")
) / 100.0

# Leave a small base-asset buffer so a fee deducted in base currency
# does not cause the protective SELL quantity to exceed the asset received.
PROTECTIVE_SELL_BUFFER_PCT = float(
    os.getenv("M1_PROTECTIVE_SELL_BUFFER_PCT", "0.40")
) / 100.0

COOLDOWN_BARS_AFTER_CLOSE = int(
    os.getenv("M1_COOLDOWN_BARS_AFTER_CLOSE", "2")
)

# True M1 scalper regime filter: M1 entries only trade with M5 trend.
USE_M5_TREND_FILTER = os.getenv("M1_USE_M5_TREND_FILTER", "true").lower() == "true"
REQUIRE_M5_EMA_SLOPE = os.getenv("M1_REQUIRE_M5_EMA_SLOPE", "true").lower() == "true"
M5_MIN_EMA_SEPARATION_ATR = float(
    os.getenv("M1_M5_MIN_EMA_SEPARATION_ATR", "0.05")
)

# Skip a setup if fee-aware TP becomes too far away to still qualify as a scalp.
MAX_TARGET_MOVE_PCT = float(
    os.getenv("M1_MAX_TARGET_MOVE_PCT", "1.75")
) / 100.0


STATE_FILE = Path(os.getenv("M1_STATE_FILE", "ndax_crypto_m1_shared_state.json"))
LOG_FILE = Path(os.getenv("M1_LOG_FILE", r"C:\NDAX_BOT\m1_log.txt"))

# Private API credentials.
NDAX_API_KEY = os.getenv("NDAX_API_KEY", "").strip()
NDAX_API_SECRET = os.getenv("NDAX_API_SECRET", "").strip()
NDAX_USER_ID = os.getenv("NDAX_USER_ID", "").strip()

# Optional login/password are NOT required for normal API-key signing in this bot.
# They are intentionally not used.
# ============================================================
# LOGGING
# ============================================================

def log(message: str):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"{stamp} | {message}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ============================================================
# SHARED M1/M5 POSITION LOCK
# ============================================================

SHARED_POSITION_LOCK_FILE = Path(
    os.getenv(
        "SHARED_POSITION_LOCK_FILE",
        r"C:\NDAX_BOT\ndax_shared_position.lock"
    )
)

SHARED_TRADE_LOG_FILE = Path(
    os.getenv(
        "SHARED_TRADE_LOG_FILE",
        r"C:\NDAX_BOT\log.txt"
    )
)


def write_shared_trade_line(symbol, pnl_cad, mode):
    """
    Write exactly ONE compact line per CLOSED trade to the shared log.
    Verbose diagnostics continue to the bot-specific M1/M5 log files.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = (
        f"{stamp} | {'M1'} | "
        f"{symbol} | {mode} | Profit=C${float(pnl_cad):+.2f}"
    )

    try:
        SHARED_TRADE_LOG_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with SHARED_TRADE_LOG_FILE.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(line + "\\n")
    except Exception as exc:
        log(
            f"SHARED TRADE LOG ERROR | "
            f"{type(exc).__name__}: {exc}"
        )

BOT_LOCK_NAME = BOT_VERSION


def read_shared_lock():
    if not SHARED_POSITION_LOCK_FILE.exists():
        return None

    try:
        return json.loads(
            SHARED_POSITION_LOCK_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        return {
            "bot": "UNKNOWN",
            "symbol": "UNKNOWN",
            "warning": "Unreadable lock file",
        }


def acquire_shared_lock(symbol):
    """
    Atomically create a cross-bot lock.
    If either M1 or M5 already owns the lock, no second bot position can open.
    """
    payload = {
        "bot": BOT_LOCK_NAME,
        "symbol": symbol,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    SHARED_POSITION_LOCK_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with SHARED_POSITION_LOCK_FILE.open(
            "x",
            encoding="utf-8",
        ) as f:
            json.dump(payload, f, indent=2)

        log(
            f"SHARED LOCK ACQUIRED | Bot={BOT_LOCK_NAME} | "
            f"Symbol={symbol}"
        )
        return True

    except FileExistsError:
        holder = read_shared_lock() or {}
        log(
            f"SHARED LOCK BLOCK | ExistingBot={holder.get('bot')} | "
            f"Symbol={holder.get('symbol')}"
        )
        return False

    except Exception as exc:
        log(
            f"SHARED LOCK ERROR | Could not acquire | "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def release_shared_lock(symbol=None):
    if not SHARED_POSITION_LOCK_FILE.exists():
        return True

    holder = read_shared_lock() or {}

    # Never remove another bot's lock.
    if holder.get("bot") != BOT_LOCK_NAME:
        return False

    if symbol and holder.get("symbol") not in (None, symbol):
        return False

    try:
        SHARED_POSITION_LOCK_FILE.unlink()
        log(
            f"SHARED LOCK RELEASED | Bot={BOT_LOCK_NAME} | "
            f"Symbol={holder.get('symbol')}"
        )
        return True
    except Exception as exc:
        log(
            f"SHARED LOCK ERROR | Could not release | "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def reconcile_shared_lock(state):
    """
    Safe restart behavior:
    - If this bot has a remembered open position, make sure it owns the lock.
    - If this bot owns the lock but has no remembered position, remove its stale lock.
    - Never clear the other bot's lock automatically.
    """
    positions = state.get("positions", {})
    holder = read_shared_lock()

    if positions:
        symbol = next(iter(positions.keys()))

        if holder is None:
            acquire_shared_lock(symbol)
            return

        if holder.get("bot") == BOT_LOCK_NAME:
            return

        log(
            f"SHARED LOCK WARNING | This bot remembers open {symbol}, "
            f"but lock belongs to {holder.get('bot')} / "
            f"{holder.get('symbol')}. No new trades will open."
        )
        return

    if holder and holder.get("bot") == BOT_LOCK_NAME:
        release_shared_lock(holder.get("symbol"))


# ============================================================
# STATE
# ============================================================

def default_state():
    return {
        "version": BOT_VERSION,
        "positions": {},
        "last_processed_bar": {},
        "last_closed_bar_ms": {},
        "paper_cash_cad": PAPER_CAD_BALANCE,
        "last_heartbeat": 0.0,
        "account_id": None,
        "daily": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "trades": 0,
            "realized_pnl_cad": 0.0,
            "wins": 0,
            "losses": 0,
        },
        "stats": {
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "realized_pnl_cad": 0.0,
        },
    }


def load_state():
    if not STATE_FILE.exists():
        return default_state()

    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        base = default_state()
        base.update(data)
        base.setdefault("positions", {})
        base.setdefault("last_processed_bar", {})
        base.setdefault("last_closed_bar_ms", {})
        base.setdefault("paper_cash_cad", PAPER_CAD_BALANCE)
        base.setdefault("last_heartbeat", 0.0)
        base.setdefault("account_id", None)
        base.setdefault("daily", {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "trades": 0,
            "realized_pnl_cad": 0.0,
            "wins": 0,
            "losses": 0,
        })
        base["daily"].setdefault("wins", 0)
        base["daily"].setdefault("losses", 0)
        base.setdefault("stats", {
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "realized_pnl_cad": 0.0,
        })
        return base
    except Exception as exc:
        log(f"STATE WARNING | Could not read state: {exc}. Starting fresh.")
        return default_state()


def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


# ============================================================
# EXCHANGE
# ============================================================

def build_exchange():
    cfg = {
        "enableRateLimit": True,
        "timeout": 20000,
    }

    if NDAX_API_KEY:
        cfg["apiKey"] = NDAX_API_KEY
    if NDAX_API_SECRET:
        cfg["secret"] = NDAX_API_SECRET
    if NDAX_USER_ID:
        cfg["uid"] = NDAX_USER_ID

    exchange = ccxt.ndax(cfg)

    # CCXT's NDAX class marks login/password as required globally because those
    # fields are needed for signIn()/withdraw, but normal private API-key signing
    # uses apiKey + secret + uid. We never use withdraw/signIn here.
    try:
        exchange.requiredCredentials["login"] = False
        exchange.requiredCredentials["password"] = False
    except Exception:
        pass

    exchange.load_markets()
    return exchange


def private_credentials_present():
    return bool(NDAX_API_KEY and NDAX_API_SECRET and NDAX_USER_ID)


def _normalize_account_ids(response):
    ids = []
    if isinstance(response, list):
        for item in response:
            if isinstance(item, dict):
                value = item.get("AccountId") or item.get("accountId") or item.get("id")
            else:
                value = item
            try:
                if value is not None:
                    ids.append(int(value))
            except Exception:
                pass
    elif isinstance(response, dict):
        for key in ("AccountIds", "accountIds", "accounts", "Accounts"):
            if key in response:
                return _normalize_account_ids(response[key])
    return ids


def discover_account_id(exchange, state):
    """
    Uses NDAX's signed GetUserAccounts endpoint directly.
    This avoids CCXT fetch_accounts(), which unnecessarily requires
    the user's NDAX email address.
    """
    if NDAX_ACCOUNT_ID_ENV:
        account_id = int(NDAX_ACCOUNT_ID_ENV)
    elif state.get("account_id"):
        account_id = int(state["account_id"])
    else:
        response = exchange.privateGetGetUserAccounts({
            "omsId": 1,
            "UserId": NDAX_USER_ID,
            "UserName": "",
        })
        ids = _normalize_account_ids(response)
        if not ids:
            raise RuntimeError(
                "NDAX GetUserAccounts returned no account IDs: " + str(response)
            )
        account_id = ids[0]

    state["account_id"] = account_id
    save_state(state)

    # Pre-populate CCXT account cache so create_order/fetch_order/cancel_order
    # do not call fetch_accounts() and ask for an email login.
    exchange.accounts = [{
        "id": str(account_id),
        "type": None,
        "currency": None,
        "info": str(account_id),
    }]
    exchange.options["accountId"] = int(account_id)

    return int(account_id)


def fetch_private_balances(exchange, account_id):
    response = exchange.privateGetGetAccountPositions({
        "omsId": 1,
        "AccountId": int(account_id),
    })

    balances = {}
    for item in response or []:
        symbol = item.get("ProductSymbol") or item.get("productSymbol")

        if not symbol:
            product_id = item.get("ProductId") or item.get("productId")
            # Look up ProductId in CCXT's loaded currencies when possible.
            try:
                currency = exchange.safe_currency(str(product_id))
                if currency:
                    symbol = currency.get("code")
            except Exception:
                symbol = None

        if not symbol:
            continue

        amount = float(item.get("Amount") or item.get("amount") or 0.0)
        hold = float(item.get("Hold") or item.get("hold") or 0.0)

        balances[str(symbol).upper()] = {
            "total": amount,
            "hold": hold,
            "free": max(0.0, amount - hold),
        }

    return balances


def test_private_connection(exchange, state):
    if not private_credentials_present():
        log(
            "PRIVATE API | Not configured. "
            "Need NDAX_API_KEY + NDAX_API_SECRET + NDAX_USER_ID."
        )
        return False, None

    try:
        account_id = discover_account_id(exchange, state)
        balances = fetch_private_balances(exchange, account_id)

        cad = balances.get("CAD", {}).get("free", 0.0)
        btc = balances.get("BTC", {}).get("free", 0.0)

        log(
            f"PRIVATE API | AUTHENTICATED | AccountId={account_id} | "
            f"FreeCAD=C${cad:.2f} | FreeBTC={btc:.10f}"
        )
        return True, account_id
    except Exception as exc:
        log(
            f"PRIVATE API | AUTH TEST FAILED | "
            f"{type(exc).__name__}: {exc}"
        )
        return False, None


def roll_daily_state(state):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily = state.get("daily", {})
    if daily.get("date") != today:
        state["daily"] = {
            "date": today,
            "trades": 0,
            "realized_pnl_cad": 0.0,
            "wins": 0,
            "losses": 0,
        }
        save_state(state)


def daily_safety_pass(state):
    roll_daily_state(state)
    daily = state["daily"]

    if int(daily.get("trades", 0)) >= MAX_TRADES_PER_UTC_DAY:
        log(
            f"DAILY SAFETY BLOCK | Trades={daily.get('trades')} "
            f">= {MAX_TRADES_PER_UTC_DAY}"
        )
        return False

    pnl = float(daily.get("realized_pnl_cad", 0.0))
    if pnl <= -abs(MAX_DAILY_LOSS_CAD):
        log(
            f"DAILY SAFETY BLOCK | RealizedPnL=C${pnl:.2f} "
            f"<= -C${MAX_DAILY_LOSS_CAD:.2f}"
        )
        return False

    return True


# ============================================================
# INDICATORS

# ============================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def true_range(df):
    prev_close = df["close"].shift(1)
    parts = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def atr(df, period=14):
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx_components(df, period=14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    tr_smoothed = true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    plus_smoothed = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_smoothed = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_smoothed / tr_smoothed.replace(0, np.nan)
    minus_di = 100 * minus_smoothed / tr_smoothed.replace(0, np.nan)

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx_line = dx.ewm(alpha=1 / period, adjust=False).mean()

    return (
        adx_line.fillna(0.0),
        plus_di.fillna(0.0),
        minus_di.fillna(0.0),
    )


def add_indicators(df):
    df = df.copy()
    df["ema9"] = ema(df["close"], 9)
    df["ema20"] = ema(df["close"], 20)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)

    adx_line, plus_di, minus_di = adx_components(df, 14)
    df["adx14"] = adx_line
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std(ddof=0)
    df["bb_upper"] = mid + 2.0 * std
    df["bb_lower"] = mid - 2.0 * std
    df["avg_volume20"] = df["volume"].rolling(20).mean()
    return df


# ============================================================
# NDAX CANDLE HANDLING
# ============================================================

def fetch_completed_tf(exchange, symbol, timeframe, limit, min_bars=60):
    raw = exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        limit=limit,
    )

    if not raw or len(raw) < min_bars:
        raise RuntimeError(
            f"{symbol}: insufficient {timeframe} OHLCV bars "
            f"({0 if not raw else len(raw)})"
        )

    df = pd.DataFrame(
        raw,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    # NDAX candle timestamp behaves as the period-end timestamp.
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - CANDLE_CLOSE_GRACE_SECONDS * 1000
    completed = df[df["timestamp"] <= cutoff_ms].copy()

    if len(completed) < min_bars:
        raise RuntimeError(
            f"{symbol}: only {len(completed)} confirmed-completed "
            f"{timeframe} bars"
        )

    completed = add_indicators(completed).reset_index(drop=True)
    newest_raw = int(df.iloc[-1]["timestamp"])
    newest_completed = int(completed.iloc[-1]["timestamp"])

    return completed, newest_raw, newest_completed


def fetch_completed_m1(exchange, symbol):
    return fetch_completed_tf(
        exchange,
        symbol,
        TIMEFRAME,
        OHLCV_LIMIT,
        min_bars=80,
    )


def fetch_completed_m5_trend(exchange, symbol):
    return fetch_completed_tf(
        exchange,
        symbol,
        TREND_TIMEFRAME,
        TREND_OHLCV_LIMIT,
        min_bars=60,
    )

def fetch_spread_bps(exchange, symbol):
    ticker = exchange.fetch_ticker(symbol)
    bid = ticker.get("bid")
    ask = ticker.get("ask")

    if bid is None or ask is None:
        return None, bid, ask, ticker

    bid = float(bid)
    ask = float(ask)

    if bid <= 0 or ask <= bid:
        return None, bid, ask, ticker

    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 10000.0, bid, ask, ticker


# ============================================================
# SIGNAL
# ============================================================

def evaluate_signal(m1_df, m5_df):
    # M1 completed[-1] is the most recently confirmed M1 bar.
    c1 = m1_df.iloc[-1]
    c2 = m1_df.iloc[-2]

    # M5 trend confirmation.
    t1 = m5_df.iloc[-1]
    t2 = m5_df.iloc[-2]

    atr_value = float(c1["atr14"])
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None, {"reason": "ATR invalid"}

    ema_sep_atr = abs(float(c1["ema9"]) - float(c1["ema20"])) / atr_value
    band_width_atr = (
        float(c1["bb_upper"]) - float(c1["bb_lower"])
    ) / atr_value

    avg_vol = float(c1["avg_volume20"])
    signal_vol = float(c1["volume"])
    vol_ratio = signal_vol / avg_vol if avg_vol > 0 else 0.0

    body = abs(float(c1["close"]) - float(c1["open"]))
    body_atr = body / atr_value

    bullish_continuation = (
        float(c1["close"]) > float(c1["open"])
        and float(c1["close"]) > float(c2["close"])
    )
    bearish_continuation = (
        float(c1["close"]) < float(c1["open"])
        and float(c1["close"]) < float(c2["close"])
    )

    adx_value = float(c1["adx14"])
    plus_di = float(c1["plus_di"])
    minus_di = float(c1["minus_di"])
    rsi_value = float(c1["rsi14"])

    m5_atr = float(t1["atr14"])
    if not math.isfinite(m5_atr) or m5_atr <= 0:
        return None, {"reason": "M5 ATR invalid"}

    m5_sep_atr = (
        abs(float(t1["ema9"]) - float(t1["ema20"])) / m5_atr
    )

    m5_up = float(t1["ema9"]) > float(t1["ema20"])
    m5_down = float(t1["ema9"]) < float(t1["ema20"])

    if REQUIRE_M5_EMA_SLOPE:
        m5_up = m5_up and float(t1["ema9"]) > float(t2["ema9"])
        m5_down = m5_down and float(t1["ema9"]) < float(t2["ema9"])

    diag = {
        "bar_ms": int(c1["timestamp"]),
        "close": float(c1["close"]),
        "ema9": float(c1["ema9"]),
        "ema20": float(c1["ema20"]),
        "ema_sep_atr": ema_sep_atr,
        "rsi": rsi_value,
        "atr": atr_value,
        "adx": adx_value,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "bb_width_atr": band_width_atr,
        "volume_ratio": vol_ratio,
        "body_atr": body_atr,
        "m5_up": m5_up,
        "m5_down": m5_down,
        "m5_sep_atr": m5_sep_atr,
    }

    if adx_value < MIN_ADX:
        diag["reason"] = f"M1 ADX {adx_value:.1f} < {MIN_ADX:.1f}"
        return None, diag

    if ema_sep_atr < MIN_EMA_SEPARATION_ATR:
        diag["reason"] = (
            f"M1 EMA/ATR {ema_sep_atr:.3f} < "
            f"{MIN_EMA_SEPARATION_ATR:.3f}"
        )
        return None, diag

    if band_width_atr < MIN_BAND_WIDTH_ATR:
        diag["reason"] = (
            f"M1 BB/ATR {band_width_atr:.2f} < "
            f"{MIN_BAND_WIDTH_ATR:.2f}"
        )
        return None, diag

    if vol_ratio < VOL_RATIO_THRESH:
        diag["reason"] = (
            f"M1 volume ratio {vol_ratio:.2f} < {VOL_RATIO_THRESH:.2f}"
        )
        return None, diag

    if body_atr > MAX_SIGNAL_BODY_ATR:
        diag["reason"] = (
            f"M1 signal body {body_atr:.2f} ATR > "
            f"{MAX_SIGNAL_BODY_ATR:.2f}"
        )
        return None, diag

    if USE_M5_TREND_FILTER and m5_sep_atr < M5_MIN_EMA_SEPARATION_ATR:
        diag["reason"] = (
            f"M5 EMA/ATR {m5_sep_atr:.3f} < "
            f"{M5_MIN_EMA_SEPARATION_ATR:.3f}"
        )
        return None, diag

    buy_ok = (
        float(c1["ema9"]) > float(c1["ema20"])
        and BUY_RSI_MIN <= rsi_value <= BUY_RSI_MAX
        and plus_di > minus_di
        and bullish_continuation
        and (not USE_M5_TREND_FILTER or m5_up)
    )

    sell_ok = (
        float(c1["ema9"]) < float(c1["ema20"])
        and SELL_RSI_MIN <= rsi_value <= SELL_RSI_MAX
        and minus_di > plus_di
        and bearish_continuation
        and (not USE_M5_TREND_FILTER or m5_down)
    )

    if buy_ok:
        diag["reason"] = "M1 BUY continuation + M5 trend confirmed"
        return "buy", diag

    if sell_ok:
        diag["reason"] = "M1 BEARISH continuation + M5 trend confirmed"
        return "sell_signal", diag

    diag["reason"] = "M1/M5 momentum criteria not met"
    return None, diag


# ============================================================
# RISK / PLAN
# ============================================================

def calculate_long_plan(exchange, symbol, entry, atr_value, cash_cap):
    raw_stop_distance = atr_value * ATR_STOP_MULTIPLIER
    minimum_stop_distance = entry * MIN_STOP_PCT
    stop_distance = max(raw_stop_distance, minimum_stop_distance)

    if stop_distance > atr_value * MAX_STOP_ATR:
        return None, "Stop distance exceeds ATR safety cap"

    stop_price = entry - stop_distance
    if stop_price <= 0:
        return None, "Invalid stop price"

    fee = ESTIMATED_TAKER_FEE_PCT

    risk_per_unit = (
        (entry - stop_price)
        + (entry * fee)
        + (stop_price * fee)
    )

    if risk_per_unit <= 0:
        return None, "Risk per unit invalid"

    risk_amount = RISK_CAD / risk_per_unit
    max_notional = min(MAX_POSITION_CAD, cash_cap)
    allocation_amount = max_notional / entry
    amount = min(risk_amount, allocation_amount)

    if amount <= 0:
        return None, "Amount <= 0"

    try:
        amount = float(exchange.amount_to_precision(symbol, amount))
        stop_price = float(exchange.price_to_precision(symbol, stop_price))
    except Exception:
        pass

    if amount <= 0:
        return None, "Rounded amount <= 0"

    market = exchange.market(symbol)
    min_amount = (market.get("limits", {}).get("amount", {}) or {}).get("min")
    min_cost = (market.get("limits", {}).get("cost", {}) or {}).get("min")

    notional = amount * entry

    if min_amount is not None and amount < float(min_amount):
        return None, f"Amount {amount} below exchange minimum {min_amount}"

    if min_cost is not None and notional < float(min_cost):
        return None, f"Notional C${notional:.2f} below exchange minimum C${float(min_cost):.2f}"

    actual_risk = amount * risk_per_unit

    # Fee-aware target for desired net reward/risk.
    target_net_per_unit = RR_RATIO * risk_per_unit
    target = (
        target_net_per_unit + entry * (1.0 + fee)
    ) / (1.0 - fee)

    try:
        target = float(exchange.price_to_precision(symbol, target))
    except Exception:
        pass

    target_move_pct = (target - entry) / entry
    if target_move_pct > MAX_TARGET_MOVE_PCT:
        return None, (
            f"Fee-aware target move {target_move_pct*100:.2f}% "
            f"> scalp cap {MAX_TARGET_MOVE_PCT*100:.2f}%"
        )

    protective_amount = amount * (1.0 - PROTECTIVE_SELL_BUFFER_PCT)
    try:
        protective_amount = float(
            exchange.amount_to_precision(symbol, protective_amount)
        )
    except Exception:
        pass

    if protective_amount <= 0:
        return None, "Protective sell amount rounded to zero"

    return {
        "symbol": symbol,
        "side": "long",
        "planned_entry": entry,
        "stop": stop_price,
        "target": target,
        "amount": amount,
        "protective_amount": protective_amount,
        "notional_cad": notional,
        "risk_cad": actual_risk,
        "risk_cap_cad": RISK_CAD,
        "rr": RR_RATIO,
        "fee_rate": fee,
    }, None


# ============================================================
# PAPER TRADING
# ============================================================

def count_positions(state):
    return len(state.get("positions", {}))


def open_paper(state, symbol, plan, bar_ms):
    if count_positions(state) >= MAX_OPEN_POSITIONS:
        log(f"{symbol} | PORTFOLIO BLOCK | MaxOpenPositions={MAX_OPEN_POSITIONS}")
        return

    if plan["notional_cad"] > state["paper_cash_cad"]:
        log(
            f"{symbol} | PAPER BLOCK | Need C${plan['notional_cad']:.2f}, "
            f"available C${state['paper_cash_cad']:.2f}"
        )
        return

    if not acquire_shared_lock(symbol):
        return

    state["paper_cash_cad"] -= plan["notional_cad"]
    state["positions"][symbol] = {
        **plan,
        "mode": "paper",
        "entry": plan["planned_entry"],
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "signal_bar_ms": bar_ms,
    }
    save_state(state)

    log(
        f"{symbol} | PAPER BUY | Amount={plan['amount']:.10f} | "
        f"Entry=C${plan['planned_entry']:.2f} | SL=C${plan['stop']:.2f} | "
        f"TP=C${plan['target']:.2f} | Notional=C${plan['notional_cad']:.2f} | "
        f"Risk≈C${plan['risk_cad']:.2f}"
    )


def close_paper(state, symbol, exit_price, reason, bar_ms):
    pos = state["positions"].get(symbol)
    if not pos:
        return

    amount = float(pos["amount"])
    entry = float(pos["entry"])
    fee = float(pos.get("fee_rate", ESTIMATED_TAKER_FEE_PCT))

    entry_cost = amount * entry
    exit_value = amount * exit_price
    pnl = (
        exit_value
        - entry_cost
        - entry_cost * fee
        - exit_value * fee
    )

    state["paper_cash_cad"] += (
        exit_value
        - entry_cost * fee
        - exit_value * fee
    )
    del state["positions"][symbol]
    state["last_closed_bar_ms"][symbol] = int(bar_ms)
    release_shared_lock(symbol)
    save_state(state)

    log(
        f"{symbol} | PAPER CLOSE | {reason} | Exit=C${exit_price:.2f} | "
        f"NetPnL≈C${pnl:.2f} | PaperCash=C${state['paper_cash_cad']:.2f}"
    )

    write_shared_trade_line(
        symbol,
        pnl,
        "PAPER",
    )


# ============================================================
# LIVE ORDER HELPERS
# ============================================================

def wait_for_order_fill(exchange, symbol, order_id, timeout_seconds=25):
    started = time.time()
    last_order = None

    while time.time() - started < timeout_seconds:
        try:
            order = exchange.fetch_order(order_id, symbol)
            last_order = order
            status = (order.get("status") or "").lower()
            filled = float(order.get("filled") or 0.0)
            amount = float(order.get("amount") or 0.0)

            if status == "closed" or (amount > 0 and filled >= amount * 0.999):
                return order
        except Exception as exc:
            log(f"{symbol} | ORDER STATUS RETRY | {type(exc).__name__}: {exc}")

        time.sleep(1.5)

    return last_order


def emergency_market_sell(exchange, symbol, amount, reason):
    try:
        amount = float(exchange.amount_to_precision(symbol, amount))
        if amount <= 0:
            log(f"{symbol} | EMERGENCY SELL FAILED | Amount <= 0")
            return False

        order = exchange.create_order(
            symbol,
            "market",
            "sell",
            amount,
            None,
        )
        log(
            f"{symbol} | EMERGENCY MARKET SELL SENT | "
            f"Reason={reason} | OrderId={order.get('id')} | Amount={amount}"
        )
        return True
    except Exception as exc:
        log(
            f"{symbol} | CRITICAL | EMERGENCY SELL FAILED | "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def place_native_stop(exchange, symbol, amount, stop_price):
    # CCXT maps triggerPrice to NDAX StopMarket.
    params = {
        "triggerPrice": stop_price,
    }

    stop_order = exchange.create_order(
        symbol,
        "market",
        "sell",
        amount,
        None,
        params,
    )

    return stop_order


def open_live(exchange, state, symbol, plan, bar_ms):
    if not LIVE_ALLOWED:
        log(
            f"{symbol} | LIVE BLOCK | "
            f"Set LIVE_TRADING=true and LIVE_ARM=YES_REAL_ORDERS to enable."
        )
        return

    if not private_credentials_present():
        log(f"{symbol} | LIVE BLOCK | Missing API key/secret/user ID.")
        return

    if count_positions(state) >= MAX_OPEN_POSITIONS:
        log(f"{symbol} | PORTFOLIO BLOCK | MaxOpenPositions={MAX_OPEN_POSITIONS}")
        return

    if not acquire_shared_lock(symbol):
        return

    try:
        buy = exchange.create_order(
            symbol,
            "market",
            "buy",
            plan["amount"],
            None,
        )
    except Exception as exc:
        log(f"{symbol} | LIVE BUY FAILED | {type(exc).__name__}: {exc}")
        release_shared_lock(symbol)
        return

    buy_id = buy.get("id")
    log(
        f"{symbol} | LIVE BUY SENT | OrderId={buy_id} | "
        f"Amount={plan['amount']}"
    )

    filled_order = buy
    if buy_id:
        fetched = wait_for_order_fill(exchange, symbol, buy_id)
        if fetched:
            filled_order = fetched

    filled = float(
        filled_order.get("filled")
        or filled_order.get("amount")
        or plan["amount"]
    )

    avg = filled_order.get("average")
    if avg is None:
        avg = filled_order.get("price")
    if avg is None or float(avg) <= 0:
        # Fall back to latest ticker only if exchange did not expose fill price.
        ticker = exchange.fetch_ticker(symbol)
        avg = ticker.get("ask") or ticker.get("last") or plan["planned_entry"]

    entry = float(avg)

    if filled <= 0:
        log(f"{symbol} | LIVE BUY UNKNOWN FILL | No positive filled amount.")
        release_shared_lock(symbol)
        return

    sell_amount = min(
        plan["protective_amount"],
        filled * (1.0 - PROTECTIVE_SELL_BUFFER_PCT),
    )
    sell_amount = float(exchange.amount_to_precision(symbol, sell_amount))

    if sell_amount <= 0:
        log(f"{symbol} | CRITICAL | Protective amount <= 0 after BUY.")
        emergency_market_sell(exchange, symbol, filled, "No protective amount")
        release_shared_lock(symbol)
        return

    # Recalculate stop and TP from actual fill while retaining original distance.
    original_distance = plan["planned_entry"] - plan["stop"]
    stop_price = entry - original_distance

    fee = ESTIMATED_TAKER_FEE_PCT
    risk_per_unit = (
        (entry - stop_price)
        + entry * fee
        + stop_price * fee
    )
    target_net_per_unit = RR_RATIO * risk_per_unit
    target_price = (
        target_net_per_unit + entry * (1.0 + fee)
    ) / (1.0 - fee)

    stop_price = float(exchange.price_to_precision(symbol, stop_price))
    target_price = float(exchange.price_to_precision(symbol, target_price))

    # Verify actual intended risk did not exceed the CAD cap materially.
    actual_risk = sell_amount * risk_per_unit
    if actual_risk > RISK_CAD * 1.05:
        log(
            f"{symbol} | CRITICAL RISK CHECK | Actual≈C${actual_risk:.2f} "
            f"> cap C${RISK_CAD:.2f}; emergency exit."
        )
        emergency_market_sell(exchange, symbol, sell_amount, "Post-fill risk exceeded")
        release_shared_lock(symbol)
        return

    # Create downside protection immediately.
    try:
        stop_order = place_native_stop(
            exchange,
            symbol,
            sell_amount,
            stop_price,
        )
        stop_id = stop_order.get("id")
    except Exception as exc:
        log(
            f"{symbol} | CRITICAL | NATIVE STOP FAILED | "
            f"{type(exc).__name__}: {exc}"
        )
        emergency_market_sell(
            exchange,
            symbol,
            sell_amount,
            "Native stop creation failed",
        )
        release_shared_lock(symbol)
        return

    state["positions"][symbol] = {
        "mode": "live",
        "entry_order_id": buy_id,
        "stop_order_id": stop_id,
        "entry": entry,
        "amount": filled,
        "managed_sell_amount": sell_amount,
        "stop": stop_price,
        "target": target_price,
        "risk_cad": actual_risk,
        "rr": RR_RATIO,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "signal_bar_ms": bar_ms,
    }
    state["daily"]["trades"] = int(state["daily"].get("trades", 0)) + 1
    save_state(state)

    log(
        f"{symbol} | LIVE POSITION PROTECTED | "
        f"Entry=C${entry:.2f} | Amount={filled:.10f} | "
        f"NativeStop=C${stop_price:.2f} | StopOrderId={stop_id} | "
        f"LocalTP=C${target_price:.2f} | Risk≈C${actual_risk:.2f}"
    )



def record_closed_trade(state, symbol, entry, exit_price, amount, reason):
    """
    Record an approximate net realized P/L using the configured fee rate.
    This gives an immediate bot-side trade summary in C:\\NDAX_BOT\\log.txt.
    NDAX account history remains the final source of truth for exact fees/fills.
    """
    fee = ESTIMATED_TAKER_FEE_PCT
    entry_value = float(entry) * float(amount)
    exit_value = float(exit_price) * float(amount)
    estimated_fees = entry_value * fee + exit_value * fee
    gross_pnl = exit_value - entry_value
    net_pnl = gross_pnl - estimated_fees

    daily = state.setdefault("daily", {})
    stats = state.setdefault("stats", {})

    daily["realized_pnl_cad"] = float(daily.get("realized_pnl_cad", 0.0)) + net_pnl
    stats["realized_pnl_cad"] = float(stats.get("realized_pnl_cad", 0.0)) + net_pnl
    stats["closed_trades"] = int(stats.get("closed_trades", 0)) + 1

    if net_pnl >= 0:
        daily["wins"] = int(daily.get("wins", 0)) + 1
        stats["wins"] = int(stats.get("wins", 0)) + 1
        result = "WIN"
    else:
        daily["losses"] = int(daily.get("losses", 0)) + 1
        stats["losses"] = int(stats.get("losses", 0)) + 1
        result = "LOSS"

    log(
        f"TRADE RESULT | {symbol} | {result} | Reason={reason} | "
        f"Entry=C${entry:.2f} | Exit=C${exit_price:.2f} | "
        f"Amount={amount:.10f} | GrossPnL≈C${gross_pnl:.2f} | "
        f"EstFees≈C${estimated_fees:.2f} | NetPnL≈C${net_pnl:.2f}"
    )
    log(
        f"SESSION SUMMARY | Closed={stats.get('closed_trades',0)} | "
        f"Wins={stats.get('wins',0)} | Losses={stats.get('losses',0)} | "
        f"CumulativeNetPnL≈C${float(stats.get('realized_pnl_cad',0.0)):.2f} | "
        f"TodayNetPnL≈C${float(daily.get('realized_pnl_cad',0.0)):.2f}"
    )

    write_shared_trade_line(
        symbol,
        net_pnl,
        "LIVE",
    )

# ============================================================
# POSITION MANAGEMENT
# ============================================================

def manage_paper(exchange, state, symbol, bar_ms):
    pos = state["positions"].get(symbol)
    if not pos or pos.get("mode") != "paper":
        return

    ticker = exchange.fetch_ticker(symbol)
    price = ticker.get("bid") or ticker.get("last")
    if price is None:
        return

    price = float(price)

    if price <= float(pos["stop"]):
        close_paper(state, symbol, price, "STOP", bar_ms)
    elif price >= float(pos["target"]):
        close_paper(state, symbol, price, "TARGET", bar_ms)


def live_stop_status(exchange, symbol, stop_id):
    if not stop_id:
        return "unknown", None

    try:
        order = exchange.fetch_order(stop_id, symbol)
        return (order.get("status") or "unknown").lower(), order
    except Exception:
        return "unknown", None


def manage_live(exchange, state, symbol, bar_ms):
    pos = state["positions"].get(symbol)
    if not pos or pos.get("mode") != "live":
        return

    ticker = exchange.fetch_ticker(symbol)
    bid = ticker.get("bid") or ticker.get("last")
    if bid is None:
        return
    bid = float(bid)

    stop_id = pos.get("stop_order_id")
    status, stop_order = live_stop_status(exchange, symbol, stop_id)

    # If the native stop filled, record closure.
    if status == "closed":
        exit_price = None
        if stop_order:
            exit_price = stop_order.get("average") or stop_order.get("price")

        log(
            f"{symbol} | LIVE STOP EXECUTED | "
            f"StopOrderId={stop_id} | Exit≈{exit_price}"
        )

        if exit_price is not None:
            record_closed_trade(
                state,
                symbol,
                float(pos["entry"]),
                float(exit_price),
                float(pos["managed_sell_amount"]),
                "STOP",
            )

        del state["positions"][symbol]
        state["last_closed_bar_ms"][symbol] = int(bar_ms)
        release_shared_lock(symbol)
        save_state(state)
        return

    target = float(pos["target"])

    # Take profit is managed locally. Native downside stop remains resting
    # on exchange until the moment the TP exit is sent.
    if bid >= target:
        amount = float(pos["managed_sell_amount"])

        # Cancel native stop first so it cannot also sell after TP exit.
        try:
            if stop_id:
                exchange.cancel_order(stop_id, symbol)
                log(f"{symbol} | TP EXIT | Native stop canceled: {stop_id}")
        except Exception as exc:
            log(
                f"{symbol} | TP EXIT WARNING | Could not cancel stop: "
                f"{type(exc).__name__}: {exc}"
            )
            # Do not send another SELL when stop cancellation is uncertain.
            return

        try:
            sell = exchange.create_order(
                symbol,
                "market",
                "sell",
                amount,
                None,
            )
            log(
                f"{symbol} | LIVE TARGET SELL SENT | "
                f"OrderId={sell.get('id')} | Amount={amount} | Bid≈C${bid:.2f}"
            )

            exit_price = bid
            sell_id = sell.get("id")
            if sell_id:
                filled_sell = wait_for_order_fill(exchange, symbol, sell_id)
                if filled_sell:
                    px = filled_sell.get("average") or filled_sell.get("price")
                    if px is not None and float(px) > 0:
                        exit_price = float(px)

            record_closed_trade(
                state,
                symbol,
                float(pos["entry"]),
                float(exit_price),
                amount,
                "TARGET",
            )

            del state["positions"][symbol]
            state["last_closed_bar_ms"][symbol] = int(bar_ms)
            release_shared_lock(symbol)
            save_state(state)
        except Exception as exc:
            log(
                f"{symbol} | CRITICAL TP SELL FAILED AFTER STOP CANCEL | "
                f"{type(exc).__name__}: {exc}"
            )

            # Recreate downside protection immediately.
            try:
                new_stop = place_native_stop(
                    exchange,
                    symbol,
                    amount,
                    float(pos["stop"]),
                )
                pos["stop_order_id"] = new_stop.get("id")
                state["positions"][symbol] = pos
                save_state(state)
                log(
                    f"{symbol} | PROTECTION RESTORED | "
                    f"NewStopOrderId={pos['stop_order_id']}"
                )
            except Exception as stop_exc:
                log(
                    f"{symbol} | CRITICAL | FAILED TO RESTORE STOP | "
                    f"{type(stop_exc).__name__}: {stop_exc}"
                )


# ============================================================
# COOLDOWN
# ============================================================

def cooldown_complete(state, symbol, bar_ms):
    last_closed = int(state["last_closed_bar_ms"].get(symbol, 0))
    if last_closed <= 0 or COOLDOWN_BARS_AFTER_CLOSE <= 0:
        return True

    return (
        bar_ms - last_closed
        >= COOLDOWN_BARS_AFTER_CLOSE * TIMEFRAME_MS
    )


# ============================================================
# SCAN
# ============================================================

def scan_symbol(exchange, state, symbol):
    if symbol not in exchange.symbols:
        log(f"{symbol} | ERROR | Symbol unavailable on NDAX.")
        return

    m1_df, newest_raw_ms, completed_ms = fetch_completed_m1(
        exchange,
        symbol,
    )

    m5_df, _, m5_completed_ms = fetch_completed_m5_trend(
        exchange,
        symbol,
    )

    # Manage open position every cycle, not only on new candle.
    if symbol in state["positions"]:
        if state["positions"][symbol].get("mode") == "live":
            manage_live(exchange, state, symbol, completed_ms)
        else:
            manage_paper(exchange, state, symbol, completed_ms)

    last_processed = int(state["last_processed_bar"].get(symbol, 0))
    if completed_ms == last_processed:
        return

    state["last_processed_bar"][symbol] = completed_ms
    save_state(state)

    completed_time = datetime.fromtimestamp(
        completed_ms / 1000,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M:%S")

    raw_time = datetime.fromtimestamp(
        newest_raw_ms / 1000,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M:%S")

    log(
        f"{symbol} | NEW CONFIRMED M1 BAR | End={completed_time} UTC | "
        f"NewestRawEnd={raw_time} UTC"
    )

    spread_bps, bid, ask, _ = fetch_spread_bps(exchange, symbol)

    if spread_bps is None:
        log(f"{symbol} | FILTER | Invalid Bid/Ask Bid={bid} Ask={ask}")
        return

    symbol_max_spread = max_spread_bps_for(symbol)

    if spread_bps > symbol_max_spread:
        log(
            f"{symbol} | FILTER | Spread={spread_bps:.1f} bps "
            f"> max {symbol_max_spread:.1f}"
        )
        return

    signal, diag = evaluate_signal(m1_df, m5_df)

    log(
        f"{symbol} | CHECK | "
        f"ADX={diag.get('adx', 0):.1f} | "
        f"RSI={diag.get('rsi', 0):.1f} | "
        f"+DI/-DI={diag.get('plus_di', 0):.1f}/"
        f"{diag.get('minus_di', 0):.1f} | "
        f"EMA/ATR={diag.get('ema_sep_atr', 0):.3f} | "
        f"BB/ATR={diag.get('bb_width_atr', 0):.2f} | "
        f"VolRatio={diag.get('volume_ratio', 0):.2f} | "
        f"M5Trend={'UP' if diag.get('m5_up') else ('DOWN' if diag.get('m5_down') else 'FLAT')} | "
        f"M5EMA/ATR={diag.get('m5_sep_atr', 0):.3f} | "
        f"Spread={spread_bps:.1f}bps | "
        f"Result={diag.get('reason', 'n/a')}"
    )

    if signal == "sell_signal":
        log(
            f"{symbol} | BEARISH SIGNAL | "
            f"Spot M1 bot does not open short positions."
        )
        return

    if signal != "buy":
        return

    if symbol in state["positions"]:
        log(f"{symbol} | SKIP | Existing bot position already open.")
        return

    if count_positions(state) >= MAX_OPEN_POSITIONS:
        log(
            f"{symbol} | PORTFOLIO BLOCK | "
            f"Open={count_positions(state)} >= Max={MAX_OPEN_POSITIONS}"
        )
        return

    shared_holder = read_shared_lock()
    if shared_holder is not None:
        log(
            f"{symbol} | SHARED PORTFOLIO BLOCK | "
            f"Bot={shared_holder.get('bot')} | "
            f"Symbol={shared_holder.get('symbol')}"
        )
        return

    if not cooldown_complete(state, symbol, completed_ms):
        log(
            f"{symbol} | COOLDOWN | Waiting "
            f"{COOLDOWN_BARS_AFTER_CLOSE} completed M1 bars."
        )
        return

    if ask is None or float(ask) <= 0:
        log(f"{symbol} | ENTRY BLOCK | Invalid ask.")
        return

    entry = float(ask)
    atr_value = float(diag["atr"])

    cash_cap = (
        float(state["paper_cash_cad"])
        if not LIVE_ALLOWED
        else MAX_POSITION_CAD
    )

    plan, error = calculate_long_plan(
        exchange,
        symbol,
        entry,
        atr_value,
        cash_cap,
    )

    if not plan:
        log(f"{symbol} | RISK BLOCK | {error}")
        return

    log(
        f"{symbol} | BUY SIGNAL | "
        f"Entry≈C${plan['planned_entry']:.2f} | "
        f"SL=C${plan['stop']:.2f} | TP=C${plan['target']:.2f} | "
        f"Amount={plan['amount']:.10f} | "
        f"Notional=C${plan['notional_cad']:.2f} | "
        f"Risk≈C${plan['risk_cad']:.2f}"
    )

    if LIVE_ALLOWED:
        open_live(
            exchange,
            state,
            symbol,
            plan,
            completed_ms,
        )
    else:
        open_paper(
            state,
            symbol,
            plan,
            completed_ms,
        )


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat(exchange, state):
    now = time.time()
    last = float(state.get("last_heartbeat", 0.0))

    if now - last < HEARTBEAT_SECONDS:
        return

    state["last_heartbeat"] = now
    save_state(state)

    mode = "LIVE-ARMED" if LIVE_ALLOWED else "PAPER"
    log(
        f"HEARTBEAT | Mode={mode} | "
        f"OpenPositions={count_positions(state)} | "
        f"DailyTrades={state['daily'].get('trades',0)} | "
        f"DailyPnL≈C${float(state['daily'].get('realized_pnl_cad',0.0)):.2f} | "
        f"Symbols={','.join(SYMBOLS)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 78)
    log(f"{BOT_VERSION} INITIALIZING")
    log(f"LogFile={LOG_FILE}")
    log(f"SharedPositionLock={SHARED_POSITION_LOCK_FILE}")
    log(f"SharedTradeLog={SHARED_TRADE_LOG_FILE}")
    log(
        f"Mode={'LIVE-ARMED' if LIVE_ALLOWED else 'PAPER'} | "
        f"RequestedLive={LIVE_TRADING} | "
        f"Symbols={SYMBOLS} | TF=M1 + M5 trend | "
        f"RiskCap=C${RISK_CAD:.2f} | RR={RR_RATIO:.1f}:1 | "
        f"MaxPosition=C${MAX_POSITION_CAD:.2f} | "
        f"MaxOpen={MAX_OPEN_POSITIONS}"
    )
    log(
        f"Filters | ADX>={MIN_ADX:.1f} | RSI BUY={BUY_RSI_MIN:.1f}-{BUY_RSI_MAX:.1f} | "
        f"EMA/ATR>={MIN_EMA_SEPARATION_ATR:.3f} | "
        f"BB/ATR>={MIN_BAND_WIDTH_ATR:.2f} | "
        f"VolRatio>={VOL_RATIO_THRESH:.2f} | "
        f"DefaultSpread={MAX_SPREAD_BPS_DEFAULT:.1f}bps"
    )
    log(
        f"M1 Stop=max(ATR x {ATR_STOP_MULTIPLIER:.2f}, "
        f"{MIN_STOP_PCT*100:.2f}% price) | "
        f"CandleGrace={CANDLE_CLOSE_GRACE_SECONDS}s"
    )
    log(
        f"Production safety | MinCADReserve=C${MIN_CAD_RESERVE:.2f} | "
        f"MaxTrades/UTCday={MAX_TRADES_PER_UTC_DAY} | "
        f"MaxDailyLoss=C${MAX_DAILY_LOSS_CAD:.2f}"
    )
    log(
        f"M1 scalp profile | M5Trend={'ON' if USE_M5_TREND_FILTER else 'OFF'} | "
        f"M5Slope={'ON' if REQUIRE_M5_EMA_SLOPE else 'OFF'} | "
        f"M5EMA/ATR>={M5_MIN_EMA_SEPARATION_ATR:.3f} | "
        f"MaxTargetMove={MAX_TARGET_MOVE_PCT*100:.2f}%"
    )

    if LIVE_TRADING and not LIVE_ALLOWED:
        log(
            "LIVE SAFETY BLOCK | LIVE_TRADING=true but LIVE_ARM is not "
            "YES_REAL_ORDERS. No real orders can be sent."
        )

    exchange = build_exchange()
    log(f"NDAX markets loaded: {len(exchange.symbols)}")

    for symbol in SYMBOLS:
        log(
            f"{symbol} | "
            f"{'AVAILABLE' if symbol in exchange.symbols else 'NOT AVAILABLE'}"
        )

    for symbol in SYMBOLS:
        log(
            f"{symbol} | MaxSpread="
            f"{max_spread_bps_for(symbol):.1f}bps"
        )

    # IMPORTANT: state must exist BEFORE private authentication.
    state = load_state()
    roll_daily_state(state)
    reconcile_shared_lock(state)

    private_ok, account_id = test_private_connection(exchange, state)

    if LIVE_ALLOWED and not private_ok:
        raise RuntimeError(
            "LIVE MODE ABORTED: NDAX private authentication failed."
        )

    save_state(state)

    log(
        f"State | PaperCash=C${state['paper_cash_cad']:.2f} | "
        f"OpenPositions={count_positions(state)} | "
        f"DailyTrades={state['daily'].get('trades',0)} | "
        f"DailyPnL=C${float(state['daily'].get('realized_pnl_cad',0.0)):.2f}"
    )
    log("=" * 78)

    while True:
        cycle_started = time.time()

        roll_daily_state(state)
        heartbeat(exchange, state)

        for symbol in SYMBOLS:
            try:
                scan_symbol(exchange, state, symbol)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                log(
                    f"{symbol} | ERROR | "
                    f"{type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

            time.sleep(1.0)

        elapsed = time.time() - cycle_started
        time.sleep(max(1.0, POLL_SECONDS - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("BOT STOPPED BY USER")
