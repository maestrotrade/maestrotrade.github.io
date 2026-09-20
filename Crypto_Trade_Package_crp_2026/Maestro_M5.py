#!/usr/bin/env python3
"""
NDAX MAESTRO M15 SCALP CURR V1.0
Paper-forward-test crypto translation of the Forex MAESTRO SCALP CURR entry family.

ENTRY PHILOSOPHY (translated from the supplied Forex EA):
  CLOSED M15:
    - EMA9 > EMA21 > EMA50
    - EMA9 slope > 0, EMA21 slope >= -0.01 ATR
    - M30 and H1 close above EMA50
    - bullish pullback OR bullish continuation
    - exhaustion / weak-cross / direction-conflict vetoes
  LIVE M15:
    - immediate timing clues = RSI delta, MACD delta, price swing vs prior M15 close
    - 3/3 agreement -> entry
    - 2/3 -> only if ADX>=23, DI dominance>=3, direction score>=3,
             RSI acceleration>=0.50 and MACD acceleration>0
    - 0/3 or 1/3 -> WAIT and recheck same M15 bar later

NDAX adaptations:
  - LONG ONLY: NDAX spot cannot open a naked short.
  - order-book bid/ask used for executable spread
  - fee/slippage-aware paper fills and CAD risk sizing
  - same credential variables / read-only account ping style as prior MAESTRO bots
  - paper mode by default; real orders remain hard-gated
"""

import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import MAESTRO_ENTRY as ENTRY

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BOT_VERSION = "MAESTRO_M5_CRYPTO_V127_2"
TIMEFRAME = "5m"
M30_TIMEFRAME = "30m"
H1_TIMEFRAME = "1h"
TIMEFRAME_MS = 300000
OHLCV_LIMIT = 240
TREND_LIMIT = 140


# ============================================================
# ENV HELPERS
# ============================================================
def env_bool(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y", "on"}

def env_float(name, default):
    return float(os.getenv(name, str(default)))

def env_int(name, default):
    return int(os.getenv(name, str(default)))


# ============================================================
# SAFETY / AUTH
# ============================================================
PAPER_MODE = env_bool("M15_PAPER_MODE", True)
PAPER_START_BALANCE_CAD = env_float("M15_PAPER_START_BALANCE_CAD", 500.0)
LIVE_TRADING = env_bool("LIVE_TRADING", False)
LIVE_ARM = os.getenv("LIVE_ARM", "").strip()
LIVE_ALLOWED = (not PAPER_MODE) and LIVE_TRADING and LIVE_ARM == "YES_REAL_ORDERS"

NDAX_API_KEY = os.getenv("NDAX_API_KEY", "").strip()
NDAX_API_SECRET = os.getenv("NDAX_API_SECRET", "").strip()
NDAX_USER_ID = os.getenv("NDAX_USER_ID", "").strip()
NDAX_ACCOUNT_ID_ENV = os.getenv("NDAX_ACCOUNT_ID", "").strip()

SYMBOLS = [x.strip() for x in os.getenv(
    "NDAX_SYMBOLS", "BTC/CAD,ETH/CAD,XRP/CAD,SOL/CAD,ADA/CAD"
).split(",") if x.strip()]
AUTO_DISCOVER = env_bool("NDAX_AUTO_DISCOVER_CAD_MARKETS", True)
AUTO_QUOTE = os.getenv("NDAX_AUTO_QUOTE", "CAD").strip().upper()
FAST_UNIVERSE_SIZE = env_int("M15_FAST_UNIVERSE_SIZE", 5)

# Hard paper-test risk limits.
RISK_CAD = float(ENTRY.PARAMS["risk_cad"])
MAX_DAILY_LOSS_CAD = float(ENTRY.PARAMS["max_daily_loss_cad"])
MAX_POSITION_CAD = float(ENTRY.PARAMS["max_position_cad"])
MIN_NOTIONAL_CAD = float(ENTRY.PARAMS["min_live_notional_cad"])
MIN_CAD_RESERVE = float(ENTRY.PARAMS["min_cad_reserve"])
MAX_OPEN_POSITIONS = int(ENTRY.PARAMS["max_open_positions"])
MAX_TRADES_DAY = int(ENTRY.PARAMS["max_trades_day"])
MAX_CONSEC_LOSSES = int(ENTRY.PARAMS["max_consec_losses"])
LOSS_PAUSE_MINUTES = int(ENTRY.PARAMS["loss_pause_minutes"])
SYMBOL_LOSS_COOLDOWN_MINUTES = int(ENTRY.PARAMS["symbol_loss_cooldown_minutes"])

# ============================================================
# SCALP CURR ENTRY TRANSLATION
# ============================================================
SCALP_PULL_TOUCH_ATR = env_float("M15_SCALP_PULL_TOUCH_ATR", 0.25)
SCALP_PULL_MAX_STRETCH_ATR = env_float("M15_SCALP_PULL_MAX_STRETCH_ATR", 1.10)
SCALP_CONT_MAX_STRETCH_ATR = env_float("M15_SCALP_CONT_MAX_STRETCH_ATR", 1.35)
SCALP_CONT_MIN_RSI = env_float("M15_SCALP_CONT_MIN_RSI", 52.0)

# Controlled crypto transition lane:
# allows strong early bullish structure before EMA21 fully clears EMA50.
SCALP_TRANSITION_ENABLED = env_bool("M15_SCALP_TRANSITION_ENABLED", True)
SCALP_TRANSITION_MIN_RSI = env_float("M15_SCALP_TRANSITION_MIN_RSI", 50.5)
SCALP_TRANSITION_MIN_ADX = env_float("M15_SCALP_TRANSITION_MIN_ADX", 18.0)
SCALP_TRANSITION_MIN_DI_RATIO = env_float("M15_SCALP_TRANSITION_MIN_DI_RATIO", 1.10)
SCALP_TRANSITION_MIN_S9_ATR = env_float("M15_SCALP_TRANSITION_MIN_S9_ATR", 0.04)
SCALP_TRANSITION_MIN_S21_ATR = env_float("M15_SCALP_TRANSITION_MIN_S21_ATR", 0.00)
SCALP_TRANSITION_MAX_GAP21_50_ATR = env_float("M15_SCALP_TRANSITION_MAX_GAP21_50_ATR", 0.65)
SCALP_TRANSITION_MAX_STRETCH_ATR = env_float("M15_SCALP_TRANSITION_MAX_STRETCH_ATR", 1.15)
SCALP_TRANSITION_REQUIRE_M30_H1 = env_bool("M15_SCALP_TRANSITION_REQUIRE_M30_H1", True)

# --------------------------- V3 RESEARCH ENTRY LANES ---------------------------
SCALP_OPPORTUNITY_ENABLED = env_bool("M15_SCALP_OPPORTUNITY_ENABLED", True)
SCALP_OPPORTUNITY_MIN_RSI = env_float("M15_SCALP_OPPORTUNITY_MIN_RSI", 49.5)
SCALP_OPPORTUNITY_MIN_ADX = env_float("M15_SCALP_OPPORTUNITY_MIN_ADX", 16.0)
SCALP_OPPORTUNITY_MIN_DI_RATIO = env_float("M15_SCALP_OPPORTUNITY_MIN_DI_RATIO", 1.05)
SCALP_OPPORTUNITY_MIN_S9_ATR = env_float("M15_SCALP_OPPORTUNITY_MIN_S9_ATR", 0.015)
SCALP_OPPORTUNITY_MIN_S21_ATR = env_float("M15_SCALP_OPPORTUNITY_MIN_S21_ATR", -0.02)
SCALP_OPPORTUNITY_MAX_GAP21_50_ATR = env_float("M15_SCALP_OPPORTUNITY_MAX_GAP21_50_ATR", 0.90)
SCALP_OPPORTUNITY_MAX_STRETCH_ATR = env_float("M15_SCALP_OPPORTUNITY_MAX_STRETCH_ATR", 1.00)
SCALP_OPPORTUNITY_REQUIRE_ONE_MACRO_UP = env_bool("M15_SCALP_OPPORTUNITY_REQUIRE_ONE_MACRO_UP", True)

SCALP_BREAKOUT_ENABLED = env_bool("M15_SCALP_BREAKOUT_ENABLED", True)
SCALP_BREAKOUT_LOOKBACK_BARS = env_int("M15_SCALP_BREAKOUT_LOOKBACK_BARS", 6)
SCALP_BREAKOUT_BUFFER_ATR = env_float("M15_SCALP_BREAKOUT_BUFFER_ATR", 0.05)
SCALP_BREAKOUT_MIN_BODY_ATR = env_float("M15_SCALP_BREAKOUT_MIN_BODY_ATR", 0.12)
SCALP_BREAKOUT_MIN_RSI = env_float("M15_SCALP_BREAKOUT_MIN_RSI", 52.0)
SCALP_BREAKOUT_MIN_ADX = env_float("M15_SCALP_BREAKOUT_MIN_ADX", 18.0)
SCALP_BREAKOUT_MIN_DI_RATIO = env_float("M15_SCALP_BREAKOUT_MIN_DI_RATIO", 1.08)
SCALP_BREAKOUT_MAX_STRETCH_ATR = env_float("M15_SCALP_BREAKOUT_MAX_STRETCH_ATR", 1.05)

SCALP_CONT_ENTRY_MAX_STRETCH_ATR = env_float("M15_SCALP_CONT_ENTRY_MAX_STRETCH_ATR", 0.95)
SCALP_CONT_MATURE_RSI = env_float("M15_SCALP_CONT_MATURE_RSI", 66.0)
SCALP_CONT_MATURE_ADX = env_float("M15_SCALP_CONT_MATURE_ADX", 32.0)
SCALP_CONT_MATURE_MAX_STRETCH_ATR = env_float("M15_SCALP_CONT_MATURE_MAX_STRETCH_ATR", 0.80)

SCALP_TRACK_REJECTED = env_bool("M15_SCALP_TRACK_REJECTED", True)
SCALP_TRACK_POST_TRADE = env_bool("M15_SCALP_TRACK_POST_TRADE", True)
SCALP_TRACK_MINUTES = os.getenv("M15_SCALP_TRACK_MINUTES", "1,3,5,10,15,30,60")
SCALP_MAX_TRACK_WATCHES = env_int("M15_SCALP_MAX_TRACK_WATCHES", 200)

# --------------------------- V4 EARLY RECLAIM ---------------------------
# Converts good rejected near-entries into actionable watches. The aim is to
# enter near the original opportunity rather than after a 0.4%-0.7% run.
EARLY_RECLAIM_ENABLED = env_bool("M15_EARLY_RECLAIM_ENABLED", True)
EARLY_RECLAIM_MIN_AGE_SEC = env_int("M15_EARLY_RECLAIM_MIN_AGE_SEC", 25)
EARLY_RECLAIM_MAX_AGE_SEC = env_int("M15_EARLY_RECLAIM_MAX_AGE_SEC", 900)
EARLY_RECLAIM_MIN_MOVE_BPS = env_float("M15_EARLY_RECLAIM_MIN_MOVE_BPS", 2.0)
EARLY_RECLAIM_MAX_MOVE_BPS = env_float("M15_EARLY_RECLAIM_MAX_MOVE_BPS", 12.0)
EARLY_RECLAIM_MAX_ADVERSE_BPS = env_float("M15_EARLY_RECLAIM_MAX_ADVERSE_BPS", 10.0)
EARLY_RECLAIM_MIN_SCORE = env_int("M15_EARLY_RECLAIM_MIN_SCORE", 3)
EARLY_RECLAIM_MAX_OPP_SCORE = env_int("M15_EARLY_RECLAIM_MAX_OPP_SCORE", 1)
EARLY_RECLAIM_MIN_DI_RATIO = env_float("M15_EARLY_RECLAIM_MIN_DI_RATIO", 1.08)
EARLY_RECLAIM_MIN_RSI = env_float("M15_EARLY_RECLAIM_MIN_RSI", 51.0)
EARLY_RECLAIM_MAX_STRETCH_ATR = env_float("M15_EARLY_RECLAIM_MAX_STRETCH_ATR", 0.95)
EARLY_RECLAIM_CONFIRM_HITS = env_int("M15_EARLY_RECLAIM_CONFIRM_HITS", 1)

# Recovery pullback: a red/neutral closed pullback may still be valid if it
# holds EMA21 and live momentum subsequently reclaims.
SCALP_RECOVERY_PULLBACK_ENABLED = env_bool("M15_SCALP_RECOVERY_PULLBACK_ENABLED", True)
SCALP_RECOVERY_MAX_STRETCH_ATR = env_float("M15_SCALP_RECOVERY_MAX_STRETCH_ATR", 0.85)
SCALP_RECOVERY_MIN_DI_RATIO = env_float("M15_SCALP_RECOVERY_MIN_DI_RATIO", 0.95)

SCALP_EXHAUSTION_MAX_STRETCH_ATR = env_float("M15_SCALP_EXHAUSTION_MAX_STRETCH_ATR", 1.85)
SCALP_EXHAUSTION_CONDITIONAL_ATR = env_float("M15_SCALP_EXHAUSTION_CONDITIONAL_ATR", 1.35)
SCALP_EXTREME_RSI = env_float("M15_SCALP_EXTREME_RSI", 69.0)
SCALP_WEAK_ADX = env_float("M15_SCALP_WEAK_ADX", 18.0)

# Weak young cross veto copied from Forex SCALP behavior.
SCALP_WEAK_CROSS_ADX = env_float("M15_SCALP_WEAK_CROSS_ADX", 18.0)
SCALP_WEAK_CROSS_MIN_GAP_ATR = env_float("M15_SCALP_WEAK_CROSS_MIN_GAP_ATR", 0.10)
SCALP_WEAK_CROSS_MIN_BODY_ATR = env_float("M15_SCALP_WEAK_CROSS_MIN_BODY_ATR", 0.15)

# Immediate timing gate.
SCALP_IMM_RSI_DEADBAND = env_float("M15_SCALP_IMM_RSI_DEADBAND", 0.20)
SCALP_IMM_SWING_ATR = env_float("M15_SCALP_IMM_SWING_ATR", 0.03)

# Exact SCALP family quality rule:
# 3/3 accepted; 2/3 must be strong.
SCALP_STRONG_2OF3_MIN_ADX = env_float("M15_SCALP_STRONG_2OF3_MIN_ADX", 23.0)
SCALP_STRONG_2OF3_MIN_DI_DOM = env_float("M15_SCALP_STRONG_2OF3_MIN_DI_DOM", 3.0)
SCALP_STRONG_2OF3_MIN_SCORE = env_int("M15_SCALP_STRONG_2OF3_MIN_SCORE", 3)
SCALP_STRONG_2OF3_MIN_RSI_ACCEL = env_float("M15_SCALP_STRONG_2OF3_MIN_RSI_ACCEL", 0.50)

SCALP_DIRECT_3OF3_MIN_ADX = env_float("M15_SCALP_DIRECT_3OF3_MIN_ADX", 18.0)
SCALP_DIRECT_3OF3_MIN_SCORE = env_int("M15_SCALP_DIRECT_3OF3_MIN_SCORE", 3)
SCALP_DIRECT_3OF3_MIN_DI_DOM = env_float("M15_SCALP_DIRECT_3OF3_MIN_DI_DOM", 2.0)

# --------------------------- V6 LIVE TREND-DIP / RESTART ---------------------------
LIVE_DIP_ENABLED = env_bool("M15_LIVE_DIP_ENABLED", True)
LIVE_DIP_REQUIRE_BOTH_MACRO_UP = env_bool("M15_LIVE_DIP_REQUIRE_BOTH_MACRO_UP", True)
LIVE_DIP_MIN_ADX = env_float("M15_LIVE_DIP_MIN_ADX", 16.0)
LIVE_DIP_MIN_RSI = env_float("M15_LIVE_DIP_MIN_RSI", 50.0)
LIVE_DIP_MAX_RSI = env_float("M15_LIVE_DIP_MAX_RSI", 66.0)
LIVE_DIP_MIN_DI_RATIO = env_float("M15_LIVE_DIP_MIN_DI_RATIO", 1.05)
LIVE_DIP_MAX_STRETCH_ATR = env_float("M15_LIVE_DIP_MAX_STRETCH_ATR", 0.85)
LIVE_DIP_MAX_EMA21_DISTANCE_ATR = env_float("M15_LIVE_DIP_MAX_EMA21_DISTANCE_ATR", 0.95)
LIVE_DIP_MIN_RSI_DELTA = env_float("M15_LIVE_DIP_MIN_RSI_DELTA", 0.15)
LIVE_DIP_MIN_MOVE_ATR = env_float("M15_LIVE_DIP_MIN_MOVE_ATR", 0.00)
LIVE_DIP_MIN_CLUES = env_int("M15_LIVE_DIP_MIN_CLUES", 3)

# A separate current-bar breakout lane for clean trend restarts.
LIVE_BREAKOUT_ENABLED = env_bool("M15_LIVE_BREAKOUT_ENABLED", True)
LIVE_BREAKOUT_BUFFER_ATR = env_float("M15_LIVE_BREAKOUT_BUFFER_ATR", 0.03)
LIVE_BREAKOUT_MAX_STRETCH_ATR = env_float("M15_LIVE_BREAKOUT_MAX_STRETCH_ATR", 0.95)
LIVE_BREAKOUT_MIN_ADX = env_float("M15_LIVE_BREAKOUT_MIN_ADX", 18.0)
LIVE_BREAKOUT_MIN_DI_RATIO = env_float("M15_LIVE_BREAKOUT_MIN_DI_RATIO", 1.10)
LIVE_BREAKOUT_MIN_RSI = env_float("M15_LIVE_BREAKOUT_MIN_RSI", 52.0)

LIVE_ENTRY_COOLDOWN_SEC = env_int("M15_LIVE_ENTRY_COOLDOWN_SEC", 300)

# --------------------------- V8 MAESTRO M15 ---------------------------
# Ported conceptually from the attached MAESTRO V76 engine:
# 1) V103-like M15 trend/pullback/breakout lane
# 2) HEALTHY_TREND lane with SCALP H1 exception
# 3) M5 + M1 confirmation
# 4) not-late/not-hostile proof before entry
MAESTRO_M15_ENABLED = env_bool("M15_MAESTRO_ENABLED", True)

# Healthy-trend structure.
MAESTRO_HT_MIN_ADX = env_float("M15_MAESTRO_HT_MIN_ADX", 18.0)
MAESTRO_HT_MAX_ADX = env_float("M15_MAESTRO_HT_MAX_ADX", 36.0)
MAESTRO_HT_MIN_DI_DOM = env_float("M15_MAESTRO_HT_MIN_DI_DOM", 3.0)
MAESTRO_HT_MIN_S9_ATR = env_float("M15_MAESTRO_HT_MIN_S9_ATR", 0.012)
MAESTRO_HT_MIN_S21_ATR = env_float("M15_MAESTRO_HT_MIN_S21_ATR", -0.008)
MAESTRO_HT_MAX_TRANSITION_GAP_ATR = env_float("M15_MAESTRO_HT_MAX_TRANSITION_GAP_ATR", 0.12)
MAESTRO_HT_MIN_STRETCH_ATR = env_float("M15_MAESTRO_HT_MIN_STRETCH_ATR", 0.04)
MAESTRO_HT_MAX_STRETCH_ATR = env_float("M15_MAESTRO_HT_MAX_STRETCH_ATR", 0.90)
MAESTRO_HT_MAX_BODY_ATR = env_float("M15_MAESTRO_HT_MAX_BODY_ATR", 0.85)
MAESTRO_HT_RESET_TO_EMA9_ATR = env_float("M15_MAESTRO_HT_RESET_TO_EMA9_ATR", 0.55)

# Like V76 SCALP HEALTHY_TREND: M30 is required; H1 may be bypassed only by
# strong M15 ADX/DI dominance.
MAESTRO_HT_ALLOW_H1_EXCEPTION = env_bool("M15_MAESTRO_HT_ALLOW_H1_EXCEPTION", True)
MAESTRO_HT_H1_EXCEPTION_MIN_ADX = env_float("M15_MAESTRO_HT_H1_EXCEPTION_MIN_ADX", 23.0)
MAESTRO_HT_H1_EXCEPTION_MIN_DI_DOM = env_float("M15_MAESTRO_HT_H1_EXCEPTION_MIN_DI_DOM", 5.0)

# Lower-TF confirmation copied from MAESTRO scoring philosophy.
MAESTRO_M5_MIN_SCORE = env_int("M15_MAESTRO_M5_MIN_SCORE", 2)
MAESTRO_M1_MIN_SCORE = env_int("M15_MAESTRO_M1_MIN_SCORE", 2)
MAESTRO_IMMEDIATE_MIN_AGREE = env_int("M15_MAESTRO_IMMEDIATE_MIN_AGREE", 2)
MAESTRO_IMMEDIATE_MAX_OPPOSE = env_int("M15_MAESTRO_IMMEDIATE_MAX_OPPOSE", 1)

# Not-late/not-hostile proof, adapted from V76 MAESTRO_M15.
MAESTRO_M1_MAX_STRETCH_ATR = env_float("M15_MAESTRO_M1_MAX_STRETCH_ATR", 1.10)
MAESTRO_M1_MAX_IMPULSE_ATR = env_float("M15_MAESTRO_M1_MAX_IMPULSE_ATR", 0.55)
MAESTRO_M15_MIN_LIVE_DI_DOM = env_float("M15_MAESTRO_M15_MIN_LIVE_DI_DOM", -2.0)

# V103-like verified lane. Strict M30+H1 remains available when the full macro
# structure exists.
MAESTRO_V103_ENABLED = env_bool("M15_MAESTRO_V103_ENABLED", True)
MAESTRO_V103_MIN_ADX = env_float("M15_MAESTRO_V103_MIN_ADX", 18.0)
MAESTRO_V103_MIN_RSI = env_float("M15_MAESTRO_V103_MIN_RSI", 51.0)
MAESTRO_V103_MIN_S9_ATR = env_float("M15_MAESTRO_V103_MIN_S9_ATR", 0.030)
MAESTRO_V103_MIN_S21_ATR = env_float("M15_MAESTRO_V103_MIN_S21_ATR", 0.010)
MAESTRO_V103_PULL_TOUCH_ATR = env_float("M15_MAESTRO_V103_PULL_TOUCH_ATR", 0.35)
MAESTRO_V103_PULL_MAX_STRETCH_ATR = env_float("M15_MAESTRO_V103_PULL_MAX_STRETCH_ATR", 1.30)
MAESTRO_V103_BREAKOUT_MAX_STRETCH_ATR = env_float("M15_MAESTRO_V103_BREAKOUT_MAX_STRETCH_ATR", 1.50)
MAESTRO_V103_MIN_BODY_ATR = env_float("M15_MAESTRO_V103_MIN_BODY_ATR", 0.08)
MAESTRO_V103_MAX_BODY_ATR = env_float("M15_MAESTRO_V103_MAX_BODY_ATR", 0.95)

# MAESTRO M15 uses a wider technical stop and runner-style fail-safe target.
MAESTRO_MIN_STOP_ATR = float(ENTRY.PARAMS["maestro_min_stop_atr"])
MAESTRO_RR = float(ENTRY.PARAMS["maestro_rr"])

# --------------------------- V118.2 LOW-MAE MARKET-STATE LIBRARY ---------------------------
# Crypto translation of the MT4 V118.2 EL201/501/1001/2001 launch families.
V118_ENABLED = env_bool("M15_V118_ENABLED", True)
V118_MIN_SOURCE_RSI = env_float("M15_V118_MIN_SOURCE_RSI", 47.0)
V118_MAX_SOURCE_STRETCH_ATR = env_float("M15_V118_MAX_SOURCE_STRETCH_ATR", 1.10)

# Learning is always observational. Adaptive execution is allowed in PAPER only by default.
LEARNING_ENABLED = env_bool("M15_LEARNING_ENABLED", True)
LEARNING_APPLY_PAPER = env_bool("M15_LEARNING_APPLY_PAPER", True)
LEARNING_APPLY_LIVE = env_bool("M15_LEARNING_APPLY_LIVE", False)
LEARNING_MIN_SAMPLES = env_int("M15_LEARNING_MIN_SAMPLES", 12)
LEARNING_MIN_EXPECTANCY_CAD = env_float("M15_LEARNING_MIN_EXPECTANCY_CAD", -0.05)
LEARNING_SHADOW_BAD_LANES = env_bool("M15_LEARNING_SHADOW_BAD_LANES", True)
LEARNING_STATE_FILE = Path(os.getenv(
    "M15_LEARNING_STATE_FILE", str(LOGS_DIR / "maestro_m15_v118_learning.json")
)) if "BASE_DIR" in globals() else None

# Do not require H1 for the old LIVE_RECOVERY lane; M30 + strong local proof is
# enough, matching the healthy-trend philosophy.
LIVE_RECOVERY_STRONG_EXCEPTION_ADX = env_float("M15_LIVE_RECOVERY_STRONG_EXCEPTION_ADX", 23.0)
LIVE_RECOVERY_STRONG_EXCEPTION_DI_DOM = env_float("M15_LIVE_RECOVERY_STRONG_EXCEPTION_DI_DOM", 5.0)

# --------------------------- V7 EARLY-FILL ENTRY ---------------------------
# Prefer the liquid CAD pairs. Wide-spread symbols can still be observed and
# traded only when the live spread becomes unusually favorable.
CORE_SYMBOLS = {
    x.strip() for x in os.getenv(
        "M15_CORE_SYMBOLS", "BTC/CAD,ETH/CAD,XRP/CAD,SOL/CAD,ADA/CAD"
    ).split(",") if x.strip()
}
CORE_MAX_SPREAD_BPS = env_float("M15_CORE_MAX_SPREAD_BPS", 55.0)
NONCORE_MAX_SPREAD_BPS = env_float("M15_NONCORE_MAX_SPREAD_BPS", 38.0)

# Live recovery from a closed pullback. This is the main new lane.
LIVE_RECOVERY_ENABLED = env_bool("M15_LIVE_RECOVERY_ENABLED", True)
LIVE_RECOVERY_MIN_RSI = env_float("M15_LIVE_RECOVERY_MIN_RSI", 47.0)
LIVE_RECOVERY_MAX_RSI = env_float("M15_LIVE_RECOVERY_MAX_RSI", 64.0)
LIVE_RECOVERY_MIN_RSI_DELTA = env_float("M15_LIVE_RECOVERY_MIN_RSI_DELTA", 0.35)
LIVE_RECOVERY_MIN_ADX = env_float("M15_LIVE_RECOVERY_MIN_ADX", 15.0)
LIVE_RECOVERY_MIN_DI_RATIO = env_float("M15_LIVE_RECOVERY_MIN_DI_RATIO", 0.95)
LIVE_RECOVERY_MAX_STRETCH_ATR = env_float("M15_LIVE_RECOVERY_MAX_STRETCH_ATR", 0.75)
LIVE_RECOVERY_MAX_GAP21_50_ATR = env_float("M15_LIVE_RECOVERY_MAX_GAP21_50_ATR", 0.55)
LIVE_RECOVERY_REQUIRE_M30_UP = env_bool("M15_LIVE_RECOVERY_REQUIRE_M30_UP", True)

# Passive paper entry: arm an inside-spread limit rather than always paying the ask.
PAPER_LIMIT_ENTRY_ENABLED = bool(ENTRY.PARAMS["paper_limit_entry_enabled"])
V124_EXECUTION_PRIMARY = (str(ENTRY.PARAMS.get("v124_primary_execution_timeframe","ALL")).upper() == "ALL" or TIMEFRAME == str(ENTRY.PARAMS.get("v124_primary_execution_timeframe","ALL")))
PAPER_LIMIT_INSIDE_SPREAD_FRACTION = float(ENTRY.PARAMS["paper_limit_inside_spread_fraction"])
PAPER_LIMIT_MAX_WAIT_SEC = int(ENTRY.PARAMS["paper_limit_max_wait_sec"])
PAPER_LIMIT_MAX_CHASE_BPS = float(ENTRY.PARAMS["paper_limit_max_chase_bps"])
PAPER_LIMIT_MIN_RECLAIM_BPS = float(ENTRY.PARAMS["paper_limit_min_reclaim_bps"])

# V9 adaptive maker/taker paper execution.
PAPER_LIMIT_FALLBACK_ENABLED = bool(ENTRY.PARAMS["paper_limit_fallback_enabled"])
PAPER_LIMIT_FALLBACK_SEC = int(ENTRY.PARAMS["paper_limit_fallback_sec"])
PAPER_LIMIT_FALLBACK_MAX_SIGNAL_MOVE_BPS = float(ENTRY.PARAMS["paper_limit_fallback_max_signal_move_bps"])
PAPER_LIMIT_FALLBACK_MAX_ADVERSE_BPS = float(ENTRY.PARAMS["paper_limit_fallback_max_adverse_bps"])
PAPER_LIMIT_USE_LAST_TOUCH = bool(ENTRY.PARAMS["paper_limit_use_last_touch"])

# Lower-TF combo logic: strong M5 + strong immediate M15 can use M1 as a
# confirmation/veto rather than always requiring 2/3.
MAESTRO_STRONG_COMBO_ENABLED = env_bool("M15_MAESTRO_STRONG_COMBO_ENABLED", True)
MAESTRO_STRONG_COMBO_M5_SCORE = env_int("M15_MAESTRO_STRONG_COMBO_M5_SCORE", 3)
MAESTRO_STRONG_COMBO_M1_SCORE = env_int("M15_MAESTRO_STRONG_COMBO_M1_SCORE", 1)
MAESTRO_STANDARD_COMBO_M5_SCORE = env_int("M15_MAESTRO_STANDARD_COMBO_M5_SCORE", 2)
MAESTRO_STANDARD_COMBO_M1_SCORE = env_int("M15_MAESTRO_STANDARD_COMBO_M1_SCORE", 2)

# Fetch M5/M1 only when M15+M30 already form a plausible long candidate.
MAESTRO_PRECHECK_ENABLED = env_bool("M15_MAESTRO_PRECHECK_ENABLED", True)
MAESTRO_PRECHECK_MAX_GAP21_50_ATR = env_float("M15_MAESTRO_PRECHECK_MAX_GAP21_50_ATR", 0.80)
MAESTRO_PRECHECK_MIN_RSI = env_float("M15_MAESTRO_PRECHECK_MIN_RSI", 48.0)
MAESTRO_PRECHECK_MIN_DI_DOM = env_float("M15_MAESTRO_PRECHECK_MIN_DI_DOM", -1.0)

# Scalp exit behavior after a trade has shown real profit but momentum stalls.
STALL_EXIT_ENABLED = env_bool("M15_STALL_EXIT_ENABLED", True)
STALL_EXIT_MIN_PEAK_CAD = env_float("M15_STALL_EXIT_MIN_PEAK_CAD", 0.20)
STALL_EXIT_MIN_HOLD_SEC = env_int("M15_STALL_EXIT_MIN_HOLD_SEC", 180)
STALL_EXIT_MIN_NET_CAD = env_float("M15_STALL_EXIT_MIN_NET_CAD", 0.05)

EMERGING_WINNER_ENABLED = env_bool("M15_EMERGING_WINNER_ENABLED", True)
EMERGING_WINNER_MIN_PEAK_CAD = env_float("M15_EMERGING_WINNER_MIN_PEAK_CAD", 0.10)
EMERGING_WINNER_MAX_PEAK_CAD = env_float("M15_EMERGING_WINNER_MAX_PEAK_CAD", 0.30)
EMERGING_WINNER_KEEP_FRACTION = env_float("M15_EMERGING_WINNER_KEEP_FRACTION", 0.55)
EMERGING_WINNER_STALL_SEC = env_int("M15_EMERGING_WINNER_STALL_SEC", 35)

# Execution economics.
MAX_SPREAD_BPS = env_float("M15_MAX_SPREAD_BPS_DEFAULT", 65.0)
MAX_SPREAD_ATR_RATIO = env_float("M15_MAX_SPREAD_ATR_RATIO", 3.00)
ESTIMATED_TAKER_FEE_PCT = env_float("M15_ESTIMATED_TAKER_FEE_PCT", 0.20) / 100.0
PAPER_SLIPPAGE_BPS = env_float("M15_PAPER_SLIPPAGE_BPS", 4.0)
SCALP_MAX_ROUNDTRIP_FRICTION_CAD = env_float("M15_SCALP_MAX_ROUNDTRIP_FRICTION_CAD", 1.75)
SCALP_MIN_TARGET_TO_FRICTION = env_float("M15_SCALP_MIN_TARGET_TO_FRICTION", 1.10)

SCALP_MIN_NET_TARGET_CAD = env_float("M15_SCALP_MIN_NET_TARGET_CAD", 0.20)
SCALP_PREFERRED_SPREAD_BPS = env_float("M15_SCALP_PREFERRED_SPREAD_BPS", 20.0)
SCALP_HIGH_SPREAD_MIN_NET_TARGET_CAD = env_float("M15_SCALP_HIGH_SPREAD_MIN_NET_TARGET_CAD", 0.45)

# Forex SCALP stop/target style.
SCALP_MIN_STOP_ATR = env_float("M15_SCALP_MIN_STOP_ATR", 0.70)
SCALP_SWING_LOOKBACK = env_int("M15_SCALP_SWING_LOOKBACK", 5)
SCALP_SWING_BUFFER_ATR = env_float("M15_SCALP_SWING_BUFFER_ATR", 0.10)
SCALP_RR = env_float("M15_SCALP_RR", 1.80)

# ============================================================
# EXIT / PROFIT PROTECTION
# ============================================================
BREAK_EVEN_TRIGGER_R = env_float("M15_BREAK_EVEN_TRIGGER_R", 0.75)
BREAK_EVEN_LOCK_R = env_float("M15_BREAK_EVEN_LOCK_R", 0.10)

HARD_CASH_STOP_ENABLED = env_bool("M15_ENABLE_HARD_CASH_STOP", True)
HARD_CASH_STOP_CAD = min(env_float("M15_HARD_CASH_LOSS_CAD", 3.00), 3.00)

FAST_FAILURE_SECONDS = env_int("M15_FAST_FAILURE_SECONDS", 300)
FAST_FAILURE_R = env_float("M15_FAST_FAILURE_R", 0.65)

FAILED_ENTRY_GRACE_SECONDS = env_int("M15_FAILED_ENTRY_GRACE_SECONDS", 240)
FAILED_ENTRY_MAX_MFE_R = env_float("M15_FAILED_ENTRY_MAX_MFE_R", 0.08)
FAILED_ENTRY_EXIT_R = env_float("M15_FAILED_ENTRY_EXIT_R", 0.35)
FAILED_ENTRY_REQUIRE_2OF3_BEARISH = env_bool("M15_FAILED_ENTRY_REQUIRE_2OF3_BEARISH", False)

CAD_PROTECT_START = env_float("M15_CAD_PROFIT_PROTECT_START", 0.30)
CAD_GIVEBACK = env_float("M15_CAD_PROFIT_GIVEBACK", 0.20)
CAD_STRONG_PEAK = env_float("M15_CAD_STRONG_PEAK", 0.65)
CAD_STRONG_LOCK = env_float("M15_CAD_STRONG_LOCK", 0.20)
CAD_RUNNER_PEAK = env_float("M15_CAD_RUNNER_PEAK", 1.00)
CAD_RUNNER_LOCK = env_float("M15_CAD_RUNNER_LOCK", 0.50)
CAD_BIG_PEAK = env_float("M15_CAD_BIG_PEAK", 2.00)
CAD_BIG_LOCK = env_float("M15_CAD_BIG_LOCK", 1.20)

PEAK_PROTECT_ENABLED = env_bool("M15_PEAK_PROTECT_ENABLED", True)
PEAK_PROTECT_MIN_CAD = env_float("M15_PEAK_PROTECT_MIN_CAD", 0.30)
PEAK_PROTECT_FRACTION = max(0.65, min(env_float("M15_PEAK_PROTECT_FRACTION", 0.65), 0.95))
PEAK_PROTECT_MIN_FLOOR_CAD = env_float("M15_PEAK_PROTECT_MIN_FLOOR_CAD", 0.20)

MAX_HOLD_MINUTES = float(ENTRY.PARAMS["max_hold_minutes"])
STALE_MAX_R = float(ENTRY.PARAMS["stale_max_r"])

POLL_SECONDS = env_int("M15_POLL_SECONDS", 5)
HEARTBEAT_SECONDS = env_int("M15_HEARTBEAT_SECONDS", 60)
WAIT_LOG_SECONDS = env_int("M15_WAIT_LOG_SECONDS", 20)
BAD_SYMBOL_COOLDOWN_SECONDS = env_int("M15_BAD_SYMBOL_COOLDOWN_SECONDS", 1800)

# ============================================================
# FILES
# ============================================================
BASE_DIR = Path(os.getenv("M15_BASE_DIR", r"C:\NDAX_BOT"))
LOGS_DIR = Path(os.getenv("MAESTRO_LOGS_DIR", str(BASE_DIR / "Logs")))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR = Path(os.getenv("MAESTRO_JSON_DIR", str(BASE_DIR / "Json")))
JSON_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = Path(os.getenv("M5_STATE_FILE", str(JSON_DIR / "maestro_m5_state.json")))
LOG_FILE = Path(os.getenv("M5_LOG_FILE", str(LOGS_DIR / "maestro_m5.log")))
EVENTS_FILE = Path(os.getenv("M5_EVENTS_FILE", str(JSON_DIR / "maestro_m5_events.jsonl")))
TRADES_CSV = Path(os.getenv("M5_TRADES_CSV", str(LOGS_DIR / "maestro_m5_trades.csv")))
SIGNALS_CSV = Path(os.getenv("M5_SIGNALS_CSV", str(LOGS_DIR / "maestro_m5_signals.csv")))
ORDERS_FILE = Path(os.getenv("M5_ORDERS_FILE", str(LOGS_DIR / "orders_m5.txt")))
OPPORTUNITIES_FILE = Path(os.getenv("M5_OPPORTUNITIES_FILE", str(JSON_DIR / "maestro_m5_opportunities.jsonl")))
MARKET_RESEARCH_FILE = Path(os.getenv("M5_MARKET_RESEARCH_FILE", str(JSON_DIR / "maestro_m5_market_research.jsonl")))

LEARNING_STATE_FILE = Path(os.getenv(
    "M15_LEARNING_STATE_FILE", str(LOGS_DIR / "maestro_m15_v118_learning.json")
))

for fp in (STATE_FILE, LOG_FILE, EVENTS_FILE, TRADES_CSV, SIGNALS_CSV, ORDERS_FILE, OPPORTUNITIES_FILE, MARKET_RESEARCH_FILE, LEARNING_STATE_FILE):
    fp.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# GENERAL HELPERS
# ============================================================
def utc_now():
    return datetime.now(timezone.utc)

def utc_iso():
    return utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")

def log(msg):
    line = f"{utc_iso()} | {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def event(kind, **kwargs):
    with EVENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc_iso(), "event": kind, **kwargs}, default=str) + "\n")

def append_csv(path, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def append_order_line(text):
    with ORDERS_FILE.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


# ============================================================
# STATE
# ============================================================
def default_state():
    return {
        "account_id": None,
        "paper_equity_cad": PAPER_START_BALANCE_CAD,
        "day": utc_now().strftime("%Y-%m-%d"),
        "daily_pnl": 0.0,
        "trades_today": 0,
        "wins": 0,
        "losses": 0,
        "consecutive_losses": 0,
        "pause_until": 0.0,
        "positions": {},
        "last_entry_bar": {},
        "symbol_cooldown_until": {},
        "bad_symbol_until": {},
        "last_wait_log": {},
        "last_heartbeat": 0.0,
        "path_watch": [],
        "tracked_reject_bar": {},
        "early_reclaim": {},
        "last_entry_ts": {},
        "pending_limits": {},
    }

def load_state():
    if not STATE_FILE.exists():
        return default_state()
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        st = default_state()
        st.update(raw)
        return st
    except Exception:
        return default_state()

def append_opportunity_research(record):
    try:
        payload = dict(record or {})
        payload.setdefault("utc", utc_iso())
        payload.setdefault("bot_version", BOT_VERSION)
        payload.setdefault("timeframe", TIMEFRAME)
        with OPPORTUNITIES_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception as exc:
        log(f"OPPORTUNITY_LOG_ERROR | {exc}")

def v120_opportunity_record(symbol, diag, bid, ask, status, block_reason=""):
    diag = dict(diag or {})
    px = (float(bid)+float(ask))/2.0
    sts = diag.get("bar_ts") or int(time.time())
    source = diag.get("entry_mode") or diag.get("source_mode") or diag.get("reason") or "UNKNOWN"
    oid = ENTRY.opportunity_id(symbol,"BUY",source,sts,px)
    append_opportunity_research({
        "event":"V120_OPPORTUNITY","opportunity_id":oid,"symbol":symbol,"side":"BUY",
        "status":status,"block_reason":block_reason,"source_mode":source,
        "signal_ts":sts,"signal_price":px,"bid":float(bid),"ask":float(ask),
        "spread_bps":spread_bps(float(bid),float(ask)),
        "research_spec":ENTRY.opportunity_research_spec(),"market_state":diag
    })
    return oid

def append_market_research(record):
    try:
        payload=dict(record or {})
        payload.setdefault("utc",utc_iso()); payload.setdefault("bot_version",BOT_VERSION)
        payload.setdefault("bot_timeframe",TIMEFRAME)
        with MARKET_RESEARCH_FILE.open("a",encoding="utf-8") as fh:
            fh.write(json.dumps(payload,default=str)+"\n")
    except Exception as exc: log(f"MARKET_RESEARCH_LOG_ERROR | {exc}")

def discover_all_cad_spot_symbols(exchange):
    try:
        markets=exchange.load_markets()
        return sorted(set(sym for sym,m in markets.items()
            if m.get("active",True) and m.get("spot") is not False
            and str(m.get("quote","")).upper()==str(ENTRY.PARAMS["scan_quote"]).upper()))
    except Exception as exc:
        log(f"MARKET_DISCOVERY_ERROR | {exc}"); return []

def v120_market_snapshot(exchange,symbol):
    frames={}
    for tf0 in ENTRY.PARAMS["scan_timeframes"]:
        try:
            raw=exchange.fetch_ohlcv(symbol,tf0,limit=80)
            if not raw or len(raw)<25: continue
            df=add_indicators(pd.DataFrame(raw,columns=["timestamp","open","high","low","close","volume"]))
            c=df.iloc[-2] if len(df)>1 else df.iloc[-1]
            frames[tf0]={"ts":int(c["timestamp"]),"o":float(c["open"]),"h":float(c["high"]),"l":float(c["low"]),
              "c":float(c["close"]),"v":float(c["volume"]),"ema9":float(c["ema9"]),"ema21":float(c["ema21"]),
              "ema50":float(c["ema50"]),"rsi14":float(c["rsi14"]),"atr14":float(c["atr14"]),
              "adx14":float(c["adx14"]),"plus_di":float(c["plus_di"]),"minus_di":float(c["minus_di"]),
              "macd_hist":float(c["macd_hist"])}
        except Exception: continue
    try:
        t=exchange.fetch_ticker(symbol); bid=float(t.get("bid") or 0); ask=float(t.get("ask") or 0); last=float(t.get("last") or 0)
        px=(bid+ask)/2 if bid>0 and ask>0 else last
    except Exception: return None
    if px<=0 or not frames: return None
    return {"symbol":symbol,"price":px,"bid":bid,"ask":ask,
      "spread_bps":spread_bps(bid,ask) if bid>0 and ask>0 else None,"frames":frames}

def v120_scan_all_markets(exchange,state):
    """Incremental full-market research scanner so research cannot block the trading loop."""
    now=time.time()
    interval=int(ENTRY.PARAMS["scan_interval_sec"])
    if now-float(state.get("v120_market_scan_ts",0)) < interval:
        return

    symbols=discover_all_cad_spot_symbols(exchange)
    if not symbols:
        state["v120_market_scan_ts"]=now
        save_state(state)
        return

    cursor=int(state.get("v120_market_scan_cursor",0)) % len(symbols)
    batch_size=2
    batch=[symbols[(cursor+i) % len(symbols)] for i in range(min(batch_size,len(symbols)))]
    completed=0
    for sym in batch:
        snap=v120_market_snapshot(exchange,sym)
        if not snap:
            continue
        snap.update({"event":"V120_MARKET_STATE",
          "research_id":ENTRY.opportunity_id(sym,"BUY","FULL_MARKET_SCAN",int(now),snap["price"]),
          "scan_universe_count":len(symbols),"checkpoints_sec":list(ENTRY.PARAMS["scan_checkpoint_sec"]),
          "target_return_pct":list(ENTRY.PARAMS["scan_target_return_pct"]),
          "missed_win_min_mfe_pct":float(ENTRY.PARAMS["missed_win_min_mfe_pct"]),
          "missed_win_max_mae_pct":float(ENTRY.PARAMS["missed_win_max_mae_pct"])})
        append_market_research(snap)
        completed += 1

    state["v120_market_scan_cursor"]=(cursor+len(batch)) % len(symbols)
    state["v120_market_scan_ts"]=now
    state["v120_market_universe_count"]=len(symbols)
    save_state(state)
    log(f"RESEARCH_SCAN | batch={completed}/{len(batch)} | full_universe={len(symbols)} | next_cursor={state['v120_market_scan_cursor']}")

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

def reset_day(state):
    d = utc_now().strftime("%Y-%m-%d")
    if state.get("day") != d:
        state["day"] = d
        state["daily_pnl"] = 0.0
        state["trades_today"] = 0
        state["consecutive_losses"] = 0
        state["pause_until"] = 0.0
        save_state(state)



def _track_minutes():
    vals = []
    for raw in str(SCALP_TRACK_MINUTES).split(","):
        try:
            vals.append(int(raw.strip()))
        except Exception:
            pass
    return tuple(sorted(set(v for v in vals if v > 0)))


def add_path_watch(state, symbol, anchor_price, kind, reason, bar_ts=None):
    if anchor_price <= 0:
        return
    state.setdefault("path_watch", []).append({
        "symbol": symbol,
        "anchor_ts": time.time(),
        "anchor_price": float(anchor_price),
        "kind": str(kind),
        "reason": str(reason),
        "bar_ts": bar_ts,
        "done": [],
        "mfe_pct": 0.0,
        "mae_pct": 0.0,
    })
    if len(state["path_watch"]) > SCALP_MAX_TRACK_WATCHES:
        state["path_watch"] = state["path_watch"][-SCALP_MAX_TRACK_WATCHES:]
    save_state(state)


def process_path_watch(state, symbol, bid, ask, live):
    watches = state.setdefault("path_watch", [])
    if not watches:
        return
    now = time.time()
    mid = (bid + ask) / 2.0
    c = live.iloc[-1]
    ema9 = float(c["ema9"]); ema21 = float(c["ema21"]); ema50 = float(c["ema50"])
    rsi_v = float(c["rsi14"]); macd_v = float(c["macd_hist"])
    cps = _track_minutes()
    changed = False

    for w in watches:
        if w.get("symbol") != symbol:
            continue
        anchor = float(w.get("anchor_price", 0.0))
        if anchor <= 0:
            continue
        move_pct = (mid - anchor) / anchor * 100.0
        w["mfe_pct"] = max(float(w.get("mfe_pct", 0.0)), move_pct)
        w["mae_pct"] = min(float(w.get("mae_pct", 0.0)), move_pct)
        elapsed = (now - float(w["anchor_ts"])) / 60.0
        done = set(w.get("done", []))
        for cp in cps:
            if elapsed >= cp and cp not in done:
                log(
                    f"PATH_TRACK | {w.get('kind')} | {symbol} | +{cp}m | "
                    f"reason={w.get('reason')} | anchor={anchor:.8f} | now={mid:.8f} | "
                    f"move={move_pct:+.3f}% | MFE={float(w['mfe_pct']):+.3f}% | "
                    f"MAE={float(w['mae_pct']):+.3f}% | RSI={rsi_v:.1f} | "
                    f"MACD={macd_v:+.8f} | EMA9/21/50={ema9:.8f}/{ema21:.8f}/{ema50:.8f}"
                )
                done.add(cp)
                changed = True
        w["done"] = sorted(done)

    max_cp = max(cps or (60,))
    state["path_watch"] = [
        w for w in watches
        if len(set(w.get("done", []))) < len(cps)
        or (now - float(w["anchor_ts"])) < (max_cp + 10) * 60
    ]
    if changed:
        save_state(state)



def arm_early_reclaim(state, symbol, mid, diag, q):
    if not EARLY_RECLAIM_ENABLED:
        return
    state.setdefault("early_reclaim", {})[symbol] = {
        "armed_ts": time.time(),
        "anchor": float(mid),
        "low": float(mid),
        "high": float(mid),
        "confirm_hits": 0,
        "bar_ts": diag.get("bar_ts"),
        "diag": dict(diag),
        "last_log": 0.0,
    }
    log(
        f"EARLY_RECLAIM_ARMED | {symbol} | anchor={mid:.8f} | "
        f"mode={diag.get('entry_mode')} | score={q.get('score')}/4 | "
        f"opp={q.get('opp_score')}/4 | immediate={q.get('agree')}/3"
    )
    save_state(state)


def cancel_early_reclaim(state, symbol, reason, mid):
    w = state.get("early_reclaim", {}).get(symbol)
    if not w:
        return
    log(
        f"EARLY_RECLAIM_CANCEL | {symbol} | {reason} | "
        f"anchor={float(w['anchor']):.8f} | now={mid:.8f}"
    )
    del state["early_reclaim"][symbol]
    save_state(state)


def process_early_reclaim(exchange, state, symbol, bid, ask, spr, live, m15):
    w = state.get("early_reclaim", {}).get(symbol)
    if not w:
        return None

    mid = (bid + ask) / 2.0
    age = time.time() - float(w["armed_ts"])
    anchor = float(w["anchor"])
    w["low"] = min(float(w.get("low", anchor)), mid)
    w["high"] = max(float(w.get("high", anchor)), mid)

    move_bps = (mid - anchor) / anchor * 10000.0
    adverse_bps = (float(w["low"]) - anchor) / anchor * 10000.0

    lc = live.iloc[-1]
    atrv = float(lc["atr14"])
    ema9 = float(lc["ema9"]); ema21 = float(lc["ema21"])
    pdi = float(lc["plus_di"]); mdi = float(lc["minus_di"])
    rsi_v = float(lc["rsi14"]); macd_v = float(lc["macd_hist"])
    di_ratio = pdi / max(mdi, 1e-9)
    score = direction_score_live(live)
    opp = opposite_score_live(live)
    stretch = max(0.0, (mid - ema21) / atrv) if atrv > 0 else 999.0

    # V127 telemetry-derived timing repair:
    # both latest XRP EARLY_RECLAIM trades entered before a meaningful retest,
    # went adverse, then rallied strongly only after the adverse-wall exit.
    # Require the watch to experience a real pullback first, then reclaim.
    v127_retest_bps = -5.0
    if adverse_bps <= v127_retest_bps:
        w["v127_saw_retest"] = True
    positive_now = (
        bool(w.get("v127_saw_retest", False))
        and move_bps >= EARLY_RECLAIM_MIN_MOVE_BPS
        and move_bps <= EARLY_RECLAIM_MAX_MOVE_BPS
        and score >= EARLY_RECLAIM_MIN_SCORE
        and opp <= EARLY_RECLAIM_MAX_OPP_SCORE
        and di_ratio >= EARLY_RECLAIM_MIN_DI_RATIO
        and rsi_v >= EARLY_RECLAIM_MIN_RSI
        and macd_v >= 0.0
        and ema9 >= ema21
        and mid >= ema21
        and stretch <= EARLY_RECLAIM_MAX_STRETCH_ATR
        and spr <= MAX_SPREAD_BPS
    )
    if positive_now:
        w["confirm_hits"] = int(w.get("confirm_hits", 0)) + 1
    else:
        w["confirm_hits"] = max(0, int(w.get("confirm_hits", 0)) - 1)

    if time.time() - float(w.get("last_log", 0.0)) >= 25:
        log(
            f"EARLY_RECLAIM_WAIT | {symbol} | age={age:.0f}s | "
            f"move={move_bps:+.1f}bps | adverse={adverse_bps:+.1f}bps | "
            f"score={score}/4 opp={opp}/4 | DIratio={di_ratio:.2f} | "
            f"RSI={rsi_v:.1f} | stretch={stretch:.2f}ATR | hits={w['confirm_hits']} | "
            f"v127_retest={int(bool(w.get('v127_saw_retest', False)))}"
        )
        w["last_log"] = time.time()

    if adverse_bps <= -EARLY_RECLAIM_MAX_ADVERSE_BPS:
        cancel_early_reclaim(state, symbol, "ADVERSE_MOVE", mid)
        return None
    if age >= EARLY_RECLAIM_MAX_AGE_SEC:
        cancel_early_reclaim(state, symbol, "TIMEOUT", mid)
        return None
    if move_bps > EARLY_RECLAIM_MAX_MOVE_BPS:
        cancel_early_reclaim(state, symbol, "CHASED_TOO_FAR", mid)
        return None
    if age < EARLY_RECLAIM_MIN_AGE_SEC:
        save_state(state)
        return None
    v127_required_hits = max(2, int(EARLY_RECLAIM_CONFIRM_HITS))
    if int(w.get("confirm_hits", 0)) < v127_required_hits:
        save_state(state)
        return None

    diag = dict(w["diag"])
    diag["reason"] = "ENTRY_READY_EARLY_RECLAIM"
    diag["quality_reason"] = "ENTRY_READY_EARLY_RECLAIM"
    diag["entry_mode"] = "EARLY_RECLAIM_" + str(diag.get("entry_mode") or "SETUP")
    diag["score"] = score
    diag["opp_score"] = opp
    diag["imm_agree"] = 2
    diag["imm_oppose"] = 0
    diag["rsi_delta"] = 0.0
    diag["macd_delta"] = 0.0
    diag["move_atr"] = (mid - float(m15.iloc[-1]["close"])) / max(float(m15.iloc[-1]["atr14"]), 1e-9)
    diag["live_swing"] = "UP"
    diag["di_dom"] = pdi - mdi
    diag["stretch_atr"] = stretch

    log(
        f"EARLY_RECLAIM_READY | {symbol} | anchor={anchor:.8f} | now={mid:.8f} | "
        f"move={move_bps:+.1f}bps | score={score}/4 | DIratio={di_ratio:.2f} | "
        f"RSI={rsi_v:.1f} | stretch={stretch:.2f}ATR"
    )
    del state["early_reclaim"][symbol]
    save_state(state)
    return diag


# ============================================================
# NDAX / AUTH - SAME STYLE AS PRIOR MAESTRO BOTS
# ============================================================
def build_exchange():
    cfg = {"enableRateLimit": True, "timeout": 20_000}
    if NDAX_API_KEY:
        cfg["apiKey"] = NDAX_API_KEY
    if NDAX_API_SECRET:
        cfg["secret"] = NDAX_API_SECRET
    if NDAX_USER_ID:
        cfg["uid"] = NDAX_USER_ID
    exchange = ccxt.ndax(cfg)
    try:
        exchange.requiredCredentials["login"] = False
        exchange.requiredCredentials["password"] = False
    except Exception:
        pass
    exchange.load_markets()
    return exchange

def discover_account_id(exchange, state):
    if NDAX_ACCOUNT_ID_ENV.isdigit():
        account_id = int(NDAX_ACCOUNT_ID_ENV)
    elif str(state.get("account_id") or "").isdigit():
        account_id = int(state["account_id"])
    else:
        if not NDAX_USER_ID:
            raise RuntimeError("NDAX_USER_ID is required for account discovery")
        response = exchange.privateGetGetUserAccounts({"omsId": 1, "UserId": NDAX_USER_ID, "UserName": ""})
        ids = []
        for item in response or []:
            val = item.get("AccountId") if isinstance(item, dict) else item
            if str(val).isdigit():
                ids.append(int(val))
        if not ids:
            raise RuntimeError("GetUserAccounts returned no valid AccountId")
        account_id = ids[0]
    state["account_id"] = account_id
    save_state(state)
    return account_id

def fetch_private_balances(exchange, account_id):
    response = exchange.privateGetGetAccountPositions({
        "omsId": 1, "AccountId": int(account_id)
    })
    out = {}
    for item in response or []:
        sym = item.get("ProductSymbol") or item.get("productSymbol")
        if not sym:
            continue
        amount = float(item.get("Amount") or item.get("amount") or 0.0)
        hold = float(item.get("Hold") or item.get("hold") or 0.0)
        out[str(sym).upper()] = {
            "total": amount, "hold": hold, "free": max(0.0, amount - hold)
        }
    return out

def readonly_account_ping(exchange, state):
    try:
        aid = discover_account_id(exchange, state)
        balances = fetch_private_balances(exchange, aid)
        free_cad = float(balances.get("CAD", {}).get("free", 0.0))
        nonzero = sum(
            1 for k, v in balances.items()
            if k != "CAD" and float(v.get("total", 0.0)) != 0.0
        )
        log(
            f"ACCOUNT_PING_OK | AccountId={aid} | free_CAD=C${free_cad:.2f} | "
            f"nonzero_assets={nonzero} | READ_ONLY"
        )
    except Exception as exc:
        log(f"ACCOUNT_PING_WARN | private read failed; paper engine can still run | {exc}")

def choose_universe(exchange):
    if not AUTO_DISCOVER:
        return [s for s in SYMBOLS if s in exchange.markets]

    # V120.3: use the complete active CAD spot universe.
    # Do not rank every market with fetch_ticker() during startup; that delayed/stalled
    # the engine before the normal evaluation loop could begin.
    candidates = []
    for sym, market in exchange.markets.items():
        base = str(market.get("base", "")).upper()
        quote = str(market.get("quote", "")).upper()
        if (
            market.get("active", True)
            and market.get("spot", True)
            and quote == AUTO_QUOTE
            and base not in {"CAD", "USD", "USDC", "USDT"}
        ):
            candidates.append(sym)
    return sorted(set(candidates))


# ============================================================
# INDICATORS
# ============================================================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)

def true_range(df):
    prev = df["close"].shift(1)
    return pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)

def add_indicators(df):
    x = df.copy()
    x["ema9"] = ema(x["close"], 9)
    x["ema21"] = ema(x["close"], 21)
    x["ema50"] = ema(x["close"], 50)
    x["rsi14"] = rsi(x["close"], 14)
    tr = true_range(x)
    x["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    up = x["high"].diff()
    down = -x["low"].diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=x.index)
    mdm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=x.index)
    atrw = x["atr14"].replace(0, np.nan)
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atrw
    mdi = 100 * mdm.ewm(alpha=1/14, adjust=False).mean() / atrw
    dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
    x["plus_di"] = pdi.fillna(0.0)
    x["minus_di"] = mdi.fillna(0.0)
    x["adx14"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0.0)

    mac = ema(x["close"], 12) - ema(x["close"], 26)
    sig = ema(mac, 9)
    x["macd_hist"] = mac - sig
    # V118.1+ learning fields: Bollinger geometry, EMA20 state and volume ratio.
    x["ema20"] = ema(x["close"], 20)
    bb_mid = x["close"].rolling(20).mean()
    bb_std = x["close"].rolling(20).std(ddof=0)
    x["bb_mid"] = bb_mid
    x["bb_upper"] = bb_mid + 2.0 * bb_std
    x["bb_lower"] = bb_mid - 2.0 * bb_std
    x["vol_avg10"] = x["volume"].rolling(10).mean()
    x["vol_ratio"] = (x["volume"] / x["vol_avg10"].replace(0, np.nan)).fillna(0.0)
    return x

def fetch_closed(exchange, symbol, timeframe, limit):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw or len(raw) < 60:
        raise RuntimeError(f"Insufficient {timeframe} candles")
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df = df.iloc[:-1].copy()  # closed bars only
    return add_indicators(df).reset_index(drop=True)

def fetch_live_m15(exchange, symbol):
    raw = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)
    if not raw or len(raw) < 60:
        raise RuntimeError("Insufficient live M15 candles")
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    return add_indicators(df).reset_index(drop=True)


def fetch_live_tf(exchange, symbol, timeframe, limit=120):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw or len(raw) < 60:
        raise RuntimeError(f"Insufficient live {timeframe} candles")
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    return add_indicators(df).reset_index(drop=True)



# ============================================================
# EXECUTABLE QUOTE
# ============================================================
def spread_bps(bid, ask):
    if bid <= 0 or ask <= bid:
        return float("nan")
    return (ask - bid) / ((ask + bid) / 2.0) * 10000.0

def configure_entry_module():
    ENTRY.configure({
        "utc_iso":utc_iso,"event":event,"log":log,"spread_bps":spread_bps,
        "PAPER_MODE":PAPER_MODE,"TRADES_CSV":TRADES_CSV,"BOT_VERSION":BOT_VERSION,
        "BASE_DIR":BASE_DIR,"MAESTRO_MIN_STOP_ATR":MAESTRO_MIN_STOP_ATR,"MAESTRO_RR":MAESTRO_RR
    })

def fetch_quote(exchange, symbol):
    ticker_bid = ticker_ask = 0.0
    try:
        t = exchange.fetch_ticker(symbol)
        ticker_bid = float(t.get("bid") or 0.0)
        ticker_ask = float(t.get("ask") or 0.0)
    except Exception:
        pass

    try:
        book = exchange.fetch_order_book(symbol, limit=5)
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if bids and asks:
            bid = float(bids[0][0])
            ask = float(asks[0][0])
            if bid > 0 and ask > bid:
                return bid, ask, spread_bps(bid, ask), "ORDER_BOOK"
    except Exception:
        pass

    if ticker_bid > 0 and ticker_ask > ticker_bid:
        return ticker_bid, ticker_ask, spread_bps(ticker_bid, ticker_ask), "TICKER"
    raise RuntimeError("No valid executable quote")


# ============================================================
# SCALP CURR SIGNAL TRANSLATION
# ============================================================
def direction_score_live(live):
    c = live.iloc[-1]
    score = 0
    if float(c["ema9"]) > float(c["ema21"]): score += 1
    if float(c["plus_di"]) > float(c["minus_di"]): score += 1
    if float(c["rsi14"]) >= 52.0: score += 1
    if float(c["macd_hist"]) > 0.0: score += 1
    return score

def opposite_score_live(live):
    c = live.iloc[-1]
    score = 0
    if float(c["ema9"]) < float(c["ema21"]): score += 1
    if float(c["minus_di"]) > float(c["plus_di"]): score += 1
    if float(c["rsi14"]) <= 48.0: score += 1
    if float(c["macd_hist"]) < 0.0: score += 1
    return score

def scalp_closed_setup(m15, m30, h1):
    c = m15.iloc[-1]
    p = m15.iloc[-2]
    atrv = float(c["atr14"])
    if atrv <= 0:
        return False, {"reason":"ATR_INVALID"}

    c1 = float(c["close"]); o1 = float(c["open"])
    h1c = float(c["high"]); l1 = float(c["low"])
    e9 = float(c["ema9"]); e21 = float(c["ema21"]); e50 = float(c["ema50"])
    e9p = float(p["ema9"]); e21p = float(p["ema21"])
    s9 = (e9 - e9p) / atrv
    s21 = (e21 - e21p) / atrv
    r1 = float(c["rsi14"]); r2 = float(p["rsi14"])
    adxv = float(c["adx14"])
    pdi = float(c["plus_di"]); mdi = float(c["minus_di"])
    mac = float(c["macd_hist"])

    m30c = m30.iloc[-1]
    h1row = h1.iloc[-1]
    m30_bull = float(m30c["close"]) > float(m30c["ema50"])
    h1_bull = float(h1row["close"]) > float(h1row["ema50"])

    stretch = abs(c1 - e21) / atrv
    touch = SCALP_PULL_TOUCH_ATR * atrv

    buy_trend = (
        e9 > e21 > e50
        and s9 > 0.0
        and s21 >= -0.01
        and m30_bull
        and h1_bull
    )

    gap21_50_atr = (e21 - e50) / atrv
    di_ratio = pdi / max(mdi, 1e-9)

    # V2 controlled transition: EMA9 already above EMA21 and both rising,
    # but EMA21 may still be slightly below EMA50. This is deliberately
    # narrower than a generic "macro transition" lane.
    transition_macro_ok = (m30_bull and h1_bull) if SCALP_TRANSITION_REQUIRE_M30_H1 else (m30_bull or h1_bull)
    buy_transition = (
        SCALP_TRANSITION_ENABLED
        and not buy_trend
        and e9 > e21
        and s9 >= SCALP_TRANSITION_MIN_S9_ATR
        and s21 >= SCALP_TRANSITION_MIN_S21_ATR
        and transition_macro_ok
        and r1 >= SCALP_TRANSITION_MIN_RSI
        and adxv >= SCALP_TRANSITION_MIN_ADX
        and di_ratio >= SCALP_TRANSITION_MIN_DI_RATIO
        and gap21_50_atr >= -SCALP_TRANSITION_MAX_GAP21_50_ATR
        and abs(c1 - e21) / atrv <= SCALP_TRANSITION_MAX_STRETCH_ATR
        and mac > 0.0
    )

    opportunity_macro_ok = (
        (m30_bull or h1_bull)
        if SCALP_OPPORTUNITY_REQUIRE_ONE_MACRO_UP
        else True
    )
    buy_opportunity = (
        SCALP_OPPORTUNITY_ENABLED
        and not buy_trend
        and not buy_transition
        and e9 > e21
        and s9 >= SCALP_OPPORTUNITY_MIN_S9_ATR
        and s21 >= SCALP_OPPORTUNITY_MIN_S21_ATR
        and opportunity_macro_ok
        and r1 >= SCALP_OPPORTUNITY_MIN_RSI
        and adxv >= SCALP_OPPORTUNITY_MIN_ADX
        and di_ratio >= SCALP_OPPORTUNITY_MIN_DI_RATIO
        and gap21_50_atr >= -SCALP_OPPORTUNITY_MAX_GAP21_50_ATR
        and stretch <= SCALP_OPPORTUNITY_MAX_STRETCH_ATR
        and mac >= 0.0
    )

    buy_pull = (
        l1 <= e9 + touch
        and c1 >= e9
        and c1 > o1
        and stretch <= SCALP_PULL_MAX_STRETCH_ATR
    )

    buy_pull_recovery = (
        SCALP_RECOVERY_PULLBACK_ENABLED
        and l1 <= e9 + touch
        and c1 >= e21
        and stretch <= SCALP_RECOVERY_MAX_STRETCH_ATR
        and di_ratio >= SCALP_RECOVERY_MIN_DI_RATIO
        and (m30_bull or h1_bull)
    )
    buy_cont = (
        c1 > e9
        and r1 > SCALP_CONT_MIN_RSI
        and r1 >= r2
        and pdi > mdi
        and mac > 0.0
        and stretch <= SCALP_CONT_MAX_STRETCH_ATR
    )

    lb = max(2, min(SCALP_BREAKOUT_LOOKBACK_BARS, len(m15) - 2))
    prior_range_high = max(float(m15.iloc[-i]["high"]) for i in range(2, lb + 2))
    breakout_level = prior_range_high + SCALP_BREAKOUT_BUFFER_ATR * atrv
    breakout_body_atr = abs(c1 - o1) / atrv
    buy_breakout = (
        SCALP_BREAKOUT_ENABLED
        and c1 >= breakout_level
        and c1 > o1
        and breakout_body_atr >= SCALP_BREAKOUT_MIN_BODY_ATR
        and r1 >= SCALP_BREAKOUT_MIN_RSI
        and adxv >= SCALP_BREAKOUT_MIN_ADX
        and di_ratio >= SCALP_BREAKOUT_MIN_DI_RATIO
        and stretch <= SCALP_BREAKOUT_MAX_STRETCH_ATR
        and mac > 0.0
        and (m30_bull or h1_bull)
    )

    setup = (
        ((buy_trend or buy_transition or buy_opportunity) and (buy_pull or buy_pull_recovery or buy_cont))
        or buy_breakout
    )

    if buy_breakout:
        mode = "BREAKOUT"
    elif buy_transition and buy_pull:
        mode = "TRANSITION_PULLBACK"
    elif buy_transition and buy_pull_recovery:
        mode = "TRANSITION_RECOVERY_PULLBACK"
    elif buy_transition and buy_cont:
        mode = "TRANSITION_CONTINUATION"
    elif buy_opportunity and buy_pull:
        mode = "OPPORTUNITY_PULLBACK"
    elif buy_opportunity and buy_pull_recovery:
        mode = "OPPORTUNITY_RECOVERY_PULLBACK"
    elif buy_opportunity and buy_cont:
        mode = "OPPORTUNITY_CONTINUATION"
    elif buy_pull_recovery:
        mode = "RECOVERY_PULLBACK"
    else:
        mode = "PULLBACK" if buy_pull else ("CONTINUATION" if buy_cont else "-")

    body_atr = abs(c1 - o1) / atrv
    gap_atr = abs(e9 - e21) / atrv
    raw_cross = e9 > e21 and e9p <= e21p
    weak_cross = (
        raw_cross
        and adxv < SCALP_WEAK_CROSS_ADX
        and (gap_atr < SCALP_WEAK_CROSS_MIN_GAP_ATR or
             body_atr < SCALP_WEAK_CROSS_MIN_BODY_ATR)
    )

    extreme = r1 >= SCALP_EXTREME_RSI
    weak_trend = adxv < SCALP_WEAK_ADX
    exhaustion = (
        stretch > SCALP_EXHAUSTION_MAX_STRETCH_ATR
        or (
            stretch > SCALP_EXHAUSTION_CONDITIONAL_ATR
            and (extreme or weak_trend)
        )
    )

    continuation_chase = False
    if mode in {"CONTINUATION", "TRANSITION_CONTINUATION", "OPPORTUNITY_CONTINUATION"}:
        mature = (r1 >= SCALP_CONT_MATURE_RSI or adxv >= SCALP_CONT_MATURE_ADX)
        allowed_stretch = (
            SCALP_CONT_MATURE_MAX_STRETCH_ATR
            if mature else SCALP_CONT_ENTRY_MAX_STRETCH_ATR
        )
        continuation_chase = stretch > allowed_stretch

    diag = {
        "bar_ts": int(c["timestamp"]),
        "reason": "CLOSED_SETUP_READY" if setup else ("M15_TRANSITION_NEAR" if buy_transition else "M15_TREND_SETUP"),
        "entry_mode": mode,
        "buy_trend": buy_trend,
        "buy_transition": buy_transition,
        "buy_opportunity": buy_opportunity,
        "buy_breakout": buy_breakout,
        "buy_pull": buy_pull,
        "buy_pull_recovery": buy_pull_recovery,
        "buy_cont": buy_cont,
        "m30_bull": m30_bull,
        "h1_bull": h1_bull,
        "ema9": e9, "ema21": e21, "ema50": e50,
        "s9": s9, "s21": s21,
        "rsi": r1, "rsi_prev": r2,
        "adx": adxv, "plus_di": pdi, "minus_di": mdi,
        "macd": mac, "stretch_atr": stretch,
        "body_atr": body_atr, "gap_atr": gap_atr,
        "gap21_50_atr": gap21_50_atr,
        "di_ratio": di_ratio,
        "raw_cross": raw_cross,
        "weak_cross": weak_cross,
        "exhaustion": exhaustion,
        "continuation_chase": continuation_chase,
        "breakout_level": breakout_level,
        "atr": atrv,
        "closed_price": c1,
    }

    if not setup:
        return False, diag
    if exhaustion:
        diag["reason"] = "EXHAUSTION_STRETCH"
        return False, diag
    if continuation_chase:
        diag["reason"] = "WAIT_CONTINUATION_CHASE"
        return False, diag
    if weak_cross:
        diag["reason"] = "WAIT_WEAK_TRANSITION"
        return False, diag
    return True, diag

def immediate_quality(live, prior_closed, mid, atrv):
    live_c = live.iloc[-1]
    r0 = float(live_c["rsi14"])
    r1 = float(prior_closed["rsi14"])
    rsi_delta = r0 - r1

    mac0 = float(live_c["macd_hist"])
    mac1 = float(prior_closed["macd_hist"])
    mac_delta = mac0 - mac1

    prev_close = float(prior_closed["close"])
    move_atr = (mid - prev_close) / atrv if atrv > 0 else 0.0

    agree = 0
    oppose = 0
    if rsi_delta >= SCALP_IMM_RSI_DEADBAND:
        agree += 1
    elif rsi_delta <= -SCALP_IMM_RSI_DEADBAND:
        oppose += 1

    if mac_delta > 0.0:
        agree += 1
    elif mac_delta < 0.0:
        oppose += 1

    if move_atr >= SCALP_IMM_SWING_ATR:
        agree += 1
        swing = "UP"
    elif move_atr <= -SCALP_IMM_SWING_ATR:
        oppose += 1
        swing = "DOWN"
    else:
        swing = "FLAT"

    score = direction_score_live(live)
    opp = opposite_score_live(live)
    adxv = float(prior_closed["adx14"])
    pdi = float(live_c["plus_di"])
    mdi = float(live_c["minus_di"])
    di_dom = pdi - mdi

    if score < 2 or score <= opp:
        return False, {
            "reason":"LIVE_DIRECTION_CONFLICT","agree":agree,"oppose":oppose,
            "rsi_delta":rsi_delta,"mac_delta":mac_delta,"move_atr":move_atr,
            "swing":swing,"score":score,"opp_score":opp,"di_dom":di_dom,
        }

    if agree <= 1:
        return False, {
            "reason":"SCALP_NEED_3OF3_OR_STRONG_2OF3","agree":agree,"oppose":oppose,
            "rsi_delta":rsi_delta,"mac_delta":mac_delta,"move_atr":move_atr,
            "swing":swing,"score":score,"opp_score":opp,"di_dom":di_dom,
        }

    if agree >= 3:
        direct_ok = (
            adxv >= SCALP_DIRECT_3OF3_MIN_ADX
            and score >= SCALP_DIRECT_3OF3_MIN_SCORE
            and di_dom >= SCALP_DIRECT_3OF3_MIN_DI_DOM
            and swing != "DOWN"
        )
        return direct_ok, {
            "reason":"ENTRY_READY_3OF3" if direct_ok else "WAIT_3OF3_LOW_QUALITY",
            "agree":agree,"oppose":oppose,
            "rsi_delta":rsi_delta,"mac_delta":mac_delta,"move_atr":move_atr,
            "swing":swing,"score":score,"opp_score":opp,"di_dom":di_dom,
        }

    # Exact strong 2/3 SCALP-family lane.
    strong = (
        adxv >= SCALP_STRONG_2OF3_MIN_ADX
        and di_dom >= SCALP_STRONG_2OF3_MIN_DI_DOM
        and score >= SCALP_STRONG_2OF3_MIN_SCORE
        and rsi_delta >= SCALP_STRONG_2OF3_MIN_RSI_ACCEL
        and mac_delta > 0.0
    )
    research_2of3 = (
        agree == 2
        and score >= 3
        and opp <= 1
        and di_dom >= 2.0
        and rsi_delta > -0.10
        and mac_delta >= 0.0
        and swing != "DOWN"
        and move_atr >= -0.02
    )
    accepted = strong or research_2of3
    qreason = (
        "ENTRY_READY_STRONG_2OF3" if strong
        else ("ENTRY_READY_RESEARCH_2OF3" if research_2of3
              else "SCALP_STRONG_2OF3_NOT_CONFIRMED")
    )
    return accepted, {
        "reason":qreason,
        "agree":agree,"oppose":oppose,
        "rsi_delta":rsi_delta,"mac_delta":mac_delta,"move_atr":move_atr,
        "swing":swing,"score":score,"opp_score":opp,"di_dom":di_dom,
    }



def live_trend_entry(m15, live, m30, h1, bid, ask):
    """
    V6: evaluate the CURRENT M15 bar inside an already bullish regime.

    This solves the V5 bottleneck where closed M15 showed Trend=1 but
    Pull=0/Cont=0 for long periods. We wait for the current bar to restart
    upward near EMA9/21 instead of waiting another 15 minutes.
    """
    if not LIVE_DIP_ENABLED and not LIVE_BREAKOUT_ENABLED:
        return False, None

    c = m15.iloc[-1]
    lc = live.iloc[-1]
    lp = live.iloc[-2]

    atrv = float(c["atr14"])
    if atrv <= 0:
        return False, None

    mid = (bid + ask) / 2.0
    e9 = float(c["ema9"])
    e21 = float(c["ema21"])
    e50 = float(c["ema50"])

    m30c = m30.iloc[-1]
    h1c = h1.iloc[-1]
    m30_up = float(m30c["close"]) > float(m30c["ema50"])
    h1_up = float(h1c["close"]) > float(h1c["ema50"])
    macro_ok = (m30_up and h1_up) if LIVE_DIP_REQUIRE_BOTH_MACRO_UP else (m30_up or h1_up)

    # Closed regime is allowed to be slightly transitional, but EMA9 must lead.
    regime_ok = (
        macro_ok
        and e9 > e21
        and e21 >= e50 - 0.20 * atrv
    )
    if not regime_ok:
        return False, None

    rsi_now = float(lc["rsi14"])
    rsi_prev = float(lp["rsi14"])
    rsi_delta = rsi_now - rsi_prev

    mac_now = float(lc["macd_hist"])
    mac_prev = float(lp["macd_hist"])
    mac_delta = mac_now - mac_prev

    pdi = float(lc["plus_di"])
    mdi = float(lc["minus_di"])
    di_ratio = pdi / max(mdi, 1e-9)

    adxv = float(c["adx14"])
    stretch = max(0.0, (mid - e21) / atrv)
    dist21 = abs(mid - e21) / atrv

    live_open = float(lc["open"])
    prior_high = float(c["high"])
    prior_close = float(c["close"])
    move_atr = (mid - prior_close) / atrv

    clues = 0
    if mid > live_open:
        clues += 1
    if rsi_delta >= LIVE_DIP_MIN_RSI_DELTA:
        clues += 1
    if mac_delta > 0.0:
        clues += 1
    if pdi > mdi:
        clues += 1

    dip_ok = (
        LIVE_DIP_ENABLED
        and adxv >= LIVE_DIP_MIN_ADX
        and LIVE_DIP_MIN_RSI <= rsi_now <= LIVE_DIP_MAX_RSI
        and di_ratio >= LIVE_DIP_MIN_DI_RATIO
        and dist21 <= LIVE_DIP_MAX_EMA21_DISTANCE_ATR
        and stretch <= LIVE_DIP_MAX_STRETCH_ATR
        and mid >= e21
        and move_atr >= LIVE_DIP_MIN_MOVE_ATR
        and clues >= LIVE_DIP_MIN_CLUES
    )

    breakout_level = prior_high + LIVE_BREAKOUT_BUFFER_ATR * atrv
    breakout_ok = (
        LIVE_BREAKOUT_ENABLED
        and mid >= breakout_level
        and adxv >= LIVE_BREAKOUT_MIN_ADX
        and rsi_now >= LIVE_BREAKOUT_MIN_RSI
        and di_ratio >= LIVE_BREAKOUT_MIN_DI_RATIO
        and stretch <= LIVE_BREAKOUT_MAX_STRETCH_ATR
        and mac_now >= 0.0
        and mac_delta >= 0.0
        and rsi_delta >= 0.0
    )

    if not dip_ok and not breakout_ok:
        return False, None

    mode = "LIVE_BREAKOUT" if breakout_ok else "LIVE_DIP_RECLAIM"
    score = direction_score_live(live)
    opp = opposite_score_live(live)

    diag = {
        "bar_ts": int(c["timestamp"]),
        "reason": "ENTRY_READY_" + mode,
        "entry_mode": mode,
        "quality_reason": "ENTRY_READY_" + mode,
        "buy_trend": True,
        "buy_transition": False,
        "buy_opportunity": False,
        "buy_breakout": breakout_ok,
        "buy_pull": dip_ok,
        "buy_pull_recovery": dip_ok,
        "buy_cont": breakout_ok,
        "m30_bull": m30_up,
        "h1_bull": h1_up,
        "ema9": e9,
        "ema21": e21,
        "ema50": e50,
        "s9": (e9 - float(m15.iloc[-2]["ema9"])) / atrv,
        "s21": (e21 - float(m15.iloc[-2]["ema21"])) / atrv,
        "rsi": rsi_now,
        "rsi_prev": rsi_prev,
        "adx": adxv,
        "plus_di": pdi,
        "minus_di": mdi,
        "macd": mac_now,
        "stretch_atr": stretch,
        "body_atr": abs(mid - live_open) / atrv,
        "gap_atr": abs(e9 - e21) / atrv,
        "gap21_50_atr": (e21 - e50) / atrv,
        "di_ratio": di_ratio,
        "raw_cross": False,
        "weak_cross": False,
        "exhaustion": False,
        "continuation_chase": False,
        "atr": atrv,
        "closed_price": float(c["close"]),
        "score": score,
        "opp_score": opp,
        "imm_agree": clues,
        "imm_oppose": 0,
        "rsi_delta": rsi_delta,
        "macd_delta": mac_delta,
        "move_atr": move_atr,
        "live_swing": "UP",
        "di_dom": pdi - mdi,
    }
    return True, diag



def symbol_spread_cap(symbol):
    return CORE_MAX_SPREAD_BPS if symbol in CORE_SYMBOLS else NONCORE_MAX_SPREAD_BPS


def arm_paper_limit(state, symbol, bid, ask, plan, diag):
    if not V124_EXECUTION_PRIMARY:
        log(f"V127_RESEARCH_ONLY | {symbol} | timeframe={TIMEFRAME} | mode={diag.get('entry_mode')} | would_arm=1")
        return
    spread = ask - bid
    frac = max(0.0, min(PAPER_LIMIT_INSIDE_SPREAD_FRACTION, 0.95))
    limit_price = bid + spread * frac
    state.setdefault("pending_limits", {})[symbol] = {
        "armed_ts": time.time(),
        "limit_price": float(limit_price),
        "signal_bid": float(bid),
        "signal_ask": float(ask),
        "signal_mid": float((bid + ask) / 2.0),
        "best_bid": float(bid),
        "best_ask": float(ask),
        "plan": dict(plan),
        "diag": dict(diag),
        "reclaim_low": float((bid + ask) / 2.0),
    }
    log(
        f"LIMIT_ARMED | {symbol} | limit={limit_price:.8f} | "
        f"bid={bid:.8f} | ask={ask:.8f} | mode={diag.get('entry_mode')} | "
        f"spread={spread_bps(bid, ask):.1f}bps"
    )
    save_state(state)


def cancel_paper_limit(state, symbol, reason, bid, ask):
    p = state.get("pending_limits", {}).get(symbol)
    if not p:
        return
    log(
        f"LIMIT_CANCEL | {symbol} | {reason} | limit={float(p['limit_price']):.8f} | "
        f"bid={bid:.8f} | ask={ask:.8f}"
    )
    del state["pending_limits"][symbol]
    save_state(state)


def process_paper_limit(exchange, state, symbol, bid, ask, live):
    p = state.get("pending_limits", {}).get(symbol)
    if not p:
        return False

    # Do not keep a pending order alive if execution economics deteriorate.
    if spread_bps(bid, ask) > symbol_spread_cap(symbol):
        cancel_paper_limit(state, symbol, "SPREAD_WIDENED", bid, ask)
        return False

    age = time.time() - float(p["armed_ts"])
    mid = (bid + ask) / 2.0
    p["best_bid"] = max(float(p.get("best_bid", bid)), bid)
    p["best_ask"] = min(float(p.get("best_ask", ask)), ask)
    p["reclaim_low"] = min(float(p.get("reclaim_low", mid)), mid)

    limit_price = float(p["limit_price"])
    signal_ask = float(p["signal_ask"])
    signal_mid = float(p.get("signal_mid", (float(p["signal_bid"]) + signal_ask) / 2.0))

    signal_move_bps = (mid - signal_mid) / max(signal_mid, 1e-9) * 10000.0
    adverse_bps = (float(p["reclaim_low"]) - signal_mid) / max(signal_mid, 1e-9) * 10000.0
    reclaim_bps = (mid - float(p["reclaim_low"])) / max(float(p["reclaim_low"]), 1e-9) * 10000.0

    lc = live.iloc[-1]
    lp = live.iloc[-2]
    rsi_up = float(lc["rsi14"]) >= float(lp["rsi14"])
    macd_up = float(lc["macd_hist"]) >= float(lp["macd_hist"])
    pdi = float(lc["plus_di"]); mdi = float(lc["minus_di"])
    momentum_votes = int(rsi_up) + int(macd_up) + int(pdi >= mdi)
    momentum_ok = momentum_votes >= 2

    # 1) Executable ask touched limit.
    if ask <= limit_price:
        plan = dict(p["plan"]); diag = dict(p["diag"])
        del state["pending_limits"][symbol]
        save_state(state)
        open_position_at_price(state, symbol, plan, limit_price, bid, ask, diag, "PASSIVE_ASK_TOUCH")
        return True

    # 2) NDAX last trade touched/undercut our passive limit.
    if PAPER_LIMIT_USE_LAST_TOUCH:
        try:
            tk = exchange.fetch_ticker(symbol)
            last_px = float(tk.get("last") or 0.0)
        except Exception:
            last_px = 0.0
        if last_px > 0 and last_px <= limit_price:
            plan = dict(p["plan"]); diag = dict(p["diag"])
            del state["pending_limits"][symbol]
            save_state(state)
            open_position_at_price(state, symbol, plan, limit_price, bid, ask, diag, "PASSIVE_LAST_TOUCH")
            return True

    # 3) Controlled dip/reclaim at the ask.
    dipped = float(p["best_ask"]) < signal_ask
    if (
        dipped
        and reclaim_bps >= PAPER_LIMIT_MIN_RECLAIM_BPS
        and signal_move_bps <= PAPER_LIMIT_MAX_CHASE_BPS
        and adverse_bps >= -PAPER_LIMIT_FALLBACK_MAX_ADVERSE_BPS
        and momentum_ok
        and spread_bps(bid, ask) <= float(ENTRY.PARAMS["reclaim_market_max_spread_bps"])
        and age >= 15
    ):
        plan = dict(p["plan"]); diag = dict(p["diag"])
        del state["pending_limits"][symbol]
        save_state(state)
        open_position_at_price(state, symbol, plan, ask, bid, ask, diag, "RECLAIM_MARKET")
        return True

    # 4) V9 adaptive fallback: if the signal is still intact after a short maker
    # attempt, take the ask rather than timing out every valid setup.
    if (
        PAPER_LIMIT_FALLBACK_ENABLED
        and age >= PAPER_LIMIT_FALLBACK_SEC
        and signal_move_bps <= PAPER_LIMIT_FALLBACK_MAX_SIGNAL_MOVE_BPS
        and adverse_bps >= -PAPER_LIMIT_FALLBACK_MAX_ADVERSE_BPS
        and momentum_ok
    ):
        plan = dict(p["plan"]); diag = dict(p["diag"])
        del state["pending_limits"][symbol]
        save_state(state)
        open_position_at_price(state, symbol, plan, ask, bid, ask, diag, "ADAPTIVE_TAKER")
        return True

    if age >= PAPER_LIMIT_MAX_WAIT_SEC:
        cancel_paper_limit(state, symbol, "TIMEOUT", bid, ask)
        return False

    # Cancel only if price has actually chased away from the signal midpoint.
    if signal_move_bps > PAPER_LIMIT_MAX_CHASE_BPS:
        cancel_paper_limit(state, symbol, "CHASED", bid, ask)
        return False

    save_state(state)
    return False


def open_position_at_price(state, symbol, plan, fill_price, bid, ask, diag, fill_mode):
    # Same position accounting as open_position(), but uses the actual paper fill price.
    entry = float(fill_price)
    stop_dist = float(plan["stop_distance"])
    target_dist = stop_dist * float(plan.get("plan_rr", SCALP_RR))
    amount = float(plan["amount"])

    pos = {
        "entry": entry,
        "amount": amount,
        "stop": entry - stop_dist,
        "initial_stop": entry - stop_dist,
        "target": entry + target_dist,
        "initial_risk": stop_dist,
        "opened_ts": time.time(),
        "peak_net": -999999.0,
        "mfe_price": entry,
        "mae_price": entry,
        "break_even": False,
        "trade_path_done": [],
        "spread_bps": spread_bps(bid, ask),
        "planned_risk_cad": float(plan["planned_risk_cad"]),
        "diag": diag,
        "fill_mode": fill_mode,
        "v124_entry_mid": float((bid + ask) / 2.0),
        "v124_entry_bid": float(bid),
        "v124_entry_ask": float(ask),
    }
    state["positions"][symbol] = pos
    state["trades_today"] += 1
    state["last_entry_bar"][symbol] = diag.get("bar_ts")
    state.setdefault("last_entry_ts", {})[symbol] = time.time()
    save_state(state)

    log(
        f"PAPER_BUY | {symbol} | fill={fill_mode} | mode={diag.get('entry_mode')} | "
        f"entry={entry:.8f} | amount={amount:.8f} | "
        f"stop={pos['stop']:.8f} | target={pos['target']:.8f} | "
        f"risk<=C${plan['planned_risk_cad']:.2f} | spread={spread_bps(bid, ask):.1f}bps"
    )
    append_order_line(
        f"{utc_iso()} | OPEN | {symbol} | fill={fill_mode} | mode={diag.get('entry_mode')} | "
        f"entry={entry:.8f} | stop={pos['stop']:.8f} | target={pos['target']:.8f} | "
        f"risk=C${plan['planned_risk_cad']:.2f}"
    )
    try:
        oid = v120_opportunity_record(symbol, diag, bid, ask, "EXECUTED")
        pos["opportunity_id"] = oid
        pos["v120_signal_price"] = (float(bid)+float(ask))/2.0
        pos["v120_signal_ts"] = int(time.time())
        pos["v120_checkpoints_done"] = []
    except Exception as exc:
        log(f"V120_EXEC_RESEARCH_ERROR | {symbol} | {exc}")
    event("paper_buy", symbol=symbol, position=pos)


def live_recovery_entry(m15, live, m30, h1, bid, ask):
    if not LIVE_RECOVERY_ENABLED:
        return False, None

    c = m15.iloc[-1]
    lc = live.iloc[-1]
    lp = live.iloc[-2]
    atrv = float(c["atr14"])
    if atrv <= 0:
        return False, None

    mid = (bid + ask) / 2.0
    e9 = float(c["ema9"]); e21 = float(c["ema21"]); e50 = float(c["ema50"])
    m30_up = float(m30.iloc[-1]["close"]) > float(m30.iloc[-1]["ema50"])
    h1_up = float(h1.iloc[-1]["close"]) > float(h1.iloc[-1]["ema50"])

    rsi_now = float(lc["rsi14"]); rsi_prev = float(lp["rsi14"])
    mac_now = float(lc["macd_hist"]); mac_prev = float(lp["macd_hist"])
    pdi = float(lc["plus_di"]); mdi = float(lc["minus_di"])
    di_ratio = pdi / max(mdi, 1e-9)
    adxv = float(c["adx14"])
    stretch = max(0.0, (mid - e21) / atrv)
    gap21_50 = (e21 - e50) / atrv
    current_green = mid > float(lc["open"])

    # Main intent: catch ETH/BTC-style recovery while price is still near EMA21.
    ok = (
        (m30_up if LIVE_RECOVERY_REQUIRE_M30_UP else (m30_up or h1_up))
        and e9 >= e21 - 0.05 * atrv
        and gap21_50 >= -LIVE_RECOVERY_MAX_GAP21_50_ATR
        and adxv >= LIVE_RECOVERY_MIN_ADX
        and LIVE_RECOVERY_MIN_RSI <= rsi_now <= LIVE_RECOVERY_MAX_RSI
        and (rsi_now - rsi_prev) >= LIVE_RECOVERY_MIN_RSI_DELTA
        and di_ratio >= LIVE_RECOVERY_MIN_DI_RATIO
        and mac_now > mac_prev
        and current_green
        and mid >= e21
        and stretch <= LIVE_RECOVERY_MAX_STRETCH_ATR
    )
    if not ok:
        return False, None

    score = direction_score_live(live)
    opp = opposite_score_live(live)
    return True, {
        "bar_ts": int(c["timestamp"]),
        "reason": "ENTRY_READY_LIVE_RECOVERY",
        "entry_mode": "LIVE_RECOVERY",
        "quality_reason": "ENTRY_READY_LIVE_RECOVERY",
        "buy_trend": e9 > e21 > e50,
        "buy_transition": True,
        "buy_opportunity": True,
        "buy_breakout": False,
        "buy_pull": True,
        "buy_pull_recovery": True,
        "buy_cont": False,
        "m30_bull": m30_up,
        "h1_bull": h1_up,
        "ema9": e9, "ema21": e21, "ema50": e50,
        "s9": (e9 - float(m15.iloc[-2]["ema9"])) / atrv,
        "s21": (e21 - float(m15.iloc[-2]["ema21"])) / atrv,
        "rsi": rsi_now, "rsi_prev": rsi_prev,
        "adx": adxv, "plus_di": pdi, "minus_di": mdi,
        "macd": mac_now, "stretch_atr": stretch,
        "body_atr": abs(mid - float(lc["open"])) / atrv,
        "gap_atr": abs(e9 - e21) / atrv,
        "gap21_50_atr": gap21_50,
        "di_ratio": di_ratio,
        "raw_cross": False, "weak_cross": False,
        "exhaustion": False, "continuation_chase": False,
        "atr": atrv, "closed_price": float(c["close"]),
        "score": score, "opp_score": opp,
        "imm_agree": 3, "imm_oppose": 0,
        "rsi_delta": rsi_now-rsi_prev,
        "macd_delta": mac_now-mac_prev,
        "move_atr": (mid-float(c["close"])) / atrv,
        "live_swing": "UP",
        "di_dom": pdi-mdi,
    }



def maestro_precandidate(m15, m30, h1):
    """
    Cheap precheck before M5/M1 API calls.
    It intentionally admits transition/recovery structure but requires M30
    support; this reduces scan latency while preserving the MAESTRO lane.
    """
    if not MAESTRO_PRECHECK_ENABLED:
        return True
    c = m15.iloc[-1]
    p = m15.iloc[-2]
    atrv = float(c["atr14"])
    if atrv <= 0:
        return False

    e9 = float(c["ema9"]); e21 = float(c["ema21"]); e50 = float(c["ema50"])
    s9 = (e9 - float(p["ema9"])) / atrv
    s21 = (e21 - float(p["ema21"])) / atrv
    gap = (e21 - e50) / atrv
    dom = float(c["plus_di"]) - float(c["minus_di"])
    m30_up = maestro_macro_bull(m30.iloc[-1])

    return (
        m30_up
        and float(c["rsi14"]) >= MAESTRO_PRECHECK_MIN_RSI
        and dom >= MAESTRO_PRECHECK_MIN_DI_DOM
        and gap >= -MAESTRO_PRECHECK_MAX_GAP21_50_ATR
        and e9 >= e21 - 0.10 * atrv
        and s9 >= -0.03
        and s21 >= -0.04
    )


def maestro_m5_score(m5):
    """MAESTRO V76 M5 score: RSI acceleration, MACD acceleration, RSI level."""
    if m5 is None or len(m5) < 3:
        return 0, 0.0, 0.0
    c = m5.iloc[-1]
    p = m5.iloc[-2]
    rd = float(c["rsi14"]) - float(p["rsi14"])
    md = float(c["macd_hist"]) - float(p["macd_hist"])
    score = 0
    if rd >= 0.50: score += 1
    if md > 0.0: score += 1
    if float(c["rsi14"]) >= 55.0: score += 1
    return score, rd, md


def maestro_m1_score(m1, mid):
    """MAESTRO V76 M1 score: RSI accel, MACD accel, >=0.02 ATR directional move."""
    if m1 is None or len(m1) < 3:
        return 0, 0.0, 0.0, 0.0
    c = m1.iloc[-1]
    p = m1.iloc[-2]
    atrv = float(c["atr14"])
    rd = float(c["rsi14"]) - float(p["rsi14"])
    md = float(c["macd_hist"]) - float(p["macd_hist"])
    mv = (mid - float(p["close"])) / atrv if atrv > 0 else 0.0
    score = 0
    if rd >= 0.20: score += 1
    if md > 0.0: score += 1
    if mv >= 0.02: score += 1
    return score, rd, md, mv


def maestro_not_late_not_hostile(m1, live_m15):
    """
    Adaptation of V76 MEV74NotLateNotHostile for MAESTRO_M15:
    M1 stretch <=1.10 ATR, M1 impulse <=0.55 ATR, M15 live DI dominance >= -2.
    """
    if m1 is None or len(m1) < 3:
        return False, {"reason":"MAESTRO_M1_DATA_MISSING"}
    c = m1.iloc[-1]
    p = m1.iloc[-2]
    atr1 = float(c["atr14"])
    if atr1 <= 0:
        return False, {"reason":"MAESTRO_M1_ATR_INVALID"}

    stretch = abs(float(c["close"]) - float(c["ema21"])) / atr1
    impulse = (float(c["close"]) - float(p["close"])) / atr1
    lc = live_m15.iloc[-1]
    dom = float(lc["plus_di"]) - float(lc["minus_di"])

    ok = (
        stretch <= MAESTRO_M1_MAX_STRETCH_ATR
        and impulse <= MAESTRO_M1_MAX_IMPULSE_ATR
        and dom >= MAESTRO_M15_MIN_LIVE_DI_DOM
    )
    return ok, {
        "reason":"MAESTRO_NOT_LATE_OK" if ok else "MAESTRO_NOT_LATE_BLOCK",
        "m1_stretch":stretch,
        "m1_impulse":impulse,
        "live_di_dom":dom,
    }


def maestro_macro_bull(row):
    return float(row["close"]) > float(row["ema50"])


def maestro_healthy_trend_candidate(m15, live, m30, h1, m5, m1, bid, ask):
    """
    Crypto translation of V76 MEHealthyTrendEntryProof (SCALP personality).
    Key difference from prior NDAX versions: H1 is NOT always mandatory.
    """
    if not MAESTRO_M15_ENABLED:
        return False, None

    c = m15.iloc[-1]
    p = m15.iloc[-2]
    pp = m15.iloc[-3]
    atrv = float(c["atr14"])
    if atrv <= 0:
        return False, None

    mid = (bid + ask) / 2.0
    e9 = float(c["ema9"]); e21 = float(c["ema21"]); e50 = float(c["ema50"])
    e9p = float(p["ema9"]); e21p = float(p["ema21"])
    e9pp = float(pp["ema9"]); e21pp = float(pp["ema21"])
    s9 = (e9-e9p)/atrv
    s21 = (e21-e21p)/atrv
    s9prev = (e9p-e9pp)/atrv
    s21prev = (e21p-e21pp)/atrv

    adx = float(c["adx14"])
    adx2 = float(p["adx14"])
    plus = float(c["plus_di"]); minus = float(c["minus_di"])
    plus2 = float(p["plus_di"]); minus2 = float(p["minus_di"])
    dom = plus-minus
    dom2 = plus2-minus2

    m30_up = maestro_macro_bull(m30.iloc[-1])
    h1_up = maestro_macro_bull(h1.iloc[-1])
    h1_exception = (
        MAESTRO_HT_ALLOW_H1_EXCEPTION
        and adx >= MAESTRO_HT_H1_EXCEPTION_MIN_ADX
        and dom >= MAESTRO_HT_H1_EXCEPTION_MIN_DI_DOM
    )
    htf_ok = m30_up and (h1_up or h1_exception)

    # V76 uses strict stack. Crypto gets a tiny EMA21/50 transition allowance
    # because the V7 log repeatedly had valid M30 trends with EMA21 just below EMA50.
    stack = e9 > e21 and e21 >= e50 - MAESTRO_HT_MAX_TRANSITION_GAP_ATR*atrv
    slope_persistent = (
        s9 >= MAESTRO_HT_MIN_S9_ATR
        and s21 >= MAESTRO_HT_MIN_S21_ATR
        and s9prev >= -0.010
        and s21prev >= -0.015
    )
    price_persistent = (
        float(c["close"]) > float(c["ema21"])
        and float(p["close"]) > float(p["ema21"])
        and float(pp["close"]) > float(pp["ema21"])
    )

    stretch = abs(float(c["close"])-e21)/atrv
    body = abs(float(c["close"])-float(c["open"]))/atrv
    dist9 = abs(float(c["low"])-e9)/atrv
    prev_dist9 = abs(float(p["close"])-float(p["ema9"]))/atrv
    shallow_reset = (
        dist9 <= MAESTRO_HT_RESET_TO_EMA9_ATR
        or prev_dist9 <= MAESTRO_HT_RESET_TO_EMA9_ATR
        or stretch <= 0.55
    )

    base_ok = (
        stack and slope_persistent and price_persistent
        and MAESTRO_HT_MIN_ADX <= adx <= MAESTRO_HT_MAX_ADX
        and adx >= adx2-2.5
        and dom >= MAESTRO_HT_MIN_DI_DOM
        and dom2 >= 0.0
        and htf_ok and shallow_reset
        and MAESTRO_HT_MIN_STRETCH_ATR <= stretch <= MAESTRO_HT_MAX_STRETCH_ATR
        and body <= MAESTRO_HT_MAX_BODY_ATR
    )
    if not base_ok:
        return False, None

    m5_score, m5_rd, m5_md = maestro_m5_score(m5)
    m1_score, m1_rd, m1_md, m1_mv = maestro_m1_score(m1, mid)
    qok, q = immediate_quality(live, c, mid, atrv)
    notlate, nl = maestro_not_late_not_hostile(m1, live)

    live_stretch = abs(mid-float(live.iloc[-1]["ema21"])) / max(float(live.iloc[-1]["atr14"]), 1e-9)
    live_body = abs(float(live.iloc[-1]["close"])-float(live.iloc[-1]["open"])) / max(float(live.iloc[-1]["atr14"]), 1e-9)

    strong_combo = (
        MAESTRO_STRONG_COMBO_ENABLED
        and q.get("agree",0) >= 3
        and q.get("oppose",0) == 0
        and m5_score >= MAESTRO_STRONG_COMBO_M5_SCORE
        and m1_score >= MAESTRO_STRONG_COMBO_M1_SCORE
    )
    standard_combo = (
        m5_score >= MAESTRO_STANDARD_COMBO_M5_SCORE
        and m1_score >= MAESTRO_STANDARD_COMBO_M1_SCORE
    )
    confirm_ok = (
        q.get("agree",0) >= MAESTRO_IMMEDIATE_MIN_AGREE
        and q.get("oppose",0) <= MAESTRO_IMMEDIATE_MAX_OPPOSE
        and (strong_combo or standard_combo)
        and notlate
        and live_stretch <= MAESTRO_HT_MAX_STRETCH_ATR + 0.20
        and live_body <= 1.00
    )
    if not confirm_ok:
        return False, {
            "reason":"MAESTRO_HEALTHY_WAIT",
            "entry_mode":"MAESTRO_HEALTHY_TREND",
            "bar_ts":int(c["timestamp"]),
            "m5_score":m5_score,"m1_score":m1_score,
            "imm_agree":q.get("agree",0),"imm_oppose":q.get("oppose",0),
            "m1_stretch":nl.get("m1_stretch",99.0),
            "m1_impulse":nl.get("m1_impulse",99.0),
        }

    score = direction_score_live(live)
    opp = opposite_score_live(live)
    diag = {
        "bar_ts":int(c["timestamp"]),
        "reason":"ENTRY_READY_MAESTRO_HEALTHY_TREND",
        "entry_mode":"MAESTRO_HEALTHY_TREND",
        "quality_reason":"ENTRY_READY_MAESTRO_HEALTHY_TREND",
        "buy_trend":True,"buy_transition":not (e21>e50),
        "buy_opportunity":True,"buy_breakout":False,
        "buy_pull":True,"buy_pull_recovery":True,"buy_cont":False,
        "m30_bull":m30_up,"h1_bull":h1_up,
        "ema9":e9,"ema21":e21,"ema50":e50,
        "s9":s9,"s21":s21,"rsi":float(c["rsi14"]),"rsi_prev":float(p["rsi14"]),
        "adx":adx,"plus_di":plus,"minus_di":minus,
        "macd":float(c["macd_hist"]),"stretch_atr":stretch,
        "body_atr":body,"gap_atr":abs(e9-e21)/atrv,
        "gap21_50_atr":(e21-e50)/atrv,"di_ratio":plus/max(minus,1e-9),
        "raw_cross":False,"weak_cross":False,"exhaustion":False,
        "continuation_chase":False,"atr":atrv,"closed_price":float(c["close"]),
        "score":score,"opp_score":opp,
        "imm_agree":q.get("agree",0),"imm_oppose":q.get("oppose",0),
        "rsi_delta":q.get("rsi_delta",0.0),"macd_delta":q.get("mac_delta",0.0),
        "move_atr":q.get("move_atr",0.0),"live_swing":q.get("swing","FLAT"),
        "di_dom":float(live.iloc[-1]["plus_di"])-float(live.iloc[-1]["minus_di"]),
        "m5_score":m5_score,"m1_score":m1_score,
        "m5_rsi_delta":m5_rd,"m1_rsi_delta":m1_rd,
        "m1_move_atr":m1_mv,
        "plan_min_stop_atr":MAESTRO_MIN_STOP_ATR,
        "plan_rr":MAESTRO_RR,
    }
    return True, diag


def maestro_v103_candidate(m15, live, m30, h1, m5, m1, bid, ask):
    """Strict V103-like lane preserved for periods when M30 and H1 fully align."""
    if not (MAESTRO_M15_ENABLED and MAESTRO_V103_ENABLED):
        return False, None

    c=m15.iloc[-1]; p=m15.iloc[-2]
    atrv=float(c["atr14"])
    if atrv<=0: return False,None
    mid=(bid+ask)/2.0
    e9=float(c["ema9"]);e21=float(c["ema21"]);e50=float(c["ema50"])
    s9=(e9-float(p["ema9"]))/atrv
    s21=(e21-float(p["ema21"]))/atrv
    plus=float(c["plus_di"]);minus=float(c["minus_di"])
    stretch=abs(float(c["close"])-e21)/atrv
    body=abs(float(c["close"])-float(c["open"]))/atrv

    # Stricter macro translation: close > EMA21 > EMA50 and EMA21 non-falling.
    def strict_macro(df):
        a=df.iloc[-1]; b=df.iloc[-2]
        return (
            float(a["close"])>float(a["ema21"])>float(a["ema50"])
            and float(a["ema21"])>=float(b["ema21"])
        )
    macro = strict_macro(m30) and strict_macro(h1)
    structure=(
        float(c["close"])>e9>e21>e50
        and s9>=MAESTRO_V103_MIN_S9_ATR
        and s21>=MAESTRO_V103_MIN_S21_ATR
    )
    core=(
        structure and macro and plus>minus
        and float(c["adx14"])>=MAESTRO_V103_MIN_ADX
        and float(c["rsi14"])>=MAESTRO_V103_MIN_RSI
    )
    if not core: return False,None

    pull=(
        float(c["low"])<=e21+MAESTRO_V103_PULL_TOUCH_ATR*atrv
        and float(c["close"])>float(c["open"])
        and stretch<=MAESTRO_V103_PULL_MAX_STRETCH_ATR
    )
    prev_high=max(float(m15.iloc[-2]["high"]),float(m15.iloc[-3]["high"]))
    compact=MAESTRO_V103_MIN_BODY_ATR<=body<=MAESTRO_V103_MAX_BODY_ATR
    cont=compact and float(c["close"])>prev_high and stretch<=MAESTRO_V103_BREAKOUT_MAX_STRETCH_ATR
    if not (pull or cont): return False,None

    qok,q=immediate_quality(live,c,mid,atrv)
    m5_score,m5_rd,m5_md=maestro_m5_score(m5)
    m1_score,m1_rd,m1_md,m1_mv=maestro_m1_score(m1,mid)
    notlate,nl=maestro_not_late_not_hostile(m1,live)
    strong_combo = (
        q.get("agree",0) >= 3 and q.get("oppose",0) == 0
        and m5_score >= MAESTRO_STRONG_COMBO_M5_SCORE
        and m1_score >= MAESTRO_STRONG_COMBO_M1_SCORE
    )
    standard_combo = (
        m5_score >= MAESTRO_STANDARD_COMBO_M5_SCORE
        and m1_score >= MAESTRO_STANDARD_COMBO_M1_SCORE
    )
    if not (
        q.get("agree",0)>=2 and q.get("oppose",0)<=1
        and (strong_combo or standard_combo) and notlate
    ):
        return False,None

    score=direction_score_live(live);opp=opposite_score_live(live)
    return True,{
        "bar_ts":int(c["timestamp"]),
        "reason":"ENTRY_READY_MAESTRO_V103",
        "entry_mode":"MAESTRO_V103_"+("PULLBACK_RESUME" if pull else "BREAKOUT_CONTINUATION"),
        "quality_reason":"ENTRY_READY_MAESTRO_V103",
        "buy_trend":True,"buy_transition":False,"buy_opportunity":False,
        "buy_breakout":cont,"buy_pull":pull,"buy_pull_recovery":pull,"buy_cont":cont,
        "m30_bull":True,"h1_bull":True,
        "ema9":e9,"ema21":e21,"ema50":e50,"s9":s9,"s21":s21,
        "rsi":float(c["rsi14"]),"rsi_prev":float(p["rsi14"]),"adx":float(c["adx14"]),
        "plus_di":plus,"minus_di":minus,"macd":float(c["macd_hist"]),
        "stretch_atr":stretch,"body_atr":body,"gap_atr":abs(e9-e21)/atrv,
        "gap21_50_atr":(e21-e50)/atrv,"di_ratio":plus/max(minus,1e-9),
        "raw_cross":False,"weak_cross":False,"exhaustion":False,
        "continuation_chase":False,"atr":atrv,"closed_price":float(c["close"]),
        "score":score,"opp_score":opp,"imm_agree":q.get("agree",0),
        "imm_oppose":q.get("oppose",0),"rsi_delta":q.get("rsi_delta",0.0),
        "macd_delta":q.get("mac_delta",0.0),"move_atr":q.get("move_atr",0.0),
        "live_swing":q.get("swing","FLAT"),
        "di_dom":float(live.iloc[-1]["plus_di"])-float(live.iloc[-1]["minus_di"]),
        "m5_score":m5_score,"m1_score":m1_score,"m1_move_atr":m1_mv,
        "plan_min_stop_atr":MAESTRO_MIN_STOP_ATR,
        "plan_rr":MAESTRO_RR,
    }


# ============================================================
# PLAN / P&L
# ============================================================
def paper_entry_price(ask):
    return ask * (1 + PAPER_SLIPPAGE_BPS / 10000.0)

def paper_exit_price(bid):
    return bid * (1 - PAPER_SLIPPAGE_BPS / 10000.0)

def net_pnl(entry, exit_price, amount):
    gross = (exit_price - entry) * amount
    fees = (
        entry * amount * ESTIMATED_TAKER_FEE_PCT
        + exit_price * amount * ESTIMATED_TAKER_FEE_PCT
    )
    return gross - fees, fees

def calculate_plan(exchange, symbol, ask, spread_bps_value, m15, equity, diag=None):
    c = m15.iloc[-1]
    atrv = float(c["atr14"])
    if atrv <= 0 or ask <= 0:
        return None, "BAD_PLAN_INPUT"

    lookback = min(SCALP_SWING_LOOKBACK, len(m15))
    recent_low = min(float(m15.iloc[-i]["low"]) for i in range(1, lookback + 1))

    min_stop_atr = float((diag or {}).get("plan_min_stop_atr", SCALP_MIN_STOP_ATR))
    plan_rr = float((diag or {}).get("plan_rr", SCALP_RR))
    stop_distance = max(
        min_stop_atr * atrv,
        ask - recent_low + SCALP_SWING_BUFFER_ATR * atrv,
    )
    stop = ask - stop_distance
    if stop <= 0:
        return None, "BAD_STOP"

    # Planned loss includes entry + stop-side taker fees.
    loss_per_unit = (
        stop_distance
        + ask * ESTIMATED_TAKER_FEE_PCT
        + stop * ESTIMATED_TAKER_FEE_PCT
    )
    if loss_per_unit <= 0:
        return None, "BAD_RISK_UNIT"

    amount = RISK_CAD / loss_per_unit
    cash_cap = max(0.0, min(MAX_POSITION_CAD, equity - MIN_CAD_RESERVE))
    amount = min(amount, cash_cap / ask)

    try:
        amount = float(exchange.amount_to_precision(symbol, amount))
        stop = float(exchange.price_to_precision(symbol, stop))
    except Exception:
        pass

    notional = amount * ask
    actual_risk = amount * loss_per_unit
    if amount <= 0 or notional < MIN_NOTIONAL_CAD:
        return None, f"NOTIONAL_TOO_SMALL_{notional:.2f}"

    target = ask + stop_distance * plan_rr
    try:
        target = float(exchange.price_to_precision(symbol, target))
    except Exception:
        pass

    spread_cost = amount * ask * max(0.0, spread_bps_value) / 10000.0
    roundtrip_fee = 2.0 * amount * ask * ESTIMATED_TAKER_FEE_PCT
    slip_cost = 2.0 * amount * ask * PAPER_SLIPPAGE_BPS / 10000.0
    friction = spread_cost + roundtrip_fee + slip_cost
    target_gross = amount * max(0.0, target - ask)
    edge_ratio = target_gross / max(friction, 1e-9)

    modeled_net_target = target_gross - friction
    min_net_required = (
        SCALP_MIN_NET_TARGET_CAD
        if spread_bps_value <= SCALP_PREFERRED_SPREAD_BPS
        else SCALP_HIGH_SPREAD_MIN_NET_TARGET_CAD
    )
    if friction > SCALP_MAX_ROUNDTRIP_FRICTION_CAD:
        return None, (
            f"SCALP_COST_GATE friction=C${friction:.2f} "
            f"> cap=C${SCALP_MAX_ROUNDTRIP_FRICTION_CAD:.2f}"
        )
    if modeled_net_target < min_net_required:
        return None, (
            f"SCALP_NET_EDGE_GATE targetGross=C${target_gross:.2f} "
            f"friction=C${friction:.2f} netTarget=C${modeled_net_target:.2f} "
            f"< min=C${min_net_required:.2f} ratio={edge_ratio:.2f}"
        )
    if edge_ratio < SCALP_MIN_TARGET_TO_FRICTION:
        return None, (
            f"SCALP_EDGE_GATE targetGross=C${target_gross:.2f} "
            f"friction=C${friction:.2f} ratio={edge_ratio:.2f}"
        )

    return {
        "amount": amount,
        "notional_cad": notional,
        "stop": stop,
        "target": target,
        "stop_distance": stop_distance,
        "planned_risk_cad": actual_risk,
        "stop_atr": stop_distance / atrv,
        "friction_cad": friction,
        "edge_ratio": edge_ratio,
        "modeled_net_target_cad": modeled_net_target,
        "plan_rr": plan_rr,
        "min_stop_atr": min_stop_atr,
    }, None


# ============================================================
# POSITIONS / EXITS
# ============================================================
def open_position(state, symbol, plan, ask, spread, diag):
    if not V124_EXECUTION_PRIMARY:
        log(f"V127_RESEARCH_ONLY | {symbol} | timeframe={TIMEFRAME} | mode={diag.get('entry_mode')} | would_open=1")
        return
    entry = paper_entry_price(ask)
    stop_dist = ask - float(plan["stop"])
    target_dist = float(plan["target"]) - ask

    pos = {
        "entry": entry,
        "amount": float(plan["amount"]),
        "stop": entry - stop_dist,
        "initial_stop": entry - stop_dist,
        "target": entry + target_dist,
        "initial_risk": stop_dist,
        "opened_ts": time.time(),
        "peak_net": -999999.0,
        "mfe_price": entry,
        "mae_price": entry,
        "break_even": False,
        "trade_path_done": [],
        "spread_bps": spread,
        "planned_risk_cad": float(plan["planned_risk_cad"]),
        "diag": diag,
        "v124_entry_mid": float(entry / (1.0 + max(0.0, spread) / 20000.0)),
        "v124_entry_bid": float(entry / (1.0 + max(0.0, spread) / 10000.0)),
        "v124_entry_ask": float(entry),
    }
    state["positions"][symbol] = pos
    state["trades_today"] += 1
    state["last_entry_bar"][symbol] = diag["bar_ts"]
    state.setdefault("last_entry_ts", {})[symbol] = time.time()
    save_state(state)

    log(
        f"PAPER_BUY | {symbol} | mode={diag['entry_mode']} | "
        f"quality={diag['quality_reason']} | entry={entry:.8f} | "
        f"notional=C${plan['notional_cad']:.2f} | risk<=C${plan['planned_risk_cad']:.2f} | "
        f"stop={pos['stop']:.8f} | target={pos['target']:.8f} | "
        f"spread={spread:.1f}bps | friction=C${plan['friction_cad']:.2f} | "
        f"edge={plan['edge_ratio']:.2f} | netTarget=C${plan.get('modeled_net_target_cad',0.0):.2f} | "
        f"score={diag['score']}/4 | immediate={diag['imm_agree']}/3"
    )
    append_order_line(
        f"{utc_iso()} | OPEN | {symbol} | mode={diag['entry_mode']} | "
        f"quality={diag['quality_reason']} | entry={entry:.8f} | "
        f"stop={pos['stop']:.8f} | target={pos['target']:.8f} | "
        f"risk=C${plan['planned_risk_cad']:.2f} | spread={spread:.1f}bps"
    )
    try:
        oid = v120_opportunity_record(symbol, diag, bid, ask, "EXECUTED")
        pos["opportunity_id"] = oid
        pos["v120_signal_price"] = (float(bid)+float(ask))/2.0
        pos["v120_signal_ts"] = int(time.time())
        pos["v120_checkpoints_done"] = []
    except Exception as exc:
        log(f"V120_EXEC_RESEARCH_ERROR | {symbol} | {exc}")
    event("paper_buy", symbol=symbol, position=pos)

def close_position(state, symbol, bid, reason):
    pos = state["positions"][symbol]
    exit_px = paper_exit_price(bid)
    pnl, fees = net_pnl(float(pos["entry"]), exit_px, float(pos["amount"]))
    pnl = float(pnl)

    state["paper_equity_cad"] += pnl
    state["daily_pnl"] += pnl
    if pnl > 0:
        state["wins"] += 1
        state["consecutive_losses"] = 0
    else:
        state["losses"] += 1
        state["consecutive_losses"] += 1
        state["symbol_cooldown_until"][symbol] = (
            time.time() + SYMBOL_LOSS_COOLDOWN_MINUTES * 60
        )
        if state["consecutive_losses"] >= MAX_CONSEC_LOSSES:
            state["pause_until"] = time.time() + LOSS_PAUSE_MINUTES * 60

    held = time.time() - float(pos["opened_ts"])
    row = {
        "closed_utc": utc_iso(),
        "build": BOT_VERSION,
        "symbol": symbol,
        "reason": reason,
        "entry": round(float(pos["entry"]), 10),
        "exit": round(exit_px, 10),
        "net_pnl_cad": round(pnl, 4),
        "fees_cad": round(fees, 4),
        "peak_net_cad": round(float(pos["peak_net"]), 4),
        "mfe_price": round(float(pos["mfe_price"]), 10),
        "mae_price": round(float(pos["mae_price"]), 10),
        "held_seconds": int(held),
        "entry_mode": pos["diag"].get("entry_mode"),
        "entry_quality": pos["diag"].get("quality_reason"),
        "entry_score": pos["diag"].get("score"),
        "entry_immediate": pos["diag"].get("imm_agree"),
        "paper_equity_cad": round(state["paper_equity_cad"], 2),
        "daily_pnl_cad": round(state["daily_pnl"], 2),
    }
    append_csv(TRADES_CSV, row)
    append_order_line(
        f"{utc_iso()} | CLOSE | {symbol} | reason={reason} | "
        f"entry={pos['entry']:.8f} | exit={exit_px:.8f} | "
        f"net=C${pnl:+.2f} | peak=C${pos['peak_net']:+.2f} | held={int(held)}s"
    )
    log(
        f"PAPER_EXIT | {symbol} | {reason} | net=C${pnl:+.2f} | "
        f"peak=C${pos['peak_net']:+.2f} | fees=C${fees:.2f} | "
        f"held={held/60:.1f}m | equity=C${state['paper_equity_cad']:.2f}"
    )
    try:
        if pos.get("opportunity_id"):
            append_opportunity_research({
                "event":"V120_EXIT_RESEARCH","opportunity_id":pos.get("opportunity_id"),
                "symbol":symbol,"entry":float(pos["entry"]),"exit":float(exit_px),
                "held_sec":int(held),"net_cad":float(pnl),
                "peak_net_cad":float(pos["peak_net"]),"exit_reason":reason,
                "mfe_price":float(pos["mfe_price"]),"mae_price":float(pos["mae_price"]),
                "post_exit_horizon_sec":int(ENTRY.PARAMS["post_exit_horizon_sec"])
            })
    except Exception as exc:
        log(f"V120_EXIT_RESEARCH_ERROR | {symbol} | {exc}")
    event("paper_exit", **row)
    if SCALP_TRACK_POST_TRADE:
        add_path_watch(
            state, symbol, exit_px, "POST_TRADE",
            f"{reason}|entry={float(pos['entry']):.8f}|net={pnl:+.2f}",
            pos.get("diag", {}).get("bar_ts")
        )
    del state["positions"][symbol]
    save_state(state)

def manage_position(exchange, state, symbol, live, bid):
    pos = state["positions"].get(symbol)
    if not pos:
        return

    entry = float(pos["entry"])
    risk_dist = float(pos["initial_risk"])
    if risk_dist <= 0:
        return

    est_exit = paper_exit_price(bid)
    net, _ = net_pnl(entry, est_exit, float(pos["amount"]))
    current_r = (bid - entry) / risk_dist
    held = time.time() - float(pos["opened_ts"])

    pos["mfe_price"] = max(float(pos["mfe_price"]), bid)
    pos["mae_price"] = min(float(pos["mae_price"]), bid)
    pos["peak_net"] = max(float(pos["peak_net"]), float(net))
    peak = float(pos["peak_net"])

    # V120: fixed trajectory checkpoints from original signal/entry.
    try:
        if pos.get("opportunity_id"):
            elapsed = int(time.time()) - int(pos.get("v120_signal_ts", int(time.time())))
            done = set(pos.get("v120_checkpoints_done", []))
            for cp in ENTRY.PARAMS["opportunity_checkpoint_sec"]:
                cp=int(cp)
                if elapsed >= cp and cp not in done:
                    entry0=float(pos["entry"])
                    append_opportunity_research({
                        "event":"V120_CHECKPOINT","opportunity_id":pos.get("opportunity_id"),
                        "symbol":symbol,"checkpoint_sec":cp,"price":float(bid),
                        "net_cad":float(net),"entry":entry0,
                        "mfe_pct":((float(pos["mfe_price"])/entry0)-1.0)*100.0,
                        "mae_pct":((float(pos["mae_price"])/entry0)-1.0)*100.0
                    })
                    done.add(cp)
            pos["v120_checkpoints_done"]=sorted(done)
    except Exception as exc:
        log(f"V120_CHECKPOINT_ERROR | {symbol} | {exc}")

    # Structured in-trade path: entry-relative price, MFE/MAE and net P&L.
    checkpoints = (1, 3, 5, 10, 15, 30)
    done_path = set(pos.get("trade_path_done", []))
    held_min = held / 60.0
    for cp in checkpoints:
        if held_min >= cp and cp not in done_path:
            mfe_pct = (float(pos["mfe_price"]) - entry) / entry * 100.0
            mae_pct = (float(pos["mae_price"]) - entry) / entry * 100.0
            log(
                f"TRADE_PATH | {symbol} | +{cp}m | entry={entry:.8f} | bid={bid:.8f} | "
                f"net=C${net:+.2f} | R={current_r:+.2f} | "
                f"MFE={mfe_pct:+.3f}% | MAE={mae_pct:+.3f}%"
            )
            done_path.add(cp)
    pos["trade_path_done"] = sorted(done_path)

    c = live.iloc[-1]
    p = live.iloc[-2]
    bearish_clues = 0
    if float(c["plus_di"]) < float(c["minus_di"]):
        bearish_clues += 1
    if float(c["rsi14"]) < float(p["rsi14"]):
        bearish_clues += 1
    if float(c["macd_hist"]) < float(p["macd_hist"]):
        bearish_clues += 1
    bearish_live = bearish_clues >= 2

    # V120.2 Forex-style trade-state authority.
    # C$0.25 is the economic discovery budget for an UNPROVEN bad entry.
    # A trade must first demonstrate favorable excursion before it earns breathing room.
    mfe_cad = max(0.0, float(peak))
    trade_state = str(pos.get("v120_trade_state", "UNPROVEN"))
    proven_threshold = float(ENTRY.PARAMS["entry_proven_mfe_cad"])
    growing_threshold = float(ENTRY.PARAMS["growing_mfe_cad"])
    runner_threshold = float(ENTRY.PARAMS["runner_mfe_cad"])

    if mfe_cad >= runner_threshold:
        trade_state = "RUNNER"
    elif mfe_cad >= growing_threshold:
        trade_state = "GROWING"
    elif mfe_cad >= proven_threshold and held >= float(ENTRY.PARAMS["entry_proven_min_hold_sec"]):
        trade_state = "PROVEN"
    pos["v120_trade_state"] = trade_state

    reason = None

    # V124: distinguish market-direction failure from NDAX transaction-cost deficit.
    # A newly opened spot trade starts economically negative because of spread + fees.
    # The C$0.25 discovery wall therefore applies to adverse UNDERLYING movement from
    # the entry midpoint, while a separate net catastrophe stop remains absolute.
    entry_mid_v124 = float(pos.get("v124_entry_mid", entry))
    initial_spread_bps_v124 = float(pos.get("spread_bps", 0.0))
    approx_mid_v124 = float(bid) * (1.0 + max(0.0, initial_spread_bps_v124) / 20000.0)
    market_move_cad_v124 = (approx_mid_v124 - entry_mid_v124) * float(pos["amount"])
    pos["v124_market_move_cad"] = market_move_cad_v124

    if trade_state == "UNPROVEN":
        if net <= -float(ENTRY.PARAMS.get("v124_net_catastrophe_stop_cad", 1.50)):
            reason = "V125_NET_CATASTROPHE_STOP"
        elif held >= float(ENTRY.PARAMS.get("v124_market_adverse_grace_sec", 8)):
            if market_move_cad_v124 <= -float(ENTRY.PARAMS.get("v124_market_adverse_max_cad", 0.25)):
                reason = "V125_MARKET_ADVERSE_WALL"

    # PROVEN/GROWING/RUNNER: breathing is earned from positive MFE.
    # Retention floor rises with the quality/size of the demonstrated move.
    if reason is None and trade_state in ("PROVEN","GROWING","RUNNER"):
        if trade_state == "PROVEN":
            giveback_frac = 0.70
            lock_frac = 0.0
        elif trade_state == "GROWING":
            giveback_frac = float(ENTRY.PARAMS["growing_max_giveback_fraction"])
            lock_frac = float(ENTRY.PARAMS["growing_profit_lock_fraction"])
        else:
            # Stronger runner gets more absolute room as peak grows, while retaining
            # a larger fraction of demonstrated profit.
            giveback_frac = float(ENTRY.PARAMS["runner_max_giveback_fraction"])
            lock_frac = float(ENTRY.PARAMS["runner_profit_lock_fraction"])
            if mfe_cad >= 2.0 * runner_threshold:
                giveback_frac = float(ENTRY.PARAMS["strong_runner_max_giveback_fraction"])
                lock_frac = float(ENTRY.PARAMS["strong_runner_profit_lock_fraction"])

        giveback_floor = mfe_cad * (1.0 - giveback_frac)
        profit_lock = max(float(ENTRY.PARAMS["minimum_positive_lock_cad"]), mfe_cad * lock_frac)
        retention_floor = max(giveback_floor, profit_lock if trade_state != "PROVEN" else 0.0)
        pos["v120_retention_floor_cad"] = retention_floor
        pos["v120_allowed_giveback_cad"] = max(0.0, mfe_cad - retention_floor)

        if net <= retention_floor and (bearish_live or trade_state in ("GROWING","RUNNER")):
            reason = "V120_ADAPTIVE_PROFIT_RETENTION"

    # Persist state transitions for learning.
    previous_state = str(pos.get("v120_last_logged_state", ""))
    if trade_state != previous_state:
        try:
            append_opportunity_research({
                "event":"V120_TRADE_STATE","opportunity_id":pos.get("opportunity_id"),
                "symbol":symbol,"trade_state":trade_state,"previous_state":previous_state,
                "held_sec":held,"net_cad":net,"mfe_cad":mfe_cad,
                "retention_floor_cad":pos.get("v120_retention_floor_cad"),
                "allowed_giveback_cad":pos.get("v120_allowed_giveback_cad"),
                "bearish_live":bool(bearish_live)
            })
        except Exception as exc:
            log(f"V120_STATE_LOG_ERROR | {symbol} | {exc}")
        pos["v120_last_logged_state"] = trade_state
    if bid <= float(pos["stop"]):
        reason = "BE_STOP" if pos.get("break_even") else "STOP"
    elif bid >= float(pos["target"]):
        reason = "TARGET"
    elif HARD_CASH_STOP_ENABLED and net <= -HARD_CASH_STOP_CAD:
        reason = "HARD_CASH_STOP"

    if reason is None and current_r >= BREAK_EVEN_TRIGGER_R and net > 0.05:
        new_stop = entry + BREAK_EVEN_LOCK_R * risk_dist
        if new_stop > float(pos["stop"]) and new_stop < bid:
            try:
                new_stop = float(exchange.price_to_precision(symbol, new_stop))
            except Exception:
                pass
            pos["stop"] = new_stop
            pos["break_even"] = True
            log(
                f"BE_LOCKED | {symbol} | R={current_r:.2f} | "
                f"net=C${net:+.2f} | stop={new_stop:.8f}"
            )

    if reason is None and held >= FAST_FAILURE_SECONDS and current_r <= -FAST_FAILURE_R and bearish_live:
        reason = "FAST_DIRECTION_FAILURE"

    mfe_r = (float(pos["mfe_price"]) - entry) / risk_dist
    failed_entry_bearish_ok = (
        bearish_clues >= 2 if FAILED_ENTRY_REQUIRE_2OF3_BEARISH else bearish_clues >= 3
    )
    if (
        reason is None
        and held >= FAILED_ENTRY_GRACE_SECONDS
        and mfe_r <= FAILED_ENTRY_MAX_MFE_R
        and current_r <= -FAILED_ENTRY_EXIT_R
        and failed_entry_bearish_ok
    ):
        reason = "FAILED_ENTRY_EXIT"

    # V8 / MAESTRO V76 emerging-winner conversion:
    # preserve a small winner before it disappears, but only after deterioration/stall.
    if (
        reason is None
        and EMERGING_WINNER_ENABLED
        and EMERGING_WINNER_MIN_PEAK_CAD <= peak < EMERGING_WINNER_MAX_PEAK_CAD
        and net > 0.0
    ):
        emerg_floor = peak * EMERGING_WINNER_KEEP_FRACTION
        since_peak_proxy = held >= EMERGING_WINNER_STALL_SEC
        if net <= emerg_floor and (bearish_clues >= 2 or since_peak_proxy):
            reason = "EMERGING_WINNER_HARVEST"

    # V5: aggressive peak protection. Once NET peak profit reaches the arm
    # threshold, preserve at least 65% of that best NET profit.
    if reason is None and PEAK_PROTECT_ENABLED and peak >= PEAK_PROTECT_MIN_CAD:
        floor = max(PEAK_PROTECT_MIN_FLOOR_CAD, peak * PEAK_PROTECT_FRACTION)
        if net <= floor:
            log(
                f"PEAK65_TRIGGER | {symbol} | peak=C${peak:+.2f} | "
                f"floor=C${floor:+.2f} | net=C${net:+.2f} | "
                f"protect={PEAK_PROTECT_FRACTION*100:.0f}%"
            )
            reason = "PEAK65_PROFIT_LOCK"

    if (
        reason is None
        and STALL_EXIT_ENABLED
        and held >= STALL_EXIT_MIN_HOLD_SEC
        and peak >= STALL_EXIT_MIN_PEAK_CAD
        and net >= STALL_EXIT_MIN_NET_CAD
        and bearish_clues >= 2
    ):
        reason = "PROFIT_STALL_EXIT"

    # V119: preserve a small demonstrated winner if it later weakens.
    if (
        reason is None
        and peak >= float(ENTRY.PARAMS["small_winner_peak_cad"])
        and held >= float(ENTRY.PARAMS["small_winner_min_hold_sec"])
        and net <= float(ENTRY.PARAMS["small_winner_floor_cad"])
        and bearish_live
    ):
        reason = "V119_SMALL_WINNER_RETENTION"

    # V119: do not STALE-close a still-positive-price continuation solely because
    # fees keep net P/L below zero; require actual loss-location or bearish state.
    if reason is None and held >= MAX_HOLD_MINUTES * 60 and current_r <= STALE_MAX_R:
        if (not bool(ENTRY.PARAMS["stale_requires_loss_or_bearish"])) or current_r < 0.0 or bearish_live:
            reason = "STALE"
        else:
            log(f"V119_STALE_CONTINUE | {symbol} | R={current_r:+.2f} | net=C${net:+.2f} | peak=C${peak:+.2f}")

    if reason:
        close_position(state, symbol, bid, reason)
    else:
        save_state(state)


def v121_fast_position_guard(exchange, state):
    """Priority risk loop for already-open positions.

    Entry discovery may scan the full CAD universe, but an open position must not
    wait for its own symbol to come around again. This guard is intentionally
    lightweight: executable quote only, then stop/economic-wall enforcement.
    """
    for psym in list(state.get("positions", {}).keys()):
        pos = state.get("positions", {}).get(psym)
        if not pos:
            continue
        try:
            bid, ask, spr, src = fetch_quote(exchange, psym)
            entry = float(pos["entry"])
            amount = float(pos["amount"])
            est_exit = paper_exit_price(bid)
            net, _ = net_pnl(entry, est_exit, amount)
            pos["mfe_price"] = max(float(pos.get("mfe_price", entry)), bid)
            pos["mae_price"] = min(float(pos.get("mae_price", entry)), bid)
            pos["peak_net"] = max(float(pos.get("peak_net", net)), float(net))

            # Technical price stop is sized from the V121 C$0.25 planned-risk budget.
            if bid <= float(pos["stop"]):
                log(f"V121_FAST_GUARD | {psym} | STOP | net=C${net:+.2f} | src={src}")
                close_position(state, psym, bid, "V121_FAST_RISK_STOP")
                continue

            # V124 fast guard: do not mistake NDAX fee/spread deficit for adverse
            # underlying movement. Protect direction after a short discovery grace.
            if str(pos.get("v120_trade_state", "UNPROVEN")) == "UNPROVEN":
                held_v124 = time.time() - float(pos.get("opened_ts", time.time()))
                entry_mid_v124 = float(pos.get("v124_entry_mid", entry))
                current_mid_v124 = (float(bid) + float(ask)) / 2.0
                market_move_cad_v124 = (current_mid_v124 - entry_mid_v124) * amount
                pos["v124_market_move_cad"] = market_move_cad_v124
                if float(net) <= -float(ENTRY.PARAMS.get("v124_net_catastrophe_stop_cad", 1.50)):
                    log(f"V127_FAST_GUARD | {psym} | NET_CATASTROPHE | net=C${net:+.2f} | market=C${market_move_cad_v124:+.2f} | src={src}")
                    close_position(state, psym, bid, "V125_NET_CATASTROPHE_STOP")
                    continue
                if held_v124 >= float(ENTRY.PARAMS.get("v124_market_adverse_grace_sec", 8)) and market_move_cad_v124 <= -float(ENTRY.PARAMS.get("v124_market_adverse_max_cad", 0.25)):
                    log(f"V127_FAST_GUARD | {psym} | MARKET_ADVERSE | net=C${net:+.2f} | market=C${market_move_cad_v124:+.2f} | held={held_v124:.1f}s | src={src}")
                    close_position(state, psym, bid, "V125_MARKET_ADVERSE_WALL")
                    continue
        except Exception as exc:
            log(f"V121_FAST_GUARD_ERROR | {psym} | {type(exc).__name__}: {exc}")


def v122_execution_order(state, universe):
    """Preserve every execution market but revisit plausible markets first."""
    spreads = state.get("v122_last_spread_bps", {})
    core_rank = {sym: i for i, sym in enumerate(("BTC/CAD","ETH/CAD","XRP/CAD","SOL/CAD","ADA/CAD"))}
    def key(sym):
        if sym in core_rank:
            return (0, core_rank[sym], 0.0, sym)
        seen = float(spreads.get(sym, 1e9))
        return (1 if seen < 1e9 else 2, 0, seen, sym)
    return sorted(list(universe), key=key)


def v122_record_spread(state, symbol, spr):
    state.setdefault("v122_last_spread_bps", {})[symbol] = float(spr)


# ============================================================
# LOGGING / BLOCKS
# ============================================================
def should_log_wait(state, symbol, reason):
    key = f"{symbol}|{reason}"
    now = time.time()
    last = float(state["last_wait_log"].get(key, 0.0))
    if now - last >= WAIT_LOG_SECONDS:
        state["last_wait_log"][key] = now
        return True
    return False

def log_signal_row(symbol, spread, diag):
    append_csv(SIGNALS_CSV, {
        "utc": utc_iso(),
        "symbol": symbol,
        "bar_ts": diag.get("bar_ts"),
        "reason": diag.get("reason"),
        "entry_mode": diag.get("entry_mode"),
        "spread_bps": round(float(spread), 2),
        "buy_trend": int(bool(diag.get("buy_trend"))),
        "buy_transition": int(bool(diag.get("buy_transition"))),
        "buy_opportunity": int(bool(diag.get("buy_opportunity"))),
        "buy_breakout": int(bool(diag.get("buy_breakout"))),
        "buy_pull": int(bool(diag.get("buy_pull"))),
        "buy_pull_recovery": int(bool(diag.get("buy_pull_recovery"))),
        "buy_cont": int(bool(diag.get("buy_cont"))),
        "m30_bull": int(bool(diag.get("m30_bull"))),
        "h1_bull": int(bool(diag.get("h1_bull"))),
        "s9": round(float(diag.get("s9", 0)), 4),
        "s21": round(float(diag.get("s21", 0)), 4),
        "rsi": round(float(diag.get("rsi", 0)), 2),
        "adx": round(float(diag.get("adx", 0)), 2),
        "plus_di": round(float(diag.get("plus_di", 0)), 2),
        "minus_di": round(float(diag.get("minus_di", 0)), 2),
        "ema9": round(float(diag.get("ema9", 0)), 10),
        "ema21": round(float(diag.get("ema21", 0)), 10),
        "ema50": round(float(diag.get("ema50", 0)), 10),
        "gap9_21_atr": round(float(diag.get("gap_atr", 0)), 4),
        "gap21_50_atr": round(float(diag.get("gap21_50_atr", 0)), 4),
        "di_ratio": round(float(diag.get("di_ratio", 0)), 3),
        "stretch_atr": round(float(diag.get("stretch_atr", 0)), 3),
        "raw_cross": int(bool(diag.get("raw_cross"))),
        "weak_cross": int(bool(diag.get("weak_cross"))),
        "exhaustion": int(bool(diag.get("exhaustion"))),
        "continuation_chase": int(bool(diag.get("continuation_chase"))),
        "score": diag.get("score", ""),
        "opp_score": diag.get("opp_score", ""),
        "imm_agree": diag.get("imm_agree", ""),
        "imm_oppose": diag.get("imm_oppose", ""),
        "rsi_delta": round(float(diag.get("rsi_delta", 0)), 3),
        "macd_delta": round(float(diag.get("macd_delta", 0)), 8),
        "move_atr": round(float(diag.get("move_atr", 0)), 4),
    })

def trading_block(state):
    if state["daily_pnl"] <= -MAX_DAILY_LOSS_CAD:
        return "DAILY_LOSS_CAP"
    if state["trades_today"] >= MAX_TRADES_DAY:
        return "TRADE_CAP"
    if time.time() < float(state["pause_until"]):
        return "LOSS_PAUSE"
    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return "POSITION_OPEN"
    return None

def heartbeat(state, universe):
    if time.time() - float(state["last_heartbeat"]) < HEARTBEAT_SECONDS:
        return
    closed = state["wins"] + state["losses"]
    wr = 100.0 * state["wins"] / closed if closed else 0.0
    log(
        f"HEARTBEAT | mode={'PAPER' if PAPER_MODE else 'LIVE'} | "
        f"symbols={len(universe)} | open={len(state['positions'])} | "
        f"paper_equity=C${state['paper_equity_cad']:.2f} | "
        f"dayPnL=C${state['daily_pnl']:+.2f} | trades={state['trades_today']} | "
        f"winRate={wr:.1f}%"
    )
    state["last_heartbeat"] = time.time()
    save_state(state)


# ============================================================
# MAIN
# ============================================================
def v123_fast_pending_guard(exchange, state):
    """Service armed paper limits between universe evaluations.

    V122 could arm a valid limit early in a cycle and then not revisit that symbol for
    several minutes. V123 checks only already-armed symbols between evaluations, so
    passive fills/cancellations are observed promptly without rescanning the universe.
    """
    pending = list(state.get("pending_limits", {}).keys())
    for psym in pending:
        try:
            if psym in state.get("positions", {}):
                continue
            pbid, pask, pspr, _ = fetch_quote(exchange, psym)
            plive = fetch_live_m15(exchange, psym)
            process_paper_limit(exchange, state, psym, pbid, pask, plive)
        except Exception as exc:
            log(f"V123_PENDING_GUARD_ERROR | {psym} | {type(exc).__name__}: {exc}")


def main():
    configure_entry_module()
    log("=" * 90)
    log(
        f"{BOT_VERSION} START | PAPER_MODE={PAPER_MODE} | "
        f"LIVE_ALLOWED={LIVE_ALLOWED} | ENTRY_ENGINE=FOREX_SCALP_CURR_TRANSLATION"
    )

    exchange = build_exchange()
    state = load_state()
    readonly_account_ping(exchange, state)
    universe = choose_universe(exchange)
    log("UNIVERSE | " + ",".join(universe))

    cycle_no = 0
    while True:
        cycle_no += 1
        cycle_started = time.time()
        reset_day(state)
        universe = v122_execution_order(state, universe)
        log(f"LOOP_START | cycle={cycle_no} | universe={len(universe)} | open={len(state['positions'])} | scheduler=V127_2_MISSED_WIN_BLOCK_REPAIR")
        heartbeat(state, universe)
        v121_fast_position_guard(exchange, state)
        v123_fast_pending_guard(exchange, state)

        # V120.1 research-only full CAD spot universe scan.
        if bool(ENTRY.PARAMS["scan_all_markets"]):
            try: v120_scan_all_markets(exchange,state)
            except Exception as exc: log(f"V120_MARKET_SCAN_ERROR | {exc}")

        for symbol_index, symbol in enumerate(universe, start=1):
            try:
                # V123: protect open trades and service armed limits before hunting another entry.
                v121_fast_position_guard(exchange, state)
                v123_fast_pending_guard(exchange, state)
                log(f"EVAL | cycle={cycle_no} | symbol={symbol} | progress={symbol_index}/{len(universe)}")
                heartbeat(state, universe)
                if time.time() < float(state["bad_symbol_until"].get(symbol, 0.0)):
                    continue

                bid, ask, spr, quote_src = fetch_quote(exchange, symbol)
                v122_record_spread(state, symbol, spr)

                m15 = fetch_closed(exchange, symbol, TIMEFRAME, OHLCV_LIMIT)
                live = fetch_live_m15(exchange, symbol)
                process_path_watch(state, symbol, bid, ask, live)

                if symbol in state["positions"]:
                    manage_position(exchange, state, symbol, live, bid)
                    continue

                state.setdefault("pending_limits", {})
                if symbol in state["pending_limits"]:
                    if process_paper_limit(exchange, state, symbol, bid, ask, live):
                        continue
                    if symbol in state.get("pending_limits", {}):
                        continue

                block = trading_block(state)
                if block:
                    continue

                state.setdefault("early_reclaim", {})
                early_diag = process_early_reclaim(
                    exchange, state, symbol, bid, ask, spr, live, m15
                )
                if early_diag is not None:
                    early_plan, early_err = calculate_plan(
                        exchange, symbol, ask, spr, m15,
                        float(state["paper_equity_cad"]), early_diag
                    )
                    if early_err:
                        log(f"{symbol} | EARLY_RECLAIM_BLOCK | {early_err}")
                    else:
                        log_signal_row(symbol, spr, early_diag)
                        if PAPER_MODE and PAPER_LIMIT_ENTRY_ENABLED:
                            arm_paper_limit(state, symbol, bid, ask, early_plan, early_diag)
                            continue
                        elif PAPER_MODE:
                            open_position(state, symbol, early_plan, ask, spr, early_diag)
                            continue
                        elif LIVE_ALLOWED:
                            order = exchange.create_order(symbol, "market", "buy", early_plan["amount"])
                            log(f"LIVE_BUY_SENT | {symbol} | orderId={order.get('id')} | EARLY_RECLAIM")
                            continue
                if time.time() < float(state["symbol_cooldown_until"].get(symbol, 0.0)):
                    continue

                # V9: execution economics gate comes BEFORE all entry lanes.
                atr_fast = float(m15.iloc[-1]["atr14"])
                spread_atr_fast = (ask - bid) / atr_fast if atr_fast > 0 else 999.0
                active_spread_cap = min(MAX_SPREAD_BPS, symbol_spread_cap(symbol))
                if spr > active_spread_cap:
                    if should_log_wait(state, symbol, "SPREAD_TOO_WIDE"):
                        log(
                            f"{symbol} | WAIT | SPREAD_TOO_WIDE | "
                            f"book={spr:.1f}bps | sprATR={spread_atr_fast:.2f} | src={quote_src}"
                        )
                    continue
                if spread_atr_fast > MAX_SPREAD_ATR_RATIO:
                    if should_log_wait(state, symbol, "SPREAD_ATR_TOO_HIGH"):
                        log(
                            f"{symbol} | WAIT | SPREAD_ATR_TOO_HIGH | "
                            f"book={spr:.1f}bps | sprATR={spread_atr_fast:.2f}"
                        )
                    continue

                m30 = fetch_closed(exchange, symbol, M30_TIMEFRAME, TREND_LIMIT)
                h1 = fetch_closed(exchange, symbol, H1_TIMEFRAME, TREND_LIMIT)

                # Shared FOREX-style ENTRY module.
                v118_m5 = fetch_live_tf(exchange, symbol, "5m", 140)
                v118_m1 = fetch_live_tf(exchange, symbol, "1m", 180)
                if ENTRY.V118_ENABLED:
                    v118_ok, v118_diag, learn_ok, lane_stat = ENTRY.assess_long(
                        symbol,bid,ask,v118_m1,v118_m5,m15,m30,"CRYPTO_M5_SOURCE"
                    )
                    v118_lane=v118_diag.get("entry_mode","EL000_NONE")
                    if v118_ok:
                        if not learn_ok:
                            if lane_stat:
                                log(f"{symbol} | V118_LEARNING_SHADOW | {v118_lane} | n={lane_stat['n']} | exp=C${lane_stat['expectancy']:+.3f}")
                        else:
                            v118_plan,v118_err=calculate_plan(exchange,symbol,ask,spr,m15,float(state["paper_equity_cad"]),v118_diag)
                            if v118_err:
                                log(f"{symbol} | V118_ENTRY_BLOCK | {v118_err}")
                            else:
                                log_signal_row(symbol,spr,v118_diag)
                                log(f"V118_ENTRY_READY | M5 | {symbol} | lane={v118_lane} | IA={v118_diag['imm_agree']}/3 | M1+M5={v118_diag['score']}/6")
                                ENTRY.record_executed(symbol,bid,ask,v118_m1,v118_m5,m15,m30,"CRYPTO_M5_SOURCE",v118_lane)
                                if PAPER_MODE and PAPER_LIMIT_ENTRY_ENABLED:
                                    arm_paper_limit(state,symbol,bid,ask,v118_plan,v118_diag); continue
                                elif PAPER_MODE:
                                    open_position(state,symbol,v118_plan,ask,spr,v118_diag); continue
                                elif LIVE_ALLOWED:
                                    order=exchange.create_order(symbol,"market","buy",v118_plan["amount"])
                                    log(f"LIVE_BUY_SENT | {symbol} | orderId={order.get('id')} | {v118_lane}"); continue

                m5_live = None
                m1_live = None
                maestro_pre = maestro_precandidate(m15, m30, h1)
                if maestro_pre:
                    m5_live = v118_m5
                    m1_live = v118_m1

                # Strict historical V103-like lane first when full macro aligns.
                mv_ok, mv_diag = (False, None)
                if maestro_pre:
                    mv_ok, mv_diag = maestro_v103_candidate(
                        m15, live, m30, h1, m5_live, m1_live, bid, ask
                    )
                if mv_ok:
                    mv_plan, mv_err = calculate_plan(
                        exchange, symbol, ask, spr, m15,
                        float(state["paper_equity_cad"]), mv_diag
                    )
                    if mv_err:
                        if should_log_wait(state, symbol, mv_err):
                            log(f"{symbol} | MAESTRO_V103_BLOCK | {mv_err}")
                    else:
                        log_signal_row(symbol, spr, mv_diag)
                        log(
                            f"MAESTRO_V103_READY | {symbol} | mode={mv_diag['entry_mode']} | "
                            f"M5={mv_diag.get('m5_score',0)}/3 | M1={mv_diag.get('m1_score',0)}/3 | "
                            f"Immediate={mv_diag.get('imm_agree',0)}/3 | "
                            f"ADX={mv_diag['adx']:.1f} | DIDom={mv_diag['di_dom']:.1f} | "
                            f"stretch={mv_diag['stretch_atr']:.2f}ATR | spread={spr:.1f}bps"
                        )
                        if PAPER_MODE and PAPER_LIMIT_ENTRY_ENABLED:
                            arm_paper_limit(state, symbol, bid, ask, mv_plan, mv_diag)
                            continue
                        elif PAPER_MODE:
                            open_position(state, symbol, mv_plan, ask, spr, mv_diag)
                            continue

                # V76 HEALTHY_TREND translation. This is the important H1-exception
                # lane for crypto when M30 + strong M15 + M5/M1 all agree.
                mh_ok, mh_diag = (False, None)
                if maestro_pre:
                    mh_ok, mh_diag = maestro_healthy_trend_candidate(
                        m15, live, m30, h1, m5_live, m1_live, bid, ask
                    )
                if mh_diag is not None and not mh_ok and should_log_wait(state, symbol, mh_diag.get("reason","MAESTRO_HEALTHY_WAIT")):
                    log(
                        f"{symbol} | MAESTRO_HEALTHY_WAIT | "
                        f"M5={mh_diag.get('m5_score',0)}/3 | M1={mh_diag.get('m1_score',0)}/3 | "
                        f"Immediate={mh_diag.get('imm_agree',0)}/3 | Oppose={mh_diag.get('imm_oppose',0)}/3"
                    )
                if mh_ok:
                    mh_plan, mh_err = calculate_plan(
                        exchange, symbol, ask, spr, m15,
                        float(state["paper_equity_cad"]), mh_diag
                    )
                    if mh_err:
                        if should_log_wait(state, symbol, mh_err):
                            log(f"{symbol} | MAESTRO_HEALTHY_BLOCK | {mh_err}")
                    else:
                        log_signal_row(symbol, spr, mh_diag)
                        log(
                            f"MAESTRO_HEALTHY_READY | {symbol} | "
                            f"M30=UP | H1={'UP' if mh_diag['h1_bull'] else 'EXCEPTION'} | "
                            f"M5={mh_diag.get('m5_score',0)}/3 | M1={mh_diag.get('m1_score',0)}/3 | "
                            f"Immediate={mh_diag.get('imm_agree',0)}/3 | "
                            f"ADX={mh_diag['adx']:.1f} | DIDom={mh_diag['di_dom']:.1f} | "
                            f"stretch={mh_diag['stretch_atr']:.2f}ATR | spread={spr:.1f}bps"
                        )
                        if PAPER_MODE and PAPER_LIMIT_ENTRY_ENABLED:
                            arm_paper_limit(state, symbol, bid, ask, mh_plan, mh_diag)
                            continue
                        elif PAPER_MODE:
                            open_position(state, symbol, mh_plan, ask, spr, mh_diag)
                            continue

                # V7 first opportunity: live pullback recovery near EMA21.
                rec_ok, rec_diag = live_recovery_entry(m15, live, m30, h1, bid, ask)
                if rec_ok:
                    rec_plan, rec_err = calculate_plan(
                        exchange, symbol, ask, spr, m15,
                        float(state["paper_equity_cad"]), rec_diag
                    )
                    if rec_err:
                        if should_log_wait(state, symbol, rec_err):
                            log(f"{symbol} | LIVE_RECOVERY_BLOCK | {rec_err}")
                    else:
                        log_signal_row(symbol, spr, rec_diag)
                        log(
                            f"LIVE_RECOVERY_READY | {symbol} | "
                            f"RSI={rec_diag['rsi']:.1f} | RSId={rec_diag['rsi_delta']:+.2f} | "
                            f"DI={rec_diag['plus_di']:.1f}/{rec_diag['minus_di']:.1f} | "
                            f"stretch={rec_diag['stretch_atr']:.2f}ATR | spread={spr:.1f}bps"
                        )
                        if PAPER_MODE and PAPER_LIMIT_ENTRY_ENABLED:
                            arm_paper_limit(state, symbol, bid, ask, rec_plan, rec_diag)
                            continue
                        elif PAPER_MODE:
                            open_position(state, symbol, rec_plan, ask, spr, rec_diag)
                            continue

                # V6/V7 current-bar trend restart.
                live_ok, live_diag = live_trend_entry(m15, live, m30, h1, bid, ask)
                live_cooldown_ok = (
                    time.time() - float(state.setdefault("last_entry_ts", {}).get(symbol, 0.0))
                    >= LIVE_ENTRY_COOLDOWN_SEC
                )
                if live_ok and live_cooldown_ok:
                    live_plan, live_err = calculate_plan(
                        exchange, symbol, ask, spr, m15,
                        float(state["paper_equity_cad"]), live_diag
                    )
                    if live_err:
                        if should_log_wait(state, symbol, live_err):
                            log(f"{symbol} | LIVE_ENTRY_BLOCK | {live_err}")
                    else:
                        log_signal_row(symbol, spr, live_diag)
                        log(
                            f"LIVE_TREND_ENTRY_READY | {symbol} | mode={live_diag['entry_mode']} | "
                            f"score={live_diag['score']}/4 | clues={live_diag['imm_agree']}/4 | "
                            f"RSI={live_diag['rsi']:.1f} | RSId={live_diag['rsi_delta']:+.2f} | "
                            f"MACDd={live_diag['macd_delta']:+.8f} | "
                            f"DI={live_diag['plus_di']:.1f}/{live_diag['minus_di']:.1f} | "
                            f"stretch={live_diag['stretch_atr']:.2f}ATR | spread={spr:.1f}bps"
                        )
                        if PAPER_MODE and PAPER_LIMIT_ENTRY_ENABLED:
                            arm_paper_limit(state, symbol, bid, ask, live_plan, live_diag)
                            continue
                        elif PAPER_MODE:
                            open_position(state, symbol, live_plan, ask, spr, live_diag)
                            continue
                        elif LIVE_ALLOWED:
                            order = exchange.create_order(symbol, "market", "buy", live_plan["amount"])
                            log(f"LIVE_BUY_SENT | {symbol} | orderId={order.get('id')} | {live_diag['entry_mode']}")
                            continue

                setup_ok, diag = scalp_closed_setup(m15, m30, h1)
                bar_ts = diag.get("bar_ts")

                # Do not enter twice from the same closed M15 signal bar.
                if state["last_entry_bar"].get(symbol) == bar_ts:
                    continue

                atrv = float(diag.get("atr") or m15.iloc[-1]["atr14"])
                spread_atr = (ask - bid) / atrv if atrv > 0 else 999.0

                if not setup_ok:
                    log_signal_row(symbol, spr, diag)
                    if should_log_wait(state, symbol, diag["reason"]):
                        log(
                            f"{symbol} | WAIT | {diag['reason']} | "
                            f"Trend={int(diag.get('buy_trend',0))} "
                            f"Trans={int(diag.get('buy_transition',0))} "
                            f"Opp={int(diag.get('buy_opportunity',0))} "
                            f"Break={int(diag.get('buy_breakout',0))} "
                            f"Pull={int(diag.get('buy_pull',0))} "
                            f"Rec={int(diag.get('buy_pull_recovery',0))} "
                            f"Cont={int(diag.get('buy_cont',0))} | "
                            f"S9={float(diag.get('s9',0)):+.3f} | "
                            f"S21={float(diag.get('s21',0)):+.3f} | "
                            f"RSI={float(diag.get('rsi',0)):.1f} | "
                            f"ADX={float(diag.get('adx',0)):.1f} | "
                            f"DI={float(diag.get('plus_di',0)):.1f}/{float(diag.get('minus_di',0)):.1f} | "
                            f"M30={'UP' if diag.get('m30_bull') else 'DOWN'} | "
                            f"H1={'UP' if diag.get('h1_bull') else 'DOWN'} | "
                            f"EMA9/21/50={float(diag.get('ema9',0)):.6f}/"
                            f"{float(diag.get('ema21',0)):.6f}/"
                            f"{float(diag.get('ema50',0)):.6f} | "
                            f"G21_50={float(diag.get('gap21_50_atr',0)):+.3f}ATR | "
                            f"stretch={float(diag.get('stretch_atr',0)):.2f}"
                        )
                    if (
                        SCALP_TRACK_REJECTED
                        and diag.get("reason") in {"M15_TREND_SETUP", "WAIT_WEAK_TRANSITION"}
                        and state.get("tracked_shadow_bar", {}).get(symbol) != bar_ts
                    ):
                        shadow_mid = (bid + ask) / 2.0
                        add_path_watch(state, symbol, shadow_mid, "REJECTED_STRUCTURE", str(diag.get("reason")), bar_ts)
                        state.setdefault("tracked_shadow_bar", {})[symbol] = bar_ts
                        save_state(state)
                    continue

                mid = (bid + ask) / 2.0
                quality_ok, q = immediate_quality(live, m15.iloc[-1], mid, atrv)
                diag.update({
                    "score": q["score"],
                    "opp_score": q["opp_score"],
                    "imm_agree": q["agree"],
                    "imm_oppose": q["oppose"],
                    "rsi_delta": q["rsi_delta"],
                    "macd_delta": q["mac_delta"],
                    "move_atr": q["move_atr"],
                    "quality_reason": q["reason"],
                    "live_swing": q["swing"],
                    "di_dom": q["di_dom"],
                })

                if not quality_ok:
                    diag["reason"] = q["reason"]
                    log_signal_row(symbol, spr, diag)
                    if (
                        SCALP_TRACK_REJECTED
                        and q["reason"] in {"SCALP_STRONG_2OF3_NOT_CONFIRMED", "SCALP_NEED_3OF3_OR_STRONG_2OF3"}
                        and state.get("tracked_reject_bar", {}).get(symbol) != bar_ts
                    ):
                        add_path_watch(state, symbol, mid, "REJECTED_NEAR_ENTRY", q["reason"], bar_ts)
                        state.setdefault("tracked_reject_bar", {})[symbol] = bar_ts
                        save_state(state)

                    # V4: a structurally good rejection becomes an actionable early-reclaim watch.
                    if (
                        EARLY_RECLAIM_ENABLED
                        and q["reason"] in {"SCALP_STRONG_2OF3_NOT_CONFIRMED", "SCALP_NEED_3OF3_OR_STRONG_2OF3"}
                        and q["score"] >= 3
                        and q["opp_score"] <= 1
                        and q.get("agree", 0) >= 1
                        and q.get("swing") != "DOWN"
                        and float(diag.get("plus_di", 0.0)) > float(diag.get("minus_di", 0.0))
                        and float(diag.get("stretch_atr", 99.0)) <= 1.05
                    ):
                        existing = state.get("early_reclaim", {}).get(symbol)
                        if not existing or existing.get("bar_ts") != bar_ts:
                            arm_early_reclaim(state, symbol, mid, diag, q)
                    if should_log_wait(state, symbol, diag["reason"]):
                        log(
                            f"{symbol} | WAIT | {q['reason']} | "
                            f"mode={diag['entry_mode']} | score={q['score']}/4 opp={q['opp_score']}/4 | "
                            f"Immediate={q['agree']}/3 oppose={q['oppose']}/3 | "
                            f"RSId={q['rsi_delta']:+.2f} | MACDd={q['mac_delta']:+.8f} | "
                            f"Swing={q['swing']} | MoveATR={q['move_atr']:+.3f} | "
                            f"ADX={diag['adx']:.1f} | DIDom={q['di_dom']:.1f}"
                        )
                    continue

                diag["reason"] = q["reason"]
                log_signal_row(symbol, spr, diag)

                plan, err = calculate_plan(
                    exchange, symbol, ask, spr, m15,
                    float(state["paper_equity_cad"]), diag
                )
                if err:
                    if should_log_wait(state, symbol, err):
                        log(f"{symbol} | ENTRY_BLOCK | {err} | sizingRisk=C${RISK_CAD:.2f} | economicWall=C${float(ENTRY.PARAMS.get(chr(101)+chr(110)+chr(116)+chr(114)+chr(121)+chr(95)+chr(97)+chr(100)+chr(118)+chr(101)+chr(114)+chr(115)+chr(101)+chr(95)+chr(109)+chr(97)+chr(120)+chr(95)+chr(99)+chr(97)+chr(100),0.25)):.2f}")
                    if (
                        SCALP_TRACK_REJECTED
                        and state.get("tracked_reject_bar", {}).get(symbol) != bar_ts
                    ):
                        add_path_watch(state, symbol, mid, "REJECTED_COST_OR_PLAN", err, bar_ts)
                        state.setdefault("tracked_reject_bar", {})[symbol] = bar_ts
                        save_state(state)
                    continue

                log(
                    f"SCALP_CURR_ENTRY_READY | {symbol} | mode={diag['entry_mode']} | "
                    f"quality={q['reason']} | score={q['score']}/4 | "
                    f"Immediate={q['agree']}/3 | RSId={q['rsi_delta']:+.2f} | "
                    f"MACDd={q['mac_delta']:+.8f} | Swing={q['swing']} | "
                    f"ADX={diag['adx']:.1f} | DIDom={q['di_dom']:.1f} | "
                    f"stretch={diag['stretch_atr']:.2f} | risk=C${plan['planned_risk_cad']:.2f}"
                )

                if PAPER_MODE and PAPER_LIMIT_ENTRY_ENABLED:
                    arm_paper_limit(state, symbol, bid, ask, plan, diag)
                elif PAPER_MODE:
                    open_position(state, symbol, plan, ask, spr, diag)
                elif LIVE_ALLOWED:
                    # Intentionally conservative: this test build is designed for paper validation.
                    # Live path remains unavailable unless explicitly armed and PAPER mode disabled.
                    order = exchange.create_order(symbol, "market", "buy", plan["amount"])
                    log(f"LIVE_BUY_SENT | {symbol} | orderId={order.get('id')}")
                else:
                    log(f"{symbol} | ENTRY_BLOCK | LIVE_NOT_ARMED")

            except Exception as exc:
                log(f"{symbol} | ERROR | {type(exc).__name__}: {exc}")
                state["bad_symbol_until"][symbol] = (
                    time.time() + BAD_SYMBOL_COOLDOWN_SECONDS
                )
                save_state(state)

        log(f"LOOP_END | cycle={cycle_no} | universe={len(universe)} | elapsed_sec={time.time()-cycle_started:.1f}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
