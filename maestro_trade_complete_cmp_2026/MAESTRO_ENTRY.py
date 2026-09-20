#!/usr/bin/env python3
"""Shared MAESTRO NDAX crypto entry intelligence, modeled on FOREX MAESTRO_ENTRY.mqh."""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

ENTRY_BUILD = "MAESTRO_ENTRY_V127_2"

# ============================================================
# CENTRAL TRADING PARAMETERS
# Going forward, strategy/entry/trading tuning belongs HERE.
# .env is reserved for credentials, live/paper arming, symbols and paths.
# ============================================================
PARAMS = {
    "maestro_min_stop_atr": 0.95,
    "maestro_rr": 3.00,
    "v118_min_source_rsi": 47.0,
    "v118_max_source_stretch_atr": 1.10,
    "learning_enabled": True,
    "learning_apply_paper": True,
    "learning_apply_live": False,
    "learning_min_samples": 12,
    "learning_min_expectancy_cad": -0.05,
    "learning_shadow_bad_lanes": True,
    # V119 execution/exit tuning: future changes stay here, not in ENV.
    "paper_limit_entry_enabled": True,
    "paper_limit_inside_spread_fraction": 0.35,
    "paper_limit_max_wait_sec": 120,
    "paper_limit_max_chase_bps": 6.0,
    "paper_limit_min_reclaim_bps": 1.0,
    "paper_limit_fallback_enabled": False,
    "paper_limit_fallback_sec": 20,
    "paper_limit_fallback_max_signal_move_bps": 4.0,
    "paper_limit_fallback_max_adverse_bps": 8.0,
    "paper_limit_use_last_touch": True,
    "reclaim_market_max_spread_bps": 25.0,
    "max_hold_minutes": 30.0,
    "stale_max_r": 0.10,
    "stale_requires_loss_or_bearish": True,
    "small_winner_peak_cad": 0.15,
    "small_winner_floor_cad": 0.05,
    "small_winner_min_hold_sec": 120,
    # Entry lane below is based only on clean missed-winner evidence.
    "el301_enabled": True,
    "el301_max_spread_bps": 25.0,
    "el301_min_rsi": 55.0,
    "el301_max_rsi": 68.0,
    "el301_min_adx": 18.0,
    "el301_min_ia": 2,
    "el301_min_m1_score": 3,
    "el301_min_m5_score": 2,
    "el301_min_slope": 0.010,
    "el301_min_move_atr": 0.05,
    "el301_max_move_atr": 0.65,
    "el301_max_stretch_atr": 1.15,
    "el301_max_gap_atr": 1.10,
    # V121 EL302: stricter clean-reclaim propagation fingerprint.
    # Derived from the previously observed clean low-MAE XRP missed-winner pattern;
    # forward validation is required before calling it a proven winning lane.
    "el302_enabled": True,
    "el302_max_spread_bps": 25.0,
    "el302_min_ia": 3,
    "el302_min_m1_score": 3,
    "el302_min_m5_score": 2,
    "el302_min_slope": 0.010,
    "el302_min_move_atr": 0.03,
    "el302_max_move_atr": 0.55,
    "el302_max_stretch_atr": 0.80,
    "el302_max_gap_atr": 0.90,
    "el302_min_adx": 18.0,
    "el302_min_di_dom": 0.0,
    "el302_min_rsi": 54.0,
    "el302_max_rsi": 66.0,

    # V123 EL303: flexible macro-aligned reclaim/propagation lane.
    # Purpose: recover the historically observed low-MAE XRP-style missed-winner family
    # without opening the M15 structural gate globally. Requires M30/H1 bullish context,
    # M1/M5 propagation, positive DI, controlled stretch/gap and executable spread.
    "el303_enabled": True,
    "el303_max_spread_bps": 25.0,
    "el303_min_ia": 2,
    "el303_max_io": 1,
    "el303_min_m1_score": 2,
    "el303_min_m5_score": 2,
    # V127 winner/loser separator: latest EL303 BTC loss was only 4/6 combined;
    # retain the lane but require the stronger 5/6 propagation fingerprint.
    "el303_min_combined_score": 5,
    "el303_min_slope": 0.000,
    "el303_min_move_atr": -0.03,
    "el303_max_move_atr": 0.50,
    "el303_max_stretch_atr": 1.00,
    "el303_max_gap_atr": 1.05,
    "el303_min_adx": 16.0,
    "el303_min_di_dom": 0.0,
    "el303_min_rsi": 50.0,
    "el303_max_rsi": 66.0,

    # V126 EL304: telemetry-derived MACRO PULLBACK SPRING.
    # V124 missed-win evidence repeatedly showed M15_TREND_SETUP blocks with M30/H1
    # bullish context, shallow/temporary M15 weakness, then low-MAE upside expansion.
    # Examples in the supplied batch: XRP +0.719% MFE / 0.000% MAE at 30m,
    # ADA +0.368% / 0.000%, ETH +0.210% / 0.000%, SOL +0.237% / 0.000%.
    # This lane does NOT buy every weak M15 setup: executable spread plus renewed
    # M1/M5 propagation and controlled stretch are still mandatory.
    "el304_enabled": True,
    "el304_max_spread_bps": 25.0,
    "el304_min_ia": 2,
    "el304_max_io": 1,
    "el304_min_m1_score": 2,
    "el304_min_m5_score": 2,
    "el304_min_combined_score": 4,
    "el304_min_slope": -0.020,
    "el304_min_move_atr": -0.10,
    "el304_max_move_atr": 0.45,
    "el304_max_stretch_atr": 0.80,
    "el304_max_gap_atr": 1.10,
    "el304_min_adx": 12.0,
    "el304_min_di_dom": -6.0,
    "el304_min_rsi": 44.0,
    "el304_max_rsi": 58.0,
    "el304_min_m15_gap_atr": -0.35,
    "el304_max_m15_stretch_atr": 0.80,

    # V127.1 — ALL MISSED-WINNER FAMILIES FROM THE LATEST V126 BATCH.
    # 261 unique low-MAE missed-winner candidates (MFE>=0.20%, MAE>=-0.20%)
    # were found across M1/M5/M15 PATH_TRACK. They collapse into four blocker families:
    # M15_TREND_SETUP, SCALP_NEED_3OF3_OR_STRONG_2OF3,
    # SCALP_STRONG_2OF3_NOT_CONFIRMED, and WAIT_WEAK_TRANSITION.
    # These are generalized families, never symbol-specific recipes.
    "el305_enabled": True,   # structural compression -> propagation launch
    "el305_max_spread_bps": 25.0,
    "el305_min_ia": 2,
    "el305_max_io": 1,
    "el305_min_combined_score": 5,
    "el305_min_adx": 12.0,
    "el305_min_di_dom": -3.0,
    "el305_min_rsi": 46.0,
    "el305_max_rsi": 64.0,
    "el305_max_stretch_atr": 0.95,
    "el305_max_gap_atr": 1.05,
    "el305_min_m15_gap_atr": -0.45,

    "el306_enabled": True,   # strong 2-of-3 near-entry confirmation repair
    "el306_max_spread_bps": 25.0,
    "el306_min_ia": 2,
    "el306_max_io": 1,
    "el306_min_combined_score": 5,
    "el306_min_adx": 13.0,
    "el306_min_di_dom": 0.0,
    "el306_min_rsi": 48.0,
    "el306_max_rsi": 66.0,
    "el306_max_stretch_atr": 0.90,
    "el306_max_gap_atr": 0.95,

    "el307_enabled": True,   # weak-transition -> confirmed recovery
    "el307_max_spread_bps": 25.0,
    "el307_min_ia": 2,
    "el307_max_io": 1,
    "el307_min_combined_score": 5,
    "el307_min_adx": 12.0,
    "el307_min_di_dom": -2.0,
    "el307_min_rsi": 45.0,
    "el307_max_rsi": 62.0,
    "el307_max_stretch_atr": 0.85,
    "el307_max_gap_atr": 1.00,
    "el307_min_slope": 0.000,

    # V120 complete opportunity/counterfactual learning instrument.
    "opportunity_tracking_enabled": True,
    "opportunity_horizon_sec": 1800,
    "opportunity_checkpoint_sec": [15,30,60,120,300,600,1200,1800],
    "opportunity_target_cad": [1.0,2.0,5.0,10.0,20.0],
    "counterfactual_entry_delay_sec": [15,30,60,120,300],
    "opportunity_dedupe_sec": 180,
    "post_exit_horizon_sec": 1800,
    # V120.1 full-market learning.
    "scan_all_markets": True,
    "scan_quote": "CAD",
    "scan_interval_sec": 60,
    "scan_timeframes": ["1m","5m","15m","30m"],
    "scan_checkpoint_sec": [15,30,60,120,300,600,1200,1800,3600],
    "scan_target_return_pct": [0.10,0.20,0.50,1.00,2.00],
    "missed_win_min_mfe_pct": 0.20,
    "missed_win_max_mae_pct": 0.20,
    # Risk authority; scanner is observation-only.
    "risk_cad": 3.00,
    "max_daily_loss_cad": 25.00,
    "max_position_cad": 150.00,
    "min_live_notional_cad": 25.00,
    "min_cad_reserve": 30.00,
    "max_open_positions": 2,
    "max_trades_day": 20,
    "max_consec_losses": 3,
    "loss_pause_minutes": 30,
    "symbol_loss_cooldown_minutes": 2,
    # Profit retention research thresholds.
    "retention_be_trigger_r": 0.35,
    "retention_lock_trigger_r": 0.60,
    "retention_lock_floor_r": 0.15,
    "retention_strong_trigger_r": 1.00,
    "retention_strong_giveback_fraction": 0.35,

    # V120.2 Forex-style asymmetric economic authority.
    # Bad/unproven launches may not consume more than C$0.25 economic loss.
    "entry_adverse_max_cad": 0.25,
    "entry_proven_mfe_cad": 0.10,
    "entry_proven_min_hold_sec": 5,
    "entry_zero_mfe_fast_fail_cad": 0.18,
    # V124: entry discovery loss is judged on underlying market movement, not
    # the unavoidable initial fee/spread accounting deficit.
    "v124_market_adverse_max_cad": 0.25,
    "v124_market_adverse_grace_sec": 8,
    "v124_net_catastrophe_stop_cad": 1.50,
    "v124_primary_execution_timeframe": "ALL",

    # Earned breathing for trades that first prove themselves.
    "growing_mfe_cad": 0.35,
    "runner_mfe_cad": 0.75,
    "growing_max_giveback_fraction": 0.55,
    "runner_max_giveback_fraction": 0.42,
    "strong_runner_max_giveback_fraction": 0.35,
    "minimum_positive_lock_cad": 0.05,
    "growing_profit_lock_fraction": 0.20,
    "runner_profit_lock_fraction": 0.35,
    "strong_runner_profit_lock_fraction": 0.50,
}


def _b(n,d=False): return os.getenv(n,str(d)).strip().lower() in {"1","true","yes","y","on"}
def _f(n,d): return float(os.getenv(n,str(d)))
def _i(n,d): return int(os.getenv(n,str(d)))

def opportunity_id(symbol, side, source_mode, signal_ts, signal_price):
    import hashlib
    bucket = int(float(signal_ts) // max(1, int(PARAMS["opportunity_dedupe_sec"])))
    raw = f"{symbol}|{side}|{source_mode}|{bucket}|{round(float(signal_price),8)}"
    return "OPP_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16].upper()

def opportunity_research_spec():
    return {
        "enabled": bool(PARAMS["opportunity_tracking_enabled"]),
        "horizon_sec": int(PARAMS["opportunity_horizon_sec"]),
        "checkpoints_sec": list(PARAMS["opportunity_checkpoint_sec"]),
        "targets_cad": list(PARAMS["opportunity_target_cad"]),
        "counterfactual_delays_sec": list(PARAMS["counterfactual_entry_delay_sec"]),
        "post_exit_horizon_sec": int(PARAMS["post_exit_horizon_sec"]),
    }

def configure(host):
    global utc_iso,event,log,spread_bps,PAPER_MODE,TRADES_CSV,BOT_VERSION,BASE_DIR
    global MAESTRO_MIN_STOP_ATR,MAESTRO_RR
    global V118_ENABLED,V118_MIN_SOURCE_RSI,V118_MAX_SOURCE_STRETCH_ATR
    global LEARNING_ENABLED,LEARNING_APPLY_PAPER,LEARNING_APPLY_LIVE
    global LEARNING_MIN_SAMPLES,LEARNING_MIN_EXPECTANCY_CAD,LEARNING_SHADOW_BAD_LANES,LEARNING_STATE_FILE
    utc_iso=host["utc_iso"]; event=host["event"]; log=host["log"]; spread_bps=host["spread_bps"]
    PAPER_MODE=host["PAPER_MODE"]; TRADES_CSV=host["TRADES_CSV"]; BOT_VERSION=host["BOT_VERSION"]; BASE_DIR=host["BASE_DIR"]
    MAESTRO_MIN_STOP_ATR=host["MAESTRO_MIN_STOP_ATR"]; MAESTRO_RR=host["MAESTRO_RR"]
    V118_ENABLED=True
    V118_MIN_SOURCE_RSI=float(PARAMS["v118_min_source_rsi"])
    V118_MAX_SOURCE_STRETCH_ATR=float(PARAMS["v118_max_source_stretch_atr"])
    LEARNING_ENABLED=bool(PARAMS["learning_enabled"])
    LEARNING_APPLY_PAPER=bool(PARAMS["learning_apply_paper"])
    LEARNING_APPLY_LIVE=bool(PARAMS["learning_apply_live"])
    LEARNING_MIN_SAMPLES=int(PARAMS["learning_min_samples"])
    LEARNING_MIN_EXPECTANCY_CAD=float(PARAMS["learning_min_expectancy_cad"])
    LEARNING_SHADOW_BAD_LANES=bool(PARAMS["learning_shadow_bad_lanes"])
    LEARNING_STATE_FILE=Path(os.getenv("MAESTRO_ENTRY_LEARNING_STATE_FILE",str(BASE_DIR/"Json"/"maestro_entry_crypto_learning.json")))

def _tf_market_state(df, side="BUY"):
    c = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
    p = df.iloc[-3] if len(df) >= 3 else c
    atr = max(float(c["atr14"]), 1e-12)
    close = float(c["close"]); open_ = float(c["open"])
    high = float(c["high"]); low = float(c["low"])
    ema9 = float(c["ema9"]); ema20 = float(c.get("ema20", c["ema21"])); ema50 = float(c["ema50"])
    ema20p = float(p.get("ema20", p["ema21"]))
    sign = 1.0 if side == "BUY" else -1.0
    bb_u = float(c.get("bb_upper", close)); bb_l = float(c.get("bb_lower", close))
    return {
        "atr": atr,
        "move": sign * (float(c["close"]) - float(p["close"])) / atr,
        "dir_body": sign * (close - open_) / atr,
        "body": abs(close - open_) / atr,
        "range": (high - low) / atr,
        "upper_wick": (high - max(open_, close)) / atr,
        "lower_wick": (min(open_, close) - low) / atr,
        "rsi": float(c["rsi14"]),
        "drsi": sign * (float(c["rsi14"]) - float(p["rsi14"])),
        "macd": sign * float(c["macd_hist"]),
        "dmacd": sign * (float(c["macd_hist"]) - float(p["macd_hist"])),
        "adx": float(c["adx14"]),
        "di_dom": sign * (float(c["plus_di"]) - float(c["minus_di"])),
        "align": bool(ema9 > ema20 > ema50) if side == "BUY" else bool(ema9 < ema20 < ema50),
        "slope": sign * (ema20 - ema20p) / atr,
        "gap": abs(ema20 - ema50) / atr,
        "stretch": abs(close - ema20) / atr,
        "structure": bool(high >= float(p["high"]) and low >= float(p["low"]) and close >= float(p["close"]))
                     if side == "BUY" else
                     bool(high <= float(p["high"]) and low <= float(p["low"]) and close <= float(p["close"])),
        "bb_width": (bb_u - bb_l) / atr if bb_u > bb_l else 0.0,
        "bb_pos": (close - bb_l) / (bb_u - bb_l) if bb_u > bb_l else 0.5,
        "vol_ratio": float(c.get("vol_ratio", 0.0)),
    }

def v118_market_snapshot(symbol, event_name, source_mode, lane, bid, ask, m1, m5, m15, m30):
    if not LEARNING_ENABLED:
        return
    mid = (bid + ask) / 2.0
    atr1 = max(float(m1.iloc[-2]["atr14"]), 1e-12)
    recent = m1.iloc[-11:-1] if len(m1) >= 12 else m1.iloc[:-1]
    hi10 = float(recent["high"].max()); lo10 = float(recent["low"].min())
    snap = {
        "build": BOT_VERSION, "event": event_name, "symbol": symbol, "side": "BUY",
        "source_mode": source_mode, "el_lane": lane, "utc": utc_iso(),
        "spread_bps": spread_bps(bid, ask), "spread_atr": (ask-bid)/atr1,
        "dist_high10_atr": (hi10-mid)/atr1, "dist_low10_atr": (mid-lo10)/atr1,
        "breakout10_atr": -(hi10-mid)/atr1,
        "atr1_vs_m5": atr1 / max(float(m5.iloc[-2]["atr14"])/5.0, 1e-12),
        "atr1_vs_m15": atr1 / max(float(m15.iloc[-1]["atr14"])/15.0, 1e-12),
        "m1": _tf_market_state(m1), "m5": _tf_market_state(m5),
        "m15": _tf_market_state(m15), "m30": _tf_market_state(m30),
    }
    event("V1181_MARKET_STATE", **snap)

def _v118_votes(m1, m5):
    c1, p1, pp1 = m1.iloc[-1], m1.iloc[-2], m1.iloc[-3]
    c5, p5, pp5 = m5.iloc[-1], m5.iloc[-2], m5.iloc[-3]
    ia = int(float(p1["close"]) > float(pp1["close"])) \
       + int(float(p1.get("ema20", p1["ema21"])) > float(p1["ema50"])) \
       + int(float(p5.get("ema20", p5["ema21"])) > float(p5["ema50"]))
    io = 3 - ia
    m1s = int(float(p1["close"]) > float(pp1["close"])) \
        + int(float(pp1["close"]) >= float(m1.iloc[-4]["close"])) \
        + int(float(p1.get("ema20", p1["ema21"])) > float(p1["ema50"]))
    m5s = int(float(p5["close"]) > float(pp5["close"])) \
        + int(float(pp5["close"]) >= float(m5.iloc[-4]["close"])) \
        + int(float(p5.get("ema20", p5["ema21"])) > float(p5["ema50"]))
    return ia, io, m1s, m5s

def v118_candidate(m1, m5, m15, m30, bid, ask):
    mid = (bid + ask) / 2.0
    c = m1.iloc[-2]; p = m1.iloc[-5]
    atr = max(float(c["atr14"]), 1e-12)
    e20 = float(c.get("ema20", c["ema21"])); e50 = float(c["ema50"])
    e20p = float(p.get("ema20", p["ema21"])); e50p = float(p["ema50"])
    ia, io, m1s, m5s = _v118_votes(m1, m5)
    a1 = e20 > e50
    p5 = m5.iloc[-2]
    a5 = float(p5.get("ema20", p5["ema21"])) > float(p5["ema50"])
    slope = (e20 - e20p) / atr
    gap = abs(e20-e50) / atr
    widen = gap - abs(e20p-e50p) / atr
    stretch = abs(mid-e20) / atr
    move = (float(c["close"]) - float(m1.iloc[-3]["close"])) / atr
    adx = float(c["adx14"])
    dom = float(c["plus_di"]) - float(c["minus_di"])
    rsi15 = float(m15.iloc[-1]["rsi14"])
    source_ok = rsi15 >= V118_MIN_SOURCE_RSI and stretch <= V118_MAX_SOURCE_STRETCH_ATR

    # V126 macro-pullback context used by EL304.
    m15c = m15.iloc[-1]
    m15_atr = max(float(m15c["atr14"]), 1e-12)
    m15_e21 = float(m15c["ema21"])
    m15_e50 = float(m15c["ema50"])
    m15_gap_atr = (m15_e21 - m15_e50) / m15_atr
    m15_stretch_atr = abs(mid - m15_e21) / m15_atr
    m30_up = bool(float(m30.iloc[-1]["close"]) > float(m30.iloc[-1]["ema50"]))

    # V121 clean-reclaim lane plus V119 missed-winner lane.
    # EL302 is intentionally stricter than EL301: full immediate agreement,
    # aligned M1/M5 structure, low stretch/gap and non-negative DI dominance.
    spread_now = spread_bps(bid, ask)
    c302 = (
        bool(PARAMS["el302_enabled"])
        and spread_now <= float(PARAMS["el302_max_spread_bps"])
        and ia >= int(PARAMS["el302_min_ia"])
        and m1s >= int(PARAMS["el302_min_m1_score"])
        and m5s >= int(PARAMS["el302_min_m5_score"])
        and a1 and a5
        and slope >= float(PARAMS["el302_min_slope"])
        and float(PARAMS["el302_min_move_atr"]) <= move <= float(PARAMS["el302_max_move_atr"])
        and stretch <= float(PARAMS["el302_max_stretch_atr"])
        and gap <= float(PARAMS["el302_max_gap_atr"])
        and adx >= float(PARAMS["el302_min_adx"])
        and dom >= float(PARAMS["el302_min_di_dom"])
        and float(PARAMS["el302_min_rsi"]) <= rsi15 <= float(PARAMS["el302_max_rsi"])
    )
    c303 = (
        bool(PARAMS["el303_enabled"])
        and spread_now <= float(PARAMS["el303_max_spread_bps"])
        and ia >= int(PARAMS["el303_min_ia"])
        and io <= int(PARAMS["el303_max_io"])
        and m1s >= int(PARAMS["el303_min_m1_score"])
        and m5s >= int(PARAMS["el303_min_m5_score"])
        and (m1s + m5s) >= int(PARAMS["el303_min_combined_score"])
        and a1 and a5
        and slope >= float(PARAMS["el303_min_slope"])
        and float(PARAMS["el303_min_move_atr"]) <= move <= float(PARAMS["el303_max_move_atr"])
        and stretch <= float(PARAMS["el303_max_stretch_atr"])
        and gap <= float(PARAMS["el303_max_gap_atr"])
        and adx >= float(PARAMS["el303_min_adx"])
        and dom >= float(PARAMS["el303_min_di_dom"])
        and float(PARAMS["el303_min_rsi"]) <= rsi15 <= float(PARAMS["el303_max_rsi"])
        and bool(float(m30.iloc[-1]["close"]) > float(m30.iloc[-1]["ema50"]))
    )
    c304 = (
        bool(PARAMS["el304_enabled"])
        and spread_now <= float(PARAMS["el304_max_spread_bps"])
        and m30_up
        and ia >= int(PARAMS["el304_min_ia"])
        and io <= int(PARAMS["el304_max_io"])
        and m1s >= int(PARAMS["el304_min_m1_score"])
        and m5s >= int(PARAMS["el304_min_m5_score"])
        and (m1s + m5s) >= int(PARAMS["el304_min_combined_score"])
        and slope >= float(PARAMS["el304_min_slope"])
        and float(PARAMS["el304_min_move_atr"]) <= move <= float(PARAMS["el304_max_move_atr"])
        and stretch <= float(PARAMS["el304_max_stretch_atr"])
        and gap <= float(PARAMS["el304_max_gap_atr"])
        and adx >= float(PARAMS["el304_min_adx"])
        and dom >= float(PARAMS["el304_min_di_dom"])
        and float(PARAMS["el304_min_rsi"]) <= rsi15 <= float(PARAMS["el304_max_rsi"])
        and m15_gap_atr >= float(PARAMS["el304_min_m15_gap_atr"])
        and m15_stretch_atr <= float(PARAMS["el304_max_m15_stretch_atr"])
    )
    # V127.1 generalized missed-winner lanes.
    # EL305 captures the repeated M15_TREND_SETUP family only after lower-timeframe
    # propagation has reasserted itself. EL306 captures the two near-entry 2-of-3
    # blocker families with stronger 5/6 confirmation. EL307 captures the
    # WAIT_WEAK_TRANSITION family after recovery. All retain an economic spread ceiling.
    c305 = (
        bool(PARAMS["el305_enabled"])
        and spread_now <= float(PARAMS["el305_max_spread_bps"])
        and ia >= int(PARAMS["el305_min_ia"]) and io <= int(PARAMS["el305_max_io"])
        and (m1s + m5s) >= int(PARAMS["el305_min_combined_score"])
        and a1 and a5
        and adx >= float(PARAMS["el305_min_adx"])
        and dom >= float(PARAMS["el305_min_di_dom"])
        and float(PARAMS["el305_min_rsi"]) <= rsi15 <= float(PARAMS["el305_max_rsi"])
        and stretch <= float(PARAMS["el305_max_stretch_atr"])
        and gap <= float(PARAMS["el305_max_gap_atr"])
        and m15_gap_atr >= float(PARAMS["el305_min_m15_gap_atr"])
    )
    c306 = (
        bool(PARAMS["el306_enabled"])
        and spread_now <= float(PARAMS["el306_max_spread_bps"])
        and ia >= int(PARAMS["el306_min_ia"]) and io <= int(PARAMS["el306_max_io"])
        and (m1s + m5s) >= int(PARAMS["el306_min_combined_score"])
        and a1 and a5
        and adx >= float(PARAMS["el306_min_adx"])
        and dom >= float(PARAMS["el306_min_di_dom"])
        and float(PARAMS["el306_min_rsi"]) <= rsi15 <= float(PARAMS["el306_max_rsi"])
        and stretch <= float(PARAMS["el306_max_stretch_atr"])
        and gap <= float(PARAMS["el306_max_gap_atr"])
    )
    c307 = (
        bool(PARAMS["el307_enabled"])
        and spread_now <= float(PARAMS["el307_max_spread_bps"])
        and ia >= int(PARAMS["el307_min_ia"]) and io <= int(PARAMS["el307_max_io"])
        and (m1s + m5s) >= int(PARAMS["el307_min_combined_score"])
        and a1 and a5
        and slope >= float(PARAMS["el307_min_slope"])
        and adx >= float(PARAMS["el307_min_adx"])
        and dom >= float(PARAMS["el307_min_di_dom"])
        and float(PARAMS["el307_min_rsi"]) <= rsi15 <= float(PARAMS["el307_max_rsi"])
        and stretch <= float(PARAMS["el307_max_stretch_atr"])
        and gap <= float(PARAMS["el307_max_gap_atr"])
    )

    c301 = (
        bool(PARAMS["el301_enabled"])
        and spread_now <= float(PARAMS["el301_max_spread_bps"])
        and ia >= int(PARAMS["el301_min_ia"])
        and m1s >= int(PARAMS["el301_min_m1_score"])
        and m5s >= int(PARAMS["el301_min_m5_score"])
        and a1 and a5
        and slope >= float(PARAMS["el301_min_slope"])
        and float(PARAMS["el301_min_move_atr"]) <= move <= float(PARAMS["el301_max_move_atr"])
        and stretch <= float(PARAMS["el301_max_stretch_atr"])
        and gap <= float(PARAMS["el301_max_gap_atr"])
        and adx >= float(PARAMS["el301_min_adx"])
        and float(PARAMS["el301_min_rsi"]) <= rsi15 <= float(PARAMS["el301_max_rsi"])
    )
    c20 = ia>=3 and io==0 and m1s>=3 and m5s>=3 and a1 and a5 and slope>=0.070 and widen>=0.010 and adx>=20 and dom>=5 and 0.02<=move<=0.45 and stretch<=0.55 and gap<=0.70
    c10 = ia>=2 and io==0 and m1s>=3 and m5s>=3 and a1 and a5 and slope>=0.050 and widen>=0.005 and adx>=18 and dom>=4 and 0.00<=move<=0.50 and stretch<=0.65 and gap<=0.75
    c5  = ia>=2 and io<=1 and m1s>=3 and m5s>=2 and a1 and a5 and slope>=0.035 and widen>=0.000 and adx>=16 and dom>=3 and 0.00<=move<=0.55 and stretch<=0.75 and gap<=0.80
    # V127: C2/EL201 must show at least 5/6 combined M1+M5 propagation.
    # The latest 4/6 BTC EL201 trade produced only +0.108% MFE and finished negative after costs.
    c2  = ia>=2 and io<=1 and m1s>=2 and m5s>=2 and (m1s+m5s)>=5 and a1 and a5 and slope>=0.020 and widen>=-0.010 and adx>=14 and dom>=2 and -0.02<=move<=0.65 and stretch<=0.95 and gap<=0.90
    lane = "EL302_V121_CLEAN_RECLAIM" if c302 else "EL303_V123_MACRO_RECLAIM" if c303 else "EL304_V126_MACRO_PULLBACK_SPRING" if c304 else "EL305_V1271_STRUCTURAL_PROPAGATION" if c305 else "EL306_V1271_STRONG_2OF3_CONFIRM" if c306 else "EL307_V1271_WEAK_TRANSITION_RECOVERY" if c307 else "EL301_V119_MISSED_WIN_RECLAIM" if c301 else "EL2001_C20_LOW_MAE_RUNNER" if c20 else "EL1001_C10_LOW_MAE_EXPANSION" if c10 else "EL501_C5_LOW_MAE_LAUNCH" if c5 else "EL201_C2_LOW_MAE_PROPAGATION" if c2 else "EL000_NONE"
    ok = bool(source_ok and lane != "EL000_NONE")
    diag = {
        "bar_ts": int(m15.iloc[-1]["timestamp"]), "reason": "V118_ENTRY_PASS" if ok else "V118_ENTRY_LIBRARY_WAIT",
        "entry_mode": lane, "quality_reason": "V118_LOW_MAE_MARKET_STATE",
        "score": m1s + m5s, "opp_score": io, "imm_agree": ia, "imm_oppose": io,
        "rsi_delta": float(c["rsi14"])-float(m1.iloc[-3]["rsi14"]),
        "macd_delta": float(c["macd_hist"])-float(m1.iloc[-3]["macd_hist"]),
        "move_atr": move, "di_dom": dom, "adx": adx, "rsi": rsi15,
        "v126_m15_gap_atr": m15_gap_atr, "v126_m15_stretch_atr": m15_stretch_atr,
        "v126_m30_up": int(m30_up),
        "v1272_missed_win_lane": int(bool(c304 or c305 or c306 or c307)),
        "stretch_atr": stretch, "gap21_50_atr": gap, "s9": slope, "s21": widen,
        "plus_di": float(c["plus_di"]), "minus_di": float(c["minus_di"]),
        "ema9": float(c["ema9"]), "ema21": e20, "ema50": e50,
        "m30_bull": float(m30.iloc[-1]["close"]) > float(m30.iloc[-1]["ema50"]),
        "h1_bull": False, "source_ok": source_ok,
        "plan_min_stop_atr": MAESTRO_MIN_STOP_ATR, "plan_rr": MAESTRO_RR,
    }
    return ok, diag

def learning_lane_stats():
    stats = {}
    if not LEARNING_ENABLED or not TRADES_CSV.exists() or TRADES_CSV.stat().st_size == 0:
        return stats
    try:
        df = pd.read_csv(TRADES_CSV)
        if "entry_mode" not in df.columns or "net_pnl_cad" not in df.columns:
            return stats
        df = df[df["entry_mode"].astype(str).str.startswith("EL")]
        for lane, g in df.groupby("entry_mode"):
            pnl = pd.to_numeric(g["net_pnl_cad"], errors="coerce").dropna()
            if len(pnl):
                stats[str(lane)] = {
                    "n": int(len(pnl)), "wins": int((pnl > 0).sum()),
                    "expectancy": float(pnl.mean()), "net": float(pnl.sum()),
                    "win_rate": float((pnl > 0).mean()),
                }
    except Exception as exc:
        log(f"LEARNING_READ_ERROR | {type(exc).__name__}: {exc}")
    return stats

def learning_allows_lane(lane):
    stats = learning_lane_stats()
    if LEARNING_STATE_FILE:
        try:
            LEARNING_STATE_FILE.write_text(json.dumps({
                "updated_utc": utc_iso(), "build": BOT_VERSION, "lanes": stats
            }, indent=2), encoding="utf-8")
        except Exception:
            pass
    apply = (PAPER_MODE and LEARNING_APPLY_PAPER) or ((not PAPER_MODE) and LEARNING_APPLY_LIVE)
    s = stats.get(lane)
    if not apply or not LEARNING_SHADOW_BAD_LANES or not s or s["n"] < LEARNING_MIN_SAMPLES:
        return True, s
    return s["expectancy"] >= LEARNING_MIN_EXPECTANCY_CAD, s


def assess_long(symbol,bid,ask,m1,m5,m15,m30,source_mode):
    ok,diag=v118_candidate(m1,m5,m15,m30,bid,ask)
    lane=diag.get("entry_mode","EL000_NONE")
    v118_market_snapshot(symbol,"PASS" if ok else "BLOCK",source_mode,lane,bid,ask,m1,m5,m15,m30)
    allowed,stats=learning_allows_lane(lane) if ok else (False,None)
    return ok,diag,allowed,stats

def record_executed(symbol,bid,ask,m1,m5,m15,m30,source_mode,lane):
    v118_market_snapshot(symbol,"EXECUTED",source_mode,lane,bid,ask,m1,m5,m15,m30)
