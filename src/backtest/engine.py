"""Walk-forward backtest engine with realistic costs and funding pass-through.

Inputs
------
weights : DataFrame indexed by rebalance date, columns are symbols. Each row
          should sum to ~0 (dollar neutral) and the gross (sum of |w|) sets the
          notional exposure (gross 1.0 = 100% gross).
returns : DataFrame indexed by daily date, columns are symbols.
funding : DataFrame indexed by daily date, columns are symbols. Daily funding
          rate paid by longs (received by shorts). Use 0 if missing.

Convention
----------
- Weights are forward-filled between rebalance dates (held positions).
- Costs are charged on the day weights change: turnover * (fee+slippage).
- Funding is applied to the *previous day's* held position (overnight).
- Output: portfolio gross/net daily return series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SLIPPAGE_BPS, TAKER_FEE_BPS


def align_weights(weights: pd.DataFrame, returns_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex weights to the daily return index, forward-filling holdings.
    Days before the first rebalance get zero weight."""
    w = weights.reindex(returns_index).ffill().fillna(0.0)
    return w


def apply_backtest(weights: pd.DataFrame, returns: pd.DataFrame,
                   funding: pd.DataFrame | None = None,
                   fee_bps: float = TAKER_FEE_BPS,
                   slip_bps: float = SLIPPAGE_BPS) -> pd.DataFrame:
    cols = sorted(set(weights.columns) & set(returns.columns))
    weights = weights[cols].copy()
    returns = returns[cols].copy()

    daily_w = align_weights(weights, returns.index)

    # gross return: weight * daily return
    gross_pnl = (daily_w.shift(1) * returns).sum(axis=1)

    # turnover: change in weight; first day full position
    dw = daily_w.diff().fillna(daily_w)
    turnover = dw.abs().sum(axis=1)
    cost_per_unit_turnover = (fee_bps + slip_bps) / 10000.0
    cost = turnover * cost_per_unit_turnover

    # funding pass-through: long pays funding, short receives (Binance convention).
    # PnL impact on a long position = -funding * weight; on a short = +|funding| * |weight|
    # Combined: -funding * weight (sign already encodes side).
    if funding is not None and not funding.empty:
        f = funding.reindex(returns.index).reindex(columns=cols).fillna(0.0)
        funding_pnl = -(daily_w.shift(1) * f).sum(axis=1)
    else:
        funding_pnl = pd.Series(0.0, index=returns.index)

    net_pnl = gross_pnl - cost + funding_pnl

    out = pd.DataFrame(
        {
            "gross_ret": gross_pnl,
            "cost": cost,
            "funding_pnl": funding_pnl,
            "net_ret": net_pnl,
            "turnover": turnover,
            "gross_exposure": daily_w.abs().sum(axis=1),
        }
    )
    return out
