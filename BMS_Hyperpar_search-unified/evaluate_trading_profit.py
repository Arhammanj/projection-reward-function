"""
Pure electricity trading profit on the held-out test days (300-364).

    Profit_t     = price_sell_t * discharge_energy_t - price_buy_t * charge_energy_t
    Total profit = sum_t Profit_t

No training or agent rollout happens here. The test-day trajectory was
already produced and saved by the latest training run
(newest/report_plots_5000_real.py -> newest/profit_traces_real.npz), which
recorded, for every timestep of days 300-364, the ACTUAL executed
charge/discharge power (after the battery physics clamps the intended
action to stay within [min_soc, max_soc]) and the grid price at that
timestep. This script only re-derives the pure trading-profit formula from
that saved trajectory and reports/saves the requested statistics -- it
never reads the trace's own reward/operation-cost, so degradation cost,
SOC penalties, projection-gap penalties, price-alignment bonus, and every
other reward-shaping term are excluded by construction.

Note on price data: this environment (src/mg_env.py) models a single
time-of-use price series applied symmetrically to both imports and
exports -- there is no separate buy/sell spread in the dataset. price_buy_t
and price_sell_t are therefore numerically identical here; they are kept as
distinct columns so the formula's intent stays explicit and a real bid/ask
spread could be plugged in later.

Usage:
    python evaluate_trading_profit.py
    python evaluate_trading_profit.py --trace-file path/to/other_traces.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.resolve()
DEFAULT_TRACE_FILE = ROOT / 'newest' / 'profit_traces_real.npz'
DEFAULT_OUT_DIR = ROOT / 'evaluation'


def load_timestep_trace(trace_file: Path) -> pd.DataFrame:
    """Load the saved test-day rollout and recompute pure trading profit."""
    data = np.load(trace_file)

    day = data['day'].astype(int)
    hour = data['hour'].astype(int)
    p_ch_kw = data['p_ch'].astype(float)     # executed charge power (kW), >= 0
    p_dch_kw = data['p_dch'].astype(float)   # executed discharge power (kW), >= 0
    price = data['sell_price'].astype(float)

    # Environment timestep length, in hours -- inferred from the data itself
    # (steps per day), so this script has no dependency on the env code.
    steps_per_day = int((day == day[0]).sum())
    dt_hours = 24.0 / steps_per_day

    charge_kwh = p_ch_kw * dt_hours
    discharge_kwh = p_dch_kw * dt_hours
    price_buy = price
    price_sell = price

    profit = price_sell * discharge_kwh - price_buy * charge_kwh

    return pd.DataFrame(dict(
        day=day, hour=hour,
        charge_kwh=charge_kwh, discharge_kwh=discharge_kwh,
        price_buy=price_buy, price_sell=price_sell,
        profit=profit,
    ))


def summarize(timestep_df: pd.DataFrame):
    daily_df = timestep_df.groupby('day', as_index=False).agg(
        profit=('profit', 'sum'),
        energy_charged_kwh=('charge_kwh', 'sum'),
        energy_discharged_kwh=('discharge_kwh', 'sum'),
    )

    summary = dict(
        n_test_days=int(len(daily_df)),
        total_profit=float(daily_df['profit'].sum()),
        avg_daily_profit=float(daily_df['profit'].mean()),
        std_daily_profit=float(daily_df['profit'].std(ddof=0)),
        best_daily_profit=float(daily_df['profit'].max()),
        best_day=int(daily_df.loc[daily_df['profit'].idxmax(), 'day']),
        worst_daily_profit=float(daily_df['profit'].min()),
        worst_day=int(daily_df.loc[daily_df['profit'].idxmin(), 'day']),
        total_energy_discharged_kwh=float(daily_df['energy_discharged_kwh'].sum()),
        total_energy_charged_kwh=float(daily_df['energy_charged_kwh'].sum()),
    )
    return daily_df, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trace-file', type=Path, default=DEFAULT_TRACE_FILE,
                         help='Saved per-timestep test rollout (.npz) to evaluate.')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.trace_file.exists():
        raise FileNotFoundError(
            f"No trace file at {args.trace_file}. This script evaluates an "
            f"already-saved test rollout -- it does not train or run the "
            f"agent. Point --trace-file at a .npz with day/hour/p_ch/p_dch/"
            f"sell_price columns (see newest/report_plots_5000_real.py)."
        )

    timestep_df = load_timestep_trace(args.trace_file)
    daily_df, summary = summarize(timestep_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestep_path = args.out_dir / 'trading_profit_timestep.csv'
    daily_path = args.out_dir / 'trading_profit_daily.csv'
    summary_path = args.out_dir / 'trading_profit_summary.json'

    timestep_df.to_csv(timestep_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    d0, d1 = int(timestep_df['day'].min()), int(timestep_df['day'].max())
    print(f"=== Pure Electricity Trading Profit (test days {d0}-{d1}) ===")
    print(f"Source trace                   : {args.trace_file}")
    print(f"Total trading profit           : {summary['total_profit']:.4f}")
    print(f"Average daily trading profit   : {summary['avg_daily_profit']:.4f}")
    print(f"Std dev of daily profit        : {summary['std_daily_profit']:.4f}")
    print(f"Best daily profit              : {summary['best_daily_profit']:.4f} (day {summary['best_day']})")
    print(f"Worst daily profit             : {summary['worst_daily_profit']:.4f} (day {summary['worst_day']})")
    print(f"Total energy discharged (sold) : {summary['total_energy_discharged_kwh']:.4f} kWh")
    print(f"Total energy charged (bought)  : {summary['total_energy_charged_kwh']:.4f} kWh")

    print(f"\nSaved timestep-level results to: {timestep_path}")
    print(f"Saved daily-level results to   : {daily_path}")
    print(f"Saved summary to               : {summary_path}")


if __name__ == '__main__':
    main()
