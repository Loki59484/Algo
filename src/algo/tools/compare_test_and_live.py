import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TESTER_FILE = ROOT_DIR / "data" / "reports" / "tester_trades.csv"
LIVETRADER_FILE = ROOT_DIR / "data" / "reports" / "livetrader_trades.csv"

def compare_reports():
    if not TESTER_FILE.exists() or not LIVETRADER_FILE.exists():
        print("❌ Error: Both trade reports must exist before running comparison.")
        return

    df_test = pd.read_csv(TESTER_FILE)
    df_live = pd.read_csv(LIVETRADER_FILE)

    print("=" * 70)
    print("⚖️  QUANTITATIVE RECONCILIATION REPORT (TESTER VS LIVETRADER)")
    print("=" * 70)
    print(f"Total Trades Logged  -> Tester: {len(df_test)} | LiveTrader: {len(df_live)}")
    print(f"Total Net PnL        -> Tester: ₹{df_test['net_pnl'].sum():,.2f} | LiveTrader: ₹{df_live['net_pnl'].sum():,.2f}")
    print("-" * 70)

    # --- NEW: DAILY BEHAVIOR ALIGNMENT ---
    compare_daily_behavior(df_test, df_live)

    # --- TRADE-LEVEL MISMATCHES ---
    print("\n🔍 CHECKING FOR EXECUTION DIVERGENCE ON MATCHING DATES...")
    merged = pd.merge(
        df_test, df_live,
        on=["date", "side"],
        suffixes=("_tester", "_livetrader"),
        how="outer"
    )

    # Find trades where they took the same trade but exited differently or at different prices
    mismatches = merged[
        (merged["buy_price_tester"].notna() & merged["buy_price_livetrader"].notna()) &
        ((merged["buy_price_tester"] != merged["buy_price_livetrader"]) |
         (merged["remark_tester"] != merged["remark_livetrader"]))
    ]

    if not mismatches.empty:
        print(f"⚠️ Found {len(mismatches)} trades with execution or pricing divergence:")
        display_cols = [
            "date", "side", 
            "buy_price_tester", "buy_price_livetrader",
            "net_pnl_tester", "net_pnl_livetrader",
            "remark_tester", "remark_livetrader"
        ]
        print(mismatches[display_cols].head(10).to_string(index=False))
    else:
        print("✅ PERFECT EXECUTION CONCURRENCY on matching trades!")
    print("=" * 70)

def compare_daily_behavior(df_test, df_live):
    """Groups trades by date to see if the engines agree on WHEN to trade."""
    
    # Count trades and sum PnL per day for both engines
    test_daily = df_test.groupby('date').agg(
        tester_trades=('trade_id', 'count'),
        tester_pnl=('net_pnl', 'sum')
    ).reset_index()
    
    live_daily = df_live.groupby('date').agg(
        live_trades=('trade_id', 'count'),
        live_pnl=('net_pnl', 'sum')
    ).reset_index()

    # Merge the daily summaries
    daily_compare = pd.merge(test_daily, live_daily, on='date', how='outer').fillna(0)
    
    # Convert counts to integers
    daily_compare['tester_trades'] = daily_compare['tester_trades'].astype(int)
    daily_compare['live_trades'] = daily_compare['live_trades'].astype(int)
    
    # Identify days where the number of trades didn't match
    divergent_days = daily_compare[daily_compare['tester_trades'] != daily_compare['live_trades']]
    
    print("\n📅 DAILY BEHAVIOR ALIGNMENT")
    print(f"Total Unique Trading Days Evaluated: {len(daily_compare)}")
    
    if divergent_days.empty:
        print("✅ Both engines traded on the exact same days with the exact same frequency.")
    else:
        print(f"⚠️ Found {len(divergent_days)} days where the engines disagreed on taking a trade:")
        print("-" * 70)
        print(divergent_days.to_string(index=False))
        print("-" * 70)

if __name__ == "__main__":
    compare_reports()