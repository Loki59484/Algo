#!/home/loki/Research/Algo/myenv/bin/python
import pandas as pd
import sys
from pathlib import Path
from num2words import num2words
import argparse

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from core.datatypes import to_ist


def setup_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        metavar="FILE",
        help="Accepts path to accept the path to the csv file to be analysed",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = setup_cli()
    if args.file:
        file = args.file
    else:
        file = input("Enter csv file path here: ")
    df = pd.read_csv(file)
    df["Month"] = to_ist(df["buy_timestamp"]).dt.strftime("%Y-%m")

    # Build a comprehensive daily tear sheet
    monthly_analysis_df = (
        df.groupby("Month")
        .agg(
            Total_pnl=("pnl", "sum"),
            Gross_Profit=("pnl", lambda x: x[x > 0].sum()),
            Gross_Loss=("pnl", lambda x: x[x < 0].sum()),
            Total_Trades=("pnl", "count"),
            Win_Rate=(
                "pnl",
                lambda x: (x > 0).mean() * 100,
            ),  # Returns % of winning trades
        )
        .reset_index()
    )
    from matplotlib import pyplot as plt

    # Round the financials for a clean look
    monthly_analysis_df = monthly_analysis_df.round(2)
    profit_df = df[df["pnl"] > 0].reset_index(drop=True)
    total_profit = profit_df["pnl"].sum().round()
    loss_df = df[df["pnl"] < 0].reset_index(drop=True)
    total_loss = loss_df["pnl"].sum().round()
    pd.set_option("display.max_rows", None)
    winning_trades = profit_df["pnl"]
    losing_trades = loss_df["pnl"]

    avg_win = winning_trades.mean() if not winning_trades.empty else 0
    avg_loss = abs(losing_trades.mean()) if not losing_trades.empty else 1
    system_rr = avg_win / avg_loss
    print(monthly_analysis_df)
    print(total_loss, " -- ", num2words(total_loss, lang="en_IN"))
    print("Total lossing trades: ", len(loss_df))
    print(total_profit, " -- ", num2words(total_profit, lang="en_IN"))
    print("Total winnig trades: ", len(profit_df))
    print("Total win rate: ", (len(profit_df) / (len(loss_df) + len(profit_df))) * 100)
    print(f"Average Win: ₹{avg_win:.2f}")
    print(f"Average Loss: ₹{avg_loss:.2f}")
    print(f"True System R:R: 1 : {system_rr:.2f}")
    print("Losing Calls", len(loss_df[(loss_df["side"] == "CE")]))
    print("Losing Puts", len(loss_df[(loss_df["side"] == "PE")]))
    print("Winning Puts", len(profit_df[(profit_df["side"] == "PE")]))
    print("Winning Calls", len(profit_df[(profit_df["side"] == "CE")]))
    