from __future__ import annotations

import logging
from dataclasses import asdict, field
from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from algo.core.datatypes import Funds, Position

logger = logging.getLogger(__name__)


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Trade:
    trade_id: None | str = None
    instrument_key: None | str = None
    side: None | str = None
    buy_timestamp: None | datetime = None
    buy_price: None | float = None
    buy_qty: None | int = None
    sell_timestamp: None | datetime = None
    sell_price: None | float = None
    sell_qty: None | int = None
    movement: None | float = None
    pnl: None | float = None
    remark: None | str = ""
    buy_conditions: None | dict = None
    sell_conditions: None | dict = None
    total: None | float = None
    charges: None | float = None


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class Portfolio:
    funds: Funds = field(default_factory=lambda: Funds(starting_capital=0.0))
    positions: dict[str, Position] = field(default_factory=dict)
    report: list[Trade] = field(default_factory=list)

    def get_report(self, verbose: bool = True) -> pd.DataFrame:
        data = [asdict(trade) for trade in self.report]
        df = pd.json_normalize(data)
        if df.empty:
            return pd.DataFrame()

        if "buy_timestamp" in df.columns:
            df = df.sort_values(by="buy_timestamp").reset_index(drop=True)

        display_df = df.copy()
        if "trade_id" in display_df.columns:
            display_df = display_df.drop(columns=["trade_id"])

        float_cols = display_df.select_dtypes(include=["float"]).columns
        display_df[float_cols] = display_df[float_cols].round(2)

        if verbose:
            from num2words import num2words

            report_cols = [
                c
                for c in [
                    "buy_timestamp",
                    "pnl",
                    "remark",
                    "buy_qty",
                    "sell_qty",
                    "buy_price",
                    "sell_price",
                ]
                if c in display_df.columns
            ]
            print(display_df[report_cols].to_markdown(tablefmt="pretty", floatfmt=".2f"))

            total_mov = display_df["movement"].sum() if "movement" in display_df.columns else 0.0
            final_pnl = display_df["pnl"].sum() if "pnl" in display_df.columns else 0.0
            print(f"Total movement: {total_mov:.2f}")
            print(f"Final pnl: {final_pnl:.2f}")
            print(f"Final pnl (words): {num2words(round(final_pnl), lang='en_IN')}")

        return df
