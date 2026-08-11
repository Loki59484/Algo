import argparse
import datetime as dt
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Configure Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.upstox_methods import DATA_DIR, UpstoxClient
from core.datatypes import FinancialState

logger = logging.getLogger(__name__)


class Planner:
    """Manages trading plans, chronological targets, and CLI reporting."""

    def __init__(self):
        self.ustox = UpstoxClient()
        self.checkpoint_path = DATA_DIR / "target_checkpoints.json"
        
        # Load Shared Financial State
        self.financial_state_file = ROOT_DIR / "tools" / "finance_state.json"
        self.financial_state = FinancialState.load_from_file(self.financial_state_file)

        # Trading & Calculation state 
        self.target: float = 0.0
        self.pnl: float = 0.0
        self.starting_capital: float = self.financial_state.trading_capital
        self.growth_factor: float = self.financial_state.growth_factor

        # Plan state
        self.df_plan: pd.DataFrame = pd.DataFrame()
        self.days_remaining: int = 0
        self.surplus: float = 0.0
        self.chronological_target: float = 0.0
    
    def setup_cli(self) -> argparse.Namespace:
        """Parses command line arguments."""
        parser = argparse.ArgumentParser(description="Launch Trading Planner")
        parser.add_argument("-t", "--target", type=float, default=None, help="The target profit for the plan.")
        parser.add_argument("-s", "--starting", type=float, default=None, help="The starting amount to be invested.")
        parser.add_argument("-m", "--multiplier", type=float, default=None, help="Multiplier for the target profit.")
        parser.add_argument("-f", "--force", action="store_true", help="Force create a new plan, ignoring existing checkpoints.")
        # New CLI flag: Controls whether financial_state.json gets updated
        parser.add_argument("--no-save", action="store_true", help="Do not save overrides to finance_state.json.")
        return parser.parse_args()

    # -------------------------------------------------------------------------
    # PLAN CREATION & LIFECYCLE
    # -------------------------------------------------------------------------
    
    def create(
        self, 
        target: float = None, 
        starting: float = None, 
        multiplier: float = None, 
        force: bool = False,
        save_state: bool = False  # Explicit parameter to toggle persisting to finance_state.json
    ) -> pd.DataFrame:
        self.pnl = self._calculate_current_pnl()

        # 1. Update shared state if arguments are passed
        if target is not None:
            self.financial_state.target = target
        if starting is not None:
            self.financial_state.trading_capital = starting
        if multiplier is not None:
            self.financial_state.growth_factor = multiplier

        # Set calculation variables
        active_target = self.financial_state.target
        self.starting_capital = self.financial_state.trading_capital
        self.growth_factor = self.financial_state.growth_factor

        logger.info("Fetching lifetime history from Upstox API...")
        report, charges = self._fetch_pnl_report()
        
        # 2. Calculate gross target using external PnL from shared state
        past_losses = self.financial_state.external_pnl + charges
        for item in report:
            past_profits = int(item.get("sell_amount", 0)) - int(float(item.get("buy_amount", 0)))
            past_losses -= past_profits
            
        gross_target = round(active_target + past_losses)
        self.target = net_target = gross_target - self.pnl

        # 3. Generate or update the plan
        has_checkpoint = self.checkpoint_path.exists() and self.checkpoint_path.stat().st_size > 0
        
        try:
            if force or not has_checkpoint:
                print("Creating a new plan...")
                df = self._generate_compound_plan(
                    gross_target, 
                    self.starting_capital, 
                    self.growth_factor
                )
            else:
                print("Following the current plan...")
                df = pd.read_json(self.checkpoint_path)

            self._update_booked_pnl(df, report)

            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df["Cumulative_Profit"] = df["Profit"].cumsum()
            df["Status"] = np.where(df["Remainder"] >= net_target, 1, 0)
            
            self._evaluate_current_position(df, net_target)
            
            # Save checkpoints
            df.to_json(self.checkpoint_path)
            self.df_plan = df

            # 4. Conditionally save financial state back to disk
            if save_state:
                logger.info("Persisting financial state updates to disk...")
                self._cache_state()

            if __name__ == "__main__":
                print(df[["Days", "Date", "Profit", "Booked_PnL", "Cumulative_Profit", "Remainder", "Status"]].to_markdown(index=False))
            
            return df

        except Exception as e:
            logger.exception(f"Error evaluating plan: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def _cache_state(self):
        """Saves any updates made to the Financial State back to disk."""
        self.financial_state.save_to_file(self.financial_state_file)

    # -------------------------------------------------------------------------
    # PRIVATE HELPER METHODS
    # -------------------------------------------------------------------------

    def _generate_compound_plan(self, gross_target: float, capital: float, base_multiplier: float) -> pd.DataFrame:
        """Generates the day-by-day mathematical trading plan using volatility-adjusted multipliers."""
        VOLATILITY_WEIGHTS = {
            0: 1.2,  
            1: 1.5,   
            2: 1.2,  
            3: 1.5,   
            4: 0.9,
        }

        n = 0
        pltdata = {"Days": [], "Date": [], "Profit": [], "Booked_PnL": [], "Remainder": []}
        x_nplus1 = round(gross_target)
        current_capital = capital
        current_date = dt.datetime.today().date()
        
        holidays_path = ROOT_DIR / "holidays.json"
        if holidays_path.exists() and holidays_path.stat().st_size != 0:
            holiday_dates = pd.to_datetime(pd.read_json(holidays_path)["date"]).dt.date.values
        else:
            holiday_dates = pd.to_datetime(pd.DataFrame(self.ustox.get_holidays())["date"]).dt.date.values

        while x_nplus1 > 0:
            while current_date.weekday() > 4 or current_date in holiday_dates:
                current_date += dt.timedelta(days=1)
                
            weekday = current_date.weekday()
            day_weight = VOLATILITY_WEIGHTS.get(weekday, 1.0)
            daily_multiplier = base_multiplier * day_weight
            
            prof = round(daily_multiplier * current_capital)
            
            if prof <= 0:
                raise ValueError(f"Calculated daily profit is {prof}. Check starting amount and multiplier.")
            
            prof = min(prof, x_nplus1)
            x_nplus1 -= prof
            
            pltdata["Profit"].append(prof)
            pltdata["Booked_PnL"].append(0.0) 
            pltdata["Remainder"].append(x_nplus1)
            pltdata["Days"].append(n + 1)
            pltdata["Date"].append(current_date.strftime("%Y-%m-%d"))
            
            current_capital += prof
            current_date += dt.timedelta(days=1)
            n += 1

        return pd.DataFrame(pltdata)

    def _evaluate_current_position(self, df: pd.DataFrame, net_target: float) -> None:
        """Determines the current day's target and surplus based on the live PnL."""
        current_idx_arr = df.index[df["Status"] == 1]
        
        if len(current_idx_arr) == 0:
            self.days_remaining = len(df)
        else:
            current_idx = current_idx_arr[-1]
            self.days_remaining = len(df) - (current_idx + 1)
        
        today_str = dt.datetime.today().date().strftime("%Y-%m-%d")
        upcoming_days = df[df["Date"] >= today_str]
        
        target_row = upcoming_days.iloc[0] if not upcoming_days.empty else df.iloc[-1]
        self.chronological_target = target_row["Profit"]
        self.surplus = round(target_row["Remainder"] - net_target, 2)

    def _fetch_pnl_report(self) -> Tuple[List[Dict[str, Any]], int]:
        """Fetches historical PnL and charges from Upstox for relevant financial years."""
        report = []
        total_charges = 0
        
        fyears = {
            2025: {"code": "2526", "start": dt.datetime(2025, 4, 1).date(), "end": dt.datetime(2026, 3, 31).date()},
            2026: {"code": "2627", "start": dt.datetime(2026, 4, 1).date(), "end": dt.datetime(2027, 3, 31).date()}
        }
        
        for fy, data in fyears.items():
            report.extend(self.ustox.get_pnl_report(
                from_date=data["start"], to_date=data["end"], pagenumber=1, pagesize=5000, financial_year=data["code"]
            ))
            total_charges += int(self.ustox.get_charges_report(
                from_date=data["start"], to_date=data["end"], financial_year=data["code"]
            ))
            
        return report, total_charges

    def _update_booked_pnl(self, df: pd.DataFrame, report: List[Dict[str, Any]]) -> None:
        """Updates the plan DataFrame with actual booked PnL history."""
        if "Booked_PnL" not in df.columns:
            df["Booked_PnL"] = 0.0

        historical_pnl_map = {}
        for item in report:
            trade_date = item.get("date", "")[:10] 
            trade_profit = int(item.get("sell_amount", 0)) - int(float(item.get("buy_amount", 0)))
            if trade_date:
                historical_pnl_map[trade_date] = historical_pnl_map.get(trade_date, 0) + trade_profit

        df["Booked_PnL"] = df["Date"].map(historical_pnl_map).fillna(0.0)

        today_str = dt.datetime.today().date().strftime("%Y-%m-%d")
        if today_str in df["Date"].values:
            df.loc[df["Date"] == today_str, "Booked_PnL"] = self.pnl

    def _calculate_current_pnl(self) -> float:
        """Fetches live positions to calculate today's realized PnL."""
        positions = self.ustox.get_positions()
        return sum(item.get("realised", 0.0) for item in positions)

    # -------------------------------------------------------------------------
    # UTILITIES & CLI OUTPUT
    # -------------------------------------------------------------------------

    def generate_trading_dates(self, num_days: int, start_date: dt.date = None) -> List[dt.date]:
        """Generates a list of valid trading dates, skipping weekends and holidays."""
        holidays_path = ROOT_DIR / "holidays.json"
        
        if holidays_path.exists() and holidays_path.stat().st_size != 0:
            holidays = pd.read_json(holidays_path)
        else:
            holidays = pd.DataFrame(self.ustox.get_holidays())
            holidays.to_json(holidays_path)
            
        holiday_dates = pd.to_datetime(holidays["date"]).dt.date.values
        current_date = start_date if start_date else dt.datetime.today().date()
        
        valid_dates = []
        while len(valid_dates) < num_days:
            if current_date.weekday() <= 4 and current_date not in holiday_dates:
                valid_dates.append(current_date)
            current_date += dt.timedelta(days=1)
            
        return valid_dates

    def to_cli(self) -> None:
        """Prints a formatted terminal report of the current trading plan."""
        print("=" * 75)
        print("📊 TRADING PLAN REPORT")
        print("=" * 75)
        print(f"Target Remaining              : ₹{self.target:,.2f}")
        print(f"Day's Starting Amount         : ₹{self.starting_capital:,.2f}")
        print(f"Today's Planned Target        : ₹{self.chronological_target:,.2f}")
        print(f"Today's Current P&L           : ₹{self.pnl:,.2f}")
        
        if self.surplus > 0:
            surplus_text = f"+₹{self.surplus:,.2f} (Ahead of plan 🚀)"
        elif self.surplus < 0:
            surplus_text = f"-₹{abs(self.surplus):,.2f} (Behind plan ⚠️)"
        else:
            surplus_text = "₹0.00 (Exactly on track)"
            
        print(f"Performance Surplus           : {surplus_text}")
        print(f"Projected Trading Days Left   : {self.days_remaining} days")
        
        if self.df_plan is not None and not self.df_plan.empty:
            final_date_str = self.df_plan.iloc[-1]["Date"]
            final_date_obj = dt.datetime.strptime(final_date_str, "%Y-%m-%d").date()
            days_left = (final_date_obj - dt.datetime.today().date()).days
            
            print(f"Projected Normal Days Left    : {days_left} days")
            print(f"Projected Target Date         : {final_date_obj.strftime('%d %B %Y')}")
        print("=" * 75)


if __name__ == "__main__":
    planner = Planner()
    args = planner.setup_cli()
    
    # When executed directly from CLI, save_state defaults to True unless --no-save is explicitly passed
    should_save = not args.no_save
    
    planner.create(
        target=args.target, 
        starting=args.starting, 
        multiplier=args.multiplier, 
        force=args.force,
        save_state=should_save
    )
    planner.to_cli()