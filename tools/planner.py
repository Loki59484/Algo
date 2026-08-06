import argparse
import datetime as dt
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import set_key

# Configure Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.upstox_methods import DATA_DIR, ENV_PATH, UpstoxClient

logger = logging.getLogger(__name__)


class Planner:
    """Manages trading plans, chronological targets, and CLI reporting."""
    
    DEFAULT_STARTING_AMOUNT = 100000.0
    DEFAULT_LIFETIME_TARGET = 50000000.0

    def __init__(self):
        self.ustox = UpstoxClient()
        self.checkpoint_path = DATA_DIR / "target_checkpoints.json"
        
        # Path to the CFO Tracker's state file
        self.cfo_state_file = ROOT_DIR / "tools" / "finance_state.json"

        # Trading state
        self.target: float = 0.0
        self.groww_pnl: float = 468045.54
        self.starting_amount: float = 0.0
        self.pnl: float = 0.0
        
        # Plan state
        self.df_plan: pd.DataFrame = pd.DataFrame()
        self.days_remaining: int = 0
        self.surplus: float = 0.0
        self.chronological_target: float = 0.0

        # Environment variables
        self.previous_target: str = os.getenv("previous_target", "0")
        self.target_profit: str = os.getenv("target_profit", "0")
        self.prev_days: str = os.getenv("prev_days", "0")
        self.prev_starting: str = os.getenv("prev_starting", "0")

    def setup_cli(self) -> argparse.Namespace:
        """Parses command line arguments."""
        parser = argparse.ArgumentParser(
            description="Launch Trading Planner",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument(
            "-f", "--force", action="store_true", help="Force create a new plan, ignoring existing checkpoints"
        )
        parser.add_argument(
            "-m", "--multiplier", type=float, default=0.1, help="Multiplier for the target profit"
        )
        return parser.parse_args()

    # -------------------------------------------------------------------------
    # PLAN CREATION & LIFECYCLE
    # -------------------------------------------------------------------------
    
    def create(self, args: argparse.Namespace) -> pd.DataFrame:
        """Main entry point to establish the baseline and generate the plan."""
        if args.force:
            self._reset_environment_variables()

        # 1. Establish current live state
        self.pnl = self._calculate_current_pnl()
        self.starting_amount = self._calculate_starting_amount()

        # 2. Establish gross target and fetch historical data
        if args.force or not self.checkpoint_path.exists() or self.checkpoint_path.stat().st_size == 0:
            logger.info("Fetching lifetime history from Upstox API...")
            report, charges = self._fetch_pnl_report()
            
            gross_target = self.groww_pnl + charges
            for item in report:
                gross_target -= int(item.get("sell_amount", 0)) - int(float(item.get("buy_amount", 0)))
            gross_target = round(gross_target)
        else:
            logger.info("Loading baseline target from saved plan...")
            df = pd.read_json(self.checkpoint_path)
            report, _ = self._fetch_pnl_report() # Fetched to update Booked_PnL safely
            
            self._update_booked_pnl(df, report)

        # 3. Establish net target from CFO dashboard settings
        financial_target = self._get_financial_target()
        self.target = net_target = gross_target = financial_target - self.pnl

        # 4. Generate or update the plan
        prev_days_int = int(self.prev_days) if self.prev_days else 0
        force_flag = True if prev_days_int == 0 else args.force
        
        return self.predict_days(net_target, gross_target, self.starting_amount, args.multiplier, force_flag)

    def predict_days(self, net_target: float, gross_target: float, capital: float, multiplier: float, force: bool = False) -> pd.DataFrame:
        """Calculates compounded targets and evaluates plan status."""
        try:
            if not self.checkpoint_path.exists() or self.checkpoint_path.stat().st_size == 0 or force:
                print("Creating a new plan...")
                df = self._generate_compound_plan(gross_target, capital, multiplier)
                
                # ADDED: Instantly map today's PnL so the first CLI print is accurate
                today_str = dt.datetime.today().date().strftime("%Y-%m-%d")
                if today_str in df["Date"].values:
                    df.loc[df["Date"] == today_str, "Booked_PnL"] = self.pnl
            else:
                print("Following the current plan...")
                df = pd.read_json(self.checkpoint_path)

            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df["Cumulative_Profit"] = df["Profit"].cumsum()
            
            # Evaluate Status
            df["Status"] = np.where(df["Remainder"] >= net_target, 1, 0)
            
            self._evaluate_current_position(df, gross_target, net_target, capital)
            
            df.to_json(self.checkpoint_path)
            self.df_plan = df

            if __name__ == "__main__":
                print(df[["Days", "Date", "Profit", "Booked_PnL", "Cumulative_Profit", "Remainder", "Status"]].to_markdown(index=False))
            
            return df

        except Exception as e:
            logger.exception(f"Error predicting days: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # PRIVATE HELPER METHODS
    # -------------------------------------------------------------------------

    def _get_financial_target(self) -> float:
        """Reads the financial target from the CFO Tracker state, falls back to default if missing."""

        try:
            if self.cfo_state_file.exists() and self.cfo_state_file.stat().st_size > 0:
                with open(self.cfo_state_file, "r") as f:
                    state = json.load(f)
                    return float(state.get("target", self.DEFAULT_LIFETIME_TARGET))

        except (ValueError, IOError) as e:
            logger.warning(f"Could not read target from {self.cfo_state_file.name}: {e}. Using default.")
            
        return self.DEFAULT_LIFETIME_TARGET

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
        
        # Pre-fetch holidays to check for valid trading days
        holidays_path = ROOT_DIR / "holidays.json"
        if holidays_path.exists() and holidays_path.stat().st_size != 0:
            holiday_dates = pd.to_datetime(pd.read_json(holidays_path)["date"]).dt.date.values
        else:
            holiday_dates = pd.to_datetime(pd.DataFrame(self.ustox.get_holidays())["date"]).dt.date.values

        while x_nplus1 > 0:
            # 2. Find the next valid trading day (Skip weekends & holidays)
            while current_date.weekday() > 4 or current_date in holiday_dates:
                current_date += dt.timedelta(days=1)
                
            weekday = current_date.weekday()
            
            # 3. Calculate today's specific multiplier
            day_weight = VOLATILITY_WEIGHTS.get(weekday, 1.0)
            daily_multiplier = base_multiplier * day_weight
            
            # 4. Calculate profit based on the running compounded capital
            prof = round(daily_multiplier * current_capital)
            
            if prof <= 0:
                raise ValueError(f"Calculated daily profit is {prof}. Check starting amount and multiplier.")
            
            prof = min(prof, x_nplus1) # Prevent overshooting on the final day
            x_nplus1 -= prof
            
            pltdata["Profit"].append(prof)
            pltdata["Booked_PnL"].append(0.0) 
            pltdata["Remainder"].append(x_nplus1)
            pltdata["Days"].append(n + 1)
            pltdata["Date"].append(current_date.strftime("%Y-%m-%d"))
            
            # 5. Compound the capital for the next day's calculation
            current_capital += prof
            current_date += dt.timedelta(days=1)
            n += 1

        return pd.DataFrame(pltdata)

    def _evaluate_current_position(self, df: pd.DataFrame, gross_target: float, net_target: float, capital: float) -> None:
        """Determines the current day's target and surplus based on the live PnL."""
        current_idx_arr = df.index[df["Status"] == 1]
        
        if len(current_idx_arr) == 0:
            curr_target_str = str(gross_target)
            target_prof_str = str(df["Profit"].iloc[0])
            p_days_str = str(len(df))
        else:
            current_idx = current_idx_arr[-1]
            next_idx = min(current_idx + 1, len(df) - 1)
            curr_target_str = str(df["Remainder"].iloc[current_idx])
            target_prof_str = str(df["Profit"].iloc[next_idx])
            p_days_str = str(len(df) - (current_idx + 1))
            
        self._update_environment_variables(curr_target_str, target_prof_str, p_days_str, str(capital))
        self.days_remaining = int(p_days_str)
        
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
            
        df.to_json(self.checkpoint_path)

    def _calculate_starting_amount(self) -> float:
        """Safely calculates or falls back to a valid starting capital amount."""
        try:
            return self.DEFAULT_STARTING_AMOUNT
        except (TypeError, KeyError):
            try:
                amt = float(self.prev_starting) if self.prev_starting else self.DEFAULT_STARTING_AMOUNT
                return amt if amt > 0 else self.DEFAULT_STARTING_AMOUNT
            except ValueError:
                return self.DEFAULT_STARTING_AMOUNT

    def _calculate_current_pnl(self) -> float:
        """Fetches live positions to calculate today's realized PnL."""
        positions = self.ustox.get_positions()
        return sum(item.get("realised", 0.0) for item in positions)

    def _update_environment_variables(self, curr_target: str, target_prof: str, prev_days: str, prev_start: str) -> None:
        """Updates state parameters safely to memory and .env file."""
        self.previous_target = os.environ["previous_target"] = curr_target
        self.target_profit = os.environ["target_profit"] = target_prof
        self.prev_days = os.environ["prev_days"] = prev_days
        self.prev_starting = os.environ["prev_starting"] = prev_start

        set_key(ENV_PATH, "previous_target", curr_target)
        set_key(ENV_PATH, "target_profit", target_prof)
        set_key(ENV_PATH, "prev_days", prev_days)
        set_key(ENV_PATH, "prev_starting", prev_start)

    def _reset_environment_variables(self) -> None:
        """Resets the environment tracking variables to zero."""
        self._update_environment_variables("0", "0", "0", "0")

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
        print(f"Day's Starting Amount         : ₹{self.starting_amount:,.2f}")
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
    planner.create(args)
    planner.to_cli()