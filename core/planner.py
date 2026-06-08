from dearpygui.dearpygui import *
from dotenv import set_key
from upstox_methods import *
from pprint import pprint
import traceback
import datetime as dt
import pandas as pd
import numpy as np
import os
import argparse

# Assuming config and path constants are handled in your environment
from config import config 
logger = logging.getLogger(__name__)

class Planner:
    def __init__(self):
        self.ustox = UpstoxClient()
        
        self.target = 0.0
        self.groww_pnl = 468045.54
        self.starting_amount = 0.0
        self.pnl = 0.0
        
        self.days_remaining = 0
        self.df_plan = None
        
        # New attributes for chronological tracking
        self.surplus = 0.0
        self.chronological_target = 0.0
        
        self.previous_target = os.getenv("previous_target")
        self.target_profit = os.getenv("target_profit")
        self.prev_days = os.getenv("prev_days")
        self.prev_starting = os.getenv("prev_starting")


    def setup_cli(self):
        parser = argparse.ArgumentParser(
            description="Launch Trading Planner",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument('-f', '--force', action='store_true', help="Force create a new plan, ignoring existing checkpoints")
        parser.add_argument('-m', '--multiplier', type=float, default=0.1, help="Multiplier for the target profit")
        return parser.parse_args()

    def predict_days(self, net_target, gross_target, y1, a, force=False):
        def profit(y, n):
            return round(a * ((1 + a) ** n) * y1)

        checkpoint_path = DATA_DIR / "target_checkpoints.json"
        
        try:
            if not os.path.exists(checkpoint_path) or os.stat(checkpoint_path).st_size == 0 or force:
                print("Creating a new plan...")
                n = 0
                
                # Removed the Status array from initialization
                pltdata = dict(Days=[], Date=[], Profit=[], Remainder=[])
                
                # 1. BUILD THE ROADMAP USING GROSS TARGET
                x_nplus1 = round(gross_target)
                
                while x_nplus1 > 0:
                    prof = profit(y1, n)
                    x_nplus1 -= prof # Deduct profit instantly for the end-of-day remainder
                    
                    pltdata["Profit"].append(prof)
                    pltdata["Remainder"].append(x_nplus1)
                    pltdata["Days"].append(n + 1) # Start visually at Day 1
                    
                    n += 1

                trading_dates = self.generate_trading_dates(n)
                pltdata["Date"] = [d.strftime("%Y-%m-%d") for d in trading_dates]
                
                df = pd.DataFrame(pltdata)
            else:
                print("Following the current plan...")
                df = pd.read_json(checkpoint_path)
                
            # Force standard string format to undo Pandas JSON parsing
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            
            # 2. EVALUATE YOUR STATUS USING NET TARGET
            df["Status"] = np.where(df["Remainder"] >= net_target, 1, 0)
            current_idx_arr = df.index[df["Status"] == 1]
            
            # Handle the logic if 0 days are completed vs multiple days completed
            if len(current_idx_arr) == 0:
                # No days fully completed yet
                curr_target_str = str(gross_target)
                target_prof_str = str(df['Profit'].iloc[0])
                p_days_str = str(len(df))
                next_idx = 0
            else:
                # At least one day completed
                current_idx = current_idx_arr[-1]
                next_idx = min(current_idx + 1, len(df) - 1)
                
                curr_target_str = str(df['Remainder'].iloc[current_idx])
                target_prof_str = str(df['Profit'].iloc[next_idx])
                p_days_str = str(len(df) - (current_idx + 1))
                
            # Update env variables safely
            self.previous_target = os.environ["previous_target"] = curr_target_str
            self.target_profit = os.environ["target_profit"] = target_prof_str
            self.prev_days = os.environ["prev_days"] = p_days_str
            self.prev_starting = os.environ["prev_starting"] = f"{y1}"

            set_key(ENV_PATH, "previous_target", self.previous_target)
            set_key(ENV_PATH, "target_profit", self.target_profit)
            set_key(ENV_PATH, "prev_days", self.prev_days)
            set_key(ENV_PATH, "prev_starting", self.prev_starting)
            
            df.to_json(checkpoint_path)
            self.df_plan = df
            self.days_remaining = int(p_days_str)
            
            # --- SURPLUS & CHRONOLOGICAL TRACKING ---
            today_str = dt.datetime.today().date().strftime("%Y-%m-%d")
            
            if today_str in df["Date"].values:
                today_row = df[df["Date"] == today_str].iloc[0]
                self.chronological_target = today_row["Profit"]
                planned_remainder = today_row["Remainder"]
            else:
                # Fallback if checking on a weekend/holiday
                self.chronological_target = df['Profit'].iloc[next_idx]
                planned_remainder = df['Remainder'].iloc[next_idx]
                
            # 3. CALCULATE SURPLUS USING NET TARGET
            self.surplus = round(planned_remainder - net_target, 2)
            
            # Print cleanly without index numbers and without truncating to 10 rows
            if __name__ == '__main__':
                print(df[["Days", "Date", "Profit", "Remainder", "Status"]].to_markdown(index=False))
            return df
            
        except Exception as e:
            traceback.print_exc()
            return pd.DataFrame()

    def create(self, args):
        try:
            if args.force:
                self.previous_target = os.environ["previous_target"]= "0"
                self.target_profit = os.environ["target_profit"]= "0"
                self.prev_days = os.environ["prev_days"]= "0"
                self.prev_starting = os.environ["prev_starting"]= "0"

            funds = self.ustox.get_funds()
            position = self.ustox.get_positions()
            self.pnl = sum(item["realised"] for item in position)
            
            self.starting_amount = round(
                funds["equity"]["available_margin"] - funds["equity"]["adhoc_margin"] + funds["equity"]["used_margin"] 
            )
            self.starting_amount = self.starting_amount if self.starting_amount > 0 else 30000

        except TypeError:
            self.starting_amount = self.prev_starting

        checkpoint_path = DATA_DIR / "target_checkpoints.json"

        # 1. Establish the GROSS TARGET (Baseline roadmap)
        if args.force or not os.path.exists(checkpoint_path) or os.stat(checkpoint_path).st_size == 0:
            logger.info("Fetching lifetime history from Upstox API...")
            gross_target = self.groww_pnl
            
            from_date = dt.datetime(2025, 4, 1).date()
            report = []
            charges = 0
            
            fyears = {
                2025: {"code": "2526", "from_date": dt.datetime(2025, 4, 1).date(), "to_date": dt.datetime(2026, 3, 31).date()},
                2026: {"code": "2627", "from_date": dt.datetime(2026, 4, 1).date(), "to_date": dt.datetime(2027, 3, 31).date()}
            }
            
            for fyear, fyear_data in fyears.items():
                size = 5000
                report.extend(self.ustox.get_pnl_report(from_date=fyear_data["from_date"], to_date=fyear_data["to_date"], pagenumber=1, pagesize=size, financial_year=fyear_data["code"]))
                charges += int(self.ustox.get_charges_report(from_date=fyear_data["from_date"], to_date=fyear_data["to_date"], financial_year=fyear_data["code"]))
                
            for item in report:
                gross_target -= int(item["sell_amount"]) - int(float(item["buy_amount"]))
                
            gross_target += charges
            gross_target = round(gross_target)
            
        else:
            logger.info("Loading baseline target from saved plan...")
            df = pd.read_json(checkpoint_path)
            
            gross_target = df['Remainder'].iloc[0] + df['Profit'].iloc[0]

        # 2. Establish the NET TARGET (Where you are right now after today's P&L)
        net_target = round(gross_target - self.pnl, 2)
        self.target = net_target  # Update class attribute for the UI

        # 3. Pass BOTH to the predictor
        force_flag = True if self.prev_days == 0 else args.force
        return self.predict_days(net_target, gross_target, self.starting_amount, args.multiplier, force_flag)
        

    def to_cli(self):
        print("="*75)
        print("📊 TRADING PLAN REPORT")
        print("="*75)
        print(f"Target Remaining       : ₹{self.target}")
        print(f"Day's Starting Amount  : ₹{self.starting_amount}")
        print(f"Today's Planned Target : ₹{self.chronological_target}")
        print(f"Today's Current P&L    : ₹{self.pnl}")
        
        # Format the surplus dynamically
        if self.surplus > 0:
            surplus_text = f"+₹{self.surplus} (Ahead of plan 🚀)"
        elif self.surplus < 0:
            surplus_text = f"-₹{abs(self.surplus)} (Behind plan ⚠️)"
        else:
            surplus_text = f"₹0.0 (Exactly on track)"
            
        print(f"Performance Surplus    : {surplus_text}")
        print(f"Projected Days Left    : {self.days_remaining} days")
        
        if self.df_plan is not None and not self.df_plan.empty:
            final_date = self.df_plan.iloc[-1]["Date"]
            # Convert string back to a readable format
            final_date_obj = dt.datetime.strptime(final_date, "%Y-%m-%d")
            print(f"Projected Target Date  : {final_date_obj.strftime('%d %B %Y')}")
        print("="*75)

    def generate_trading_dates(self, num_days, start_date=None):
        holidays_path = ROOT_DIR / "holidays.json"
        
        if os.path.exists(holidays_path) and not os.path.getsize(holidays_path) == 0:
            holidays = pd.read_json(holidays_path)
        else:
            holidays = pd.DataFrame(self.ustox.get_holidays())
            holidays.to_json(holidays_path)
            
        holiday_dates = pd.to_datetime(holidays["date"]).dt.date.values
        current_date = start_date if start_date else dt.datetime.today().date()
        
        valid_dates = []
        while len(valid_dates) < num_days:
            # Check if weekend OR if date is in Upstox holiday list
            if current_date.weekday() <= 4 and current_date not in holiday_dates:
                valid_dates.append(current_date)
            current_date += dt.timedelta(days=1)
            
        return valid_dates

    def report_window(self, show=True):
        def setup():
            with font_registry():
                with font(f"{self.ustox.directory}fonts/ProggyClean.ttf", 18) as defaultfont:
                    add_font_range_hint(mvFontRangeHint_Default)
                    
            with window(tag="Report", show=show, height=300, width=500):
                add_text(f"Target : {self.target}")
                add_text(f"Groww target : {self.groww_pnl}")
                add_text(f"Upstox target : {self.target - self.groww_pnl}")
                add_text(f"Status : {max(0, self.previous_target - self.target)}")
                add_text(f"Day's Target: {self.target_profit}")
                add_text(f"Projected days: {self.days_remaining} days")
                if self.projected_date:
                    add_text(f"Projected date: {self.projected_date.date().strftime('%d %B %Y')}")
                    
            bind_item_font(item="Report", font=defaultfont)

        create_context()
        if not is_dearpygui_running():
            try:
                create_viewport(title="Report", resizable=True, height=300, width=500)
                setup()
                set_primary_window("Report", True) # Fixed: Mismatched window tag
                setup_dearpygui()
                show_viewport()
                start_dearpygui()
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            setup()


if __name__ == "__main__":
    planner = Planner()
    args = planner.setup_cli()
    planner.create(args)
    planner.to_cli()
    