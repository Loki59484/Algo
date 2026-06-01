from dearpygui.dearpygui import *
import matplotlib.pyplot as plt
from upstox_methods import *
from pprint import pprint
import traceback
import datetime as dt
import pandas as pd
import numpy as np
import config
import os

"""
Edit only checkpoint file if it exists
Create it not. 
"""
ustox = UpstoxClient()

def predicted_days(x1, y1, a, force=False):

    def yn(n):
        yn = ((1 + a) ** (n)) * y1
        return yn

    def profit(y, n):
        p_nplus1 = a * ((1 + a) ** n) * y1
        return round(p_nplus1)

    def xn(x, n):
        xnplus1 = x - ((1 + a) ** n) * y1
        return xnplus1, x - xnplus1

    checkpoint_path = ustox.directory + "/target_checkpoints.json"
    config_path = ustox.directory + "/config.py"
    try:
        if (
            not os.path.exists(checkpoint_path)
            or os.stat(checkpoint_path).st_size == 0
            or force
        ):
            print("Creating a new plan.")
            n = 0
            pltdata = dict(Days=[], Profit=[], Remainder=[], Status=[1])
            x_nplus1 = round(x1)
            prof = 0
            while x_nplus1 > 0:
                pltdata["Profit"].append(prof)
                pltdata["Remainder"].append(x_nplus1)
                pltdata["Days"].append(n)
                prof = profit(y1, n)
                x_nplus1 -= prof
                n += 1

            pltdata["Status"].extend([0] * (len(pltdata["Days"]) - 1))
            df = pd.DataFrame(pltdata)
        else:
            print("Following the current plan.")
            df = pd.read_json(checkpoint_path)
        df["Status"] = np.where(df["Remainder"] >= x1, 1, 0)

        current_idx = df.index[df["Status"] == 1]
        if len(current_idx) == 0:
            current_idx = 0
        else:
            current_idx = current_idx[-1]

        with open(config_path, "r+") as f:
            content = f.readlines()
            for idx, line in enumerate(content):
                if "previous_target" in line:
                    content[idx] = (
                        f"previous_target = {df['Remainder'].iloc[current_idx]}\n"
                    )
                elif "target_profit" in line:
                    content[idx] = (
                        f"target_profit = {df['Profit'].iloc[current_idx+1]}\n"
                    )
                elif "prev_days" in line:
                    content[idx] = f"prev_days = {len(df)-current_idx}\n"
                elif "prev_starting" in line:
                    content[idx] = f"prev_starting = {y1}\n"
            f.seek(0)
            f.writelines(content)
            df.to_json(checkpoint_path)

        if __name__ == "__main__":
            df.plot(x="Days", y="Remainder", title="Predicted Path")
            df.plot(x="Days", y="Profit", title="Predicted Profit")
            print(df)
        return len(df) - current_idx
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)


try:
    funds = ustox.get_funds()
    position = ustox.get_positions()
    # Realized Profil or Loss for the current day.
    pnl = sum(item["realised"] for item in position)
    # The available trading amount for the current day (P&L not included).
    starting_amount = round(
        funds["equity"]["available_margin"] - funds["equity"]["adhoc_margin"] + funds["equity"]["used_margin"]
    )
except TypeError:
    # starting amount if funds available
    starting_amount = config.prev_starting

groww_pnl = target = 468045.54


# Percentage of the starting amount to be targeted per day
target_factor = 0.17
# P&L report and target from upstox
from_date = dt.datetime(2025, 4, 1).date()
size = ustox.get_report_metadata(from_date=from_date)["page_size_limit"]
report = ustox.get_pnl_report(from_date=from_date, pagenumber=1, pagesize=size)
charges = int(ustox.get_charges_report(from_date=from_date))
for item in report:
    target -= int(item["sell_amount"]) - int(float(item["buy_amount"]))
target += charges
target = round(target)

target -= pnl
if config.prev_days == 0:
    projected_days = predicted_days(target, starting_amount, target_factor, True)
else:
    projected_days = predicted_days(target, starting_amount, target_factor)


def report_cli(days, date, target, pnl):
    print("Target: ", target)
    print("Day's Starting Amount: ", starting_amount)
    print("Day's Target: ", config.target_profit)
    print("Today's Profit: ", pnl)
    print("Projected Days: ", days)
    print("Projected Date: ", date)


def projected_date_calc(days):
    if (
        os.path.exists(ustox.directory + "holidays.json")
        and not os.path.getsize(ustox.directory + "holidays.json") == 0
    ):
        holidays = pd.read_json(ustox.directory + "holidays.json")
    else:
        holidays = pd.DataFrame(ustox.get_holidays())
        holidays.to_json(
            ustox.directory + "holidays.json",
        )
    current_date = dt.datetime.today()
    while days != 0:
        if current_date.weekday() > 4 or (
            current_date in holidays["date"]
            and ("NSE" in element for element in holidays["closed_exchanges"])
        ):
            current_date += dt.timedelta(days=1)
            continue
        current_date += dt.timedelta(days=1)
        days -= 1
    return current_date


def report_window(show=True):
    def setup():
        with font_registry():
            with font(f"{ustox.directory}fonts/ProggyClean.ttf", 18) as defaultfont:
                add_font_range_hint(mvFontRangeHint_Default)
        with window(tag="Report", show=show, height=300, width=500):
            add_text(f"Target : {target}")
            add_text(f"Groww target : {groww_pnl}")
            add_text(f"Upstox target : {target-groww_pnl}")
            add_text(
                f"Status : {config.previous_target-target if config.previous_target>target else 0}"
            )
            add_text(f"Day's Target: {config.target_profit}")
            add_text(f"Projected days: {projected_days} days")
            add_text(
                f"Projected date: {projected_date_calc(projected_days).date().strftime('%d %B %Y')}"
            )
        bind_item_font(item="Report", font=defaultfont)

    create_context()
    if not is_dearpygui_running():
        try:
            create_viewport(
                title="Report",
                resizable=True,
                height=300,
                width=500,
            )
            setup()
            set_primary_window("primary", True)
            setup_dearpygui()
            show_viewport()
            start_dearpygui()
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        setup()


if __name__ == "__main__":
    report_cli(
        projected_days,
        projected_date_calc(projected_days).date().strftime("%d %B %Y"),
        target=target,
        pnl=pnl
    )
    # plt.show()
