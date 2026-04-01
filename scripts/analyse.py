import pandas as pd
import sys
from pathlib import Path
from num2words import num2words
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from core.datatypes import to_ist

if __name__ == "__main__":

        file = input("Enter csv file path here: ")
        df = pd.read_csv(file)
        df['DM_diff'] = df['DMP']-df['DMN']
        
        df['Month'] = to_ist(df['Buy_timestamp']).dt.strftime('%Y-%m')

        # Build a comprehensive daily tear sheet
        monthly_analysis_df = df.groupby('Month').agg(
        Total_PnL=('PnL', 'sum'),
        Gross_Profit=('PnL', lambda x: x[x > 0].sum()),
        Gross_Loss=('PnL', lambda x: x[x < 0].sum()),
        Total_Trades=('PnL', 'count'),
        Win_Rate=('PnL', lambda x: (x > 0).mean() * 100) # Returns % of winning trades
        ).reset_index()

        # Round the financials for a clean look
        monthly_analysis_df = monthly_analysis_df.round(2)
        profit_df = df[df['PnL']>0].reset_index(drop=True)
        total_profit = profit_df['PnL'].sum().round() 
        loss_df = df[df['PnL']<0].reset_index(drop=True)
        total_loss = loss_df['PnL'].sum().round() 
        pd.set_option('display.max_rows', None)
        winning_trades = profit_df['PnL']
        losing_trades = loss_df['PnL']

        avg_win = winning_trades.mean() if not winning_trades.empty else 0
        avg_loss = abs(losing_trades.mean()) if not losing_trades.empty else 1 
        system_rr = avg_win / avg_loss
        print(monthly_analysis_df)
        print(total_loss,' -- ',num2words(total_loss,lang='en_IN'))
        print('Total lossing trades: ',len(loss_df))
        print(total_profit,' -- ',num2words(total_profit,lang='en_IN') )
        print('Total winnig trades: ',len(profit_df))
        print('Total win rate: ',(len(profit_df)/(len(loss_df)+len(profit_df)))*100)
        print(f"Average Win: ₹{avg_win:.2f}")
        print(f"Average Loss: ₹{avg_loss:.2f}")
        print(f"True System R:R: 1 : {system_rr:.2f}")
        print("Losing Calls",len(loss_df[(loss_df['Side']=='CE')]))
        print("Losing Puts",len(loss_df[(loss_df['Side']=='PE')]))
        print("Winning Puts",len(profit_df[(profit_df['Side']=='PE')]))
        print("Winning Calls",len(profit_df[(profit_df['Side']=='CE')]))

        
