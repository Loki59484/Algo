from core import upstox_methods as ustox

client = ustox.UpstoxClient()

trades = client.get_trades_for_day()
print("Sorting trades")
buy_trades = [trade for trade in trades if trade['transaction_type']=='BUY']
sell_trades = [trade for trade in trades if trade['transaction_type']=='SELL']
print("getting order details")

buy_orders = [client.get_order_details(trade['order_id']) for trade in buy_trades]
sell_orders = [client.get_order_details(trade['order_id']) for trade in sell_trades]

buy_orders = [item for item in buy_orders if item.status == 'complete']
sell_orders = [item for item in sell_orders if item.status == 'complete']

print("Calculating brokerage")
total_brokerage = (
    sum([client.get_brokerage(item.price,item.instrument_token,item.quantity) for item in buy_orders]) 
    + sum([client.get_brokerage(item.price,item.instrument_token,item.quantity) for item in sell_orders])
    )
