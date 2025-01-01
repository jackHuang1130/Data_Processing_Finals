import twstock
import mplfinance as mpf
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()

# LINE Notify Token 
LINE_NOTIFY_TOKEN = os.getenv("LINE_NOTIFY_TOKEN")
LINE_NOTIFY_URL = 'https://notify-api.line.me/api/notify'

def send_line_notify(message, image_path=None):
    headers = {'Authorization': f'Bearer {LINE_NOTIFY_TOKEN}'}
    data = {'message': message}
    files = {'imageFile': open(image_path, 'rb')} if image_path else None
    response = requests.post(LINE_NOTIFY_URL, headers=headers, data=data, files=files)
    return response.status_code

def get_stock_data(stock_no, days=30):
    stock = twstock.Stock(stock_no)
    today = datetime.today()
    # 獲取日期
    dates=stock.date
    # 獲取價格
    prices=stock.price
    # 獲取最高價
    highs=stock.high
    # 獲取最低價
    lows=stock.low
    # 獲取開盤價
    opens=stock.open
    # 獲取收盤價
    closes=stock.close
    # 獲取成交量
    volumes=stock.capacity

    data = []
    for i in range(len(dates)):
        if dates[i] >= today - timedelta(days=days):
            data.append({
                'date': dates[i],
                'open': opens[i],
                'high': highs[i],
                'low': lows[i],
                'close': closes[i],
                'volume': volumes[i]
            })
    return data

def plot_k_line(stock_data, stock_name, realtime_price):
    df_data = {
        'Date': [d['date'] for d in stock_data],
        'Open': [d['open'] for d in stock_data],
        'High': [d['high'] for d in stock_data],
        'Low': [d['low'] for d in stock_data],
        'Close': [d['close'] for d in stock_data],
        'Volume': [d['volume'] for d in stock_data]
    }
    
    df = pd.DataFrame(df_data)
    df.set_index('Date', inplace=True)

    rcpdict = { 'font.family' : 'Microsoft JhengHei', 'font.size':10 }
    s = mpf.make_mpf_style(base_mpf_style='yahoo',rc=rcpdict)
    mpf.plot(df, 
             type='candle',
             style=s, 
             savefig=f'{stock_name}_kline.png', 
             title=(f'{stock_name} 近一個月 K 線圖'), 
             figratio=(15,15),
             figscale=1.5
             )
    
    return f'{stock_name}_kline.png'

if __name__ == "__main__":
    stocks = {'台積電': '2330', '聯發科': '2454'}

    for stock_name, stock_no in stocks.items():
        stock_data = get_stock_data(stock_no, days=30)
        
        #調取即時股價
        realtime_price = twstock.realtime.get(stock_no)["realtime"]["latest_trade_price"]
        image_path = plot_k_line(stock_data, stock_name, realtime_price)

        message = f'{stock_name} 即時股價: {realtime_price} 元' if realtime_price else f'{stock_name} 即時股價無法取得'

        send_line_notify(message, image_path)
        print(f'{stock_name} 的即時股價和K線圖已發送至LINE Notify。')
