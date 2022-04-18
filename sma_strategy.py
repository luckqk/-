import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.getcwd(), '.env'))
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN")

pro = ts.pro_api(TUSHARE_TOKEN)
data = pro.daily(ts_code='600030.SH', start_date='20160630', end_date='20170630')
data = pd.DataFrame(data)
data.rename(columns={'close': 'price'}, inplace=True)
# inplace 覆盖
data.set_index('trade_date', inplace=True)
# need to reorder from older time to latest
data.sort_values(by='trade_date', ascending=True, inplace=True)

data['SMA_10'] = data['price'].rolling(10).mean()
data['SMA_60'] = data['price'].rolling(60).mean()
data[['price','SMA_10','SMA_60']].plot(title='HS300 stock price | 10 & 60 days SMAs',
                                       figsize=(10, 6))
# add this and plot will show in pycharm
# plt.show()
# 1 做多， -1 做空
data['position'] = np.where(data['SMA_10'] > data['SMA_60'], 1, -1)
# 去掉空值，NaN
data.dropna(inplace=True)
# 每天收益率
data['returns'] = np.log(data['price'] / data['price'].shift(1))
# 分布图, bins粒度
data['returns'].hist(bins=35)
# 策略收益
data['strategy'] = data['position'].shift(1) * data['returns']

# compare returns and strategy
data[['returns', 'strategy']].sum()
# e^([ln(pday1/pday2)+ln(pday2/pday3)])
data[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))


