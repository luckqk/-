import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.getcwd(), '.env'))
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN")

# momentum 强者恒强
pro = ts.pro_api(TUSHARE_TOKEN)
data = pro.daily(ts_code='600030.SH', start_date='20160630', end_date='20180630')
data.rename(columns={'close': 'price'}, inplace=True)
data.set_index('trade_date', inplace=True)
data['returns'] = np.log(data['price'] / data['price'].shift(1))
# np.sign if data > 0 1, <0 -1
data['position'] = np.sign(data['returns'])
data['strategy'] = data['position'].shift(1) * data['returns']
data[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))

# 优化
# rolling(x) 挑选过去x天的数据进行处理
data['position_5'] = np.sign(data['returns'].rolling(5).mean())
data['strategy_5'] = data['position_5'].shift(1) * data['returns']

# 离散算法
data['returns_dis'] = data['price'] / data['price'].shift(1)-1
# cumprod 累乘
data['returns_dis_cum'] = (data['returns_dis']+1).cumprod()

price_plot = ['returns_dis_cum']
# 枚举时间获取最佳参数
for days in [10,20,30,60]:
    price_plot.append('sty_cumr_%dd' % days)
    data['position_%dd' % days] = np.where(data['returns'].rolling(days).mean()>0,1,-1)
    data['strategy_%dd' % days] = data['position_%dd' % days].shift(1) * data['returns']
    data['sty_cumr_%dd' % days] = (data['strategy_%dd' % days]+1).cumprod()