import matplotlib.pyplot as plt
import pandas as pd

from pyecharts import options as opts
def qushi(df):

    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

    plt.figure(figsize=(12, 7))
    df_sales = df
    df_sales=df_sales.drop(df_sales[df_sales['买家实际支付金额']==0].index)
    print(df_sales)
    df_sales['订单创建时间'] = pd.to_datetime(df_sales['订单创建时间'])
    df_sales = df_sales.set_index('订单创建时间')
    df_sales_day = df_sales.resample('D')['订单编号'].count()  # 按天计数
    x = df_sales_day.index
    y = df_sales_day
    plt.plot(x, y, color='red', marker='d')
    plt.title('每日订单数')
    plt.xlabel('日期')
    plt.ylabel('订单数')
    plt.show()

def dili(df):
    df_sales = df
    df_sales['订单创建时间'] = pd.to_datetime(df_sales['订单创建时间'])
    df_sales = df_sales.set_index('订单创建时间')
    df_sales_map = df_sales.groupby('收货地址')['收货地址'].count().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    df_sales_map.plot(kind='bar', color='#87CEFA')

    result = []
    for i in df_sales_map.index:
        if i.endswith('自治区'):
            if i == '内蒙古自治区':
                i = i[:3]
                result.append(i)
            else:
                i = i[:2]
                result.append(i)
        else:
            result.append(i)

    df_sales_map.index = result
    df_sales_map.index = df_sales_map.index.str.strip('省')
    print(df_sales_map.index)
    from pyecharts.charts import Map

    sales_map = Map().add('订单数', [list(i) for i in df_sales_map.items()]).set_global_opts(
        visualmap_opts=opts.VisualMapOpts(max_=max(df_sales_map) * 0.6))  # 设置最大数据范围

    sales_map.render()