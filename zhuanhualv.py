import pandas as pd
from pyecharts.charts import Funnel
from pyecharts import options as opts
def zhuanhua(df):
    #计算各环节订单数
    dict_convs=dict()
    key="总订单数"
    dict_convs[key]=len(df['订单编号'])

    key='付款订单数'
    dict_convs[key]=df['订单付款时间'].notnull().sum()

    key='到款订单数'
    #总金额减去退款金额不为0，则认为到款了，即买家实际支付金额不为0即可
    sum_daokuan=(df['买家实际支付金额'] !=0).sum()
    dict_convs[key]=sum_daokuan

    key='未退款订单数'
    #即付款了且未退款
    dict_convs[key]=len(df[(df['买家实际支付金额'] != 0) & (df['退款金额'] == 0)])
    print(dict_convs)
    #计算总体转化率
    df_convs=pd.Series(dict_convs,name='订单数').to_frame()
    print(df_convs)
    cvr_total = df_convs['订单数']/df_convs.loc['总订单数','订单数']
    df_convs['总转化率'] = cvr_total
    df_convs[['总转化率']] = df_convs[['总转化率']].applymap(lambda x :'%.2f%%'  %  (x*100))
    print(df_convs)
    funnel = Funnel().add('总转化率',
                          [list(z) for z in zip(df_convs.index, cvr_total)],
                          label_opts=opts.LabelOpts(position='inside'))
    funnel.render_notebook()
    #计算单一环节转化率
    cvr_single = df_convs['订单数'].shift()                               # 将数据向下移一位，第一行变成NAN
    cvr_single = cvr_single.fillna(df_convs.loc['总订单数','订单数'])             # 将第一行的NAN填上订单总数的数据，用于计算
    cvr_single = df_convs['订单数']/cvr_single
    df_convs['单一环节转化率'] = cvr_single
    df_convs[['单一环节转化率']] = df_convs[['单一环节转化率']].applymap(lambda x :'%.2f%%'  %  (x*100))
    print(df_convs)
    funnel = Funnel().add('单一环节转化率',
                          [list(z) for z in zip(df_convs.index, cvr_single)],
                          label_opts=opts.LabelOpts(position='inside'))
    funnel.render_notebook()
