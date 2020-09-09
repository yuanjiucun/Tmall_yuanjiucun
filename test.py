import pandas as pd
import zhuanhualv
import numpy as np
from scipy import cluster
import qushitu
import seaborn as sns  #用于绘制热图的工具包
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import decomposition as skldec #用于主成分分析降维的包

from sklearn.cluster import AgglomerativeClustering

df=pd.read_csv(r'C:\Users\Administrator\Desktop\课件学习\商务建模与大数据分析\tmall_order_report.csv',engine='python',encoding='utf-8')
print(df.head())
print("========")
print(df.columns)
print("=========")
print(df.info())
df=df.rename(columns={'收货地址 ':'收货地址','订单付款时间 ':'订单付款时间'})
print(df.columns)
print(df.duplicated().sum())    #重复值
print(df.isnull().sum())        #缺失值

zhuanhualv.zhuanhua(df)

qushitu.qushi(df)

diqu={
    '上海':0,
    '广东省':1,
    '江苏省':2,
    '浙江省':3,
    '北京':4,
    '四川省':5,
    '山东省':6,
    '辽宁省':7,
    '天津':8,
    '湖南省':9,
    '河北省':10,
    '重庆':11,
    '河南省':12,
    '云南省':13,
    '安徽省':14,
    '陕西省':15,
    '福建省':16,
    '山西省':17,
    '广西壮族自治区':18,
    '江西省':19,
    '吉林省':20,
    '黑龙江省':21,
    '贵州省':22,
    '内蒙古自治区':23,
    '海南省':24, '甘肃省':25,
    '湖北省':26,
    '新疆维吾尔自治区':27,
    '宁夏回族自治区':28,
    '青海省':29,
    '西藏自治区':30
}
df['收货地址']=df['收货地址'].map(diqu)
df['订单创建时间'] = pd.to_datetime(df['订单创建时间'])
df['订单创建时间'] = df['订单创建时间'].apply(lambda x:x.strftime('%Y%m%d'))
df['订单付款时间']=df['订单付款时间'].values.astype(str)
df['订单付款时间']=df['订单付款时间'].str[0:10]
df['订单付款时间']=df['订单付款时间'].apply(lambda x:str(x[0:4]+x[5:7]+x[8:10]))
df['订单付款时间']=df['订单付款时间'].replace('nan',-1)
print(df['订单付款时间'].replace('nan',-1))

scaler = MinMaxScaler()
scaler.fit(df)
scaler.data_max_
my_matrix_normorlize=scaler.transform(df)
print(my_matrix_normorlize)

#samples = my_matrix_normorlize.values
print('samples的维度',my_matrix_normorlize.shape)

"""mergings = linkage(my_matrix_normorlize,method='single')
print('mergings维度',mergings.shape)
#层次分析可视化，leaf的字体不旋转，大小为10。
#这里我们不显示每一条数据的具体名字标签（varieties），默认以数字标签显示
dendrogram(mergings,leaf_rotation=0,leaf_font_size=10)
plt.show()"""
modal=AgglomerativeClustering(my_matrix_normorlize,affinity="euclidean",linkage="complete")
print(modal)
dendrogram(modal,leaf_rotation=0,leaf_font_size=10)
plt.show()
label = cluster.hierarchy.cut_tree(modal,height=0.9)
label = label.reshape(label.size,)
pca = skldec.PCA(n_components = 2)    #选择方差95%的占比
pca.fit(df)   #主城分析时每一行是一个输入数据
result = pca.transform(df)  #计算结果
print(result)


plt.figure()  #新建一张图进行绘制
plt.scatter(result[:, 0], result[:, 1], c=label, edgecolor='k') #绘制两个主成分组成坐标的散点图
for i in range(result[:,0].size):
    plt.text(result[i,0],result[i,1],df.index[i])     #在每个点边上绘制数据名称
x_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[0]*100.0),2)   #x轴标签字符串
y_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[1]*100.0),2)   #y轴标签字符串
plt.xlabel(x_label)    #绘制x轴标签
plt.ylabel(y_label)    #绘制y轴标签
plt.show()
sns.clustermap(my_matrix_normorlize,method ='ward',metric='euclidean')
#订单趋势图
#qushitu.qushi(df)
#地理图
#qushitu.dili(df)
#层次聚类分析
"""from scipy.cluster.hierarchy import linkage, dendrogram
df['订单创建时间'] = pd.to_datetime(df['订单创建时间'])
df_cengci=df
#df_cengci=df_cengci.drop(['收货地址'],axis=1)
df_cengci=df_cengci.drop(['订单编号'],axis=1)
diqu={
    '上海':0,
    '广东省':1,
    '江苏省':2,
    '浙江省':3,
    '北京':4,
    '四川省':5,
    '山东省':6,
    '辽宁省':7,
    '天津':8,
    '湖南省':9,
    '河北省':10,
    '重庆':11,
    '河南省':12,
    '云南省':13,
    '安徽省':14,
    '陕西省':15,
    '福建省':16,
    '山西省':17,
    '广西壮族自治区':18,
    '江西省':19,
    '吉林省':20,
    '黑龙江省':21,
    '贵州省':22,
    '内蒙古自治区':23,
    '海南省':24, '甘肃省':25,
    '湖北省':26,
    '新疆维吾尔自治区':27,
    '宁夏回族自治区':28,
    '青海省':29,
    '西藏自治区':30
}
df_cengci['收货地址']=df_cengci['收货地址'].map(diqu)

df['订单创建时间'] = pd.to_datetime(df['订单创建时间'])
df_cengci['订单创建时间'] = df_cengci['订单创建时间'].apply(lambda x:x.strftime('%Y%m%d'))
df_cengci['订单付款时间']=df_cengci['订单付款时间'].values.astype(str)
df_cengci['订单付款时间']=df_cengci['订单付款时间'].str[0:10]
df_cengci['订单付款时间']=df_cengci['订单付款时间'].apply(lambda x:str(x[0:4]+x[5:7]+x[8:10]))
df_cengci['订单付款时间']=df_cengci['订单付款时间'].replace('nan',-1)
print(df_cengci['订单付款时间'].replace('nan',-1))
samples = df_cengci.values
print(samples)
print('samples的维度',samples.shape)

mergings = linkage(samples,method='single')
print('mergings维度',mergings.shape)
#层次分析可视化，leaf的字体不旋转，大小为10。
#这里我们不显示每一条数据的具体名字标签（varieties），默认以数字标签显示
dendrogram(mergings,leaf_rotation=0,leaf_font_size=10)
plt.show()
print(mergings)
"""

#在图中显示的数字是最细粒度的叶子，相当于每个样本数据点。

