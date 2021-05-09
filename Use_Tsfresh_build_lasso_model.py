import math
import numpy as np
import pandas as pd

import gc
import time
import calendar 
from datetime import date
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import tsfresh
from tsfresh import extract_features, extract_relevant_features, select_features

%matplotlib inline


#data pre_process
data= pd.read_excel("TOP1000_code_04.xlsx",encoding='gbk',errors='ignore',dtype=str)
data=data[data['goods_code']!='0200011831']
data_ts=data[['goods_code','ym','sale']]
data_ts=data_ts[data_ts['ym']!='2016_12']
data_ts=data_ts[data_ts['ym']!='2019_03']


def split_ts(data):
    data['ym']=data['ym'].apply(lambda x: datetime.datetime.strptime(x,"%Y_%m"))
    data['sale']=data['sale'].astype(float)
    #划分数据集
    ts_test=data[(data['ym']<='2019-01-01')&(data['ym']>='2018-12-01')]
    ts_train=data[(data['ym']<'2018-12-01')]
    ts_test=ts_test.sort_values(by=['goods_code','ym'])
    ts_train=ts_train.sort_values(by=['goods_code','ym'])
    ts_test=ts_test.reset_index()
    ts_train=ts_train.reset_index()
    ts_test_1= pd.DataFrame(ts_test,columns=["goods_code", "sale","ym"])
    ts_test_1['sale']=ts_test_1['sale'].astype(float)
    ts_test_pro= tsfresh.extract_features(ts_test_1, column_id="goods_code",column_sort='ym')
    ts_train_1= pd.DataFrame(ts_train,columns=["goods_code", "sale","ym"])
    ts_train_1['sale']=ts_train_1['sale'].astype(float)
    ts_train_pro= tsfresh.extract_features(ts_train_1, column_id="goods_code",column_sort='ym')
    #释放临时变量
    del ts_test,ts_train,ts_test_1,ts_train_1
    gc.collect()
    return ts_test_pro,ts_train_pro


def data_clean(data):
    #固定进入模型的字段
    data=data[['goods_code','ym','sale']]
    data.astype(str)
    data['sale']=data['sale'].astype(float)
    data.sort_values(by=['goods_code','ym'],inplace=True)
    data['goods_code']=data['goods_code'].astype(str)
    data_m1=pd.DataFrame(data['ym'].groupby(data['goods_code']).count())
    data_m1=data_m1.rename(index=str,columns={'ym':'long'})
    data_m1=data_m1.reset_index()
    #选择大于15个月的值进行预测
    data_m2=data_m1.merge(data,on='goods_code',how='right')
    data_m2=data_m2[data_m2['long']>=15]
    data_m2['ym']=data_m2['ym'].apply(lambda x: datetime.datetime.strptime(x,"%Y_%m"))
    data_m2['month'] = data_m2['ym'].dt.month
    data_m2['month_day']=data_m2['month'].map(lambda x : calendar.monthrange(2019,x)[1])
    data_m2['dis_month'] = data_m2['ym'].map(lambda x: (datetime.datetime(2019, 3, 8)-x).days//28)
    data_m2.sort_values(['goods_code','ym'],inplace=True)
    return data_m2

def train_model():
    start_time=time.time()
    data_inp=data_clean(df)
    pivot = data_inp.pivot(index='goods_code', columns='dis_month', values='sale')
    #对变量重新命名
    col_name=[]
    for i in range(len(pivot.columns)):
        col_name.append('sales_'+str(i))
    pivot.columns=col_name
    pivot.fillna(0, inplace=True)
    sub=pivot.reset_index()
    test_features=['goods_code']
    trian_features = ['goods_code']
    for i in range(1,3):
        test_features.append('sales_' + str(i))
    #前面21个月作为训练集
    for i in range(3,23):
        trian_features.append('sales_' + str(i))

    sub.fillna(0, inplace=True)
    sub.drop_duplicates(subset=['goods_code'],keep='first',inplace=True)
    #最近的两个月作为测试集
    for i in range(1,3):
        test_features.append('sales_' + str(i))
   
    for i in range(3,23):
        trian_features.append('sales_' + str(i))
    X_train = sub[trian_features]
    y_train = sub[['sales_0', 'goods_code']]
    X_test = sub[test_features]    
    sales_type = 'sales_'
    
    #平均数特征
    X_train['mean_sale'] = X_train.apply(
        lambda x: np.mean([x[sales_type+'3'], x[sales_type+'4'],x[sales_type+'5'], 
                              x[sales_type+'6'], x[sales_type+'7'],x[sales_type+'8'], x[sales_type+'9'], 
                           x[sales_type+'10'], x[sales_type+'11'],x[sales_type+'12'],x[sales_type+'13'], 
                              x[sales_type+'14'],
                           x[sales_type+'15'], x[sales_type+'16'], x[sales_type+'17'],x[sales_type+'18'],
                           x[sales_type+'19'], x[sales_type+'20'], x[sales_type+'21'], x[sales_type+'22']]), axis=1)
    
    X_test['mean_sale'] = X_test.apply(
        lambda x: np.mean([x[sales_type+'1'], x[sales_type+'2']]), axis=1)
    train_mean=X_train['mean_sale']
    test_mean=X_test['mean_sale']
    train_mean=pd.Series(train_mean)
    test_mean=pd.Series(test_mean)
    
     #众数特征
    X_train['median_sale'] = X_train.apply(
        lambda x: np.median([ x[sales_type+'3'], x[sales_type+'4'],
                      x[sales_type+'5'], x[sales_type+'6'], x[sales_type+'7'],x[sales_type+'8'], 
                             x[sales_type+'9'], x[sales_type+'10'], x[sales_type+'11'],x[sales_type+'12'],
                             x[sales_type+'13'], x[sales_type+'14'],x[sales_type+'15'], x[sales_type+'16'], 
                             x[sales_type+'17'],x[sales_type+'18'], x[sales_type+'19'], x[sales_type+'20'],
                             x[sales_type+'21'], x[sales_type+'22']]), axis=1)
    X_test['median_sale'] = X_test.apply(
        lambda x: np.median([x[sales_type+'1'], x[sales_type+'2']]), axis=1)
    
    #标准差特征
    X_train['std_sale'] = X_train.apply(
        lambda x: np.std([ x[sales_type+'3'], x[sales_type+'4'],x[sales_type+'5'], x[sales_type+'6'], 
                          x[sales_type+'7'],x[sales_type+'8'], x[sales_type+'9'], x[sales_type+'10'], 
                          x[sales_type+'11'],x[sales_type+'12'],x[sales_type+'13'], x[sales_type+'14'],
                        x[sales_type+'15'], x[sales_type+'16'], x[sales_type+'17'],x[sales_type+'18'], 
                        x[sales_type+'19'], x[sales_type+'20'], x[sales_type+'21'], x[sales_type+'22']]), axis=1)
    X_test['std_sale'] = X_test.apply(
        lambda x: np.std([x[sales_type+'1'], x[sales_type+'2']]), axis=1)
    
    train_median=X_train['median_sale']
    test_median=X_test['median_sale']

    train_std=X_train['std_sale']
    test_std=X_test['std_sale']

    X_train = sub[trian_features]
    X_test = sub[test_features]
    
    formas_train=[train_mean,train_median,train_std]
    formas_test=[test_mean,test_median,test_std]
    train_inp=pd.concat(formas_train,axis=1)
    test_inp=pd.concat(formas_test,axis=1)
    
    #残差特征
    lr_Y=y_train['sales_0']
    lr_train_x=train_inp
    re_train= sm.OLS(lr_Y,lr_train_x).fit()
    train_inp['resid']=re_train.resid
    
    lr_Y=y_train['sales_0']
    lr_test_x=test_inp
    re_test= sm.OLS(lr_Y,lr_test_x).fit()
    test_inp['resid']=re_test.resid
    
    train_inp=pd.concat([y_train,train_inp],axis=1)
    
    ts_test_pro,ts_train_pro=split_ts(df)
    
    ts_train_=ts_train_pro.reset_index()
    train_inp=pd.merge(train_inp,ts_train_,left_on='goods_code',right_on='id',how='left')
    test_inp=pd.concat([y_train,test_inp],axis=1)
    
    ts_test_=ts_test_pro.reset_index()
    test_inp=pd.merge(test_inp,ts_test_,left_on='goods_code',right_on='id',how='left')
    train_inp.drop(['sales_0','goods_code'],axis=1,inplace=True)
    test_inp.drop(['sales_0','goods_code'],axis=1,inplace=True)
    
    train_inp.fillna(0,inplace=True)
    train_inp.replace(np.inf,0,inplace=True)
    test_inp.replace(np.inf,0,inplace=True)
    test_inp.fillna(0,inplace=True)

    #lasso
    ss = StandardScaler()
    train_inp_s= ss.fit_transform(train_inp) 
    test_inp_s= ss.transform(test_inp)
    alpha_ridge = [1e-4,1e-3,1e-2,0.1,1]

    coeffs = {}
    for alpha in alpha_ridge:
        r = Lasso(alpha=alpha, normalize=True, max_iter=1000000)
        r = r.fit(train_inp_s, y_train['sales_0'])

    grid_search = GridSearchCV(Lasso(alpha=alpha, normalize=True), scoring='neg_mean_squared_error',
                           param_grid={'alpha': alpha_ridge}, cv=5, n_jobs=-1)
    grid_search.fit(train_inp_s, y_train['sales_0'])
    
    alpha = alpha_ridge
    rmse = list(np.sqrt(-grid_search.cv_results_['mean_test_score']))
    plt.figure(figsize=(6,5))
    
    lasso_cv = pd.Series(rmse, index = alpha)
    lasso_cv.plot(title = "Validation - LASSO", logx=True)
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    plt.show()
    
    least_lasso=min(alpha)
    lasso = Lasso(alpha=least_lasso,normalize=True)
    model_lasso=lasso.fit(train_inp_s,y_train['sales_0'])
    
    print("lasso feature.......................")
    lasso_coef = pd.Series(model_lasso.coef_,index = train_inp.columns)
    lasso_coef=lasso_coef[lasso_coef!=0.0000]
    lasso_coef=lasso_coef.astype(float)
    print(".....lasso_coef..............")

    print(lasso_coef.sort_values(ascending=False).head(10))
    print(" R^2，拟合优度")
    
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef = pd.concat([lasso_coef.sort_values().head(5), 
                     lasso_coef.sort_values().tail(5)])#选头尾各10条

    imp_coef.plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")
    
    print(lasso.score(train_inp_s,y_train['sales_0']))
    
    print(lasso.get_params())  
    print('参数信息')
    print(lasso.set_params(fit_intercept=False)) 
    lasso_preds =model_lasso.predict(test_inp_s)
    #绘制预测结果和真实值散点图
    fig, ax = plt.subplots()
    ax.scatter(y_train['sales_0'],lasso_preds)
    ax.plot([y_train['sales_0'].min(), y_train['sales_0'].max()], [y_train['sales_0'].min(), y_train['sales_0'].max()], 'k--', lw=4)
    ax.set_xlabel('y_true')
    ax.set_ylabel('Pred')
    plt.show()
    y_pred=pd.DataFrame(lasso_preds,columns=['y_pred'])
    
    matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
    preds = pd.DataFrame({"preds":y_pred['y_pred'], "true":y_train['sales_0']}) 
    preds["residuals"] = preds["true"] - preds["preds"]
    
    print("打印预测值描述.....................")
    preds=preds.astype(float)
    print(preds.head())
    print(preds.describe())
    print(preds.shape)
    preds.plot(x = "preds", y = "residuals",kind = "scatter")
    plt.title("True and residuals")
    plt.show()
    
    data_out=[y_train['goods_code'],y_train['sales_0'],y_pred]
    result=pd.concat(data_out,axis=1)
    #计算mape
    result['mape']=abs((result['sales_0']-result['y_pred'])/result['sales_0']*100)    
    return result,lasso_coef


if __name__ == '__main__':
    df=data_ts
    result_f,lasso_coef_f=train_model()
    
    del df
    gc.collect()    