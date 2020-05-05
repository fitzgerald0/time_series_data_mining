
# -*- coding:utf-8 -*-
"""
Name  : multi_prophet_model.py
Time  : 2020/4/28 14:11
Author : hjs
"""


"""
针对多个序列数据的prophet预测，比如，10万个sku序列
从数据库读取到数据预处理和回测，预测的个人算法框架
回测最近7天，预测未来28天

"""
import gc
from dateutil.relativedelta import relativedelta
from fbprophet import Prophet
import pandas as pd
import numpy as np
import time
import datetime
import os
from holiday_data import holiday_df#自定义假期数据
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
#全局超参数，使用计算核数
use_cpu = cpu_count() // 4

def sale_ds():
    
    sql = """select * from scmtemp.csh_scode1_29_dateset"""
    hive_conn = conn_hive()#该处为定义的读取hive，因隐私问题，不放出
    df = pd.read_sql(sql, hive_conn)
    df.columns = [col.lower().split('.')[-1] for col in df.columns]
    df.drop_duplicates(subset=['store_code', 'goods_code', 'ds'], inplace=True)
    df['store_code'] = df['store_code'].astype(str)
    df = df[df['store_code'] != 'None']
    df['store_sku'] = df['store_code'].astype(
        str) + '-' + df['goods_code'].astype(str)
    df.drop(columns=['store_code', 'goods_code'], inplace=True)
    df.rename(columns={'qty_fix': 'y'}, inplace=True)
    print('finish the data reading....')
    return df

def replace_fill(data, name):

    """
    先尝试使用上周的数据填补，再针对极端的数据进行cap，保障序列的完整和平滑性
    :param data:单个序列
    :param name: 序列名称，store_sku
    :return: 修复后的一条序列
    """
    data['ds'] = pd.to_datetime(data['ds'], format='%Y-%m-%d')
    data['y'] = data['y'].astype(float)
    data.loc[data['y'] <= 0, 'y'] = np.NaN
    data.loc[data['y'].isnull(), 'y'] = data['y'].shift(7).values[0]
    data.loc[data['y'].isnull(), 'y'] = data['y'].shift(-7).values[0]
    data.loc[data['y'].isnull(), 'y'] = data['y'].shift(-14).values[0]
    data.loc[data['y'].isnull(), 'y'] = data['y'].shift(14).values[0]
    data['y'] = data['y'].interpolate(methon='nearest', order=3)
    low = data[data['y'] > 0]['y'].quantile(0.10)
    high = data[data['y'] > 0]['y'].quantile(0.90)
    data.loc[data['y'] < low, 'y'] = np.NaN
    data.loc[data['y'] > high, 'y'] = np.NaN
    data['y'] = data['y'].fillna(data['y'].mean())
    data['store_sku'] = name
    return data


def multi_fill(data):
    start_time = time.time()
    data['store_sku'] = data['store_sku'].astype(str)
    data_grouped = data.groupby(data.store_sku)
    results = Parallel(
        n_jobs=use_cpu)(
        delayed(replace_fill)(
            group,
            name) for name,
        group in data_grouped)
    p_predict = pd.concat(results)
    end_time = time.time()
    del data
    gc.collect()
    print('read data end etl have use {} minutes'.format(
        round((end_time - start_time) / 60, 2)))
    return p_predict


def predict_cap(data, result, columns):
    """
    :param data:修正后的输入数据
    :param result: 预测值
    :param columns: 预测值columns
    :return:每个序列上下限使用原始输入数据进行修正的结果
    """
    data_list = set(result['store_sku'].unique())
    data_df = data[data['store_sku'].isin(data_list)][['store_sku', 'y']]
    for i in data_df['store_sku'].unique():
        low = (1 + 0.1) * data_df[data_df['store_sku'] == i]['y'].min()
        hight = (1 + 0.05) * data_df[data_df['store_sku'] == i]['y'].max()
        result.loc[(result['store_sku'] == i) & (
            result[columns] < low), columns] = low
        result.loc[(result['store_sku'] == i) & (
            result[columns] > hight), columns] = hight
    return result


def data_tranform(data):
    """
    :param data:全部序列的数据
    :return: 针对所以数据处理后的结果，如，针对某一天赋值为0，做对数处理
    """
    data = data[['store_sku', 'ds', 'y']]
    data.drop_duplicates(subset=['store_sku', 'ds'], inplace=True)
    data.sort_values(['store_sku', 'ds'], ascending=[
                     True, True], inplace=True)
    data['ds'] = data['ds'].astype(str)
    data['ds'] = data['ds'].apply(
        lambda x: datetime.datetime.strptime(
            x, "%Y-%m-%d"))
    data.loc[data['y'] == np.nan, 'y'] = data.shift(7).iloc[-1:, :]
    data.loc[data['y'] == np.nan, 'y'] = data.shift(-7).iloc[-1:, :]
    data.loc[data['y'] == np.nan, 'y'] = data.shift(-14).iloc[-1:, :]
    data['y'] = np.log1p(data['y'])
    data = data.dropna(axis=0)
    return data


def prophet_train(data, name, holiday_df, model_type='test'):
    # 选择model_type:test表示回测，否则预测未来时间点
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        holidays=holiday_df,
        holidays_prior_scale=10)
    model.add_seasonality(
        name='weekly',
        period=7,
        fourier_order=3,
        prior_scale=0.10
        # ,mode='additive'
    )
    if model_type == 'test':
        data_train = data.iloc[:-6]
        model.fit(data_train)
        future = model.make_future_dataframe(periods=7, freq='d')
    else:
        model.fit(data)
        future = model.make_future_dataframe(periods=7 * 4, freq='d')
    forecast = model.predict(future)
    forecast['store_sku'] = name
    print('---this runing id is :{0} ---'.format(name))
    return forecast


def multi_process(data, holiday_df, model_type):
    data['store_sku'] = data['store_sku'].astype(str)
    data_grouped = data.groupby(data.store_sku)
    results = Parallel(
        n_jobs=use_cpu)(
        delayed(prophet_train)(
            group,
            name,
            holiday_df,
            model_type) for name,
        group in data_grouped)
    p_predict = pd.concat(results)
    return p_predict


def prophet_main(data, holiday_df_, model_type, true_time=False):
    start = time.time()
    if true_time is False:
        true_time = pd.datetime.now().strftime('%Y-%m-%d')
    else:
        true_time = datetime.datetime.strptime(true_time, "%Y-%m-%d")
        true_time = str(
            (true_time +
             datetime.timedelta(
                 days=7)).strftime('%Y-%m-%d'))

    df = data_tranform(data)
    df['ds'] = pd.to_datetime(df['ds'], format('Y%-%m-%d'))
    df = df[df['ds'] < true_time]
    df['ds'] = df['ds'].astype(str)
    df['ds'] = pd.to_datetime(df['ds'])
    holiday_df_['ds'] = pd.to_datetime(holiday_df_['ds'])
    holiday_df_['ds'] = holiday_df_['ds'].astype(str)
    holiday_df_['ds'] = holiday_df_['ds'].apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    # parallel
    pro_back = multi_process(df, holiday_df_, model_type)
    if model_type == 'test':
        pro_back = pd.merge(
            df, pro_back, on=[
                'store_sku', 'ds'], how='inner')
    else:
        print('this is forecast model！')
    pro_back.rename(columns={'yhat': 'pro_pred'}, inplace=True)
    pro_back['pro_pred'] = np.expm1(pro_back['pro_pred'])
    # 盖帽异常值
    pro_back.loc[pro_back['pro_pred'] < 0, 'pro_pred'] = 0
    pro_back_adj = predict_cap(df, pro_back, 'pro_pred')
    low = pro_back_adj['pro_pred'].quantile(0.05)
    hight = pro_back_adj['pro_pred'].quantile(0.95)
    pro_back_adj.loc[pro_back_adj['pro_pred'] < low, 'pro_pred'] = low
    pro_back_adj.loc[pro_back_adj['pro_pred'] > hight, 'pro_pred'] = hight
    pro_back_adj['pro_pred'] = pro_back_adj['pro_pred'].round(2)
    if model_type == 'test':
        today_date = str(
            (datetime.datetime.strptime(
                true_time,
                "%Y-%m-%d") -
                datetime.timedelta(
                days=7)).strftime('%Y-%m-%d'))
        back_result = pro_back_adj[pro_back_adj['ds'] >= today_date][[
            'store_sku', 'ds', 'pro_pred']].drop_duplicates(subset=['store_sku', 'ds'])
    else:
        result = pro_back_adj[pro_back_adj['ds'] >= true_time]
        back_result = result[['store_sku', 'ds', 'pro_pred']
                             ].drop_duplicates(subset=['store_sku', 'ds'])
        print(
            'prophet model use : {} minutes'.format(
                (time.time() - start) // 60))
    return back_result


if __name__ == '__main__':
    sale_df = sale_ds()
    sale_df['ds'] = pd.to_datetime(sale_df['ds'])
    sale_df = sale_df[['store_sku', 'ds', 'y']]
    # 控制长度,不使用疫情时期的数据，且周期不用太长，关注最近的几个完整周期即可
    start_day = (
        sale_df['ds'].max() -
        relativedelta(
            days=63)).strftime('%Y-%m-%d')
    sale_df = sale_df[sale_df['ds'] >= start_day][['store_sku', 'ds', 'y']]
    # 筛选条件：1 序列长度大于等于14，且过去最少有七天的销售记录；
    # 条件1，保障模型有两个完整的周期数据；
    # 条件2，避免出现0，0，0，0，0，0，1，0，1这样非常稀疏的数据出现
    sale_set = sale_df.groupby(
        ['store_sku']).filter(
        lambda x: len(x) >= 14 and np.sum(
            x['y']) > 7)
    print('min date is {},max date is {}'.format(
        sale_set['ds'].min(), sale_set['ds'].max()))
    sale_data = multi_fill(sale_set)
    holiday_df_ = holiday_df()
    # 回测最近7天
    model_type = 'test'
    # 回测开始时间，如果为false，则回测从过去的第七天开始
    true_time = False
    pro_mape = prophet_main(sale_data, holiday_df_, model_type, true_time)
    pro_mape = pd.merge(
        pro_mape, sale_set, on=[
            'store_sku', 'ds'], how='inner')
    pro_mape['mape'] = np.abs(
        pro_mape['y'] - pro_mape['pro_pred']) / pro_mape['y'] * 100
    print('mape------', pro_mape['mape'].mean())
    pro_mape.to_excel('pro_mape_428.xlsx', index=False)

    # 以下为预测未来28天
    model_type = 'train'
    prophet_forecast = prophet_main(
         sale_data, holiday_df_, model_type, true_time)
    prophet_forecast.to_excel('prophet_forecast_428.xlsx', index=False)
