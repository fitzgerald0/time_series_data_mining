#-*- coding:utf-8 -*-
"""
Name  : sarimax_model.py
Time  : 2019/9/8 18:17
Author : hjs
"""

import time
from itertools import product
import numpy as np
import pandas as pd
from math import sqrt
# from multiprocessing import cpu_count
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')
from warnings import catch_warnings, filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX


# 传入数据和参数，输出模型预测
def model_forecast(history, config):
    order, sorder, trend = config
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# 模型评估指标,mape
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 划分训练集和测试集
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# one-step滚动向前预测
def forward_valid(data, n_test, cfg):
    predictions = list()
    train, test = train_test_split(data, n_test)
    history = [x for x in train]
    for i in range(len(test)):
        yhat = model_forecast(history, cfg)
        predictions.append(yhat)
        history.append(test[i])
    error = mape(test, predictions)
    return error


# 模型评估
def score_model(data, n_test, cfg, debug=False):
    result = None
    key = str(cfg)
    if debug:
        result = forward_valid(data, n_test, cfg)
    else:
        try:
            with catch_warnings():
                filterwarnings("ignore")
                result = forward_valid(data, n_test, cfg)
        except:
            error = None

    return (key, result)


# 网格搜索
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # 使用计算机全部的cpu核数多进程并行
        executor = Parallel(n_jobs=-1, backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)

    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    scores = [r for r in scores if r[1] != None]
    scores.sort(key=lambda x: x[1])
    return scores


# 生成参数列表
def sarima_configs(seasonal=[0]):
    p = d = q = [0, 1, 2]
    pdq = list(product(p, d, q))
    s = 0
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(product(p, d, q))]
    t = ['n', 'c', 't', 'ct']
    return list(product(pdq, seasonal_pdq, t))


# 模型训练
def train_model(sale_df):
    sum_ = 0
    n_test = 3
    p_b, d_b, q_b = [], [], []
    P_b, D_b, Q_b = [], [], []
    m_b, t_b = [], []
    model_id, error = [], []
    for i in sale_df['store_code'].unique():
        data = sale_df[sale_df['store_code'] == i]['y']
        data = [i for i in data]
        cfg_list = sarima_configs()
        scores = grid_search(data, cfg_list, n_test, parallel=True)
        p_b.append(int(scores[0][0][2]))
        d_b.append(int(scores[0][0][5]))
        q_b.append(int(scores[0][0][8]))
        P_b.append(int(scores[0][0][13]))
        D_b.append(int(scores[0][0][16]))
        Q_b.append(int(scores[0][0][19]))
        m_b.append(int(scores[0][0][22]))
        t_b.append(str(scores[0][0][27]))
        model_id.append(i)
        error.append(scores[1][-1])
        params_df = pd.DataFrame(
            {'store_code': model_id, 'map': error, 'p': p_b, 'd': d_b, 'q': q_b, 'P': P_b, 'D': D_b, 'Q': Q_b, 'm': m_b,
             't': t_b})
    return params_df


# 模型预测
def one_step_forecast(data, order, seasonal_order, t, h_fore):
    predictions = list()
    data = [i for i in data]
    for i in range(h_fore):
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order, trend=t, enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        yhat = model_fit.predict(len(data), len(data))
        data.append(yhat[0])
        predictions.append(yhat[0])
    return predictions


def forecast_model(sale_df, params_df):
    h_fore = 4
    fore_list = []
    model_id = []
    for i in sale_df['store_code'].unique():
        params_list = params_df[params_df['store_code'] == i]
        data = sale_df[sale_df['store_code'] == i]['y']
        p = params_df[params_df['store_code'] == i].iloc[:, 2].values[0]
        d = params_df[params_df['store_code'] == i].iloc[:, 3].values[0]
        q = params_df[params_df['store_code'] == i].iloc[:, 4].values[0]
        P = params_df[params_df['store_code'] == i].iloc[:, 5].values[0]
        D = params_df[params_df['store_code'] == i].iloc[:, 6].values[0]
        Q = params_df[params_df['store_code'] == i].iloc[:, 7].values[0]
        m = params_df[params_df['store_code'] == i].iloc[:, 8].values[0]
        t = params_df[params_df['store_code'] == i].iloc[:, 9].values[0]
        order = (p, d, q)
        seasonal_order = (P, D, Q, m)
        all_fore = one_step_forecast(data, order, seasonal_order, t, h_fore)
        fore_list.append(all_fore)

        # 以下为，多步预测，如果不使用滚动预测，则不调one_step_forecast函数
        # model=SARIMAX(data, order=order,seasonal_order=seasonal_order,trend=t,enforce_stationarity=False,
        #                                                enforce_invertibility=False)
        # forecast_=model.fit(disp=-1).forecast(steps=h_fore)
        # fore_list_flatten = [x for x in forecast_]
        # fore_list.append(fore_list_flatten)
        model_id.append(i)
    df_forecast = pd.DataFrame({'store_code': model_id, 'fore': fore_list})
    return df_forecast

if __name__ == '__main__':
    start_time = time.time()
    sale_df = pd.read_excel('/home/test01/store_forecast/sale_df.xlsx')
    params_df = train_model(sale_df)
    forecast_out = forecast_model(sale_df, params_df)
    end_time = time.time()
    use_time = (end_time - start_time) // 60
    print('finish the process use', use_time, 'mins')