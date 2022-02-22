
# @Time    : 2022/2/21 21:18
# @Author  : huangjisheng
# @File    : comb_time_series.py
# @Software: PyCharm
# !/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
class Loss_func(object):
    def __init__(self):
        self.y_true = y_true
        self.y_pred = y_pred

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

    def smape(y_true, y_pred):
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

use_loss = Loss_func.mape
loss_name = getattr(use_loss, '__name__')

forecast_df = pd.read_excel('comb_forecast_result.xlsx')

"""
features: 为不同模型预测结果
默认真实值为y
"""

features = ['auto_arima','holt_winters','sarimax','xgboost','prophet','ma']

def simple_avg_forecast(forecast_df):
    forecast_df['naive_comb_forecast'] = 0
    for fea in features:
        forecast_df['naive_comb_forecast'] += forecast_df[fea]
    forecast_df['naive_comb_forecast'] = forecast_df['naive_comb_forecast'] / len(features)
    return forecast_df


def weight_avg_forecast(forecast_df):
    forecast_df['{}_sum'.format(loss_name)] = 0
    forecast_df['{}_max'.format(loss_name)] = 0
    for fea in features:
        forecast_df["{}_{}".format(fea, loss_name)] = forecast_df.apply(lambda x: use_loss(x['y'], x[fea]), axis=1)
        forecast_df["{}_{}".format(fea, loss_name)] = forecast_df["{}_{}".format(fea, loss_name)].apply(
            lambda x: 0 if x <= 0 else x)

    for fea in features:
        forecast_df['{}_max'.format(loss_name)] = forecast_df.apply(
            lambda x: max(x['{}_max'.format(loss_name)], x["{}_{}".format(fea, loss_name)]), axis=1)

    for fea in features:
        forecast_df['{}_sum'.format(loss_name)] += forecast_df['{}_max'.format(loss_name)] - forecast_df[
            "{}_{}".format(fea, loss_name)]

    for fea in features:
        forecast_df["{}_weight_{}".format(fea, loss_name)] = (forecast_df['{}_max'.format(loss_name)] - forecast_df[
            "{}_{}".format(fea, loss_name)]) / forecast_df['{}_sum'.format(loss_name)]
    forecast_df['weight_avg_forecast'] = 0
    for fea in features:
        forecast_df['weight_avg_forecast'] += forecast_df["{}_weight_{}".format(fea, loss_name)] * forecast_df[fea]
    return forecast_df


def lasso_comb_forecast(forecast_df, target_col='y'):
    reg_data = forecast_df[features]
    target = [target_col]
    reg_target = forecast_df[target]
    lassocv = LassoCV()
    lassocv.fit(reg_data, reg_target)
    alpha = lassocv.alpha_
    print('best alpha is : {}'.format(alpha))
    lasso = Lasso(alpha=alpha)
    lasso.fit(reg_data, reg_target)
    num_effect_coef = np.sum(lasso.coef_ != 0)
    print('all coef num : {}. not equal coef num : {}'.format(len(lasso.coef_), num_effect_coef))
    lasso_coefs = lasso.coef_
    lst = zip(lasso_coefs, features)
    loss_coef_df = pd.DataFrame.from_dict(lst)
    loss_coef_df.columns = ['coef', 'feature']
    t = 'lasso_comb_forecast='
    for i in loss_coef_df['feature'].unique():
        coef = loss_coef_df[loss_coef_df['feature'] == i]['coef'].values[0]
        temp = str(i) + '*' + str(coef) + '+'
        t += temp
    forecast_df.eval(t[:-1], inplace=True)
    for fea in features:
        forecast_df['lasso_coef_{}'.format(fea)] = loss_coef_df[loss_coef_df['feature'] == i]['coef'].values[0]
    return forecast_df


def corr_comb_forecast(forecast_df):
    forecast_df_corr = forecast_df.corr()
    df_corr = pd.DataFrame(forecast_df_corr['y'].sort_values(ascending=False)[1:])

    print(df_corr)
    forecast_df_corr_re = forecast_df_corr.reset_index()
    corr_select_fea = forecast_df_corr_re[forecast_df_corr_re['index'] == 'y']
    corr_select_fea = corr_select_fea[features]
    corr_select_fea[features] = abs(corr_select_fea[features])
    corr_select_fea_min = min([corr_select_fea[fea].values[0]] for fea in corr_select_fea[features])[0]

    t_sum = 0
    for fea in features:
        corr_select_fea['corr_norm_{}'.format(fea)] = corr_select_fea[fea] - corr_select_fea_min
        print(corr_select_fea['corr_norm_{}'.format(fea)].values[0])
        t_sum += corr_select_fea['corr_norm_{}'.format(fea)].values[0]

    for fea in features:
        corr_select_fea['corr_norm_{}'.format(fea)] = corr_select_fea['corr_norm_{}'.format(fea)] / t_sum
    forecast_df['corr_forecast'] = 0
    for fea in features:
        forecast_df['corr_forecast'] += corr_select_fea['corr_norm_{}'.format(fea)].values[0] * forecast_df[fea]
        forecast_df['corr_norm_{}'.format(fea)] = corr_select_fea['corr_norm_{}'.format(fea)].values[0]
    return forecast_df


forecast_df = simple_avg_forecast(forecast_df)
forecast_df = weight_avg_forecast(forecast_df)
forecast_df = lasso_comb_forecast(forecast_df)
forecast_df = corr_comb_forecast(forecast_df)