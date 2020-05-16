#-*- coding:utf-8 -*-
"""
Name  : prophet_spark_demo.py
Time  : 2020/5/16 10:25
Author : hjs
"""


import datetime
from dateutil.relativedelta import relativedelta
from fbprophet import Prophet
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *

spark = SparkSession. \
    Builder(). \
    config("spark.sql.crossJoin.enabled", "true"). \
    config("spark.sql.execution.arrow.enabled", "true"). \
    enableHiveSupport(). \
    getOrCreate()


def holiday_set():
    """"
    更新周期:年
    更新方式:手动
    优化点:
    1.观察该节假日是否存在假日效应，如果存在则设置，促销也可以作为假日因素考虑.
    2.如果日期周期短，可以把不同的假日合并
    """
    new_year_day = pd.DataFrame({
        'holiday': 'new_year_day',
        'ds': pd.to_datetime(['2020-01-01', '2021-01-01']),
        'lower_window': -2,
        'upper_window': 1,
    })
    # 受疫情的影响,2020年春节时间长
    spring_festival = pd.DataFrame({
        'holiday': 'spring_festival',
        'ds': pd.to_datetime(['2020-01-24']),
        'lower_window': -4,
        'upper_window': 14,
    })
    valentine_day = pd.DataFrame({
        'holiday': 'valentine_day',
        'ds': pd.to_datetime(['2019-02-14', '2020-02-14']),
        'lower_window': -1,
        'upper_window': 0,
    })
    tomb_sweeping = pd.DataFrame({
        'holiday': 'tomb_sweeping',
        'ds': pd.to_datetime(['2019-04-05', '2020-04-05']),
        'lower_window': 0,
        'upper_window': 1,
    })
    labour_day = pd.DataFrame({
        'holiday': 'labour_day',
        'ds': pd.to_datetime(['2019-05-01', '2020-05-01']),
        'lower_window': -1,
        'upper_window': 2,
    })
    children_day = pd.DataFrame({
        'holiday': 'children_day',
        'ds': pd.to_datetime(['2019-06-01', '2020-06-01']),
        'lower_window': -1,
        'upper_window': 0,
    })

    shopping_618 = pd.DataFrame({
        'holiday': 'shopping_618',
        'ds': pd.to_datetime(['2019-06-18', '2020-06-18']),
        'lower_window': 0,
        'upper_window': 1,
    })
    mid_autumn = pd.DataFrame({
        'holiday': 'mid_autumn',
        'ds': pd.to_datetime(['2019-09-13']),
        'lower_window': 0,
        'upper_window': 0,
    })
    national_day = pd.DataFrame({
        'holiday': 'national_day',
        'ds': pd.to_datetime(['2019-10-01', '2020-10-01']),
        'lower_window': -1,
        'upper_window': 6,
    })

    double_eleven = pd.DataFrame({
        'holiday': 'double_eleven',
        'ds': pd.to_datetime(['2019-11-11', '2020-11-11']),
        'lower_window': -1,
        'upper_window': 0,
    })
    year_sale = pd.DataFrame({
        'holiday': 'year_sale',
        'ds': pd.to_datetime(['2019-12-05', '2019-12-31']),
        'lower_window': 0,
        'upper_window': 1,
    })
    double_twelve = pd.DataFrame({
        'holiday': 'double_twelve',
        'ds': pd.to_datetime(['2019-12-12', '2020-12-12']),
        'lower_window': 0,
        'upper_window': 0,
    })

    christmas_day = pd.DataFrame({
        'holiday': 'christmas_day',
        'ds': pd.to_datetime(['2019-12-25', '2020-12-25']),
        'lower_window': -1,
        'upper_window': 0,
    })

    holidays_df = pd.concat(
        (new_year_day,
         spring_festival,
         valentine_day,
         tomb_sweeping,
         labour_day,
         children_day,
         shopping_618,
         mid_autumn,
         national_day,
         double_eleven,
         year_sale,
         double_twelve,
         christmas_day))

    holidays_set = holidays_df[['ds', 'holiday',
                                'lower_window', 'upper_window']].reset_index()
    return holidays_set


holiday_df = holiday_set()



def sale_ds(df):
    df['ds'] = pd.to_datetime(df['ds'])
    df = df[['store_sku', 'ds', 'y']]
    # 控制长度,周期不用太长，关注最近的几个完整周期即可
    start_day = (
            df['ds'].max() -
            relativedelta(
                days=63)).strftime('%Y-%m-%d')
    df = df[df['ds'] >= start_day][['store_sku', 'ds', 'y']]
    # 筛选条件：1 序列长度大于等于14，且过去最少有七天的销售记录；
    # 条件1，保障模型有两个完整的周期数据；
    # 条件2，避免出现0，0，0，0，0，0，1，0，1这样数据稀疏的数据出现
    sale_set = df.groupby(
        ['store_sku']).filter(
        lambda x: len(x) >= 14 and np.sum(
            x['y']) > 7)
    return sale_set


def replace_fill(data):
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
    data.loc[data['y'].isnull(), 'y'] = data['y'].interpolate(methon='nearest', order=3)
    low = data[data['y'] > 0]['y'].quantile(0.10)
    high = data[data['y'] > 0]['y'].quantile(0.90)
    data.loc[data['y'] < low, 'y'] = np.NaN
    data.loc[data['y'] > high, 'y'] = np.NaN
    data['y'] = data['y'].fillna(data['y'].mean())
    data['y'] = np.log1p(data['y'])
    return data


def prophet_train(data):
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=False,
        holidays=holiday_df,
        holidays_prior_scale=10)
    model.add_seasonality(
        name='weekly',
        period=7,
        fourier_order=3,
        prior_scale=0.10)
    model.fit(data)
    future = model.make_future_dataframe(periods=7, freq='d')
    forecast = model.predict(future)
    forecast['pro_pred'] = np.expm1(forecast['yhat'])
    forecast_df=forecast[['store_sku','ds','pro_pred']]
    # 对预测值修正
    forecast_df.loc[forecast_df['pro_pred'] < 0, 'pro_pred'] = 0
    low = (1 + 0.1) * data['y'].min()
    hight = min((1 + 0.05) * data['y'].max(), 10000)
    forecast_df.loc[forecast_df['pro_pred'] < low, 'pro_pred'] = low
    forecast_df.loc[forecast_df['pro_pred'] > hight, 'pro_pred'] = hight
    return forecast

def prophet_main(data):
    true_time = pd.datetime.now().strftime('%Y-%m-%d')
    data.dropna(inplace=True)
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds'] < true_time]
    data['ds'] = data['ds'].astype(str)
    data['ds'] = pd.to_datetime(data['ds'])
    # 异常值替换
    data = replace_fill(data)
    pro_back = prophet_train(data)
    return pro_back

schema = StructType([
    StructField("store_sku", StringType()),
    StructField("ds", StringType()),
    StructField("pro_pred", DoubleType())
])

@pandas_udf(schema, functionType=PandasUDFType.GROUPED_MAP)
def run_model(data):
    data['store_sku']=data['store_sku'].astype(str)
    df = prophet_main(data)
    uuid = data['store_sku'].iloc[0]
    df['store_sku']=unid
    df['ds']=df['ds'].astype(str)
    df['pro_pred']=df['pro_pred'].astype(float)
    cols=['store_sku','ds','pro_pred']
    return df[cols]

data = spark.sql(
    """
    select concat(store_code,'_',goods_code) as store_sku,qty_fix as y,ds
    from scmtemp.csh_etl_predsku_store_sku_sale_fix_d""")
data.createOrReplaceTempView('data')
sale_predict = data.groupby(['store_sku']).apply(run_model)
sale_predict.createOrReplaceTempView('test_read_data')
# 保存到数据库
spark.sql(f"drop table if exists scmtemp.tmp_hjs_store_sku_sale_prophet")
spark.sql(f"create table scmtemp.tmp_hjs_store_sku_sale_prophet as select * from store_sku_predict_29 ")
print('完成预测')
