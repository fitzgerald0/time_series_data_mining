# -*- coding: utf-8 -*-
# @Time    : 2020/10/3 14:58
# @Author  : hjs
# @File    : adfuller_test.py

#this is example for spark udf run adfuller test

import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *

spark = SparkSession. \
    Builder(). \
    config("spark.sql.crossJoin.enabled", "true"). \
    enableHiveSupport(). \
    getOrCreate()


df=spark.sql("select * from test.app_forecast_input_fix")


schema = StructType([
    StructField("store_id", StringType()),
    StructField("is_adfuller", DoubleType())
])

@pandas_udf(schema, functionType=PandasUDFType.GROUPED_MAP)
def adfuller_func(df):
    df.sort_values(by=['date'],ascending=[True],inplace=True)
    adfuller_result=adfuller(df['qty'],autolag='AIC')
    is_adfuller=None
    if adfuller_result[1]<0.05:
        is_adfuller=1
    else:
        is_adfuller=0
    result=pd.DataFrame({'store_id':df['store_id'].iloc[0],'is_adfuller':[is_adfuller]})
    return result

adfuller_result = df.groupby(['store_id']).apply(adfuller_func)
adfuller_result.printSchema()
adfuller_result.createOrReplaceTempView('adfuller_result')
spark.sql("""drop table if exists test.adfuller_result""")
spark.sql("""create table test.adfuller_result as select * from adfuller_result""")



