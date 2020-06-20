#-*- coding:utf-8 -*-
"""
Name  : pyspark_ml_predict.py
Time  : 2020/6/20 17:36
Author : hjs
"""

from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
spark = SparkSession. \
    Builder(). \
    config("spark.sql.crossJoin.enabled", "true"). \
    config("spark.sql.execution.arrow.enabled", "false"). \
    enableHiveSupport(). \
    getOrCreate()

today = (datetime.datetime.today()).strftime('%Y-%m-%d')

df = spark.sql(f"""
    select store_code,goods_code,ds,qty
	from xxx.temp_store_sale
	where ds>='{prev28_day}' and ds<'{today}'
    union
    select s.store_code,s.goods_code,d.ds,0 as qty
    from
    (select stat_date as ds from xxx.dim_date where stat_date<'{after7_day}' and stat_date>='{today}') d
    join
    (select
    distinct
    store_code,goods_code
    from sxxx.temp_store_sale
    ) s on 1=1""")

#读取最佳参数
best_param_set=spark.sql(f"select regparam,fitIntercept, elasticNetParam from xxx.regression_model_best_param order by update_date desc,update_time desc limit 1 ").collect()

reg_vec=best_param_set.select('regparam')
reg_b= [row.regparam for row in reg_vec][0]
reg_b=float(reg_b)

inter_vec =best_param_set.select('fitIntercept')
inter_b = [row.fitIntercept for row in inter_vec][0]
#str --> boole
if inter_b=='false':
    inter_b=False
else:
    inter_b=True

elastic_vec =best_param_set.select('elasticNetParam')
elastic_b = [row.elasticNetParam for row in elastic_vec][0]
elastic_b=float(elastic_b)

#特征处理

df=df.withColumn('dayofweek',dayofweek('ds'))
df = df.withColumn("dayofweek", df["dayofweek"].cast(StringType()))

#是否月末编码
df=df.withColumn('day',dayofmonth('ds'))
df = df.withColumn('day', df["day"].cast(StringType()))
df = df.withColumn('month_end',when(df['day'] <=25,0).otherwise(1))

#星期编码--将星期转化为了0-1变量
dayofweek_ind = StringIndexer(inputCol='dayofweek', outputCol='dayofweek_index')
dayofweek_ind_model = dayofweek_ind.fit(df)
dayofweek_ind_ = dayofweek_ind_model.transform(df)
onehotencoder = OneHotEncoder(inputCol='dayofweek_index', outputCol='dayofweek_Vec')
df = onehotencoder.transform(dayofweek_ind_)


inputCols=[
"dayofweek_Vec",
"month_end"]

assembler = VectorAssembler(inputCols=inputCols, outputCol="features")


#使用where自定义切分数据集
train_data=df.where(df['ds'] <today)
test_data=df.where(df['ds'] >=today)

train_mod01 = assembler.transform(train_data)
train_mod02 = train_mod01.selectExpr("features","qty as label")

test_mod01 = assembler.transform(test_data)
test_mod02 = test_mod01.select("store_code","goods_code","ds","features")

# build train the model
lr = LinearRegression(maxIter=100,regParam=reg_b, fitIntercept=inter_b,elasticNetParam=elastic_b, solver="normal")
model = lr.fit(train_mod02)


# predict
predictions = model.transform(test_mod02)
print('print the schema')
predictions.printSchema()
predictions.select("store_code","goods_code","ds","prediction").show(5)
log.info('predictions shape'+str(predictions.count()))

test_store_predict=predictions.select("store_code","goods_code","ds","prediction").createOrReplaceTempView('test_store_predict')
spark.sql(f"""create table xxx.regression_test_store_predict as select * from test_store_predict""")
