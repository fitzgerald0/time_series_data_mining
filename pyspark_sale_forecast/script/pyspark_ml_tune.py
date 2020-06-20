#-*- coding:utf-8 -*-
"""
Name  : pyspark_ml_tune.py
Time  : 2020/6/20 17:26
Author : hjs
"""


"""
tune the  best param for linear model

"""

import pandas as pd
import datetime
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *
from pyspark.sql.types import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#spark初始化
spark = SparkSession. \
    Builder(). \
    config("spark.sql.crossJoin.enabled", "true"). \
    config("spark.sql.execution.arrow.enabled", "false"). \
    enableHiveSupport(). \
    getOrCreate()

today = (datetime.datetime.today()).strftime('%Y-%m-%d')
update_time = str(datetime.datetime.now().strftime('%b-%m-%y %H:%M:%S')).split(' ')[-1]
#spark.sql数据读取
df = spark.sql("""
select store_code,goods_code,ds,qty as label
from xxx.temp_store_sale
where ds>='2020-05-22'
""")

#数据此时是spark.dataframe格式，用类sql的形式进行操作
df = df.withColumn('dayofweek', dayofweek('ds'))
df = df.withColumn("dayofweek", df["dayofweek"].cast(StringType()))
# 是否月末编码
df = df.withColumn('day', dayofmonth('ds'))
df = df.withColumn('day', df["day"].cast(StringType()))
df = df.withColumn('month_end', when(df['day'] <= 25, 0).otherwise(1))

# 星期编码--将星期转化为了0-1变量，从周一至周天
dayofweek_ind = StringIndexer(inputCol='dayofweek', outputCol='dayofweek_index')
dayofweek_ind_model = dayofweek_ind.fit(df)
dayofweek_ind_ = dayofweek_ind_model.transform(df)
onehotencoder = OneHotEncoder(inputCol='dayofweek_index', outputCol='dayofweek_Vec')
df = onehotencoder.transform(dayofweek_ind_)


#此时产生的dayofweek_Vec是一个向量
inputCols = [
    "dayofweek_Vec",
    "month_end"]


assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
#数据集划分,此时是随机切分，不考虑时间的顺序
train_data_1, test_data_1 = df.randomSplit([0.7, 0.3])
train_data=assembler.transform(train_data_1)
test_data = assembler.transform(test_data_1)


lr_params = ({'regParam': 0.00}, {'fitIntercept': True}, {'elasticNetParam': 0.5})

lr = LinearRegression(maxIter=100, regParam=lr_params[0]['regParam'], \
                      fitIntercept=lr_params[1]['fitIntercept'], \
                      elasticNetParam=lr_params[2]['elasticNetParam'])

lrParamGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.005, 0.01, 0.1, 0.5]) \
    .addGrid(lr.fitIntercept, [False, True]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.5, 1.0]) \
    .build()

model = lr.fit(train_data)
pred = model.evaluate(test_data)

#调参前的模型评估
eval = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mae")
mae = eval.evaluate(pred.predictions, {eval.metricName: "mae"})
r2 = eval.evaluate(pred.predictions, {eval.metricName: "r2"})

# 本次不需要预测，故注释掉
#predictions = model.transform(test_data)
#predictions.printSchema()

cross_valid = CrossValidator(estimator=lr, estimatorParamMaps=lrParamGrid, evaluator=RegressionEvaluator(),
                          numFolds=5)

cvModel = cross_valid.fit(train_data)

best_parameters = [(
    [{key.name: paramValue} for key, paramValue in zip(params.keys(), params.values())], metric) \
    for params, metric in zip(
        cvModel.getEstimatorParamMaps(),
        cvModel.avgMetrics)]

lr_best_params = sorted(best_parameters, key=lambda el: el[1], reverse=True)[0]


#借用pd.DataFrame把以上关键参数转换为结构化数据
pd_best_params = pd.DataFrame({
    'regParam':[lr_best_params[0][0]['regParam']],
    'fitIntercept':[lr_best_params[0][1]['fitIntercept']],
    'elasticNetParam':[lr_best_params[0][2]['elasticNetParam']]
}
)

pd_best_params['update_date'] = today
pd_best_params['update_time'] = update_time
pd_best_params['model_type'] = 'linear'

# 最优参数进行再次模型训练
lr = LinearRegression(maxIter=100, regParam=lr_best_params[0][0]['regParam'], \
                      fitIntercept=lr_best_params[0][1]['fitIntercept'], \
                      elasticNetParam=lr_best_params[0][2]['elasticNetParam'])

model = lr.fit(train_data)

pred = model.evaluate(test_data)

eval = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mae")

#回归模型评估参数有很多种，基于模型建模效果得检验是R方,本文作为一个时序预测，更关注得是mae，所以我们输出两个评估结果
r2_tune = eval.evaluate(pred.predictions, {eval.metricName: "r2"})
mae_tune = eval.evaluate(pred.predictions, {eval.metricName: "mae"})

pd_best_params['mae_value'] = str(mae_tune)
#虽然这里pd_best_params中mae_value的数据类型为str，但是在写入过程中，会有一个类型推断，
# 所以最后在hive中查看会知道这是一个float类型
#pd.DataFrame-->spark.dataframe 然后写入表中，以追加得形式写入hive，得到的最优参数供模型预测使用
spark.createDataFrame(pd_best_params).write.mode("append").format('hive').saveAsTable(
    'xxx.regression_model_best_param')
