# 第x章:销量预测spark实战

**本章目录**

1.Spark.ML与DataFrame简介
2.销量预测特征工程
2.销量预测特征选择和超参数调优
4.销量预测Spark算法模型实战



*本文是PySpark销量预测系列第一篇，后面会陆续通过实战案例详细介绍PySpark销量预测流程，包含特征工程、特征筛**选、超参搜索、预测算法。*



在零售销量预测领域，销售小票数据动辄上千万条，这个量级在单机版上进行数据分析/挖掘是非常困难的，所以我们需要借助大数据利器--Spark来完成。

Spark作为一个快速通用的分布式计算平台，可以高效的使用内存，向用户呈现高级API，这些API将转换为复杂的并行程序，用户无需深入底层。

由于数据挖掘或分析人员，大多数熟悉的编程语言是Python，所以本章我们介绍Spark的Python版--PySpark。本节先介绍必要的基础知识比如DataFrame和ML库，在后续章节中会给出基于Spark机器学习的特征生成/特征选择/超参数调优以及机器学习销量预测算法。



## **1.Spark.DataFrame与Spark.ML简介**



从Spark 2.0开始，Spark机器学习API是基于DataFrame的Spark.ML ,而之前基于RDD的Spark.MLlib已进入维护模式，不再更新加入新特性。基于DataFrame的Spark.ML是在RDD的基础上进一步的封装，也是更加强大方便的机器学习API,同时如果已经习惯了Python机器学习库如sklearn等，那么你会发现ML用起来很亲切。

下面我们就开始介绍DataFrame和ML

DataFrame 从属于 Spark SQL 模块，适用于结构化/数据库表以及字典结构的数据，执行数据读取操作返回的数据格式就是DataFrame，同时熟悉Python的pandas库或者R语言的同学来说，更是觉得亲切，Spark.DataFrame正是借鉴了二者。DataFrame的主要优点是Spark引擎在一开始就为其提供了性能优化，与Java或者Scala相比，Python中的RDD非常慢。每当使用RDD执行PySpark程序时，在PySpark驱动器中，启动Py4j使用JavaSparkContext的JVM，PySpark将数据分发到多个节点的Python子进程中，此时Python和JVM之间是有很多上下文切换和通信开销，而DataFrame存在的意义就是优化PySpark的查询性能。

以上我们交代了Spark.DataFrame的由来，下面介绍其常见操作。

![img](https://mmbiz.qpic.cn/mmbiz_png/wufCEEo7jqpvGdhYMibALv6dicjuBqZAfam5icHErcAox8RE2ZhtpJFibS7TKZnyODqCTMQyY2BzeLSxicmoaSN0Y7w/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



### 1.1 Spark.DataFrame生成

(1)使用toDF(基于RDD)

```python
from pyspark import SparkConf,SparkContext
from pyspark.sql import Row
conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)
df = sc.parallelize([ \
    Row(name='Alice', age=5, height=80), \
    Row(name='Alice', age=5, height=80), \
    Row(name='Alice', age=10, height=80)]).toDF()

#查看数据类型
df.dtypes
#[('age', 'bigint'), ('height', 'bigint'), ('name', 'string')]
查看df类型
type(df)
#class 'pyspark.sql.dataframe.DataFrame'>
```

可以将DataFrame视为关系数据表，在其上进行类似于SQL的操作，同时与平时建SQL表需要指定数据类型不同的是，此时数据列的类型是自动推断，这也是其强大之处。

(2)读取本地文件

```python 
from pyspark.sql import SparkSession

spark = SparkSession.builder \
		.master("local") \
	  .appName("Test Create DataFrame") \
	  .config("spark.some.config.option", "some-value") \
    .getOrCreate()
df = spark.read.csv('python/test_spark/ts_dataset.csv')

```

同理还可以读取parquet/json文件

```python
df_parquet=spark.read.parquet('....')

df_json = spark.read.format('json').load('python/test_spark/ts_dataset.json')

```

以上两种方式中，第一种是Spark1.x版本中以RDD作为主要API的方式，第二种的SparkSession是随着spark2.x引入，封装了SparkContext、SparkConf、sqlContext等，为用户提供统一的接口来使用Spark各项功能的更高级抽象的启动方式。

强调一点是，我们通过会话SparkSession读取出来的数据类型就是DataFrame，而第一种还需要在RDD的基础上使用toDF进行转换。如果当前读者使用的spark版本是2，那么，推荐使用第二种方式。



(3)读取HIVE表

```python
from pyspark.sql import SparkSession
spark = SparkSession. \
    Builder(). \
    config("spark.sql.crossJoin.enabled", "true"). \
    config("spark.sql.execution.arrow.enabled", "true"). \
    enableHiveSupport(). \
    getOrCreate()
df=spark.sql("""select regparam,fitIntercept, elasticNetParam from temp.model_best_param""")

```

这种类型和上文直接读取本地文件类似，Spark任务在创建时，是默认支持Hive，可以直接访问现有的 Hive支持的存储格式。解释一下，Apache Hive是Hadoop上一种常见的结构化数据源，支持包含HDFS在内的多种存储系统上的表，由于实际工作中我们使用spark.sql读取数据操作的机会更多，也是spark最核心组件之一，所以这里重点讲解一些Spark.SQL。与Spark其他的组件一样，在使用的时候是需要提前引入Spark.SQL，但也无需依赖大量的包，如果需要把Spark.SQL连接到一个部署好的Hive上，则需要把hive-site.xml复制到spark的配置文件目录中，该部分内容参考网络上其他的教程。以上代码中enableHiveSupport的调用使得SparkSession支持Hive。如果是Spark 1.x版本，则使用以下方式引用。

```python
from pyspark.sql import HiveContext
hiveCtx=HiveContext(sc)
data=hiveCtx.sql("select regparam,fitIntercept, elasticNetParam from temp.model_best_para ")
```





(4)pandas.DataFrame转换而来

既然使用python进行数据处理，尤其是结构化数据，那么pandas一定绕不开，所以我们经常会有把做过一些处理的pandas.DataFrame数据转换为Spark.DataFrame的诉求，好在Spark.DataFrame在设计之初就参考并考虑到了这个问题，所以实现方式也相当简单。

```python
import pandas as pd
df = pd.read_csv('python/test_spark/ts_dataset.csv')
#将pandas.Dataframe 转换成-->spark.dataFrame 
spark_df=spark.createDataFrame(df)
#将spark.dataFrame 转换成--> pandas.Dataframe
pd_df = spark_df.toPandas()
```

以上将Spark.DataFrame 转换成--> pandas.Dataframe的过程，不建议对超过10G的数据执行该操作。

本节开头我们也说了Spark.DataFrame是从属于Spark.sql的，Spark.sql作为Spark最重要的组件，是可以从各种结构化数据格式和数据源读取和写入的，所以上面我们也展示了读取json/csv等本地以及数据库中的数据。同时spark还允许用户通过thrift的jdbc远程访问数据库。总的来说 Spark 隐藏了分布式计算的复杂性， Spark SQL 、DataFrame更近一步用统一而简洁的API接口隐藏了数据分析的复杂性。从开发速度和性能上来说，DataFrame + SQL 无疑是大数据分析的最好选择。

### 1.2 Spark.DataFrame操作

以上我们强调了Spark.DataFrame可以灵活的读取各种数据源，数据读取加载后就是对其进行处理了，下面介绍读取DataFrame格式的数据以后执行的一些简单的操作。

(1)展示DataFrame

```python
spark_df.show()
```

- 打印DataFrame的Schema信息

```
spark_df.printSchema()
```

- 显示前n行

```
spark_df.head(5)
```

- 显示数据长度与列名

```python
df.count()
df.columns
```



(2)操作DataFrame列

- 选择列

```
ml_dataset=spark_df.select("features", "label")
```

- 增加列

```python
from pyspark.sql.functions import *
#注意这个*号，这里是导入了sql.functions中所有的函数，所以下文的abs就是由此而来
df2 = spark_df.withColumn("abs_age", abs(df2.age))
```

- 删除列

```python
df3= spark_df.drop("age")
```

- 筛选

```python
df4= spark_df.where(spark_df["age"]>20)
```



以上只是简单的展示了一小部分最为常见的DataFrame操作，更详尽的内容请查阅官方文档或者其他参考资料。



### 1.3 Spark.ML简介

以上我们介绍了与Spark.ML机器学习密切相关的数据类型和基本操作--Spark.DataFrame

犹如我们通过pandas.DataFrame对数据做加工，下面我们看看用这些清洗过后的制作佳肴的过程--机器学习建模。

ML包括三个主要的抽象类：转换器（Transformer）、评估器（Estimator）和管道（Pipeline）。

转换器，顾名思义就是在原对象的基础上对DataFrame进行转换操作，常见的有spark.ml.feature中的对特征做归一化，分箱，降度，OneHot等数据处理，通过`transform()`方法将一个DataFrame转换成另一个DataFrame。

评估器，评估器是用于机器学习诸如预测或分类等算法，训练一个DataFrame并生成一个模型。用实现fit()方法来拟合模型。

```python
from pyspark.ml.feature import MinMaxScaler
#定义/引入转换类
max_min_scaler = MinMaxScaler(inputCol="age", outputCol="age_scaler")
#fit数据
max_min_age = max_min_scaler.fit(df)
#执行转换
max_min_age_=max_min_age.transform(spark_df)
```



管道
管道这一概念同样受Python的Scikit-Learn库的影响，PySpark ML中的管道指从转换到评估的端到端的过程，为简化机器学习过程并使其具备可扩展性，采用了一系列 API 定义并标准化机器学习工作流，包含数据读取、预处理、特征加工、特征选择、模型拟合、模型验证、模型评估等一系列工作，对DataFrame数据执行计算操作。Spark机器学习部分其他的如特征生成，模型训练，模型保存，数据集划分/超参数调优，后面我们会有实际案例进行详细阐述。另外，随着Spark.3.0的发布，最近的ML简介可以通过此链接了解。

http://spark.apache.org/docs/latest/ml-guide.html



顺便介绍几本手头上的相关书籍

1.Spark快速大数据分析，本书有些旧，主要是spark.1.x为主，少量的spark.2.X介绍，如果想要了解或者不得不使用rdd based  APIs进行数据分析或者想深入spark更底层学习一点scala等函数式编程入门的还是不错的选择，比较全面通俗。豆瓣评分7.9

2.PySpark实战指南，用python进行spark数据分析那就不得不提到这本书，倒不见得有多好，只是目前市面上也没有更好的专门使用python介绍spark的了，本书从rdd到mllib的介绍以及ml包的介绍，可以通过书中提供的api介绍了解使用python进行spark机器学习的过程，当然机器学习的一些细节是没有涉及到的，总的来说更多的是展示流程和api的使用。

至于spark乃至于hadoop的书市面上可就非常多了，个人也不是专长做这一块的，所以也就不好品论。

