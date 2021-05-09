

利用SARIMAX 进行销量预测

本文延续一个月前推送的销量预测模型系列，从传统的时间序列SARIMAX 算法讲解销量预测模型。

主要涉及到python的pandas，statsmodels,joblib等模块，通过对多个模型进行并行网格搜索寻找评价指标map最小模型的参数，虽然可以使用的模型非常多，从传统时间序列，到机器学习，深度学习算法。但是作为计量经济学主要内容之一，时间序列因为其强大成熟完备的理论基础，应作为我们处理带有时序效应的数据时的首先需要尝试的模型类型，且往往效果不错。本文只是实用代码的角度讲解其中statsmodels中SARIMAX 的用法，不涉及公式。

下面从代码开始剖析，以便于照例上手实际操作。

SARIMAX 是在差分移动自回归模型（ARIMA）的基础上加上季节性（S,Seasonal）和外部变量(X,eXogenous)

也就是说以ARIMA基础加上周期性和季节性，适用于时间序列中带有明显周期性和季节性特征的数据。



由于参数众多，所以下面简单介绍其含义，免得勿用或者遗漏。

| 参数                        | 含义                                             | 是否必须 |
| :-------------------------- | ------------------------------------------------ | :------- |
| **endog**                   | 观察（自）变量 y                                 | 是       |
| **exog**                    | 外部变量                                         | 否       |
| **order**                   | 自回归，差分，滑动平均项 (p,d,q)                 | 否       |
| **seasonal_order**          | 季节因素的自回归，差分，移动平均，周期 (P,D,Q,s) | 否       |
| **trend**                   | 趋势，c表示常数，t:线性，ct:常数+线性            | 否       |
| **measurement_error**       | 自变量的测量误差                                 | 否       |
| **time_varying_regression** | 外部变量是否存在不同的系数                       | 否       |
| **mle_regression**          | 是否选择最大似然极大参数估计方法                 | 否       |
| **simple_differencing**     | 简单差分，是否使用部分条件极大似然               | 否       |
| **enforce_stationarity**    | 是否在模型种使用强制平稳                         | 否       |
| **enforce_invertibility**   | 是否使用移动平均转换                             | 否       |
| **hamilton_representation** | 是否使用汉密尔顿表示                             | 否       |
| **concentrate_scale**       | 是否允许标准误偏大                               | 否       |
| **trend_offset**            | 是否存在趋势                                     | 否       |
| **use_exact_diffuse**       | 是否使用非平稳的初始化                           | 否       |
| **kwargs                    | 接受不定数量的参数，如空间状态矩阵和卡尔曼滤波   | 否       |



参数说明：

以上我们列出了改函数中所有可能有到的参数，可以看到很多参数不是必须指定，比如，甚至只是需要给定，endog(观察变量)，算法就可以运行起来，也正是这样，SARIMAX，具有极大的灵活性，主要体现在：

1 如果不指定seasonal_order，或者季节性参数都为0，那么就是普通的ARIMA模型；

2 exog，外部因子没有也可以不用指定，所以目前用python的statsmodels进行时间序列分析时，用SARIMAX就好了；

3，其他的参数如无必要，则不需要修改，因为函数默认的参数在大多数时候是最优的；

4上表多次提到，我们也多次见到，关于初始化，我们知道模型拟合所用迭代算法，是需要提供一个初始值的，在初始值的基础上不断迭代，一般情况下是随机的指定，在大多数梯度下降算法陷入局部最优的时候，可以尝试更改初始值，和上条一样，如无不要勿动；

5，关于拟合算法，我们一般都是假定给定的数据满足正态分布，所有使用极大似然算法求解最优参数；

6，关于是否强制平稳和移动平均转换，一般设置为False，保持灵活性。

总的来说，SARIMA 模型通过(p,d,q) (P,D,Q)m 不同的组合，囊括了ARIMA, ARMA, AR, MA模型，通过指定的模型评估准则，选择最优模型。



关于模型选择标准，提供以下三种思路：

1，使用AIC信息准则

通过Akaike information criteria (AIC)进行模型选择，用最大似然函数拟合模型，虽然我们的目标是似然函数最大化，但并非越大越好，我们同时需要考虑模型复杂度，所以常常使用AIC和BIC作为模型优劣的衡量标准，我们说的优劣是不同的模型进行比较的时候，只能说在这么多备选模型中，最小AIC的模型刻画的真实数据表达的信息损失最小，是一个相对指标。

AIC=-2 ln(*L*) + 2 *k* 

BIC=-2 ln(*L*) + ln(n)*k 

AIC在样本容量很大时，拟合所得数值会因为样本容量而放大。(通常超过1000的样本称之为大样本容量)

AIC准则和BIC准则作为模型选择的准则，可以有效弥补根据自相关图和偏自相关图定阶的主观性。

AIC是statsmodels模块中SARIMAX 函数默认模型评估准则。



2使用 Box-Jenkins建模分析过程

当然也可以使用Box-Jenkins 建模流程，如下：建立在反复验证是否满足假设前提，并对参数调整。

![img](https://img-blog.csdnimg.cn/20190220160453715.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE2MDUwNTYx,size_16,color_FFFFFF,t_70)





3，使用自定义模型评估准则，比如，MAPE（平均绝对误差百分比），从总体上评估模型预测准确率:
$$
M A P E=\frac{\sum_{i=1}^{n}\left(\frac{\left|y_{i}-e_{i}\right|}{y_{i}} \times 100\right)}{n}
$$


同时MAPE是反映误差大小的相对值，不同模型对同一数据的模型进行评估才有比较的意义。

需要说明的是，建模评估标准以上就列举了三种，最好综合考虑，同时可能不同的业务有不同的述求，AIC是信息论的角度度量信息损失大小， Box-Jenkins是传统的层层假设之下的时间序列统计建模准则，往往在应对单一序列模型，通过眼观图形和检验参数的显著性来判定，效果佳。而如果我们关注的是模型准确率，那么最好的当然是定义mape函数，不能无脑的用函数原始的参数，这点尤其值得关注，包括XGBoost模型中，目标损失函数不能直接使用RMSE。

导入模块

```python
import time
from itertools  import product
import numpy as np
import pandas as pd
from math import sqrt
from joblib import Parallel,delayed
import warnings
warnings.filterwarnings('ignore')
from warnings import catch_warnings,filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
```



定义模型,该部分的关键参数已经在上文全部罗列，因为重要所以从原文档中全部总结翻译过来。

```python
#传入数据和参数，输出模型预测
def model_forecast(history,config):
    order, sorder, trend = config
    model = SARIMAX(history, order=order, 		 seasonal_order=sorder,trend=trend,enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]
```



```python
#模型评估指标,mape
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100    

#划分训练集和测试集
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]
```

使用one-step滚动向前预测法，每次预测值再加入数据中，接着预测下一个值，而不是一次预测多个值。依据经验，我们可以知道多数情况下，其实滚动逐步预测比多步预测效果更佳，所以应该尝试滚动预测。

```python
#one-step滚动向前预测
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
```



当模型的移动平均或者自回归阶数较高时，模型计算极大似然的时候可能会抛出很多警告，比如常见的自由度为0，模型非正定，所以这里需要设置忽视警告,还如AIC有时会得到NaN值，这样的数据精度问题，在高阶模型求解中会出现，所以我们需要用到python中的try-except异常处理控制流，把可能报错的语法块放置在try中，以免程序中断。如果需要查看警告或者调试，则debug这里可以设置为True,但是大概率程序会报错退出，而你，一脸朦胧。对数据进行标准化等处理，其实在一定程度上是可以避免一些计算方面的问题，同时也会提高计算求解效率。

```python
#模型评估
def score_model(data,n_test,cfg,debug=False):
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
```



网格搜索

网格搜索非常耗时，时间复杂度非常高，为指数型。在条件允许的情况下，尤其是PC和服务器计算力极大提高的当下，做法通常都是用空间换时间，压榨计算资源，使用多线程并行，以便可以在短时间内得到结果。

所以，我们使用Joblib模块中的`Parallel`和`delayed`函数并行求解多个模型，Joblib模块也是常常在机器学习任务**grid search**和**Cross validation**中为了提高计算速度需要必备的。

```python
#网格搜索
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        #使用计算机全部的cpu核数多进程并行
        executor = Parallel(n_jobs=-1, backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
        
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    scores = [r for r in scores if r[1] != None]
    scores.sort(key=lambda x: x[1])
    return scores

#生成参数列表
def sarima_configs(seasonal=[0]):   
    p = d = q = [0,1,2]
    pdq = list(product(p, d, q))
    s = 0
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(product(p, d, q))]
    t=['n','c','t','ct']
    return list(product(pdq,seasonal_pdq,t))
```



还需要唠叨的是，本文引进的itertools模块中的product，是对迭代对象创建笛卡尔积的函数，穷尽迭代对象的所有组合，返回迭代对象组合的元组。解释一下 ，上面sarima_configs函数中的，p、 d、 q其实都是0、1、2，因为更高阶的并不多见，且高阶会导致模型非常复杂，往往0，1，2也就够了，季节性这里，设置了一个默认的0，是因为本文使用的是周这样的汇总时间点，通过前期数据探索作图看出设置为4，或者12都没有意义，所以为了节省计算资源，指定为0，不让程序计算选择该参数。以上函数，我们自己可以写嵌套循环，但是python内置的模块和成熟的模块在计算性能和规范上会比自己手写的优很多，所以这也是不要重复造轮子的理念，除非自己造的轮子更好，能解决一个新需求。既然讲到计算性能，因为本文涉及到了很多循环迭代，那么如果可以的话，建议使用Profile 这个内置的模块分析每个函数花费的时间。

以下为模型训练函数，n_test表示预测三个值，因为我个人使用的场景比较固定，所以就直接写在函数内部作为局部变量了，为了保持函数的灵活性，作为全局参数或者函数的形参当然是更好，另，下面这种列表元素append追加似乎不太优雅。（过早优化是万恶之源，emmm,就这样子，逃）

```python
#模型训练
def train_model(sale_df):
    sum_=0
    n_test = 3
    p_b,d_b,q_b=[],[],[]
    P_b,D_b,Q_b=[],[],[]
    m_b,t_b=[],[]
    model_id,error=[],[]
    for i in sale_df['store_code'].unique():
        data=sale_df[sale_df['store_code']==i]['y']
        data=[i for i in data]
        cfg_list = sarima_configs()
        scores = grid_search(data,cfg_list,n_test,parallel=True)
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
        params_df=pd.DataFrame({'store_code': model_id, 'map': error,'p':p_b,'d':d_b,'q':q_b,'P':P_b,'D':D_b,'Q':Q_b,'m':m_b,'t':t_b})
    return params_df
```



通过模型训练得到的最优参数，传递，滚动预测四个时间点。

```python
#定义预测函数，传入数据和参数，返回预测值
def one_step_forecast(data,order,seasonal_order,t,h_fore):
    predictions=list()
    data=[i for i in data]
    for i in range(h_fore):
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order,trend=t,enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        yhat = model_fit.predict(len(data), len(data))
        data.append(yhat[0])
        predictions.append(yhat[0])
    return predictions


#用for循环，多个序列预测
def forecast_model(sale_df,params_df):
    h_fore=4
    fore_list=[]
    model_id=[]
    for i in sale_df['store_code'].unique():
        params_list=params_df[params_df['store_code']==i]
        data=sale_df[sale_df['store_code']==i]['y']
        p=params_df[params_df['store_code']==i].iloc[:,2].values[0]
        d=params_df[params_df['store_code']==i].iloc[:,3].values[0]
        q=params_df[params_df['store_code']==i].iloc[:,4].values[0]
        P=params_df[params_df['store_code']==i].iloc[:,5].values[0]
        D=params_df[params_df['store_code']==i].iloc[:,6].values[0]
        Q=params_df[params_df['store_code']==i].iloc[:,7].values[0]
        m=params_df[params_df['store_code']==i].iloc[:,8].values[0]
        t=params_df[params_df['store_code']==i].iloc[:,9].values[0]
        order=(p, d, q)
        seasonal_order=(P,D,Q,m)
        all_fore=one_step_forecast(data,order,seasonal_order,t,h_fore)
        fore_list.append(all_fore)
        
        #以下为，多步预测，如果不使用滚动预测，则不调one_step_forecast函数
        #model=SARIMAX(data, order=order,seasonal_order=seasonal_order,trend=t,enforce_stationarity=False,
        #                                                enforce_invertibility=False)
        #forecast_=model.fit(disp=-1).forecast(steps=h_fore)
        #fore_list_flatten = [x for x in forecast_]
        #fore_list.append(fore_list_flatten)
        model_id.append(i)
    df_forecast = pd.DataFrame({'store_code': model_id, 'fore': fore_list})
    return df_forecast

```

以下就是主函数了

```python
if __name__ == '__main__':
    start_time=time.time()
    sale_df=pd.read_excel('/home/test01/store_forecast/sale_df.xlsx')
    params_df=train_model(sale_df)
    forecast_out=forecast_model(sale_df,params_df)
    end_time=time.time()
    use_time=(end_time-start_time)//60
    print('finish the process use',use_time,'mins')
```



以下展示为本次模型所得结果，每个门店一个序列模型

| store_code | mape     | p    | d    | q    | P    | D    | Q    | m    | t    |
| ---------- | -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1F         | 6.305378 | 1    | 0    | 2    | 2    | 0    | 1    | 0    | c    |
| 62         | 1.889192 | 0    | 2    | 2    | 0    | 0    | 2    | 0    | t    |
| CS         | 1.425515 | 1    | 2    | 2    | 0    | 0    | 0    | 0    | c    |
| 2H         | 2.144674 | 0    | 1    | 2    | 1    | 0    | 2    | 0    | c    |
| 32         | 5.289745 | 0    | 2    | 2    | 0    | 0    | 0    | 0    | t    |



五个模型总体MAPE为3.4%,效果还不错，本文没有还没有用到SARIMAX 中的X,也就是eXogenous外部因素，一来，看到模型得到的MAPE已经达到3.4%,准确率到了96.6%，但是限于时间的关系，就没有加入如天气客流等外部因素。

本文完全站在工程实现的角度，考虑多种参数组合模型，通过并行网格搜索，回测得到我们定义的准则MAPE最小的模型参数，并把最优参数作为模型预测未来值的参数，滚动预测未来4个时间点。以上就是结合自己经验和体会以及查阅的资料针对几个关键点进行阐述而成。如有误，欢迎指正留言，完整的程序和数据会放在github上，因为是企业数据，不便于公布，所以在UCI上找了一个类似的数据集，如有需要请点击【阅读原文】

