#本文为使用tensorflow概率建模

%matplotlib inline
import pandas as pd
import numpy as np
import gc
#import matplotlib as mpl
from matplotlib import pylab as plt
import seaborn as sns


import collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts


#绘制预测与真实值时间序列图

def plot_forecast(x, y,
                  forecast_mean, forecast_scale, forecast_samples,
                  title, x_locator=None, x_formatter=None):
    """Plot a forecast distribution against the 'true' time series."""
    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    num_steps = len(y)
    num_steps_forecast = forecast_mean.shape[-1]
    num_steps_train = num_steps - num_steps_forecast
    

    ax.plot(x, y, lw=2, color=c1, label='ground truth')
    forecast_steps = np.arange(
      x[num_steps_train],
      x[num_steps_train]+num_steps_forecast,
      dtype=x.dtype)
    #
    ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)
    #绘制预测期望值以及针对最后三个月的100个采样结果
    ax.plot(forecast_steps, forecast_mean, lw=2, ls='--', color='r',
           label='forecast')
    ax.fill_between(forecast_steps,
                   forecast_mean-2*forecast_scale,
                   forecast_mean+2*forecast_scale, color=c2, alpha=0.2)

    ymin, ymax = min(np.min(forecast_samples), np.min(y)), max(np.max(forecast_samples), np.max(y))
    yrange = ymax-ymin
    ax.set_ylim([ymin - yrange*0.1, ymax + yrange*0.1])
    ax.set_title("{}".format(title))
    ax.legend()
 
    if x_locator is not None:
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)
        fig.autofmt_xdate()
        
    return fig, ax

#建构模型，并计算计算损失函数
def cal_loss(training_data):
    
    #设置全局默认图形
    tf.reset_default_graph()
    #遵循加法模型，设置趋势
    trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
    
    #设置季节性
    seasonal = tfp.sts.Seasonal(
          num_seasons=12, observed_time_series=observed_time_series)
    #模型拟合,之所以用sum，而不是我们在建模中常见的fit定义，是因为，
    #模型时间序列为加法模型，有如上文提到的趋势，季节性，周期性等成分相加
    #默认的先验分布为正态（normal）
    ts_model = sts.Sum([trend, seasonal], observed_time_series=observed_time_series)

    #构建变分损失函数和后验
    with tf.variable_scope('sts_elbo', reuse=tf.AUTO_REUSE):
        elbo_loss, variational_posteriors = tfp.sts.build_factored_variational_loss(
          ts_model,observed_time_series=training_data)
    
    return ts_model,elbo_loss,variational_posteriors


#模型训练，输出后验分布
def run(training_data):
    
    ts_model,elbo_loss,variational_posteriors=cal_loss(training_data)
    num_variational_steps = 401 
    num_variational_steps = int(num_variational_steps)

    #训练模型，ELBO作为在变分推断的损失函数
    train_vi = tf.train.AdamOptimizer(0.1).minimize(elbo_loss)
    
    #创建会话,并通过上下文管理器方式对张量Tensor对象进行计算
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_variational_steps):
            _, elbo_ = sess.run((train_vi, elbo_loss))
            
        if i % 20 == 0:
            print("step {} -ELBO {}".format(i, elbo_))
        #求解后验参数
        q_samples_ = sess.run({k: q.sample(3)
                             for k, q in variational_posteriors.items()})
        
        
        print("打印变分推断参数信息:")
        for param in ts_model.parameters:
            print("{}: {} +- {}".format(param.name,
                      np.mean(q_samples_[param.name], axis=0),
                      np.std(q_samples_[param.name], axis=0)))

    data_t_dist = tfp.sts.forecast(ts_model,observed_time_series=training_data,\
                                   parameter_samples=q_samples_,num_steps_forecast=num_forecast_steps)
    return  data_t_dist


#模型预测
def forecast(training_data):
    data_t_dist=run(training_data)
    with tf.Session() as sess:
        data_t_mean, data_t_scale, data_t_samples = sess.run(
          (data_t_dist.mean()[..., 0],
           data_t_dist.stddev()[..., 0],
           data_t_dist.sample(num_samples)[..., 0]))
        
    return data_t_mean,data_t_scale, data_t_samples


#计算回测
def get_mape(data_t,forecsat):
    true_=data_t[-num_forecast_steps:]
    true_=true_.iloc[:,-1]
    true_=true_.reset_index()
    forecsat=pd.DataFrame(forecsat,columns=['focecast'])
    mape_=pd.concat([pd.DataFrame(true_),forecsat],axis=1)
    mape_['mape']=abs(mape_.iloc[:,-2]-mape_.iloc[:,-1])/mape_.iloc[:,-2]*100
    return mape_



if __name__ == '__main__':
    
    #读取数据集

    data_t=pd.read_csv("../input/ts_sale.csv")
    data_t=data_t[['sale','ym']]
    data_t=data_t.set_index('ym')
    #data_t.to_csv('/input/ts_sale')
    print('序列长度',len(data_t))
    
    #设置超参数
    num_forecast_steps =3 # 最后三个月作为预测值，以便于计算回测mape
    num_samples=100    #设定采样次数
    
    training_data = data_t[:-num_forecast_steps]
    data_dates=np.array(data_t.index,dtype='datetime64[M]')

    observed_time_series=training_data
    data_t_mean,data_t_scale, data_t_samples=forecast(training_data)
    
    data_y=pd.Series(data_t['sale'])
    fig, ax = plot_forecast(
    data_dates, data_y,
    data_t_mean,data_t_scale, data_t_samples,title="forecast")
    ax.axvline(data_dates[-num_forecast_steps], linestyle="--")
    ax.legend(loc="upper left")
    ax.set_ylabel("sale")
    ax.set_xlabel("year_month")
    fig.autofmt_xdate()
    
    mape=get_mape(data_t,data_t_mean)
    print(mape)
    print('mape:',mape['mape'].mean())
