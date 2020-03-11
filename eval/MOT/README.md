### 0、指标说明

![跟踪用指标说明][1]


### 1、使用流程
装一下这个：
https://github.com/cheind/py-motmetrics

里面有基本的教程。

或者看这里也行：

#### 1.1 数据准备

* ground_truth数据，真实的跟踪数据，需要指出对于每一帧，目标A的坐标框以及目标A的trackID。

* track数据，跟踪数据，需要指出每一帧，对于跟踪算法而言，目标A的坐标框以及目标A的trackID。

####1.2 写入accumulator


1） 利用motmetrics创建accumulator。

2） 对于每一帧，获取：

    1、这一帧中每个被检测物体的跟踪id。
    2、这一帧中每个被检测物体的跟踪框位置。
    3、利用motmetrics中提供的算法计算gt框和track框的距离。
    4、按照格式将1和3的数据写入accumulator
    
####1.3 计算矩阵
利用motmetrics计算MOTA等指标。

####1.4 验证
将gt数据作为track数据输入，得到：

          num_frames  mota  motp
    acc         794   1.0   0.0

认为基本可信。


[1]: https://img-blog.csdn.net/20180308160506213?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveXVocTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70