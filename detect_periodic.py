'''
Author: Puyang Deng
Date: 2021-05-20 15:39:44
LastEditTime: 2021-06-04 09:07:51
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Detection\test.py
'''
import math
import os
from time import time
import cv2
import pandas as pd
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import scipy.signal as signal
from utils.video_tools import get_frames
from utils.yolo.MyYOLO import YOLODetector
from detect_region import DetectRegions
from statsmodels.tsa.seasonal import seasonal_decompose

class TrafficFlowReader():
    """
    视频交通流读取器
    """

    def __init__(self, yolocfgfile='./utils/yolo/yoloconfig.ini', 
    regioncfgfile = '/home/adminroot/Desktop/Detection/confs/temp.conf', save_path = './timeseries/'):
        """
        Initialize Class attributes
        Args:
            yolo配置文件
            区域配置文件
            存储地址
        """
        self.yolocfgfile = yolocfgfile # yolo配置文件
        self.regioncfgfile = regioncfgfile # 区域配置文件
        self.save_path = save_path #时间序列存储路径


    def getTrafficFlow(self, video):
        """
        给定视频计算指定区域内的车流统计时间序列

        Args:
                video (str): 要计算的视频位置 
        Returns:
                detection [list]: 视频指定区域内的车流时间序列
        """
        yolo = YOLODetector(self.yolocfgfile)
        detection_regions  = DetectRegions(self.regioncfgfile)
        detection = []

        for i, img in enumerate(get_frames(video)):
            res = yolo.Infer(img, img_name=str(i)+"res.jpg")
            count = 0 
            for obj in res:
                if obj[5] == 1 or 2:# count if id is 1(car) or 2(bus)
                    if detection_regions.JudgePointInRegion(((obj[0]+obj[2])/2, (obj[1]+obj[3])/2),detection_regions.polygon_list[0]) > 0:
                        #检测是否在区域内
                        count += 1

            detection.append(count)
        return detection

    def ts2csv(self, ts, save_name):
        """
        存储时间序列（list）到csv
        """
        name_attribute = ['count']
        writerCSV=pd.DataFrame(columns=name_attribute,data=ts)
        writerCSV.to_csv(self.save_path+save_name, encoding='utf-8', index=False)



class CongestionDetector():

    def __init__(self, csv_path, fps) -> None:
        """
        Initialize Class attributes
        Args:
            csv_path (str): 要计算的时间序列位置 
            fps（int）： 该时序视屏帧数

        """

        self.fps = fps#视频帧数
        self.ts = self.csv2ts(csv_path)
        self.ts_resample, self.ts_smooth = self.resample(self.ts, fps)
        self.x, self.y = self.fftTransfer(self.ts_smooth)
        #根据y值（频域增幅）取周期
        arr = np.vstack((self.x, self.y)).T
        sortedArr = arr[arr[:,1].argsort()[::-1]]
        print('period,frespike',sortedArr)
        # print('zhouqichangdu',len(self.ts_smooth)/3)
        p = []
        for period in sortedArr:
            if period[0] < len(self.ts_smooth)/3:#若增幅周期没有超过时长1/3
                p.append(period[0])

        self.cycle = p[0]#增幅最大周期
        self.limit = max(self.ts)*0.8#车流上限

    def csv2ts(self,csv,col='count'):
        """
        读取csv的一个colum到np.array
        """
        df = pd.read_csv(csv)
        arr = df[[col]].to_numpy()

        return arr

    def normalize(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """

        x = np.asarray(x)
        return (x - x.min()) / (np.ptp(x))

    def resample(self, ts, n):
        """
        平滑曲线（均值）
        """
        if len(ts)%n==0:
            temp = ts.reshape(int(len(ts)/n),n)
        else:
            temp = ts[0:-(len(ts)%n)].reshape(int(len(ts[0:-(len(ts)%n)])/n),n)
        t = []
        for i in temp:
            a = np.average(i)
            t.append(a)

        tmp_smooth = signal.savgol_filter(t,n*2+5,3)

        return t, tmp_smooth
    
    def data_search(self, data, level):
        """[summary]

        Args:
            data ([type]): [description]
            level ([type]): [description]

        Returns:
            [type]: [description]
        """

        list = []
        temp = []
        for i in range(len(data)):
            if data[i] > level:
                temp.append(data[i])
            else:
                list.append(temp)
                temp = []
        return [i for i in list if i]

    def fftTransfer(self, timeseries, n=5, fmin=0.01):

        yf = abs(fft(timeseries))  # 取绝对值
        yfnormlize = self.normalize(yf)  # 归一化处理
        yfhalf = yfnormlize[range(int(len(timeseries) / 2))]  # 由于对称性，只取一半区间

        xf = np.arange(len(timeseries))  # 频率
        xhalf = xf[range(int(len(timeseries) / 2))]  # 取一半区间


        fwbest = yfhalf[signal.argrelextrema(yfhalf, np.greater)]
        xwbest = signal.argrelextrema(yfhalf, np.greater)

        xorder = np.argsort(-fwbest)  # 对获取到的极值进行降序排序，也就是频率越接近，越排前
        xworder = list()
        xworder.append(xwbest[x] for x in xorder)  # 返回频率从大到小的极值顺序
        fworder = list()
        fworder.append(fwbest[x] for x in xorder)  # 返回幅度

        if len(fwbest) <= n:
            fwbest = fwbest[fwbest >= fmin].copy()
            return len(timeseries)/xwbest[0][:len(fwbest)], fwbest    #转化为周期输出
        else:
            fwbest = fwbest[fwbest >= fmin].copy()
            return len(timeseries)/xwbest[0][:len(fwbest)], fwbest  # 只返回前n个数   #转化为周期输出

    def decompose(self):
        """[summary]

        Returns:
            [type]: [description]
        """

        timeseries = self.ts_smooth
        decomposition = seasonal_decompose(timeseries, model='additive', freq =self.cycle)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(411)
        ax1.plot(timeseries, label='Original')
        ax1.legend(loc='best')
        ax2 = fig.add_subplot(412)
        ax2.plot(trend, label='Trend')
        ax2.legend(loc='best')
        ax3 = fig.add_subplot(413)
        ax3.plot(seasonal, label='Seasonality')
        ax3.legend(loc='best')
        ax4 = fig.add_subplot(414)
        ax4.plot(residual, label='Residuals')
        ax4.legend(loc='best')
        fig.tight_layout()
        plt.show(block=False)
    
        return trend, seasonal, residual


    def visualize(self):
        
        plt.figure(figsize=(15,15))
        plt.subplot(411)
        x = np.arange(len(self.ts)) # x轴
        plt.plot(x, self.ts)
        plt.title('Original wave')

        plt.subplot(412)
        x = np.arange(len(self.ts_resample)) # x轴
        plt.plot(x, self.ts_resample)
        plt.title('Resample wave')

        plt.subplot(413)
        x = np.arange(len(self.ts_smooth)) # x轴
        plt.plot(x, self.ts_smooth)
        plt.title('Smooth wave')

        plt.subplot(414)
        yf = abs(fft(self.ts_smooth))  # 取绝对值
        yfnormlize = self.normalize(yf)  # 归一化处理
        yfhalf = yfnormlize[range(int(len(self.ts_smooth) / 2))]  # 由于对称性，只取一半区间

        xf = np.arange(len(self.ts_smooth))  # 频率
        xhalf = xf[range(int(len(self.ts_smooth) / 2))]  # 取一半区间
        plt.plot(xhalf, yfhalf, 'r')
        plt.title('FFT of Mixed wave(half side frequency range)', fontsize=10, color='#7A378B')

        fwbest = yfhalf[signal.argrelextrema(yfhalf, np.greater)]
        xwbest = signal.argrelextrema(yfhalf, np.greater)
        plt.plot(xwbest[0][:5], fwbest[:5], 'o', c='yellow')
        plt.show(block=False)
        plt.show()

    def main(self, csv_path):
        """
        判断函数
        """

        # #根据y值（频域增幅）取周期
        # arr = np.vstack((self.x, self.y)).T
        # sortedArr = arr[arr[:,1].argsort()[::-1]]
        # print('period,frespike',sortedArr)
        # # print('zhouqichangdu',len(self.ts_smooth)/3)
        # p = []
        # for period in sortedArr:
        #     if period[0] < len(self.ts_smooth)/3:#若增幅周期没有超过时长1/3
        #         p.append(period[0])

        # cycle = p[0]#增幅最大周期
        # limit = max(self.ts)*0.8#车流上限

        ts = self.csv2ts(csv_path)
        ts_resample, ts_smooth = self.resample(ts, self.fps)

        over = self.data_search(ts_resample, self.limit)#超限列

        t = []
        for period in over:
            if len(period) > self.cycle:
                t.append(period)

        if len(t) != 0:
            print("该路段拥堵")
            return t
        else:
            print("该路段畅通")





# def csv2ts(csv,col='count'):
#     """
#      读取csv的一个colum到np.array
#     """
#     df = pd.read_csv(csv)
#     arr = df[[col]].to_numpy()

#     return arr

   
# def resample(ts, n):
#     """
#     平滑曲线（均值）
#     """
#     if len(ts)%n==0:
#         temp = ts.reshape(int(len(ts)/n),n)
#     else:
#         temp = ts[0:-(len(ts)%n)].reshape(int(len(ts[0:-(len(ts)%n)])/n),n)
#     t = []
#     for i in temp:
#         a = np.average(i)
#         t.append(a)

#     tmp_smooth = signal.savgol_filter(t,n*2+5,3)

#     return t, tmp_smooth


# def normalize(x):

#     x = np.asarray(x)
#     return (x - x.min()) / (np.ptp(x))


# def fftTransfer(timeseries, n=5, fmin=0.2):

#     yf = abs(fft(timeseries))  # 取绝对值
#     yfnormlize = normalize(yf) # 归一化处理
#     yfhalf = yfnormlize[range(int(len(timeseries) / 2))]  # 由于对称性，只取一半区间

#     xf = np.arange(len(timeseries))  # 频率
#     xhalf = xf[range(int(len(timeseries) / 2))]  # 取一半区间

#     plt.figure(figsize=(15,10))
#     plt.subplot(211)
#     x = np.arange(len(timeseries))  # x轴
#     plt.plot(x, timeseries)
#     plt.title('Original wave')
#     plt.subplot(212)
#     plt.plot(xhalf, yfhalf, 'r')
#     plt.title('FFT of Mixed wave(half side frequency range)', fontsize=10, color='#7A378B')

#     fwbest = yfhalf[signal.argrelextrema(yfhalf, np.greater)]
#     xwbest = signal.argrelextrema(yfhalf, np.greater)
#     plt.plot(xwbest[0][:n], fwbest[:n], 'o', c='yellow')
#     plt.show(block=False)
#     plt.show()

#     xorder = np.argsort(-fwbest)  # 对获取到的极值进行降序排序，也就是频率越接近，越排前
#     # print('xorder = ', xorder)
#     # print(type(xorder))
#     xworder = list()
#     xworder.append(xwbest[x] for x in xorder)  # 返回频率从大到小的极值顺序
#     fworder = list()
#     fworder.append(fwbest[x] for x in xorder)  # 返回幅度

#     if len(fwbest) <= n:
#         fwbest = fwbest[fwbest >= fmin].copy()
#         return len(timeseries)/xwbest[0][:len(fwbest)], fwbest    #转化为周期输出
#     else:
#         fwbest = fwbest[fwbest >= fmin].copy()
#         # print(len(fwbest))
#         # print(xwbest)
#         return len(timeseries)/xwbest[0][:len(fwbest)], fwbest  # 只返回前n个数   #转化为周期输出

# def decompose(timeseries, frequence):

#     decomposition = seasonal_decompose(timeseries, model='additive', freq =frequence)
#     trend = decomposition.trend
#     seasonal = decomposition.seasonal
#     residual = decomposition.resid

#     fig = plt.figure(figsize=(15,5))
#     ax1 = fig.add_subplot(411)
#     ax1.plot(timeseries, label='Original')
#     ax1.legend(loc='best')
#     ax2 = fig.add_subplot(412)
#     ax2.plot(trend, label='Trend')
#     ax2.legend(loc='best')
#     ax3 = fig.add_subplot(413)
#     ax3.plot(seasonal, label='Seasonality')
#     ax3.legend(loc='best')
#     ax4 = fig.add_subplot(414)
#     ax4.plot(residual, label='Residuals')
#     ax4.legend(loc='best')
#     fig.tight_layout()
#     plt.show(block=False)
    
#     return trend, seasonal, residual

# def resample(ts, n):
#     if len(ts)%n==0:
#         temp = ts.reshape(int(len(ts)/n),n)
#     else:
#         temp = ts[0:-(len(ts)%n)].reshape(int(len(ts[0:-(len(ts)%n)])/n),n)
#     t = []
#     for i in temp:
#         a = np.average(i)
#         t.append(a)
#     return t