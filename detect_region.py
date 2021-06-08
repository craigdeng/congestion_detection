'''
Author: your name
Date: 2021-05-26 16:35:50
LastEditTime: 2021-05-26 19:17:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Detection/detect_region.py
'''
import cv2
import configparser
import ast
import numpy as np

class DetectRegions():

    def __init__(self, config_name):
        """构造函数
        Args:
            config_name (string):配置文件路径
        """
        self.config = configparser.RawConfigParser()
        self.config.read(config_name)
        self.polygon_list, self.line_list = self.ReadConfig()
        
    def ReadConfig(self):
        """读取配置文件数据存入polygon_list和line_list
        """
        polygon_list = []
        line_list = []
        if self.config.has_section("polygon_list") != 0:
            polygon_num = self.config.getint('polygon_list', 'polygon_num')
            if polygon_num>0:
                for i in range(polygon_num):
                    option = 'polygon'+str(i)
                    polygon = self.config.get('polygon_list', option)
                    polygon = ast.literal_eval(polygon)
                    polygon_list.append(np.array(polygon))
        if self.config.has_section("line_list") != 0:
            line_num = self.config.getint('line_list', 'line_num')
            if line_num>0:             
                for i in range(line_num):
                    option = 'line'+str(i)
                    line = self.config.get('line_list', option)
                    line = ast.literal_eval(line)
                    line_list.append(np.array(line))
        return polygon_list, line_list

    def JudgePointInRegion(self, point, region):
        """判断某一点与某一配置区域的距离，若返回值为正，表示点在多边形内部，返回值为负，表示在多边形外部，返回值为0，表示在多边形上
        Args:
            point (tuple): 需要判定的点(x, y)
            region (ndarray): 区域，包含区域多边形每个点的坐标
        """
        distance = cv2.pointPolygonTest(region, point, True)
        return distance
        

if __name__ == '__main__':

    detect_regions=DetectRegions('/home/adminroot/Desktop/Detection/temp.conf')
    detect_regions.JudgePointInRegion((1600, 200),detect_regions.polygon_list[0])
