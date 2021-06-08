import cv2
import numpy as np
import configparser

WIN_NAME = 'Image'

class MouseAction():
    """基于鼠标操作绘制检测区域
    """
    def __init__(self,image, config_path):
        """初始化
        Args:
            image (mat):待配置区域图片
            config_path(string):配置文件地址
        """
        self.his_image = image.copy()
        self.image_copy = image.copy()
        self.size = image.shape
        self.temp_points = []
        self.all_polygons = []
        self.all_lines = []
        self.draw_type = 1
        self.config_path = config_path
    def ChangeDrawType(self):
        """切换绘制模式，画线或画框
        """
        self.draw_type = not self.draw_type
    
    def PixelToPercent(self, points):
        """将像素点值按照图像大小归一化
        """
        size = self.image_copy.shape
        print(size[1],size[0])  #打印图片宽度与高度
        for i in range(len(points)):
            points[i]=list(points[i])
            points[i][0] = round(points[i][0] / self.size[1], 3)
            points[i][1] = round(points[i][1] / self.size[0], 3)

    def MouseHandler(self,event, x, y, flags, data):
        """鼠标处理函数
        Args:
            x (int):鼠标点选的x坐标
            y (int):鼠标点选的y坐标
            flags ():CV_EVENT_FLAG的组合，未使用
            data ():用户定义的传递到setMouseCallback函数调用的参数，未使用
        """
        if event == cv2.EVENT_LBUTTONDOWN :#左键按下
            print('x = %d, y = %d' % (x, y))
            cv2.circle(self.image_copy, (x, y), 5, (0, 255, 0), 2)
            self.temp_points.append((x, y))  # 用于画点
            for i in range(len(self.temp_points) - 1):
                cv2.line(self.image_copy, self.temp_points[i], self.temp_points[i + 1], (0, 0, 255), 2)

        if event == cv2.EVENT_RBUTTONDOWN :#右键按下保存
            if self.draw_type:
                """保存多边形框"""
                if len(self.temp_points)>2:
                    for i in range(len(self.temp_points) - 1):
                        cv2.line(self.image_copy, self.temp_points[i], self.temp_points[i + 1], (0, 255, 255), 2)
                    cv2.line(self.image_copy, self.temp_points[-1], self.temp_points[0], (0, 255, 255), 2)
                    self.his_image = self.image_copy.copy()
                    # self.PixelToPercent(self.temp_points)
                    self.all_polygons.append(self.temp_points)
                    self.temp_points = []
                else:
                    print("not polygon")
            else:
                """保存折线"""
                for i in range(len(self.temp_points) - 1):
                    cv2.line(self.image_copy, self.temp_points[i], self.temp_points[i + 1], (255, 0, 255), 2)
                self.his_image = self.image_copy.copy()
                # self.PixelToPercent(self.temp_points)
                self.all_lines.append(self.temp_points)
                self.temp_points = []

        if event == cv2.EVENT_MBUTTONDOWN :#中键按下删除上一个动作，直到之前保存的一组结果，无法继续删除
            del(self.temp_points[-1])
            print(self.temp_points)
            self.image_copy = self.his_image.copy()
            for i in range(len(self.temp_points)):
                cv2.circle(self.image_copy, self.temp_points[i], 5, (0, 255, 0), 2)
            for i in range(len(self.temp_points) - 1):
                cv2.line(self.image_copy, self.temp_points[i], self.temp_points[i + 1], (0, 0, 255), 2)

        cv2.imshow(WIN_NAME, self.image_copy)

    def GetPoints(self):
        """显示图片，开始配置区域
        """
        cv2.imshow(WIN_NAME,self.image_copy)
        cv2.setMouseCallback(WIN_NAME, self.MouseHandler)

    def SavePoints(self):
        """将绘制的点存入配置文件
        """
        conf= configparser.ConfigParser()
        conf.read(self.config_path)  # 文件路径
        if len(self.all_polygons):
            if conf.has_section("polygon_list") == 0:
                conf.add_section("polygon_list")
            conf.set("polygon_list", "polygon_num", str(len(self.all_polygons)))
            for i in range(len(self.all_polygons)):
                conf.set("polygon_list", "polygon"+str(i), str(self.all_polygons[i])[1:-1]) 
        if len(self.all_lines):
            if conf.has_section("line_list") == 0:
                conf.add_section("line_list")
            conf.set("line_list", "line_num", str(len(self.all_lines)))
            for i in range(len(self.all_lines)):
                conf.set("line_list", "line"+str(i), str(self.all_lines[i])[1:-1]) 

        conf.write(open(self.config_path, 'w'))


if __name__ == '__main__':

    image = cv2.imread('/home/adminroot/Desktop/Detection/confs/con1.2.png')
    cfg_path = '/home/adminroot/Desktop/Detection/confs/temp.conf'
    size = image.shape
    print(size[1],size[0])  #打印图片宽度与高度

    mouse_action = MouseAction(image, cfg_path)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME,1000, 800)

    mouse_action.GetPoints()
    while(True):
        key = cv2.waitKey(0)
        if key == 9:
            mouse_action.ChangeDrawType()
        elif key == 27:
            break
    cv2.destroyAllWindows()
    mouse_action.SavePoints()


