import math
import os
import cv2

from utils.video_tools import get_frames
from utils.yolo.MyYOLO import YOLODetector


class AbnormalParkingDetect():
    def __init__(self, stop_thresh=600):
        """类初始化
        """
        self.c0_list = [[0, 0, 0, 0]]  # 初始化起始点 [x,y,n,x]
        self.stop_thresh = stop_thresh  # 停车判断阈值， 连续stop_thresh帧不动后，判断为异常停车

    def Detect(yolo_res):
        """根据yolo检测结果，判断是否存在异常停车

        Args:
            yolo_res ([list]): yolo检测结果

        Returns:
            flag [boolean]: 是否发生异常停车的标志
            stop_list: 如果发生异常停车，车的中心点坐标列表 其中元素为 [x,y,n,s] x,y表示车中心坐标位置，n 出现次数，s车检测框面积
        """
        stop_list = []
        flag = False
        if len(yolo_res) == 0:
            return
        c1_list = BboxToPoint(yolo_res)
        new_c0_list = CheckPoints(c1_list, self.c0_list)
        # print(new_c0_list)
        for p in new_c0_list:
            if p[2] > self.stop_thresh:
                flag = True
                stop_list.append([p[0], p[1]])
        self.c0_list = new_c0_list
        return flag, stop_list



def BboxToPoint(yolo_res):
    """将yolo检测结果转成bbox中心点列表

    Args:
        yolo_res ([type]): [description]
    """
    # 所有检测框中心点, 元素为(x,y,n, s) x,y 为中心点坐标, n 为当前位置 bbox出现次数，初始化为1 s: 当前框面积 用于设定移动阈值
    center_points = []
    for res_i in yolo_res:
        xmin, ymin, xmax, ymax, score, label = res_i
        cx, cy = (xmin+xmax)//2, (ymin + ymax)//2
        s = (ymax - ymin) * (xmax - xmin)
        center_points.append([cx, cy, 1, s])
    return center_points


def GetDistance(p1, p2):
    """ 计算两个点欧式距离

    Args:
        p1 (list): x,y 点1
        p2 (list): x,y 点2
    Returns:
        [float]: 距离
    """
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def CheckPoints(c1, c0, img=None, save_path=None, debug_mode=False):
    """匹配前后两组点

    Args:
        c1 (list): 当前帧（t）bbox中心点坐标 [x1,y1,1] 1表示出现一次
        c0 (list]): 上一帧(t-1) bbox中心点坐标 [x2,y2,n] n表示当前位置已经累计出现n次
        img (array) : 调试用 显示两组点和最小距离
        save_path: (str) :调试用 保存检测结果
    Returns:
        [list]: 当前帧车辆位置列表 [x,y,n,s] x,y 表示中心坐标位置，n表示在该位置已经出现的次数，s表示bbox面积
    """
    final_obj = []  # 保存当前帧中所有bbox中心点 及其历史位置信息 [x,y,n] n表示过去帧中坐标为x,y的bbox的个数
    if debug_mode:
        print(save_path)
    for x1, y1, _, s in c1:
        min_d = 9999999
        min_x2, min_y2, min_n = 0, 0, 1
        for x2, y2, n, _ in c0:
            if x2 == 0 and y2 == 0:
                d = 0
            else:
                d = GetDistance((x1, y1), (x2, y2))
            if debug_mode:
                print('x1,y1,x2,y2,d ', x1, y1, x2, y2, d)
            if d < min_d:
                min_d = d
                min_n = n
                # debug
                min_x2, min_y2 = x2, y2
        if debug_mode:
            print(int(x1), int(y1), round(d, 2), min_n, s)
        if s < 10000:
            same_obj_thresh = 4  # 对于bbox较小的车辆，判断是相同车辆的阈值设为4
        else:
            same_obj_thresh = 10  # 对于bbox较大的车辆，判断是相同车辆的阈值设为10 (根据经验设定)
        if min_d < same_obj_thresh:
            final_obj.append([x1, y1, min_n+1, s])
        else:
            final_obj.append([x1, y1, 1, s])
        # debug
        if debug_mode:
            cv2.circle(img, (int(x1), int(y1)), 10,
                       (0, 0, 0), -1)
            cv2.line(img, (int(x1), int(y1)),
                     (int(min_x2), int(min_y2)), (0, 0, 255), thickness=2)
            cv2.imwrite(save_path, img)
    return final_obj


def main(video_path):
    """ 主函数，用于检测异常停车

    Args:
        video_path (str): 测试视频路径
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(BASE_DIR)
    yolocfgfile = os.path.join(BASE_DIR, "utils/yolo/yoloconfig.ini")
    yolo = YOLODetector(yolocfgfile=yolocfgfile)
    print('yolo load ok!')
    c0_list = [[0, 0, 0, 0]]  # 初始化
    stop_thresh = 600
    imgi = 1
    for img in get_frames(video_path):
        stop_list = []  # 所有异常停止的车中心点坐标
        flag = False
        # debug
        if imgi > 1000000:
            break
            imgi += 1
            continue
        yolo_res = yolo.Infer(img, target_label=2)
        if len(yolo_res) == 0:
            imgi += 1
            continue
        else:  # 有车
            c1_list = BboxToPoint(yolo_res)
            # new_c0_list = CheckPoints(c1_list, c0_list)
            new_c0_list = CheckPoints(
                # c1_list, c0_list, img, save_path='./res/6/'+str(imgi)+'.jpg')
                c1_list, c0_list, img, save_path='./res/6_13_5/'+str(imgi)+'.jpg')
            print(new_c0_list)
            for p in new_c0_list:
                if p[2] > stop_thresh:
                    flag = True
                    stop_list.append([p[0], p[1]])
        if flag:
            print('Waring!!! Car Stop!!!')
            # save_path = './res/6_stop/' + str(imgi) + '.jpg'
            save_path = './res/6_13_5_stop/' + str(imgi) + '.jpg'
            for stop_x, stop_y in stop_list:
                print(stop_x, stop_y)
                cv2.circle(img, (int(stop_x), int(stop_y)), 100,
                           (0, 0, 255), thickness=2)
            cv2.imwrite(save_path, img)

        c0_list = new_c0_list
        imgi += 1


# if __name__ == "__main__":
#     # debug
#     # video_path = './data/172_1_6_13_5.mp4'
#     # video_path = './data/172_1_8_193_2.mp4'
#     # video_path = './data/6.mp4'
#     # main(video_path)

#     # use
#     # detector = AbnormalParkingDetect(stop_thresh=600)
#     # yolo_res = Yolo(input_img)
#     # flag, stop_list = detector.Detect(yolo_res)