import math
import time
import traceback
import utils
from virtualMouse import VirtualMouse
import configparser
import logging
from tkinter import messagebox
import tkinter

if __name__ == '__main__':
    # 初始化日志
    utils.get_logger(console=True)

    # 推出默认的TK窗口
    window = tkinter.Tk()
    window.withdraw()

    # 读取配置文件
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        index = int(config['common']['index'])
        hand = config['common']['hand']
        smooth = int(config['common']['smooth'])
        show = config['common']['show'] == "1"
        crop1_arr = config['common']['crop1'].split(",")
        crop1 = (int(crop1_arr[0]), int(crop1_arr[1]))
        crop2_arr = config['common']['crop2'].split(",")
        crop2 = (int(crop2_arr[0]), int(crop2_arr[1]))
        if crop2 != (0, 0) and not math.isclose((crop2[0] - crop1[0]) / (crop2[1] - crop1[1]), 16/9, abs_tol=0.05):
            messagebox.showerror("错误", "截取范围长宽比必须为16:9")
            exit()
        # pt1_arr = config['common']['pt1'].split(",")
        # pt1 = (int(pt1_arr[0]), int(pt1_arr[1]))
        # pt2_arr = config['common']['pt2'].split(",")
        # pt2 = (int(pt2_arr[0]), int(pt2_arr[1]))

        # 将摄像图片转换为指定分辨率，在一个范围内映射屏幕坐标
        w, h = 1280, 720
        pt1, pt2 = (100, 100), (1180, 620)

        control = VirtualMouse(index, hand, show)
        control.recognize(crop1, crop2, w, h, pt1, pt2, smooth)
    except:
        logging.info("读取配置出错：" + traceback.format_exc())
        exit()
