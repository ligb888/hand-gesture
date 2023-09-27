import math
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
        rtsp = config['common']['rtsp']
        cap_width = config['common']['cap_width']
        cap_height = config['common']['cap_height']
        cap_fps = int(config['common']['cap_fps'])
        hand = config['common']['hand']
        smooth = int(config['common']['smooth'])
        show = config['common']['show'] == "1"
        crop1_arr = config['common']['crop1'].split(",")
        crop1 = (int(crop1_arr[0]), int(crop1_arr[1]))
        crop2_arr = config['common']['crop2'].split(",")
        crop2 = (int(crop2_arr[0]), int(crop2_arr[1]))
        if not math.isclose(cap_width / cap_height, 1.68, abs_tol=0.1):
            messagebox.showerror("错误", "摄像头分辨率长宽比必须为16:9或16:10")
            exit()
        if cap_fps < 24:
            messagebox.showerror("错误", "摄像头帧率最低为24")
            exit()
        if crop2 != (0, 0) and not math.isclose((crop2[0] - crop1[0]) / (crop2[1] - crop1[1]), 1.68, abs_tol=0.1):
            messagebox.showerror("错误", "截取范围长宽比必须为16:9或16:10")
            exit()
        elif crop2 != (0, 0) and crop2[0] - crop1[0] < 500:
            messagebox.showerror("错误", "截取范围分辨率太低")
            exit()

        # 将摄像图片转换为指定分辨率，在一个范围内映射屏幕坐标
        w, h = 1280, 720
        pt1, pt2 = (100, 100), (1180, 620)

        logging.info("读取配置完成")
        control = VirtualMouse(index, rtsp, hand, show)
        control.recognize(cap_width, cap_height, cap_fps, crop1, crop2, w, h, pt1, pt2, smooth)
    except:
        logging.info("读取配置出错：" + traceback.format_exc())
        exit()
