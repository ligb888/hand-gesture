import time
import traceback
import utils
from virtualMouse import VirtualMouse
import configparser
import logging

if __name__ == '__main__':
    # 初始化日志
    utils.get_logger(console=True)

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
        pt1_arr = config['common']['pt1'].split(",")
        pt1 = (int(pt1_arr[0]), int(pt1_arr[1]))
        pt2_arr = config['common']['pt2'].split(",")
        pt2 = (int(pt2_arr[0]), int(pt2_arr[1]))

        control = VirtualMouse(index, hand, show)
        control.recognize(crop1, crop2, pt1, pt2, smooth)
    except:
        logging.info("读取配置出错：" + traceback.format_exc())
        exit()
