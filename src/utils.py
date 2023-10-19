from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import re


def get_logger(name="main", level=logging.INFO, console=False, path="./logs"):
    logger = logging.getLogger()
    if len(logger.handlers) > 0:
        return logger

    if not os.path.exists(path):
        os.makedirs(path)

    formatter = logging.Formatter("%(asctime)s-%(process)d-%(processName)s-%(levelname)s-%(filename)s[:%(lineno)d]-%(message)s")

    handler = TimedRotatingFileHandler(
        filename=path + '/' + name + '.log', encoding="utf-8",
        when="MIDNIGHT", interval=1, backupCount=10
    )
    handler.suffix = "%Y%m%d.log"
    handler.extMatch = re.compile(r"^" + name + "_\d{4}\d{2}\d{2}.log$")
    handler.setLevel(level)
    handler.setFormatter(formatter)

    system_handler = logging.StreamHandler()
    system_handler.setLevel(level)
    system_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)
    if console:
        logger.addHandler(system_handler)

    # logging.info("logger initialization")
    return logger


class Utils:
    def __init__(self):
        pass

    # 添加中文
    def cv2AddChineseText(self,img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "./fonts/simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)