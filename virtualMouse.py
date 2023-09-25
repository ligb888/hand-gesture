import math
import copy
import cv2
import handProcess
import time
import numpy as np
import pyautogui
from utils import Utils
import autopy
import logging
from tkinter import messagebox


# 识别控制类
class VirtualMouse:
    def __init__(self, index=0, rtsp="", hand="Right", show=False):
        # image实例，以便另一个类调用
        self.image = None
        self.index = index
        self.rtsp = rtsp
        self.hand = hand
        self.show = show

    # 主函数
    def recognize(self, crop1, crop2, w, h, pt1, pt2, smooth):
        # 调用手势识别类
        handprocess = handProcess.HandProcess(False, 1)
        # 初始化基础工具：绘制图像，绘制文本等
        utils = Utils()
        # 计时，用于帧率计算
        fpsTime = time.time()
        # 初始化OpenCV对象，为了获取usb摄像头的图像
        if self.index == -1:
            cap = cv2.VideoCapture(self.rtsp)
        else:
            cap = cv2.VideoCapture(self.index)
        # 返回电脑屏幕的宽和高(1920.0, 1080.0)
        # wScr, hScr = autopy.screen.size()
        wScr, hScr = pyautogui.size()
        # 获取视频宽度、高度
        wCam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hCam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(rf"屏幕分辨率：{wScr},{hScr}")
        logging.info(rf"摄像头分辨率：{wCam},{hCam}")
        if wCam < 500:
            messagebox.showerror("错误", "摄像头分辨率太低")
            exit()
        if crop2 == (0, 0) and not math.isclose(wCam/hCam, 1.68, abs_tol=0.1):
            messagebox.showerror("错误", "摄像头的长宽比不是16:9或16/10，请截取指定范围")
            exit()
        # 柔和处理参数，使鼠标运动平滑
        stepX, stepY = 0, 0
        # 事件触发需要的时间，初始化为0
        action_trigger_time = {
            'single_click': 0,
            'double_click': 0,
            'right_click': 0,
            'enter': 0
        }
        # 用此变量记录鼠标是否处于按下状态
        mouseDown = False
        action_last = ""

        logging.info("初始化完成，开始进行图像处理")
        while cap.isOpened():#只要相机持续打开，则不间断循环
            try:
                # 等待5毫秒，判断按键是否为Esc，判断窗口是否正常，来控制程序退出
                if cv2.waitKey(10) & 0xFF == 27:
                    break
                # 获取视频的一帧图像，返回值两个。第一个为判断视频是否成功获取。第二个为获取的图像，若未成功获取，返回none
                success, img = cap.read()
                # 视频获取为空的话，进行下一次循环，重新捕捉画面
                if not success:
                    logging.info("空帧")
                    continue
                # 镜像，需要根据镜头位置来调整
                img = cv2.flip(img, 1)
                # 深度拷贝
                self.image = copy.deepcopy(img)
                # 将图片格式设置为只读状态，可以提高图片格式转化的速度
                self.image.flags.writeable = False
                # 截图部分区域，进行手势识别
                if crop2 != (0, 0):
                    self.image = self.image[crop1[1]:crop2[1], crop1[0]:crop2[0]]
                self.image = cv2.resize(self.image, (w, h))
                # 将BGR格式存储的图片转为RGB
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                # 使用mediapipe，将图像输入手指检测模型，得到结果
                self.image = handprocess.processOneHand(self.image, self.hand, self.show)

                # 调用手势识别文件获取手势动作
                self.image, action, key_point = handprocess.checkHandAction(self.image, drawKeyFinger=self.show)
                # 通过手势识别得到手势动作，将其画在图像上显示
                action_zh = handprocess.action_labels[action]

                if key_point:
                    x3 = np.interp(key_point[0], (pt1[0], pt2[0]), (0, wScr))
                    y3 = np.interp(key_point[1], (pt1[1], pt2[1]), (0, hScr))

                    # 柔和处理,通过限制步长，让鼠标移动平滑
                    finalX = stepX + (x3 - stepX) / smooth
                    finalY = stepY + (y3 - stepY) / smooth
                    # 计时停止，用于计算帧率
                    now = time.time()
                    # 判断鼠标为点击状态
                    if action_zh == '鼠标拖拽':
                        # 如果鼠标上一个状态不是点击，则判断为按下鼠标
                        if not mouseDown:
                            pyautogui.mouseDown(button='left')
                            mouseDown = True
                        autopy.mouse.move(finalX, finalY)
                    else:
                        # 如果鼠标上一个状态是点击，则判断为放开鼠标左键
                        if mouseDown:
                            pyautogui.mouseUp(button='left')
                            mouseDown = False
                    # 回车
                    if action_zh == '回车' and action_last != action_zh \
                            and (now - action_trigger_time['enter'] > 1):
                        pyautogui.press('enter')
                        action_trigger_time['enter'] = now
                    # 根据识别得到的鼠标手势，控制鼠标做出相应的动作
                    if action_zh == '鼠标移动':
                        autopy.mouse.move(finalX, finalY)
                    elif action_zh == '单击准备':
                        pass
                    elif action_zh == '触发单击' and action_last != action_zh:
                        pyautogui.click()
                    elif action_zh == '右击准备':
                        pass
                    elif action_zh == '触发右击' and action_last != action_zh \
                            and (now - action_trigger_time['right_click'] > 1):
                        pyautogui.click(button='right')
                        action_trigger_time['right_click'] = now
                    elif action_zh == '向上滑页':
                        pyautogui.scroll(30)
                    elif action_zh == '向下滑页':
                        pyautogui.scroll(-30)

                    stepX, stepY = finalX, finalY

                action_last = action_zh
                # logging.info(action_zh)

                # 显示画面
                if self.show:
                    # 将手框出来
                    cv2.rectangle(self.image, pt1, pt2, (0, 255, 255), 5)
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                    # 显示刷新率FPS
                    cTime = time.time()
                    fps_text = 1 / (cTime - fpsTime)
                    fpsTime = cTime
                    self.image = utils.cv2AddChineseText(self.image, "帧率: " + str(int(fps_text)), (10, 30),
                                                         textColor=(255, 0, 255), textSize=50)
                    cv2.imshow('virtual mouse', self.image)
            except:
                pass

        cap.release()
