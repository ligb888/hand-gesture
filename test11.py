# 更多疑问，欢迎私信交流。thujiang000
# 导入OpenCV,用于图像处理，图像显示
import cv2
# 导入mediapipe，手势识别
import handProcess

# 导入其他依赖包
import time
import numpy as np
# 鼠标控制包
import pyautogui
from utils import Utils
import autopy


# 识别控制类
class VirtualMouse:
    def __init__(self):

        # image实例，以便另一个类调用
        self.image=None

    # 主函数
    def recognize(self):
        # 调用手势识别类
        handprocess = handProcess.HandProcess(False,1)
        # 初始化基础工具：绘制图像，绘制文本等
        utils = Utils()
        # 计时，用于帧率计算
        fpsTime = time.time()
        # 初始化OpenCV对象，为了获取usb摄像头的图像
        cap = cv2.VideoCapture(0)
        # 视频分辨率
        resize_w = 960
        resize_h = 720

        # 控制边距
        frameMargin = 100

        # 利用pyautogui库获取屏幕尺寸
        screenWidth, screenHeight = pyautogui.size()

        # 柔和处理参数，使鼠标运动平滑
        stepX, stepY = 0, 0
        finalX, finalY = 0, 0
        smoothening = 7
        # 事件触发需要的时间，初始化为0
        action_trigger_time = {
            'single_click':0,
            'double_click':0,
            'right_click':0
        }
        # 用此变量记录鼠标是否处于按下状态
        mouseDown = False
        while cap.isOpened():#只要相机持续打开，则不间断循环
            action_zh = ''
            # 获取视频的一帧图像，返回值两个。第一个为判断视频是否成功获取。第二个为获取的图像，若未成功获取，返回none
            success, self.image = cap.read()
            # 修改图片大小
            self.image = cv2.resize(self.image, (resize_w, resize_h))
            # 视频获取为空的话，进行下一次循环，重新捕捉画面
            if not success:
                print("空帧")
                continue

            # 将图片格式设置为只读状态，可以提高图片格式转化的速度
            self.image.flags.writeable = False
            # 将BGR格式存储的图片转为RGB
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # 镜像，需要根据镜头位置来调整
            self.image = cv2.flip(self.image, 1)
            # 使用mediapipe，将图像输入手指检测模型，得到结果
            self.image = handprocess.processOneHand(self.image)

            # 将手框出来
            cv2.rectangle(self.image, (frameMargin, frameMargin), (resize_w - frameMargin, resize_h - frameMargin),(255, 0, 255), 2)

            # 调用手势识别文件获取手势动作
            self.image,action,key_point = handprocess.checkHandAction(self.image,drawKeyFinger=True)
            # 通过手势识别得到手势动作，将其画在图像上显示
            action_zh = handprocess.action_labels[action]

            if key_point:
                # np.interp为插值函数，简而言之，看key_point[0]的值在(frameMargin, resize_w - frameMargin)中所占比例，
                # 然后去(0, screenWidth)中线性寻找相应的值，作为返回值
                # 利用插值函数，输入手势食指的位置，找到其在框中的百分比，等比例映射到
                x3 = np.interp(key_point[0], (frameMargin, resize_w - frameMargin), (0, screenWidth))
                y3 = np.interp(key_point[1], (frameMargin, resize_h - frameMargin), (0, screenHeight))

                # 柔和处理,通过限制步长，让鼠标移动平滑
                finalX = stepX + (x3 - stepX) / smoothening
                finalY = stepY + (y3 - stepY) / smoothening
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

                # 根据识别得到的鼠标手势，控制鼠标做出相应的动作
                if action_zh == '鼠标移动':
                    try:
                        autopy.mouse.move(finalX, finalY)
                    except:
                        pass
                elif action_zh == '单击准备':
                    pass
                elif action_zh == '触发单击' and (now - action_trigger_time['single_click'] > 0.3):
                    pyautogui.click()
                    action_trigger_time['single_click'] = now


                elif action_zh == '右击准备':
                    pass
                elif action_zh == '触发右击' and (now - action_trigger_time['right_click'] > 2):
                    pyautogui.click(button='right')
                    action_trigger_time['right_click'] = now

                elif action_zh == '向上滑页':
                    pyautogui.scroll(30)
                elif action_zh == '向下滑页':
                    pyautogui.scroll(-30)

                stepX, stepY = finalX, finalY

            self.image.flags.writeable = True
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

            # 显示刷新率FPS
            cTime = time.time()
            fps_text = 1/(cTime-fpsTime)
            fpsTime = cTime

            self.image = utils.cv2AddChineseText(self.image, "帧率: " + str(int(fps_text)),  (10, 30), textColor=(255, 0, 255), textSize=50)

            # 显示画面
            # self.image = cv2.resize(self.image)
            # self.image = cv2.resize(self.image, (resize_w//2, resize_h//2))
            cv2.imshow('virtual mouse', self.image)
            # 等待5毫秒，判断按键是否为Esc，判断窗口是否正常，来控制程序退出
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

# 开始程序
control = VirtualMouse()
control.recognize()