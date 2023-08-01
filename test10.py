import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector  # 手部检测方法
import time
import autopy

# （1）导数视频数据
wScr, hScr = autopy.screen.size()  # 返回电脑屏幕的宽和高(1920.0, 1080.0)
wCam, hCam = 1280, 720  # 视频显示窗口的宽和高
pt1, pt2 = (100, 100), (1100, 500)  # 虚拟鼠标的移动范围，左上坐标pt1，右下坐标pt2

cap = cv2.VideoCapture(0)  # 0代表自己电脑的摄像头
cap.set(3, wCam)  # 设置显示框的宽度1280
cap.set(4, hCam)  # 设置显示框的高度720

pTime = 0  # 设置第一帧开始处理的起始时间

pLocx, pLocy = 0, 0  # 上一帧时的鼠标所在位置

smooth = 4  # 自定义平滑系数，让鼠标移动平缓一些

# （2）接收手部检测方法
detector = HandDetector(mode=False,  # 视频流图像
                        maxHands=1,  # 最多检测一只手
                        detectionCon=0.8,  # 最小检测置信度
                        minTrackCon=0.5)  # 最小跟踪置信度

# （3）处理每一帧图像
while True:

    # 图片是否成功接收、img帧图像
    success, img = cap.read()

    # 翻转图像，使自身和摄像头中的自己呈镜像关系
    img = cv2.flip(img, flipCode=1)  # 1代表水平翻转，0代表竖直翻转

    # 在图像窗口上创建一个矩形框，在该区域内移动鼠标
    cv2.rectangle(img, pt1, pt2, (0, 255, 255), 5)

    # （4）手部关键点检测
    # 传入每帧图像, 返回手部关键点的坐标信息(字典)，绘制关键点后的图像
    hands, img = detector.findHands(img, flipType=False)  # 上面反转过了，这里就不用再翻转了
    # print(hands)

    # 如果能检测到手那么就进行下一步
    if hands:

        # 获取手部信息hands中的21个关键点信息
        lmList = hands[0]['lmList']  # hands是由N个字典组成的列表，字典包每只手的关键点信息

        # 获取食指指尖坐标，和中指指尖坐标
        x1, y1, z1 = lmList[8]  # 食指尖的关键点索引号为8
        x2, y2, z1 = lmList[12]  # 中指指尖索引12

        # （5）检查哪个手指是朝上的
        fingers = detector.fingersUp(hands[0])  # 传入
        # print(fingers) 返回 [0,1,1,0,0] 代表 只有食指和中指竖起

        # 如果食指竖起且中指弯下，就认为是移动鼠标
        if fingers[1] == 1 and fingers[2] == 0:
            # 开始移动时，在食指指尖画一个圆圈，看得更清晰一些
            cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)  # 颜色填充整个圆

            # （6）确定鼠标移动的范围
            # 将食指的移动范围从预制的窗口范围，映射到电脑屏幕范围
            x3 = np.interp(x1, (pt1[0], pt2[0]), (0, wScr))
            y3 = np.interp(y1, (pt1[1], pt2[1]), (0, hScr))

            # （7）平滑，使手指在移动鼠标时，鼠标箭头不会一直晃动
            cLocx = pLocx + (x3 - pLocx) / smooth  # 当前的鼠标所在位置坐标
            cLocy = pLocy + (y3 - pLocy) / smooth

            # （8）移动鼠标
            autopy.mouse.move(cLocx, cLocy)  # 给出鼠标移动位置坐标

            # 更新前一帧的鼠标所在位置坐标，将当前帧鼠标所在位置，变成下一帧的鼠标前一帧所在位置
            pLocx, pLocy = cLocx, cLocy

        # （9）如果食指和中指都竖起，指尖距离小于某个值认为是点击鼠标
        if fingers[1] == 1 and fingers[2] == 1:  # 食指和中指都竖起

            # 计算食指尖和中指尖之间的距离distance,绘制好了的图像img,指尖连线的信息info
            distance, info, img = detector.findDistance((x1, y1), (x2, y2), img)
            # print(distance)

            # 当指间距离小于50（像素距离）就认为是点击鼠标
            if distance < 50:
                # 在食指尖画个绿色的圆，表示点击鼠标
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)

                # 点击鼠标
                autopy.mouse.click()

    # （10）显示图像
    # 查看FPS
    cTime = time.time()  # 处理完一帧图像的时间
    fps = 1 / (cTime - pTime)
    pTime = cTime  # 重置起始时间

    # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 显示图像，输入窗口名及图像数据
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:  # 每帧滞留20毫秒后消失，ESC键退出
        break

# 释放视频资源
cap.release()
cv2.destroyAllWindows()