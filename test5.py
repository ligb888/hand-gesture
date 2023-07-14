# 手势拖拽方块

import cv2
import numpy as np

# mediapipe 参数
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 获取摄像头视频流
cap = cv2.VideoCapture(0)
# 获取画面宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#  方块参数
square_x = 100
square_y = 100
square_width = 100

L1 = 0
L2 = 0
on_square = False

while True:
    # 读取视频帧数据
    ret, frame = cap.read()

    # 对图像进行处理
    # 镜像处理
    frame = cv2.flip(frame, 1)

    # mediapip处理
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    # Draw the hand annotations on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 解析results
    # 判断是否出现双手
    if results.multi_hand_landmarks:
        # 解析每一双手
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制21个关键点
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            # print(hand_landmarks)
            x_list = []
            y_list = []
            for landmark in hand_landmarks.landmark:
                # 添加x坐标
                x_list.append(landmark.x)
                # 添加y坐标
                y_list.append(landmark.y)
            # 获取食指指尖x y坐标
            index_finger_x = int(x_list[8] * width)
            index_finger_y = int(y_list[8] * height)
            # 画一个圆验证坐标
            # cv2.circle(frame,(index_finger_x,index_finger_y),30,(255,0,255),-1)

            # 判断食指指尖在不在方块上
            if (index_finger_x > square_x) and (index_finger_x < (square_x + square_width)) and (
                    index_finger_y > square_y) and (index_finger_y < (square_y + square_width)):
                L1 = abs(index_finger_x - square_x)
                L2 = abs(index_finger_y - square_y)
                on_square = True

            if on_square:
                square_x = index_finger_x - L1
                square_y = index_finger_y - L2
    else:
        on_square = False
    # 画一个方块
    cv2.rectangle(frame, (square_x, square_y), (square_x + square_width, square_y + square_width), (255, 0, 0), -1)

    # 显示
    cv2.imshow('virtual drag', frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
