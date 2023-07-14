import math
import traceback

import cv2 as cv
import numpy as np
import mediapipe as mp
from numpy import linalg


def distance(lm1, lm2):
    x = lm1.x - lm2.x
    y = lm1.y - lm2.y
    z = lm1.z - lm2.z
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


def lm_distance(landmark):
    lm_data = []
    lm0 = landmark[0]
    for j, lm in enumerate(landmark):
        lm_data.append(distance(lm, lm0))
    return lm_data


def gesture(landmark, distance_data):
    mode = ""
    dis1 = distance(landmark[4], landmark[17])-0.03
    dis2 = distance(landmark[2], landmark[17])
    dis3 = distance(landmark[12], landmark[8])
    if dis1 < dis2 and distance_data[8] > distance_data[6] and \
        distance_data[12] < distance_data[10] and distance_data[16] < distance_data[14] and \
        distance_data[20] < distance_data[18]:
        mode = "one"
    elif dis1 < dis2 and distance_data[8] > distance_data[6] and \
        distance_data[12] > distance_data[10] and distance_data[16] < distance_data[14] and \
        distance_data[20] < distance_data[18] and dis3 < 0.025:
        mode = "two"
    return mode


def confirm_gesture(history):
    count_dict = {}
    last = history[-15:]
    for i, item in enumerate(last):
        if len(item) == 0:
            continue
        if item[0] in count_dict.keys():
            count_dict[item[0]] = count_dict[item[0]] + 1
        else:
            count_dict[item[0]] = 1
    for key, item in enumerate(count_dict):
        if count_dict[item] >= 10:
            return item
    return ""


def moved(history):
    last = history[-15:]



if __name__ == '__main__':
    # 历史手势
    history = []
    # 视频设备号
    DEVICE_NUM = 0
    # 接入USB摄像头时，注意修改cap设备的编号
    cap = cv.VideoCapture(DEVICE_NUM)
    # 加载手部检测函数
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    # 加载绘制函数，并设置手部关键点和连接线的形状、颜色
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=int(5))
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(10))
    figure = np.zeros(5)
    if not cap.isOpened():
        print("Can not open camera.")
        exit()

    while True:
        try:
            record = []
            ret, frame = cap.read()
            if not ret:
                print("Can not receive frame (stream end?). Exiting...")
                break
            frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(frame_RGB)
            # 读取视频图像的高和宽
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            if results.multi_hand_world_landmarks:
                curr_i = -1
                for i, item in enumerate(results.multi_handedness):
                    if item.classification[0].label == "Right":
                        curr_i = i
                        break
                if curr_i == -1:
                    continue
                landmark = results.multi_hand_world_landmarks[curr_i].landmark
                landmark2 = results.multi_hand_landmarks[curr_i].landmark
                distance_data = lm_distance(landmark)
                mode = gesture(landmark, distance_data)
                record = [mode, landmark[4], landmark2[4]]
        except:
            print("获取数据出错" + traceback.format_exc())
        finally:
            history.append(record)

        # 确认当前手势
        mode = confirm_gesture(history)
        if mode != "":
            print(mode)
        # 判断手势移动
        move_data = moved(history)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

