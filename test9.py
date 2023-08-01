import math
import time
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
    last = history[-20:]
    for i, item in enumerate(last):
        if len(item) == 0:
            continue
        if item[0] in count_dict.keys():
            count_dict[item[0]] = count_dict[item[0]] + 1
        else:
            count_dict[item[0]] = 1
    for key, item in enumerate(count_dict):
        if count_dict[item] >= 15:
            return item
    return ""


def moved(history):
    mode = ""
    move = 0
    last = history[-20:]
    arr = []
    arr_z = []
    arr_x = []
    arr_2 = []
    arr_dis = []
    move_arr = []
    # 稳定帧
    count = 0

    if len(last) == 0 or len(last[0]) == 0:
        return [mode, move]

    for i, item in enumerate(last):
        if len(item) == 0:
            continue
        arr.append(item[3].y)
        arr_z.append(item[3].z)
        arr_x.append(item[3].x)
        arr_2.append(item[2])
        move_arr.append(item[3])
    if len(arr) < 15:
        return [mode, move]
    for i, item in enumerate(arr_2):
        if i == 0:
            arr_dis.append(0)
            continue
        dis = distance(item, arr_2[0])
        arr_dis.append(dis)

    print(arr_dis)
    ma = np.max(arr_dis)
    mi = np.min(arr_dis)
    if ma - mi > 0.01:
        avg = np.mean(arr_dis)
        # 低点数组
        low_arr = []
        # 当前低点
        low = 0
        for i, item in enumerate(arr_dis):
            # 稳定帧计数
            if item < avg:
                count += 1
            else:
                count = 0

            # 点击帧统计
            if item < avg and low != 0 and low - avg > 0.005:
                low_arr.append(low)
                low = 0
            elif item > avg and item > low:
                low = item

        if count > 8 and len(low_arr) == 1:
            mode = "click"
        elif count > 8 and len(low_arr) == 2:
            mode = "double"
    else:
        # 判断是否移动
        dx = move_arr[0].x - move_arr[len(move_arr) - 1].x
        dy = move_arr[0].y - move_arr[len(move_arr) - 1].y
        move = math.sqrt(dx**2 + dy**2)
        if move > 0.05:
            for i, item in enumerate(move_arr):
                if i == 0:
                    continue
                dis = math.sqrt((move_arr[i].x - move_arr[i-1].x)**2 + (move_arr[i].y - move_arr[i-1].y)**2)
                if dis < 0.01:
                    count += 1
                else:
                    count = 0
            if count > 8:
                mode = "move"
    return [mode, move]


def get_data(history):
    record = []
    try:
        if results.multi_hand_world_landmarks:
            curr_i = -1
            for i, item in enumerate(results.multi_handedness):
                if item.classification[0].label == "Right":
                    curr_i = i
                    break
            if curr_i == -1:
                return
            landmark = results.multi_hand_world_landmarks[curr_i].landmark
            landmark2 = results.multi_hand_landmarks[curr_i].landmark
            distance_data = lm_distance(landmark)
            mode = gesture(landmark, distance_data)
            record = [mode, distance_data, landmark[4], landmark2[4]]
            # print(landmark[4].x, landmark[4].y)
    except:
        print("获取数据出错" + traceback.format_exc())
    finally:
        history.append(record)


if __name__ == '__main__':
    pTime = 0
    # 历史手势
    history = []
    skip = 0
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
        ret, frame = cap.read()
        if not ret:
            print("Can not receive frame (stream end?). Exiting...")
            break
        frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_RGB)
        # 读取视频图像的高和宽
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        try:
            if skip > 0:
                skip -= 1
                history.append([])
                continue
            # 获取数据
            get_data(history)
            # 删减历史记录
            if len(history) > 130:
                history = history[30:]
            # 确认当前手势
            mode = confirm_gesture(history)
            # 判断手势移动
            move_data = moved(history)
            if mode != "" and move_data[0] != "":
                skip = 15
                print(mode, move_data)
        except:
            print("计算出错" + traceback.format_exc())
        finally:
            # 检测帧数
            cTime = time.time()
            fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 1
            pTime = cTime
            cv.putText(frame, str(int(fps)), (10, 120), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
    cap.release()
    cv.destroyAllWindows()

