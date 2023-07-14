import cv2 as cv
import numpy as np
import mediapipe as mp
from numpy import linalg


def landmarks_to_numpy(results):
    """
    将landmarks格式的数据转换为numpy格式的数据
    numpy shape:(2, 21, 3)
    :param results:
    :return:
    """
    shape = (2, 21, 3)
    landmarks = results.multi_hand_landmarks
    if landmarks is None:
        # 没有检测到手
        return np.zeros(shape)
    elif len(landmarks) == 1:
        # 检测出一只手，先判断是左手还是右手
        label = results.multi_handedness[0].classification[0].label
        hand = landmarks[0]
        # print(label)
        if label == "Left":
            return np.array(
                [np.array([[hand.landmark[i].x, hand.landmark[i].y, hand.landmark[i].z] for i in range(21)]),
                 np.zeros((21, 3))])
        else:
            return np.array([np.zeros((21, 3)),
                             np.array(
                                 [[hand.landmark[i].x, hand.landmark[i].y, hand.landmark[i].z] for i in range(21)])])
    elif len(landmarks) == 2:
        # print(results.multi_handedness)
        lh_idx = 0
        rh_idx = 0
        for idx, hand_type in enumerate(results.multi_handedness):
            label = hand_type.classification[0].label
            if label == 'Left':
                lh_idx = idx
            if label == 'Right':
                rh_idx = idx

        lh = np.array(
            [[landmarks[lh_idx].landmark[i].x, landmarks[lh_idx].landmark[i].y, landmarks[lh_idx].landmark[i].z] for i
             in range(21)])
        rh = np.array(
            [[landmarks[rh_idx].landmark[i].x, landmarks[rh_idx].landmark[i].y, landmarks[rh_idx].landmark[i].z] for i
             in range(21)])
        return np.array([lh, rh])
    else:
        return np.zeros((2, 21, 3))


def relative_coordinate(arr, point):
    """
    转化为相对坐标
    :param arr: numpy数组
    :param point: 手掌根部坐标点
    :return:
    """
    return arr - point


def standardization(hand_arr):
    """
    均值方差归一化
    :param hand_arr:numpy数组
    :return:
    """
    return (hand_arr - np.mean(hand_arr)) / np.std(hand_arr)


def process_mark_data(hand_arr):
    """
    处理手部坐标点信息数组
    将所有点处理为相对于手掌根部点的相对位置
    :param hand_arr: 手部numpy数组
    :return:
    """
    lh_root = hand_arr[0, 0]
    rh_root = hand_arr[1, 0]
    lh_marks = relative_coordinate(hand_arr[0, 1:], lh_root)
    rh_marks = relative_coordinate(hand_arr[1, 1:], lh_root)
    if lh_marks.all() != 0:
        lh_marks = standardization(lh_marks)
    if rh_marks.all() != 0:
        rh_marks = standardization(rh_marks)
    return np.array([lh_marks, rh_marks])

if __name__ == '__main__':
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
        data = landmarks_to_numpy(results)
        data = process_mark_data(data)
        print(data)