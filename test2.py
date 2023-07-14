import time

import cv2 as cv
import numpy as np
import mediapipe as mp
from numpy import linalg

# 视频设备号
DEVICE_NUM = 0


# 手指检测
# point1-手掌0点位置，point2-手指尖点位置，point3手指根部点位置
def finger_stretch_detect(point1, point2, point3):
    result = 0
    # 计算向量的L2范数
    dist1 = np.linalg.norm((point2 - point1), ord=2)
    dist2 = np.linalg.norm((point3 - point1), ord=2)
    if dist2 > dist1:
        result = 1

    return result


# 检测手势
def detect_hands_gesture(result, landmark):
    if (result[0] == 1) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
        gesture = "good"
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
        gesture = "one"
    elif (result[0] == 0) and (result[1] == 0) and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
        gesture = "please civilization in testing"
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
        gesture = "two"
        # 计算两指的距离
        dist1 = np.linalg.norm((landmark[8] - landmark[6]), ord=2)
        dist2 = np.linalg.norm((landmark[8] - landmark[12]), ord=2)
        if dist1 > dist2:
            gesture = "two2"
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 0):
        gesture = "three"
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
        gesture = "four"
    elif (result[0] == 1) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
        gesture = "five"
    elif (result[0] == 1) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 1):
        gesture = "six"
    elif (result[0] == 0) and (result[1] == 0) and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
        gesture = "OK"
    elif (result[0] == 0) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
        gesture = "stone"
    else:
        gesture = "not in detect range..."

    return gesture


def confirm_gesture(history):
    count_dict = {}
    last = history[-15:]
    for i, item in enumerate(last):
        if len(item) < 2:
            continue
        if item[1] in count_dict.keys():
            count_dict[item[1]] = count_dict[item[1]] + 1
        else:
            count_dict[item[1]] = 1
    for key, item in enumerate(count_dict):
        if count_dict[item] >= 12:
            return item
    return ""


def gesture_moved(history, gesture_mode):
    last = history[-15:]
    if gesture_mode != "one" and gesture_mode != "two2":
        return -1
    

def detect():
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
    # 上一帧的时间戳
    pTime = 0
    # 记录历史手型和点位
    history = []
    # 手势模式
    gesture_mode = ""

    if not cap.isOpened():
        print("Can not open camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can not receive frame (stream end?). Exiting...")
            break

        # 所有手数组
        landmark_all = []
        # 当前手数据
        landmark = np.empty((21, 2))

        # mediaPipe的图像要求是RGB，所以此处需要转换图像的格式
        frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands.process(frame_RGB)
        # 读取视频图像的高和宽
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # print(result.multi_hand_landmarks)
        # 如果检测到手
        if result.multi_hand_landmarks:
            # print(result.multi_hand_world_landmarks)
            # print("=================================")
            # print(result.multi_hand_landmarks)
            # break
            # 为每个手绘制关键点和连接线
            for i, handLms in enumerate(result.multi_hand_landmarks):
                mpDraw.draw_landmarks(frame,
                                      handLms,
                                      mpHands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=handLmsStyle,
                                      connection_drawing_spec=handConStyle)

                for j, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * frame_width)
                    yPos = int(lm.y * frame_height)
                    landmark_ = [xPos, yPos]
                    landmark[j, :] = landmark_

                landmark_all.append(landmark)

            # print(len(landmark_all))
            # if len(landmark_all) >= 2 and landmark_all[0][17][0] < landmark_all[0][4][0] < landmark_all[1][4][0]:
            #     landmark = landmark_all[0]
            # elif len(landmark_all) >= 2 and landmark_all[1][17][0] < landmark_all[1][4][0] < landmark_all[0][4][0]:
            #     landmark = landmark_all[1]
            # elif len(landmark_all) == 1 and landmark_all[0][17][0] < landmark_all[0][4][0]:
            #     landmark = landmark_all[0]
            # else:
            #     landmark = None
            landmark = landmark_all[0]

            if landmark is not None:
                # 通过判断手指尖与手指根部到0位置点的距离判断手指是否伸开(拇指检测到17点的距离)
                for k in range(5):
                    if k == 0:
                        figure_ = finger_stretch_detect(landmark[17], landmark[4 * k + 2], landmark[4 * k + 4])
                    else:
                        figure_ = finger_stretch_detect(landmark[0], landmark[4 * k + 2], landmark[4 * k + 4])

                    figure[k] = figure_
                print(figure, '\n')

                gesture_result = detect_hands_gesture(figure, landmark)
                cv.putText(frame, f"{gesture_result}", (30, 60 * (i + 1)), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 5)
                history.append([landmark, gesture_result])
            else:
                history.append([])
        else:
            history.append([])

        if len(history) > 130:
            history = history[30:]

        # 检测帧数
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 1
        pTime = cTime
        cv.putText(frame, str(int(fps)), (10, 120), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

        # 通过历史记录确定手型，最近的xx帧中一个手型超过一定比例（例如80%）时就将模式切位改手型
        gesture_mode = confirm_gesture(history)

        # 判断移动


        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            print(history)
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    detect()
