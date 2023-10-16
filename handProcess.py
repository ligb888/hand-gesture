import logging

import cv2
import mediapipe as mp
import math
from utils import Utils


class HandProcess:

    def __init__(self, static_image_mode=False, max_num_hands=2):
        # 参数
        self.mp_drawing = mp.solutions.drawing_utils  # 初始化medialpipe的画图函数
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands  # 初始化手掌检测对象
        # 调用mediapipe的Hands函数，输入手指关节检测的置信度和上一帧跟踪的置信度，输入最多检测手的数目，进行关节点检测
        self.hands = self.mp_hands.Hands(static_image_mode=static_image_mode,
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.7,
                                         max_num_hands=max_num_hands)
        # 初始化一个列表来存储
        self.landmark_list = []
        self.landmark_world_list = []
        self.landmark_distance_list = []
        # 手势历史
        self.action_history = []
        # 关键动作历史
        self.key_action_history = []
        # 定义所有的手势动作对应的鼠标操作
        self.action_labels = {
            'none': '无',
            'move': '鼠标移动',
            'click_single_active': '触发单击',
            'click_single_ready': '单击准备',
            'click_right_active': '触发右击',
            'click_right_ready': '右击准备',
            'scroll_up': '向上滑页',
            'scroll_down': '向下滑页',
            'drag': '鼠标拖拽',
            'five': '五指',
            'zero': '握拳',
            'close_open': '握拳张开'
        }
        self.action_deteted = ''

    # 检查左右手在数组中的index
    def checkHandsIndex(self, handedness):
        # 判断数量
        if len(handedness) == 1:
            handedness_list = [handedness[0].classification[0].label]
        else:
            handedness_list = [handedness[0].classification[0].label, handedness[1].classification[0].label]
        return handedness_list

    # 计算两点点的距离
    def getDistance(self, pointA, pointB):
        # math.hypot为勾股定理计算两点长度的函数，得到食指和拇指的距离
        return math.hypot((pointA[0] - pointB[0]), (pointA[1] - pointB[1]))

    # 获取手指在图像中的坐标
    def getFingerXY(self, index):
        return (self.landmark_list[index][1], self.landmark_list[index][2])

    # 将手势识别的结果绘制到图像上
    def drawInfo(self, img, action):
        thumbXY, indexXY, middleXY = map(self.getFingerXY, [4, 8, 12])

        if action == 'move':
            img = cv2.circle(img, indexXY, 20, (255, 0, 255), -1)

        elif action == 'click_single_active':
            # middle_point = int((indexXY[0] + thumbXY[0]) / 2), int((indexXY[1] + thumbXY[1]) / 2)
            # img = cv2.circle(img, middle_point, 30, (0, 255, 0), -1)
            # 以前的单击准备改为触发单击
            img = cv2.circle(img, indexXY, 30, (0, 255, 0), -1)
            img = cv2.circle(img, thumbXY, 30, (0, 255, 0), -1)

        elif action == 'click_single_ready':
            img = cv2.circle(img, indexXY, 20, (255, 0, 255), -1)
            img = cv2.circle(img, thumbXY, 20, (255, 0, 255), -1)
            img = cv2.line(img, indexXY, thumbXY, (255, 0, 255), 2)


        elif action == 'click_right_active':
            # middle_point = int((indexXY[0] + middleXY[0]) / 2), int((indexXY[1] + middleXY[1]) / 2)
            # img = cv2.circle(img, middle_point, 30, (0, 255, 0), -1)
            img = cv2.circle(img, indexXY, 30, (0, 255, 0), -1)

        elif action == 'click_right_ready':
            img = cv2.circle(img, indexXY, 20, (255, 0, 255), -1)
            img = cv2.circle(img, middleXY, 20, (255, 0, 255), -1)
            img = cv2.line(img, indexXY, middleXY, (255, 0, 255), 2)

        return img

    # 返回手掌各种动作
    def checkHandAction(self, img, drawKeyFinger=True):
        upList = self.checkFingersUp()
        action = 'none'

        if len(upList) == 0:
            return img, action, None

        # 侦测距离
        dete_dist = 50
        # 食指
        key_point = self.getFingerXY(8)

        # 移动模式：单个食指在上，鼠标跟随食指指尖移动，需要smooth处理防抖
        if upList == [0, 1, 0, 0, 0]:
            action = 'move'

        # 单击：食指与拇指出现暂停移动，如果两指捏合，触发单击
        if upList == [1, 1, 0, 0, 0]:
            # l1 = self.getDistance(self.getFingerXY(4), self.getFingerXY(8))
            # action = 'click_single_active' if l1 < dete_dist else 'click_single_ready'

            # 暂改为直接触发单击
            action = 'click_single_active'
        # 拖动：食指和中指
        if upList == [0, 1, 1, 0, 0]:
            # l1 = self.getDistance(self.getFingerXY(8), self.getFingerXY(12))
            # action = 'click_right_active' if l1 < dete_dist else 'click_right_ready'
            action = 'drag'

        # 右击：大拇指加食指、中指
        if upList == [0, 1, 0, 1, 1] or upList == [0, 1, 0, 0, 1]:
            action = 'click_right_active'

        # 向上滑：四指向上
        if upList == [0, 1, 1, 1, 1]:
            action = 'scroll_up'

        # 向下滑：三指向上
        if upList == [0, 0, 1, 1, 1]:
            action = 'scroll_down'

        # 五指
        if upList == [1, 1, 1, 1, 1]:
            action = 'five'

        # 握拳
        if upList == [0, 0, 0, 0, 0]:
            action = 'zero'

        # 根据动作绘制相关点
        img = self.drawInfo(img, action) if drawKeyFinger else img

        self.action_deteted = self.action_labels[action]

        # 动作存入历史，并尝试清理
        self.action_history.append(action)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[:-30]

        # 关键动作停留超过5帧
        # flag = True
        key_action = "none"
        action_map = {}
        action_last_arr = self.action_history[-5:]
        for i, item in enumerate(action_last_arr):
            count = 1
            if item in action_map:
                count = action_map[item] + 1
            action_map[item] = count
            if count >= 3:
                key_action = item

        # 关键动作存入关键动作历史，并尝试清理
        if key_action != "none":
            self.key_action_history.append(key_action)
            if len(self.key_action_history) > 100:
                self.key_action_history = self.key_action_history[:-30]
        # 上两帧关键动作
        last_key_action = self.key_action_history[-2:]

        # 非关键动作不触发，关键动作覆盖当前动作，减少抖动
        if (action == "move" and key_action != "move") or (action == "drag" and key_action != "drag"):
            action = "none"
        elif key_action == "move" or key_action != "drag":
            action = key_action

        # 两个关键动作触发一个新的操作
        if len(last_key_action) == 2 and last_key_action[0] == "zero" and last_key_action[1] == "five":
            action = "close_open"

        return img, action, key_point

    def distance(self, lm1, lm2):
        x = lm1.x - lm2.x
        y = lm1.y - lm2.y
        z = 0
        if lm1.z is not None:
            z = lm1.z - lm2.z
        return math.sqrt(x ** 2 + y ** 2 + z ** 2)

    def distance2(self, lm1, lm2):
        x = lm1[1] - lm2[1]
        y = lm1[2] - lm2[2]
        return math.sqrt(x ** 2 + y ** 2)

    def lm_distance(self):
        self.landmark_distance_list = []
        lm0 = self.landmark_world_list[0]
        for j, lm in enumerate(self.landmark_world_list):
            self.landmark_distance_list.append(self.distance(lm, lm0))

    # 返回向上手指的数组
    def checkFingersUp(self):

        fingerTipIndexs = [4, 8, 12, 16, 20]
        upList = []
        if len(self.landmark_list) == 0:
            return upList

        self.lm_distance()

        # 拇指，比较距17点的距离
        dis1 = self.distance(self.landmark_world_list[fingerTipIndexs[0]], self.landmark_world_list[17]) - 0.02
        dis2 = self.distance(self.landmark_world_list[fingerTipIndexs[0]-2], self.landmark_world_list[17])
        # dis1 = (self.landmark_list[fingerTipIndexs[0]][0] - self.landmark_list[17][0]) ** 2 + (self.landmark_list[fingerTipIndexs[0]][1] - self.landmark_list[17][1]) ** 2
        # dis2 = (self.landmark_list[fingerTipIndexs[0]-2][0] - self.landmark_list[17][0]) ** 2 + (self.landmark_list[fingerTipIndexs[0]-2][1] - self.landmark_list[17][1]) ** 2
        if dis1 > dis2:
            upList.append(1)
        else:
            upList.append(0)

        # 其他指头，比较距手掌根部的距离
        for i in range(1, 5):
            diff = self.landmark_distance_list[fingerTipIndexs[i]] - self.landmark_distance_list[fingerTipIndexs[i] - 2]
            if diff > 0 and diff / self.landmark_distance_list[fingerTipIndexs[i]] > 0.1:
                upList.append(1)
            else:
                upList.append(0)

        # for i in range(1, 5):
        #     dis1 = self.distance2(self.landmark_list[fingerTipIndexs[i]], self.landmark_list[fingerTipIndexs[i] - 3])
        #     dis2 = self.distance2(self.landmark_list[fingerTipIndexs[i] - 2], self.landmark_list[fingerTipIndexs[i] - 3])
        #     diff = dis1 - dis2
        #     if diff > 0 and diff / dis1 > 0.35 and dis2 / dis1 > 0.35:
        #         upList.append(1)
        #     else:
        #         upList.append(0)

        return upList

    # 分析手指的位置，得到手势动作
    def processOneHand(self, img, hand, show=True):
        utils = Utils()

        results = self.hands.process(img)
        self.landmark_list = []
        self.landmark_world_list = []
        curr_i = -1

        if results.multi_hand_landmarks:
            for i, item in enumerate(results.multi_handedness):
                if item.classification[0].label == hand:
                    curr_i = i
                    break
            if curr_i == -1:
                return img

            hand_landmarks = results.multi_hand_landmarks[curr_i]
            self.landmark_world_list = results.multi_hand_world_landmarks[curr_i].landmark

            # 遍历landmark
            for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                p_x, p_y = math.ceil(finger_axis.x * w), math.ceil(finger_axis.y * h)

                self.landmark_list.append([
                    landmark_id, p_x, p_y,
                    finger_axis.z
                ])

            # 图像显示
            if show:
                # 手部连线
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                # 框和label
                x_min, x_max = min(self.landmark_list, key=lambda i: i[1])[1], \
                max(self.landmark_list, key=lambda i: i[1])[1]
                y_min, y_max = min(self.landmark_list, key=lambda i: i[2])[2], \
                max(self.landmark_list, key=lambda i: i[2])[2]

                img = cv2.rectangle(img, (x_min - 30, y_min - 30), (x_max + 30, y_max + 30), (0, 255, 0), 2)
                img = utils.cv2AddChineseText(img, self.action_deteted, (x_min - 20, y_min - 120),
                                              textColor=(255, 0, 255), textSize=60)

        return img
