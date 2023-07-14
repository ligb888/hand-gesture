import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # 调用镜头
wcap = cap.set(3, 800)
hcap = cap.set(4, 800)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
drawmp = mp.solutions.drawing_utils
topIds = [4, 8, 12, 16, 20]
pTime = 0
while True:
    success, img = cap.read()  # cap.read()会返回两个值：Ture或False 和 帧
    if success:
        list = []
        # opencv调用相机拍摄的图像格式是BGR,得转化为RGB格式便于图像处理
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        # print(result.multi_hand_landmarks) #打印手部21个点的坐标信息
        if result.multi_hand_landmarks:
            for handlm in result.multi_hand_landmarks:
                # print(landmarks) #打印坐标信息
                drawmp.draw_landmarks(img, handlm, mpHands.HAND_CONNECTIONS)
                for id, lm in enumerate(handlm.landmark):
                    h, w, c = img.shape  # 图像的长、宽、通道
                    cx, cy = int(lm.x * w), int(lm.y * h)  # 将坐标数值转为整数
                    list.append([id, cx, cy])
            if len(list) != 0:
                finger = []
                # 判断左右手
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]  # 仅取第一个检测到的手
                    # 大拇指
                    if hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x < hand_landmarks.landmark[
                        mpHands.HandLandmark.THUMB_TIP].x:
                        if list[topIds[0]][1] > list[topIds[0] - 1][1]:
                            finger.append(1)
                        else:
                            finger.append(0)
                    else:
                        if list[topIds[0]][1] < list[topIds[0] - 1][1]:
                            finger.append(1)
                        else:
                            finger.append(0)
                # 其余四指
                for id in range(1, 5):
                    if list[topIds[id]][2] < list[topIds[id] - 2][2]:
                        finger.append(1)
                        # print("---伸直----")
                        # print('id:', topIds[id], ', Y:', list[topIds[id]][2])
                        # print('id:', topIds[id] - 2, ', Y:', list[topIds[id] - 2][2])
                        # print("_________ ")
                    else:
                        finger.append(0)
                        # print("---弯曲----")
                        # print('id:', topIds[id], ', Y:', list[topIds[id]][2])
                        # print('id:', topIds[id] - 2, ', Y:', list[topIds[id] - 2][2])
                        # print("_________ ")
                    totalFinger = finger.count(1)
                    print(totalFinger)
            cv2.putText(img, str(totalFinger), (40, 350), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
        # 检测帧数
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.imshow("images", img)
    if cv2.waitKey(1) & 0xff == 27:  # 按'ESC'键退出
        break