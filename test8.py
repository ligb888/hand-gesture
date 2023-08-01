import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        frame.flags.writeable = False
        results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # we are interested in the index finger tip (id 8)
                # and middle finger tip (id 12) to perform click action
                index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # normalize the coordinates, and swap x and y
                screen_width, screen_height = pyautogui.size()
                x, y = index_fingertip.x * screen_width, screen_height - index_fingertip.y * screen_height

                # move the mouse to the index finger tip
                pyautogui.moveTo(x, y)
                # click if index and middle fingertips are close to each other
                if abs(index_fingertip.x - middle_fingertip.x) < 0.01 and abs(index_fingertip.y - middle_fingertip.y) < 0.01:
                    pyautogui.click()

        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # cap.release()
        # cv2.destroyAllWindows()
