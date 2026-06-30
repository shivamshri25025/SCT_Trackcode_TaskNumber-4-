import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            tipIds = [4, 8, 12, 16, 20]
            fingers = []

            # Thumb
            if handLms.landmark[4].x > handLms.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other Fingers
            for tip in tipIds[1:]:
                if handLms.landmark[tip].y < handLms.landmark[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total = fingers.count(1)

            if total == 0:
                gesture = "Fist"
            elif total == 5:
                gesture = "Open Hand"
            elif total == 1:
                gesture = "One Finger"
            elif total == 2:
                gesture = "Two Fingers"
            elif total == 3:
                gesture = "Three Fingers"
            elif total == 4:
                gesture = "Four Fingers"
            else:
                gesture = "Gesture"

            cv2.putText(img, gesture, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
