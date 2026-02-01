#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install opencv-python mediapipe numpy wmi


# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
import math
import wmi

# Initialize webcam
cap = cv2.VideoCapture(0)

# Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize WMI 
c = wmi.WMI(namespace='wmi')

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []

            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Thumb tip (4) and Index finger tip (8)
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # Draw connection
            cv2.circle(img, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Distance between fingers
            length = math.hypot(x2 - x1, y2 - y1)

            # Map distance to brightness (0–100)
            brightness = np.interp(length, [30, 200], [0, 100])
            brightness = int(brightness)

            # Set system brightness
            try:
                c.WmiMonitorBrightnessMethods()[0].WmiSetBrightness(brightness, 0)
            except:
                pass

            # Brightness bar mapping
            bar = np.interp(brightness, [0, 100], [400, 150])

            # Green to Red color transition
            green = int(np.interp(brightness, [0, 100], [255, 0]))
            red = int(np.interp(brightness, [0, 100], [0, 255]))

            # Draw brightness bar
            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 2)
            cv2.rectangle(img, (50, int(bar)), (85, 400), (red, green, 0), cv2.FILLED)

            # Show brightness percentage
            cv2.putText(img, f'{brightness} %',
                        (40, 430),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

    cv2.imshow("Adaptive Brightness Controller", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




