
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            middle_finger = hand_landmarks.landmark[12]  # Middle Finger
            ring_finger = hand_landmarks.landmark[16]  # Ring Finger
            thumb_tip = hand_landmarks.landmark[4]  # Thumb Tip
            index_finger = hand_landmarks.landmark[8]  # Index Finger

            # Convert middle finger position to screen coordinates
            x = int(middle_finger.x * screen_width)
            y = int(middle_finger.y * screen_height)

            # Move cursor with middle finger
            pyautogui.moveTo(x, y, duration=0.1)

            # Detect click (Ring Finger & Thumb close)
            distance = np.sqrt((thumb_tip.x - ring_finger.x)**2 + (thumb_tip.y - ring_finger.y)**2)
            if distance < 0.03:
                pyautogui.click()

            # Get Index Finger Position
            index_x = int(index_finger.x * screen_width)
            index_y = int(index_finger.y * screen_height)
            middle_x = int(middle_finger.x * screen_width)
            middle_y = int(middle_finger.y * screen_height)

            # Scrolling Up & Down
            if index_y < middle_y - 20:  
                pyautogui.scroll(10)  # Scroll Up
            elif index_y > middle_y + 20:  
                pyautogui.scroll(-10)  # Scroll Down

            # Left, Right, Up, Down Navigation
            if index_x < middle_x - 40:  
                pyautogui.press("left")  # Left Arrow
            elif index_x > middle_x + 40:  
                pyautogui.press("right")  # Right Arrow
            elif index_y < middle_y - 50:  
                pyautogui.press("up")  # Up Arrow (Large Movement)
            elif index_y > middle_y + 50:  
                pyautogui.press("down")  # Down Arrow (Large Movement)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
