import cv2
import mediapipe as mp
import time
import math

# -----------------------------
# Initialize webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# Initialize MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Variables to track movement
# -----------------------------
prev_x = None
prev_time = None

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror view
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Index finger tip = landmark 8
        index_tip = hand.landmark[8]

        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        current_time = time.time()

        # Draw point
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

        # -----------------------------
        # Calculate movement speed
        # -----------------------------
        if prev_x is not None:
            dx = x - prev_x
            dt = current_time - prev_time

            if dt > 0:
                speed = abs(dx) / dt
                cv2.putText(
                    frame,
                    f"Speed: {int(speed)}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

        prev_x = x
        prev_time = current_time

    cv2.imshow("Step 1 - Index Finger Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
