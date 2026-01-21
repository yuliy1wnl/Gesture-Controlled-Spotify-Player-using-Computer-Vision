import cv2
import mediapipe as mp
import time
import os

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# MediaPipe Hands
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
# Flick detection variables
# -----------------------------
prev_x = None
prev_time = None
last_flick_time = 0

FLICK_SPEED_THRESHOLD = 1200
COOLDOWN = 1.0

flick_text = ""

# -----------------------------
# Music control functions (macOS)
# -----------------------------
def next_track():
    os.system("osascript -e 'tell application \"Spotify\" to next track'")

def previous_track():
    os.system("osascript -e 'tell application \"Spotify\" to previous track'")

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        index_tip = hand.landmark[8]

        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        current_time = time.time()

        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

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

                if (
                    speed > FLICK_SPEED_THRESHOLD
                    and (current_time - last_flick_time) > COOLDOWN
                ):
                    if dx > 0:
                        flick_text = "NEXT TRACK"
                        print("NEXT TRACK")
                        next_track()
                    else:
                        flick_text = "PREVIOUS TRACK"
                        print("PREVIOUS TRACK")
                        previous_track()

                    last_flick_time = current_time

        prev_x = x
        prev_time = current_time

    if flick_text:
        cv2.putText(
            frame,
            flick_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    cv2.imshow("Step 3 - Flick Controls Music", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
