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
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

def is_two_finger_gesture(hand, h):
    """
    Returns True only if:
    - Index finger is up
    - Middle finger is up
    - Ring finger is down
    - Pinky finger is down
    """

    # Landmark indices
    INDEX_TIP = 8
    INDEX_PIP = 6

    MIDDLE_TIP = 12
    MIDDLE_PIP = 10

    RING_TIP = 16
    RING_PIP = 14

    PINKY_TIP = 20
    PINKY_PIP = 18

    index_up = hand.landmark[INDEX_TIP].y < hand.landmark[INDEX_PIP].y
    middle_up = hand.landmark[MIDDLE_TIP].y < hand.landmark[MIDDLE_PIP].y

    ring_down = hand.landmark[RING_TIP].y > hand.landmark[RING_PIP].y
    pinky_down = hand.landmark[PINKY_TIP].y > hand.landmark[PINKY_PIP].y

    return index_up and middle_up and ring_down and pinky_down


# -----------------------------
# Spotify volume state
# -----------------------------
spotify_volume = 50
VOLUME_STEP = 3
COOLDOWN = 0.15
last_volume_time = 0

prev_y = None
MOVE_THRESHOLD = 12  # pixels

# -----------------------------
# Spotify volume control
# -----------------------------
def set_spotify_volume(volume):
    volume = max(0, min(100, volume))
    os.system(
        f"osascript -e 'tell application \"Spotify\" to set sound volume to {volume}'"
    )

set_spotify_volume(spotify_volume)

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

        # Index & middle fingertips
        index_tip = hand.landmark[8]
        middle_tip = hand.landmark[12]

        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
        mx, my = int(middle_tip.x * w), int(middle_tip.y * h)

        # Draw points
        cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)
        cv2.circle(frame, (mx, my), 8, (0, 255, 0), -1)

        # Average Y position (two-finger drag)
        avg_y = (iy + my) // 2

        current_time = time.time()

        if prev_y is not None:
            dy = avg_y - prev_y

            if (
                abs(dy) > MOVE_THRESHOLD
                and (current_time - last_volume_time) > COOLDOWN
            ):
                if dy < 0:
                    spotify_volume += VOLUME_STEP
                    print("VOLUME UP")
                else:
                    spotify_volume -= VOLUME_STEP
                    print("VOLUME DOWN")

                spotify_volume = max(0, min(100, spotify_volume))
                set_spotify_volume(spotify_volume)

                last_volume_time = current_time

        prev_y = avg_y

        cv2.putText(
            frame,
            f"Spotify Volume: {spotify_volume}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

    cv2.imshow("Two-Finger Drag Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
