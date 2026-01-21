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

# -----------------------------
# Flick (track change) settings
# -----------------------------
prev_x = None
prev_y = None
prev_time = None
last_flick_time = 0

FLICK_SPEED_THRESHOLD = 1500
FLICK_COOLDOWN = 1.0
HORIZONTAL_DOMINANCE = 2.5

# -----------------------------
# Smooth volume control settings
# -----------------------------
VOLUME_SENSITIVITY = 0.08   # higher = faster volume change
SMOOTHING = 0.7             # higher = smoother
spotify_volume = None
prev_avg_y = None
smoothed_dy = 0


# -----------------------------
# pause and play control settings
# -----------------------------
PINCH_THRESHOLD = 35      # pixels (tune if needed)
PINCH_COOLDOWN = 0.8      # seconds
last_pinch_time = 0


# -----------------------------
# Spotify AppleScript helpers
# -----------------------------
def spotify_next():
    os.system("osascript -e 'tell application \"Spotify\" to next track'")

def spotify_previous():
    os.system("osascript -e 'tell application \"Spotify\" to previous track'")

def spotify_play_pause():
    os.system("osascript -e 'tell application \"Spotify\" to playpause'")


def set_spotify_volume(volume):
    volume = max(0, min(100, volume))
    os.system(
        f"osascript -e 'tell application \"Spotify\" to set sound volume to {volume}'"
    )

def get_spotify_volume():
    try:
        result = os.popen(
            "osascript -e 'tell application \"Spotify\" to get sound volume'"
        ).read().strip()
        return int(result)
    except:
        return 50

spotify_volume = get_spotify_volume()
print(f"Initial Spotify volume: {spotify_volume}")

# -----------------------------
# Gesture gate: EXACTLY TWO FINGERS
# -----------------------------
def is_two_finger_gesture(hand):
    INDEX_TIP, INDEX_PIP = 8, 6
    MIDDLE_TIP, MIDDLE_PIP = 12, 10
    RING_TIP, RING_PIP = 16, 14
    PINKY_TIP, PINKY_PIP = 20, 18

    index_up = hand.landmark[INDEX_TIP].y < hand.landmark[INDEX_PIP].y
    middle_up = hand.landmark[MIDDLE_TIP].y < hand.landmark[MIDDLE_PIP].y
    ring_down = hand.landmark[RING_TIP].y > hand.landmark[RING_PIP].y
    pinky_down = hand.landmark[PINKY_TIP].y > hand.landmark[PINKY_PIP].y

    return index_up and middle_up and ring_down and pinky_down

# -----------------------------
# Gesture gate: PINCH
# -----------------------------

def is_pinch(hand, w, h):
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]

    tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
    ix, iy = int(index_tip.x * w), int(index_tip.y * h)

    distance = ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5
    return distance < PINCH_THRESHOLD


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
        # =============================
        # PINCH → PLAY / PAUSE
        # =============================
        if is_pinch(hand, w, h):
            if current_time - last_pinch_time > PINCH_COOLDOWN:
                spotify_play_pause()
                print("PLAY / PAUSE")
                last_pinch_time = current_time

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        current_time = time.time()

        if is_two_finger_gesture(hand):
            index_tip = hand.landmark[8]
            middle_tip = hand.landmark[12]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            iy = int(index_tip.y * h)
            my = int(middle_tip.y * h)
            avg_y = (iy + my) // 2

            # =============================
            # SMOOTH TWO-FINGER DRAG → VOLUME
            # =============================
            if prev_avg_y is not None:
                dy = avg_y - prev_avg_y
                smoothed_dy = SMOOTHING * smoothed_dy + (1 - SMOOTHING) * dy
                delta_volume = -smoothed_dy * VOLUME_SENSITIVITY

                spotify_volume += delta_volume
                spotify_volume = max(0, min(100, int(spotify_volume)))
                set_spotify_volume(spotify_volume)

            prev_avg_y = avg_y

            # =============================
            # SAFE FLICK → TRACK CONTROL
            # =============================
            if prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy_flick = y - prev_y
                dt = current_time - prev_time

                if dt > 0:
                    speed = abs(dx) / dt

                    if (
                        speed > FLICK_SPEED_THRESHOLD
                        and abs(dx) > HORIZONTAL_DOMINANCE * abs(dy_flick)
                        and (current_time - last_flick_time) > FLICK_COOLDOWN
                    ):
                        if dx > 0:
                            spotify_next()
                            print("NEXT TRACK")
                        else:
                            spotify_previous()
                            print("PREVIOUS TRACK")

                        last_flick_time = current_time

            prev_x = x
            prev_y = y
            prev_time = current_time

        else:
            prev_x = None
            prev_y = None
            prev_avg_y = None
            smoothed_dy = 0

            cv2.putText(
                frame,
                "Show ONLY index + middle fingers",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"Spotify Volume: {spotify_volume}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

    cv2.imshow("Gesture-Controlled Spotify", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
