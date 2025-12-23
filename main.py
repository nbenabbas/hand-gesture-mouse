import cv2
import math
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

screen_width, screen_height = pyautogui.size()

# MediaPipe setup
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

def hand_gesture_mouse_control():
    cap = cv2.VideoCapture(0)
    clicking = False
    pinch_threshold = 0.04  # tweak if needed

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

                # Index fingertip (8)
                index_tip = landmarks[8]
                x = int(index_tip.x * frame.shape[1])
                y = int(index_tip.y * frame.shape[0])

                # Move mouse
                """
                pyautogui.moveTo(
                    index_tip.x * screen_width,
                    index_tip.y * screen_height
                )
                """
                target_x = index_tip.x * screen_width
                target_y = index_tip.y * screen_height

                # smoother mouse move
                smooth_x, smooth_y = None, None
                alpha = 0.5

                if smooth_x is None:
                    smooth_x, smooth_y = target_x, target_y
                else:
                    smooth_x = alpha * target_x + (1 - alpha) * smooth_x
                    smooth_y = alpha * target_y + (1 - alpha) * smooth_y

                pyautogui.moveTo(smooth_x, smooth_y)

                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

                # Thumb tip (4) for pinch
                thumb_tip = landmarks[4]
                distance = math.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2
                )

                if distance < pinch_threshold and not clicking:
                    pyautogui.click()
                    clicking = True
                elif distance >= pinch_threshold:
                    clicking = False

            cv2.imshow("Hand Gesture Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_gesture_mouse_control()
