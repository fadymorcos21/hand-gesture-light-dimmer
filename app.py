import cv2
import mediapipe as mp
import math
import time

# Initialize MediaPipe Hands and its drawing utility.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Default calibration value if no dynamic calibration has been done.
default_max_distance = 125
calibrated_max_distance = 0  # This will update when the hand is fully open.

def get_landmark_positions(hand_landmarks, frame_width, frame_height):
    """Return a list of (id, x, y) for each landmark."""
    landmark_list = []
    for idx, lm in enumerate(hand_landmarks.landmark):
        cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
        landmark_list.append((idx, cx, cy))
    return landmark_list

def fingers_status(landmarks, hand_label):
    """
    Determine which fingers are extended.
    Returns a list: [thumb, index, middle, ring, pinky] with 1 for extended, 0 for folded.
    Adjusts thumb logic based on handedness.
    """
    fingers = []
    # Thumb (different logic for right vs. left hand).
    if hand_label == "Right":
        # For a right hand, thumb is extended if its tip is to the left of its joint.
        fingers.append(1 if landmarks[4][1] < landmarks[3][1] else 0)
    else:  # Left hand
        fingers.append(1 if landmarks[4][1] > landmarks[3][1] else 0)
    
    # Other fingers: if the tip is higher (i.e. has a smaller y-value) than its pip joint, it's extended.
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(1 if landmarks[tip][2] < landmarks[pip][2] else 0)
    return fingers

def map_distance_to_brightness(distance, min_dist=0, max_dist=125):
    """
    Map the measured distance to a brightness percentage.
    When distance == min_dist (0), brightness is 0%.
    When distance == max_dist, brightness is 100%.
    """
    brightness = int((distance - min_dist) / (max_dist - min_dist) * 100)
    return max(0, min(brightness, 100))

# Start video capture.
cap = cv2.VideoCapture(0)
activation_mode = False  # Whether the system is in brightness control mode.
last_detection_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for a more natural (selfie) view.
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine handedness.
            hand_label = results.multi_handedness[i].classification[0].label  # "Left" or "Right"
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = get_landmark_positions(hand_landmarks, w, h)
            status = fingers_status(landmarks, hand_label)  # [thumb, index, middle, ring, pinky]

            # Activation: open hand (all five fingers extended) turns on brightness control.
            if sum(status) == 5:
                activation_mode = True
                last_detection_time = time.time()
                cv2.putText(frame, "Activated", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # During full open hand, update our calibrated maximum distance.
                thumb_tip = landmarks[4][1:3]
                index_tip = landmarks[8][1:3]
                current_distance = math.hypot(index_tip[0] - thumb_tip[0],
                                              index_tip[1] - thumb_tip[1])
                calibrated_max_distance = max(calibrated_max_distance, current_distance)
                # Show calibration info.
                cv2.putText(frame, f"Calib Max: {int(calibrated_max_distance)}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Deactivation: a fist (no fingers extended) turns off brightness control.
            elif sum(status) == 0:
                activation_mode = False
                cv2.putText(frame, "Deactivated", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Once activated, always calculate brightness based on the thumb-index distance.
            if activation_mode:
                thumb_tip = landmarks[4][1:3]
                index_tip = landmarks[8][1:3]
                distance = math.hypot(index_tip[0] - thumb_tip[0],
                                      index_tip[1] - thumb_tip[1])
                # Use the higher of the calibrated max or default if calibration hasn't been done.
                effective_max = calibrated_max_distance if calibrated_max_distance > default_max_distance else default_max_distance
                brightness = map_distance_to_brightness(distance, min_dist=0, max_dist=effective_max)
                cv2.putText(frame, f"Brightness: {brightness}%", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                last_detection_time = time.time()
    else:
        # If no hand is detected for over 3 seconds, turn off activation.
        if time.time() - last_detection_time > 3:
            activation_mode = False

    cv2.imshow("Hand Gesture Light Dimmer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
