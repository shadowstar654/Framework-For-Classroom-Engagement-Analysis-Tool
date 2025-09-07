# classroom_engagement.py
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Face detection
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands, \
     mp_face.FaceDetection(min_detection_confidence=0.7) as face_detection:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_results = face_detection.process(frame_rgb)
        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Detect hands
        hand_results = hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the result
        cv2.imshow('Classroom Engagement Analysis', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
