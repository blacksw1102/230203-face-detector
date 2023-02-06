import cv2
import numpy as np
import mediapipe as mp

# Load the face detection graph.
face_detection_graph = mp.solutions.face_detection.load_gpu_face_detection_model()

# Load the face recognition model.
face_recognition_model = cv2.face.LBPHFaceRecognizer_create()
face_recognition_model.read('face-trainner.yml')

# Open the camera.
camera = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera.
    ret, frame = camera.read()

    # Use the face detection graph to detect faces in the frame.
    faces = mp.solutions.face_detection.detect_faces(
        frame, face_detection_graph)

    # Iterate over the faces and recognize them.
    for face in faces:
        x, y, w, h = face.bounding_box.flatten().tolist()
        face_img = frame[y:y+h, x:x+w]
        label, confidence = face_recognition_model.predict(face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Label: {label} ({confidence})',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame.
    cv2.imshow('Face Recognition', frame)

    # Check if the user presses the 'q' key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window.
camera.release()
cv2.destroyAllWindows()
