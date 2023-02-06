"""
https://stackoverflow.com/questions/69264221/is-possible-to-face-recognition-with-mediapipe-in-python
Q: is possible to face recognition with mediapipe in python?
A: 
Mediapipe doesn't provide a face recognition method, only face detector.
The face_recognition library has really good accuracy, It's claimed accuracy is 99%+. your dataset probably isn't good enough.
"""

import os
import sys
import math
import numpy as np
import cv2
import face_recognition
import psutil
import time
import threading
import multiprocessing


cpu_usage = 0.0
gpu_usage = 0.0
is_run = True


def check_cpu_gpu_usage():
    global is_run

    while is_run:
        global cpu_usage
        cpu_usage = psutil.cpu_percent()
        print('cpu_percent:', cpu_usage)
        #print('virtual_memory:', psutil.virtual_memory().percent)
        time.sleep(0.1)


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        #print('value:', value)
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
        # encode faces

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

        print(self.known_face_names)

    def run_recognition(self):
        global is_run

        # Open a connection to the camera
        video_capture = cv2.VideoCapture(0)

        # Set the camera's frames per second (fps)
        video_capture.set(cv2.CAP_PROP_FPS, 24)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        # Run a loop to continuously capture video frames
        while is_run:
            # Capture a frame from the camera
            ret, frame = video_capture.read()

            # Check if the iframe was successfully captured
            if ret:
                # Resize and change the farme to RGB
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all faces in the current frame
                # self.face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=1, model="cnn") # cnn 기반 얼굴 검출기
                self.face_locations = face_recognition.face_locations(
                    rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, self.face_locations)

                self.face_names = []

                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=0.6)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    # Set name, confidence if match_index is exist
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(
                            face_distances[best_match_index], 0.6)

                    self.face_names.append(f'{name} ({confidence})')

                print('self.face_locations:', self.face_locations)
                print('self.face_encodings:', self.face_encodings)

                # Display annotations
                #print('face_locations:', self.face_locations)
                #print('face_names:', self.face_names)
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    print('top:', top, 'right:', right,
                          'bottom:', bottom, 'left:', left)
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top),
                                  (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35),
                                  (right, bottom), (0, 0, 255), -1)
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                # Write the text about cpu usage
                #print('cpu_usage:', cpu_usage)
                display_text = "CPU Usage: {:.2f}%".format(cpu_usage)
                cv2.putText(frame, display_text, (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the frame
                cv2.imshow("Camera", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    is_run = False

        # Release the camera and destroy all windows
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Start.')
    jobs = []
    fr = FaceRecognition()
    processes = [check_cpu_gpu_usage, fr.run_recognition]

    for i in range(len(processes)):
        process = threading.Thread(target=processes[i], daemon=True)
        jobs.append(process)

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()
    print('Done.')
