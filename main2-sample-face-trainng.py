import cv2
import numpy as np
import os
from PIL import Image

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()  # LBPH를 사용할 새 변수 생성

Face_ID = 0
pev_person_name = []
y_ID = []
x_train = []

Face_Images = os.path.join(os.getcwd(), "faces")  # 이미지 폴더 지정
print(Face_Images)

for root, dirs, files in os.walk(Face_Images):  # 파일 목록 가져오기
    for file in files:
        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):  # 이미지 파일 필터링
            path = os.path.join(root, file)
            person_name = os.path.basename(file)
            print(path, person_name)

            img = cv2.imread(path)  # 이미지 파일 가져오기
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_image, scaleFactor=1.06, minNeighbors=5)  # 얼굴 찾기

            print(Face_ID, faces)

            for (x, y, w, h) in faces:
                roi = gray_image[y:y+h, x:x+w]  # 얼굴부분만 가져오기
                x_train.append(roi)
                y_ID.append(Face_ID)

                Face_ID = Face_ID+1
                pev_person_name.append(person_name)

                recognizer.train(x_train, np.array(y_ID))  # matrix 만들기
                recognizer.save("face-trainner.yml")  # 저장하기

print('pev_person_name:', pev_person_name)
