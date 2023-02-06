import cv2
import numpy as np
import os
from PIL import Image

labels = ['Elon Musk.jpg', 'Elon Musk2.jpg', 'Jeff Bezos.jpg', 'Messi.jpg', 'Messi.jpg',
          'Messi.jpg', 'Ryan Reynolds.jpg', 'suwan.jpg', 'suwan2.jpg', 'suwan2.jpg']  # 라벨 지정

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")  # 저장된 값 가져오기

cap = cv2.VideoCapture(0)  # 카메라 실행

if cap.isOpened() == False:  # 카메라 생성 확인
    exit()

while True:
    ret, img = cap.read()  # 현재 이미지 가져오기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백으로 변환
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)  # 얼굴 인식

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # 얼굴 부분만 가져오기

        id_, conf = recognizer.predict(roi_gray)  # 얼마나 유사한지 확인
        print(id_, conf)

        if conf >= 40 and conf < 100:
            font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 지정
            name = labels[id_]  # ID를 이용하여 이름 가져오기
            cv2.putText(img, name, (x, y), font, 1, (0, 0, 255), 2)
            print('id:', id_)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Preview', img)  # 이미지 보여주기
    if cv2.waitKey(10) >= 0:  # 키 입력 대기, 10ms
        break

# 전체 종료
cap.release()
cv2.destroyAllWindows()
