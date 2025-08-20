import cv2
import face_recognition
import os
import numpy as np


#load anh tu kho nhan dang
path = 'picSetup'
images = []
className = []
myList = os.listdir(path)

for cl in myList:
    curimg = cv2.imread(f"{path}/{cl}")
    images.append(curimg)
    className.append(os.path.splitext(cl)[0])

#ma hoa
def MaHoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = MaHoa(images)
print('ma hoa thanh cong')

#Vì video có thể có nhiều frame, ta sẽ chỉ xử lý mỗi 3 frame một lần 
frame_count = 0
video_capture = cv2.VideoCapture("video-test.mp4")
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 3 != 0:
        continue
    framS = cv2.resize(frame, (0, 0), fx = 0.3, fy=0.3)  # Resize to speed up processing
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    #xac dinh vi tri khuon mat
    faceCurFram = face_recognition.face_locations(framS)
    encodeCur = face_recognition.face_encodings(framS)

    #Ktra nhan dang
    for encodeFace, faceLoc in zip(encodeCur, faceCurFram):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.5:  # Adjust threshold as needed
            name = className[matchIndex].upper()
            name = name + 'Acc :' + str(round((1 - faceDis[matchIndex]) * 100, 2)) + '%'
            
        else:
            name = "Unknown" 
        y1, x2, y2, x1 = faceLoc
        cv2.rectangle(framS, (x1, y1), (x2, y2), (0, 234, 45), 2)
        cv2.putText(framS, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        
    cv2.imshow('Video Frame', framS)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
