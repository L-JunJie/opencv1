import cv2 as cv
detector=cv.CascadeClassifier('haarcascade_eye.xml')
detector.load('haarcascade_eye.xml')
detector1=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
detector1.load('haarcascade_frontalface_default.xml')
cap=cv.VideoCapture(0)#读取摄像头，对摄像头中的人脸进行识别
while (True):
    ret, img = cap.read()  # 一帧一帧的读取
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face=detector1.detectMultiScale(gray,1.3,5)
    # face=detector.detectMultiScale(gray)
    for (x, y, w, h) in face:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # [height,width,pixels]=img.shape
    # new_img=cv.resize(img,(int(width/3),int(height/3)),interpolation=cv.INTER_CUBIC)
    cv.imshow('img', img)
    if cv.waitKey(20) & 0xff == ord('q'):
        break
cap.release()
cv.destroyAllWindows()