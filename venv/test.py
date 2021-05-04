import cv2
face_cascade = cv2.CascadeClassifier(r"C:\Users\starinfo\Desktop\camcheck-master\camcheck-master\cascades\haarcascade_frontalface_alt2.xml")
img = cv2.imread(\Users\starinfo\Desktop\camcheck-master\camcheck-master\images\alan_grant)
faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
