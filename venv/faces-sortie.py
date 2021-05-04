import numpy as np
import cv2
import datetime
import os
import pickle

maintenant = datetime.datetime.now()
os.chdir(r"C:\Users\starinfo\Desktop\Camcheck\Resultats\recognition")
fichier = open(str(maintenant.day) + '-' + str(maintenant.month) + '-' + str(maintenant.year) + '-sortie.txt', 'a')
fichier.write("Nous sommes le " + str(maintenant.day) + "/" + str(maintenant.month) + " de l'année " + str(maintenant.year))
fichier.write('\n')
face_cascade = cv2.CascadeClassifier(r"C:\Users\starinfo\Desktop\Camcheck\venv\cascades\haarcascade_frontalface_alt2.xml")
# eye_cascade = cv2.CascadeClassifier(r"C:\Users\test\Desktop\stageformationhumaine\recognition\cascades\haarcascade_eye.xml")
# smile_cascade = cv2.CascadeClassifier(r"C:\Users\test\Desktop\stageformationhumaine\recognition\cascades\haarcascade_smile.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r"C:\Users\starinfo\Desktop\Camcheck\venv\recognizers\face-trainner.yml")

labels = {"person_name": 1}
with open(r"C:\Users\starinfo\Desktop\Camcheck\venv\pickles\face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()} #permuter k et v

cap = cv2.VideoCapture(0)
while (True):
    # Capture frame-by-frame
    maintenant = datetime.datetime.now()
    ret, frame = cap.read() #ret is a boolean regarding whether or not there was a return at all, at the frame is each frame that is returned.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        id_, conf = recognizer.predict(roi_gray) #conf is probablity
        print(conf)
        if conf >= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            color = (255, 0, 0)
            stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
            fichier.write("Mm/Mr " + labels[id_] + " a sortie à l'entreprise à " + str(maintenant.hour) + " heures " + str(maintenant.minute) + " minutes.")
            fichier.write('\n')
            print(labels[id_])
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, "Inconnu!!", (x, y), font, 1, color, stroke, cv2.LINE_AA)
            img_item = "inconnu sort " + str(maintenant.hour) + " h" + str(maintenant.minute) + " m.png"
            cv2.imwrite(img_item, roi_color)
            color = (255, 0, 0)
            stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
            fichier.write("Mm/Mr " + "Inconnu" + " a sortie à l'entreprise à " + str(maintenant.hour) + " heures " + str(maintenant.minute) + " minutes.")
            fichier.write('\n')
            print("Inconnu")


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fichier.close()
file = open(str(maintenant.day) + '-' + str(maintenant.month) + '-' + str(maintenant.year) + '-sortie.txt', "r")
lines = file.readlines()
file.close()
new_list = []
for line in lines:
    if line not in new_list:
        new_list.append(line)

fichier = open(str(maintenant.day) + '-' + str(maintenant.month) + '-' + str(maintenant.year) + '-sortie.txt', 'w')
for i in new_list:
    fichier.write(i)
    fichier.write('\n')



fichier.close()
cap.release()
cv2.destroyAllWindows()
