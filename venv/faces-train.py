import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(r"C:\Users\starinfo\Desktop\Camcheck\venv\__file__"))

image_dir = os.path.join(r"C:\Users\starinfo\Desktop\Camcheck\venv\images")
#le chemin pour la data set que j'ai réalisé : le chemin doit être le nom du dossier qui contient des sousdossiers nommés avec les noms des employés et contenant plusieurs images pour que le modèle soit efficace

face_cascade = cv2.CascadeClassifier(r"C:\Users\starinfo\Desktop\Camcheck\venv\cascades\haarcascade_frontalface_alt2.xml")
#le chemin vers les fichiers en cascade, de préferences ils doivent être dans le même endroit de l'exécution




recognizer= cv2.face_LBPHFaceRecognizer.create()
#création du recognizer
current_id = 0 #personne à laquelle actuellement
label_ids = {} #nom de la personne
y_labels = [] #les coordonnés de visage
x_train = []
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"): #image en jpg ou png
			path = os.path.join(root, file)
			label = os.path.basename(root).lower()
			if label not in label_ids:
				label_ids[label] = current_id
				current_id += 1

			id_ = label_ids[label]
			pil_image = Image.open(path).convert("L") #grayscale
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			#Une image numérique est composée de pixels. Lorsqu'elle est redimensionnée,
			#le bord des formes ayant un angle particulier prend la forme d'escalier :
			# c'est le crénelage, ou aliasing. Pour supprimer cet effet visuel disgracieux, on utilise l'anticrénelage, ou anti-aliasing.
			image_array = np.array(final_image, "uint8")
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=10)


			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)
			#on fait le traitement pour chaque image pour ajouter les coordonnées de visage à un fichier qui contient les face labels à l'aide du module pickle

with open(r"C:\Users\starinfo\Desktop\Camcheck\venv\pickles\face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)
recognizer.train(x_train,np.array(y_labels))
recognizer.save(r"C:\Users\starinfo\Desktop\Camcheck\venv\recognizers\face-trainner.yml")