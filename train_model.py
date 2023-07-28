# import the necessary packages
from imutils import paths
import face_recognition
#import argparse
import pickle
import cv2
import os

# images are sent to database folder
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))

knownEncodings = []
knownNames = []

# loop image paths to compile images
for (i, imagePath) in enumerate(imagePaths):
	# take persons name from file, use it in training
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model="hog")

	#facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()