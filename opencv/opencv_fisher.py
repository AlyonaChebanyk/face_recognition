# example of creating a face embedding
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from numpy import expand_dims
from numpy import load
from keras.models import load_model
from numpy import asarray
from numpy import savez_compressed
from keras.models import Input
from keras.models import Sequential
from keras.layers import Flatten
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import cv2
from PIL import Image
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert("RGB")
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]["box"]
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    image = image.convert("L")
    face_array = asarray(image)
    return face_array


# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + "/"
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print(">loaded %d examples for class: %s" % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# # load train dataset
# trainX, trainy = load_dataset("train/")
# print(trainX.shape, trainy.shape)
# # load test dataset
# testX, testy = load_dataset("val/")
# print(testX.shape, testy.shape)
# # save arrays to one file in compressed format
# savez_compressed("5-celebrity-faces-dataset.npz", trainX, trainy, testX, testy)

model = cv2.face.FisherFaceRecognizer_create()
#
# # load dataset
data = load("5-celebrity-faces-dataset.npz")
trainX, trainy, testX, testy = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
# print(trainX.shape)
# print("Dataset: train=%d, test=%d" % (trainX.shape[0], testX.shape[0]))
# in_encoder = Normalizer(norm="l2")
# trainX = in_encoder.transform(trainX)
# testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
# model.train(trainX, trainy)
# model.save('fisher_model.yml')

model.read('fisher_model.yml')
yhat_train = []
yhat_test = []
for train in trainX:
    yhat_train.append(model.predict(train)[0])
for test in testX:
    yhat_test.append(model.predict(test)[0])
# score
yhat_train = asarray(yhat_train)
yhat_test = asarray(yhat_test)
print(testy)
print(yhat_test)
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print("Accuracy: train=%.3f, test=%.3f" % (score_train*100, score_test*100))

# 80%