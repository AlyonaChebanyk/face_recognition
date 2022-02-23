# example of creating a face embedding
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import savez_compressed
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
from sklearn.svm import SVC

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
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


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # # scale pixel values
    # face_pixels = face_pixels.astype("float32")
    # # standardize pixel values across channels (global)
    # mean, std = face_pixels.mean(), face_pixels.std()
    # face_pixels = (face_pixels - mean) / std
    # # transform face into one sample
    # samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    face_pixels = expand_dims(face_pixels, axis=0)
    yhat = model.predict(face_pixels)
    return yhat[0]


# # load train dataset
# trainX, trainy = load_dataset("train/")
# print(trainX.shape, trainy.shape)
# # load test dataset
# testX, testy = load_dataset("val/")
# print(testX.shape, testy.shape)
# # save arrays to one file in compressed format
# savez_compressed("5-celebrity-faces-dataset.npz", trainX, trainy, testX, testy)



# create a vggface2 model
vggface_model = VGGFace(model='resnet50', include_top=False, pooling='max', input_shape=(224, 224, 3))
# print(vggface_model.summary())
model = Sequential()
model.add(vggface_model)
model.add(Flatten())
# print(model.summary())
# summarize input and output shape

# # load the face dataset
# data = load("5-celebrity-faces-dataset.npz")
# trainX, trainy, testX, testy = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
# print("Loaded: ", trainX.shape, trainy.shape, testX.shape, testy.shape)
# # convert each face in the train set to an embedding
# trainX = trainX.astype("float32")
# trainX = preprocess_input(trainX, version=2)
# print("Shape:", trainX.shape)
# newTrainX = list()
# for face_pixels in trainX:
#     embedding = get_embedding(model, face_pixels)
#     newTrainX.append(embedding)
# newTrainX = asarray(newTrainX)
# print(newTrainX.shape)
# # convert each face in the test set to an embedding
# testX = testX.astype("float32")
# testX = preprocess_input(testX, version=2)
# newTestX = list()
# for face_pixels in testX:
#     embedding = get_embedding(model, face_pixels)
#     newTestX.append(embedding)
# newTestX = asarray(newTestX)
# print(newTestX.shape)
# # save arrays to one file in compressed format
# savez_compressed("5-celebrity-faces-embeddings.npz", newTrainX, trainy, newTestX, testy)

# load dataset
data = load("5-celebrity-faces-embeddings.npz")
trainX, trainy, testX, testy = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
print("Dataset: train=%d, test=%d" % (trainX.shape[0], testX.shape[0]))
in_encoder = Normalizer(norm="l2")
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel="linear", probability=True)
model.fit(trainX, trainy)
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print("Accuracy: train=%.3f, test=%.3f" % (score_train*100, score_test*100))

data = load("5-celebrity-faces-embeddings.npz")
trainX, trainy, testX, testy = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
print(trainX)