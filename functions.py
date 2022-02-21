# function file for aespa face recognition
# This also could be run on google colab
'''
Main file for aespa face recognition
This also could be run on google colab
If run on google colab
    1. Create a folder and Upload data file into your Google Drive
    2. Run these scipt before this program
        from google.colab import drive
        drive.mount('/content/gdrive')
        import os
        os.chdir('gdrive/MyDrive/[Your folder name]')
        !pip install mtcnn
        !pip install keras-facenet
'''

import cv2
from matplotlib import pyplot
import numpy as np
from numpy import load
import os
from os import listdir
from os.path import isdir
from PIL import Image
from google.colab.patches import cv2_imshow

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def imgtoarray(source,convert_type='RGB'):
    # turn source image into an array for MTCNN to predict faces in the image
    # load image from source
    image = Image.open(source)
    # convert to RGB, if needed
    image = image.convert(convert_type)
    # convert to array
    pixels = np.asarray(image)
    return pixels


# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    face_array = None
    if len(results) != 0:
      x1, y1, width, height = results[0]['box']
      # bug fix
      x1, y1 = abs(x1), abs(y1)
      x2, y2 = x1 + width, y1 + height
      # extract the face
      face = pixels[y1:y2, x1:x2]
      # resize pixels to the model size
      image = Image.fromarray(face)
      image = image.resize(required_size)
      face_array = np.asarray(image)

    return face_array

def extract_faces(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    faces_array = []
    for i in range(len(results)):
      x1, y1, width, height = results[i]['box']
      # bug fix
      x1, y1 = abs(x1), abs(y1)
      x2, y2 = x1 + width, y1 + height
      # extract the face
      face = pixels[y1:y2, x1:x2]
      # resize pixels to the model size
      image = Image.fromarray(face)
      image = image.resize(required_size)
      face_array = np.asarray(image)
      faces_array.append(face_array)
    faces_array = np.asarray(faces_array)
    return faces_array


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
        try:
            if face.all() != None:
                faces.append(face)
        except:
            pass

    return faces

def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

def makedata(train_path,test_path):
    # make training data and eval data for SVM classifier
    # which are [x,y]
    # x = embedding from Facenet_model
    # y = one-hot vector of each Aespa's member
    # save training data and eval data
    trainX, trainy = load_dataset(train_path)
    trainX, trainy = shuffle(trainX, trainy)
    testX, testy   = load_dataset(test_path)
    testX, testy   = shuffle(testX, testy)
    # save arrays to one file in compressed format
    #np.savez_compressed(save_data_path, trainX, trainy, testX, testy)
    print('train data')
    print(trainX.shape, trainy.shape)
    print('test data')
    print(testX.shape, testy.shape)
    print('Done making data')
    return [trainX, trainy, testX, testy]

# specify folder to plot
def de_face(actorname,dataplace='train'):
    # actorname = 'giselle','karina','ningning','winter'
    # dataplace ='train' or 'val'
    folder = 'data_face/'+dataplace+'/'+actorname+'/'
    i = 1
    # enumerate files
    for filename in listdir(folder):
        if i <= 14:
            # path
            path = folder + filename
            # get face
            face = extract_face(path)
            print(i, face.shape)
            # plot
            pyplot.subplot(2, 7, i)
            pyplot.axis('off')
            pyplot.imshow(face)
            pyplot.close()
            i += 1
    pyplot.show()

# show image
def showimg(source,convert_type='RGB'):
    # turn source image into an array for MTCNN to predict faces in the image
    # load image from source
    image = Image.open(source)
    # convert to RGB, if needed
    image = image.convert(convert_type)
    # show image
    pyplot.imshow(image)

    return
