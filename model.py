# model file for aespa face recognition
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
# Import Part =============================================================================

from keras.models import load_model
from mtcnn import MTCNN

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

class AespaML():
    def __init__(self,faceembedder=None,classifier=SVC(kernel='linear',probability=True)):
        self.member_name  = ['giselle','karina','ningning','winter']
        self.member_count = [0,0,0,0]
        self.song_name        = None
        self.song_total_frame = 0
        self.song_source      = None
        self.embedding_data_path   = 'aeapa_faces_data.npz'
        self.embedding_data_path1  = 'aeapa_faces_data1.npz'
        self.train_path            = 'data_face/train/'
        self.train_path1           = 'data_face/train_v2/'
        self.test_path             = 'data_face/val/'
        self.test_path1             = 'data_face/val_v2/'

        # ML models
        self.faceembedder = load_model(os.path.join('keras_facenet', 'facenet_keras.h5')) #FaceNet()
        self.classifier   = SVC(kernel='linear',probability=True)#classifier
        self.classifier1   = SVC(kernel='linear',probability=True)#classifier
        self.classifier2   = SVC(kernel='poly',degree=3,probability=True)#classifier
        self.classifier3   = SVC(kernel='rbf', probability=True)#classifier

    def reset_count(self):
        self.member_count     = [0,0,0,0]
        self.song_name        = None
        self.song_total_frame = 0
        self.song_source      = None
    def set_song(self,song):
        self.song_name        = song
        self.song_source      = 'aespa_' + song

    def make_individual_embedding(self,face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = self.faceembedder.predict(samples)
        #print(yhat)
        return yhat[0]

    def make_train_test_embedding(self):
        trainX, trainy, testX, testy = makedata(self.train_path,self.test_path)
        #trainX, trainy, testX, testy = datalist[0], datalist[1], datalist[2], datalist[3]
        # Train set
        newTrainX = []
        for face_pixels in trainX:
            embedding = self.make_individual_embedding(face_pixels)
            newTrainX.append(embedding)
        newTrainX = np.asarray(newTrainX)
        # Test set
        newTestX = []
        for face_pixels in testX:
            embedding = self.make_individual_embedding(face_pixels)
            newTestX.append(embedding)
        newTestX = np.asarray(newTestX)
        print('Done embedding making data')
        np.savez_compressed(self.embedding_data_path, newTrainX, trainy, newTestX, testy)

    def train_general(self,model,data_path,model_name='',data_name=''):
        # load embedding_data
        data = load(data_path)
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        #print('-----------------')
        #print(trainX)
        #print('-----------------')
        #print(testX)
        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)

        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        testy = out_encoder.transform(testy)

        # fit model
        model.fit(trainX, trainy)
        # predict
        yhat_train = model.predict(trainX)
        yhat_test  = model.predict(testX)
        # score
        score_train = accuracy_score(trainy, yhat_train)
        score_test = accuracy_score(testy, yhat_test)
        # summarize
        print('{} DATA - {} MODEL'.format(data_name,model_name))
        print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))




    def make_train_test_embedding1(self,v=0):
        if v==0:
            trainX, trainy, testX, testy = makedata(self.train_path,self.test_path)
            print('Data_v0')
        elif v==1:
            trainX, trainy, testX, testy = makedata(self.train_path1,self.test_path1)
            print('Data_v1')
        #trainX, trainy, testX, testy = makedata(self.train_path1,self.test_path)
        #trainX, trainy, testX, testy = datalist[0], datalist[1], datalist[2], datalist[3]
        # Train set
        newTrainX = []
        for face_pixels in trainX:
            embedding = self.make_individual_embedding(face_pixels)
            newTrainX.append(embedding)
        newTrainX = np.asarray(newTrainX)
        # Test set
        newTestX = []
        for face_pixels in testX:
            embedding = self.make_individual_embedding(face_pixels)
            newTestX.append(embedding)
        newTestX = np.asarray(newTestX)
        print('Done embedding making data')
        if v==0:
            np.savez_compressed(self.embedding_data_path, newTrainX, trainy, newTestX, testy)
            print('Save Data_v0')
        elif v==1:
            np.savez_compressed(self.embedding_data_path1, newTrainX, trainy, newTestX, testy)
            print('Save Data_v1')





    def whose_face(self,source,model,show=True):
        if show == True:
          print(source)
          showimg(source)
        faces_arrray = extract_faces(source)
        #print(face_arrray)
        #faces = load_faces(face_arrray)
        #print(faces)
        #for face in range(len(face_arrray)):
        #print(face_arrray.shape)
        prest_set = []
        for i in range(len(faces_arrray)):
          embed = self.make_individual_embedding(faces_arrray[i])
          #print(embed.shape)
          pred  = model.predict([embed])
          prest_set.append(pred[0])
          #print(pred)

        # remove duplicate
        prest_set = list(set(prest_set))
        for i in range(len(prest_set)):
          self.member_count[i] += 1
          print('Model predicts that this is {}!'.format(self.member_name[prest_set[i]]))


    def count_face(self,faces):
        # count faces from whose face is this
        return

    def train_face(self,data_train):
        # train self.faceembeder and self.classifier
        return

    def eval_face(self,data_eval):
        # evaluate trained self.faceembeder and trained self.classifier
        return

    def draw_box(self,source,savepath):
        # save img with bounding boxes of faces to savepath
        return
