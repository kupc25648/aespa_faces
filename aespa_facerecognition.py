# Main file for aespa face recognition
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
import cv2
from google.colab.patches import cv2_imshow
from keras.models import load_model
from mtcnn import MTCNN
from matplotlib import pyplot
import numpy as np
from numpy import load
import os
from os import listdir
from os.path import isdir
from PIL import Image

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Import Function Part =============================================================================

from functions import imgtoarray, extract_face, extract_faces, load_faces, load_dataset, makedata, de_face, showimg

# Import Class Part =============================================================================

from model import AespaML

test_aespa = AespaML()

# Making training data version 0 and version 1
test_aespa.make_train_test_embedding1(0)
test_aespa.make_train_test_embedding1(1)

# Train and Test using data 0 and 1 with different SVM models
old_data = test_aespa.embedding_data_path
new_data = test_aespa.embedding_data_path1

old_data_name = 'OLD'
new_data_name = 'NEW'

data_list = [[old_data,old_data_name],[new_data,new_data_name]]

classifier0 = test_aespa.classifier
classifier1 = test_aespa.classifier1
classifier2 = test_aespa.classifier2
classifier3 = test_aespa.classifier3

classifier0_name='LINEAR SVM'
classifier1_name='LINEAR SVM'
classifier2_name='POLY SVM'
classifier3_name='BRF SVM' # This might be too overfit

classifier_list = [[classifier0,classifier0_name],
                   [classifier1,classifier1_name],
                   [classifier2,classifier2_name],
                   [classifier3,classifier3_name]]

data_num = 1
for i in range(len(data_list)):
  for j in range(len(classifier_list)):
    test_aespa.train_general(classifier_list[j][0],data_list[i][0],classifier_list[j][1],data_list[i][1])

# Use model to predict from MV's image
test_aespa.whose_face('data_mv/imgs/aespa_savage/aespa_savage_frame_5269.jpg',classifier2)
