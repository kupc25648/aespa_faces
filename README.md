# aespa_faces
Main file for aespa face recognition
This also could be run on google colab

If run on google colab


1. Create a folder and Upload data file into your Google Drive

2. Run the scipt before this program

from google.colab import drive

drive.mount('/content/gdrive')

import os

os.chdir('gdrive/MyDrive/[Your folder name]')

!pip install mtcnn

!pip install keras-facenet

