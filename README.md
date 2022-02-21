# aespa_faces
Main file for aespa face recognition
This also could be run on google colab

If run on google colab


1. Create a folder and Upload data file into your Google Drive

2. Run this scipt before this program

    from google.colab import drive

    drive.mount('/content/gdrive')

    import os

    os.chdir('gdrive/MyDrive/[Your folder name]')

    !pip install mtcnn

    !pip install keras-facenet

Train and Validation data are .npz files

Or you can download raw files from 

https://drive.google.com/drive/folders/1KVAuNYk3OHEvHlKNmMeV1dyMvOZXN8vg?usp=sharing
