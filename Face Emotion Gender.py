
from statistics import mode
import numpy as np
import cv2
import tensorflow.keras
from tensorflow.keras.models import load_model
from time import sleep
import os


print("Libraries Updated")

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input


print("Libraries Updated")


# parameters for loading data and images
detection_model_path = r'Face Emotion Gender\haarcascade_frontalface_default.xml'
emotion_model_path = r'Face Emotion Gender\fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = r'Face Emotion Gender\simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX


# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)


Emotion_Count = 0
emotion_text = ''
emotion_text_Old = 'check'
Depression_Count = 0
Angry_Count = 0
Sad_Count = 0
Fear_Count = 0

while True:

    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        rgb_face = np.expand_dims(rgb_face, 0)
        rgb_face = preprocess_input(rgb_face, False)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        gender_window.append(gender_text)

        if len(gender_window) > frame_window:
            emotion_window.pop(0)
            gender_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
            gender_mode = mode(gender_window)
        except:
            continue

        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_mode,
                  color, 0, -20, 1, 1)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)

#    print(emotion_text)
    if( emotion_text_Old == emotion_text ):
#        print("Same Emotion")
        if( emotion_text == 'angry' ):
            Angry_Count = Angry_Count + 1
            if( Angry_Count >= 5 ):
                print("Angry Emotion Detected")
                sleep(3)  
                Depression_Flag = 1

        elif( emotion_text == 'sad' ):
            Sad_Count = Sad_Count + 1
            if( Sad_Count >= 5 ):
                print("Sad Emotion Detected")
                sleep(3)  
                Depression_Flag = 1

        elif( emotion_text == 'fear' ):
            Fear_Count = Fear_Count + 1
            if( Fear_Count >= 5 ):
                print("Fear Emotion Detected")
                sleep(3)  
                Depression_Flag = 1


##        if( Depression_Flag == 1):
##            if( (Sad_Count >= 4) and (Angry_Count >= 1) and (Fear_Count >= 1) ):
##                print("Less Stress or Depression Detected")
##            elif( (Sad_Count >= 2) and (Angry_Count >= 4) and (Fear_Count >= 3) ):
##                print("More Stress or Depression Detected")
##            elif( (Sad_Count >= 2) and (Angry_Count >= 4) and (Fear_Count >= 4) ):
##                print("High Stress or Depression Detected")

        if( Depression_Flag == 1):
            if( Sad_Count >= 4 ):
                print("Less Stress or Depression Detected")
            if( Angry_Count >= 4 ):
                print("More Stress or Depression Detected")
            if( Fear_Count >= 4 ):
                print("Moderate Stress or Depression Detected")

            sleep(2)
            Angry_Count = 0
            Sad_Count = 0
            Fear_Count = 0

            
    emotion_text_Old = emotion_text
    Depression_Flag = 0
    sleep(0.4)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()


print("Project End")


