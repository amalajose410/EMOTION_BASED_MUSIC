# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os

from collections import Counter
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
import base64

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 


df = pd.read_csv("muse_v6.csv")

df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name','emotional','pleasant','link','artist']]

# df = df.sort_values(by=["emotional", "pleasant"])
df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)
# df.reset_index()

df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]





# Define the PyTorch model (matching the TensorFlow structure)
class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # (62,62,1) → (62,62,32)
        self.pool1 = nn.MaxPool2d(2, 2)                          # (62,62,32) → (31,31,32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # (31,31,32) → (31,31,64)
        self.pool2 = nn.MaxPool2d(2, 2)                          # (31,31,64) → (15,15,64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (15,15,64) → (15,15,128)
        self.pool3 = nn.MaxPool2d(2, 2)                           # (15,15,128) → (7,7,128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # (7,7,128) → (7,7,128)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(2048, 1024)  # Fully connected layer
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 7)  # Output layer with 7 classes
        self.dropout2 = nn.Dropout(0.5)
        
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))

        print(x.shape)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.dropout3(x)

        return x

tf_to_pytorch = {
    'Conv_sequential_1/conv2d_1_1/BiasAdd:0.weight': 'conv1.weight',
    'Conv_sequential_1/conv2d_1_1/BiasAdd:0.bias': 'conv1.bias',
    'Conv_sequential_1/conv2d_2_1/BiasAdd:0.weight': 'conv2.weight',
    'Conv_sequential_1/conv2d_2_1/BiasAdd:0.bias': 'conv2.bias',
    'Conv_sequential_1/conv2d_3_1/BiasAdd:0.weight': 'conv3.weight',
    'Conv_sequential_1/conv2d_3_1/BiasAdd:0.bias': 'conv3.bias',
    'Conv_sequential_1/conv2d_4_1/BiasAdd:0.weight': 'conv4.weight',
    'Conv_sequential_1/conv2d_4_1/BiasAdd:0.bias': 'conv4.bias',
    'MatMul_sequential_1/dense_1_1/BiasAdd:0.weight': 'fc1.weight',
    'MatMul_sequential_1/dense_1_1/BiasAdd:0.bias': 'fc1.bias',
    'MatMul_sequential_1/dense_2_1/BiasAdd:0.weight': 'fc2.weight',
    'MatMul_sequential_1/dense_2_1/BiasAdd:0.bias': 'fc2.bias'
}

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf_weights = torch.load("model.pth", map_location=device)
pytorch_state_dict = {}
for tf_name, pt_name in tf_to_pytorch.items():
    if tf_name in tf_weights:
        pytorch_state_dict[pt_name] = tf_weights[tf_name]
model = EmotionNet().to(device)
model.load_state_dict(pytorch_state_dict)
model.eval()
# print(model)

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# OpenCV Face detection
cv2.ocl.setUseOpenCL(False)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Streamlit UI
st.title("Emotion-based Music Recommendation")
col1, col2, col3 = st.columns(3)

list_emotions = []
status_placeholder = st.empty()
video_placeholder = st.empty()

# Preprocessing function
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Emotion scanning function
if col2.button('SCAN EMOTION (Click here)'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Cannot access webcam.")
    else:
        count = 0
        list_emotions.clear()
        status_placeholder.text("Scanning for emotions...")

        while count < 20:  # Scan 20 frames
            ret, frame = cap.read()
            if not ret:
                status_placeholder.text("Error: Cannot read video frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]  # Extract face ROI
                processed_img = transform(roi_gray).unsqueeze(0).to(device)  # Apply transformations

                # Make prediction
                with torch.no_grad():
                    output = model(processed_img)
                    # print(output)
                    prediction = torch.argmax(output, dim=1).item()
                    detected_emotion = emotion_dict[prediction]

                # Store detected emotion
                list_emotions.append(detected_emotion)

                # Draw bounding box and emotion label
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                cv2.putText(frame, detected_emotion, (x + 20, y - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            count += 1

        cap.release()
        cv2.destroyAllWindows()

        # Process detected emotions
        list_emotions = list(dict.fromkeys(list_emotions))  # Remove duplicates
        status_placeholder.text("Emotion detection complete.")
        st.success(f"Detected emotions: {list_emotions}")

# Music Recommendation
def recommend_music(emotions):
    emotion_dfs = {
        "Neutral": df_neutral,
        "Angry": df_angry,
        "Fearful": df_fear,
        "Happy": df_happy,
        "Sad": df_sad
    }

    music_df = pd.DataFrame()
    sample_sizes = [30, 20, 15, 10, 5]  # Sample size per emotion
    for i, emotion in enumerate(emotions):
        if emotion in emotion_dfs and i < len(sample_sizes):
            music_df = pd.concat([music_df, emotion_dfs[emotion].sample(n=sample_sizes[i])], ignore_index=True)

    return music_df

if list_emotions:
    recommended_songs = recommend_music(list_emotions)
    st.subheader("Recommended Songs")
    for _, row in recommended_songs.iterrows():
        st.markdown(f"🎵 [{row['name']} - {row['artist']}]({row['link']})")
