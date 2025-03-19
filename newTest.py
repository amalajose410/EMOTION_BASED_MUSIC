def fun(list):

    data = pd.DataFrame()

    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry':
             data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'fear':
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v == 'happy':
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)

    elif len(list) == 2:
        times = [30,20]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':    
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':             
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])

    elif len(list) == 3:
        times = [55,20,15]
        for i in range(len(list)): 
            v = list[i]          
            t = times[i]

            if v == 'Neutral':              
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':               
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':             
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':               
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:      
                data = pd.concat([df_sad.sample(n=t)])


    elif len(list) == 4:
        times = [30,29,18,9]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral': 
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':              
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':               
                data =pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])
    else:
        times = [10,7,6,5,2]
        for i in range(len(list)):           
            v = list[i]         
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':           
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':           
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':          
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])

    print("data of list func... :",data)
    return data

def pre(l):

    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    print("Processed Emotions:", result)

    # result = [item for items, c in Counter(l).most_common()
    #           for item in [items] * c]

    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
            print(result)
    print("Return the list of unique emotions in the order of occurrence frequency :",ul)
    return ul

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))


model.load_weights('model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

print("Loading Haarcascade Classifier...")
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion based music recommendation</b></h2>"
            , unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

list = []
status_placeholder = st.empty()
video_placeholder = st.empty()

with col1:
    pass
with col2:
    if st.button('SCAN EMOTION(Click here)'):
        try:
            # Initialize the camera
            cap = cv2.VideoCapture(0)  # Use camera 0 or replace with video file
            if not cap.isOpened():
                raise RuntimeError("Unable to access the camera or video feed.")

            count = 0
            list.clear()
            st_placeholder = st.empty()

            status_placeholder.text("Initializing scanning and processing...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.text("Error: Unable to access the webcam.")
                    break

                status_placeholder.text("Scanning for faces and processing emotions...")

                # Convert frame to grayscale and detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                count += 1

                for (x, y, w, h) in faces:
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

                    # Extract the region of interest (face) in grayscale
                    roi_gray = gray[y:y + h, x:x + w]

                    # Resize and preprocess the image for the model
                    cropped_img = cv2.resize(roi_gray, (48, 48))  # Resize to model input size
                    cropped_img = cropped_img.astype('float32') / 255.0  # Normalize to [0, 1]
                    cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)  # Add batch and channel dimensions

                    # Predict emotion
                    prediction = model.predict(cropped_img)
                    max_index = int(np.argmax(prediction))  # Get the index of the highest score

                    # Map the index to the emotion dictionary
                    detected_emotion = emotion_dict[max_index]

                    # Add detected emotion to the list
                    list.append(detected_emotion)

                    # Annotate the detected emotion on the frame
                    cv2.putText(frame, detected_emotion, (x + 20, y - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Convert the frame to an image for Streamlit
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame, channels="RGB")


                if cv2.waitKey(1) & 0xFF == ord('s'):  # Stop scanning on pressing 's'
                    status_placeholder.text("Scanning stopped by user.")
                    break
                if count >= 20:  # Limit to 20 frames for demo purposes
                    status_placeholder.text("Scanning complete: Processed 20 frames.")
                    break

            # Release resources
            cap.release()
            cv2.destroyAllWindows()

            list = pre(list)
            status_placeholder.text("Processing complete.")
            st.success("Emotions successfully detected")

        except RuntimeError as e:
            status_placeholder.error(f"Error: {e}")

        except Exception as e:
            status_placeholder.error(f"An unexpected error occurred: {e}")     

with col3:
    pass

new_df = fun(list)
st.write("")

st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended song's with artist names</b></h5>"
            , unsafe_allow_html=True)

st.write("---------------------------------------------------------------------------------------------------------------------")

try:
  
    for l,a,n,i in zip(new_df["link"],new_df['artist'],new_df['name'],range(30)):

        st.markdown("""<h4 style='text-align: center;'><a href={}>{} - {}</a></h4>"""
                    .format(l,i+1,n),unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center; color: grey;'><i>{}</i></h5>" 
                    .format(a), unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------------------------------------")
except:
    pass