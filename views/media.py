import uuid

import cv2
import numpy as np
import streamlit as st
import streamlit_shadcn_ui as ui
import hydralit_components as hc
from PIL import Image
from utils.filters import apply_gaussian_blur, convert_to_grayscale, equalize_histogram
from utils.hdfs import save_image_to_hdfs

from deepface import DeepFace


def detect_emotions(frame):
    # Detect faces, gender, and emotion
    results = DeepFace.analyze(frame, actions=['emotion', 'gender'])

    # Counters for gender
    total_women = 0
    total_men = 0
    metrics_cols = st.columns(3)

    # Display the results and process each face
    for result in results:
        face_info = result['region']
        emotion_info = result['emotion']
        gender_info = result['gender']

        # Process each face
        (x, y, w, h) = face_info['x'], face_info['y'], face_info['w'], face_info['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(frame, f"Emotion: {result['dominant_emotion']}, Gender: {result['dominant_gender']}",
                    (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Update gender counters
        if result['dominant_gender'] == 'Man':
            total_men += 1
        elif result['dominant_gender'] == 'Woman':
            total_women += 1

    # Calculate the probability of most students being satisfied (Happy or Neutral)
    satisfied_students = sum(1 for result in results if result['dominant_emotion'] in ['happy', 'neutral'])
    total_students = len(results)
    try:
        probability_satisfied = (satisfied_students / total_students) * 100
    except ZeroDivisionError:
        probability_satisfied = 0

    # Display the metrics
    with metrics_cols[0]:
        ui.metric_card(title="Total Men", content=str(total_men), description="Total number of men detected")
    with metrics_cols[1]:
        ui.metric_card(title="Total Women", content=str(total_women), description="Total number of women detected")
    with metrics_cols[2]:
        ui.metric_card(title="Total Students", content=str(total_students), description="Total number of students")

    # Display the probability
    status = {
        "title": "Student Satisfaction",
        "content": f"The majority of students seem to be {'' if probability_satisfied > 70 else 'not '}satisfied, with a probability of {probability_satisfied:.2f}%",
        "sentiment": 'good' if probability_satisfied > 70 else 'bad',
        "bar_value": probability_satisfied
    }

    hc.info_card(
        title=status['title'],
        content=status['content'],
        sentiment=status['sentiment'],
        bar_value=status['bar_value'],
        title_text_size='1.5em',
        content_text_size='1.3em',
        icon_size='1.5em',
    )
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame_rgb


def process_webcam_frame(frame):
    # Detect faces using Haar Cascade
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Detect gender and emotions for each face
    results = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Analyze gender and emotion
        face_roi = frame[y:y + h, x:x + w]
        result = DeepFace.analyze(img_path=face_roi, actions=['emotion', 'gender'], enforce_detection=False)
        results.append(result)

        # Display results on the frame
        emotion_info = result[0]['emotion']
        gender_info = result[0]['gender']

        dominant_emotion = max(emotion_info, key=emotion_info.get)
        dominant_gender = max(gender_info, key=gender_info.get)

        cv2.putText(frame, f"Emotion: {dominant_emotion}, Gender: {dominant_gender}",
                    (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    # Calculate metrics
    total_women = sum(1 for result in results if result[0]['dominant_gender'] == 'Woman')
    total_men = sum(1 for result in results if result[0]['dominant_gender'] == 'Man')
    total_students = len(results)
    satisfied_students = sum(1 for result in results if result[0]['dominant_emotion'] in ['happy', 'neutral'])
    try:
        probability_satisfied = (satisfied_students / total_students) * 100
    except ZeroDivisionError:
        probability_satisfied = 0

    # Display the metrics
    # Display the metrics
    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        ui.metric_card(key=f"total_men_card{uuid.uuid4()}", title="Total Men", content=str(total_men),
                       description="Total number of men detected")
    with metrics_cols[1]:
        ui.metric_card(key=f"total_women_card{uuid.uuid4()}", title="Total Women", content=str(total_women),
                       description="Total number of women detected")
    with metrics_cols[2]:
        ui.metric_card(key=f"total_students_card{uuid.uuid4()}", title="Total Students", content=str(total_students),
                       description="Total number of students")
    # Display the probability
    status = {
        "title": "Student Satisfaction",
        "content": f"The majority of students seem to be {'' if probability_satisfied > 70 else 'not '}satisfied, with a probability of {probability_satisfied:.2f}%",
        "sentiment": 'good' if probability_satisfied > 70 else 'bad',
        "bar_value": probability_satisfied
    }

    hc.info_card(
        title=status['title'],
        content=status['content'],
        sentiment=status['sentiment'],
        bar_value=status['bar_value'],
        title_text_size='1.5em',
        content_text_size='1.3em',
        icon_size='1.5em',
        key=f"student_satisfaction_card{uuid.uuid4()}"
    )

    return frame


def init():
    st.title("Image and Video Processing App")

    # Sidebar select option
    mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Webcam"])

    if mode == "Image":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Process the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            st.image(image_np, caption="Uploaded Image", use_column_width=True)

            # Apply image processing functions
            grayscale_image = convert_to_grayscale(image_np)
            equalized_image = equalize_histogram(image_np)
            blurred_image = apply_gaussian_blur(image_np)

            # Display processed images in a grid
            st.subheader("Processed Images")
            st.image(detect_emotions(image_np), channels="BGR", use_column_width=True)
            st.image([grayscale_image, equalized_image, blurred_image], caption=["Grayscale", "Equalized", "Blurred"],
                     width=270)

            # Save processed images to HDFS
            save_image_to_hdfs(grayscale_image, '/processed_images', 'grayscale')
            save_image_to_hdfs(equalized_image, '/processed_images', 'equalized')
            save_image_to_hdfs(blurred_image, '/processed_images', 'blurred')

    elif mode == "Video":
        # TODO: Implement video processing
        pass

    elif mode == "Webcam":
        st.write("Starting Webcam...")
        video_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        stop_button = st.button("Stop")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process webcam frame
            processed_frame = process_webcam_frame(frame)

            # Update the placeholder with the new frame
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

            # Break the loop if 'Stop' button is pressed
            if stop_button:
                break
        cap.release()
