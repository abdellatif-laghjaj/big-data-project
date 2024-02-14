import uuid
import cv2
import numpy as np
import streamlit as st
import streamlit_shadcn_ui as ui
import hydralit_components as hc
from PIL import Image
from utils.filters import apply_gaussian_blur, convert_to_grayscale, equalize_histogram, apply_laplacian, \
    apply_sobel_filter, apply_bilateral_filter, apply_median_filter, remove_blur_effect, crop_image, open_image, \
    zoom_image, close_image, erode_image, dilate_image, resize_image, rotate_image, sharpen_image, detect_edges, \
    enhance_rgb_quality, adjust_brightness, enhance_rgb_quality
from utils.hdfs import save_image_to_hdfs
from deepface import DeepFace

captions = []


@st.cache_data()
def calculate_metrics(results):
    total_women = sum(1 for result in results if result[0]['dominant_gender'] == 'Woman')
    total_men = sum(1 for result in results if result[0]['dominant_gender'] == 'Man')
    total_students = len(results)
    satisfied_students = sum(1 for result in results if result[0]['dominant_emotion'] in ['happy', 'neutral'])

    try:
        probability_satisfied = (satisfied_students / total_students) * 100
    except ZeroDivisionError:
        probability_satisfied = 0

    return total_women, total_men, total_students, probability_satisfied


@st.cache_data()
def display_metrics(total_women, total_men, total_students, probability_satisfied):
    st.subheader("Metrics")

    placeholder = st.empty()
    t_men_col, t_women_col, t_students_col = st.columns(3)

    with placeholder.container():
        t_men_col.metric(label="Total Men", value=total_men, delta=0)
        t_women_col.metric(label="Total Women", value=total_women, delta=0)
        t_students_col.metric(label="Total Students", value=total_students, delta=0)

        # Display the probability using st.success or st.error
        if probability_satisfied > 70:
            st.success(
                f"The majority of students seem to be satisfied, with a probability of {probability_satisfied:.2f}%")
        else:
            st.error(
                f"The majority of students seem to be not satisfied, with a probability of {probability_satisfied:.2f}%")


# Extract frames from a video
def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = total_frames // num_frames

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()

        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def process_image(image_np, selected_filters):
    processed_images = []
    captions = []

    if "Enhance Image Quality" in selected_filters:
        enhanced_image = enhance_rgb_quality(image_np)
        processed_images.append(enhanced_image)
        captions.append("Enhanced Image Quality")
    if "Grayscale" in selected_filters:
        grayscale_image = convert_to_grayscale(image_np)
        processed_images.append(grayscale_image)
        captions.append("Grayscale")
    if "Equalized" in selected_filters:
        equalized_image = equalize_histogram(image_np)
        processed_images.append(equalized_image)
        captions.append("Equalized")
    if "Blurred" in selected_filters:
        blurred_image = apply_gaussian_blur(image_np)
        processed_images.append(blurred_image)
        captions.append("Blurred")
    if "Laplacian" in selected_filters:
        laplacian_image = apply_laplacian(image_np)
        processed_images.append(laplacian_image)
        captions.append("Laplacian")
    if "Sobel" in selected_filters:
        sobel_image = apply_sobel_filter(image_np)
        processed_images.append(sobel_image)
        captions.append("Sobel")
    if "Sharpen" in selected_filters:
        sharpened_image = sharpen_image(image_np)
        processed_images.append(sharpened_image)
        captions.append("Sharpen")
    if "Enhance RGB" in selected_filters:
        enhanced_image = enhance_rgb_quality(image_np)
        processed_images.append(enhanced_image)
        captions.append("Enhance RGB")
    if "Remove Blur" in selected_filters:
        removed_blur_image = remove_blur_effect(image_np)
        processed_images.append(removed_blur_image)
        captions.append("Remove Blur")
    if "Median" in selected_filters:
        median_image = apply_median_filter(image_np)
        processed_images.append(median_image)
        captions.append("Median")
    if "Brightness" in selected_filters:
        brightened_image = adjust_brightness(image_np)
        processed_images.append(brightened_image)
        captions.append("Brightness")
    if "Rotation" in selected_filters:
        rotated_image = rotate_image(image_np)
        processed_images.append(rotated_image)
        captions.append("Rotation")
    if "Crop" in selected_filters:
        cropped_image = crop_image(image_np, 100, 100, 300, 300)
        processed_images.append(cropped_image)
        captions.append("Crop")
    if "Resize" in selected_filters:
        height = st.sidebar.slider("Select new height", 10, 1000, image_np.shape[0], key="resize_height")
        width = st.sidebar.slider("Select new width", 10, 1000, image_np.shape[1], key="resize_width")
        new_width, new_height = int(width), int(height)
        resized_image = resize_image(image_np, new_width, new_height)
        processed_images.append(resized_image)
        captions.append("Resize")
    if "Zoom" in selected_filters:
        zoomed_image = zoom_image(image_np)
        processed_images.append(zoomed_image)
        captions.append("Zoom")
    if "Edge Detection" in selected_filters:
        edge_detected_image = detect_edges(image_np)
        processed_images.append(edge_detected_image)
        captions.append("Edge Detection")
    if "Dilation" in selected_filters:
        dilated_image = dilate_image(image_np)
        processed_images.append(dilated_image)
        captions.append("Dilation")
    if "Erosion" in selected_filters:
        eroded_image = erode_image(image_np)
        processed_images.append(eroded_image)
        captions.append("Erosion")
    if "Opening" in selected_filters:
        opened_image = open_image(image_np)
        processed_images.append(opened_image)
        captions.append("Opening")
    if "Closing" in selected_filters:
        closed_image = close_image(image_np)
        processed_images.append(closed_image)
        captions.append("Closing")
    if "Bilateral Filter" in selected_filters:
        bilateral_image = apply_bilateral_filter(image_np)
        processed_images.append(bilateral_image)
        captions.append("Bilateral Filter")

    if len(processed_images) == 0:
        st.war("No filters selected. Please select at least one filter to apply to the image.")
        captions = []
    return processed_images, captions


# Function to detect emotions in a frame
def detect_emotions(frame):
    # Detect faces, gender, and emotion
    results = DeepFace.analyze(frame, actions=['emotion', 'gender'], enforce_detection=False)

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
        ui.metric_card(title="Total Men", content=str(total_men), description="Total number of men detected", key=f"total_men_{uuid.uuid4()}")
    with metrics_cols[1]:
        ui.metric_card(title="Total Women", content=str(total_women), description="Total number of women detected", key=f"total_women_{uuid.uuid4()}")
    with metrics_cols[2]:
        ui.metric_card(title="Total Students", content=str(total_students), description="Total number of students", key=f"total_students_{uuid.uuid4()}")

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
        key=f"info_card_{uuid.uuid4()}"
    )
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame_rgb


# Function to process the video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.write(f"Processing {total_frames} frames...")

    video_placeholder = st.empty()
    metrics_placeholder = st.empty()
    stop_button = st.button("Stop")

    total_women, total_men, total_students, probability_satisfied = 0, 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret or stop_button:
            break

        total_women, total_men, total_students, probability_satisfied, processed_frame = process_webcam_frame(frame)
        video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

        with metrics_placeholder.container():
            display_metrics(total_women, total_men, total_students, probability_satisfied)

    cap.release()


# Function to process the webcam frame
def process_webcam_frame(frame):
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        face_roi = frame[y:y + h, x:x + w]
        result = DeepFace.analyze(img_path=face_roi, actions=['emotion', 'gender'], enforce_detection=False)
        results.append(result)

        emotion_info = result[0]['emotion']
        gender_info = result[0]['gender']

        dominant_emotion = max(emotion_info, key=emotion_info.get)
        dominant_gender = max(gender_info, key=gender_info.get)

        cv2.putText(frame, f"Emotion: {dominant_emotion}, Gender: {dominant_gender}",
                    (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    total_women, total_men, total_students, probability_satisfied = calculate_metrics(results)

    return total_women, total_men, total_students, probability_satisfied, frame


def init():
    st.title("Process Media")

    # Sidebar header
    st.sidebar.title("VisioCraft")

    # Sidebar select option
    mode = st.sidebar.selectbox("Choose Mode", ["Image", "Video", "Webcam"])

    if mode == "Image":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Multi-select sidebar for filters
            selected_filters = st.sidebar.multiselect("Select Filters",
                                                      ["Enhance Image Quality", "Grayscale", "Equalized", "Blurred",
                                                       "Laplacian", "Sobel",
                                                       "Sharpen", "Enhance RGB", "Remove Blur", "Median", "Brightness",
                                                       "Rotation", "Crop", "Resize", "Zoom", "Edge Detection",
                                                       "Dilation", "Erosion", "Opening", "Closing", "Bilateral Filter"],
                                                      default=["Enhance Image Quality", "Equalized", "Edge Detection"])

            # Process the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            st.image(image_np, caption="Uploaded Image", use_column_width=True)

            # Processed Images List
            processed_images = []

            # Check if None or All Filters are selected
            if "Apply All Filters" in selected_filters:
                selected_filters = ["Enhance Image Quality", "Grayscale", "Equalized", "Blurred",
                                                       "Laplacian", "Sobel",
                                                       "Sharpen", "Enhance RGB", "Remove Blur", "Median", "Brightness",
                                                       "Rotation", "Crop", "Resize", "Zoom", "Edge Detection",
                                                       "Dilation", "Erosion", "Opening", "Closing", "Bilateral Filter"],
            # Apply image processing functions
            processed_images, captions = process_image(image_np, selected_filters)

            # Display processed images in a grid
            st.subheader("Processed Images")
            st.image(detect_emotions(image_np), channels="BGR", use_column_width=True)
            st.image(processed_images, captions, width=270)

            # Save processed images to HDFS
            for i, processed_image in enumerate(processed_images):
                save_image_to_hdfs(processed_image, '/processed_images', 'processed_img_from_image')

    elif mode == "Video":
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            # Multi-select sidebar for filters
            selected_filters = st.sidebar.multiselect("Select Filters",
                                                      ["Apply All Filters", "Enhance Image Quality", "Grayscale", "Equalized", "Blurred",
                                                       "Laplacian", "Sobel",
                                                       "Sharpen", "Enhance RGB", "Remove Blur", "Median", "Brightness",
                                                       "Rotation", "Crop", "Resize", "Zoom", "Edge Detection",
                                                       "Dilation", "Erosion", "Opening", "Closing", "Bilateral Filter"],
                                                      default=["Enhance Image Quality", "Equalized", "Edge Detection"])

            # Number of frames in the video slider
            frames_number = st.sidebar.slider("Select the number of frames to process", 1, 50, 5)

             # Save the video file
            video_path = f"./videos/{uploaded_video.name}"
            with open(video_path, "wb") as video_file:
                video_file.write(uploaded_video.read())

            # Extract frames from the video
            frames = extract_frames(video_path, frames_number)

            # Process the frames
            for i, frame in enumerate(frames):
                st.subheader(f"Frame {i + 1}")
                st.image(detect_emotions(frame), channels="BGR", use_column_width=True)

            # Apply image processing functions
            processed_frames = []
            for frame in frames:
                processed_images, captions = process_image(frame, selected_filters)
                processed_frames.append(processed_images)

            # Generate captions for the processed frames
            captions = []
            for i in range(frames_number):
                for filter in selected_filters:
                    captions.append(f"Frame {i + 1} - {filter}") 

            # Display processed frames in a grid
            st.subheader("Processed Frames")

            # Flatten the list of processed frames
            flat_processed_frames = [item for sublist in processed_frames for item in sublist]

            # Reverse the order of the flat_processed_frames
            flat_processed_frames = flat_processed_frames[::-1]

            # Display the processed frames
            st.image(flat_processed_frames, captions, width=270)

            # Save processed frames to HDFS
            for i, processed_frame in enumerate(flat_processed_frames):
                save_image_to_hdfs(processed_frame, '/processed_images', 'processed_img_from_video')
                
            # Process the video
            process_video(video_path)

    elif mode == "Webcam":
        st.write("Starting Webcam...")
        video_placeholder = st.empty()
        metrics_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        stop_button = st.button("Stop")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_women, total_men, total_students, probability_satisfied, processed_frame = process_webcam_frame(frame)
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

            if stop_button:
                break

            with metrics_placeholder.container():
                display_metrics(total_women, total_men, total_students, probability_satisfied)
        cap.release()
