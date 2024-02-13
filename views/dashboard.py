import os
import subprocess
import cv2
import numpy as np
from pyspark.sql import SparkSession
import streamlit as st
import streamlit_shadcn_ui as ui
from utils.hdfs import list_files_in_hdfs
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Function to display metrics
def display_metrics(total_videos, total_processed_images, today_processed_images):
    st.subheader("Metrics")
    st.write("Here are some metrics to help you understand the data.")

    metrics_cols = st.columns(3)

    # Total Videos Card
    with metrics_cols[0]:
        ui.metric_card(
            title="Total Videos",
            content=str(total_videos),
            description="Total number of videos",
            key="total_videos_card"
        )

    # Total Processed Images Card
    with metrics_cols[1]:
        ui.metric_card(
            title="Total Processed Images",
            content=str(total_processed_images),
            description="Total number of processed images",
            key="total_processed_images_card"
        )

    # Today's Processed Images Card
    with metrics_cols[2]:
        ui.metric_card(
            title="Today's Processed Images",
            content=str(today_processed_images),
            description="Total number of images processed today",
            key="today_processed_images_card"
        )

# Function to display recent processed images
def display_recent_images(images):
    st.subheader("Recent Processed Images")
    st.write("Here are some of the recently processed images.")
    
    # Generate captions
    captions = []
    for i in range(len(images)):
        captions.append(f"Image {i+1}")

    # Display the images
    st.image(images, width=380, caption=captions)

# Function to display charts
def display_charts(video_files, images_files):
    st.subheader("Charts")
    st.write("Here are some charts to visualize the data.")

    charts_cols = st.columns(2)

    # Chart 1: Total Videos Over Time
    with charts_cols[0]:
        st.subheader("Total Videos Over Time")
        # Assuming the date is the 6th element in the file info
        video_dates = [file_info[5] for file_info in video_files if len(file_info) > 5]
        video_counts = [len(video_files) for _ in video_dates]

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(video_dates, video_counts, marker='o')
        ax1.set_xticks(ax1.get_xticks())
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Total Videos")
        st.pyplot(fig1)

    with charts_cols[1]:
        # Chart 2: Processed Images Distribution
        st.subheader("Processed Images Distribution")
        # Assuming the date is the 6th element in the file info
        image_dates = [file_info[5] for file_info in images_files if len(file_info) > 5]

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(image_dates, bins=len(set(image_dates)), kde=True, ax=ax2)
        ax2.set_xticks(ax2.get_xticks())
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Count")
        st.pyplot(fig2)

# Download images from HDFS
def download_images_from_hdfs(hdfs_files, local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    for hdfs_file in hdfs_files:
        local_file = os.path.join(local_dir, os.path.basename(hdfs_file))
        if not os.path.exists(local_file):
            print(f"Downloading {hdfs_file} to {local_file}")
            subprocess.run(['hdfs', 'dfs', '-get', hdfs_file, local_file], check=True)
        else:
            print(f"File {local_file} already exists. Skipping download.")

# Function to read images from a directory
def read_images_from_dir(directory):
    images = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_path = os.path.join(directory, file)
            img = cv2.imread(file_path)
            img = np.array(img)
            images.append(img)
    return images

def init():
    # Constants
    videos_dir = "/student_videos"
    processed_images_dir = "/processed_images"

    # List files in HDFS
    video_files = list_files_in_hdfs(videos_dir)
    images_files = list_files_in_hdfs(processed_images_dir)

    print(images_files)

    # Get total number of video and processed images
    total_videos = len(video_files)
    total_processed_images = len(images_files)

    # Get today's processed images
    today = datetime.today().strftime('%Y-%m-%d')
    today_processed_images = 0

    for image in images_files:
        # check if the list has enough elements
        if len(image) > 7:
            # get the date
            date = image[5]

            # check if the date is today
            if date == today:
                today_processed_images += 1
        else:
            continue

    # Display metrics
    display_metrics(total_videos, total_processed_images, today_processed_images)

    # Download images from HDFS
    local_images_dir = "processed_images"
    files = list_files_in_hdfs("/processed_images")[0:6]
    download_images_from_hdfs(files, local_images_dir)

    # Read and display recent processed images
    recent_images = read_images_from_dir(local_images_dir)
    display_recent_images(recent_images)

    # Display charts
    display_charts(video_files, images_files)