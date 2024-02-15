# VisioCraft Media Processing

VisioCraft Media Processing is a Streamlit-based application that allows users to process images and videos using various filters and algorithms. It utilizes the DeepFace library for facial analysis, including emotion and gender detection.

## Features

- **Image Processing:** Upload an image, apply a variety of filters and transformations, and view the results interactively.
- **Video Processing:** Upload a video, select the number of frames to extract, apply filters to the frames, and visualize the processed frames.

- **Webcam Processing:** Start the webcam, capture real-time video, and analyze the detected faces for gender and emotion.

- **Metrics Display:** Get real-time metrics, including the total number of men, women, and students, along with the probability of student satisfaction based on detected emotions.

## Prerequisites

Make sure you have the following installed:

- Python (3.7 or later)
- Streamlit
- OpenCV
- DeepFace
- Other required dependencies (install using `pip install -r requirements.txt`)

## Installation

1. Clone the repository:

````bash
git clone https://github.com/abdellatif-laghjaj/big-data-project
cd project

```bash
pip install -r requirements.txt
````

## Usage

```bash
streamlit run app.py --server.runOnSave true
```
