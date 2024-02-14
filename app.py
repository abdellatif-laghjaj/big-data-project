from time import sleep
import uuid
import pandas as pd
from sklearn.metrics import mean_absolute_error
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date
from utils.constants import navigation_icons, navigations_labels
from views import dashboard, media, about

# Set page layout to wide
st.set_page_config(layout="wide", page_title="Emotion and Object Detection")

# Header
selected = option_menu(None, navigations_labels, icons=navigation_icons, default_index=0, orientation="horizontal")

# Main content
if selected == "Dashboard":
    dashboard.init()

if selected == "Media":
    media.init()

if selected == "About":
    about.init()
