from time import sleep
import uuid
import pandas as pd
from sklearn.metrics import mean_absolute_error
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date
from utils.constants import navigation_icons, navigations_labels


# Set page layout to wide
st.set_page_config(layout="wide", page_title="Emotion and Object Detection")

# Sidebar
st.sidebar.markdown("<h1 style='text-align: center;'>Emotion and Object Detection</h1>", unsafe_allow_html=True)

# Header
selected = option_menu(None, navigations_labels, icons=navigation_icons, default_index=0, orientation="horizontal")

# Main content
if selected == "Dashboard":
    st.markdown("<h2 style='text-align: center;'>Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This is the dashboard page.</p>", unsafe_allow_html=True)

if selected == "Processed Data":
    st.markdown("<h2 style='text-align: center;'>Processed Data</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This is the processed data page.</p>", unsafe_allow_html=True)

elif selected == "Media":
    st.markdown("<h2 style='text-align: center;'>Media</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This is the media page.</p>", unsafe_allow_html=True)

elif selected == "About":
    st.markdown("<h2 style='text-align: center;'>About</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This is the about page.</p>", unsafe_allow_html=True)