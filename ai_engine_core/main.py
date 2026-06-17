import streamlit as st
from views.detection import render_detection
from utils.icons import icon_path

# -----------------------------------------------------------------------------
# Config & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="云盾：多模态虚假新闻检测竞技场",
    page_icon=icon_path("shield"),
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    # Direct Render
    render_detection()

if __name__ == "__main__":
    main()
