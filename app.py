import streamlit as st
from my_pages import home, about, projects_main, contact

# 사이드바 메뉴
st.sidebar.title("Portfolio Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Me", "Projects", "Contact"])

# 각 페이지 연결
if page == "Home":
    home.app()
elif page == "About Me":
    about.app()
elif page == "Projects":
    projects_main.app()
elif page == "Contact":
    contact.app()
