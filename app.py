import streamlit as st
from my_pages import home, projects_main, contact

# 사이드바 메뉴
st.sidebar.title("Portfolio")
page = st.sidebar.radio("Go to", ["Home", "Projects", "Contact"])

# 각 페이지 연결
if page == "Home":
    home.app()
elif page == "Projects":
    projects_main.app()
elif page == "Contact":
    contact.app()
