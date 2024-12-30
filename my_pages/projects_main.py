import streamlit as st
from my_pages.projects import project1, project2, project3, project3_data

def app():
    st.title("Projects")
    st.write("프로젝트를 소개합니다!")

    # 프로젝트 선택
    selected_project = st.radio("Select a project to view details:", ["Overview", "이미지 복원", "Youtrue", "Korea Culture","Korea Culture - dataset"])

    if selected_project == "Overview":
        st.subheader("Project Overview")
        st.write("""
        - **이미지 복원**: [Github](https://github.com/Geon-05/dacon_image_u-net_DeepFill-v2)
        - **Youtrue**: [Website](https://youtrue.duckdns.org)
        - **Korea Culture**: [Github](https://github.com/Geon-05/koreaculture_project1_chatbot)
        """)
    elif selected_project == "이미지 복원":
        project1.app()
    elif selected_project == "Youtrue":
        project2.app()
    elif selected_project == "Korea Culture":
        project3.app()
    elif selected_project == "Korea Culture - dataset":
        project3_data.app()
