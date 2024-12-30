import streamlit as st

def app():
    st.title("Projects")
    st.write("프로젝트를 소개합니다!")

    # 프로젝트 선택
    selected_project = st.radio("Select a project to view details:", ["Overview", "이미지 복원", "Youtrue", "Korea Culture"])

    if selected_project == "Overview":
        st.subheader("Project Overview")
        st.write("""
        - **이미지 복원**: [Github](https://github.com/Geon-05/dacon_image_u-net_DeepFill-v2)
        - **Youtrue**: [Website](https://youtrue.duckdns.org)
        - **Korea Culture**: [Github](https://github.com/Geon-05/koreaculture_project1_chatbot)
        """)
    elif selected_project == "이미지 복원":
        from my_pages.projects import project1  # 함수 내부에서 임포트
        project1.app()
    elif selected_project == "Youtrue":
        from my_pages.projects import project2  # 함수 내부에서 임포트
        project2.app()
    elif selected_project == "Korea Culture":
        from my_pages.projects import project3  # 함수 내부에서 임포트
        project3.app()
