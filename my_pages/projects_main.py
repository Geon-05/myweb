import streamlit as st

def app():
    st.title("Projects")
    st.write("Welcome to the Projects Page! Here are some of my recent works.")

    # 프로젝트 선택
    selected_project = st.radio("Select a project to view details:", ["Overview", "Project 1", "Project 2", "Project 3"])

    if selected_project == "Overview":
        st.subheader("Project Overview")
        st.write("""
        - **Project 1**: Brief description of Project 1.
        - **Project 2**: Brief description of Project 2.
        - **Project 3**: Brief description of Project 3.
        """)
    elif selected_project == "Project 1":
        from my_pages.projects import project1  # 함수 내부에서 임포트
        project1.app()
    elif selected_project == "Project 2":
        from my_pages.projects import project2  # 함수 내부에서 임포트
        project2.app()
    elif selected_project == "Project 3":
        from my_pages.projects import project3  # 함수 내부에서 임포트
        project3.app()
