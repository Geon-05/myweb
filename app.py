import streamlit as st
from PIL import Image

# 사이드바 설정
st.sidebar.title("Portfolio Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Me", "Projects", "Contact"])

# 홈 페이지
if page == "Home":
    st.title("Welcome to My Portfolio")
    st.subheader("Explore my work and projects.")
    image = Image.open("your_profile_picture.jpg")  # 본인 프로필 사진 경로
    st.image(image, caption="Your Name", use_column_width=True)
    st.write("Hi! I'm [Your Name], a passionate developer who loves creating impactful solutions.")

# 소개 페이지
elif page == "About Me":
    st.title("About Me")
    st.write("""
    I'm [Your Name], a [Your Profession/Field] enthusiast. 
    Here are a few things about me:
    - 🌱 Currently learning [topics you're learning].
    - 📚 Experienced in [your skills].
    - 🎯 Goals: [your professional goals].
    """)

# 프로젝트 페이지
elif page == "Projects":
    st.title("Projects")
    st.write("Here are some of my recent projects:")
    st.markdown("""
    - **[Project 1 Name](#)**: Brief description of project 1.
    - **[Project 2 Name](#)**: Brief description of project 2.
    - **[Project 3 Name](#)**: Brief description of project 3.
    """)
    st.write("Check out more on my [GitHub](https://github.com/your_username).")

# 연락처 페이지
elif page == "Contact":
    st.title("Contact Me")
    st.write("Feel free to reach out to me through the following channels:")
    st.markdown("""
    - 📧 Email: [your_email@example.com](mailto:your_email@example.com)
    - 💼 LinkedIn: [Your LinkedIn](https://linkedin.com/in/your_profile)
    - 🐦 Twitter: [Your Twitter](https://twitter.com/your_handle)
    """)

# 실행 방법 안내
st.sidebar.info("Run the app using `streamlit run app.py` in your terminal.")
