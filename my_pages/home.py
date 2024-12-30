import streamlit as st
from PIL import Image

def app():
    st.title("안녕하세요! AI 개발자 지망생입니다!")
    st.subheader("원내 두번째 프로젝트!")
    image = Image.open("img/koreaculture/스크린샷 2024-12-23 182041.png")
    st.image(image, caption="youtrue")
    st.write("""
             학원에서 진행한 첫 팀 프로젝트입니다.
             여가생활 정보를 추천하는 챗봇이며
             사용자 기반 추천 정보제공과 여가생활에 드는 평균비용을 안내하는 챗봇입니다.
             """)
