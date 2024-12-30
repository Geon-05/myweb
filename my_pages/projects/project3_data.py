import streamlit as st
from src.culture import base_data
import pandas as pd
import altair as alt

def app():
    data_dict = base_data()
    
    st.title("Sample Dataset")
    
    st.subheader("사용자 데이터")
    st.dataframe(data_dict["user_culture_df"])\

    df_melt = data_dict["user_culture_df"].melt(id_vars=["ID"], var_name="Question", value_name="Value", ignore_index=False)

    heatmap = alt.Chart(df_melt).mark_rect().encode(
        x=alt.X('Question:N', sort=list(data_dict["user_culture_df"].columns[2:])),
        y=alt.Y('ID:O', sort=list(data_dict["user_culture_df"]['ID'])),
        color=alt.Color('Value:Q', scale=alt.Scale(domain=[0,1], scheme='blues'))
    ).properties(
        width=800,
        height=600
    )

    st.altair_chart(heatmap, use_container_width=True)

    
    st.subheader("선호 콘텐츠")
    st.dataframe(data_dict["user_ranking_df"])
    
    st.subheader("콘텐츠 비용")
    st.dataframe(data_dict["user_cost_df"])
    
    st.subheader("사용자 생활권")
    st.dataframe(data_dict["user_area_df"])
    
    st.subheader("내부 서비스 코드 정의")
    st.dataframe(data_dict["code_service_df"])
    
    st.subheader("콘텐츠 코드 정의")
    st.dataframe(data_dict["code_culture_df"])
    
    st.subheader("지역 코드 정의")
    st.dataframe(data_dict["code_area_df"])