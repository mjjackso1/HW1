import streamlit as st

st.set_page_config(page_title="HW Manager", layout="wide")

hw1 = st.Page("HW/HW1.py", title="HW 1")
hw2 = st.Page("HW/HW2.py", title="HW 2")
hw3 = st.Page("HW/HW3.py", title="HW 3")
hw4 = st.Page("HW4.py", title="HW 4", default=True)

pg = st.navigation([hw4, hw3, hw2, hw1])
pg.run()
