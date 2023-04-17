import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import classifier
import numpy as np
import time

st.set_page_config(page_title="My Finance App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme = """
    [theme]
    base="dark"
    backgroundColor="#2f77ce"
    textColor="#f5f4f4"
    font="serif"
"""

#st.set_config(raw_config=theme)


# Navigation Bar
menu = ["Home", "Model", "Stock Charts", "More"]
choice = st.sidebar.selectbox("Select a page", menu)

st.markdown(
    """
    <style>
    body {
        background-color: ##000000;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Home Section
if choice == "Home":
    st.write("When approaching the task of building a project that integrates Finance and Artificial Intelligence, we wanted to address a problem we felt was being neglected. We identified that one of the biggest problems in the financial sector, is the reputation of its fundamental pillar, Wall Street.")

    st.write("Wall Street has a reputation for being ruthless and cold, which is largely due to the high-pressure environment of the financial industry. The traders and investors who work on Wall Street are under constant pressure to perform and make profits for their clients, which can create a cutthroat culture that rewards individual success above all else.")

    st.write ("In addition to the pressure to perform, the culture of Wall Street is known for its intense competitiveness, where individuals are pitted against one another in a Darwinian struggle for survival. This can lead to a focus on personal gain and a disregard for the well-being of others, which contributes to the perception of Wall Street as being cold and ruthless.")

    st.write("So how are we trying to solve this issue? We realized that we can’t change the intrinsic nature of wall street is, but we can put a filter on the lens in which its viewed from. Zodiac signs are often thought of as a deeply personal aspect of an individual's identity. People often identify strongly with their zodiac sign and view it as a reflection of their personality traits and characteristics. Many people believe that their zodiac sign can offer insight into their strengths, weaknesses, and even their romantic compatibility with others.")

    st.write("Our astrological investing algorithm is a tool that uses a variety of astrological factors, such as planetary alignments and transits, to identify potential investment opportunities that may be more favorable for individuals with certain zodiac signs in an effort to help investors connect with stocks that may align with their astrological profile.")

    st.write("By using astrological data in this way, the algorithm can help investors find stocks that more aligned with their personal strengths, preferences, and risk tolerance. This can lead to a more personalized investment strategy that takes into account not just financial considerations, but also the unique characteristics and tendencies of the investor.")

    st.write("While not everyone may believe in astrology, those who do can find value in using astrological investing algorithms as a way to deepen their understanding of themselves and the investment landscape. By using this tool to identify stocks that may resonate with their astrological profile, investors may be able to make more informed decisions that are better aligned with their personal goals and values.")
 

# Model Section
elif choice == "Model":
    st.header("Main Model goes here...")
    
# Stock Charts Section (Classifier)
elif choice == "Stock Charts":
    st.header("Stock Charts")
    st.subheader("Select a stock to see what times the highest and lowest values occur")
    df = pd.read_csv('stocks.csv')
    options = list(df.columns)[2:-2]  
    option = st.selectbox('Select a column to plot', options)
    if st.button('Generate Plot'):
        classifier.plot_stock(option)   
        st.image('plots/stock_plot.png')

# About Section
else:
    st.header("About Us")
    



