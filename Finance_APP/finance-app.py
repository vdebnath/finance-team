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
menu = ["Home", "Model", "Stock Charts", "Technologies Used", "About Us"]
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
    st.header("Purpose")
    st.write("When approaching the task of building a project that integrates Finance and Artificial Intelligence, we wanted to address a problem we felt was being neglected. We identified that one of the biggest problems in the financial sector, is the reputation of its fundamental pillar, Wall Street.")

    st.write("Wall Street has a reputation for being ruthless and cold, which is largely due to the high-pressure environment of the financial industry. The traders and investors who work on Wall Street are under constant pressure to perform and make profits for their clients, which can create a cutthroat culture that rewards individual success above all else.")

    st.write ("In addition to the pressure to perform, the culture of Wall Street is known for its intense competitiveness, where individuals are pitted against one another in a Darwinian struggle for survival. This can lead to a focus on personal gain and a disregard for the well-being of others, which contributes to the perception of Wall Street as being cold and ruthless.")

    st.write("So how are we trying to solve this issue? We realized that we can’t change the intrinsic nature of wall street is, but we can put a filter on the lens in which its viewed from. Zodiac signs are often thought of as a deeply personal aspect of an individual's identity. People often identify strongly with their zodiac sign and view it as a reflection of their personality traits and characteristics. Many people believe that their zodiac sign can offer insight into their strengths, weaknesses, and even their romantic compatibility with others.")

    st.write("Our astrological investing algorithm is a tool that uses a variety of astrological factors, such as planetary alignments and transits, to identify potential investment opportunities that may be more favorable for individuals with certain zodiac signs in an effort to help investors connect with stocks that may align with their astrological profile.")

    st.write("By using astrological data in this way, the algorithm can help investors find stocks that more aligned with their personal strengths, preferences, and risk tolerance. This can lead to a more personalized investment strategy that takes into account not just financial considerations, but also the unique characteristics and tendencies of the investor.")

    st.write("While not everyone may believe in astrology, those who do can find value in using astrological investing algorithms as a way to deepen their understanding of themselves and the investment landscape. By using this tool to identify stocks that may resonate with their astrological profile, investors may be able to make more informed decisions that are better aligned with their personal goals and values.")

#Technologies Used Section
elif choice == "Technologies Used":
    st.header("Technologies Used")
    st.write("PyTorch was used to create the neural network (NN).")
    st.write("Architecture: 40 hidden layer, 1000 neurons each, ReLU activation and dropout after each layer.")
    st.write("- Optimizer: Adam.")
    st.write("- Loss: Mean Squared Error (MSELoss).")
    st.write("- Pandas was used for preprocessing of data.")
    st.write("- NASA’s API for planetary data.")
    st.write("- Our Stock API  was used to get the prices on a 30-minute change from the past 2 years.")
    st.write("- Streamlit for deployment of our info page. The model will not be showcased (yet).")


    st.header("Pipeline Description")
    st.write("- API called to read in data.")
    st.write("- Use Planet Transformer script to process the data.")
    st.write("- Scaling was done on the data using the Standard Normalization scaling technique. i.e., X - mean(X) / sd(X) to get it to standard normal distribution.")
    st.write("- Data was then joined on date so that there was a long format for stocks and planets.")
    st.write("- Transform data into a multi-dimensional tensor and put it through the model.")
    st.write("- Using voodoo magic and harnessing the power of the stars and aligning our chakras to the correct frequency so that our model would arrive at the optimal minimum loss.")


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
elif choice == "About Us":
    st.header("About Us")
    st.write("Ryan is a senior studying Finance at the Broad College of Business, he has interests in Data Analytics, Urban Planning and Coffee")
    st.write("Alexander is a senior pursuing a degree in data science and statistics at the College of Natural Science. His passion lies in utilizing statistical learning and data analytics to solve real-world problems such as predicting which stock to buy when Venus is retrograde")
    st.write("Mateja is a junior studying Comp Data Science at College of Engineering. His passion lies in using projects as a way to learn new things. Traveling as a real profession.")
    st.write("Vashcar is a junior studying Computer Engineering at the College of Engineering. He has interest in Machine Learning and Web Development.")
    st.write("Matthew is a junior studying computer science at the college of engineering. His hobbies are playing basketball, learning new things about cs, and reading")
    st.write("Ankan is a junior studying Comp Sci at the College of Engineering. He likes Web Development, Traveling, Animation, and Drawing.")

    



