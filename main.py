import streamlit as st
import pandas as pd
import numpy as np
import time
import yfinance as yf
import plotly.express as px


with st.sidebar:
    st.write("sidebar")
    st.slider("slider", -10, 20, 5)

    st.button("click me!", on_click=st.balloons)

    audio_file = open("audio.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="mp3")

st.header("Header")

st.camera_input("Your camera")

st.write("Here's our first attempt at using data to create a table:")
st.write(
    pd.DataFrame(
        {"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})
)


tickerSymbol = "GOOGL"
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period="1d", start="2010-05-31")

st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)

df = px.data.gapminder()
fig = px.scatter(
    df,
    x="gdpPercap",
    y="lifeExp",
    animation_frame="year",
    animation_group="country",
    size="pop",
    color="continent",
    hover_name="country",
    facet_col="continent",
    log_x=True,
    size_max=45,
    range_x=[100, 100000],
    range_y=[25, 90],
)

st.plotly_chart(fig)


df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=["lat", "lon"]
)

st.map(df)

st.code("""val x = 10""", language="scala")

st.file_uploader("drop file")


# interactive plot
@st.cache(suppress_st_warning=True)
def create_anki_df():
    return pd.DataFrame({
        'day': list(range(1, 101)),
        'repetitions': [
            60 + np.random.randint(-6 + int(i/3), 6 + int(i/3))
            for i in range(100)],
        'learned': [20 + np.random.randint(-19, 19) for _ in range(100)],
        'topic': np.random.choice(
            ['cuisine', 'buisnes', 'traveling', 'health'], 100),
        'time': list(np.abs(np.random.randn(100)*2 + 30)),
    })


df = create_anki_df()
low, high = st.slider('Set days range', 1, 100, (25, 75))

st.header('Anki repetitions')

fig = px.scatter(
    df[(df['day'] >= low) & (df['day'] <= high)], x='time', y='repetitions',
    size='learned', color='topic',
    range_x=(df.time.min() - 1, df.time.max() + 1),
    range_y=(df.repetitions.min() - 2, df.repetitions.max() + 2))
st.plotly_chart(fig)

st.dataframe(df[(df['day'] >= low) & (df['day'] <= high)])


progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.1)
    progress_bar.progress(i + 1)
