import streamlit as st
import pandas as pd
import numpy as np
# import time
import yfinance as yf
import plotly.express as px
from annotated_text import annotated_text
from sklearn.svm import SVC
import plotly.graph_objs as go
# from plotly import tools
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt


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


st.header('NER')


@st.cache(suppress_st_warning=True)
def load_NER_result():
    return [
        "This ",
        ("is", "verb", "#8ef"),
        " some ",
        ("annotated", "adj", "#faa"),
        ("text", "noun", "#afa"),
        " for those of ",
        ("you", "pronoun", "#fea"),
        " who ",
        ("like", "verb", "#8ef"),
        " this sort of ",
        ("thing", "noun", "#afa"),
        "."
    ]


def select_entities_description(ner_res: list, ner_classes: list):
    res = [None] * len(ner_res)
    for i in range(len(ner_res)):
        if isinstance(ner_res[i], tuple) and ner_res[i][1] not in ner_classes:
            res[i] = ner_res[i][0]
        else:
            res[i] = ner_res[i]
    return res


ner_res = load_NER_result()
ner_classes = st.multiselect(
    'Select entities types', ['verb', 'adj', 'noun', 'pronoun'])

annotated_text(
    *select_entities_description(ner_res, ner_classes)
)


st.header('SVM example')


@st.cache(suppress_st_warning=True)
def load_nonlinear():
    n_per_cls = 201

    start, stop = 1, 3
    range_ = stop - start
    x1 = np.linspace(start, stop, n_per_cls)
    y1 = np.sin((x1 - start) * np.pi / range_) - 0.25 \
        + np.random.randn(n_per_cls) * 0.2

    start, stop = 2, 4
    range_ = stop - start
    x2 = np.linspace(start, stop, n_per_cls)
    y2 = np.sin((x2 - start) * np.pi / range_) * (-1) + 0.25 \
        + np.random.randn(n_per_cls) * 0.2

    cls_ = ['a'] * n_per_cls + ['b'] * n_per_cls

    return pd.DataFrame({
        'x': np.concatenate((x1, x2)).tolist(),
        'y': np.concatenate((y1, y2)).tolist(), 'cls_': cls_
    })


df = load_nonlinear()
train, test = train_test_split(df, test_size=0.2)
X_train, y_train = train[['x', 'y']].to_numpy(), train['cls_'].to_numpy()
X_test, y_test = test[['x', 'y']].to_numpy(), test['cls_'].to_numpy()

clf = SVC(gamma=2, C=1, probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
st.subheader('SVM scores')
st.write('Accuracy: %f' % accuracy_score(y_test, y_pred))
st.write('Precision: %f' % precision_score(
    y_test, y_pred, average="binary", pos_label="a"))
st.write('Recall: %f' % recall_score(
    y_test, y_pred, average="binary", pos_label="a"))
st.write('F1-score: %f' % f1_score(
    y_test, y_pred, average="binary", pos_label="a"))

fig, ax = plt.subplots()
plot_confusion_matrix(clf, X_test, y_test, ax=ax)
st.pyplot(fig)
# plot_confusion_matrix

x_min, x_max = df['x'].min() - 1, df['x'].max() + 1
y_min, y_max = df['y'].min() - 1, df['y'].max() + 1
h = 1000

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
y_ = np.linspace(y_min, y_max, h)

Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, :1]
Z = Z.reshape(xx.shape)
fig = make_subplots(rows=1, cols=1)

trace1 = go.Heatmap(
    x=xx[0], y=y_, z=Z,
    colorscale='Jet',
    showscale=False)

trace2 = go.Scatter(
    x=X_test[:, 0], y=X_test[:, 1],
    mode='markers',
    showlegend=False,
    marker=dict(size=10,
                color=[1 if val == 'a' else 0 for val in y_test],
                colorscale='Jet',
                line=dict(color='black', width=1))
    )

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)

st.plotly_chart(fig)


# progress_bar = st.progress(0)
# for i in range(100):
#     time.sleep(0.1)
#     progress_bar.progress(i + 1)
