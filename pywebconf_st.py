"""
Purpose:
    Start Python Web Conf UI
"""

# Python imports
import random
from typing import Type, Union, Dict, Any, List, Tuple

# 3rd party imports
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
from wordcloud import WordCloud
from yellowbrick.classifier import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def sidebar() -> None:
    """
    Purpose:
        Shows the side bar
    Args:
        N/A
    Returns:
        N/A
    """

    st.sidebar.title("Python Web Conf")

    # Create the Navigation Section
    st.sidebar.image(
        "https://2022.pythonwebconf.com/python-web-conference-2022/@@images/logo_image"
    )

    pages = ["Home", "Playground", "Schedule", "Team"]
    default_page = 0
    page = st.sidebar.selectbox("Go To", options=pages, index=default_page)

    if page == "Home":
        home_page()
    elif page == "Playground":
        playground_page()
    elif page == "Schedule":
        schedule_page()
    elif page == "Team":
        team_page()
    else:
        st.error("Invalid Page")


def app() -> None:
    """
    Purpose:
        Controls the app flow
    Args:
        N/A
    Returns:
        N/A
    """

    # Spin up the sidebar, will control which page is loaded in the
    # main app
    sidebar()


def data_prep(df: pd.DataFrame) -> Tuple[List, List, List, List]:
    """
    Purpose:
        Prep data for modeling
    Args:
        df - Pandas dataframe
    Returns:
        test_features - test set features
        train_features - train set feautres
        test_target -  test set target
        train_target - train set target
    """
    # Specify the target classes
    target_string = st.selectbox("Select Target Column", df.columns)
    target = np.array(df[target_string])

    # Select Features you want
    feature_cols = st.multiselect("Select Modeling Features", df.columns)

    # Get all features
    features = df[feature_cols]
    featurestmp = np.array(features)
    feats = []
    # find all bad rows
    for index, featarr in enumerate(featurestmp):
        try:
            featarr = featarr.astype(float)
            feats.append(featarr)
        except Exception as error:

            st.error(error)
            st.error(featarr)
            st.stop()

    featuresarr = np.array(feats)

    # Split Data
    randInt = random.randint(1, 200)

    (
        test_features,
        train_features,
        test_target,
        train_target,
    ) = train_test_split(featuresarr, target, test_size=0.75, random_state=randInt)

    return (
        test_features,
        train_features,
        test_target,
        train_target,
    )


def show_classification_report(
    df: pd.DataFrame,
) -> None:
    """
    Purpose:
        Renders a classification_report
    Args:
        df - Pandas dataframe
    Returns:
        N/A
    """

    # Prep data for model training
    (
        test_features,
        train_features,
        test_target,
        train_target,
    ) = data_prep(df)

    if st.button("Train Model"):

        st.header("Classification Report")

        st.markdown(
            "The classification report visualizer displays the precision, recall, F1, and support scores for the model. In order to support easier interpretation and problem detection, the report integrates numerical scores with a color-coded heatmap. All heatmaps are in the range (0.0, 1.0) to facilitate easy comparison of classification models across different classification reports."
        )

        # Instantiate the visualizer
        visualizer = classification_report(
            GaussianNB(),
            train_features,
            train_target,
            test_features,
            test_target,
            support=True,
        )

        # Get the viz
        fig = visualizer.fig
        ax = visualizer.show()
        fig.axes.append(ax)

        # show the viz
        st.write(fig)


def gen_wordcloud(df: pd.DataFrame, repeat: bool) -> None:
    """
    Purpose:
        Generate Word Cloud from Column
    Args:
        df - Pandas dataframe
    Returns:
        N/A
    """

    # List of all non-numeric fields of given dataframe
    non_num_cols = df.select_dtypes(include=object).columns
    # selected column
    column = st.selectbox("Column", non_num_cols)
    column = df[column]
    # generate word cloud image from unique values of selected non-numeric field
    wc = WordCloud(
        max_font_size=25, background_color="white", repeat=repeat, height=500, width=800
    ).generate(" ".join(column.unique()))
    # Display the generated image:
    st.image(wc.to_image())


def bar_chart(
    df: pd.DataFrame,
):
    """
    Purpose:
        Renders bar chart
    Args:
        df - Pandas dataframe
    Returns:
        N/A
    """

    # Bar Chart Example
    x_col = st.selectbox("Select x axis for bar chart", df.columns)
    xcol_string = x_col + ":O"
    if st.checkbox("Show as continuous?", key="bar_chart_x_is_cont"):
        xcol_string = x_col + ":Q"
    y_col = st.selectbox("Select y axis for bar chart", df.columns)
    z_col = st.selectbox("Select z axis for bar chart", df.columns)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(x=xcol_string, y=y_col, color=z_col, tooltip=list(df.columns))
        .interactive()
        .properties(title="Bar Chart for " + x_col + "," + y_col)
        .configure_title(
            fontSize=20,
        )
        .configure_axis(labelFontSize=20, titleFontSize=20)
        .configure_legend(labelFontSize=20, titleFontSize=20)
    )

    st.altair_chart(chart, use_container_width=True)


def show_metrics(df: pd.DataFrame) -> None:
    """
    Purpose:
        Render mean,max,min,std,count metrics of numeric fields
    Args:
        df - Pandas dataframe
    Returns:
        N/A
    """

    # List of all numeric fields of given dataframe
    columns = df.select_dtypes(include="number").columns
    # selected column
    column = st.selectbox("Column", options=columns)
    column = df[column]
    # Rendering metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean", round(column.mean(), 2))
    col2.metric("Max", column.max())
    col3.metric("Min", column.min())
    col4.metric("Std", round(column.std(), 2))
    col5.metric("Count", int(column.count()))


def playground_page():
    """
    Purpose:
        Render playground page
    Args:
        N/A
    Returns:
        N/A
    """
    st.header("Playground")
    df = pd.read_csv("data/iris.csv")

    bar_chart(df)

    st.subheader("Metrics")
    show_metrics(df)

    talk_df = pd.read_csv("data/talks.csv")

    st.subheader("WordCloud")
    repeat = st.checkbox("Repeat words?")
    gen_wordcloud(talk_df, repeat)

    st.subheader("Classification Report")
    show_classification_report(df)


def write_talk_data(datum, col):
    """
    Purpose:
        Render schedule data
    Args:
        datum - data
        col - column to write
    Returns:
        N/A
    """
    col.write(datum["title"])
    col.write(datum["speaker"])


def schedule_page():
    """
    Purpose:
        Render schedule page
    Args:
        N/A
    Returns:
        N/A
    """
    st.header("Schedule")

    talk_data = pd.read_csv("data/talks.csv")
    with st.expander("Monday,March 21,2022"):

        datum = talk_data.iloc[0]

        col1, col2 = st.columns([1, 3])

        col1.write(datum["time"])
        col2.header("KEYNOTE")
        col2.subheader(datum["title"])
        col2.write(datum["speaker"])

    with st.expander("Tuesday,March 22,2022"):

        datum = talk_data.iloc[2]

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        # Header rows
        col1.write("TIME(US EDT/UTC-4)")
        col2.write("APP DEV 1")
        col3.write("APP DEV 2")
        col4.write("CLOUD")
        col5.write("CULTURE")
        col6.write("PYDATA")
        col7.write("TUTORIALS")

        # Data Rows
        col1.write(datum["time"])
        write_talk_data(datum, col2)

        datum = talk_data.iloc[3]
        write_talk_data(datum, col3)

        datum = talk_data.iloc[4]
        write_talk_data(datum, col4)

        datum = talk_data.iloc[5]
        write_talk_data(datum, col5)

        datum = talk_data.iloc[6]
        write_talk_data(datum, col6)

        datum = talk_data.iloc[7]
        write_talk_data(datum, col7)


def render_team_member(datum):
    """
    Purpose:
        Render team members
    Args:
        N/A
    Returns:
        N/A
    """
    st.image(datum["picture"])
    st.write(datum["name"])
    st.write(datum["title"])
    st.markdown(f"[Linkedin]({datum['linkedin']})")
    st.markdown(f"[Twitter]({datum['twitter']})")


def team_page():
    """
    Purpose:
        Show team page
    Args:
        N/A
    Returns:
        N/A
    """
    st.header("Meet the Team")

    st.subheader(
        "Meet the Sixie team behind the 4th annual 2022 Python Web Conference:"
    )
    team_data = pd.read_csv("data/team.csv")
    # st.write(team_data)

    col1, col2, col3 = st.columns(3)

    with col1:
        datum = team_data.iloc[0]
        render_team_member(datum)

        datum = team_data.iloc[3]
        render_team_member(datum)

    with col2:
        datum = team_data.iloc[1]
        render_team_member(datum)

        datum = team_data.iloc[4]
        render_team_member(datum)

    with col3:
        datum = team_data.iloc[2]
        render_team_member(datum)


def home_page():
    """
    Purpose:
        Show home page
    Args:
        N/A
    Returns:
        N/A
    """
    with st.echo(code_location="below"):
        st.title("Python Web Conf")
        st.subheader("The most in-depth Python conference for web developers")
        st.image(
            "https://2022.pythonwebconf.com/python-web-conference-2022/@@images/logo_image"
        )

        st.write("https://2022.pythonwebconf.com/")


def main() -> None:
    """
    Purpose:
        Controls the flow of the streamlit app
    Args:
        N/A
    Returns:
        N/A
    """

    # Start the streamlit app
    app()


if __name__ == "__main__":
    main()
