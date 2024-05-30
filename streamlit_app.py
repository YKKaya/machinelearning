import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile

# Page title
st.set_page_config(page_title='Stock Performance Prediction', page_icon='📈')
st.title('📈 Stock Performance Prediction')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to build a machine learning (ML) model for predicting stock performance. This includes data fetching from Yahoo Finance, data pre-processing, ML model building, and post-model analysis.')

    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, go to the sidebar and 1. Select a stock ticker and 2. Adjust the model parameters using the various slider widgets. This will initiate the ML model building process, display the model results, and allow users to download the generated models and accompanying data.')

    st.markdown('**Under the hood**')
    st.markdown('Data sets:')
    st.code('- Stock data from Yahoo Finance', language='markdown')

    st.markdown('Libraries used:')
    st.code('''- Pandas for data wrangling
- yfinance for fetching stock data
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
    ''', language='markdown')

# Sidebar for accepting input parameters
with st.sidebar:
    st.header('1. Input Data')

    st.markdown('**1. Select stock ticker**')
    stock_ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT)", "AAPL")

    @st.cache_data
    def load_stock_data(ticker):
        # Use try-except block to handle potential errors
        try:
            data = yf.download(ticker, progress=False)
            if data.empty:
                raise ValueError("No data found for ticker")
            return data
        except Exception as e:
            st.error(f"Error fetching data for ticker {ticker}: {e}")
            return None

    df = load_stock_data(stock_ticker)

    if df is not None:
        st.header('2. Set Parameters')
        parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        with st.expander('What is data split ratio?'):
            st.write("Data split ratio determines the proportion of data that will be used for training the model versus the proportion used for testing the model. Typically, a higher percentage is allocated for training.")

        st.subheader('2.1. Learning Parameters')
        parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
        with st.expander('What is n_estimators?'):
            st.write("The number of trees in the forest. Increasing this number can improve model performance but also increases computation time.")

        parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
        with st.expander('What is max_features?'):
            st.write("The number of features to consider when looking for the best split. 'All' uses all features, 'sqrt' uses the square root of the number of features, and 'log2' uses the logarithm base 2 of the number of features.")

        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
        with st.expander('What is min_samples_split?'):
            st.write("The minimum number of samples required to split an internal node. A higher value can make the model more conservative.")

        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
        with st.expander('What is min_samples_leaf?'):
            st.write("The minimum number of samples required to be at a leaf node. This value can affect the model's ability to generalize.")

        st.subheader('2.2. General Parameters')
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
        with st.expander('What is random_state?'):
            st.write("The seed used by the random number generator. This ensures reproducibility of the model training.")

        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])
        with st.expander('What is criterion?'):
            st.write("The function to measure the quality of a split. 'squared_error' is the default for regression tasks.")

        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        with st.expander('What is bootstrap?'):
            st.write("Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.")

        parameter_oob_score = st.select_slider('Use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
        with st.expander('What is oob_score?'):
            st.write("Whether to use out-of-bag samples to estimate the R^2 on unseen data. Out-of-bag samples are the samples that are not included in the bootstrap sample for a tree.")

        sleep_time = st.slider('Sleep time', 0, 3, 0)
        with st.expander('What is sleep time?'):
            st.write("Sleep time is used to simulate processing time, giving users visual feedback on long operations. It can be set to zero for faster execution.")

# Initiate the model building process
if df is not None and not df.empty:
    with st.spinner("Running..."):
        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Preparing data ...")
        time.sleep(sleep_time)
        df = df.dropna()  # Dropping rows with missing values
        X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = df['Adj Close']

        st.write("Splitting data ...")
        time.sleep(sleep_time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - parameter_split_size) / 100, random_state=parameter_random_state)

        st.write("Model training ...")
        time.sleep(sleep_time)

        max_features = None if parameter_max_features == 'all' else parameter_max_features

        rf = RandomForestRegressor(
            n_estimators=parameter_n_estimators,
            max_features=max_features,
            min_samples_split=parameter_min_samples_split,
            min_samples_leaf=parameter_min_samples_leaf,
            random_state=parameter_random_state,
            criterion=parameter_criterion,
            bootstrap=parameter_bootstrap,
            oob_score=parameter_oob_score)
        rf.fit(X_train, y_train)

        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)

        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        st.write("Displaying performance metrics ...")
        time.sleep(sleep_time)
        parameter_criterion_string = ' '.join([x.capitalize() for x in parameter_criterion.split('_')])

        rf_results = pd.DataFrame({
            'Method': ['Random forest'],
            f'Training {parameter_criterion_string}': [train_mse],
            'Training R2': [train_r2],
            f'Test {parameter_criterion_string}': [test_mse],
            'Test R2': [test_r2]
        }).round(3)

    st.success("Model training complete!")

    # Display data info
    st.header('Input data', divider='rainbow')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="No. of samples", value=X.shape[0])
    col2.metric(label="No. of X variables", value=X.shape[1])
    col3.metric(label="No. of Training samples", value=X_train.shape[0])
    col4.metric(label="No. of Test samples", value=X_test.shape[0])

    with st.expander('Initial dataset', expanded=True):
        st.dataframe(df, height=210, use_container_width=True)
    with st.expander('Train split', expanded=False):
        train_col1, train_col2 = st.columns((3, 1))
        with train_col1:
            st.markdown('**X**')
            st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
        with train_col2:
            st.markdown('**y**')
            st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
    with st.expander('Test split', expanded=False):
        test_col1, test_col2 = st.columns((3, 1))
        with test_col1:
            st.markdown('**X**')
            st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
        with test_col2:
            st.markdown('**y**')
            st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

    # Zip dataset files
    df.to_csv('dataset.csv', index=False)
    X_train.to_csv('X_train.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    list_files = ['dataset.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
    with zipfile.ZipFile('dataset.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    with open('dataset.zip', 'rb') as datazip:
        st.download_button(
            label='Download ZIP',
            data=datazip,
            file_name="dataset.zip",
            mime="application/octet-stream"
        )

    # Display model parameters
    st.header('Model parameters', divider='rainbow')
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Data split ratio (% for Training Set)", value=parameter_split_size)
    col2.metric(label="Number of estimators (n_estimators)", value=parameter_n_estimators)
    col3.metric(label="Max features (max_features)", value='all' if max_features is None else max_features)

    # Display feature importance plot
    importances = rf.feature_importances_
    feature_names = list(X.columns)
    forest_importances = pd.Series(importances, index=feature_names)
    df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})

    bars = alt.Chart(df_importance).mark_bar(size=40).encode(
        x='value:Q',
        y=alt.Y('feature:N', sort='-x')
    ).properties(height=250)

    col1, col2 = st.columns((2, 3))
    with col1:
        st.header('Model performance', divider='rainbow')
        st.dataframe(rf_results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))
    with col2:
        st.header('Feature importance', divider='rainbow')
        st.altair_chart(bars, theme='streamlit', use_container_width=True)

    # Prediction results
    st.header('Prediction results', divider='rainbow')
    df_train = pd.DataFrame({'actual': y_train, 'predicted': y_train_pred, 'class': 'train'}).reset_index(drop=True)
    df_test = pd.DataFrame({'actual': y_test, 'predicted': y_test_pred, 'class': 'test'}).reset_index(drop=True)
    df_prediction = pd.concat([df_train, df_test], axis=0)

    col1, col2 = st.columns((2, 3))
    with col1:
        st.dataframe(df_prediction, height=320, use_container_width=True)
    with col2:
        scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
            x='actual',
            y='predicted',
            color='class'
        )
        st.altair_chart(scatter, theme='streamlit', use_container_width=True)

# Ask for valid stock ticker if none is provided
else:
    st.warning('👈 Enter a valid stock ticker to get started!')
