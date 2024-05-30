import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile

# Page configuration
st.set_page_config(page_title='ML Model Building', page_icon='🤖', layout='wide')
st.title('🤖 ML Model Building')

# Sidebar - About section
with st.sidebar.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to build a machine learning (ML) model in an end-to-end workflow. This includes data upload, data pre-processing, ML model building, and post-model analysis.')

    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, go to the sidebar and 1. Select a dataset and 2. Adjust the model parameters using the various slider widgets. This will initiate the ML model building process, display the model results, and allow users to download the generated models and accompanying data.')

    st.markdown('**Under the hood**')
    st.markdown('Data sets:')
    st.code('- Drug solubility data set', language='markdown')

    st.markdown('Libraries used:')
    st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
    ''', language='markdown')

# Sidebar - Data input
st.sidebar.header('1. Input Data')
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

@st.cache_data
def load_example_data():
    return pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

@st.cache_data
def convert_df(input_df):
    return input_df.to_csv(index=False).encode('utf-8')

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=False)
else:
    example_data = st.sidebar.checkbox('Load example data')
    if example_data:
        df = load_example_data()
    else:
        df = None

if df is not None:
    example_csv = load_example_data()
    csv = convert_df(example_csv)
    st.sidebar.download_button(
        label="Download example CSV",
        data=csv,
        file_name='delaney_solubility_with_descriptors.csv',
        mime='text/csv',
    )

    # Sidebar - Model parameters
    st.sidebar.header('2. Set Parameters')
    parameter_split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.sidebar.subheader('2.1. Learning Parameters')
    with st.sidebar.expander('See parameters'):
        parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
        parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.sidebar.subheader('2.2. General Parameters')
    with st.sidebar.expander('See parameters'):
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])
        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.select_slider('Use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

    sleep_time = st.sidebar.slider('Sleep time', 0, 3, 0)

    # Main content - Model building process
    with st.spinner("Running..."):
        st.subheader("Model Building Process")

        # Loading and preparing data
        st.markdown("### Step 1: Loading and Preparing Data")
        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Preparing data ...")
        time.sleep(sleep_time)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Splitting data
        st.markdown("### Step 2: Splitting Data")
        st.write("Splitting data ...")
        time.sleep(sleep_time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - parameter_split_size) / 100, random_state=parameter_random_state)

        # Training model
        st.markdown("### Step 3: Training Model")
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

        # Making predictions
        st.markdown("### Step 4: Making Predictions")
        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)

        # Evaluating performance
        st.markdown("### Step 5: Evaluating Performance")
        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Displaying results
        st.markdown("### Step 6: Displaying Results")
        parameter_criterion_string = ' '.join([x.capitalize() for x in parameter_criterion.split('_')])
        rf_results = pd.DataFrame({
            'Metric': ['Train MSE', 'Train R2', 'Test MSE', 'Test R2'],
            'Value': [train_mse, train_r2, test_mse, test_r2]
        }).round(3)
        st.dataframe(rf_results)

    # Data information
    st.markdown("### Input Data Information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("No. of samples", X.shape[0])
    col2.metric("No. of features", X.shape[1])
    col3.metric("Training samples", X_train.shape[0])
    col4.metric("Test samples", X_test.shape[0])

    with st.expander("Initial Dataset", expanded=True):
        st.dataframe(df)

    with st.expander("Train/Test Split"):
        train_col1, train_col2 = st.columns((3, 1))
        with train_col1:
            st.subheader("Train X")
            st.dataframe(X_train)
        with train_col2:
            st.subheader("Train y")
            st.dataframe(y_train)

        test_col1, test_col2 = st.columns((3, 1))
        with test_col1:
            st.subheader("Test X")
            st.dataframe(X_test)
        with test_col2:
            st.subheader("Test y")
            st.dataframe(y_test)

    # Zipping dataset files
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

    # Model parameters
    st.markdown("### Model Parameters")
    col1, col2, col3 = st.columns(3)
    col1.metric("Data split ratio", parameter_split_size)
    col2.metric("Number of estimators", parameter_n_estimators)
    col3.metric("Max features", 'all' if max_features is None else max_features)

    # Feature importance plot
    importances = rf.feature_importances_
    feature_names = list(X.columns)
    forest_importances = pd.Series(importances, index=feature_names)
    df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'importance'})

    bars = alt.Chart(df_importance).mark_bar(size=40).encode(
        x='importance:Q',
        y=alt.Y('feature:N', sort='-x')
    ).properties(height=400)

    col1, col2 = st.columns((1, 2))
    with col1:
        st.markdown("### Model Performance")
        st.dataframe(rf_results)
    with col2:
        st.markdown("### Feature Importance")
        st.altair_chart(bars, use_container_width=True)

    # Prediction results
    st.markdown("### Prediction Results")
    df_train = pd.DataFrame({'actual': y_train, 'predicted': y_train_pred, 'set': 'train'})
    df_test = pd.DataFrame({'actual': y_test, 'predicted': y_test_pred, 'set': 'test'})
    df_prediction = pd.concat([df_train, df_test])

    scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
        x='actual',
        y='predicted',
        color='set'
    )

    col1, col2 = st.columns((1, 2))
    with col1:
        st.dataframe(df_prediction)
    with col2:
        st.altair_chart(scatter, use_container_width=True)

else:
    st.warning('👈 Upload a CSV file or check *"Load example data"* to get started!')
