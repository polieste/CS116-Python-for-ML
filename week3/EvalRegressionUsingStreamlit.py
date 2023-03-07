from numpy import sqrt
from numpy import absolute
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import streamlit as st
import mpld3
import streamlit.components.v1 as components
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Linear Regression Model")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes and write to local disk:
    bytes_data = uploaded_file.read()
    file_path =  uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(bytes_data)
    df = pd.read_csv(file_path)
    st.markdown("## Data From Your Files")
    st.dataframe(df)
    features_full = list(df.columns)
    features = features_full[:-1]

    st.markdown("## Input Features")
    check_boxes = [st.checkbox(
        feature, key=feature) for feature in features]


    checked_features = [feature for feature, checked in zip(
        features, check_boxes) if checked]

    chosen_fe = []
    for feature in checked_features:
        chosen_fe.append(feature)

    st.markdown("## Output Feature")
    st.write(features_full[-1])

    st.markdown("## Choose Data Separation Strategy")
    strag1 = st.checkbox("Train/Test Split")
    strag2 = st.checkbox("K-Fold Cross Validation")

    if strag1:
        st.markdown("### Train/Test Split")
        ratio = st.text_input('Input Train Ratio: ')

    if strag2:
        st.markdown("### K-Fold Cross Validation")
        global k_fold
        k_fold = st.text_input('Input Value of K: ')

    if st.button('Run'):
        objList = df.select_dtypes(include="object").columns

        # Label Encoding for object to numeric conversion
        le = LabelEncoder()
        for feat in objList:
            df[feat] = le.fit_transform(df[feat].astype(str))

        y = df[features_full[-1]]
        X = df[list(chosen_fe)]

        if strag1:
            import numpy as np
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=(1-float(ratio)))

            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)

            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            st.write('MAE Score: ', mae)
            st.write('MSE Score: ', mse)

            name = ['MAE', 'MSE']
            score = [mae, mse]
            X_axis = np.arange(len(name))
            # creating the bar plot

            fig = plt.figure(figsize=(10, 5))
            plt.yscale("log")
            plt.bar(X_axis, score, color=['Red', 'Blue'],
                    width=0.4)
            plt.xticks(X_axis, name)
            plt.xlabel("MSE & MAE scores")
            plt.ylabel("Scores")
            plt.title("Loss function", fontsize=20)
            st.pyplot(fig)

        if strag2:
            # st.write(k_fold)
            k_fold = int(k_fold)
            cv = KFold(n_splits=k_fold, random_state=1, shuffle=True)

            model = LinearRegression()
            mae_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error',
                                         cv=cv, n_jobs=-1)
            mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error',
                                         cv=cv, n_jobs=-1)
            mse_scores = absolute(mse_scores)
            mae_scores = absolute(mae_scores)

            st.write('MAE Score: ', mean(absolute(mae_scores)))
            st.write('MSE Score: ', mean(absolute(mse_scores)))

            def plot_result(x_label, y_label, plot_title, train_data, val_data):
                # Set size of plot
                fig = plt.figure(figsize=(12, 6))
                labels = []
                for i in range(k_fold):
                    temp = str(i+1) + 'th Fold'
                    labels.append(temp)

                X_axis = np.arange(k_fold)
                ax = plt.gca()
                plt.yscale("log")
                plt.bar(X_axis-0.2, train_data, 0.4,
                        color='Red', label='MAE')
                plt.bar(X_axis+0.2, val_data, 0.4, color='blue', label='MSE')
                plt.title(plot_title, fontsize=20)
                plt.xticks(np.arange(k_fold), labels)
                plt.xlabel(x_label, fontsize=14)
                plt.ylabel(y_label, fontsize=14)
                plt.legend()
                plt.grid(True)
                st.pyplot(fig)

            plot_title = "MSE & MAE scores in" + str(k_fold) + "th Fold"
            with st.container():
                plot_result("MSE & MAE Scores", "Error Score",
                            "MSE & MAE scores in 10th Fold", mae_scores, mse_scores)
