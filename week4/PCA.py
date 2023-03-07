from tkinter import HORIZONTAL
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from io import StringIO
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
import altair as alt
import matplotlib.pyplot as plt
st.markdown("# Classification Model with PCA")

dataframe = ''
output = 'E'
features = ['A', 'B']

X = ['.', '.']
y = ['.', '.']

is_uploaded = True
if is_uploaded:
    # Can be used wherever a "file-like" object is accepted:
    wine = load_wine(as_frame=True)
    dataframe = pd.DataFrame(data=np.c_[wine['data'], wine['target']],
                             columns=wine['feature_names'] + ['target'])
    st.write(dataframe)
    # st.write(dataframe.head())
    features = list(dataframe.keys())[0:-1]
    output = list(dataframe.keys())[-1]
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]

selected_features = wine['feature_names']

ns_components = [*range(1, len(selected_features)+1)]
# pca = PCA(n_components=n_components)

Split_option = st.radio('Select a data splitting method',
                        ("Train/Test split", "KFold"))

if Split_option == "KFold" and is_uploaded:
    number_fold = int(st.number_input(
        'K', value=4, min_value=1, max_value=len(X)))
else:
    train_radio = st.number_input(
        'Train radio', value=0.8, min_value=0.5, max_value=0.9)


if st.button('Run'):
    precisions = []
    recalls = []
    f1s = []

    for n_components in ns_components:
        pca = PCA(n_components=n_components)
        if selected_features is not None:
            remove_features = []
            add_features = []
            for selected_feature in selected_features:
                try:
                    float(X.loc[:, selected_feature][0])
                except ValueError:
                    one_hot = pd.get_dummies(X[selected_feature])
                    X = X.drop(selected_feature, axis=1)
                    X = X.join(one_hot)

                    for key in one_hot.keys():
                        add_features.append(key)
                    remove_features.append(selected_feature)

            for feature in remove_features:
                selected_features.remove(feature)
            for feature in add_features:
                selected_features.append(feature)

            if Split_option == "KFold":

                kfold = StratifiedKFold(
                    n_splits=number_fold, shuffle=True, random_state=1)
                result = []
                fold = 1
                for train_index, test_index in kfold.split(X, y):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    X_train = X_train.loc[:, selected_features]
                    pca.fit(X_train)
                    X_train = pca.transform(X_train)
                    X_test = X_test.loc[:, selected_features]
                    X_test = pca.transform(X_test)

                    reg = LogisticRegression().fit(X_train, y_train)
                    y_predict = reg.predict(X_test)
                    precision = precision_score(
                        y_test, y_predict, average='weighted')
                    recall = recall_score(
                        y_test, y_predict, average='weighted')
                    f1 = f1_score(y_test, y_predict, average='weighted')
                    result.append((precision, recall, f1))
                    fold += 1

                data = pd.DataFrame(
                    result, columns=['Precision', 'Recall', 'F1-score'])

                precision = data['Precision']
                recall = data['Recall']
                f1 = data['F1-score']

                ind = np.arange(len(precision)) # the x locations for the groups
                width = 0.2 # the width of the bars
                label = ()
                for i in ind:
                    s = 'Fold' + str(i + 1)
                    label += (s,)

                fig, ax = plt.subplots()
                rects1 = ax.bar(ind - width/2 - width, precision, width, label='Precision')
                rects2 = ax.bar(ind, recall, width, label='Recall')
                rects3 = ax.bar(ind + width/2 + width, f1, width, label='F1-score')

                ax.set_ylabel('Precision, recall and f1 scores')
                ax.set_xlabel('KFold')
                ax.set_title('Score')
                ax.set_xticks(ind)
                ax.set_xticklabels(label)

                ax.legend()

                st.pyplot(plt)
                st.write('Recall score trung bình: ', recall.mean())
                st.write('F1-score trung bình: ', f1.mean())

                precisions.append(precision.mean())
                recalls.append(recall.mean())
                f1s.append(f1.mean())

            else:

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=train_radio, random_state=1, stratify=y)
                X_train = X_train.loc[:, selected_features]
                X_test = X_test.loc[:, selected_features]
                pca.fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)
                reg = LogisticRegression().fit(X_train, y_train)
                y_predict = reg.predict(X_test)
                precision = precision_score(
                    y_test, y_predict, average='weighted')
                recall = recall_score(y_test, y_predict, average='weighted')
                f1 = f1_score(y_test, y_predict, average='weighted')
                result = [(precision, recall, f1)]

                data = pd.DataFrame(
                    result, columns=['Precision', 'Recall', 'F1-score'])

                precision = data['Precision']
                recall = data['Recall']
                f1 = data['F1-score']
                precisions.append(precision.mean())
                recalls.append(recall.mean())
                f1s.append(f1.mean())
        else:
            st.write("Please select feature(s)")

    st.markdown('### Precision over PCA number')
    chart_data = pd.DataFrame(
        np.array(precisions), columns=["Score"])
    st.bar_chart(chart_data)
    st.write("Số chiều tối ưu: ", np.array(precisions).argmax()+1)

    st.markdown('### Recall over PCA number')
    chart_data1 = pd.DataFrame(
        np.array(recalls), columns=["Score"])
    st.bar_chart(chart_data1)
    st.write("Số chiều tối ưu: ", np.array(recalls).argmax()+1)

    st.markdown('### F1-score over PCA number')
    chart_data2 = pd.DataFrame(
        np.array(f1s),  columns=["Score"])
    st.bar_chart(chart_data2)
    st.write("Số chiều tối ưu: ", np.array(f1s).argmax()+1)
