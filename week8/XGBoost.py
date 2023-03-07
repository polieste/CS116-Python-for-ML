import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model,svm,tree
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,log_loss
import math
standard_scaler = StandardScaler()
le = preprocessing.LabelEncoder()




uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file,sep=',')#note
    cols=dataframe.columns
    st.write(dataframe)




st.write("Input features")


data_dict={}
kiemtra_IO=True
if uploaded_file==None:
    st.write("non_data")
else:
    lst_input=[]
    for col_name in cols:
        A=st.checkbox(col_name) #create checkbox
        if A:
            data_dict[col_name]=dataframe[col_name]
            lst_input.append(col_name)



  
    st.write("output:")
    #add output:
    for col_name in cols:
        B = st.checkbox("output: "+col_name)
        danhdau=0
        if B:  
            #check if output==input
            for c in lst_input:
                if col_name==c:
                    danhdau=1
            if danhdau==0:
                data_dict[col_name] = dataframe[col_name]
            else:
                kiemtra_IO=False

    data=pd.DataFrame(data_dict)
    st.write("DATA being selected")
    st.write(data)



#normalize data:

    cols_=data.columns
    if len(cols_)>1:#choice more than 1 column
        for col_name in cols_[:-1]:

            data[[col_name]]=standard_scaler.fit_transform(data[[col_name]])
        data[cols_[-1]]=le.fit_transform(data[cols_[-1]])
        st.write("Normalized data")
        st.write(data)

        X = data.loc[:, [c for c in cols_[:-1]]]
        y = data[cols_[-1]]

        traintest_split=st.checkbox("Train - Test Split")

        if traintest_split:

            #st.write("X:")
            #st.write(X)
            #st.write("y:")
            #st.write(y)
            title = st.text_input( 'ratio',0.33)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(title), random_state=42)
        
            names=[f'total:{X.shape[0]}',f'train:{len(X_train[cols_[0]])}',f'test:{len(X_test[cols_[0]])}']
            values=[len(X[cols_[0]]),len(X_train[cols_[0]]),len(X_test[cols_[0]])]
            fig, ax = plt.subplots()
            ax=plt.bar(names,values,color=["r","g","b"])
            st.pyplot(fig)

            run=st.button("run")
            if run and len(data.columns)!=1 and kiemtra_IO==True:
                # train
                st.write("TRAIN:")
                logistic_reg = linear_model.LogisticRegression()
                hist = logistic_reg.fit(X_train, y_train)
                #y_pred=logistic_reg.predict_proba(X_test)
                y_pred = logistic_reg.predict(X_test)
                y_pred_proba=logistic_reg.predict_proba(X_test)
                #st.write("predict:")
                #st.write(np.array(y_pred))
                #st.write("labels truths:")
                #st.write(y_test,)
                precision=precision_score(y_true=y_test,y_pred=y_pred)
                recall=recall_score(y_true=y_test,y_pred=y_pred)
                f1_score=f1_score(y_true=y_test,y_pred=y_pred)
                log_loss=log_loss(y_true=y_test,y_pred=y_pred_proba[:,1])
                confusion_mt=confusion_matrix(y_true=y_test,y_pred=y_pred)
                #f1_score=2*precision*recall/(precision+recall)
                st.write(f"precision,recall,f1_score,log_loss:{precision,recall,f1_score,log_loss}")
                st.write("confusion_matrix",confusion_mt)
                #log_loss1 = (-(y_test * np.log(y_pred_proba[:, 1]) + (1 - y_test) * np.log(1 - y_pred_proba[:, 1]))).sum()/len(y_test)
                #st.write(log_loss1)

                names=[f"precision:{round(precision,2)}",f"recall:{round(recall,2)}",f"f1_score{round(f1_score,2)}",f"log_loss:{round(log_loss,2)}"]
                pos=np.arange(len(names))
                values=[precision,recall,f1_score,log_loss]

                fig, ax = plt.subplots()
                ax = plt.xticks(pos, names)
                ax = plt.bar(pos, values, width=0.4,color=['red','blue','green','yellow'])  # vẽ biểu đồ cột
                ax = plt.title("Evaluate")
                ax = plt.legend()
                st.pyplot(fig)





        k_fold = st.checkbox("k_fold")
        if k_fold:

            k = st.text_input("k=", 4)
            #K-Fold CV
            kfold = KFold(n_splits=int(k), shuffle=True)

            # K-fold Cross Validation model evaluation
            fold_idx = 1

            logistic_reg = linear_model.LogisticRegression()
            SVM=svm.SVC()
            decision_tree=tree.DecisionTreeClassifier()
            xg_boost=XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')






            for train_ids, val_ids in kfold.split(X, y):
                st.write("Start train Fold ", fold_idx)

                # Train model
                xtrain=X.loc[[id for id in train_ids],:]
                ytrain=y.loc[[id for id in train_ids]]
                xtext=X.loc[[id for id in val_ids],:]
                ytest=y.loc[[id for id in val_ids]]

                hist_logis_reg = logistic_reg.fit( xtrain,ytrain)
                hist_svm=SVM.fit(xtrain,ytrain)
                hist_de_tree=decision_tree.fit(xtrain,ytrain)
                hist_xgboost=xg_boost.fit(xtrain,ytrain)

                # Test
                y_pred_logis_reg = logistic_reg.predict(xtext)
                y_pred_svm = SVM.predict(xtext)
                y_pred_decisiontree = decision_tree.predict(xtext)
                y_pred_xgboost = xg_boost.predict(xtext)


                f1_logis_reg = f1_score(y_true=ytest, y_pred=y_pred_logis_reg)
                f1_svm= f1_score(y_true=ytest, y_pred=y_pred_svm)
                f1_decisiontree = f1_score(y_true=ytest, y_pred=y_pred_decisiontree)
                f1_xgboost = f1_score(y_true=ytest, y_pred=y_pred_xgboost)


                
                lst_names = ["logistic","SVM","decision tree","xg boost"]
                pos = np.arange(len(lst_names))
                lst_f1 = [f1_logis_reg,f1_svm,f1_decisiontree,f1_xgboost]

                st.write("f1 score of logistic regression:", f1_logis_reg)
                st.write("f1 score of svm:", f1_svm)
                st.write("f1 score of decision tree:", f1_decisiontree)
                st.write("f1 score of xgboost:", f1_xgboost)

                fig, ax = plt.subplots()
                ax = plt.xticks(pos, lst_names)
                ax = plt.bar(pos, lst_f1, width=0.2, label="f1 score")  # plot

                ax = plt.title("Evaluate")
                ax = plt.legend()
                st.pyplot(fig)



                fold_idx = fold_idx + 1






