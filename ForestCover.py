# ***********************************************************************************************
# -*- coding: utf-8 -*-
#
# Title: Forest Cover Type Prediction
# Course & Group : PGP in AI & ML - AIML2020 Cohort 4, Group 1
# Subject: Capstone Project - PCAMZC321
# Description: This is the python script used to start the ForestCover GUI Web Application
#
# Lead Instructor:
# Mr. Satyaki Dasgupta
#
# Student Names :
## Chandresh Khaneja | 2020AIML062
## Saurabh Gupta     | 2020AIML065
## Sudheendran T L   | 2020AIML003
## Sudhir Valluri    | 2020AIML001
#
# ***********************************************************************************************
#
# Steps to start this GUI
# 1) Pre-requisite : Make sure to install streamlit python module using "pip install streamlit"
# 2) Open anaconda command prompt or anaconda power shell
# 3) Change the current working directory to the path where this ForestCover.py file is located
# 4) Launch the GUI using "streamlit run ForestCover.py" command
#
# ***********************************************************************************************
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st

from covtype import createBinnedData, createCatData, createTargetEncData, optimizeDataFrameSize
from covtype import minMaxScaleData, performIncrementalPCA, performSMOTETomek, removeLOFOutliers, stdScaleData
from os.path import exists
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def DisplayStatistics(data):
	st.write("The first few rows...")
	st.write(data.head())
	st.write("The last few rows...")
	st.write(data.tail())
	st.write("The shape of the data...")
	st.write(data.shape)
	st.write("Are there any missing values in the data?")
	st.write(data.isnull().values.any())
	st.write("Are there any duplicate observations in the data?")
	st.write(data.duplicated().values.any())
	st.write("Some statistics about the data...")
	st.write(data.describe())

def OnFileUpload(uploaded_file, org_cols):
	with st.spinner('Please wait...'):
		data = pd.read_csv(uploaded_file , index_col=False, names=org_cols)
	st.success('Here is some information about the uploaded data')
	return data

def OnPreProcessing(actions, data, cols, ncols, xcols, ycols):
	ydata = data[ycols[0]]
	with st.spinner('Please wait...'):
		for i in range(len(actions)):
			if ( "Cat" in actions[i] ):
				data = createCatData(data, cols)
				st.success('Created Categorical Data Successfully...')
			if ( "Target" in actions[i] ):
				data = createTargetEncData(data, cols, ycols[0])
				st.success('Created Target Encoded Data Successfully...')
			if ( "Outliers" in actions[i] ):
				data, ydata, removed = removeLOFOutliers(data.iloc[:,:-1], data.iloc[:,data.shape[1]-1], ycols[0])
				data[ycols[0]]=ydata
				st.success('Renoved Outliers Successfully...')
			if ( "SMOTE" in actions[i] ):
				data, ydata= performSMOTETomek(data.iloc[:,:-1], data.iloc[:,data.shape[1]-1], xcols, ycols)
				data[ycols[0]]=ydata
				st.success('Performed SMOTE Successfully...')
			if ( "BINNING" in actions[i] ):
				data = createBinnedData(data.iloc[:,:-1], ncols)
				data[ycols[0]]=ydata
				st.success('BINNED Data Successfully...')
			if ( "Max" in actions[i] ):
				data = minMaxScaleData(data.iloc[:,:-1], xcols)
				data[ycols[0]]=ydata
				st.success('Scaled Using Min-Max Scaler Successfully...')
			if ( "Standard" in actions[i] ):
				data = stdScaleData(data.iloc[:,:-1], xcols)
				data[ycols[0]]=ydata
				st.success('Scaled Using Standard Scaler Successfully...')
			if ( "PCA" in actions[i] ):
				data = performIncrementalPCA(data.iloc[:,:-1])
				data[ycols[0]]=ydata
				st.success('Performed PCA Successfully...')
	return data

def OnPredict(model_name, data, ycols):

	file_name = model_name+'.sav'
	if ( exists(file_name) != True ) :
		st.error("The model {} is not available. Try a different model!".format(file_name))
		return

	with st.spinner('Please wait...'):
		model = pickle.load(open(file_name, 'rb'))
		xtest = data.drop(columns=ycols)
		ytest = data[ycols[0]]
		ypred = model.predict(xtest)

		st.success('Prediction Completed Successfully. Model used was : '+str(model_name))
		st.write('Accuracy metrcs on the prediction made...')
		st.write(f"{100*accuracy_score(ytest, ypred):.2f}%")

		st.write('Classification Report on the prediction made...')
		cr = classification_report(ytest, ypred, output_dict=True)
		df = pd.DataFrame(cr).transpose()
		st.write(df.head(7))

		st.write("Confusion Matrix of the prediction made...")
		cm = confusion_matrix(ytest, ypred)
		st.write(cm)
		st.write("Confusion Matrix - Graphical")

		f, ax = plt.subplots(figsize=(15,15))
		sns.set(font_scale=1.25)
		st.write(sns.heatmap(cm, square=True, annot=True, fmt="d", cmap="RdYlGn"))
		st.pyplot(f)

		f, ax = plt.subplots(figsize=(15,15))
		sns.set(font_scale=1.25)
		st.write(sns.heatmap(cm/np.sum(cm), square=True, annot=True, fmt='.2%', cmap='RdYlGn'))
		st.pyplot(f)
		st.balloons()

	return

def main():

	actions_list = ['Perform Cat. Encoding', 'Perform Target Encoding', 'Remove Outliers', 'Perform SMOTE',
				 'Perform BINNING', 'Perform Min-Max Scaling', 'Perform Standard Scaling', 'Perform PCA']

	model_list = ['ComplementNB','DecisionTreeClassifier','ExtraTreesClassifier', 'GaussianNB',
			   'KNeighborsClassifier','LGBMClassifier','LinearDiscriminantAnalysis', 'LinearSVC',
			   'LogisticRegression','MLPClassifier','MultinomialNB','QuadraticDiscriminantAnalysis',
			   'RandomForestClassifier','RidgeClassifier','RidgeClassifierCV','SGDClassifier', 'VotingClassifier', 'XGBClassifier']

	model_dict = { 'ComplementNB':'CNB', 'DecisionTreeClassifier':'DTC', 'ExtraTreesClassifier':'ETC',
			   'GaussianNB':'GNB', 'KNeighborsClassifier':'kNN', 'LGBMClassifier':'LGBM',
			   'LinearDiscriminantAnalysis':'LDA', 'LinearSVC':'LSVC', 'LogisticRegression':'LR',
			   'MLPClassifier':'MLP', 'MultinomialNB':'MNB', 'QuadraticDiscriminantAnalysis':'QDA',
			   'RandomForestClassifier':'RFC', 'RidgeClassifier':'RIDGE', 'RidgeClassifierCV':'RCV',
			   'SGDClassifier':'SGD', 'VotingClassifier':'V-SOFT111-ETC-kNN-LGBM' , 'XGBClassifier':'XGB'}

	dname_list= ["Original Data", "Target Enoded Data", "Data with outliers removed", "Target Class Balanced Data",
			   "MinMax Scaled Data", "Standard Scaled Data", "Data with Numerical Columns Binned", "PCA Reduced Data"]

	dname_dict = {"Original Data":"Org Data", "Target Enoded Data":"Target Enc Data",
			   "Data with outliers removed":"No Outliers Data", "Target Class Balanced Data":"Balanced Data",
			   "MinMax Scaled Data":"MinMax Scaled Data", "Standard Scaled Data":"Std Scaled Data",
			   "Data with Numerical Columns Binned":"Binned Data", "PCA Reduced Data":"PCA Data"}

	org_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am',  'Hillshade_Noon', 'Hillshade_3pm',
		'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6',
		'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19',
		'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32',
		'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39', 'Soil_Type_40', 'Cover_Type']
	ncols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
			  'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
			  'Horizontal_Distance_To_Fire_Points']
	ccols = ['Soil_Type', 'Wilderness_Area']
	ycols = ['Cover_Type']
	xcols = ncols + ccols
	cols = xcols + ycols

	st.markdown('<h1 style=text-align:center>Forest Cover Type Prediction</center>', unsafe_allow_html=True)
	st.markdown('<h2 style=text-align:center>PGP in AI & ML - AIML2020 Cohort 4, Group 1</center>', unsafe_allow_html=True)
	st.markdown('<h2 style=text-align:center>Capstone Project - PCAMZC321</center>', unsafe_allow_html=True)
	st.image('covtype.jpg', width=700)

	# Sidebar - Header
	st.sidebar.header("Forest Cover Type Prediction")

	# Sidebar - Upload Data File
	uploaded_file = st.sidebar.file_uploader("Upload Your .csv File", type='csv')
	if ( uploaded_file is not None ):
		data = OnFileUpload(uploaded_file, org_cols)
		DisplayStatistics(optimizeDataFrameSize(data))
	else: return

	# Sidebar -  Pre-processing multi-select widget & button
	preprocess_actions = st.sidebar.multiselect('Select Pre-Processing Needed', actions_list, [actions_list[0]])
	if ( len(preprocess_actions) ):
		if st.sidebar.button('Perform Pre-processing'):
			data = OnPreProcessing(preprocess_actions, data, cols, ncols, xcols, ycols)
			DisplayStatistics(optimizeDataFrameSize(data))

	# Sidebar -  Model & Dataset Select widgets & Predict button
	model = st.sidebar.selectbox('Select Model', model_list)
	dataset = st.sidebar.selectbox('Select Dataset', dname_list)
	if ( ( model is not None ) and ( dataset is not None ) ):
		if st.sidebar.button('Predict'):
			model_name = model_dict.get(model) + "-" + dname_dict.get(dataset)
			data = OnPreProcessing(preprocess_actions, data, cols, ncols, xcols, ycols)
			OnPredict(model_name, data, ycols)

if __name__ == '__main__':
	main()

# end of file