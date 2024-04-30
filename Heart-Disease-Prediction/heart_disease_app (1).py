#GUI SYTEM for APPLicatioon
#Importing libraries
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


#GUI
class HeartDiseaseApp:
    def __init__(self, root, LR_model, SVM_Model, column_transformer, sec_scalar, simpleImp):
        self.root = root
        self.root.title("Heart Disease Prediction App")

	# Store the Logistic Regression model and transformers
        self.LR_model = LR_model
        self.SVM_model = SVM_Model
        self.column_transformer = column_transformer
        self.sec_scalar = sec_scalar
        self.simpleImp = simpleImp

        # Labels
        ttk.Label(root, text="Age:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(root, text="Resting BP:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(root, text="Cholesterol:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(root, text="Fasting Blood Sugar:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(root, text="Max HR:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(root, text="Old Peak:").grid(row=5, column=0, padx=10, pady=5, sticky="w")


        # Entry Widgets
        self.age_entry = ttk.Entry(root)
        self.cholesterol_entry = ttk.Entry(root)
        self.resting_bp_entry = ttk.Entry(root)
        self.fasting_bs_entry = ttk.Entry(root)
        self.old_peak_entry = ttk.Entry(root)
        self.max_hr_entry = ttk.Entry(root)

        self.age_entry.grid(row=0, column=1, padx=10, pady=5)
        self.resting_bp_entry.grid(row=1, column=1, padx=10, pady=5)
        self.cholesterol_entry.grid(row=2, column=1, padx=10, pady=5)
        self.fasting_bs_entry.grid(row=3, column=1, padx=10, pady=5)
        self.max_hr_entry.grid(row=4, column=1, padx=10, pady=5)
        self.old_peak_entry.grid(row=5, column=1, padx=10, pady=5)

        # Button to predict
        ttk.Button(root, text="Predict", command=self.predict).grid(row=6, column=0, columnspan=2, pady=10)

    # Make prediction using user inputs
    def predict(self):

        # Get user inputs
        age = float(self.age_entry.get())
        cholesterol = float(self.cholesterol_entry.get())
        resting_bp = float(self.resting_bp_entry.get())
        fasting_bs = float(self.fasting_bs_entry.get())
        old_peak = float(self.old_peak_entry.get())
        max_hr = float(self.max_hr_entry.get())

        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': ['M'],
            'ChestPainType': ['ATA'],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'RestingECG': ['Normal'],
            'MaxHR': [max_hr],
            'ExerciseAngina': ['N'],
            'Oldpeak': [old_peak],
            'ST_Slope': ['Flat'],
            'HeartDisease': [1]
        })

        input_data_Processed = input_data.iloc[:, :-1].values #Getting inputted data from GUI

	# Preprocess and standardize data
        input_data_Processed[:, [3, 4]] = self.simpleImp.transform(input_data_Processed[:, [3, 4]])
        input_data_Processed = np.array(self.column_transformer.transform(input_data_Processed))
        input_data_Processed[:, [9,10,11,13,14]] = self.sec_scalar.transform(input_data_Processed[:, [9,10,11,13,14]])

	# Get prediction from the models
        prediction = self.LR_model.predict(input_data_Processed)
        predictionSVM = self.SVM_model.predict(input_data_Processed)

        # Display the prediction in the GUI
        result_label = ttk.Label(self.root, text=f"LR Prediction: {prediction}" + f" SVM Prediction: {predictionSVM}")
        result_label.grid(row=7, column=0, columnspan=2, pady=10)
        
if __name__ == "__main__":
    # Import dataset
        filePath = 'C:/Users/AmitG/Downloads/heart.csv'
        dataSet = pd.read_csv(filePath, na_values={'RestingBP': 0, 'Cholesterol': 0})

    # Rest of the preprocessing steps (imputation, encoding, scaling)

        #Separate data into input and output
        X = dataSet.iloc[:, : -1].values
        Y = dataSet.iloc[:, -1].values

        #Imputer
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        X[:, [3, 4]] = imputer.fit_transform(X[:, [3, 4]])

        #Encoding
        colTrans = ColumnTransformer(transformers = [('encoder', OneHotEncoder(drop = 'first'), [1,2,6,8,10])], remainder = 'passthrough')
        X = np.array(colTrans.fit_transform(X))

	#Separate data into sets for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1) #Loading Training Data into variables

        #Standardize the data
        secScalar = StandardScaler()
        X_train[:, [9,10,11,13,14]] = secScalar.fit_transform(X_train[:, [9,10,11,13,14]])
        X_test[:, [9,10,11,13,14]] = secScalar.transform(X_test[:, [9,10,11,13,14]])

        # Load Logistic Regression model
        LR = LogisticRegression()
        LR.fit(X_train, y_train)

        # Load SVM model
        svm = SVC(kernel = 'linear')
        svm.fit(X_train, y_train)


    	#Call the GUI and load corresponding parameters
        root = tk.Tk()
        app = HeartDiseaseApp(root, LR_model=LR, SVM_Model= svm, column_transformer=colTrans, sec_scalar=secScalar, simpleImp= imputer)
        root.mainloop()
