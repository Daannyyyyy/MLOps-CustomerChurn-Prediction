Customer Churn Prediction – MLOps Practice Project

This project is a simple end-to-end machine learning pipeline built while learning the basics of MLOps.
The goal is to predict whether a customer will churn (leave the service) using customer data.

This project focuses more on clean structure and understanding the ML workflow than on complex models.

What this project does

Loads a real-world dataset

Cleans and prepares the data

Trains a machine learning model

Evaluates the model performance

Follows a clean and modular project structure

Project folder structure
mlops/
│
├── data/
│   └── churn.csv        # Dataset used for training
│
├── src/
│   ├── train.py         # Main script to train the model
│   └── utils.py         # Helper functions for data processing
│
├── venv/                # Python virtual environment
│
├── requirements.txt     # Python dependencies
│
└── README.md            # Project description

Tools and libraries used

Python

Pandas

Scikit-learn

VS Code

Virtual Environment (venv)

How the pipeline works

The dataset is loaded from a CSV file

Features and target column are separated

Data is preprocessed:

ID columns are removed

Categorical values are converted to numbers

Missing values are handled

Data is split into training and testing sets

A Logistic Regression model is trained

Model accuracy is calculated on test data

What I learned from this project

Machine learning models only work with numerical data

Why preprocessing is a critical step

How to structure ML code using utils.py and train.py

Importance of train-test split

How real ML pipelines are built step by step

How to run the project
Create a virtual environment
python -m venv venv

Activate the virtual environment (Windows)
venv\Scripts\activate

Install required packages
pip install -r requirements.txt

Run the training script
python src/train.py

Output

After running the script, you will see:

Dataset shape

Feature and target information

Preprocessing status

Final model accuracy

Example output:

Model accuracy: 0.7+

Dataset information

Customer churn dataset

Contains customer details, services, and billing information

Target column: Churn

Future improvements

Save the trained model

Add logging

Track experiments

Implement CI/CD for ML

Deploy the model as an API

Author

This project was built while learning Machine Learning and MLOps fundamentals, focusing on strong basics and clean code.