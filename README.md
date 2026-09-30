\# Flask API for Deep Learning Model



\## Task 4 - Developing a Flask API for Deep Learning Models



This project demonstrates how a trained deep learning model can be exposed as a REST API using the Flask framework.



The project uses a neural network trained on the Iris dataset. The trained model is saved in Keras format and loaded by the Flask application.



\---



\## Objective



The objective of this task is to expose a trained deep learning model as a REST API using Flask.



The API:



\- Loads a trained deep learning model

\- Accepts JSON input

\- Performs prediction

\- Returns JSON responses

\- Implements error handling

\- Provides health checking



\---



\## Technologies Used



\- Python

\- Flask

\- TensorFlow / Keras

\- NumPy

\- Scikit-learn

\- Requests

\- REST API

\- Git

\- GitHub



\---



\## Dataset



The Iris dataset is used for this implementation.



The model uses four input features:



1\. Sepal Length

2\. Sepal Width

3\. Petal Length

4\. Petal Width



The model predicts one of three classes:



\- Setosa

\- Versicolor

\- Virginica



\---



\## Project Structure



```text

Task-4-Flask-Deep-Learning-API/

│

├── model/

│   ├── iris\_deep\_learning\_model.keras

│   ├── scaler\_mean.npy

│   ├── scaler\_scale.npy

│   └── class\_names.npy

│

├── screenshots/

│

├── test/

│   └── test\_api.py

│

├── app.py

├── train\_model.py

├── requirements.txt

├── README.md

└── .gitignore

