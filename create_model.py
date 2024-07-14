# Import the BentoML library for deploying machine learning models
import bentoml

# Import necessary libraries from scikit-learn
from sklearn import datasets
from sklearn import svm

# Load the Iris flower dataset for classification
iris = datasets.load_iris()
print("Iris dataset loaded!")  # Optional line to confirm data loading

# Separate the data (features) and target labels from the loaded dataset
X, y = iris.data, iris.target

# Create a Support Vector Classifier (SVC) object with specific parameters
clf = svm.SVC(gamma="scale")  # Gamma is a hyperparameter for SVMs

# Train the classifier on the features (X) and target labels (y)
clf.fit(X, y)
print("Model training completed!")  # Optional line to confirm training

# Save the trained model using BentoML for deployment
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f"Model saved as: {saved_model}")
