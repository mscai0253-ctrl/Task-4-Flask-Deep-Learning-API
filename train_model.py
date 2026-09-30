import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale input features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create deep learning model
model = Sequential([
    Dense(16, activation="relu", input_shape=(4,)),
    Dense(16, activation="relu"),
    Dense(3, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model - only 5 epochs
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=8,
    validation_split=0.1,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nModel Evaluation")
print("----------------")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Save model
model.save("model/iris_deep_learning_model.keras")

# Save scaler parameters
np.save("model/scaler_mean.npy", scaler.mean_)
np.save("model/scaler_scale.npy", scaler.scale_)

# Save class names
class_names = iris.target_names
np.save("model/class_names.npy", class_names)

print("\nModel saved successfully!")
print("Location: model/iris_deep_learning_model.keras")