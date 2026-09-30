from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = "model/iris_deep_learning_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler_mean = np.load("model/scaler_mean.npy")
    scaler_scale = np.load("model/scaler_scale.npy")
    class_names = np.load("model/class_names.npy", allow_pickle=True)

    print("Model loaded successfully!")

except Exception as e:
    model = None
    scaler_mean = None
    scaler_scale = None
    class_names = None

    print(f"Error loading model: {e}")


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Iris Deep Learning Flask API",
        "status": "running",
        "version": "1.0"
    })


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    if model is not None:

        return jsonify({
            "status": "healthy",
            "model_loaded": True
        })

    return jsonify({
        "status": "unhealthy",
        "model_loaded": False
    }), 500


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check JSON
        data = request.get_json()

        if data is None:

            return jsonify({
                "success": False,
                "error": "Request must contain JSON data"
            }), 400


        # Check features
        if "features" not in data:

            return jsonify({
                "success": False,
                "error": "Missing 'features' field"
            }), 400


        features = data["features"]


        # Check list
        if not isinstance(features, list):

            return jsonify({
                "success": False,
                "error": "'features' must be a list"
            }), 400


        # Check number of features
        if len(features) != 4:

            return jsonify({
                "success": False,
                "error": "Exactly 4 features are required"
            }), 400


        # Convert input to NumPy array
        input_data = np.array(
            features,
            dtype=np.float32
        )


        # Scale input
        input_data = (
            input_data - scaler_mean
        ) / scaler_scale


        # Add batch dimension
        input_data = np.expand_dims(
            input_data,
            axis=0
        )


        # Generate prediction
        prediction = model.predict(
            input_data,
            verbose=0
        )


        # Get predicted class
        predicted_index = int(
            np.argmax(prediction[0])
        )


        predicted_class = str(
            class_names[predicted_index]
        )


        # Confidence
        confidence = float(
            np.max(prediction[0])
        )


        # Return response
        return jsonify({

            "success": True,

            "prediction": {
                "class_id": predicted_index,
                "class_name": predicted_class,
                "confidence": round(
                    confidence,
                    4
                )
            }

        })


    except ValueError:

        return jsonify({

            "success": False,

            "error": "Features must contain valid numeric values"

        }), 400


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )