from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "model/superkart_sales_model.pkl"
)

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Expected model features
# --------------------------------------------------

MODEL_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "message": "SuperKart Sales Prediction API is running"
    })


# --------------------------------------------------
# Single prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No input data provided"
            }), 400

        # Check whether all expected columns exist
        missing_features = [
            col
            for col in MODEL_FEATURES
            if col not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        # Create DataFrame
        input_df = pd.DataFrame(
            [data],
            columns=MODEL_FEATURES
        )

        # Prediction
        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_sales": round(
                float(prediction),
                2
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Batch prediction
# --------------------------------------------------

@app.route("/predict_batch", methods=["POST"])
def predict_batch():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No input data provided"
            }), 400

        if not isinstance(data, list):
            return jsonify({
                "error":
                "Batch input must be a list of records"
            }), 400

        input_df = pd.DataFrame(data)

        missing_features = [
            col
            for col in MODEL_FEATURES
            if col not in input_df.columns
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        # Ensure same column order as training
        input_df = input_df[MODEL_FEATURES]

        predictions = model.predict(
            input_df
        )

        results = input_df.copy()

        results["Predicted_Sales"] = (
            predictions.round(2)
        )

        return jsonify(
            results.to_dict(
                orient="records"
            )
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )