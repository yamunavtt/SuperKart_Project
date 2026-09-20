import streamlit as st
import pandas as pd
import requests
import os

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:5000"
)
# Configure the Streamlit page
st.set_page_config(
    page_title="SuperKart Sales Forecast",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 SuperKart Sales Forecasting")

st.write(
    "Predict product-store sales using the "
    "SuperKart machine learning model."
)

#Create Single and Batch Prediction tabs
single_tab, batch_tab = st.tabs(
    [
        "Single Prediction",
        "Batch Prediction"
    ]

#Single prediction interface
    with single_tab:

    st.subheader("Single Sales Prediction")

    col1, col2 = st.columns(2)

    with col1:

        product_weight = st.number_input(
            "Product Weight",
            min_value=0.0,
            value=12.5,
            step=0.1
        )

        sugar_content = st.selectbox(
            "Product Sugar Content",
            [
                "Low Sugar",
                "Regular",
                "No Sugar"
            ]
        )

        allocated_area = st.number_input(
            "Product Allocated Area",
            min_value=0.0,
            max_value=1.0,
            value=0.06,
            step=0.01
        )

        product_mrp = st.number_input(
            "Product MRP",
            min_value=0.0,
            value=180.5,
            step=1.0
        )

        product_id_char = st.selectbox(
            "Product ID Category",
            [
                "FD",
                "DR",
                "NC"
            ]
        )


    with col2:

        store_size = st.selectbox(
            "Store Size",
            [
                "Small",
                "Medium",
                "High"
            ]
        )

        city_type = st.selectbox(
            "Store Location City Type",
            [
                "Tier 1",
                "Tier 2",
                "Tier 3"
            ]
        )

        store_type = st.selectbox(
            "Store Type",
            [
                "Departmental Store",
                "Food Mart",
                "Supermarket Type1",
                "Supermarket Type2"
            ]
        )

        store_age = st.number_input(
            "Store Age (Years)",
            min_value=0,
            max_value=100,
            value=16,
            step=1
        )

        product_type_category = st.selectbox(
            "Product Type Category",
            [
                "Perishables",
                "Non Perishables"
            ]
        )

#Send single prediction to Flask
    if st.button(
        "Predict Sales",
        type="primary"
    ):

        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar_content,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age,
            "Product_Type_Category":
                product_type_category
        }

        try:

            response = requests.post(
                f"{BACKEND_URL}/predict",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:

                result = response.json()

                predicted_sales = result[
                    "predicted_sales"
                ]

                st.success(
                    f"Predicted Sales: "
                    f"{predicted_sales:,.2f}"
                )

            else:

                st.error(
                    f"Prediction failed: "
                    f"{response.text}"
                )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Unable to connect to backend: {e}"
            )

# Batch prediction interface
with batch_tab:

    st.subheader("Batch Sales Prediction")

    st.write(
        "Upload a CSV file containing multiple "
        "product-store records."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

#Define the required columns
    required_columns = [
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

#Read and validate uploaded CSV
    if uploaded_file is not None:

        batch_df = pd.read_csv(
            uploaded_file
        )

        st.write("Uploaded Data")

        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )

        missing_columns = [
            col
            for col in required_columns
            if col not in batch_df.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        else:

            st.success(
                "CSV structure is valid."
            )

        if not missing_columns:

            if st.button(
                "Run Batch Prediction",
                type="primary"
            ):

                try:

                    batch_input = (
                        batch_df[
                            required_columns
                        ]
                        .to_dict(
                            orient="records"
                        )
                    )

                    response = requests.post(
                        f"{BACKEND_URL}/predict_batch",
                        json=batch_input,
                        timeout=60
                    )

                    if response.status_code == 200:

                        predictions = (
                            response.json()
                        )

                        result_df = pd.DataFrame(
                            predictions
                        )

                        st.success(
                            "Batch prediction completed."
                        )

                        st.dataframe(
                            result_df,
                            use_container_width=True
                        )

                    else:

                        st.error(
                            f"Prediction failed: "
                            f"{response.text}"
                        )

                except requests.exceptions.RequestException as e:

                    st.error(
                        f"Backend connection failed: {e}"
                    )

        # Allow prediction results to be downloaded
            result_df = pd.DataFrame(predictions)

                     csv_data = (
                            result_df
                            .to_csv(index=False)
                            .encode("utf-8")
                        )

                        st.download_button(
                            label="Download Predictions",
                            data=csv_data,
                            file_name=(
                                "superkart_predictions.csv"
                            ),
                            mime="text/csv"
                        )
    
    
)
