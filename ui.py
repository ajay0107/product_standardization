import streamlit as st
import pandas as pd
import openai
import pandas as pd
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    # This is the default and can be omitted
    api_key=OPENAI_API_KEY,
)

# Function to call GPT-4o for dish standardization
def standardize_dish_name(dish_name):
    prompt = f"""
    Standardize the given dish name into two levels:
    - Level 1: A structured and readable version of the dish name.
    - Level 2: A broader category that the dish belongs to (e.g., "burger", "pizza", "pasta").

    Example:
    Input: "burge chicken"
    Output: ["chicken burger", "burger"]

    Input: "ham brger"
    Output: ["ham burger", "burger"]

    Input: "veg piz"
    Output: ["Veg Pizza", "Pizza"]

    Input: "butt chikn msla"
    Output: ["Butter Chicken Masala", "Indian"]

    Input: "{dish_name}"
    Output:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant for food standardization."},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )  

        # Extract response text
        result_text = response.choices[0].message.content.strip()
        print(f"Result Text is: {result_text}")

        # Ensure output format is valid
        result_text = result_text.replace("[", "").replace("]", "").replace('"', "").strip()
        result_list = result_text.split(", ")

        if len(result_list) == 2:
            return result_list[0].strip(), result_list[1].strip()
        
        elif len(result_list) == 1:
            return result_list[0].strip(), "Unknown"
        else:
            return dish_name, "Error"  # Fallback if response is malformed

    except Exception as e:
        print(f"Error processing '{dish_name}': {e}")
        return dish_name, "Error"  # Return original if error occurs

def create_standardized_prod_names(df):
    try:
        # Ensure there's a column named "dish_name"
        if "dish_name" not in df.columns:
            raise ValueError("CSV file must contain a column named 'dish_name'")
        # Apply standardization function with delay to avoid rate limits
        df[["level1_standard_name", "level2_standard_name"]] = df["dish_name"].apply(
            lambda x: pd.Series(standardize_dish_name(x))
        )
        return df
    except Exception as e:
        print(f"Unexpected error: {e}")


# Set page configuration
st.set_page_config(page_title="Product Data Enhancement", layout="wide")

# Custom CSS with sidebar-specific styling
st.markdown("""
    <style>
        .stApp {
            background-color: #F0F2F6;
            font-family: 'Arial', sans-serif;
        }
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0E1117 !important;
        }
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] .stAlert,
        section[data-testid="stSidebar"] .stAlert * {
            color: white !important;
        }
        /* Main content styling */
        .stTitle {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }
        .stSubtitle {
            font-size: 20px;
            text-align: center;
            color: #555;
            margin-bottom: 30px;
        }
        .upload-box {
            border: 2px dashed #0073E6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            background-color: #FFFFFF;
            font-weight: bold;
            color: #0073E6;
        }
        .stDataFrame {
            border-radius: 10px;
            background-color: #FFF;
        }
        .stButton>button {
            background-color: #0073E6;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #005BB5;
        }
        .custom-warning {
            background-color: #FFF3CD;
            padding: 15px;
            border-radius: 8px;
            color: black !important;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
            border: 1px solid #FFC107;
        }
        /* Black text rules */
        h3 {
            color: black !important;
        }
        .stAlert, .stAlert * {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<div class='stTitle'>Product Data Enhancement Tool</div>", unsafe_allow_html=True)
st.markdown("<div class='stSubtitle'>Upload your product CSV file to process and enhance product data</div>", unsafe_allow_html=True)

# Sidebar for use case selection
st.sidebar.title("Select Use Case")
use_case = st.sidebar.radio("", ["Standardize Product Names", "Extract Food Attributes"])

st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.info("📌 Upload a CSV file containing product names (and descriptions for attributes). The tool will process and return an enhanced dataset.")

# Upload section
st.markdown("<div class='upload-box'>📤 Drag & Drop or Click to Upload CSV File</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Reading CSV file... Please wait!"):
        df = pd.read_csv(uploaded_file)
    st.success(f"✅ {uploaded_file.name} uploaded successfully!")

    # Show Data Preview
    st.subheader("📌 Uploaded Data Preview")    
    st.dataframe(df.head())

    # Processing message
    with st.spinner("Processing Data... Please wait!"):
        st.subheader("🚀 Standardizing Product Names...")
        st.info("AI Model is analyzing and standardizing product data...")
        std_prod_nm_output_df = create_standardized_prod_names(df)

    st.subheader("📊 Standardized Product Data Preview")
    st.dataframe(std_prod_nm_output_df.head())  # Replace with actual processed data

    # Download processed data button
    st.subheader("📥 Download Standardized Product Data")
    output_file = "processed_data.csv"
    df.to_csv(output_file, index=False)

    st.download_button(
        label="📩 Download Standardized Product Data CSV",
        data=df.to_csv(index=False),
        file_name=output_file,
        mime="text/csv"
    )

else:
    st.markdown("<div class='custom-warning'>⚠️ Please upload a CSV file to proceed.</div>", unsafe_allow_html=True)