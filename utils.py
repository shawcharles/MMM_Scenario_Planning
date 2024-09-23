import streamlit as st
import pandas as pd
import tempfile
import yaml
import joblib

# Miller Stone color palette
custom_palette = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
]
base_color = "#C7C7C7"  # Base color

#  Function to assign colors to all components using the custom color palette
def assign_colors_to_all(media_columns, extra_features_cols, promos, prophet_columns):
    all_columns = media_columns + extra_features_cols + promos + prophet_columns + ["Base", "intercept"]
    channel_colors = {channel: custom_palette[i % len(custom_palette)] for i, channel in enumerate(all_columns)}
    channel_colors["Base"] = base_color  # Set the base color for "Base"
    return channel_colors

# Function to get numeric style with precision
def get_numeric_style_with_precision(precision: int) -> dict:
    return {"type": ["numericColumn", "customNumericFormat"], "precision": precision}

# Define precision styles
PRECISION_ZERO = get_numeric_style_with_precision(0)
PRECISION_ONE = get_numeric_style_with_precision(1)
PRECISION_TWO = get_numeric_style_with_precision(2)
PINLEFT = {"pinned": "left"}

# Function to highlight cells based on conditions
def highlight(color, condition):
    code = f"""
        function(params) {{
            color = "{color}";
            if ({condition}) {{
                return {{
                    'backgroundColor': color
                }}
            }}
        }};"""
    return JsCode(code)

# Function to calculate optimal column widths
def calculate_column_widths(df):
    max_lengths = {}
    for column in df.columns:
        max_lengths[column] = max(df[column].astype(str).map(len).max(), len(column)) * 10
    return max_lengths

# Function to load the configuration
def load_config(config_file: str) -> dict:
    try:
        with open(config_file, "r") as f:
            config = yaml.full_load(f)
        return config
    except FileNotFoundError:
        st.error(f"Couldn't load config file: {config_file}")
        st.stop()

# Function to determine prophet columns based on config
def determine_prophet_columns(config):
    prophet_config = config.get('prophet', {})
    prophet_columns = []
    if prophet_config.get('yearly_seasonality', False):
        prophet_columns.append('yearly')
    if prophet_config.get('trend', False):
        prophet_columns.append('trend')
    if prophet_config.get('weekly_seasonality', False):
        prophet_columns.append('weekly')
    return prophet_columns


# Function to handle file uploads and store them in session state
def handle_file_uploads():
    # Define session state keys for uploaded files
    file_keys = {
        "uploaded_data_file": "data_file",
        "uploaded_model_file": "model_file",
        "uploaded_config_file": "config_file",
        "uploaded_prophet_file": "prophet_file"
    }

    with st.expander("Upload Files", expanded=True):
        col1, col2 = st.columns(2)

        # File upload columns
        with col1:
            uploaded_data_file = st.file_uploader("Upload your data file (CSV)", type="csv", key=file_keys["uploaded_data_file"])
            uploaded_model_file = st.file_uploader("Upload your model file (NetCDF)", type="nc", key=file_keys["uploaded_model_file"])
        with col2:
            uploaded_config_file = st.file_uploader("Upload your config file (YAML)", type="yml", key=file_keys["uploaded_config_file"])
            uploaded_prophet_file = st.file_uploader("Upload your Prophet model file (PKL)", type="pkl", key=file_keys["uploaded_prophet_file"])

    # Check and process each file upload
    if uploaded_data_file and not st.session_state.get(f"{file_keys['uploaded_data_file']}_processed", False):
        st.session_state.data = pd.read_csv(uploaded_data_file)
        st.session_state.data_uploaded = True

    if uploaded_model_file and not st.session_state.get(f"{file_keys['uploaded_model_file']}_processed", False):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_model_file.read())
            st.session_state.model_path = temp_file.name
        st.session_state.model_uploaded = True
    
    if uploaded_config_file and not st.session_state.get(f"{file_keys['uploaded_config_file']}_processed", False):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_config_file.read())
            st.session_state.config_path = temp_file.name
        st.session_state.config_uploaded = True
    
    if uploaded_prophet_file and not st.session_state.get(f"{file_keys['uploaded_prophet_file']}_processed", False):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_prophet_file.read())
            st.session_state.prophet_path = temp_file.name
        st.session_state.prophet_uploaded = True

# Function to check if all required files are uploaded and processed
def are_all_files_uploaded():
    required_files = [
        'data_uploaded',
        'model_uploaded',
        'config_uploaded',
        'prophet_uploaded'
    ]
    # Check if each required file is uploaded and processed
    return all(st.session_state.get(file_key, False) for file_key in required_files)
