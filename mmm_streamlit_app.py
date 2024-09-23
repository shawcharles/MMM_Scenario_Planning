import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import holidays
import plotly.graph_objects as go
import plotly.express as px
import xarray as xr
import pymc as pm
import tempfile
import yaml
from pymc_marketing.mmm import MMM
import os
from graphs import *
from preprocessing import *
from utils import *
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Define constants
MAX_TABLE_HEIGHT = 500

# Miller Stone color palette
custom_palette = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
]

# Use #C7C7C7 for the base color
base_color = "#C7C7C7"
# Function to assign colors to all components using the custom color palette
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


with open("styles.css") as f:
    css = f.read()

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
        }};
    """
    return JsCode(code)

# Function to calculate optimal column widths
def calculate_column_widths(df):
    max_lengths = {}
    for column in df.columns:
        max_lengths[column] = max(df[column].astype(str).map(len).max(), len(column)) * 10
    return max_lengths

def ensure_legends_under_graph(fig, legend_y=-0.6):
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=legend_y,
            xanchor="center",
            x=0.5,
           
        ),
        margin=dict(l=10, r=50, t=40, b=40),
    )

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

# Function to load and predict prophet model
def load_and_predict_prophet(prophet_path, future_dates):
    model = joblib.load(prophet_path)
    future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)
    return forecast

# Function to create features
def create_features(df, date_column, extra_features_cols):
    df = df.copy()
    if 'holidays' in extra_features_cols:
        us_holidays = holidays.US(years=[min(df[date_column].dt.year), max(df[date_column].dt.year)])
        df['holidays'] = df[date_column].apply(lambda x: 1 if x in us_holidays else 0)
    if 'dayofweek' in extra_features_cols:
        df['dayofweek'] = df[date_column].dt.dayofweek
    if 'weekofyear' in extra_features_cols:
        df['weekofyear'] = df[date_column].dt.isocalendar().week.astype(int)
    if 'dayofmonth' in extra_features_cols:
        df['dayofmonth'] = df[date_column].dt.day
    return df

# Function to scale prophet columns
def scale_prophet_columns(data, prophet_columns):
    scaler = MinMaxScaler()
    for feature in prophet_columns:
        if feature in data.columns:
            data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
    return data

# Function to generate out-of-sample data
def generate_out_of_sample_data(n_new, channel_spends, promo_periods, last_date, interval_type, 
                                date_column, media_columns, extra_features_cols, promos, prophet_columns, 
                                prophet_path, X):
    freq = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'}.get(interval_type)
    new_dates = pd.date_range(start=last_date, periods=1 + n_new, freq=freq)[1:]
    if interval_type == 'weekly':
        new_dates = new_dates[new_dates.weekday == 0]

    X_out_of_sample = pd.DataFrame({date_column: new_dates})

    for column in media_columns:
        X_out_of_sample[column] = channel_spends[column]

    for promo in promos:
        X_out_of_sample[promo] = 0

    for promo_period in promo_periods:
            promo_start_date, promo_end_date, selected_promos = promo_period
            promo_dates_range = pd.date_range(start=promo_start_date, end=promo_end_date, freq=freq)
            for promo_date in promo_dates_range:
                for promo in selected_promos:
                    X_out_of_sample.loc[X_out_of_sample[date_column] == promo_date, promo] = 1

    for column in extra_features_cols:
        if column in X.columns:
            if X[column].dtype == 'float':
                X_out_of_sample[column] = X[column].iloc[-1] * (1 + np.random.normal(0, 0.05, size=n_new))
            else:
                X_out_of_sample[column] = X[column].iloc[-1]

    if prophet_columns:
        forecast = load_and_predict_prophet(prophet_path, new_dates)
        for column in prophet_columns:
            X_out_of_sample[column] = forecast['yhat'].values

    X_out_of_sample = create_features(X_out_of_sample, date_column, extra_features_cols)
    X_out_of_sample = scale_prophet_columns(X_out_of_sample, prophet_columns)

    return X_out_of_sample


def display_and_save_vif_table(data, drop_columns=None, date_column=None, target_column=None, high_vif_threshold=10):
    """
    Function to calculate Variance Inflation Factor (VIF), display it with conditional formatting,
    and save the resulting VIF table to st.session_state for later use.
    
    Parameters:
    data: pd.DataFrame - Input data
    drop_columns: list - List of columns to exclude from the VIF calculation
    date_column: str - Date column to exclude from the VIF calculation
    target_column: str - Target column to exclude from the VIF calculation
    high_vif_threshold: float - Threshold for highlighting high VIF values
    
    Returns:
    None
    """
    # Drop unnecessary columns such as date or target column
    if drop_columns is None:
        drop_columns = []
    if date_column:
        drop_columns.append(date_column)
    if target_column:
        drop_columns.append(target_column)
    
    X = data.drop(columns=drop_columns)
    
    # Add a constant for the intercept (as needed for VIF calculation)
    X_with_const = sm.add_constant(X)

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(X.shape[1])]  # i+1 to skip the constant
    
    # Save the VIF table to session state
    st.session_state['vif_table'] = vif_data
    # Apply conditional formatting based on VIF value
    styled_vif_table = vif_data.style.set_properties(**{
        'background-color': 'white'  # White background for cells and headers
    })

    # Apply the highlight for high VIF values using `applymap` on the "VIF" column
    styled_vif_table = styled_vif_table.applymap(
        lambda x: highlight_high_vif_values(x, high_vif_threshold=high_vif_threshold), 
        subset=["VIF"]
    )
    styled_vif_table = styled_vif_table.set_table_styles([
        {
            'selector': 'thead th',  # 'thead th' targets header cells
            'props': [('background-color', 'white')]  # Set white background for header
        },
        {
            'selector': 'tbody td',  # 'tbody td' targets table body cells
            'props': [('background-color', 'white')]  # Set white background for body cells
        }
    ])
    # Display the table in Streamlit
    st.subheader("VIF Table")
    st.dataframe(styled_vif_table, height=500,hide_index = True)
    
def highlight_high_vif_values(vif_value: float, high_vif_threshold: float=7.0) -> str:
    if vif_value > high_vif_threshold:
        weight = 'bold'
        color = '#d1615d' 
    else:
        weight = 'normal'
        color = '#6a9f58' 
    style = f'font-weight: {weight}; color: {color}'
    return style

# Function to calculate media contributions for new data
def new_data_media_contributions(X: pd.DataFrame, mmm: MMM, original_scale: bool = True):
    mmm._data_setter(X)
    
    with mmm.model:
        posterior_predictive = pm.sample_posterior_predictive(
            mmm.idata,
            var_names=["channel_contributions", "intercept", "control_contributions"],
        )
    
    channel_contributions = posterior_predictive.posterior_predictive["channel_contributions"]
    intercept = posterior_predictive.posterior_predictive["intercept"]
    control_contributions = posterior_predictive.posterior_predictive["control_contributions"]

    if original_scale:
        channel_contributions = xr.DataArray(
            mmm.get_target_transformer().inverse_transform(
                channel_contributions.data.reshape(-1, 1)
            ).reshape(channel_contributions.shape),
            dims=channel_contributions.dims,
            coords=channel_contributions.coords,
        )
        
        control_contributions = xr.DataArray(
            mmm.get_target_transformer().inverse_transform(
                control_contributions.data.reshape(-1, 1)
            ).reshape(control_contributions.shape),
            dims=control_contributions.dims,
            coords=control_contributions.coords,
        )
    
    channel_contributions_df = channel_contributions.to_dataframe(name='contribution').reset_index()
    control_contributions_df = control_contributions.to_dataframe(name='contribution').reset_index()
    control_contributions_df = control_contributions_df.rename(columns={'control': 'channel'})

    intercept_mean = intercept.mean(dim=("chain", "draw")).item()
    dates = channel_contributions_df['date'].unique()
    intercept_df = pd.DataFrame({
        'date': dates,
        'contribution': intercept_mean,
        'channel': 'intercept'
    })

    if 'date' in channel_contributions_df.columns:
        channel_contributions_df['date'] = pd.to_datetime(channel_contributions_df['date'], errors='coerce')
    if 'date' in control_contributions_df.columns:
        control_contributions_df['date'] = pd.to_datetime(control_contributions_df['date'], errors='coerce')

    return channel_contributions_df, intercept_df, control_contributions_df

# Function to initialize session state
def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'prophet_path' not in st.session_state:
        st.session_state.prophet_path = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'extra_features_cols' not in st.session_state:
        st.session_state.extra_features_cols = None
    if 'media_columns' not in st.session_state:
        st.session_state.media_columns = None
    if 'date_column' not in st.session_state:
        st.session_state.date_column = None
    if 'interval_type' not in st.session_state:
        st.session_state.interval_type = None
    if 'promos' not in st.session_state:
        st.session_state.promos = None
    if 'prophet_columns' not in st.session_state:
        st.session_state.prophet_columns = None
    if 'last_date' not in st.session_state:
        st.session_state.last_date = None
    if 'n_historical' not in st.session_state:
        st.session_state.n_historical = None
    if 'n_new' not in st.session_state:
        st.session_state.n_new = 1
    if 'channel_spends' not in st.session_state:
        st.session_state.channel_spends = None
    if 'X_out_of_sample' not in st.session_state:
        st.session_state.X_out_of_sample = pd.DataFrame()
    if 'prediction_triggered' not in st.session_state:
        st.session_state.prediction_triggered = False
    if 'tab2_data_created' not in st.session_state:
        st.session_state.tab2_data_created = False
    if 'previous_params' not in st.session_state:
        st.session_state.previous_params = {}
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = None
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = None
    if 'rolling_window' not in st.session_state:
        st.session_state['rolling_window'] = 1
# Function to check for parameter changes
def parameters_changed(params, previous_params):
    for key, value in params.items():
        prev_value = previous_params.get(key)
        if isinstance(value, pd.DataFrame) and isinstance(prev_value, pd.DataFrame):
            if not value.equals(prev_value):
                return True
        else:
            if prev_value != value:
                return True
    return False

# Function to update previous parameters
def update_previous_params(params):
    for key, value in params.items():
        st.session_state.previous_params[key] = value

def calculate_dialog_table_width(df, max_width=1100, char_width=8, padding=48, border=1):
    # Calculate width for each column based on the longest entry (character count)
    col_widths = [max(df[col].astype(str).map(len).max(), len(str(col))) * char_width for col in df.columns]
    
    # Sum up the column widths and add padding and border space
    total_width = sum(col_widths) + padding + (len(df.columns) * border * 2)
    
    # Ensure the total width does not exceed the dialog width (900pt in this case)
    if total_width > max_width:
        total_width = max_width  # Limit the width to the dialog's max width
    
    return int(total_width)


# Streamlit interface setup
st.set_page_config(page_title="MMM Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
st.title('MMM Dashboard')

# Tabs for Data Upload, Forecasting Data Set Creation, and Scenario Planning
tab1, tab2,tab3= st.tabs(["Data Load and Inspect","Model Validation and Results" ,"Scenario Planning Tool"])

initialize_session_state()

with tab1:
    st.header('Upload Data for Analysis')
    handle_file_uploads()
    if are_all_files_uploaded():
        # Load configuration
        config = load_config(st.session_state.config_path)

        # Determine prophet columns based on config
        prophet_columns = determine_prophet_columns(config)

        # Extract parameters from the config
        target_column = config.get('target_col', 'subscribers')
        target_type = config.get('target_type','revenue')
        extra_features_cols = config.get('extra_features_cols', [])
        media_columns = [entry['spend_col'] for entry in config.get('media', [])]
        date_column = config.get('date_col', 'date')
        interval_type = config['raw_data_granularity']
        promos = config.get('promos', [])
        if 'promo' in extra_features_cols:
            promos.append('promo')
        if 'promo_events' in extra_features_cols:
            promos.append('promo_events')

        # Now define n_historical_default based on interval_type
        if interval_type == 'daily':
            n_historical_default = 90  # 1 quarter = 90 days
        elif interval_type == 'weekly':
            n_historical_default = 13  # 1 quarter = 13 weeks
        elif interval_type == 'monthly':
            n_historical_default = 3  # 1 quarter = 3 months
        else:
            raise ValueError("Invalid interval_type. Choose from 'daily', 'weekly', or 'monthly'.")

        data = st.session_state.data
        st.success("Files Uploaded Successfully")
        data[date_column] = pd.to_datetime(data[date_column])
        X, y, X_train, y_train, X_test, y_test = split_data(data, config)
        model = MMM.load(st.session_state.model_path)
        last_date = pd.to_datetime(data[date_column].max())
        freq = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'}.get(interval_type)
        if not freq:
            raise ValueError("Invalid interval_type. Choose from 'daily', 'weekly', or 'monthly'.")

        # Store in session state
        st.session_state.data_uploaded = True
        st.session_state.data = data
        st.session_state.target_type =target_type
        st.session_state.X = X
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y = y
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.model = model
        st.session_state.config = config
        st.session_state.prophet_path = st.session_state.prophet_path
        st.session_state.target_column = target_column
        st.session_state.extra_features_cols = extra_features_cols
        st.session_state.media_columns = media_columns
        st.session_state.date_column = date_column
        st.session_state.interval_type = interval_type
        st.session_state.promos = promos
        st.session_state.prophet_columns = prophet_columns
        st.session_state.last_date = last_date
        st.session_state.n_historical_default = n_historical_default

        # Generate the channel colors for all components
        channel_colors = assign_colors_to_all(media_columns, extra_features_cols, promos, prophet_columns)

        # Filter columns to keep only the relevant ones
        columns_to_keep = [date_column, target_column] + media_columns + extra_features_cols
        data = data[columns_to_keep]
        st.subheader("Input Data")
        styled_data_table = data.style.set_properties(**{
            'background-color': 'white'  # White background for cells and headers
        })
        st.dataframe(styled_data_table,height=150,hide_index = True)

        col1, col2 = st.columns([1, 4])  # Adjust the ratio for a larger correlation map

        with col1:
            display_and_save_vif_table(data, drop_columns=[date_column, target_column], date_column=date_column, target_column=target_column)
        with col2:
            # Correlation Map
            st.subheader("Correlation Map")
            corr_matrix = data.corr()
            fig_corr = px.imshow(corr_matrix,text_auto=".2f", aspect="auto")
            fig_corr.update_layout(
                plot_bgcolor = 'rgba(0, 0, 0, 0)',paper_bgcolor='white',
                height=500,  
                margin=dict(l=10, r=50, t=10, b=10),  # Reduce the margins
                coloraxis_showscale=False,  # Hide the color bar (legend)

            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # Multi-selection for variables to be plotted, with default as the target column
        col3,col4 = st.columns([1,1])
        with col3:
            selected_variables = st.multiselect(
                "Select one or more variables:", 
                options=data.drop(columns='date').columns, 
                default=[target_column]
            )
        with col4:
            # Multi-selection for variables to be plotted on the secondary y-axis
            secondary_y_variables = st.multiselect(
                "Select variables to be plotted on the secondary y-axis:", 
                options=[var for var in selected_variables if var != target_column],  # Options exclude the target column
                default=[]
            )

        # Plot only if there are selected variables
        if selected_variables:
            fig_line = make_subplots(specs=[[{"secondary_y": True}]])

            # Track the variables added to the legend to ensure "subscribers" is shown only once
            variables_in_legend = set()

            # Add the target variable on the primary y-axis by default
            if target_column not in variables_in_legend:
                fig_line.add_trace(
                    go.Scatter(
                        x=data[date_column], 
                        y=data[target_column], 
                        mode='lines', 
                        name=target_column, 
                        line=dict(color='rgb(0,0,0)')
                    ),
                    secondary_y=False
                )
                variables_in_legend.add(target_column)

            # Add traces for selected variables, placing them on either the primary or secondary y-axis
            for var in selected_variables:
                line_color = channel_colors.get(var, 'rgb(0,0,0)')
                if var in secondary_y_variables:
                    if var not in variables_in_legend:
                        fig_line.add_trace(
                            go.Scatter(
                                x=data[date_column], 
                                y=data[var], 
                                mode='lines', 
                                name=var, 
                                line=dict(color=line_color)
                            ),
                            secondary_y=True  # Place on secondary y-axis
                        )
                        variables_in_legend.add(var)
                else:
                    if var not in variables_in_legend:
                        fig_line.add_trace(
                            go.Scatter(
                                x=data[date_column], 
                                y=data[var], 
                                mode='lines', 
                                name=var, 
                                line=dict(color=line_color)
                            ),
                            secondary_y=False  # Place on primary y-axis
                        )
                        variables_in_legend.add(var)

            # Update layout for dual-axis chart
            fig_line.update_layout(
                title="Selected Variables Over Time",
                xaxis_title="Date",
                yaxis_title=target_column,
                template='plotly_white',
                plot_bgcolor = 'rgba(0, 0, 0, 0)',paper_bgcolor='white',
                height=400,
                margin=dict(l=10, r=50, t=40, b=40),
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="top", 
                    y=-0.2,  # Position the legend below the graph
                    xanchor="center", 
                    x=0.5
                )
            )

            # Update y-axes titles
            fig_line.update_yaxes(title_text=target_column, secondary_y=False)
            fig_line.update_yaxes(title_text="Secondary Variables", secondary_y=True)

            # Display the figure in Streamlit with full width
            st.plotly_chart(fig_line, use_container_width=True)

        # Warning message if no variable is selected
        else:
            st.warning("Please select at least one variable to plot.")
if are_all_files_uploaded():
    with tab2:
        data = st.session_state.data
        model = st.session_state.model
        config = st.session_state.config
        prophet_path = st.session_state.prophet_path
        target_column = st.session_state.target_column
        extra_features_cols = st.session_state.extra_features_cols
        media_columns = st.session_state.media_columns
        date_column = st.session_state.date_column
        interval_type = st.session_state.interval_type
        promos = st.session_state.promos
        prophet_columns = st.session_state.prophet_columns
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        target_type = st.session_state.target_type
        # Retrieve the generated channel colors
        channel_colors = assign_colors_to_all(media_columns, extra_features_cols, promos, prophet_columns)

        first_row = st.columns([2, 1])
        with first_row[0]:  # Waterfall Chart spans 50% of the left column’s width
            waterfall_fig = plot_decomposition_plotly(
                model, data, target_column, extra_features_cols, media_columns, date_column, promos, prophet_columns, channel_colors
            )
            waterfall_fig .update_layout(margin=dict(r=80),height=400)
            st.plotly_chart(waterfall_fig, use_container_width=True)
        
        with first_row[1]:  # Posterior Predictive Check spans 50% of the right column’s width
            posterior_predictive = plot_posterior_predictive_with_r2(model, X_train, y_train, original_scale=True)
            posterior_predictive.update_layout(margin=dict(r=50,t=90),height=400)
            st.plotly_chart(posterior_predictive, use_container_width=True)

        second_row = st.columns([2, 1])
        with second_row[0]:  # Left 50% width
            st.container()
            total_contribution = all_contributions_plot_plotly(model, config, channel_colors)
            total_contribution.update_layout(height=400,margin=dict(l=40, r=50, t=40, b=70))
            st.plotly_chart(total_contribution, use_container_width=True)
        
        with second_row[1]:  # Right 50% width
            st.container()
            contr_share = plot_channel_contribution_share_pie(model, channel_colors)
            contr_share.update_layout(margin=dict(l=40, r=50, t=40, b=70),height=400)
            st.plotly_chart(contr_share, use_container_width=True)

        third_row = st.columns([2, 1])
        with third_row[0]: 
            st.container()
            responce_curves = plot_direct_contribution_curves(model,show_fit=True,channel_colors=channel_colors,xlim_max=1)
            responce_curves.update_layout(margin=dict(r=50,),height=400)
            st.plotly_chart(responce_curves, use_container_width=True)

        with third_row[1]: 
            st.container()
            roi = plot_roi_cpa_plotly(model, data, config, channel_colors,target_type=target_type)
            roi.update_layout(margin=dict(l=40, r=50, t=40, b=125,),height=400)
            st.plotly_chart(roi, use_container_width=True)

    with tab3:
        st.header('Scenario Planning')

        if (st.session_state.data is not None and st.session_state.model is not None and st.session_state.config is not None and 
            st.session_state.prophet_path is not None and st.session_state.target_column is not None):

            data = st.session_state.data
            model = st.session_state.model
            config = st.session_state.config
            prophet_path = st.session_state.prophet_path
            target_column = st.session_state.target_column
            extra_features_cols = st.session_state.extra_features_cols
            media_columns = st.session_state.media_columns
            date_column = st.session_state.date_column
            interval_type = st.session_state.interval_type
            promos = st.session_state.promos
            prophet_columns = st.session_state.prophet_columns
            last_date = st.session_state.last_date
            n_historical_default = st.session_state.n_historical_default
            X = data.drop(target_column, axis=1)
            y = data[target_column]

            n_new_default = 30  # Default number of new periods for predictions
            # Define a bright color palette for channels
            channel_colors = assign_colors_to_all(media_columns, extra_features_cols, promos, prophet_columns)

            if st.button("Forecasting Data Set Creation", key="forecast_button"):
                st.session_state.dialog_open = True

            # Create layout with columns and containers
            col1, col2 = st.columns(2, gap='small')
            col3, col4 = st.columns(2, gap='small')
            if st.session_state.prediction_triggered:
                n_historical = st.session_state.get('n_historical', n_historical_default)
            
                historical_data = X[pd.to_datetime(X[date_column]) >= (last_date - pd.DateOffset(days=st.session_state.get('n_historical', n_new_default) * (7 if interval_type == 'weekly' else 30 if interval_type == 'monthly' else 1)))]

                forecasted_data = st.session_state.get('X_out_of_sample', pd.DataFrame())
                with col1:
                    if 'X_out_of_sample' in st.session_state:
                        X_out_of_sample = st.session_state.X_out_of_sample
                        n_historical = st.session_state.n_historical
                        daily_budget = st.session_state.daily_budget
                        last_date = st.session_state.last_date
                        y_out_of_sample = model.sample_posterior_predictive(X_pred=X_out_of_sample, extend_idata=False)

                        actual_data = pd.DataFrame({date_column: X[date_column], 'actuals': y})
                        actual_data = actual_data[pd.to_datetime(actual_data[date_column]) >= (last_date - pd.DateOffset(days=n_historical * (7 if interval_type == 'weekly' else 30 if interval_type == 'monthly' else 1)))]

                        y_out_of_sample_groupby = y_out_of_sample["y"].to_series().groupby("date")
                        lower, upper = quantiles = [0.025, 0.975]
                        conf = y_out_of_sample_groupby.quantile(quantiles).unstack()
                        mean = y_out_of_sample_groupby.mean()


                    st.container()
                    channel_contributions_df, intercept_df, control_contributions_df = new_data_media_contributions(X_out_of_sample, model)
                    base_contributions_df = pd.concat([intercept_df, control_contributions_df])
                    base_contributions_df["channel"] = "Base"
                    overall_contributions_df = pd.concat([channel_contributions_df, base_contributions_df])
                    combined_contributions = pd.concat([channel_contributions_df, intercept_df, control_contributions_df])
                    combined_contributions['contribution'] = combined_contributions['contribution'].clip(lower=0)
                    combined_contributions['date'] = pd.to_datetime(combined_contributions['date'])
                    channel_contributions_pivot = combined_contributions.pivot_table(index='date', columns='channel', values='contribution', aggfunc='mean')
                    fig_contributions = out_of_sample_all_contributions_plot(channel_contributions_pivot, date_column, 
                                                                            extra_features_cols=extra_features_cols, 
                                                                            prophet_columns=prophet_columns, promos=promos, channel_colors=channel_colors)
                    ensure_legends_under_graph(fig_contributions)
                    fig_contributions.update_layout(height=400)
                    st.plotly_chart(fig_contributions, use_container_width=True)

                    with col2:
                        st.container()
                        fig_new_spend_contributions = plot_new_spend_contributions_plotly(
                            model=model, spend_amount=daily_budget, channel_colors=channel_colors, one_time=True
                        )
                        ensure_legends_under_graph(fig_new_spend_contributions)
                        fig_new_spend_contributions.update_layout(height=400)
                        st.plotly_chart(fig_new_spend_contributions, use_container_width=True)
                    with col3:
                        st.container()
                        fig_prediction = go.Figure()
                        fig_prediction.add_trace(go.Scatter(
                            x=actual_data[date_column], y=actual_data['actuals'], mode='lines+markers', name='Actuals', line=dict(color='blue')
                        ))
                        fig_prediction.add_trace(go.Scatter(
                            x=X_out_of_sample[date_column], y=conf[lower], mode='lines', name='Lower Interval', line=dict(color='orange', dash='dash'), showlegend=False
                        ))
                        fig_prediction.add_trace(go.Scatter(
                            x=X_out_of_sample[date_column], y=conf[upper], mode='lines', name='Upper Interval', fill='tonexty', line=dict(color='orange', dash='dash'), showlegend=False
                        ))
                        fig_prediction.add_trace(go.Scatter(
                            x=X_out_of_sample[date_column], y=mean, mode='lines+markers', name='Mean Prediction', line=dict(color='orange')
                        ))
                        fig_prediction.update_layout(
                            title='Out-of-Sample Predictions', xaxis_title='Date', yaxis_title='Target', template='plotly_white', height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),plot_bgcolor = 'rgba(0, 0, 0, 0)',paper_bgcolor='white', 
                        )
                        ensure_legends_under_graph(fig_prediction)
                        st.plotly_chart(fig_prediction, use_container_width=True)

                    with col4:
                        st.container()
                        control_colors = [custom_palette[i % len(custom_palette)] for i in range(len(extra_features_cols + promos + prophet_columns))]
                        control_colors_dict = {control: color for control, color in zip(extra_features_cols + promos + prophet_columns, control_colors)}
                        fig_control_contributions = plot_control_contributions_plotly(control_contributions_df, 'Control Contributions Over Time', control_colors_dict)
                        ensure_legends_under_graph(fig_control_contributions)
                        fig_control_contributions.update_layout(height=400)
                        st.plotly_chart(fig_control_contributions, use_container_width=True)
                    # Reset the predictiron trigger
                    st.session_state.prediction_triggered = False

        if 'dialog_open' in st.session_state and st.session_state.dialog_open:
            st.session_state.dialog_open = False

            @st.dialog("Forecasting Data Set Creation", width='large')
            def forecasting_data_set_creation():
                st.header('Scenario Planning Settings')

                # Row 1: Periods and Budget
                col1, col2, = st.columns(2)
                col3, col4 = st.columns(2)

                with col1:
                    # Step 1: Handle Periods (Historical and Prediction)
                    if interval_type == 'daily':
                        historical_label = 'Historical Days'
                        prediction_label = 'Prediction Days'
                        historical_min = 1
                        historical_max = 365
                        prediction_min = 1
                        prediction_max = 90
                    elif interval_type == 'weekly':
                        historical_label = 'Historical Weeks'
                        prediction_label = 'Prediction Weeks'
                        historical_min = 1
                        historical_max = 52
                        prediction_min = 1
                        prediction_max = 12
                    elif interval_type == 'monthly':
                        historical_label = 'Historical Months'
                        prediction_label = 'Prediction Months'
                        historical_min = 1
                        historical_max = 24
                        prediction_min = 1
                        prediction_max = 6

                    with st.expander("Periods"):
                        n_historical_d = st.session_state.get('n_historical', 60)
                        n_new_d = st.session_state.get('n_new', 30)
                        n_historical = st.slider(historical_label, historical_min, historical_max, n_historical_d, key="historical_slider")
                        n_new = st.slider(prediction_label, prediction_min, prediction_max, n_new_d, key="new_slider")
                        # If prediction weeks (n_new) have changed, update the out-of-sample dataset
                        if 'prev_n_new' not in st.session_state:
                            st.session_state['prev_n_new'] = n_new
                        if n_new != st.session_state['prev_n_new']:
                            st.session_state.n_new = n_new
                            st.session_state.prev_n_new = n_new

                            # Regenerate the out-of-sample dataset based on the new number of prediction weeks
                            st.session_state.X_out_of_sample = generate_out_of_sample_data(
                                n_new, st.session_state.channel_spends, st.session_state.promo_periods, last_date,
                                interval_type, date_column, media_columns, extra_features_cols, promos, prophet_columns, prophet_path, X)

                with col2:
                        # Step 3: Scenario Modifiers
                        with st.expander("Scenario Modifiers"):
                            if 'scenario_method' not in st.session_state:
                                st.session_state['scenario_method'] = "Fixed Percentage Adjustment"
                            if 'adjusted_spends' not in st.session_state:
                                st.session_state['adjusted_spends'] = {}

                            interval_label = interval_type.capitalize()

                            # Use session state to remember the selected scenario method
                            st.session_state['scenario_method'] = st.selectbox(
                                "Select Scenario Generation Method:",
                                ["Fixed Percentage Adjustment", "Historical Pattern"],
                                index=["Fixed Percentage Adjustment", "Historical Pattern"].index(st.session_state['scenario_method'])
                            )

                            if st.session_state['scenario_method'] == "Fixed Percentage Adjustment":
                                # Step 1: Adjustment Percentage Slider
                                adjustment_percentage = st.slider(
                                    "Adjust media spends by percentage:",
                                    min_value=-100,
                                    max_value=100,
                                    value=st.session_state.get('adjustment_percentage', 0),
                                    key='adjustment_slider'
                                )
                                st.session_state['adjustment_percentage'] = adjustment_percentage
                                # Step 2: Initialize adjusted spends dictionary if not already present
                                period_start = last_date - pd.DateOffset(weeks=1) if interval_type == 'daily' else last_date - pd.DateOffset(weeks=3)
                                recent_data = X[(pd.to_datetime(X[date_column]) >= period_start) & (pd.to_datetime(X[date_column]) <= last_date)]
                                default_spend = recent_data[media_columns].mean()
                                if 'adjusted_spends' not in st.session_state:
                                    st.session_state['adjusted_spends'] = {}
                                if 'channel_spends' not in st.session_state or st.session_state.channel_spends is None:
                                    st.session_state.channel_spends = {
                                        col: (default_spend[col] if not np.isnan(default_spend[col]) else 0)
                                        for col in media_columns
        }
                                # Step 3: Calculate and update channel spends based on adjustment percentage
                                for column in media_columns:
                                    # Calculate new spend based on adjustment percentage
                                    default_spend_value = st.session_state.channel_spends.get(column, 0) * (1 + adjustment_percentage / 100)
                                    st.session_state['adjusted_spends'][column] = default_spend_value

                                # Step 4: Display and allow manual entry for each channel spend
                                for i in range(0, len(media_columns), 2):
                                    col1, col2 = st.columns(2,gap='small')

                                    with col1:
                                        column1 = media_columns[i]
                                        # Get adjusted spend value and allow manual entry
                                        manual_spend_value1 = st.number_input(
                                            f"{column1} Spend:",
                                            value=int(st.session_state['adjusted_spends'].get(column1, 0)),
                                            step=1,
                                            key=f"{column1}_spend_input"
                                        )
                                        # Update session state with manually entered value
                                        st.session_state['adjusted_spends'][column1] = manual_spend_value1

                                    if i + 1 < len(media_columns):
                                        with col2:
                                            column2 = media_columns[i + 1]
                                            manual_spend_value2 = st.number_input(
                                                f"{column2} Spend:",
                                                value=int(st.session_state['adjusted_spends'].get(column2, 0)),
                                                step=1,
                                                key=f"{column2}_spend_input"
                                            )
                                            st.session_state['adjusted_spends'][column2] = manual_spend_value2

                                # Step 5: Update channel spends in session state based on adjusted spends
                                st.session_state.channel_spends.update(st.session_state['adjusted_spends'])
                                daily_budget = sum(st.session_state['adjusted_spends'].values())
                                # Step 6: Apply the updated channel spends to X_out_of_sample
                                if 'X_out_of_sample' in st.session_state and not st.session_state.X_out_of_sample.empty:
                                    for column in media_columns:
                                        st.session_state.X_out_of_sample[column] = st.session_state.channel_spends[column]


                            elif st.session_state['scenario_method'] == "Historical Pattern":
                                # Historical pattern inputs
                                if st.session_state.get('start_date') is None:
                                    start_date=last_date - pd.DateOffset(weeks=4)
                                    st.session_state['start_date'] = start_date
                                if st.session_state.get('end_date') is None:
                                    end_date=last_date
                                    st.session_state['end_date'] = end_date
                                start_date = st.date_input("Select the start date for historical period", value=st.session_state['start_date'] )
                                end_date = st.date_input("Select the end date for historical period", value=st.session_state['end_date'])

                                # Update session state with the new values
                                st.session_state['start_date'] = start_date
                                st.session_state['end_date'] = end_date

                                if start_date < end_date:
                                    historical_data = X[(pd.to_datetime(X[date_column]) >= pd.to_datetime(start_date)) &
                                                        (pd.to_datetime(X[date_column]) <= pd.to_datetime(end_date))]
                                    if st.session_state['rolling_window'] is None:
                                        rolling_window=min(4, len(historical_data))
                                    rolling_window = st.slider("Select rolling window size (weeks)", 1, len(historical_data),
                                                            value=st.session_state.get('rolling_window', min(4, len(historical_data))))
                                    st.session_state['rolling_window'] = rolling_window

                                    rolling_mean_spends = historical_data[media_columns].rolling(window=rolling_window).mean().dropna()

                                    if not rolling_mean_spends.empty:
                                        num_repeats = (st.session_state.n_new // len(rolling_mean_spends)) + 1
                                        updated_spends = pd.concat([rolling_mean_spends] * num_repeats, ignore_index=True).head(st.session_state.n_new)
                                        st.session_state.X_out_of_sample.update(updated_spends)

                                        daily_budget = updated_spends.sum().sum()
                                        st.write(f"Total {interval_label} Budget (Historical Pattern): {int(daily_budget)}")

                            st.session_state.daily_budget = daily_budget

                # Step 4: Apply Promos
                with col3:
                    with st.expander("Promos"):
                        if 'promo_periods' not in st.session_state:
                            st.session_state.promo_periods = []

                        promo_periods = []
                        num_periods = st.number_input("Number of Promo Periods", min_value=1, max_value=10, value=1, step=1)

                        for i in range(num_periods):
                            with st.container():
                                st.subheader(f"Promo Period {i+1}")
                                promo_start_date = st.date_input(f"Promo Start Date {i+1}", key=f"promo_start_date_{i}")
                                promo_end_date = st.date_input(f"Promo End Date {i+1}", key=f"promo_end_date_{i}")
                                selected_promos = st.multiselect(f"Select Promos for Period {i+1}", promos, key=f"selected_promos_{i}")

                                if promo_start_date and promo_end_date:
                                    promo_periods.append((promo_start_date, promo_end_date, selected_promos))

                        st.session_state.promo_periods = promo_periods

                        if 'X_out_of_sample' in st.session_state and not st.session_state.X_out_of_sample.empty:
                            for promo in promos:
                                st.session_state.X_out_of_sample[promo] = 0
                            for promo_start_date, promo_end_date, selected_promos in st.session_state.promo_periods:
                                promo_dates_range = pd.date_range(start=promo_start_date, end=promo_end_date, freq='D')
                                for promo_date in promo_dates_range:
                                    if promo_date in st.session_state.X_out_of_sample[date_column].values:
                                        for promo in selected_promos:
                                                st.session_state.X_out_of_sample.loc[st.session_state.X_out_of_sample[date_column] == promo_date, promo] = 1

                # Rest of the code remains unchanged
                with col4:
                    with st.expander("Upload Your Own Data File"):
                        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
                        if uploaded_file is not None:
                            try:
                                user_data = pd.read_csv(uploaded_file)
                                required_columns = media_columns + [date_column] + extra_features_cols + promos + prophet_columns
                                missing_columns = [col for col in required_columns if col not in user_data.columns]

                                if missing_columns:
                                    st.error(f"Missing columns: {', '.join(missing_columns)}. Please upload a file with all required columns.")
                                else:
                                    st.success("File successfully uploaded and validated.")
                                    user_data[date_column] = pd.to_datetime(user_data[date_column])
                                    st.session_state.X_out_of_sample = user_data
                                    st.dataframe(user_data.head())
                            except Exception as e:
                                st.error(f"Error processing the uploaded file: {e}")
                        else:
                            if st.session_state.X_out_of_sample.empty:
                                st.session_state.X_out_of_sample = generate_out_of_sample_data(
                                    n_new, st.session_state.channel_spends, st.session_state.promo_periods, last_date,
                                    interval_type, date_column, media_columns, extra_features_cols, promos, prophet_columns, prophet_path, X)
                        
                # Step 6: Data Editor
                X_out_of_sample_table_width = calculate_dialog_table_width(st.session_state.X_out_of_sample)
                styled_x_out_of_sample_table = st.session_state.X_out_of_sample.style.set_properties(**{'background-color': 'white'})
                edited_data = st.data_editor(styled_x_out_of_sample_table, height=200, width=X_out_of_sample_table_width, hide_index=True)

                st.session_state.X_out_of_sample = edited_data

                # Step 7: Visualization
                historical_data = X[pd.to_datetime(X[date_column]) >= (last_date - pd.DateOffset(days=n_historical * (7 if interval_type == 'weekly' else 30 if interval_type == 'monthly' else 1)))]
                fig_combined = plot_historical_forecasted_spend(historical_data, edited_data, date_column, media_columns, channel_colors, last_date)
                fig_combined.update_layout(height=400)
                st.plotly_chart(fig_combined, use_container_width=True)
                col5,col6=st.columns([1,12])
                # Submit and Reset buttons
                with col5:
                # Submit button to store all changes in session state
                    if st.button('Submit'):
                        st.session_state['scenario_method'] = st.session_state['scenario_method']
                        st.session_state['adjusted_spends'] = st.session_state['adjusted_spends']
                        st.session_state['adjustment_percentage'] = st.session_state['adjustment_percentage']
                        st.session_state['start_date'] = st.session_state['start_date']
                        st.session_state['end_date'] = st.session_state['end_date']
                        st.session_state['rolling_window'] = st.session_state['rolling_window']
                        st.session_state.n_new = n_new
                        st.session_state.X_out_of_sample = edited_data
                        st.session_state.n_historical = n_historical
                        st.session_state.daily_budget = daily_budget
                        st.session_state.last_date = last_date
                        st.session_state.prediction_triggered = True
                        st.rerun()
                with col6:
                    if st.button('Reset Forecast Dataset', key='reset_button_dialog'):
                        st.session_state.X_out_of_sample = pd.DataFrame()
                        st.session_state['scenario_method'] = "Fixed Percentage Adjustment"
                        st.session_state['adjusted_spends'] = {}
                        st.session_state['adjustment_percentage'] = 0
                        st.session_state['start_date'] = last_date - pd.DateOffset(weeks=4)
                        st.session_state['end_date'] = last_date
                        st.session_state['rolling_window'] = min(4, len(X))
                        st.rerun()

            forecasting_data_set_creation()
else:
    with tab2:
        st.warning(f"Please upload missing files on Data Load and Inspect tab")
    with tab3:
        st.warning(f"Please upload missing files on Data Load and Inspect tab")