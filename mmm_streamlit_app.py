import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import holidays
import os
import plotly.graph_objects as go
import xarray as xr
import pymc as pm
import tempfile
import yaml
import matplotlib.pyplot as plt

from pymc_marketing.mmm import DelayedSaturatedMMM

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

# Streamlit interface
st.set_page_config(layout="wide")  # Set the layout to wide mode

st.title('MMM Scenario Planning and Optimization')

# Tabs for Data Upload and Scenario Planning
tab1, tab2 = st.tabs(["Data Upload", "Scenario Planning"])

with tab1:
    st.header('Upload Data for Analysis')

    uploaded_data_file = st.file_uploader("Upload your data file (CSV)", type="csv")
    uploaded_model_file = st.file_uploader("Upload your model file (NetCDF)", type="nc")
    uploaded_config_file = st.file_uploader("Upload your config file (YAML)", type="yml")
    uploaded_prophet_file = st.file_uploader("Upload your Prophet model file (PKL)", type="pkl")

    if uploaded_data_file and uploaded_model_file and uploaded_config_file and uploaded_prophet_file:
        # Save temporary files for model, config, and prophet model
        with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
            temp_model_file.write(uploaded_model_file.read())
            model_path = temp_model_file.name

        with tempfile.NamedTemporaryFile(delete=False) as temp_config_file:
            temp_config_file.write(uploaded_config_file.read())
            config_path = temp_config_file.name

        with tempfile.NamedTemporaryFile(delete=False) as temp_prophet_file:
            temp_prophet_file.write(uploaded_prophet_file.read())
            prophet_path = temp_prophet_file.name

        # Load configuration
        config = load_config(config_path)

        # Determine prophet columns based on config
        prophet_columns = determine_prophet_columns(config)

        # Extract parameters from the config
        target_column = config.get('target_col', 'subscribers')
        extra_features_cols = config.get('extra_features_cols', [])
        media_columns = [entry['spend_col'] for entry in config.get('media', [])]
        date_column = config.get('date_col', 'date')
        interval_type = config['raw_data_granularity']
        promos = config.get('promos', [])
        if 'promo' in extra_features_cols:
            promos.append('promo')
        if 'promo_events' in extra_features_cols:
            promos.append('promo_events')

        data = pd.read_csv(uploaded_data_file)
        st.markdown("**Files Uploaded Successfully**")

        # Display the data
        st.subheader('Uploaded Data')
        st.write(data)

        # Ensure date column is in datetime format
        data[date_column] = pd.to_datetime(data[date_column])

        # Show data statistics
        st.subheader('Data Statistics')
        st.write(data.describe())

        # Download data button
        st.download_button(
            label="Download Data as CSV",
            data=data.to_csv(index=False).encode('utf-8'),
            file_name='uploaded_data.csv',
            mime='text/csv',
        )

        # Load model
        model = DelayedSaturatedMMM.load(model_path)

        # Ensure last_date is in datetime format
        last_date = pd.to_datetime(data[date_column].max())

        freq = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'}.get(interval_type)
        if not freq:
            raise ValueError("Invalid interval_type. Choose from 'daily', 'weekly', or 'monthly'.")

with tab2:
    st.header('Scenario Planning')

    # Load the uploaded files
    if 'data' in locals() and 'model' in locals() and 'config' in locals() and 'prophet_path' in locals():
        
        X = data.drop(target_column, axis=1)
        y = data[target_column]  # Ensure target column is loaded

        n_new_default = 30  # Default number of new periods for predictions

        # Load and predict with Prophet model
        def load_and_predict_prophet(prophet_path, future_dates):
            model = joblib.load(prophet_path)
            future = pd.DataFrame({'ds': future_dates})
            
            # Debugging information using Streamlit
            st.subheader("Future DataFrame for Prophet prediction:")
            st.write(future.head())
            
            # Check if future DataFrame has rows
            if future.empty:
                st.error("The future DataFrame is empty. Please check the date range generation.")
                return pd.DataFrame()  # Return an empty DataFrame to avoid further errors
            
            forecast = model.predict(future)
            return forecast

        # Create features
        def create_features(df):
            df = df.copy()
            control_columns = extra_features_cols

            if 'holidays' in control_columns:
                us_holidays = holidays.US(years=[min(df[date_column].dt.year), max(df[date_column].dt.year)])
                df['holidays'] = df[date_column].apply(lambda x: 1 if x in us_holidays else 0)
            if 'dayofweek' in control_columns:
                df['dayofweek'] = df[date_column].dt.dayofweek
            if 'weekofyear' in control_columns:
                df['weekofyear'] = df[date_column].dt.isocalendar().week.astype(int)
            if 'dayofmonth' in control_columns:
                df['dayofmonth'] = df[date_column].dt.day
            return df

        # Scale Prophet columns
        def scale_prophet_columns(data):
            scaler = MinMaxScaler()
            for feature in prophet_columns:
                if feature in data.columns:
                    data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
            return data

        # Scenario strategies
        def last_known_value(column):
            return X[column].iloc[-1]

        def apply_growth_rate(column, growth_rate=0.01):
            if column not in promos:  # Exclude promos from growth rate application
                return X[column].iloc[-1] * (1 + growth_rate)
            else:
                return X[column].iloc[-1]

        # Generate out-of-sample data based on user input

        def generate_out_of_sample_data(n_new, channel_spends, scenario, promo_periods, custom_spending_patterns):
            new_dates = pd.date_range(start=last_date, periods=1 + n_new, freq=freq)[1:]
        
            if interval_type == 'weekly':
                new_dates = new_dates[new_dates.weekday == 6]  # Ensure all dates start on Sunday
        
            # Debugging information using Streamlit
            st.subheader("Generated new_dates:")
            st.write(new_dates)
        
            if new_dates.empty:
                st.error("Generated new_dates is empty. Please check the date range generation.")
                return pd.DataFrame()  # Return an empty DataFrame to avoid further errors
        
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
        
            for pattern in custom_spending_patterns:
                channel, start_date, end_date, spend_value = pattern
                date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
                for date in date_range:
                    if date in X_out_of_sample[date_column].values:
                        X_out_of_sample.loc[X_out_of_sample[date_column] == date, channel] = spend_value
        
            for column in extra_features_cols:
                if column in X.columns:
                    if X[column].dtype == 'float':
                        X_out_of_sample[column] = X[column].iloc[-1] * (1 + np.random.normal(0, 0.05, size=n_new))
                    else:
                        X_out_of_sample[column] = X[column].iloc[-1]
        
            if prophet_columns:
                forecast = load_and_predict_prophet(prophet_path, new_dates)
                if not forecast.empty:
                    for column in prophet_columns:
                        if column in forecast.columns:
                            X_out_of_sample[column] = forecast['yhat'].values
                        else:
                            st.warning(f"Column '{column}' not found in Prophet forecast.")
                else:
                    st.error("Prophet forecast is empty. Skipping Prophet columns.")
        
            if scenario['random_events']:
                for column in X_out_of_sample.columns:
                    if column != date_column and column not in promos:
                        X_out_of_sample[column] *= (1 + np.random.normal(0, 0.1, size=n_new))
        
            if scenario['add_noise']:
                for column in X_out_of_sample.columns:
                    if column != date_column and column not in promos:
                        X_out_of_sample[column] *= (1 + np.random.normal(0, 0.05, size=n_new))
        
            X_out_of_sample = create_features(X_out_of_sample)
            X_out_of_sample = scale_prophet_columns(X_out_of_sample)
        
            # Debugging information using Streamlit
            st.subheader("Generated out-of-sample data:")
            st.write(X_out_of_sample.head())
        
            return X_out_of_sample

        def new_data_media_contributions(X: pd.DataFrame, mmm: DelayedSaturatedMMM, original_scale: bool = True):
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

        # Define a bright color palette for channels
        bright_colors = [
            "rgb(31, 119, 180)", "rgb(255, 127, 14)", "rgb(44, 160, 44)",
            "rgb(214, 39, 40)", "rgb(148, 103, 189)", "rgb(140, 86, 75)",
            "rgb(227, 119, 194)", "rgb(127, 127, 127)", "rgb(188, 189, 34)",
            "rgb(23, 190, 207)"
        ]

        channel_colors = {channel: bright_colors[i % len(bright_colors)] for i, channel in enumerate(media_columns)}
        channel_colors["Base"] = "rgb(169, 169, 169)"  # Grey color for Base
        channel_colors["intercept"] = "rgb(169, 169, 169)"  # Grey color for intercept

        # Function to plot contributions using Plotly
        def plot_contributions_plotly(contributions_df: pd.DataFrame, title: str, group_base: bool = False):
            contributions_df['date'] = pd.to_datetime(contributions_df['date'])

            if group_base:
                base_cols = ['intercept'] + extra_features_cols + prophet_columns
                existing_base_cols = [col for col in base_cols if col in contributions_df['channel'].unique()]
                if existing_base_cols:
                    base_df = contributions_df[contributions_df['channel'].isin(existing_base_cols)]
                    if not base_df.empty:
                        base_contributions = base_df.groupby('date')['contribution'].sum().reset_index()
                        base_contributions['channel'] = 'Base'
                        contributions_df = pd.concat([contributions_df[~contributions_df['channel'].isin(existing_base_cols)], base_contributions])
                    else:
                        st.warning("No existing base columns found in the contributions DataFrame.")
            
            contributions_pivot = contributions_df.pivot_table(index='date', columns='channel', values='contribution', aggfunc='mean')

            fig = go.Figure()

            for channel in contributions_pivot.columns:
                fig.add_trace(go.Scatter(
                    x=contributions_pivot.index,
                    y=contributions_pivot[channel],
                    mode='lines',
                    name=channel,
                    stackgroup='one',
                    line=dict(color=channel_colors.get(channel, 'rgb(0,0,0)'))  # Use channel_colors
                ))

            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Contribution",
                template="plotly_white"
            )

            return fig

        # Function to plot new spend contributions using Plotly
        def plot_new_spend_contributions_plotly(model, spend_amount, one_time=True, lower=0.025, upper=0.975, ylabel="Sales", channels=None, prior=False, original_scale=True):
            total_channels = len(model.channel_columns)
            contributions = model.new_spend_contributions(
                np.ones(total_channels) * spend_amount,
                one_time=one_time,
                spend_leading_up=np.ones(total_channels) * spend_amount,
                prior=prior,
                original_scale=original_scale,
            )

            contributions_groupby = contributions.to_series().groupby(level=["time_since_spend", "channel"])

            idx = pd.IndexSlice[0:]

            conf = (
                contributions_groupby.quantile([lower, upper])
                .unstack("channel")
                .unstack()
                .loc[idx]
            )

            channels = channels or model.channel_columns

            fig = go.Figure()

            for channel in channels:
                fig.add_trace(go.Scatter(
                    x=conf.index,
                    y=conf[channel][upper],
                    mode='lines',
                    name=f"{channel} Upper CI",
                    line=dict(width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=conf.index,
                    y=conf[channel][lower],
                    mode='lines',
                    name=f"{channel} Lower CI",
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f"rgba{channel_colors[channel][3:-1]},0.2)",
                    showlegend=False
                ))

            mean = contributions_groupby.mean().unstack("channel").loc[idx, channels]

            for i, channel in enumerate(channels):
                fig.add_trace(go.Scatter(
                    x=mean.index,
                    y=mean[channel],
                    mode='lines',
                    name=f"{channel} Mean",
                    line=dict(color=channel_colors.get(channel, f"rgb({i*40 % 256}, {i*80 % 256}, {i*120 % 256})"))
                ))

            fig.update_layout(
                title=f"Upcoming Sales for {spend_amount:.02f} Spend",
                xaxis_title="Time since spend",
                yaxis_title=ylabel,
                template='plotly_white'
            )

            return fig

        # Function to plot overall contributions using Plotly
        def out_of_sample_all_contributions_plot(contributions_pivot, date_column='date'):
            # Define base columns that should be included in the 'Base' category
            base_cols = ['intercept'] + extra_features_cols + prophet_columns + promos
            
            # Identify any missing base columns in the contributions_pivot DataFrame
            missing_cols = [col for col in base_cols if col not in contributions_pivot.columns]
            if missing_cols:
                st.warning(f"Missing base columns: {missing_cols}")
        
            # Find available base columns that are present in the contributions_pivot DataFrame
            available_base_cols = [col for col in base_cols if col in contributions_pivot.columns]
            
            # Sum the contributions of the available base columns to create a 'Base' column
            contributions_pivot['Base'] = contributions_pivot[available_base_cols].sum(axis=1)
            
            # Drop the individual base columns from the DataFrame, keeping only the 'Base' column
            contributions_pivot = contributions_pivot.drop(columns=available_base_cols, errors='ignore')
        
            # Order the remaining columns by their total contribution in descending order
            column_order = contributions_pivot.drop('Base', axis=1).sum(axis=0).sort_values(ascending=False).index
        
            # Initialize a Plotly figure
            fig = go.Figure()
        
            # Add the 'Base' contributions as a stacked area plot
            fig.add_trace(go.Scatter(
                x=contributions_pivot.index,
                y=contributions_pivot['Base'],
                mode='lines',
                name='Base',
                stackgroup='one',
                line=dict(color='grey', width=0),
                fill='tonexty'
            ))
        
            # Add each channel's contributions as a stacked area plot
            for channel in column_order:
                fig.add_trace(go.Scatter(
                    x=contributions_pivot.index,
                    y=contributions_pivot[channel],
                    mode='lines',
                    name=channel,
                    stackgroup='one',
                    line=dict(color=channel_colors.get(channel, 'rgb(0,0,0)'))  # Use channel_colors
                ))
        
            # Update the layout of the figure
            fig.update_layout(
                title='Total Contributions Including Base Over Time',
                xaxis_title="Date",
                yaxis_title="Revenue Contribution",
                template="plotly_white"
            )
        
            return fig

        # Determine default values and ranges for sliders based on interval type
        if interval_type == 'daily':
            historical_label = 'Historical Days'
            prediction_label = 'Prediction Days'
            historical_min = 1
            historical_max = 365
            historical_default = 60
            prediction_min = 1
            prediction_max = 90
            prediction_default = n_new_default
        elif interval_type == 'weekly':
            historical_label = 'Historical Weeks'
            prediction_label = 'Prediction Weeks'
            historical_min = 1
            historical_max = 52
            historical_default = 8
            prediction_min = 1
            prediction_max = 12
            prediction_default = n_new_default // 7
        elif interval_type == 'monthly':
            historical_label = 'Historical Months'
            prediction_label = 'Prediction Months'
            historical_min = 1
            historical_max = 24
            historical_default = 12
            prediction_min = 1
            prediction_max = 6
            prediction_default = n_new_default // 30

        # Move selectors to the sidebar
        st.sidebar.header('Scenario Planning Settings')

        # Number of historical and prediction periods
        n_historical = st.sidebar.slider(historical_label, historical_min, historical_max, historical_default)
        n_new = st.sidebar.slider(prediction_label, prediction_min, prediction_max, prediction_default)

        # Custom scenario settings
        st.sidebar.header('Custom Scenario Settings')
        channel_spends = {}
        budget_change_percentage = st.sidebar.slider('Adjust Total Budget (%)', -100, 100, 0)

        if interval_type == 'daily':
            period_start = last_date - pd.DateOffset(weeks=1)
        else:
            period_start = last_date - pd.DateOffset(weeks=3)

        recent_data = X[(pd.to_datetime(X[date_column]) >= period_start) & (pd.to_datetime(X[date_column]) <= last_date)]
        default_spend = recent_data[media_columns].mean()
        adjusted_total_budget = default_spend.sum() * (1 + budget_change_percentage / 100)

        for column in media_columns:
            adjusted_spend = adjusted_total_budget * (default_spend[column] / default_spend.sum())
            max_value = int(adjusted_spend * 3) if adjusted_spend > 0 else 150
            default_value = int(adjusted_spend) if adjusted_spend > 0 else 50
            channel_spends[column] = st.sidebar.slider(f'{column} Spend', 0, max_value, default_value)

        total_budget_period = sum(channel_spends.values()) * n_new
        daily_budget = total_budget_period / n_new

        custom_spending_patterns = []
        custom_spending = st.sidebar.checkbox('Custom Spending Pattern', value=False)
        if custom_spending:
            num_patterns = st.sidebar.number_input("Number of Custom Spending Patterns", min_value=1, max_value=10, value=1, step=1, format='%d')
            for i in range(num_patterns):
                st.sidebar.subheader(f"Custom Spending Pattern {i+1}")
                selected_channel = st.sidebar.selectbox(f"Select Channel for Pattern {i+1}", media_columns, key=f"channel_{i}")
                pattern_start_date = st.sidebar.date_input(f"Select Pattern Start Date {i+1}")
                pattern_end_date = st.sidebar.date_input(f"Select Pattern End Date {i+1}")
                spend_value = st.sidebar.number_input(f"Spending Value for Pattern {i+1}", min_value=0, value=100, step=10, format='%d')
                if pattern_start_date and pattern_end_date and selected_channel:
                    custom_spending_patterns.append((selected_channel, pattern_start_date, pattern_end_date, spend_value))

        if custom_spending_patterns:
            total_custom_spend = 0
            for pattern in custom_spending_patterns:
                _, start_date, end_date, spend_value = pattern
                total_days = (end_date - start_date).days + 1
                total_custom_spend += total_days * spend_value
            total_budget_period += total_custom_spend
            daily_budget = total_budget_period / n_new

        if budget_change_percentage > 0:
            scenario_description = f"High spend scenario with a {budget_change_percentage}% increase in all channels."
        elif budget_change_percentage < 0:
            scenario_description = f"Low spend scenario with a {budget_change_percentage}% decrease in all channels."
        else:
            scenario_description = "Baseline scenario with current spend levels."

        st.sidebar.markdown(f"**Scenario Description:** {scenario_description}")
        st.sidebar.markdown(f"**Total Budget for Period:** ${total_budget_period:.2f}")
        st.sidebar.markdown(f"**Daily Budget:** ${daily_budget:.2f}")

        promo_events = st.sidebar.checkbox('Include Promo Events', value=False)
        random_events = st.sidebar.checkbox('Include Random Events', value=False)
        add_noise = st.sidebar.checkbox('Add Noise', value=False)

        promo_periods = []
        if promo_events:
            num_periods = st.sidebar.number_input("Number of Promo Periods", min_value=1, max_value=10, value=1, step=1, format='%d')
            for i in range(num_periods):
                st.sidebar.subheader(f"Promo Period {i+1}")
                promo_start_date = st.sidebar.date_input(f"Select Promo Start Date {i+1}")
                promo_end_date = st.sidebar.date_input(f"Select Promo End Date {i+1}")
                selected_promos = st.sidebar.multiselect(f"Select Promos for Period {i+1}", promos, key=f"{i}")
                if promo_start_date and promo_end_date and selected_promos:
                    promo_periods.append((promo_start_date, promo_end_date, selected_promos))

        if st.sidebar.button('Apply Scenario'):
            custom_scenario = {
                'promo_events': promo_events,
                'random_events': random_events,
                'add_noise': add_noise
            }
            X_out_of_sample = generate_out_of_sample_data(n_new, channel_spends, custom_scenario, promo_periods, custom_spending_patterns)
        else:
            custom_scenario = {
                'promo_events': promo_events,
                'random_events': random_events,
                'add_noise': add_noise
            }
            X_out_of_sample = generate_out_of_sample_data(n_new, channel_spends, custom_scenario, promo_periods, custom_spending_patterns)

        st.subheader(f'Historical Marketing Spend Over Last {n_historical} periods, delta =  {interval_type.capitalize()}')
        fig_historical = go.Figure()

        historical_data = X[pd.to_datetime(X[date_column]) >= (last_date - pd.DateOffset(days=n_historical * (7 if interval_type == 'weekly' else 30 if interval_type == 'monthly' else 1)))]
        for column in media_columns:
            fig_historical.add_trace(go.Scatter(
                x=historical_data[date_column],
                y=historical_data[column],
                mode='lines+markers',
                name=f'Historical {column}',
                fill='tozeroy',
                line=dict(color=channel_colors[column])
            ))

        fig_historical.update_layout(
            title=f'Historical Marketing Spend Over Last {n_historical} {interval_type.capitalize()}s',
            xaxis_title='Date',
            yaxis_title='Spend',
            template='plotly_white'
        )

        st.plotly_chart(fig_historical)

        st.subheader(f'Forecasted Marketing Spend for Next {n_new} {interval_type.capitalize()}s')
        fig_forecast = go.Figure()

        for column in media_columns:
            fig_forecast.add_trace(go.Scatter(
                x=X_out_of_sample[date_column],
                y=X_out_of_sample[column],
                mode='lines+markers',
                name=f'Forecast {column}',
                fill='tozeroy',
                line=dict(color=channel_colors[column])
            ))

        if promo_events:
            for promo_period in promo_periods:
                promo_start_date, promo_end_date, selected_promos = promo_period
                promo_dates_range = pd.date_range(start=promo_start_date, end=promo_end_date, freq=freq)
                for promo_date in promo_dates_range:
                    for promo in selected_promos:
                        fig_forecast.add_trace(go.Scatter(
                            x=[promo_date],
                            y=[max(X_out_of_sample[column].max() for column in media_columns)],
                            mode='markers',
                            marker=dict(size=6, symbol='circle', color='blue'),
                            name=f'Forecast {promo} Events',
                            showlegend=False
                        ))

        fig_forecast.update_layout(
            title=f'Forecasted Marketing Spend for Next {n_new} {interval_type.capitalize()}s',
            xaxis_title='Date',
            yaxis_title='Spend',
            template='plotly_white'
        )

        st.plotly_chart(fig_forecast)

        st.subheader('Out-of-Sample Data')
        st.write(X_out_of_sample)

        st.header('Predictions')

        if st.button('Predict'):
            y_out_of_sample = model.sample_posterior_predictive(
                X_pred=X_out_of_sample, extend_idata=False
            )

            st.subheader('Out-of-Sample Predictions')
            fig_prediction = go.Figure()

            actual_data = pd.DataFrame({date_column: X[date_column], 'actuals': y})
            actual_data = actual_data[pd.to_datetime(actual_data[date_column]) >= (last_date - pd.DateOffset(days=n_historical * (7 if interval_type == 'weekly' else 30 if interval_type == 'monthly' else 1)))]
            fig_prediction.add_trace(go.Scatter(
                x=actual_data[date_column],
                y=actual_data['actuals'],
                mode='lines+markers',
                name='Actuals',
                line=dict(color='white')
            ))

            y_out_of_sample_groupby = y_out_of_sample["y"].to_series().groupby("date")
            lower, upper = quantiles = [0.025, 0.975]
            conf = y_out_of_sample_groupby.quantile(quantiles).unstack()
            fig_prediction.add_trace(go.Scatter(
                x=X_out_of_sample[date_column],
                y=conf[lower],
                mode='lines',
                name='Lower Interval',
                line=dict(color='blue', dash='dash')
            ))
            fig_prediction.add_trace(go.Scatter(
                x=X_out_of_sample[date_column],
                y=conf[upper],
                mode='lines',
                name='Upper Interval',
                fill='tonexty',
                line=dict(color='blue', dash='dash')
            ))

            mean = y_out_of_sample_groupby.mean()
            fig_prediction.add_trace(go.Scatter(
                x=X_out_of_sample[date_column],
                y=mean,
                mode='lines+markers',
                name='Mean Prediction',
                line=dict(color='blue')
            ))

            fig_prediction.update_layout(
                title='Out-of-Sample Predictions',
                xaxis_title='Date',
                yaxis_title='Target',
                template='plotly_white'
            )

            st.plotly_chart(fig_prediction)

            channel_contributions_df, intercept_df, control_contributions_df = new_data_media_contributions(X_out_of_sample, model)

            base_contributions_df = pd.concat([intercept_df, control_contributions_df])
            base_contributions_df["channel"] = "Base"
            overall_contributions_df = pd.concat([channel_contributions_df, base_contributions_df])

            st.subheader('Channel Contributions Over Time for New Spend')
            spend_amount = daily_budget
            one_time = True

            fig_new_spend_contributions = plot_new_spend_contributions_plotly(
                model=model,
                spend_amount=spend_amount,
                one_time=one_time
            )
            st.plotly_chart(fig_new_spend_contributions)
            
            st.subheader('Overall Contributions')
            combined_contributions = pd.concat([channel_contributions_df, intercept_df, control_contributions_df])
            combined_contributions['contribution'] = combined_contributions['contribution'].clip(lower=0)
            combined_contributions['date'] = pd.to_datetime(combined_contributions['date'])
            channel_contributions_pivot = combined_contributions.pivot_table(index='date', columns='channel', values='contribution', aggfunc='mean')
            fig_contributions = out_of_sample_all_contributions_plot(channel_contributions_pivot, date_column)
            st.plotly_chart(fig_contributions)

            st.subheader('Channel Contributions')
            fig_channel_contributions = plot_contributions_plotly(channel_contributions_df, 'Channel Contributions Over Time')
            st.plotly_chart(fig_channel_contributions)

            control_colors = [bright_colors[i % len(bright_colors)] for i in range(len(extra_features_cols + promos + prophet_columns))]
            control_colors_dict = {control: color for control, color in zip(extra_features_cols + promos + prophet_columns, control_colors)}

            def plot_control_contributions_plotly(contributions_df: pd.DataFrame, title: str):
                contributions_df['date'] = pd.to_datetime(contributions_df['date'])
                contributions_pivot = contributions_df.pivot_table(index='date', columns='channel', values='contribution', aggfunc='mean')
                fig = go.Figure()

                for control in contributions_pivot.columns:
                    fig.add_trace(go.Scatter(
                        x=contributions_pivot.index,
                        y=contributions_pivot[control],
                        mode='lines',
                        name=control,
                        stackgroup='one',
                        line=dict(color=control_colors_dict.get(control, bright_colors[0]))
                    ))

                fig.update_layout(
                    title=title,
                    xaxis_title="Date",
                    yaxis_title="Contribution",
                    template="plotly_white"
                )
                return fig

            st.subheader('Control Contributions')
            fig_control_contributions = plot_control_contributions_plotly(control_contributions_df, 'Control Contributions Over Time')
            st.plotly_chart(fig_control_contributions)

    else:
        st.markdown("**Please upload all required files in the Data Upload tab**")
