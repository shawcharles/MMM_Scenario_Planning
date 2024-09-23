import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import arviz as az
from xarray import DataArray
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from pymc_marketing.mmm.utils import (
    apply_sklearn_transformer_across_dim,
)
from typing import Any
import plotly.subplots as psp
from matplotlib.colors import to_rgb
import plotly.express as px
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
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

# Function to generate colors dynamically
def generate_color_palette(num_colors):
    # Generate a color palette dynamically (using a color palette like Plotly default or 'tab20')
    colors = px.colors.qualitative.Plotly
    if num_colors > len(colors):
        colors = px.colors.qualitative.Alphabet  # Fallback to another palette for more colors
    return [colors[i % len(colors)] for i in range(num_colors)]

# Function to lighten colors for filling in the bar chart or pie chart
def lighten_color(color, amount=0.5):
    c = np.array(to_rgb(color))
    white = np.array([1, 1, 1])
    new_color = c + (white - c) * amount
    return f'rgba({int(new_color[0]*255)}, {int(new_color[1]*255)}, {int(new_color[2]*255)}, 0.8)'



def get_channel_contributions_share_samples(fit_result, target_transformer) -> DataArray:
    channel_contribution = az.extract(
        data=fit_result, var_names=["channel_contributions"], combined=False
    )

    # Reverse the transformation if needed
    channel_contribution_original_scale = DataArray(
        np.reshape(
            target_transformer.inverse_transform(
                channel_contribution.data.flatten()[:, None]
            ),
            channel_contribution.shape,
        ),
        dims=channel_contribution.dims,
        coords=channel_contribution.coords,
    )

    numerator = channel_contribution_original_scale.sum(["date"])
    denominator = numerator.sum("channel")
    return numerator / denominator


def plot_channel_contribution_share_pie(model, channel_colors):
    # Extract channel contributions share samples
    channel_contributions_share: DataArray = model._get_channel_contributions_share_samples()
    
    # Convert the channels DataArray to a list of strings
    channels = channel_contributions_share.channel.values.tolist()
    values = channel_contributions_share.mean(dim=["chain", "draw"]).values

    # Generate darker borders and lighter fill for each slice
    pie_colors = [channel_colors.get(channel, 'rgb(0,0,0)') for channel in channels]
    lighter_pie_colors = [lighten_color(color, 0.5) for color in pie_colors]  # Lighter fill colors
    darker_border_colors = pie_colors  # Use original (darker) colors for borders

    # Build pie chart
    fig = go.Figure(go.Pie(
        labels=channels,
        values=values,
        marker=dict(
            colors=lighter_pie_colors,  # Lighter colors for fill
            line=dict(color=darker_border_colors, width=2)  # Darker borders
        ),
        hoverinfo="label+percent",
        textinfo="percent",
        hole=0.4
    ))

    # Update layout with styling options
    fig.update_layout(
        title_text="Channel Contribution Share",
        showlegend=False,
        template="plotly_white",  # Use the white plotly template
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the chart
        paper_bgcolor ='white',  # Transparent background for the paper
        margin=dict(l=40, r=40, t=40, b=40),  # Adjust margins for the chart
    )

    return fig




def plot_all_metrics_plotly(input_data):
    time_axis = np.arange(input_data.media_data.shape[0])
    charts = []

    for media_idx in range(input_data.media_data.shape[1]):
        values = input_data.media_data[:, media_idx]
        chart_name = f"{input_data.media_names[media_idx]} (volume)"
        charts.append((chart_name, values))

        values = input_data.media_costs_by_row[:, media_idx]
        chart_name = f"{input_data.media_names[media_idx]} (cost)"
        charts.append((chart_name, values))

    for extra_features_idx in range(input_data.extra_features_data.shape[1]):
        values = input_data.extra_features_data[:, extra_features_idx]
        chart_name = input_data.extra_features_names[extra_features_idx]
        charts.append((chart_name, values))

    charts.append((input_data.target_name, input_data.target_data))

    fig = go.Figure()

    for chart_name, values in charts:
        fig.add_trace(go.Scatter(x=time_axis, y=values, mode='lines', name=chart_name))
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text=chart_name)
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=time_axis, ticktext=input_data.date_strs))

    fig.update_layout(height=800, title="All Metrics Over Time",plot_bgcolor = 'rgba(0, 0, 0, 0)',paper_bgcolor ='white',)
    return fig


def plot_channel_contributions_plotly(model, config, channel_colors):
    date_col = config.get('date_col', 'date')

    df_attribution = (
        model.compute_channel_contribution_original_scale()
        .mean(dim=('chain', 'draw'))
        .to_dataframe(name='attribution')
        .reset_index()
        .pivot(index=date_col, columns='channel', values='attribution')
        .reset_index()
        .rename_axis(None, axis=1)
    )

    column_order = df_attribution.drop(date_col, axis=1).sum(axis=0).sort_values(ascending=False).index

    fig = go.Figure()

    for channel in column_order:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df_attribution[date_col]), 
            y=df_attribution[channel],
            mode='lines', 
            name=channel, 
            fill='tonexty', 
            stackgroup='one',
            line=dict(color=channel_colors.get(channel, 'rgb(0,0,0)'))
        ))

    fig.update_layout(
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor ='white',
        title="Channel Contributions Over Time",
        xaxis_title="Date", 
        yaxis_title="Revenue Contribution",
        legend_title="Channel"
    )

    return fig


def plot_roi_cpa_plotly(model, data, config, channel_colors, target_type='revenue'):
    channel_columns = [entry['spend_col'] for entry in config.get('media', [])]
    channel_contribution_original_scale = model.compute_channel_contribution_original_scale()

    # Check if target_type is 'conversion', if so, compute CPA
    if target_type == 'conversion':
        # Calculate CPA (Cost Per Acquisition) for each channel
        conversions = data[config.get('target_col', 'subscribers')].sum()  # Total conversions for the target column
        cpa_samples = (
            data[channel_columns].sum().to_numpy() # Total spend per channel
            / conversions  # Divide by total conversions
        )

        # Mean CPA values
        mean_cpa_values = {channel: cpa_samples[i] for i, channel in enumerate(channel_columns)}

        y_axis_title = "CPA (Cost Per Acquisition)"
        title = "CPA For Each Channel"
    else:
        # Calculate ROAS (Return on Ad Spend) for each channel
        roas_samples = (
            channel_contribution_original_scale.stack(sample=("chain", "draw")).sum("date")
            / data[channel_columns].sum().to_numpy()[..., None]
        )

        # Mean ROAS values
        mean_roas_values = {channel: roas_samples.sel(channel=channel).mean(dim="sample").item() for channel in channel_columns}

        y_axis_title = "ROI"
        title = "ROI For Each Channel"
    
    # Extract values based on the target_type
    if target_type == 'conversion':
        channels = list(mean_cpa_values.keys())
        values = list(mean_cpa_values.values())
    else:
        channels = list(mean_roas_values.keys())
        values = list(mean_roas_values.values())

    # Generate lighter fill colors and darker borders for the bars
    bar_colors = [channel_colors.get(channel, 'rgb(0,0,0)') for channel in channels]
    lighter_bar_colors = [lighten_color(color, 0.5) for color in bar_colors]  # Lighter fill colors
    darker_border_colors = bar_colors  # Darker borders

    # Create the bar chart
    fig = go.Figure()

    for i, channel in enumerate(channels):
        fig.add_trace(go.Bar(
            x=[channel],
            y=[values[i]],
            marker=dict(
                color=lighter_bar_colors[i],  # Lighter fill colors
                line=dict(color=darker_border_colors[i], width=2)  # Darker borders
            ),
            name=channel
        ))

    # Update layout with styling
    fig.update_layout(
        title=title,
        xaxis_title="Channel",
        yaxis_title=y_axis_title,
        template="plotly_white",  # Use the white plotly template
        showlegend=False,
        plot_bgcolor='rgba(0, 0, 0, 0)', 
        paper_bgcolor ='white',  
        margin=dict(l=40, r=40, t=40, b=80)  
    )

    return fig



def plot_roi_distribution_plotly(model, data, config, channel_colors):
    channel_columns = [entry['spend_col'] for entry in config.get('media', [])]
    channel_contribution_original_scale = model.compute_channel_contribution_original_scale()

    roas_samples = (
        channel_contribution_original_scale.stack(sample=("chain", "draw")).sum("date")
        / data[channel_columns].sum().to_numpy()[..., None]
    )

    fig = go.Figure()

    for channel in channel_columns:
        roas_distribution = roas_samples.sel(channel=channel)
        fig.add_trace(go.Violin(
            y=roas_distribution, 
            name=channel, 
            box_visible=True, 
            meanline_visible=True,
            line_color=channel_colors.get(channel, 'rgb(0,0,0)')
        ))

    fig.update_layout(title="ROI Distribution Across Channels",
                      yaxis_title="ROI", xaxis_title="Channel",plot_bgcolor = 'rgba(0, 0, 0, 0)',paper_bgcolor ='white',)
    return fig


def plot_decomposition_plotly(model, data, target_column, extra_features_cols, media_columns, date_column, promos, prophet_columns, channel_colors):
    all_columns = list(set(['intercept'] + extra_features_cols + prophet_columns + promos + media_columns))

    contributions = model.compute_mean_contributions_over_time(original_scale=True)
    contributions = contributions[all_columns]
    total_contributions = contributions.sum()
    sorted_contributions = total_contributions.sort_values(ascending=True)

    component_names = sorted_contributions.index
    component_values = sorted_contributions.values
    measure = ["relative"] * len(component_names)

    fig = go.Figure(go.Waterfall(
        name="Contributions",
        orientation="h",
        measure=measure,
        y=component_names,
        x=component_values,
        textposition="outside",
        text=[f'{value:.0f}' for value in component_values],
        connector={"visible": False},
        increasing={"marker": {"color": "rgba(0, 0, 255, 0.5)"}},
        decreasing={"marker": {"color": "#d9534f"}},
        totals={"marker": {"color": "#5cb85c"}}
    ))

    fig.update_layout(
        title="Decomposition Of Contributions",
        yaxis_title="metrics",
        xaxis_title="contribution",
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='white',
        xaxis=dict(
            title="contribution",
            zeroline=False,
            showgrid=False,
        ),
        yaxis=dict(
            title="metrics",
            showgrid=False,
            tickfont=dict(size=10),
            automargin=True,
        ),
        font=dict(
            family="Arial",
            size=10,
            color="black",
        ),
        waterfallgap=0.1,
        autosize=True,
        margin=dict(l=80, r=30, t=80, b=50),
    )

    # Ensure that text for both positive and negative values is placed outside the bars on the right
    fig.update_traces(
        cliponaxis=False,  # Avoid text clipping
        textposition="outside",  # Position text outside the bars
        texttemplate='%{text}',  # Ensure text formatting
    )

    return fig


def all_contributions_plot_plotly(model, config, channel_colors):
    # Compute the contributions and reset the index
    contributions = model.compute_mean_contributions_over_time(original_scale=True).reset_index()
    date_col = config.get('date_col', 'date')

    # Define base columns including intercept and extra features
    base_cols = ['intercept'] + config.get('extra_features_cols', [])
    if not all(col in contributions.columns for col in base_cols):
        raise ValueError("Not all base columns found in the contributions data.")

    # Sum the base contributions
    contributions['base'] = contributions[base_cols].sum(axis=1)

    # Compute channel contributions and pivot the dataframe
    df_attribution = (
        model.compute_channel_contribution_original_scale()
        .mean(dim=('chain', 'draw'))
        .to_dataframe(name='attribution')
        .reset_index()
        .pivot(index=date_col, columns='channel', values='attribution')
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # Merge with base contributions and fill missing values with 0
    df_attribution = pd.merge(df_attribution, contributions[[date_col, 'base']], on=date_col, how='left').fillna(0)

    # Clip all values to be above 0 to ensure no negative contributions are plotted
    df_attribution[df_attribution.columns.difference([date_col])] = df_attribution[df_attribution.columns.difference([date_col])].clip(lower=0)

    # Order columns by the sum of their values
    column_order = df_attribution.drop(date_col, axis=1).sum(axis=0).sort_values(ascending=False).index

    fig = go.Figure()

    # Ensure "Base" is colored grey
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df_attribution[date_col]), 
        y=df_attribution['base'],
        mode='lines', 
        name='Base', 
        fill='tonexty', 
        stackgroup='one',
        line=dict(color='grey', width=0),  # Ensure 'Base' color is set to grey
        fillcolor='grey'
    ))

    # Add each media channel as a trace, using the specified color
    for channel in column_order:
        if channel != 'base':  # Avoid re-adding base as it's already handled
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df_attribution[date_col]), 
                y=df_attribution[channel],
                mode='lines', 
                name=channel, 
                fill='tonexty', 
                stackgroup='one',
                line=dict(color=channel_colors.get(channel, 'rgb(0,0,0)'))
            ))

    # Update the layout of the figure
    fig.update_layout(
        title="Total Contributions Including Base Over Time",
        xaxis_title="Date", 
        yaxis_title="Revenue Contribution",
        legend_title="Channel",
        template="plotly_white",
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor ='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.31,  # Match legend position with other plots
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, b=80)  # Consistent margin setup
    )

    return fig
def plot_posterior_predictive_with_r2(
    model, X_train, y_train, original_scale: bool = False, **plt_kwargs: Any
) -> go.Figure:
    # Sample posterior predictive data
    y_train_pred = model.sample_posterior_predictive(X_train, extend_idata=True, combined=True)

    y_train_pred_mean = y_train_pred['y'].mean(dim='sample').values
    
    # Calculate R^2 score
    r_squared_training = r2_score(y_train, y_train_pred_mean)
    posterior_predictive_data = model.posterior_predictive
    # Calculate HDIs
    likelihood_hdi_95 = az.hdi(ary=posterior_predictive_data, hdi_prob=0.95, group="posterior")[model.output_var]
    likelihood_hdi_50 = az.hdi(ary=posterior_predictive_data, hdi_prob=0.50, group="posterior")[model.output_var]

    if original_scale:
        likelihood_hdi_95 = model.get_target_transformer().inverse_transform(Xt=likelihood_hdi_95)
        likelihood_hdi_50 = model.get_target_transformer().inverse_transform(Xt=likelihood_hdi_50)

    target_to_plot = np.asarray(
        model.y if original_scale else model.preprocessed_data["y"]  # type: ignore
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=model.X[model.date_column],
        y=likelihood_hdi_95[:, 0],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=model.X[model.date_column],
        y=likelihood_hdi_95[:, 1],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(width=0),
        name="95% HDI",
    ))

    fig.add_trace(go.Scatter(
        x=model.X[model.date_column],
        y=likelihood_hdi_50[:, 0],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=model.X[model.date_column],
        y=likelihood_hdi_50[:, 1],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.3)',
        line=dict(width=0),
        name="50% HDI",
    ))

    fig.add_trace(go.Scatter(
        x=model.X[model.date_column],
        y=target_to_plot,
        mode='lines',
        line=dict(color='black'),
        name='Observed'
    ))

    # Add R^2 annotation to the plot
    fig.add_annotation(
        text=f'R^2: {r_squared_training:.3f}',
        xref='paper', yref='paper',
        x=0.95, y=0.05,
        showarrow=False,
        font=dict(size=12, color='black'),
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor='black'
    )

    fig.update_layout(
        title="Posterior Predictive Check",
        xaxis_title="date",
        yaxis_title=model.output_var,
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor ='white',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,  # Adjust this value to control how far below the plot the legend appears
            xanchor="center",
            x=0.5
        ),
        **plt_kwargs
    )

    return fig


def plot_new_spend_contributions_plotly(model, spend_amount, channel_colors, one_time=True, lower=0.025, upper=0.975, 
                                        ylabel="Sales", channels=None, prior=False, original_scale=True):
    total_channels = len(model.channel_columns)
    contributions = model.new_spend_contributions(
        np.ones(total_channels) * spend_amount, one_time=one_time, spend_leading_up=np.ones(total_channels) * spend_amount,
        prior=prior, original_scale=original_scale,
    )

    contributions_groupby = contributions.to_series().groupby(level=["time_since_spend", "channel"])

    idx = pd.IndexSlice[0:]

    conf = contributions_groupby.quantile([lower, upper]).unstack("channel").unstack().loc[idx]

    channels = channels or model.channel_columns

    fig = go.Figure()

    for channel in channels:
        # Extract RGB values from the color string (e.g., '#4E79A7')
        color = channel_colors[channel]
        if color.startswith('#'):
            r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        else:
            r, g, b = (0, 0, 0)  # Default to black if color format is unexpected
        
        rgba_fillcolor = f'rgba({r},{g},{b},0.2)'  # Add 20% opacity for the fill

        # Plot the upper confidence interval
        fig.add_trace(go.Scatter(
            x=conf.index, y=conf[channel][upper], mode='lines', name=f"{channel} Upper CI", line=dict(width=0), showlegend=False
        ))
        
        # Plot the lower confidence interval with fill
        fig.add_trace(go.Scatter(
            x=conf.index, y=conf[channel][lower], mode='lines', name=f"{channel} Lower CI", line=dict(width=0),
            fill='tonexty', 
            fillcolor=rgba_fillcolor,  # Use the constructed RGBA color
            showlegend=False
        ))

    mean = contributions_groupby.mean().unstack("channel").loc[idx, channels]

    for i, channel in enumerate(channels):
        # Plot mean contributions
        fig.add_trace(go.Scatter(
            x=mean.index, y=mean[channel], mode='lines', name=f"{channel} Mean", 
            line=dict(color=channel_colors.get(channel, f"rgb({i*40 % 256}, {i*80 % 256}, {i*120 % 256})"))
        ))

    fig.update_layout(
        title=f"Upcoming Sales for {spend_amount:.02f} Spend", 
        xaxis_title="Time since spend", 
        yaxis_title=ylabel, 
        template='plotly_white', 
        plot_bgcolor='rgba(0, 0, 0, 0)', 
        paper_bgcolor ='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
    )

    return fig



def out_of_sample_all_contributions_plot(contributions_pivot, date_column='date', show_only_media=False, 
                                         extra_features_cols=None, prophet_columns=None, promos=None, channel_colors=None):
    base_cols = ['intercept'] + (extra_features_cols or []) + (prophet_columns or []) + (promos or [])
    if show_only_media:
        contributions_pivot = contributions_pivot.drop(columns=base_cols, errors='ignore')
    else:
        missing_cols = [col for col in base_cols if col not in contributions_pivot.columns]
        if missing_cols:
            st.warning(f"Missing base columns: {missing_cols}")

        available_base_cols = [col for col in base_cols if col in contributions_pivot.columns]
        contributions_pivot['Base'] = contributions_pivot[available_base_cols].sum(axis=1)
        contributions_pivot = contributions_pivot.drop(columns=available_base_cols, errors='ignore')

    column_order = contributions_pivot.drop('Base', axis=1, errors='ignore').sum(axis=0).sort_values(ascending=False).index

    fig = go.Figure()

    if not show_only_media:
        fig.add_trace(go.Scatter(
            x=contributions_pivot.index, y=contributions_pivot['Base'], mode='lines', name='Base', 
            stackgroup='one', line=dict(color='grey', width=0), fill='tonexty'
        ))

    for channel in column_order:
        fig.add_trace(go.Scatter(
            x=contributions_pivot.index, y=contributions_pivot[channel], mode='lines', name=channel, 
            stackgroup='one', line=dict(color=channel_colors.get(channel, 'rgb(0,0,0)'))
        ))

    fig.update_layout(
        title='Total Contributions Including Base Over Time' if not show_only_media else 'Channel Contributions Over Time',
        xaxis_title="Date", yaxis_title="Revenue Contribution", template="plotly_white",plot_bgcolor = 'rgba(0, 0, 0, 0)',paper_bgcolor ='white', 
        legend=dict(orientation="h", yanchor="bottom", y=-0.31, xanchor="center", x=0.5)
    )

    return fig


def plot_historical_forecasted_spend(historical_data, forecasted_data, date_column, media_columns, channel_colors, last_date):
    fig_combined = go.Figure()

    # Adding historical data
    for column in media_columns:
        fig_combined.add_trace(go.Scatter(
            x=historical_data[date_column], 
            y=historical_data[column], 
            mode='lines',  # Changed to 'lines' for consistency
            name=column, 
            stackgroup='one', 
            line=dict(color=channel_colors[column]), 
            fill='tonexty'
        ))

    # Adding forecasted data
    for column in media_columns:
        fig_combined.add_trace(go.Scatter(
            x=forecasted_data[date_column], 
            y=forecasted_data[column], 
            mode='lines',  # Changed to 'lines' for consistency
            name=column, 
            stackgroup='one', 
            line=dict(color=channel_colors[column]), 
            fill='tonexty',
            showlegend=False
        ))

    # Adding vertical line for the last date
    fig_combined.add_vline(x=last_date, line=dict(color='red', width=2, dash='dash'))

    # Update layout to match style
    fig_combined.update_layout(
        title='Historical and Forecasted Marketing Spend', 
        xaxis_title='Date', 
        yaxis_title='Spend', 
        template='plotly_white', 
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor ='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.31, xanchor="center", x=0.5),  # Adjusted to match other graph
        margin=dict(l=40, r=40, t=40, b=80)
    )

    return fig_combined


def plot_control_contributions_plotly(contributions_df: pd.DataFrame, title: str, control_colors_dict):
    contributions_df['date'] = pd.to_datetime(contributions_df['date'])
    contributions_pivot = contributions_df.pivot_table(index='date', columns='channel', values='contribution', aggfunc='mean')
    fig = go.Figure()

    for control in contributions_pivot.columns:
        fig.add_trace(go.Scatter(
            x=contributions_pivot.index, y=contributions_pivot[control], mode='lines', name=control, 
            stackgroup='one', line=dict(color=control_colors_dict.get(control, 'rgb(0,0,0)'))
        ))

    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Contribution", template="plotly_white",plot_bgcolor = 'rgba(0, 0, 0, 0)',paper_bgcolor ='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
    )
    return fig


def plot_channel_contributions_grid(model, start: float, stop: float, num: int, channel_colors: dict, absolute_xrange: bool = False) -> go.Figure:
    """
    Plot a grid of scaled channel contributions for a given grid of share values using Plotly.

    Parameters
    ----------
    model : DelayedSaturatedMMM
        The model object that contains the necessary methods and data.
    start : float
        Start of the grid. It must be equal or greater than 0.
    stop : float
        End of the grid. It must be greater than start.
    num : int
        Number of points in the grid.
    channel_colors : dict
        Dictionary mapping channel names to their corresponding colors.
    absolute_xrange : bool, optional
        If True, the x-axis is in absolute values (input units), otherwise it is in
        relative percentage values, by default False.

    Returns
    -------
    go.Figure
        Plotly figure of grid of channel contributions.
    """
    share_grid = np.linspace(start=start, stop=stop, num=num)
    contributions = model.get_channel_contributions_forward_pass_grid(start=start, stop=stop, num=num)

    fig = go.Figure()

    for channel in model.channel_columns:
        channel_contribution_total = contributions.sel(channel=channel).sum(dim="date")
        mean_contribution = channel_contribution_total.mean(dim=("chain", "draw"))
        hdi_contribution = az.hdi(ary=channel_contribution_total).x

        total_channel_input = model.X[channel].sum()
        x_range = total_channel_input * share_grid if absolute_xrange else share_grid

        # Use the provided channel color for the line and fill
        line_color = channel_colors.get(channel, 'rgb(0,0,0)')
        fill_color = f"rgba{line_color[3:-1]},0.1)"  # Adjusting the color to 10% opacity

        # Plot the HDI as a shaded area
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_range, x_range[::-1]]),
            y=np.concatenate([hdi_contribution[:, 0], hdi_contribution[::-1, 1]]),
            fill='toself',
            fillcolor=fill_color,
            line=dict(color='rgba(255,255,255,0)'),  # No line for HDI
            showlegend=False,
            hoverinfo='skip'
        ))

        # Plot the mean contribution as a line plot
        fig.add_trace(go.Scatter(
            x=x_range,
            y=mean_contribution,
            mode='lines+markers',
            name=f"{channel} contribution mean",
            line=dict(color=line_color, width=2),
            marker=dict(color=line_color)
        ))

        # If absolute_xrange, add a vertical line to indicate the current total input
        if absolute_xrange:
            fig.add_vline(
                x=total_channel_input,
                line=dict(color=line_color, dash='dash'),
                annotation_text=f"{channel} current total input",
                annotation_position="top left"
            )

    # Add a reference line at x=1 if not using absolute x-range
    if not absolute_xrange:
        fig.add_vline(x=1, line=dict(color="black", dash='dash'), annotation_text="delta = 1")

    fig.update_layout(
        title="Channel contribution as a function of cost share",
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor ='white',
        xaxis_title="Input" if absolute_xrange else "Delta",
        yaxis_title="Contribution",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        template="plotly_white"
    )

    return fig

def _plot_response_curve_fit(
        model,
        channel: str,
        color_index: int,
        xlim_max: int | None,
        label: str = "Fit Curve",
        quantile_lower: float = 0.05,
        quantile_upper: float = 0.95,
        channel_colors: dict[str, str] = None,
        show_ci: bool = False, 
    ) -> list:
        """Plot the curve fit for the given channel based on the estimation of the parameters by the model using Plotly.

        Parameters
        ----------
        channel : str
            The name of the channel for which the curve fit is being plotted.
        color_index : int
            An index used for color selection to ensure distinct colors for multiple plots.
        xlim_max: int
            The maximum value to be plot on the X-axis.
        label: str
            The label for the curve being plotted, default is "Fit Curve".
        quantile_lower: float
            The lower quantile for parameter estimation, default is 0.05.
        quantile_upper: float
            The upper quantile for parameter estimation, default is 0.95.
        channel_colors : dict[str, str], optional
            A dictionary mapping channel names to their corresponding colors.
        show_ci : bool, optional
            Whether to display the 5% to 95% confidence interval. Default is False.

        Returns
        -------
        list
            A list of Plotly trace objects for the fit curve and confidence interval.
        """

        traces = []

        if model.X is not None:
            x_mean = np.max(model.X[channel])

        # Set x_limit based on the method or xlim_max
        x_limit = xlim_max if xlim_max is not None else x_mean

        # Generate x_fit and y_fit
        x_fit = np.linspace(0, x_limit, 1000)
        upper_params = model.format_recovered_transformation_parameters(quantile=quantile_upper)
        lower_params = model.format_recovered_transformation_parameters(quantile=quantile_lower)
        mid_params = model.format_recovered_transformation_parameters(quantile=0.5)

        y_fit = model.saturation.function(
            x=x_fit, **mid_params[channel]["saturation_params"]
        ).eval()

        y_fit_lower = model.saturation.function(
            x=x_fit, **lower_params[channel]["saturation_params"]
        ).eval()
        y_fit_upper = model.saturation.function(
            x=x_fit, **upper_params[channel]["saturation_params"]
        ).eval()

        # scale all y fit values to the original scale using
        y_fit = (
            model.get_target_transformer()
            .inverse_transform(y_fit.reshape(-1, 1))
            .flatten()
        )
        y_fit_lower = (
            model.get_target_transformer()
            .inverse_transform(y_fit_lower.reshape(-1, 1))
            .flatten()
        )
        y_fit_upper = (
            model.get_target_transformer()
            .inverse_transform(y_fit_upper.reshape(-1, 1))
            .flatten()
        )

        # scale x fit values
        x_fit = model._channel_map_scales()[channel] * x_fit

        line_color = channel_colors.get(channel, f"rgb({color_index * 40 % 256}, {color_index * 80 % 256}, {color_index * 120 % 256})")

        # Create trace for the confidence interval
        if show_ci:
            traces.append(go.Scatter(
                x=np.concatenate([x_fit, x_fit[::-1]]),
                y=np.concatenate([y_fit_upper, y_fit_lower[::-1]]),
                fill='toself',
                fillcolor=f"rgba{line_color[3:-1]},0.2)",
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{label} 95% CI",
                showlegend=True
            ))

        # Create trace for the fitted curve
        traces.append(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            line=dict(color=line_color, width=2),
            name=label
        ))

        return traces


def plot_direct_contribution_curves(
        model,
        show_fit: bool = False,
        xlim_max: int | None = None,
        channels: list[str] | None = None,
        quantile_lower: float = 0.05,
        quantile_upper: float = 0.95,
        channel_colors: dict[str, str] = None,
        show_ci: bool = False,  # New parameter for showing the 5% to 95% CI
    ) -> go.Figure:
        """Plot the direct contribution curves for each marketing channel using Plotly."""

        channels_to_plot = model.channel_columns if channels is None else channels

        if not all(channel in model.channel_columns for channel in channels_to_plot):
            unknown_channels = set(channels_to_plot) - set(model.channel_columns)
            raise ValueError(
                f"The provided channels must be a subset of the available channels. Got {unknown_channels}"
            )

        if len(channels_to_plot) != len(set(channels_to_plot)):
            raise ValueError("The provided channels must be unique.")

        fig = go.Figure()

        for i, channel in enumerate(channels_to_plot):
            if model.X is not None:
                # Filter the data for the specific channel
                x = model.X[channel].to_numpy()
                y = model.compute_channel_contribution_original_scale().sel(channel=channel).mean(["chain", "draw"]).to_numpy()

                # Scatter plot of data points
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(color=channel_colors.get(channel, f"rgb({i*40 % 256}, {i*80 % 256}, {i*120 % 256})")),
                    name=channel,  # Name appears in legend
                    legendgroup=channel,  # Group traces by channel
                    showlegend=True
                ))

                # Optionally plot the fit curve using the updated method
                if show_fit:
                    fit_traces = _plot_response_curve_fit(
                        model=model,
                        channel=channel,
                        color_index=i,
                        xlim_max=xlim_max,
                        label=f"{channel} Fit Curve",
                        quantile_lower=quantile_lower,
                        quantile_upper=quantile_upper,
                        channel_colors=channel_colors,
                        show_ci=show_ci  # Pass the new parameter
                    )
                    for trace in fit_traces:
                        trace.update(legendgroup=channel, showlegend=False)  # Group with the same channel, hide extra legends
                        fig.add_trace(trace)

        # Adjust layout to place legend below the plot
        fig.update_layout(
            title="Direct Response Curves",
            xaxis_title="Spent",
            yaxis_title="Contribution",
            template="plotly_white",
            plot_bgcolor = 'rgba(0, 0, 0, 0)',
            paper_bgcolor ='white',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,  # Place the legend below the plot
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=40, r=40, t=40, b=80)  # Adjust margins if needed
        )

        return fig

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

def display_and_save_vfi_table(data, drop_columns=None, date_column=None, target_column=None, high_vif_threshold=10):
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
    st.session_state['vfi_table'] = vif_data
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
    st.subheader("VFI Table")
    st.dataframe(styled_vif_table, height=500,hide_index = True)
    
def highlight_high_vif_values(vif_value: float, high_vif_threshold: float=7.0) -> str:
    if vif_value > high_vif_threshold:
        weight = 'bold'
        color = '#d1615d'  # Red color for high VFI values
    else:
        weight = 'normal'
        color = '#6a9f58'  # Green color for normal VFI values
    style = f'font-weight: {weight}; color: {color}'
    return style