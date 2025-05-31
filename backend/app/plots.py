from model import load_data, featurize_data
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

output_dir_targets = "../plots/targets"
output_dir_macro = "../plots/macro"
output_dir_sentiment = "../plots/sentiment"
output_dir_rolling = "../plots/rolling"
output_dir_technical = "../plots/technical"
output_dir_extremes = "../plots/extremes"

def plot_targets_interactive(ticker):
    raw_df = load_data(ticker)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])

    X_train, X_test, y_train, y_test = featurize_data(
        raw_df,
        use_baseline_data=True,
        use_macro_data=True,
        use_extra_data=True,
        target_window=1
    )

    train_dates = pd.to_datetime(X_train["Date"])
    test_dates = pd.to_datetime(X_test["Date"])
    vertical_line_date = pd.to_datetime("2013-01-01")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=train_dates,
        y=raw_df[raw_df["Date"] < vertical_line_date]["Close"],
        mode="lines",
        name="Training Close Price"
    ))
    fig1.add_trace(go.Scatter(
        x=test_dates,
        y=raw_df[raw_df["Date"] > vertical_line_date]["Close"],
        mode="lines",
        name="Testing Close Price"
    ))
    fig1.add_vline(x=vertical_line_date, line_dash="dash")
    fig1.update_layout(
        title=f"{ticker} Close Price (Train/Test)",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        width=800, height=500,
        margin=dict(l=20, r=20, t=50, b=50)
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=train_dates,
        y=y_train["ClosePriceReturn"],
        mode="lines",
        name="Training ClosePriceReturn"
    ))
    fig2.add_trace(go.Scatter(
        x=test_dates,
        y=y_test["ClosePriceReturn"],
        mode="lines",
        name="Testing ClosePriceReturn"
    ))
    fig2.add_vline(x=vertical_line_date, line_dash="dash")
    fig2.update_layout(
        title=f"{ticker} ClosePriceReturn (Train/Test)",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        width=800, height=500,
        margin=dict(l=20, r=20, t=50, b=50)
    )

    all_returns = pd.concat([y_train["ClosePriceReturn"], y_test["ClosePriceReturn"]])
    fig3 = px.histogram(
        all_returns,
        nbins=50,
        histnorm="density",
        title="Distribution of Close Price Returns",
        labels={"value": "Close Price Return", "density": "Density"},
    )
    fig3.update_layout(
        width=800, height=500,
        margin=dict(l=20, r=20, t=50, b=50),
        template="plotly_white"
    )

    return fig1, fig2, fig3


def plot_macro_features_interactive():
    macro_df = pd.read_csv("../data/processed/macrodata/macrodata_processed.csv")
    macro_df["Date"] = pd.to_datetime(macro_df["Date"])

    feature_descriptions = {
        "DCOILWTICO(daily)": "Crude Oil Prices",
        "CPIAUCSL(Interpolated from monthly to daily)": "Consumer Price Index for All Urban Consumers",
        "FEDFUNDS(Interpolated from monthly to daily)": "Effective Federal Funds Rate",
        "GDP(Interpolated from quarterly to daily)": "Gross Domestic Product",
        "PSAVERT(Interpolated from monthly to daily)": "Personal Saving Rate",
        "UMCSENT(Interpolated from monthly to daily)": "Consumer Sentiment Index",
        "UNRATE(Interpolated from monthly to daily)": "Unemployment Rate",
    }

    col_mapping = {
        "interpolated_DCOILWTICO": "DCOILWTICO(daily)",
        "interpolated_CPIAUCSL": "CPIAUCSL(Interpolated from monthly to daily)",
        "interpolated_FEDFUNDS": "FEDFUNDS(Interpolated from monthly to daily)",
        "interpolated_GDP": "GDP(Interpolated from quarterly to daily)",
        "interpolated_PSAVERT": "PSAVERT(Interpolated from monthly to daily)",
        "interpolated_UMCSENT": "UMCSENT(Interpolated from monthly to daily)",
        "interpolated_UNRATE": "UNRATE(Interpolated from monthly to daily)",
    }

    figs = []

    for col, title_key in col_mapping.items():
        if col in macro_df.columns:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=macro_df["Date"],
                y=macro_df[col],
                mode="lines",
                name="Value",
                line=dict(color="blue", width=2)
            ))

            description = feature_descriptions.get(title_key, "No description available")

            fig.update_layout(
                title={
                    'text': f"{title_key}<br><sup>{description}</sup>",
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Date",
                yaxis_title="Value",
                width=800,
                height=500,
                margin=dict(l=20, r=20, t=80, b=40),
                legend_title="Feature",
                template="plotly_white"
            )

            figs.append(fig)

    return figs

def plot_moving_averages_interactive(ticker):
    raw_df = load_data(ticker)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    raw_df_last_year = raw_df[raw_df["Date"] >= "2016-10-01"]
    dates_last_year = raw_df_last_year["Date"]

    periods = [5, 10, 20, 50, 100, 200]
    periods = [10, 20, 50]


    figures = []

    for i, period in enumerate(periods):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates_last_year,
            y=raw_df_last_year["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="blue", width=2, dash="solid")
        ))

        fig.add_trace(go.Scatter(
            x=dates_last_year,
            y=raw_df_last_year[f"ClosePriceSMA{period}"],
            mode="lines",
            name=f"SMA {period}",
            line=dict(color="red", dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=dates_last_year,
            y=raw_df_last_year[f"ClosePriceEMA{period}"],
            mode="lines",
            name=f"EMA {period}",
            line=dict(color="green", dash="dot")
        ))

        fig.update_layout(
            title=f"{ticker} Close Price with SMA and EMA {period}",
            xaxis_title="Date",
            yaxis_title="Close Price",
            legend_title="Legend",
            width=700,
            height=500,
            margin=dict(l=20, r=20, t=50, b=50),
            template="plotly_white"
        )

        figures.append(fig)

    return figures

def plot_bollinger_bands_interactive(ticker):
    raw_df = load_data(ticker)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    raw_df_last_year = raw_df[raw_df["Date"] >= "2016-10-01"]
    dates_last_year = raw_df_last_year["Date"]

    periods = [5, 10, 20, 50, 100, 200]

    figures = []

    for i, period in enumerate(periods):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates_last_year,
            y=raw_df_last_year["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="blue", width=2, dash="solid")
        ))

        fig.add_trace(go.Scatter(
            x=dates_last_year,
            y=raw_df_last_year[f"Bollinger_Mean_{period}"],
            mode="lines",
            name=f"Bollinger Mean {period}",
            line=dict(color="red", width=2, dash="solid")
        ))

        fig.add_trace(go.Scatter(
            x=dates_last_year,
            y=raw_df_last_year[f"Bollinger_Upper_{period}"],
            mode="lines",
            name=f"Bollinger Upper {period}",
            line=dict(color="red", width=1, dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=dates_last_year,
            y=raw_df_last_year[f"Bollinger_Lower_{period}"],
            mode="lines",
            name=f"Bollinger Lower {period}",
            line=dict(color="red", width=1, dash="dot")
        ))

        fig.update_layout(
            title=f"{ticker} Close Price with Bollinger Bands ({period} Days)",
            xaxis_title="Date",
            yaxis_title="Close Price",
            legend_title="Legend",
            width=800,
            height=500,
            margin=dict(l=20, r=20, t=50, b=50),
            template="plotly_white"
        )

        figures.append(fig)

    return figures