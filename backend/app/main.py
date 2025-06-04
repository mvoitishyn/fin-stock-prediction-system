import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import time
from linearmodels import predict_linear_model
from neuralnetworks import predict_neural_network
from bayesianneuralnetworks import predict_bayesian_model
from process import process_dataset
from trainmodel import run_linear_single_model, run_single_neural_network, run_bayesian_single_model
from plots import plot_moving_averages_interactive, plot_macro_features_interactive, plot_bollinger_bands_interactive, plot_targets_interactive

def main():
    st.set_page_config(page_title="Thesis Financial Forecasting", layout="wide")
    st.title("Developing a Hybrid AI Model for Financial Time Series Forecasting")

    st.sidebar.title("Navigation")
    pages = [
        "Introduction",
        "Data Selection",
        "Feature engineering",
        "Prediction Target",
        "Moving averages",
        "Macro Features",
        "Bollinger Bands",
        "Train Linear Model",
        "Predict Linear Model",
        "Train Neural Network",
        "Predict Neural Network",
        "Train Bayesian Model",
        # "Predict Bayesian Model" TODO: FIX!!!
    ]
    page = st.sidebar.radio("Go to", pages)

    if page == "Introduction":
        st.write("""

        The **Demo App** is a tool built using the Python **Streamlit** library. It provides an end-to-end workflow for financial time series modeling - from data loading to prediction.

        ##### Key Features:
        1. **Select & Load Stock Data** - Choose financial instruments and load historical data.
        2. **Preprocess & Feature Engineering** - Clean and transform data to prepare it for modeling.
        3. **Exploratory Data Analysis** - Explore the market data from various sources.
        3. **Train Models** - Apply and customize machine learning models for prediction tasks.
        4. **Predict & Evaluate** - Generate forecasts and evaluate model performance interactively.

        Get started by selecting a step from the sidebar!
        """)

    if page == "Data Selection":
        st.header("Data Selection")
        ticker = st.text_input("Enter Ticker Symbol (e.g., QCOM, AMD, INTC, NVDA, TSM)", value="INTC")

        if st.button("Load Data"):
            df = pd.read_csv(f"../data/raw/parsed/{ticker}.csv")
            dates = pd.to_datetime(df["Unnamed: 0"])
            df = df.drop(columns=[col for col in df.columns if "Unnamed" in col])
            df["Date"] = dates
            summary = {
                "Date Range Start": df["Date"].min().strftime("%Y-%m-%d"),
                "Date Range End": df["Date"].max().strftime("%Y-%m-%d"),
                "Average Close": round(df["Close"].mean(), 2),
                "Max Close": df["Close"].max(),
                "Min Close": df["Close"].min(),
                "Average Volume": int(df["Volume"].mean()),
                "Max Volume": int(df["Volume"].max()),
                "Min Volume": int(df["Volume"].min()),
            }
            st.success(f"Data loaded for {ticker}!")

            st.session_state["df"] = df
            st.session_state["ticker"] = ticker

            st.subheader("Summary Statistics")

            st.dataframe(df.tail(10))
            st.dataframe(pd.DataFrame([summary]))

    if page == "Feature engineering":
        try:
            df = st.session_state["df"]

            ticker = st.session_state["ticker"]

            start_time = time.time()
            processed_df = process_dataset(ticker, "2000-01-01", "2018-01-01")
            duration = time.time() - start_time


            processed_df.to_csv(f"../data/processed/stocksdata/{ticker}_processed.csv")

            st.session_state["processed_df"] = processed_df

            min_date = processed_df["Date"].min()
            max_date = processed_df["Date"].max()

            st.markdown(f"""
            ### Processed Dataset Overview
            - **Observations:** {processed_df.shape[0]}
            - **Features:** {processed_df.shape[1]}
            - **Date Range:** {min_date} → {max_date}
            - **Execution Time:** {duration:.2f} seconds

            ### Included Features

            - **Baseline OHLCV features**:
                - `Date`, `Open`, `High`, `Low`, `Volume`

            - **Macroeconomic data features**:
                - `DCOILWTICO(Daily)` - Oil price
                - `CPIAUCSL(Interpolated from monthly to daily)` - Inflation / CPI
                - `FEDFUNDS(Interpolated from monthly to daily)` - Federal Funds Rate
                - `GDP(Interpolated from quarterly to daily)` - Gross Domestic Product
                - `PSAVERT(Interpolated from monthly to daily)` - Personal Saving Rate
                - `UMCSENT(Interpolated from monthly to daily)` - Consumer Sentiment
                - `UNRATE(Interpolated from monthly to daily)` - Unemployment Rate

            - **Extra technical and engineered features**:
                - Technical indicators: `SMA`, `EMA`, `RSI`, `ROC`, `Bollinger Bands`
                - Volatility and range measures: `Std`, `IntradayVolatility`, `OpenCloseRange`
                - Extreme event features: `ExtremeClosePriceDailyReturn_IQR`, `ExtremeVolumeDailyChangeReturn_IQR`
                - Time-based features: `Year`, `Month`, `Day`, `DayNumeric`
                - Sentiment-related features: `AnomalySentimentScore`
            """)

            st.dataframe(processed_df.head())

        except:
            st.error("Please select ticker and load the data first.")

    if page == "Prediction Target":
        st.header("Training and Testing Targets Overview")
        try:
            ticker = st.session_state["ticker"]
            fig1, fig2, fig3 = plot_targets_interactive(ticker)

            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.plotly_chart(fig3, use_container_width=True) 

        except Exception:
            st.error("Please select ticker and load the data first.")

    if page == "Moving averages":
        try:
            st.header("Exploratory Data Analysis")
            ticker = st.session_state["ticker"]
            figs = plot_moving_averages_interactive(ticker)

            for i in range(0, len(figs), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(figs):
                        with cols[j]:
                            st.plotly_chart(figs[i + j], use_container_width=True)
        except:
            st.error("Please select ticker and load the data first.")

    if page == "Macro Features":
        st.header("Macro Features Overview")
        try:
            figs = plot_macro_features_interactive()

            for i in range(0, len(figs), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(figs):
                        with cols[j]:
                            st.plotly_chart(figs[i + j], use_container_width=True)
        except:
            st.error("Please, load the data first.")
    if page == "Bollinger Bands":
        st.header("Bollinger Bands Analysis")
        try:
            ticker = st.session_state["ticker"]
            figs = plot_bollinger_bands_interactive(ticker)

            for i in range(0, len(figs), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(figs):
                        with cols[j]:
                            st.plotly_chart(figs[i + j], use_container_width=True)
        except:
            st.error("Please select ticker and load the data first.")

    if page == "Train Linear Model":
        try:
            ticker = st.session_state["ticker"]
            st.subheader(f"Running model for {ticker}: please configure model parameters")

            col1, col2 = st.columns(2)
            with col1:
                scale_bool = st.checkbox("Scale Data")
                use_macro_data_bool = st.checkbox("Use Macro Data")
            with col2:
                use_nonlinear_transformation_bool = st.checkbox("Use Nonlinear Transformation")
                use_extra_data_bool = st.checkbox("Use Extra (Sentiment/Anomaly) Data")

            target_window = st.selectbox(
                "Target Window (days)",
                options=[1, 3, 5],
                index=0
            )

            run_button = st.button("Run Linear Model")

            if run_button:
                st.info("Running model... please wait")
                start_time = time.time()
                results = run_linear_single_model(
                    ticker=ticker,
                    scale_bool=scale_bool,
                    use_nonlinear_transformation_bool=use_nonlinear_transformation_bool,
                    use_macro_data_bool=use_macro_data_bool,
                    use_extra_data_bool=use_extra_data_bool,
                    target_window=target_window
                )
                duration = time.time() - start_time
                st.write(f" **Training time:** {duration:.2f} seconds")

                if results is None:
                    st.error("Model run failed. Check your settings or data.")
                else:
                    st.success("Model run completed!")
                    st.subheader("Linear Model Results")
                    st.dataframe(results)

                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="linear_single_model_result.csv",
                        mime='text/csv',
                    )
        except:
            st.error("Please select ticker and load the data first.")


    if page == "Predict Linear Model":

        try:
            ticker = st.session_state["ticker"]
            st.subheader(f"Predicting using Linear model for {ticker}: please configure prediction parameters")

            col1, col2 = st.columns(2)
            with col1:
                scale = st.checkbox("Scale Input Data")
                use_nonlinear_transformation_bool = st.checkbox("Use Nonlinear Transformation")
                use_macro_data_bool = st.checkbox("Use Macro Data")
                use_extra_data_bool = st.checkbox("Use Extra Data (Anomalies/Sentiment)")
            with col2:
                target_window = st.selectbox(
                    "Target Window (days)",
                    options=[1, 3, 5],
                    index=0
            )

            predict_button = st.button("Run Prediction")

            if predict_button:
                st.info("Running prediction... please wait")
                try:
                    rawdf = st.session_state["processed_df"]

                    use_baseline_data_bool = True
                    model_path = f"cachedir/{ticker}_bs{use_baseline_data_bool}_mac{use_macro_data_bool}_ex{use_extra_data_bool}_sc{scale}_nt{use_nonlinear_transformation_bool}_w{target_window}_linear_model.pkl"

                    start_time = time.time()
                    predictions = predict_linear_model(
                        model_path=model_path,
                        rawdf=rawdf,
                        scaler_X=scale,
                        use_nonlinear_transformation_bool=use_nonlinear_transformation_bool,
                        use_baseline_data_bool=use_baseline_data_bool,
                        use_macro_data_bool=use_macro_data_bool,
                        use_extra_data_bool=use_extra_data_bool,
                        target_window=target_window
                    )

                    st.success("Prediction completed!")
                    duration = time.time() - start_time
                    st.write(f" **Training time:** {duration:.2f} seconds")

                    st.subheader("Prediction Results")

                    st.write(predictions)

                    try:
                        st.subheader(f"Next {target_window} prediction plot")

                        last_5_dates = rawdf["Date"].tail(5)
                        last_5_values = rawdf["ClosePriceReturn"].tail(5)


                        start_date = pd.to_datetime(np.array(last_5_dates)[-1])

                        future_dates = []
                        current = start_date + timedelta(days=1)

                        while len(future_dates) < 5:
                            if current.weekday() < 5:
                                future_dates.append(current)
                            current += timedelta(days=1)



                        trace_actual = go.Scatter(
                            x=last_5_dates,
                            y=last_5_values,
                            mode='lines+markers',
                            name='Actual + Start',
                            line=dict(color='blue')
                        )

                        future_dates = future_dates[:len(predictions)]
                        predictions = np.array(predictions).flatten()
                        trace_predicted = go.Scatter(
                            x=future_dates,
                            y=predictions,
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='red', dash='dash')
                        )
                        trace_bridge = go.Scatter(
                            x=[last_5_dates.iloc[-1], future_dates[0]],
                            y=[last_5_values.iloc[-1], predictions[0]],
                            mode='lines',
                            name='Bridge',
                            line=dict(color='gray', dash='dot')
                        )


                        layout = go.Layout(
                            title=f"Actual vs Predicted ClosePriceReturn for {ticker}",
                            xaxis=dict(title='Date'),
                            yaxis=dict(title='Return'),
                            legend=dict(x=0, y=1),
                            template="plotly_white"
                        )

                        fig = go.Figure(data=[trace_actual, trace_bridge, trace_predicted], layout=layout)

                        st.plotly_chart(fig)

                    except Exception as e:
                        st.error(f"Plotting failed: {e}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        except:
            st.error("Please select ticker and load the data first.")

    if page == "Train Neural Network":
        try:
            ticker = st.session_state["ticker"]
            st.subheader(f"Running SNN model for {ticker}: please configure model parameters")

            col1, col2 = st.columns(2)
            with col1:
                epochs = st.number_input(
                    "Training Epochs",
                    min_value=100, max_value=5000, value=1000, step=100
                )
                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f"
                )
                hidden_size = st.number_input(
                    "Hidden Neurons",
                    min_value=1, max_value=100, value=1, step=1
                )
            with col2:
                scale = st.checkbox("Scale Input Data")
                use_macro_data_bool = st.checkbox("Use Macro Data")
                use_extra_data_bool = st.checkbox("Use Extra Data (Anomalies/Sentiment)")
                target_window = st.selectbox(
                    "Target Window (days)",
                    options=[1, 3, 5],
                    index=0
                )

            run_button = st.button("Run Neural Network Model")

            if run_button:
                st.info("Training Simple Neural Network... please wait")
                start_time = time.time()
                results = run_single_neural_network(
                    ticker=ticker,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    hidden_size=hidden_size,
                    scale=scale,
                    use_macro_data_bool=use_macro_data_bool,
                    use_extra_data_bool=use_extra_data_bool,
                    target_window=target_window,
                    cachedir="cachedir"
                )

                if results is None:
                    st.error("Model training failed. Check your settings or data.")
                else:
                    duration = time.time() - start_time
                    st.write(f" **Training time:** {duration:.2f} seconds")
                    st.success("Model training completed!")
                    st.subheader("Simple Neural Network Model Results")
                    st.dataframe(results)

                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="simple_nn_single_model_result.csv",
                        mime='text/csv',
                    )
        except:
            st.error("Please select ticker and load the data first.")

    if page == "Predict Neural Network":
        try:
            ticker = st.session_state["ticker"]
            st.subheader(f"Predicting using SNN for {ticker}: please configure prediction parameters")

            col1, col2 = st.columns(2)
            with col1:
                scale = st.checkbox("Scale Input Data")
                use_macro_data_bool = st.checkbox("Use Macro Data")
                use_extra_data_bool = st.checkbox("Use Extra Data (Anomalies/Sentiment)")
            with col2:
                target_window = st.selectbox(
                    "Target Window (days)",
                    options=[1, 3, 5],
                    index=0
                )

            predict_button = st.button("Run Prediction")

            if predict_button:
                st.info("Running prediction... please wait")

                rawdf = st.session_state["processed_df"]
                use_baseline_data_bool = True
                try:
                    model_path = f"cachedir/{ticker}_bs{use_baseline_data_bool}_mac{use_macro_data_bool}_ex{use_extra_data_bool}_sc{scale}_w{target_window}_neural_network.pkl"

                    start_time = time.time()
                    predictions = predict_neural_network(
                        model_path=model_path,
                            rawdf=rawdf,
                            scaler_X=scale,
                            use_baseline_data_bool=True,
                            use_macro_data_bool=use_macro_data_bool,
                            use_extra_data_bool=use_extra_data_bool,
                            target_window=target_window
                    )

                    st.success("Prediction completed!")
                    duration = time.time() - start_time
                    st.write(f" **Training time:** {duration:.2f} seconds")
                    st.subheader("Prediction Results")
                    st.write(predictions)
                    try:
                        st.subheader(f"Next {target_window} days prediction plot")

                        last_5_dates = rawdf["Date"].tail(5)
                        last_5_values = rawdf["ClosePriceReturn"].tail(5)


                        start_date = pd.to_datetime(np.array(last_5_dates)[-1])

                        future_dates = []
                        current = start_date + timedelta(days=1)

                        while len(future_dates) < 5:
                            if current.weekday() < 5:
                                future_dates.append(current)
                            current += timedelta(days=1)



                        trace_actual = go.Scatter(
                            x=last_5_dates,
                            y=last_5_values,
                            mode='lines+markers',
                            name='Actual + Start',
                            line=dict(color='blue')
                        )

                        future_dates = future_dates[:len(predictions)]
                        predictions = np.array(predictions).flatten()
                        trace_predicted = go.Scatter(
                            x=[last_5_dates.iloc[-1]] + future_dates,
                            y=[last_5_values.iloc[-1]] + predictions.tolist(),
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='red', dash='dash', width=2),
                            marker=dict(symbol='circle', size=6)
                        )

                        layout = go.Layout(
                            title=f"Actual vs Predicted ClosePriceReturn for {ticker}",
                            xaxis=dict(title='Date'),
                            yaxis=dict(title='Return'),
                            legend=dict(x=0, y=1),
                            template="plotly_white"
                        )

                        fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)

                        st.plotly_chart(fig)

                    except Exception as e:
                        st.error(f"Plotting failed: {e}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        except:
            st.error("Please select ticker and load the data first.")


    if page == "Train Bayesian Model":
        try:
            ticker = st.session_state["ticker"]
            st.subheader(f"Running BNN model for {ticker}: please configure model parameters")

            col1, col2 = st.columns(2)
            with col1:
                trainepochs = st.number_input(
                    "Training Epochs",
                    min_value=100, max_value=5000, value=1000, step=100
                )
                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f"
                )
                clip_grad_value = st.number_input(
                    "Clip Gradient Value",
                    min_value=0.1, max_value=10.0, value=0.1, step=0.1
                )
            with col2:
                hidden_neurons = st.number_input(
                    "Hidden Neurons",
                    min_value=1, max_value=9, value=1, step=1
                )
                predsamples = st.number_input(
                    "Prediction Samples",
                    min_value=1, max_value=100, value=15, step=1
                )
                target_window = st.selectbox(
                    "Target Window (days)",
                    options=[1, 3, 5],
                    index=0
                )

            st.subheader("Data Settings")
            col3, col4, col5 = st.columns(3)
            with col3:
                scale = st.checkbox("Scale Input Data")
            with col4:
                use_macro_data_bool = st.checkbox("Use Macro Data")
            with col5:
                use_extra_data_bool = st.checkbox("Use Extra Data (Anomalies/Sentiment)")

            run_button = st.button("Run Bayesian Model")

            if run_button:
                st.info("Training Bayesian Neural Network... please wait")
                start_time = time.time()
                model, actual, predicted, results_df = run_bayesian_single_model(
                    ticker=ticker,
                    trainepochs=trainepochs,
                    learning_rate=learning_rate,
                    clip_grad_value=clip_grad_value,
                    hidden_neurons=hidden_neurons,
                    scale=scale,
                    predsamples=predsamples,
                    use_baseline_data_bool=True,
                    use_macro_data_bool=use_macro_data_bool,
                    use_extra_data_bool=use_extra_data_bool,
                    target_window=target_window,
                    cachedir="cachedir"
                )
                save_path = f"{ticker}_bs{True}_mac{use_macro_data_bool}_ex{use_extra_data_bool}_sc{scale}_w{target_window}_bayesian_model"
                st.session_state[save_path] = model


                if results_df is None:
                    st.error("Model training failed. Check your settings or data.")
                else:
                    print(model)
                    duration = time.time() - start_time
                    st.write(f" **Training time:** {duration:.2f} seconds")
                    st.success("Model training completed!")
                    st.subheader("Bayesian Model Results")
                    st.dataframe(results_df)

                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="bayesian_single_model_result.csv",
                        mime='text/csv',
                )
        except:
            st.error("Please select ticker and load the data first.")


    if page == "Predict Bayesian Model":
        import pickle

        try:
            ticker = st.session_state["ticker"]
            processedf = st.session_state["processed_df"]

            st.subheader(f"Predicting using BNN for {ticker}: please configure model parameters")

            col1, col2 = st.columns(2)
            with col1:
                scale = st.checkbox("Scale Input Data")
                use_macro_data_bool = st.checkbox("Use Macro Data")
                use_extra_data_bool = st.checkbox("Use Extra Data (Anomalies/Sentiment)")
            with col2:
                target_window = st.selectbox(
                    "Target Window (days)",
                    options=[1, 3, 5],
                    index=0
                )
                predsamples = st.number_input(
                    "Prediction Samples",
                    min_value=1, max_value=100, value=15, step=5
                )
                predict_button = st.button("Run Bayesian Prediction")

            if predict_button:
                st.info("Running prediction... please wait")

                print("try to access processeddf")

                use_baseline_data_bool = True

                try:
                    start_time = time.time()
                    save_path = f"cachedir/{ticker}_bs{True}_mac{use_macro_data_bool}_ex{use_extra_data_bool}_sc{scale}_w{target_window}_bayesian_model"
                    model = st.session_state[save_path]

                    mean_predictions, mean_variances = predict_bayesian_model(
                        processedf=processedf,
                        target_window=target_window,
                        model = model,
                        predsamples=predsamples,
                        use_baseline_data_bool=use_baseline_data_bool,
                        use_macro_data_bool=use_macro_data_bool,
                        use_extra_data_bool=use_extra_data_bool,
                        scale=scale
                    )
                    predictions = mean_predictions.detach().numpy().flatten()
                    variances = mean_variances.detach().numpy().flatten()
                    st.success("Prediction completed!")
                    duration = time.time() - start_time
                    st.write(f" **Training time:** {duration:.2f} seconds")

                    st.subheader("Prediction Results")
                    st.write(predictions)
                    try:
                        st.subheader(f"Next {target_window} days prediction plot")

                        last_5_dates = processedf["Date"].tail(5)
                        last_5_values = processedf["ClosePriceReturn"].tail(5)


                        start_date = pd.to_datetime(np.array(last_5_dates)[-1])

                        future_dates = []
                        current = start_date + timedelta(days=1)

                        while len(future_dates) < 5:
                            if current.weekday() < 5:
                                future_dates.append(current)
                            current += timedelta(days=1)

                        std_dev = variances
                        upper_bound = predictions + std_dev
                        lower_bound = predictions - std_dev

                        trace_actual = go.Scatter(
                            x=last_5_dates,
                            y=last_5_values,
                            mode='lines+markers',
                            name='Actual + Start',
                            line=dict(color='blue')
                        )

                        future_dates = future_dates[:len(predictions)]
                        predictions = np.array(predictions).flatten()
                        trace_predicted = go.Scatter(
                            x=[last_5_dates.iloc[-1]] + future_dates,
                            y=[last_5_values.iloc[-1]] + predictions.tolist(),
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='red', dash='dash', width=2),
                            marker=dict(symbol='circle', size=6)
                        )
                     
                        layout_1 = go.Layout(
                            title=f"Actual vs Predicted ClosePriceReturn for {ticker}",
                            xaxis=dict(title='Date'),
                            yaxis=dict(title='Return'),
                            legend=dict(x=0, y=1),
                            template="plotly_white"
                        )
                   
                        fig_2 = go.Figure(data=[trace_actual, trace_predicted], layout=layout_1)

                        st.plotly_chart(fig_2)


                    except Exception as e:
                        st.error(f"Plotting failed: {e}")


                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        except:
            st.error("Please select ticker and load the data first.")


if __name__ == "__main__":
    main()


