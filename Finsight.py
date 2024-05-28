# Designed & Written by James Ouyang

from urllib.error import URLError
import altair as alt
import pandas as pd
import yfinance as yf
import yahoo_fin.options as ops
from yahoo_fin.stock_info import get_data
import streamlit as st
from streamlit.hello.utils import show_code
from datetime import datetime, timedelta

import streamlit as st
from streamlit.logger import get_logger

from helpers.data import price_transform, get_predictions
from tensorflow.keras.models import load_model

LOGGER = get_logger(__name__)


def dfs():
    ticker_list = ['META', 'AMZN', 'AAPL', 'NFLX', 'GOOG', 'AMD', 'TSLA', 'NVDA']

    @st.cache_data
    def get_BIGTECH_data():
        historical_datas = {}
        for ticker in ticker_list:
            historical_datas[ticker] = get_data(ticker, start_date="03/03/2024")

        return historical_datas

######### Main App Design
    try:

        all_data = get_BIGTECH_data()

        s_ticker = st.selectbox("Choose Ticker Symbol", ticker_list)


        for key, df in all_data.items():
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'date'}, inplace=True)

        concatenated_df = pd.concat(all_data.values(), ignore_index=True)

        concatenated_df = concatenated_df[concatenated_df['ticker'] == s_ticker]

        df = concatenated_df

        def ytd_chart():
            # Calculate y-axis range
            y_min = df['low'].min() * 0.95  # 5% below the minimum low value
            y_max = df['high'].max() * 1.05  # 5% above the maximum high value

            # Base chart
            base = alt.Chart(df).encode(
                x=alt.X('date:T', title='Date')
            )

            # Rule for the high-low range
            rule = base.mark_rule().encode(
                y=alt.Y('low:Q', title='Price', scale=alt.Scale(domain=[y_min, y_max])),
                y2='high:Q'
            )

            # Bars for the open-close range
            bars = base.mark_bar().encode(
                y='open:Q',
                y2='close:Q',
                color=alt.condition("datum.open <= datum.close",
                                    alt.value("green"),  # The bar is green if the close is higher than the open
                                    alt.value("red"))  # The bar is red if the close is lower than the open
            )

            # Layer the rule and bars to create the candlestick chart
            candlestick_chart = alt.layer(rule, bars).properties(
                width=600,
                height=400,
                title='Candlestick Chart'
            )


            st.altair_chart(candlestick_chart, use_container_width=True)

        # st.dataframe(concatenated_df)

        models = ["model_training/LSTM_predict_open.h5", 
                  "model_training/LSTM_predict_high.h5",
                  "model_training/LSTM_predict_low.h5",
                  "model_training/LSTM_predict_close.h5",]
        
        date = concatenated_df.iloc[-1]['date']

        
        # st.write(f'LSTM RNN prediction based on data leading up to {date} for next day')
        prediction_entry = [date+timedelta(days=1)]

        for mpath in models:
            model = load_model(mpath)
            answer = get_predictions(new_data=concatenated_df,model=model)
            # st.write(f'{mpath[28:-3]}: ${round(answer[0],2)}')
            prediction_entry.append(answer[0])

        # def flatten_range(op,hi,lo,clo):
        #     hi = max(op,clo,hi)
        #     lo = min(op,clo,lo)
        #     return (op, hi, lo, clo)
        # def reasonable_range():
            # if prediction_entry[1] < prediction_entry[4]:
            #     prediction_entry[1], prediction_entry[4] = prediction_entry[4], prediction_entry[1]
        prediction_entry[2] = max(prediction_entry[1], prediction_entry[2], prediction_entry[4])
        prediction_entry[3] = min(prediction_entry[1], prediction_entry[3], prediction_entry[4])
    
        # reasonable_range()
        prediction_entry += [0, 0,'ticker']
        ### New prediction display

        df2 = concatenated_df[-10:]
        print(df2.columns)
        # print(prediction_entry.columns)
        df2.loc[len(df2)] = prediction_entry
        print("ADDED")

        def two_week_chart():
            # Calculate y-axis range
            y_min = df2['low'].min() * 0.99  # 5% below the minimum low value
            y_max = df2['high'].max() * 1.01  # 5% above the maximum high value

            # Base chart
            base = alt.Chart(df2).encode(
                x=alt.X('date:T', title='Date', axis=alt.Axis(labelAngle=-45), scale=alt.Scale(padding=0.1))

            )

            # Rule for the high-low range
            rule = base.mark_rule(size=2).encode(
                y=alt.Y('low:Q', title='Price', scale=alt.Scale(domain=[y_min, y_max])),
                y2='high:Q'
            )

            # Bars for normal volume
            normal_bars = base.mark_bar(size=20).encode(
                y='open:Q',
                y2='close:Q',
                color=alt.condition(
                    alt.datum.open <= alt.datum.close,
                    alt.value("green"),  # Green if close is higher than open
                    alt.value("red")     # Red if close is lower than open
                )
            ).transform_filter(
                alt.datum.volume > 0
            )

            # Bars for zero volume
            zero_volume_bars = base.mark_bar(size=22, stroke='lightblue', strokeWidth=2
                ).encode(
                y='open:Q',
                y2='close:Q',
                color=alt.condition(
                    alt.datum.open <= alt.datum.close,
                    alt.value("lightgreen"),  # Light green if close is higher than open and volume is 0
                    alt.value("lightcoral")   # Light red if close is lower than open and volume is 0
                )
            ).transform_filter(
                alt.datum.volume == 0
            )

            # Layer the rule and all bars to create the candlestick chart
            candlestick_chart = alt.layer(rule, normal_bars, zero_volume_bars).properties(
                width=600,
                height=400,
                title='2 Week Chart'
            )

            # Display the chart in the Streamlit container
            st.altair_chart(candlestick_chart, use_container_width=True)
        two_week_chart()

        st.write(f'Model prediction based on data leading up to {date} for next day')
        # prediction_entry = [date+timedelta(days=1)]

        ind = 1
        for mpath in models:
            answer = prediction_entry[ind]
            ind+=1
            st.write(f'{mpath[28:-3]}: ${round(answer,2)}')

    except URLError as e:
        st.error(
            """
            **This page requires internet access.**
            Connection error: %s
        """
            % e.reason
        )


st.set_page_config(page_title=" Stock Trader Insights", page_icon="📊")
st.markdown("# Stock Trader Insights")
st.sidebar.header("Stock Insights")

dfs()

# show_code(dfs)