# Designed & Written by James Ouyang

from urllib.error import URLError
import altair as alt
import pandas as pd
import yfinance as yf
import yahoo_fin.options as ops
from yahoo_fin.stock_info import get_data
import streamlit as st
from streamlit.hello.utils import show_code

import streamlit as st
from streamlit.logger import get_logger

from helpers.data import price_transform, get_predictions
from tensorflow.keras.models import load_model

LOGGER = get_logger(__name__)


# def run():
#     st.set_page_config(
#         page_title="Hello",
#         page_icon="👋",
#     )

# if __name__ == "__main__":
#     run()

def dfs():
    ticker_list = ['META', 'AMZN', 'AAPL', 'NFLX', 'GOOG', 'BA', 'AMD', 'TSLA']

    @st.cache_data
    def get_BIGTECH_data():
        historical_datas = {}
        for ticker in ticker_list:
            historical_datas[ticker] = get_data(ticker, start_date="03/01/2024")

        return historical_datas

######### Main App Design
    try:

        # Get some market data yahoo finance?

        # list of gpt generated facts + news?

        # I would use this to create a trading plan

        # st.multiselect: filter for US sectors tech, non-tech, commodities, etc..

        # time series predictor based on last 7 days or last 3 days can select options?
        all_data = get_BIGTECH_data()
        # data = pd.DataFrame(all_data['NFLX'])
        # st.dataframe(data)

        #multilayer comparison

        s_ticker = st.selectbox("Choose Ticker Symbol", ticker_list)


        for key, df in all_data.items():
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'date'}, inplace=True)

        concatenated_df = pd.concat(all_data.values(), ignore_index=True)

        # concatenated_df = all_data
        concatenated_df = concatenated_df[concatenated_df['ticker'] == s_ticker]

        df = concatenated_df

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

        # y_min = df['low'].min() * 0.98  # 5% below the minimum low value
        # y_max = df['high'].max() * 1.02  # 5% above the maximum high value


        # # Create the base chart with the line
        # line = alt.Chart(concatenated_df).mark_line().encode(
        #     x="date:T",
        #     y=alt.Y("close", stack=None),
        #     color="ticker:N",
        # )

        # # Create the points with filled circles
        # points = alt.Chart(concatenated_df).mark_point(filled=True).encode(
        #     x="date:T",
        #     y=alt.Y("close", stack=None),
        #     color="ticker:N",
        # )

        # # Layer the line and points together
        # chart = alt.layer(line, points).properties(
        #     width=600,
        #     height=400,
            
        # )


        st.altair_chart(candlestick_chart, use_container_width=True)

        # st.dataframe(concatenated_df)

        model = load_model("model_training/lstm_model.h5")
        print(concatenated_df.head())
        answer = get_predictions(new_data=concatenated_df,model=model)
        date = concatenated_df.iloc[-1]['date']

        st.write(f'LSTM RNN prediction based on data up to {date} for next close: {answer}')




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