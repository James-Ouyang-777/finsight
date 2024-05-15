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
            historical_datas[ticker] = get_data(ticker, start_date="01/01/2024")

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

        tickers = st.multiselect("Choose Ticker Symbols", ticker_list, [ticker_list[0]])


        for key, df in all_data.items():
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'date'}, inplace=True)

        concatenated_df = pd.concat(all_data.values(), ignore_index=True)

        concatenated_df = concatenated_df[concatenated_df['ticker'].isin(tickers)]

        chart = (
            alt.Chart(concatenated_df)
            .mark_area(opacity=0.4)
            .encode(
                x="date:T",
                y=alt.Y("close", stack=None),
                color="ticker:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)

        st.dataframe(concatenated_df)

        model = load_model("model_training/lstm_model.h5")
        print(concatenated_df.head())
        answer = get_predictions(new_data=concatenated_df,model=model)
        date = concatenated_df.iloc[-1]['date']

        st.write(f'prediction based on data up to {date} for tommorow\'s close: {answer}')




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