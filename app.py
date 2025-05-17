import os
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
import random

os.environ["GOOGLE_API_KEY"] = ""
ALPHA_VANTAGE_API_KEY = "demo"

def fetch_stock_data_av(symbol, max_retries=3):
    base_url = "https://www.alphavantage.co/query"
    
    
    try:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "compact"
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            st.warning(f"No price data found for {symbol}")
            return None
            
        
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        
        time.sleep(12)
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        response = requests.get(base_url, params=params)
        info = response.json()
        
        return {
            'history': df.sort_index(),
            'info': {
                'name': info.get('Name', symbol),
                'sector': info.get('Sector', 'N/A'),
                'market_cap': info.get('MarketCapitalization', 'N/A'),
                'summary': info.get('Description', 'N/A')
            }
        }
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=600)
def fetch_stock_data(symbol):
    return fetch_stock_data_av(symbol)

def compare_stocks(symbols):
    data = {}
    stock_info = {}
    progress_text = st.empty()
    
    for i, symbol in enumerate(symbols, 1):
        progress_text.text(f"Fetching data for {symbol} ({i}/{len(symbols)})...")
        try:
            stock_data = fetch_stock_data(symbol)
            if stock_data is not None:
                hist = stock_data['history']
                if not hist.empty:
                    start_price = float(hist['Close'].iloc[-30])
                    end_price = float(hist['Close'].iloc[-1])
                    percent_change = ((end_price - start_price) / start_price) * 100
                    data[symbol] = percent_change
                    stock_info[symbol] = stock_data['info']
            time.sleep(12)
        except Exception as e:
            st.warning(f"Could not analyze {symbol}: {str(e)}")
            continue
    
    progress_text.empty()
    if not data:
        st.error("Could not fetch data for any of the provided symbols. Please try again later.")
    return data, stock_info

def get_company_info(symbol):
    stock_data = fetch_stock_data(symbol)
    if stock_data is not None:
        return stock_data['info']
    return {
        "name": symbol,
        "sector": "N/A",
        "market_cap": "N/A",
        "summary": "Information temporarily unavailable",
    }

def get_company_news(symbol):
    try:
        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "limit": 5
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        news = data.get('feed', [])
        return news[:5] if news else []
    except Exception as e:
        st.warning(f"Could not fetch news for {symbol}")
        return []


market_analyst = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Trading market analyst that evaluates stock performance and market trends.",
    instructions=[
        "Analyze stock price movements and trading patterns.",
        "Calculate key technical indicators and trend analysis.",
        "Identify potential trading opportunities based on price action."
    ],
    show_tool_calls=True,
    markdown=True
)

def get_market_analysis(symbols):
    performance_data, _ = compare_stocks(symbols)
    if not performance_data:
        return "No valid stock data found for the given symbols."
    analysis = market_analyst.run(f"Compare these stock performances: {performance_data}")
    return analysis.content



company_researcher = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Company and market research specialist for trading decisions.",
    instructions=[
        "Analyze company fundamentals and market position.",
        "Evaluate news sentiment and market impact.",
        "Identify potential catalysts for price movement."
    ],
    markdown=True
)

def get_company_analysis(symbol):
    info = get_company_info(symbol)
    news = get_company_news(symbol)
    response = company_researcher.run(
        f"Provide an analysis for {info['name']} in the {info['sector']} sector.\n"
        f"Market Cap: {info['market_cap']}\n"
        f"Summary: {info['summary']}\n"
        f"Latest News: {news}"
    )
    return response.content


stock_strategist = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Trading strategist for market opportunities and risk management.",
    instructions=[
        "Develop trading strategies based on technical and fundamental analysis.",
        "Evaluate risk-reward ratios and market conditions.",
        "Provide specific entry and exit points for trades."
    ],
    markdown=True
)

def get_stock_recommendations(symbols):
    market_analysis = get_market_analysis(symbols)
    data = {}
    for symbol in symbols:
        data[symbol] = get_company_analysis(symbol)
        time.sleep(1)
    recommendations = stock_strategist.run(
        f"Based on the market analysis: {market_analysis}, and company news {data}, "
        f"which stocks would you recommend for investment?"
    )
    return recommendations.content


team_lead = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Lead trading analyst coordinating market insights and trade recommendations.",
    instructions=[
        "Synthesize technical, fundamental, and sentiment analysis.",
        "Generate actionable trading strategies with risk management.",
        "Provide clear buy, sell, or hold recommendations with price targets."
    ],
    markdown=True
)

def get_final_investment_report(symbols):
    market_analysis = get_market_analysis(symbols)
    company_analyses = [get_company_analysis(symbol) for symbol in symbols]
    stock_recommendations = get_stock_recommendations(symbols)

    final_report = team_lead.run(
        f"Market Analysis:\n{market_analysis}\n\n"
        f"Company Analyses:\n{company_analyses}\n\n"
        f"Stock Recommendations:\n{stock_recommendations}\n\n"
        f"Provide the full analysis of each stock with Fundamentals and market news."
        f"Generate a final ranked list in ascending order on which should I buy."
    )
    return final_report.content



st.set_page_config(page_title="AI Trading Agent", page_icon="🤖", layout="wide")


st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">🤖 AI Trading Agent</h1>
    <h3 style="text-align: center; color: #6c757d;">Advanced stock analysis and trading recommendations powered by AI</h3>
""", unsafe_allow_html=True)


st.sidebar.markdown("""
    <h2 style="color: #343a40;">Trading Configuration</h2>
    <p style="color: #6c757d;">Enter the stock symbols you want to analyze. The AI agent will analyze market data, trends, and provide trading recommendations.</p>
""", unsafe_allow_html=True)


input_symbols = st.sidebar.text_input("Enter Stock Symbols (separated by commas)", "AAPL, MSFT, GOOGL")
alpha_vantage_key = st.sidebar.text_input("Enter your Alpha Vantage API Key", type="password")


if alpha_vantage_key:
    ALPHA_VANTAGE_API_KEY = alpha_vantage_key


symbols = [symbol.strip() for symbol in input_symbols.split(",")]


if st.sidebar.button("Generate Investment Report"):
    if not symbols:
        st.sidebar.warning("Please enter at least one stock symbol.")
    elif ALPHA_VANTAGE_API_KEY == "demo":
        st.sidebar.warning("Please enter your Alpha Vantage API Key. Get one for free at: https://www.alphavantage.co/support/#api-key")
    else:
        with st.spinner("Generating investment report... This may take a few minutes due to API rate limits."):
            try:
                
                report = get_final_investment_report(symbols)
                
                
                st.subheader("Investment Report")
                st.markdown(report)
                
                
                try:
                    st.markdown("### 📊 Stock Performance (Last 30 Days)")
                    data = {}
                    for symbol in symbols:
                        stock_data = fetch_stock_data(symbol)
                        if stock_data is not None:
                            data[symbol] = stock_data['history']['Close']
                    
                    if data:
                        df = pd.concat(data, axis=1)
                        st.line_chart(df)
                except Exception as e:
                    st.warning("Could not generate performance chart.")
                    
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.info("Please try again with different symbols or wait a few minutes before retrying.")


