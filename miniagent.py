import dash
from dash import html, dcc, Input, Output
import requests
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from a .env file locally
# On Render, this is automatically handled by the Environment Variables section.
load_dotenv() 

# === API KEYS & CLIENT INITIALIZATION ===
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)


# === TICKER TO COMPANY NAME MAPPING ===
ticker_to_company = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "GOOGL": "Google", "AMZN": "Amazon",
    "META": "Meta", "TSLA": "Tesla", "BRK-B": "Berkshire Hathaway", "UNH": "UnitedHealth", "XOM": "Exxon Mobil",
    "LLY": "Eli Lilly", "JPM": "JPMorgan", "JNJ": "Johnson & Johnson", "V": "Visa", "PG": "Procter & Gamble",
    "AVGO": "Broadcom", "HD": "Home Depot", "MA": "Mastercard", "MRK": "Merck", "PEP": "PepsiCo",
    "ABBV": "AbbVie", "COST": "Costco", "BAC": "Bank of America", "ADBE": "Adobe", "KO": "Coca-Cola",
    "CVX": "Chevron", "WMT": "Walmart", "ACN": "Accenture", "TMO": "Thermo Fisher", "MCD": "McDonald's",
    "CSCO": "Cisco", "PFE": "Pfizer", "INTC": "Intel", "NFLX": "Netflix", "AMD": "AMD", "NKE": "Nike",
    "CRM": "Salesforce", "LIN": "Linde", "QCOM": "Qualcomm", "ABT": "Abbott", "TXN": "Texas Instruments",
    "NEE": "NextEra Energy", "WFC": "Wells Fargo", "AMGN": "Amgen", "DHR": "Danaher", "UNP": "Union Pacific",
    "ORCL": "Oracle", "HON": "Honeywell", "LOW": "Lowe's", "SBUX": "Starbucks"
}

tickers = list(ticker_to_company.keys())

server = dash.Dash(__name__)
app = server.server # Expose the underlying Flask server for Gunicorn/compatibility
server.title = "Market Analysis Agent with GPT"

# === RSI CALCULATION ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    # Handle division by zero for initial periods gracefully
    rs = rs.replace([float('inf'), float('-inf')], 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# === FETCH STOCK DATA ===
def fetch_stock_data(ticker):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
    res = requests.get(url)
    data = res.json()
    
    # Check for API limit error gracefully
    if "Error Message" in data or "Note" in data:
        print("Alpha Vantage API error:", data.get("Error Message") or data.get("Note"))
        return pd.DataFrame(), {'price': 'N/A', 'ma20': 'N/A', 'rsi': 'N/A', 'volume': 'N/A'}

    try:
        ts = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        latest = df.iloc[-1]
        close_price = round(latest["4. close"], 2)
        volume = int(latest["5. volume"])
        ma20 = round(df["4. close"].rolling(window=20).mean().iloc[-1], 2)
        rsi = round(compute_rsi(df["4. close"]), 2)

        return df, {
            'price': close_price,
            'ma20': ma20,
            'rsi': rsi,
            'volume': volume
        }
    except Exception as e:
        print("Stock data processing error:", e)
        return pd.DataFrame(), {'price': 'N/A', 'ma20': 'N/A', 'rsi': 'N/A', 'volume': 'N/A'}

# === FETCH NEWS HEADLINES ===
def fetch_news(ticker):
    company = ticker_to_company.get(ticker, ticker)
    url = (
        f"https://newsapi.org/v2/top-headlines?"
        f"q={company}&language=en&pageSize=5&apiKey={NEWSAPI_KEY}"
    )

    try:
        res = requests.get(url)
        # Check for NewsAPI errors
        if res.json().get("status") == "error":
            print("NewsAPI error:", res.json().get("message"))
            return ["NewsAPI Error: Check API Key or limits."]

        articles = res.json().get("articles", [])

        headlines = [a["title"] for a in articles if "title" in a and a["title"] != "[Removed]"]
        print(f"🔍 Top headlines for {company}: {headlines}")

        return headlines if headlines else ["No headlines found"]
    except Exception as e:
        print("News fetch error:", e)
        return ["No headlines found due to connection error."]

    
# === GPT SUMMARY ===
def summarize_news_with_gpt(headlines, ticker):
    if not headlines or headlines == ["No headlines found"]:
        return "No news to summarize."

    try:
        # Construct the full prompt for the model
        prompt = f"Summarize these stock-related headlines about {ticker}: " + "\n".join(headlines)
        
        # 💥 CRITICAL FIX: Use 'messages' list for the gpt-3.5-turbo model
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are a concise financial analyst summarizing news. Keep the summary under 100 words."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        
        # Access the result using the correct path: choices[0].message.content
        return response.choices[0].message.content.strip() 
        
    except Exception as e:
        print("GPT summary error:", e)
        # The log confirms this is the failure point, but we keep the handler for safety
        return "Unable to summarize news due to an internal API error. Check the Render logs for details."

# === DASH LAYOUT ===
server.layout = html.Div([
    html.H1("📈 Market Analysis Dashboard with GPT", style={'textAlign': 'center'}),
    html.Div([
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': t, 'value': t} for t in tickers],
            value='AAPL',
            style={'width': '100%'}
        )
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Br(),
    dcc.Graph(id='candlestick-chart'),
    html.Div(id='stock-data', style={'textAlign': 'center', 'fontSize': 18}),
    html.Div(id='news-data', style={'width': '70%', 'margin': 'auto', 'padding': '10px'}),
    html.Div(id='summary-data', style={'width': '70%', 'margin': 'auto', 'padding': '10px'})
])

# === CALLBACK ===
@server.callback( # Changed 'app.callback' to 'server.callback'
    [Output('candlestick-chart', 'figure'),
     Output('stock-data', 'children'),
     Output('news-data', 'children'),
     Output('summary-data', 'children')],
    [Input('stock-dropdown', 'value')]
)
def update_dashboard(ticker):
    df, data = fetch_stock_data(ticker)
    headlines = fetch_news(ticker)
    summary = summarize_news_with_gpt(headlines, ticker)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
    else:
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['1. open'],
            high=df['2. high'],
            low=df['3. low'],
            close=df['4. close']
        )])
        fig.update_layout(title=f'{ticker} Candlestick Chart', xaxis_title='Date', yaxis_title='Price')

    stock_info = html.Div([
        html.P(f"📍 Ticker: {ticker}"),
        html.P(f"💲 Price: ${data['price']}"),
        html.P(f"📈 MA20: {data['ma20']}"),
        html.P(f"⚠️ RSI: {data['rsi']}"),
        html.P(f"📉 Volume: {data['volume']}")
    ])

    news_items = html.Div([
        html.H3("📰 Top News Headlines"),
        html.Ul([html.Li(headline) for headline in headlines])
    ])

    summary_block = html.Div([
        html.H3("🧠 AI Summary"),
        html.P(summary)
    ])

    return fig, stock_info, news_items, summary_block

# === RUN ===
if __name__ == '__main__':
    server.run_server(debug=True)