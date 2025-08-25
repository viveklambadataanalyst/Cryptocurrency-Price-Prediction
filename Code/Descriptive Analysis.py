#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
from collections import Counter
import talib as ta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse, r2_score as r2, mean_absolute_error as mae
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import gc
import shap
import joblib as joblib
import warnings
warnings.filterwarnings('ignore')

#loading, cleaning and sub-setting the data
crypto_prices_original = pd.read_csv("Data\\Crypto File.csv")

#getting list of cryptos based on their available data
crypto_prices_all = crypto_prices_original.copy()
crypto_prices_all = crypto_prices_all.dropna().drop(columns=["adj_close"])
crypto_prices_all["dates"] = pd.to_datetime(crypto_prices_all["dates"])
crypto_prices_all["month"] = crypto_prices_all["dates"].dt.to_period('M')
crypto_prices_all["year"] = crypto_prices_all["dates"].dt.year
crypto_list = crypto_prices_all["symbol"].tolist()
filtered_crypto_prices = crypto_prices_all[crypto_prices_all["symbol"].isin(crypto_list)]
start_date = pd.Timestamp("2015-01-01")
end_date = pd.Timestamp("2025-01-02")
crypto_start_check = filtered_crypto_prices[filtered_crypto_prices["dates"] == start_date]["symbol"].unique()
crypto_end_check = filtered_crypto_prices[filtered_crypto_prices["dates"] == end_date]["symbol"].unique()
missing_start_cryptos = set(crypto_list) - set(crypto_start_check)
missing_end_cryptos = set(crypto_list) - set(crypto_end_check)

if missing_start_cryptos:
    print(f"These cryptos does not have {start_date}: {missing_start_cryptos}")
else:
    print(f"All cryptos have start date")
    
if missing_end_cryptos:
    print(f"These cryptos does not have {end_date}: {missing_end_cryptos}")
else:
    print(f"All cryptos have end date")

crypto_data_counts = (filtered_crypto_prices[(filtered_crypto_prices["dates"] >= start_date) & 
                                            (filtered_crypto_prices["dates"] <= end_date)]
.groupby("symbol")["dates"].count().reset_index())

crypto_data_counts.columns = ["symbol", "data_point_count"]
crypto_data_counts.astype({"data_point_count": "int"}).dtypes
crypto_data_counts = crypto_data_counts.sort_values("data_point_count", ascending=False).reset_index(drop=True)

selected_cryptos = crypto_data_counts[crypto_data_counts["data_point_count"] >= 3655*0.9]
selected_cryptos = selected_cryptos["symbol"].tolist()

#filtering dataframe based on obtained list of cryptos
crypto_prices_6 = crypto_prices_all[crypto_prices_all["symbol"].isin(selected_cryptos)].reset_index(drop=True)
print("Main Dataframe with all data\n")
print(crypto_prices_all)
print("Main Dataframe with 6 Selected Crypto data\n")
print(crypto_prices_6)

#finding the overall min and max values for each crypto	
overall_min_max_crypto = crypto_prices_6.groupby("symbol").agg({"open": ["min", "max"], "high": ["min", "max"], "low": ["min", "max"], "close": ["min", "max"]})
overall_min_max_crypto.columns = ['_'.join(col).strip('_') for col in overall_min_max_crypto.columns]
overall_min_max_crypto = overall_min_max_crypto.reset_index(drop=True)
print(overall_min_max_crypto.columns)
print("Minimum and Maximum Prices of Chosen Cryptos")
print(overall_min_max_crypto.sort_values(by=['open_min', 'open_max', 'high_min', 'high_max', 'low_min','low_max', 'close_min', 'close_max'], ascending=False))

#Simple Moving Average (SMA) for 7 and 50 days
crypto_prices_6["SMA_7"] = ta.SMA(crypto_prices_6['close'], timeperiod=7)
crypto_prices_6["SMA_50"] = ta.SMA(crypto_prices_6['close'], timeperiod=50)
crypto_prices_6["SMA_200"] = ta.SMA(crypto_prices_6['close'], timeperiod=200)

#Exponential Moving Average (EMA) for 7 and 21 days
crypto_prices_6["EMA_9"] = ta.EMA(crypto_prices_6['close'], timeperiod=9)
crypto_prices_6["EMA_21"] = ta.EMA(crypto_prices_6['close'], timeperiod=21)

#Relative Strength Index (RSI) for 7 and 14 days
crypto_prices_6[f"RSI_7"] = ta.RSI(crypto_prices_6['close'], timeperiod=7)
crypto_prices_6[f"RSI_14"] = ta.RSI(crypto_prices_6['close'], timeperiod=14)

#Average Directional Index (ADX) for 14 days
crypto_prices_6["ADX_14"] = ta.ADX(crypto_prices_6["high"], crypto_prices_6["low"], crypto_prices_6["close"], timeperiod=14)

#Fibonnaci Pivot Poits for daily and weekly points
def compute_fibonacci_pivots(df):
    df.loc[:, "P"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
    df.loc[:, "R1"] = df["P"] + (0.382 * (df["high"].shift(1) - df["low"].shift(1)))
    df.loc[:, "R2"] = df["P"] + (0.618 * (df["high"].shift(1) - df["low"].shift(1)))
    df.loc[:, "R3"] = df["P"] + (1.000 * (df["high"].shift(1) - df["low"].shift(1)))
    df.loc[:, "S1"] = df["P"] - (0.382 * (df["high"].shift(1) - df["low"].shift(1)))
    df.loc[:, "S2"] = df["P"] - (0.618 * (df["high"].shift(1) - df["low"].shift(1)))
    df.loc[:, "S3"] = df["P"] - (1.000 * (df["high"].shift(1) - df["low"].shift(1)))
    return df

crypto_prices_6 = crypto_prices_6.groupby("symbol", group_keys=False).apply(compute_fibonacci_pivots).reset_index(drop=True)

#Average True Range (ATR) for 14 days
crypto_prices_6["ATR_14"] = ta.ATR(crypto_prices_6["high"], crypto_prices_6["low"], crypto_prices_6["close"], timeperiod=14)
crypto_prices_6["ATR_PCT"] = crypto_prices_6["ATR_14"] / crypto_prices_6["close"] * 100

#Cumulative On-Balance Volume (OBV)
crypto_prices_6["OBV"] = ta.OBV(crypto_prices_6["close"], crypto_prices_6["volume"])

#classifying and observing market trends using SMA, EMA, and 30 days rolling percentage change 
rolling_window = 30
bullish_threshold = 20
bearish_threshold = -20
sideways_range = (-10, 10)
#between -10 to -20 and +10 to +20 is neutral

crypto_prices_6["rolling_pct_change"] = crypto_prices_6.groupby("symbol")["close"].pct_change(periods=rolling_window) * 100

def classify_market(pct_change):
    if pct_change > bullish_threshold:
        return "Bullish"
    elif pct_change < bearish_threshold:
        return "Bearish"
    else:
        return "Sideways"

crypto_prices_6["market_trend_percent_change"] = crypto_prices_6["rolling_pct_change"].apply(classify_market)

def label_sma(row):
    if row["SMA_50"] > row["SMA_200"]:
        return "Bullish"
    elif row["SMA_50"] < row["SMA_50"]:
        return "Bearish"
    else:
        return "Sideways"

def label_ema(row):
    if row["EMA_9"] > row["EMA_21"]:
        return "Bullish"
    elif row["EMA_9"] < row["EMA_21"]:
        return "Bearish"
    else:
        return "Sideways"

crypto_prices_6["trend_SMA_crossover"] = crypto_prices_6.apply(label_sma, axis=1)
crypto_prices_6["trend_EMA_crossover"] =  crypto_prices_6.apply(label_ema, axis=1)

def final_trend_label(row):
    labels = [row["market_trend_percent_change"], row["trend_SMA_crossover"], row["trend_EMA_crossover"]]
    count=Counter(labels)
    top = count.most_common()
    
    if len(top) == 1 or top[0][1] > top[1][1]:
        direction = top[0][0]
    else:
        for col in ["trend_SMA_crossover", "trend_EMA_crossover", "market_trend_percent_change"]:
            if row[col] == top[0][0]:
                direction = row[col]
                break
            elif row[col] == top[1][0]:
                direction = row[col]
                break
    return direction

crypto_prices_6["market_trend_direction"] = crypto_prices_6.apply(final_trend_label, axis=1)

#oberving and classifying stregth of market trend using ADX values
def label_adx_strength(adx_val):
    if adx_val>25:
        return "Strong"
    elif adx_val<20:
        return "Weak"
    else:
        return "Neutral"

crypto_prices_6["trend_strength_ADX"] = crypto_prices_6["ADX_14"].apply(label_adx_strength)

crypto_prices_6["market_label"] = crypto_prices_6["trend_strength_ADX"] + " " + crypto_prices_6["market_trend_direction"]

def label_rsi_condition(rsi_val):
    if rsi_val > 70:
        return "Overbaught"
    elif rsi_val < 30:
        return "Oversold"
    elif 50 > rsi_val < 70:
        return "Bullish Momentum"
    elif 30 > rsi_val < 50:
        return "Bearish Momentum"
    else:
        return "Neutral"

crypto_prices_6["rsi_condition"] = crypto_prices_6["RSI_14"].apply(label_rsi_condition)

def volatality_label(atr_pct):
    if atr_pct > 5:
        return "High"
    elif 2 > atr_pct < 5:
        return "Moderate"
    else:
        return "Low"

crypto_prices_6["volatality_label"] = crypto_prices_6["ATR_PCT"].apply(volatality_label)
print(crypto_prices_6["market_label"].unique())
print(crypto_prices_6["rsi_condition"].unique())
print(crypto_prices_6["volatality_label"].unique())

crypto_prices_6.to_csv("Data\\Crypto File with 6 Cryptos and Indicators.csv", index=False)
print(crypto_prices_6.isnull().sum())
cols_needed = [
    "dates", "rolling_pct_change", "ATR_PCT", "ATR_14", "S3", "S2", "S1", "R3", "R2", "R1", "P",
    "ADX_14", "RSI_14", "RSI_7", "EMA_21", "EMA_9", "SMA_200", "SMA_50", "SMA_7"
]
crypto_prices_6 = crypto_prices_6.dropna(subset=cols_needed)
print(crypto_prices_6.columns)
print(crypto_prices_6)

#duplicating dataframe for further analysis
crypto_prices_final = crypto_prices_6.copy().reset_index(drop=True)
crypto_prices_final.dropna()
print(crypto_prices_final)

#Now, we will create different dataframes for different time frames to observe trends, market movements closely and clearly
#Market Events identified
#1. Bitcoin Halving (Jul 2016) - Dataframe: btc_halving_2016 - Duration: Jan 2015 – Dec 2016
btc_halving_2016 = crypto_prices_final[(crypto_prices_final["month"] >= "2015-01") & (crypto_prices_final["month"] <= "2016-12")].reset_index(drop=True)
cols_needed = [
    "dates", "rolling_pct_change", "ATR_PCT", "ATR_14", "S3", "S2", "S1", "R3", "R2", "R1", "P",
    "ADX_14", "RSI_14", "RSI_7", "EMA_21", "EMA_9", "SMA_200", "SMA_50", "SMA_7"
]
btc_halving_2016 = btc_halving_2016.dropna(subset=cols_needed).copy()
print(btc_halving_2016[cols_needed].isna().sum())
print("Bitcoin Halving 2016 DataFrame 1\n")
print(btc_halving_2016.reset_index(drop=True))

#Plotting Correlation of BTC-USD with other cryptos for btc_halving_2016
crypto_prices_pivot = btc_halving_2016.pivot(index='dates', columns='symbol', values='close')
cryptos = ['NMC-USD', 'PPC-USD', 'LTC-USD', 'FTC-USD', 'PXC-USD']
BTC_Symbol = "BTC-USD"

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 15), sharex=True)
for i, crypto in enumerate(cryptos):
    rolling_corr = crypto_prices_pivot[crypto].rolling(window=30).corr(crypto_prices_pivot[BTC_Symbol])
    axes[i].plot(crypto_prices_pivot.index, rolling_corr, label=f"{crypto} vs BTC", color='blue')
    axes[i].axhline(0.7, color='red', linestyle='dashed')
    axes[i].axhline(0.5, color='blue', linestyle='dashed')
    axes[i].axhline(0, color='black', linestyle='dashed')
    axes[i].axhline(-0.5, color='blue', linestyle='dashed')
    axes[i].axhline(-0.7, color='red', linestyle='dashed')
    axes[i].set_ylabel("Correlation")
    axes[i].legend()
axes[-1].set_xlabel("Date")
plt.title(f"30 days Rolling Correlation: BTC and Cryptos for BTC Halving 2016")
plt.tight_layout()
plt.show()

sns.heatmap(crypto_prices_pivot.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corr Heatmap between BTC and Cryptos for BTC Halving 2016")
plt.show()

#results suggests low correlation of cryptos with BTC, there does not seems to have any affect of BTC market movements on other cryptos.
#LTC is the only coin with moderate (0.66) correlation with BTC. Hence we can somewhat make interpretation of movements and trends of LTC using BTC but cannot have strong confidence in results
#So all cryptos needs to be analyzed separately for this duration
#for this period we will choose BTC, NMC (-0.55), and PPC (-0.60) as they are least correlated

#plotting indicators for the time frame for BTC-USD
crypto = "BTC-USD"
data = btc_halving_2016[btc_halving_2016["symbol"] == crypto]

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.title(f"{crypto} - Closing Price")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["SMA_7"], label="SMA 7", color="green")
plt.plot(data["dates"], data["SMA_50"], label="SMA 50", color="red")
plt.plot(data["dates"], data["SMA_200"], label="SMA 200", color="orange")
plt.title(f"{crypto} - Closing Price and SMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["EMA_9"], label="EMA 9", color="green")
plt.plot(data["dates"], data["EMA_21"], label="EMA 21", color="red")
plt.title(f"{crypto} - Closing Price and EMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["RSI_7"], label="RSI 7", color="green")
plt.plot(data["dates"], data["RSI_14"], label="RSI 14", color="red")
plt.title(f"{crypto} - RSI - Market Condition")
plt.xlabel("Dates")
plt.ylabel("RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(50, color='orange', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ADX_14"], label="ADX 14", color="blue")
plt.title(f"{crypto} - ADX 14 - Trend Strength")
plt.xlabel("Dates")
plt.ylabel("ADX")
plt.axhline(25, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ATR_PCT"], label="ATR 14", color="blue")
plt.title(f"{crypto} - ATR 14 - Volatility")
plt.xlabel("Dates")
plt.ylabel("ATR Percentage")
plt.axhline(5, color='red', linestyle='--')
plt.axhline(2, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["OBV"], label="OBV", color="blue")
plt.title(f"{crypto} - OBV - Volatility")
plt.xlabel("Dates")
plt.ylabel("OBV")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

def plot_categorical_label(data, col, title):
    color_map = {
    'Strong Bullish': '#006400',          # Dark Green
    'Weak Bullish': '#90ee90',            # Light Green
    'Neutral Bullish': '#228B22',         # Forest Green
    'Bullish Momentum': '#32CD32',        # Lime Green

    'Strong Bearish': '#8B0000',          # Dark Red
    'Weak Bearish': '#f08080',            # Light Coral
    'Neutral Bearish': '#DC143C',         # Crimson
    'Oversold': '#FF4500',                # Orange Red

    'Strong Sideways': '#1e90ff',         # Dodger Blue
    'Weak Sideways': '#add8e6',           # Light Blue
    'Neutral Sideways': '#4682b4',        # Steel Blue

    'Neutral': '#808080',                 # Gray

    'Overbaught': '#800080',              # Purple
    'Low': '#a0522d',                     # Sienna
    'Moderate': '#ffa500',               # Orange
    'High': '#00008b',                    # Dark Blue
    }

    plot_data = data[["dates", col]].dropna()
    plt.figure(figsize=(12, 4))
    label_levels = {label: i for i, label in enumerate(plot_data[col].unique())}
    for label, level in label_levels.items():
        label_dates = plot_data[plot_data[col] == label]["dates"]
        plt.scatter(label_dates, [level] * len(label_dates), label=label, color=color_map.get(label, "black"), s=10)
    plt.title(title)
    plt.xlabel("Dates")
    plt.yticks(list(label_levels.values()), list(label_levels.keys()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

btc_data = btc_halving_2016[btc_halving_2016["symbol"] == "BTC-USD"]
plot_categorical_label(btc_data, "market_label", "BTC-Market Trend Label")
plot_categorical_label(btc_data, "rsi_condition", "BTC-Market RSI Condition")
plot_categorical_label(btc_data, "volatality_label", "BTC-Market Volatality Label")

#BTC-USD Candlestick
btc_crypto = btc_halving_2016[btc_halving_2016["symbol"] == "BTC-USD"]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=btc_crypto["dates"], open=btc_crypto["open"], high=btc_crypto["high"],
                                                  low=btc_crypto["low"], close=btc_crypto["close"],
                                                  name="Candlesticks"))

fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_50"], mode="lines", name="SMA_50", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_200"], mode="lines", name="SMA_200", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_21"], mode="lines", name="EMA_21", line=dict(color="green")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_9"], mode="lines", name="EMA_9", line=dict(color="red")))

fib_levels = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
fib_colors = {
    "P": "black", "R1": "green", "R2": "darkgreen", "R3": "lightgreen",
    "S1": "red", "S2": "darkred", "S3": "salmon"
}

for level in fib_levels:
    fig.add_trace(go.Scatter(
        x=btc_crypto["dates"], y=btc_crypto[level],
        mode="lines", name=f"Fibonacci {level}",
        line=dict(dash="dot", color=fib_colors[level])
    ))

fig.update_layout(title="BTC-USD Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
fig.show()
    
fig, ax = plt.subplots(5, 1, figsize=(12, 8))
ax[0].plot(btc_crypto["dates"], btc_crypto["RSI_14"], label="RSI_14", color="brown")
ax[0].set_title('RSI 14 (Relative Strength Index)')
ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
ax[0].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[0].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[0].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[0].legend()

ax[1].plot(btc_crypto["dates"], btc_crypto["RSI_7"], label="RSI_7", color="orange")
ax[1].set_title('RSI 7 (Relative Strength Index)')
ax[1].axhline(70, color='red', linestyle='--', label='Overbought')
ax[1].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[1].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[1].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[1].legend()

ax[2].plot(btc_crypto["dates"], btc_crypto["OBV"], label="OBV", color="blue")
ax[2].set_title('OBV (On-Balance Volume)')

ax[3].plot(btc_crypto["dates"], btc_crypto["ATR_PCT"], label="ATR_PCT", color="green")
ax[3].axhline(5, color='red', linestyle='--', label='High')
ax[3].axhline(2, color='darkgreen', linestyle='--', label='Moderate')
ax[3].axhline(1.9, color='green', linestyle='--', label='Low')
ax[3].set_title('ATR (Average True Range)')

ax[4].plot(btc_crypto["dates"], btc_crypto["ADX_14"], label="ADX_14", color="orange")
ax[4].set_title('ADX 14 (Average Directional Strength)')
ax[4].axhline(25, color='red', linestyle='--', label="Strong")
ax[4].axhline(20, color='green', linestyle='--', label="Weak")
ax[4].legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

#plotting indicators for the time frame for NMC-USD
crypto = "NMC-USD"
data = btc_halving_2016[btc_halving_2016["symbol"] == crypto]

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.title(f"{crypto} - Closing Price")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["SMA_7"], label="SMA 7", color="green")
plt.plot(data["dates"], data["SMA_50"], label="SMA 50", color="red")
plt.plot(data["dates"], data["SMA_200"], label="SMA 200", color="orange")
plt.title(f"{crypto} - Closing Price and SMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["EMA_9"], label="EMA 9", color="green")
plt.plot(data["dates"], data["EMA_21"], label="EMA 21", color="red")
plt.title(f"{crypto} - Closing Price and EMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["RSI_7"], label="RSI 7", color="green")
plt.plot(data["dates"], data["RSI_14"], label="RSI 14", color="red")
plt.title(f"{crypto} - RSI - Market Condition")
plt.xlabel("Dates")
plt.ylabel("RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(50, color='orange', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ADX_14"], label="ADX 14", color="blue")
plt.title(f"{crypto} - ADX 14 - Trend Strength")
plt.xlabel("Dates")
plt.ylabel("ADX")
plt.axhline(25, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ATR_PCT"], label="ATR 14", color="blue")
plt.title(f"{crypto} - ATR 14 - Volatility")
plt.xlabel("Dates")
plt.ylabel("ATR Percentage")
plt.axhline(5, color='red', linestyle='--')
plt.axhline(2, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["OBV"], label="OBV", color="blue")
plt.title(f"{crypto} - OBV - Volatility")
plt.xlabel("Dates")
plt.ylabel("OBV")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

def plot_categorical_label(data, col, title):
    color_map = {
    'Strong Bullish': '#006400',          # Dark Green
    'Weak Bullish': '#90ee90',            # Light Green
    'Neutral Bullish': '#228B22',         # Forest Green
    'Bullish Momentum': '#32CD32',        # Lime Green

    'Strong Bearish': '#8B0000',          # Dark Red
    'Weak Bearish': '#f08080',            # Light Coral
    'Neutral Bearish': '#DC143C',         # Crimson
    'Oversold': '#FF4500',                # Orange Red

    'Strong Sideways': '#1e90ff',         # Dodger Blue
    'Weak Sideways': '#add8e6',           # Light Blue
    'Neutral Sideways': '#4682b4',        # Steel Blue

    'Neutral': '#808080',                 # Gray

    'Overbaught': '#800080',              # Purple
    'Low': '#a0522d',                     # Sienna
    'Moderate': '#ffa500',               # Orange
    'High': '#00008b',                    # Dark Blue
    }

    plot_data = data[["dates", col]].dropna()
    plt.figure(figsize=(12, 4))
    label_levels = {label: i for i, label in enumerate(plot_data[col].unique())}
    for label, level in label_levels.items():
        label_dates = plot_data[plot_data[col] == label]["dates"]
        plt.scatter(label_dates, [level] * len(label_dates), label=label, color=color_map.get(label, "black"), s=10)
    plt.title(title)
    plt.xlabel("Dates")
    plt.yticks(list(label_levels.values()), list(label_levels.keys()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

nmc_data = btc_halving_2016[btc_halving_2016["symbol"] == "NMC-USD"]
plot_categorical_label(nmc_data, "market_label", "NMC-Market Trend Label")
plot_categorical_label(nmc_data, "rsi_condition", "NMC-Market RSI Condition")
plot_categorical_label(nmc_data, "volatality_label", "NMC-Market Volatality Label")

#NMC-USD Candlestick
nmc_crypto = btc_halving_2016[btc_halving_2016["symbol"] == "NMC-USD"]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=nmc_crypto["dates"], open=nmc_crypto["open"], high=nmc_crypto["high"],
                                                  low=nmc_crypto["low"], close=nmc_crypto["close"],
                                                  name="Candlesticks"))

fig.add_trace(go.Scatter(x=nmc_crypto["dates"], y=nmc_crypto["SMA_50"], mode="lines", name="SMA_50", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=nmc_crypto["dates"], y=nmc_crypto["SMA_200"], mode="lines", name="SMA_200", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=nmc_crypto["dates"], y=nmc_crypto["EMA_21"], mode="lines", name="EMA_21", line=dict(color="green")))
fig.add_trace(go.Scatter(x=nmc_crypto["dates"], y=nmc_crypto["EMA_9"], mode="lines", name="EMA_9", line=dict(color="red")))

fib_levels = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
fib_colors = {
    "P": "black", "R1": "green", "R2": "darkgreen", "R3": "lightgreen",
    "S1": "red", "S2": "darkred", "S3": "salmon"
}

for level in fib_levels:
    fig.add_trace(go.Scatter(
        x=nmc_crypto["dates"], y=nmc_crypto[level],
        mode="lines", name=f"Fibonacci {level}",
        line=dict(dash="dot", color=fib_colors[level])
    ))

fig.update_layout(title="NMC-USD Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
fig.show()

fig, ax = plt.subplots(5, 1, figsize=(12, 8))
ax[0].plot(nmc_crypto["dates"], nmc_crypto["RSI_14"], label="RSI_14", color="brown")
ax[0].set_title('RSI 14 (Relative Strength Index)')
ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
ax[0].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[0].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[0].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[0].legend()

ax[1].plot(nmc_crypto["dates"], nmc_crypto["RSI_7"], label="RSI_7", color="orange")
ax[1].set_title('RSI 7 (Relative Strength Index)')
ax[1].axhline(70, color='red', linestyle='--', label='Overbought')
ax[1].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[1].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[1].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[1].legend()

ax[2].plot(nmc_crypto["dates"], nmc_crypto["OBV"], label="OBV", color="blue")
ax[2].set_title('OBV (On-Balance Volume)')

ax[3].plot(nmc_crypto["dates"], nmc_crypto["ATR_PCT"], label="ATR_PCT", color="green")
ax[3].set_title('ATR (Average True Range)')
ax[3].axhline(5, color='red', linestyle='--', label='High')
ax[3].axhline(2, color='darkgreen', linestyle='--', label='Moderate')
ax[3].axhline(1.9, color='green', linestyle='--', label='Low')

ax[4].plot(nmc_crypto["dates"], nmc_crypto["ADX_14"], label="ADX_14", color="orange")
ax[4].set_title('ADX 14 (Average Directional Strength)')
ax[4].axhline(25, color='red', linestyle='--', label="Strong")
ax[4].axhline(20, color='green', linestyle='--', label="Weak")
ax[4].legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

#plotting indicators for the time frame for PPC-USD 
crypto = "PPC-USD"
data = btc_halving_2016[btc_halving_2016["symbol"] == crypto]

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.title(f"{crypto} - Closing Price")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["SMA_7"], label="SMA 7", color="green")
plt.plot(data["dates"], data["SMA_50"], label="SMA 50", color="red")
plt.plot(data["dates"], data["SMA_200"], label="SMA 200", color="orange")
plt.title(f"{crypto} - Closing Price and SMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["EMA_9"], label="EMA 9", color="green")
plt.plot(data["dates"], data["EMA_21"], label="EMA 21", color="red")
plt.title(f"{crypto} - Closing Price and EMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["RSI_7"], label="RSI 7", color="green")
plt.plot(data["dates"], data["RSI_14"], label="RSI 14", color="red")
plt.title(f"{crypto} - RSI - Market Condition")
plt.xlabel("Dates")
plt.ylabel("RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(50, color='orange', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ADX_14"], label="ADX 14", color="blue")
plt.title(f"{crypto} - ADX 14 - Trend Strength")
plt.xlabel("Dates")
plt.ylabel("ADX")
plt.axhline(25, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ATR_PCT"], label="ATR 14", color="blue")
plt.title(f"{crypto} - ATR 14 - Volatility")
plt.xlabel("Dates")
plt.ylabel("ATR Percentage")
plt.axhline(5, color='red', linestyle='--')
plt.axhline(2, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["OBV"], label="OBV", color="blue")
plt.title(f"{crypto} - OBV - Volatility")
plt.xlabel("Dates")
plt.ylabel("OBV")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

def plot_categorical_label(data, col, title):
    color_map = {
    'Strong Bullish': '#006400',          # Dark Green
    'Weak Bullish': '#90ee90',            # Light Green
    'Neutral Bullish': '#228B22',         # Forest Green
    'Bullish Momentum': '#32CD32',        # Lime Green

    'Strong Bearish': '#8B0000',          # Dark Red
    'Weak Bearish': '#f08080',            # Light Coral
    'Neutral Bearish': '#DC143C',         # Crimson
    'Oversold': '#FF4500',                # Orange Red

    'Strong Sideways': '#1e90ff',         # Dodger Blue
    'Weak Sideways': '#add8e6',           # Light Blue
    'Neutral Sideways': '#4682b4',        # Steel Blue

    'Neutral': '#808080',                 # Gray

    'Overbaught': '#800080',              # Purple
    'Low': '#a0522d',                     # Sienna
    'Moderate': '#ffa500',               # Orange
    'High': '#00008b',                    # Dark Blue
    }

    plot_data = data[["dates", col]].dropna()
    plt.figure(figsize=(12, 4))
    label_levels = {label: i for i, label in enumerate(plot_data[col].unique())}
    for label, level in label_levels.items():
        label_dates = plot_data[plot_data[col] == label]["dates"]
        plt.scatter(label_dates, [level] * len(label_dates), label=label, color=color_map.get(label, "black"), s=10)
    plt.title(title)
    plt.xlabel("Dates")
    plt.yticks(list(label_levels.values()), list(label_levels.keys()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

ppc_data = btc_halving_2016[btc_halving_2016["symbol"] == "PPC-USD"]
plot_categorical_label(ppc_data, "market_label", "PXC-Market Trend Label")
plot_categorical_label(ppc_data, "rsi_condition", "PXC-Market RSI Condition")
plot_categorical_label(ppc_data, "volatality_label", "PXC-Market Volatality Label")

#PPC-USD Candlestick
ppc_crypto = btc_halving_2016[btc_halving_2016["symbol"] == "PPC-USD"]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=ppc_crypto["dates"], open=ppc_crypto["open"], high=ppc_crypto["high"],
                                                  low=ppc_crypto["low"], close=ppc_crypto["close"],
                                                  name="Candlesticks"))

fig.add_trace(go.Scatter(x=ppc_crypto["dates"], y=ppc_crypto["SMA_50"], mode="lines", name="SMA_50", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=ppc_crypto["dates"], y=ppc_crypto["SMA_200"], mode="lines", name="SMA_200", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=ppc_crypto["dates"], y=ppc_crypto["EMA_21"], mode="lines", name="EMA_21", line=dict(color="green")))
fig.add_trace(go.Scatter(x=ppc_crypto["dates"], y=ppc_crypto["EMA_9"], mode="lines", name="EMA_9", line=dict(color="red")))

fib_levels = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
fib_colors = {
    "P": "black", "R1": "green", "R2": "darkgreen", "R3": "lightgreen",
    "S1": "red", "S2": "darkred", "S3": "salmon"
}

for level in fib_levels:
    fig.add_trace(go.Scatter(
        x=ppc_crypto["dates"], y=ppc_crypto[level],
        mode="lines", name=f"Fibonacci {level}",
        line=dict(dash="dot", color=fib_colors[level])
    ))

fig.update_layout(title="PPC-USD Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
fig.show()

fig, ax = plt.subplots(5, 1, figsize=(12, 8))
ax[0].plot(ppc_crypto["dates"], ppc_crypto["RSI_14"], label="RSI_14", color="brown")
ax[0].set_title('RSI 14 (Relative Strength Index)')
ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
ax[0].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[0].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[0].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[0].legend()

ax[1].plot(ppc_crypto["dates"], ppc_crypto["RSI_7"], label="RSI_7", color="orange")
ax[1].set_title('RSI 7 (Relative Strength Index)')
ax[1].axhline(70, color='red', linestyle='--', label='Overbought')
ax[1].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[1].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[1].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[1].legend()

ax[2].plot(ppc_crypto["dates"], ppc_crypto["OBV"], label="OBV", color="blue")
ax[2].set_title('OBV (On-Balance Volume)')

ax[3].plot(ppc_crypto["dates"], ppc_crypto["ATR_PCT"], label="ATR_PCT", color="green")
ax[3].set_title('ATR (Average True Range)')
ax[3].axhline(5, color='red', linestyle='--', label='High')
ax[3].axhline(2, color='darkgreen', linestyle='--', label='Moderate')
ax[3].axhline(1.9, color='green', linestyle='--', label='Low')

ax[4].plot(ppc_crypto["dates"], ppc_crypto["ADX_14"], label="ADX_14", color="orange")
ax[4].set_title('ADX 14 (Average Directional Strength)')
ax[4].axhline(25, color='red', linestyle='--', label="Strong")
ax[4].axhline(20, color='green', linestyle='--', label="Weak")
ax[4].legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

#2. Initial Coin Offering (ICO) Boom and Bitcoin Bull Run (Early 2017 to Early 2018) - Dataframe: ico_boom_2017 - Duration: Jan 2017 – Jan 2018
ico_boom_2017 = crypto_prices_final[(crypto_prices_final["month"] >= "2017-01") & (crypto_prices_final["month"] <= "2018-01")]
cols_needed = [
    "dates", "rolling_pct_change", "ATR_PCT", "ATR_14", "S3", "S2", "S1", "R3", "R2", "R1", "P",
    "ADX_14", "RSI_14", "RSI_7", "EMA_21", "EMA_9", "SMA_200", "SMA_50", "SMA_7"
]
ico_boom_2017 = ico_boom_2017.dropna(subset=cols_needed).copy()
print(ico_boom_2017[cols_needed].isna().sum())
print("ICO Boom 2017 DataFrame 2\n")
print(ico_boom_2017.reset_index(drop=True))

#Plotting Correlation of BTC-USD with other cryptos for ico_boom_2017
crypto_prices_pivot = ico_boom_2017.pivot(index='dates', columns='symbol', values='close')
cryptos = ['NMC-USD', 'PPC-USD', 'LTC-USD', 'FTC-USD', 'PXC-USD']
BTC_Symbol = "BTC-USD"

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 15), sharex=True)
for i, crypto in enumerate(cryptos):
    rolling_corr = crypto_prices_pivot[crypto].rolling(window=30).corr(crypto_prices_pivot[BTC_Symbol])
    axes[i].plot(crypto_prices_pivot.index, rolling_corr, label=f"{crypto} vs BTC", color='blue')
    axes[i].axhline(0.7, color='red', linestyle='dashed')
    axes[i].axhline(0.5, color='blue', linestyle='dashed')
    axes[i].axhline(0, color='black', linestyle='dashed')
    axes[i].axhline(-0.5, color='blue', linestyle='dashed')
    axes[i].axhline(-0.7, color='red', linestyle='dashed')
    axes[i].set_ylabel("Correlation")
    axes[i].legend()
axes[-1].set_xlabel("Date")
plt.title(f"30 days Rolling Correlation: BTC and Cryptos for ICO Boom 2017")
plt.tight_layout()
plt.show()

sns.heatmap(crypto_prices_pivot.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corr Heatmap between BTC and Cryptos for ICO Boom 2017")
plt.show()

#results suggests high correlation of all cryptos with BTC, BTC market movements affcts highlt on other cryptos.
#Hence we can make interpretation of movements and trends of other cryptos using BTC with confidence
#for this period we will only take BTC and apply the insights and results to all other cryptos

#plotting indicators for the time frame for BTC-USD
crypto = "BTC-USD"
data = ico_boom_2017[ico_boom_2017["symbol"] == crypto]

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.title(f"{crypto} - Closing Price")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["SMA_7"], label="SMA 7", color="green")
plt.plot(data["dates"], data["SMA_50"], label="SMA 50", color="red")
plt.plot(data["dates"], data["SMA_200"], label="SMA 200", color="orange")
plt.title(f"{crypto} - Closing Price and SMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["EMA_9"], label="EMA 9", color="green")
plt.plot(data["dates"], data["EMA_21"], label="EMA 21", color="red")
plt.title(f"{crypto} - Closing Price and EMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["RSI_7"], label="RSI 7", color="green")
plt.plot(data["dates"], data["RSI_14"], label="RSI 14", color="red")
plt.title(f"{crypto} - RSI - Market Condition")
plt.xlabel("Dates")
plt.ylabel("RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(50, color='orange', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ADX_14"], label="ADX 14", color="blue")
plt.title(f"{crypto} - ADX 14 - Trend Strength")
plt.xlabel("Dates")
plt.ylabel("ADX")
plt.axhline(25, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ATR_PCT"], label="ATR 14", color="blue")
plt.title(f"{crypto} - ATR 14 - Volatility")
plt.xlabel("Dates")
plt.ylabel("ATR Percentage")
plt.axhline(5, color='red', linestyle='--')
plt.axhline(2, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["OBV"], label="OBV", color="blue")
plt.title(f"{crypto} - OBV - Volatility")
plt.xlabel("Dates")
plt.ylabel("OBV")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

def plot_categorical_label(data, col, title):
    color_map = {
    'Strong Bullish': '#006400',          # Dark Green
    'Weak Bullish': '#90ee90',            # Light Green
    'Neutral Bullish': '#228B22',         # Forest Green
    'Bullish Momentum': '#32CD32',        # Lime Green

    'Strong Bearish': '#8B0000',          # Dark Red
    'Weak Bearish': '#f08080',            # Light Coral
    'Neutral Bearish': '#DC143C',         # Crimson
    'Oversold': '#FF4500',                # Orange Red

    'Strong Sideways': '#1e90ff',         # Dodger Blue
    'Weak Sideways': '#add8e6',           # Light Blue
    'Neutral Sideways': '#4682b4',        # Steel Blue

    'Neutral': '#808080',                 # Gray

    'Overbaught': '#800080',              # Purple
    'Low': '#a0522d',                     # Sienna
    'Moderate': '#ffa500',               # Orange
    'High': '#00008b',                    # Dark Blue
    }

    plot_data = data[["dates", col]].dropna()
    plt.figure(figsize=(12, 4))
    label_levels = {label: i for i, label in enumerate(plot_data[col].unique())}
    for label, level in label_levels.items():
        label_dates = plot_data[plot_data[col] == label]["dates"]
        plt.scatter(label_dates, [level] * len(label_dates), label=label, color=color_map.get(label, "black"), s=10)
    plt.title(title)
    plt.xlabel("Dates")
    plt.yticks(list(label_levels.values()), list(label_levels.keys()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

btc_data = ico_boom_2017[ico_boom_2017["symbol"] == "BTC-USD"]
plot_categorical_label(btc_data, "market_label", "BTC-Market Trend Label")
plot_categorical_label(btc_data, "rsi_condition", "BTC-Market RSI Condition")
plot_categorical_label(btc_data, "volatality_label", "BTC-Market Volatality Label")

#BTC-USD Candlestick
btc_crypto = ico_boom_2017[ico_boom_2017["symbol"] == "BTC-USD"]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=btc_crypto["dates"], open=btc_crypto["open"], high=btc_crypto["high"],
                                                  low=btc_crypto["low"], close=btc_crypto["close"],
                                                  name="Candlesticks"))

fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_50"], mode="lines", name="SMA_50", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_200"], mode="lines", name="SMA_200", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_21"], mode="lines", name="EMA_21", line=dict(color="green")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_9"], mode="lines", name="EMA_9", line=dict(color="red")))

fib_levels = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
fib_colors = {
    "P": "black", "R1": "green", "R2": "darkgreen", "R3": "lightgreen",
    "S1": "red", "S2": "darkred", "S3": "salmon"
}

for level in fib_levels:
    fig.add_trace(go.Scatter(
        x=btc_crypto["dates"], y=btc_crypto[level],
        mode="lines", name=f"Fibonacci {level}",
        line=dict(dash="dot", color=fib_colors[level])
    ))

fig.update_layout(title="BTC-USD Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
fig.show()

fig, ax = plt.subplots(5, 1, figsize=(12, 8))
ax[0].plot(btc_crypto["dates"], btc_crypto["RSI_14"], label="RSI_14", color="brown")
ax[0].set_title('RSI 14 (Relative Strength Index)')
ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
ax[0].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[0].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[0].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[0].legend()

ax[1].plot(btc_crypto["dates"], btc_crypto["RSI_7"], label="RSI_7", color="orange")
ax[1].set_title('RSI 7 (Relative Strength Index)')
ax[1].axhline(70, color='red', linestyle='--', label='Overbought')
ax[1].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[1].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[1].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[1].legend()

ax[2].plot(btc_crypto["dates"], btc_crypto["OBV"], label="OBV", color="blue")
ax[2].set_title('OBV (On-Balance Volume)')

ax[3].plot(btc_crypto["dates"], btc_crypto["ATR_PCT"], label="ATR_PCT", color="green")
ax[3].set_title('ATR (Average True Range)')
ax[3].axhline(5, color='red', linestyle='--', label='High')
ax[3].axhline(2, color='darkgreen', linestyle='--', label='Moderate')
ax[3].axhline(1.9, color='green', linestyle='--', label='Low')

ax[4].plot(btc_crypto["dates"], btc_crypto["ADX_14"], label="ADX_14", color="orange")
ax[4].set_title('ADX 14 (Average Directional Strength)')
ax[4].axhline(25, color='red', linestyle='--', label="Strong")
ax[4].axhline(20, color='green', linestyle='--', label="Weak")
ax[4].legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

#3. Post-bubble collapse (the Great crypto crash) (After Jan 2018) - Dataframe: crypto_crash_2018 - Duration: Feb 2018 – Dec 2018
crypto_crash_2018 = crypto_prices_final[(crypto_prices_final["month"] >= "2018-02") & (crypto_prices_final["month"] <= "2018-12")]
cols_needed = [
    "dates", "rolling_pct_change", "ATR_PCT", "ATR_14", "S3", "S2", "S1", "R3", "R2", "R1", "P",
    "ADX_14", "RSI_14", "RSI_7", "EMA_21", "EMA_9", "SMA_200", "SMA_50", "SMA_7"
]
crypto_crash_2018 = crypto_crash_2018.dropna(subset=cols_needed).copy()
print(crypto_crash_2018[cols_needed].isna().sum())
print("The Great Crypto Crash 2018 DataFrame 3\n")
print(crypto_crash_2018)

#Plotting Correlation of BTC-USD with other cryptos for crypto_crash_2018
crypto_prices_pivot = crypto_crash_2018.pivot(index='dates', columns='symbol', values='close')
cryptos = ['NMC-USD', 'PPC-USD', 'LTC-USD', 'FTC-USD', 'PXC-USD']
BTC_Symbol = "BTC-USD"

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 15), sharex=True)
for i, crypto in enumerate(cryptos):
    rolling_corr = crypto_prices_pivot[crypto].rolling(window=30).corr(crypto_prices_pivot[BTC_Symbol])
    axes[i].plot(crypto_prices_pivot.index, rolling_corr, label=f"{crypto} vs BTC", color='blue')
    axes[i].axhline(0.7, color='red', linestyle='dashed')
    axes[i].axhline(0.5, color='blue', linestyle='dashed')
    axes[i].axhline(0, color='black', linestyle='dashed')
    axes[i].axhline(-0.5, color='blue', linestyle='dashed')
    axes[i].axhline(-0.7, color='red', linestyle='dashed')
    axes[i].set_ylabel("Correlation")
    axes[i].legend()
axes[-1].set_xlabel("Date")
plt.title(f"30 days Rolling Correlation: BTC and Cryptos for Crypto Crash 2018")
plt.tight_layout()
plt.show()

sns.heatmap(crypto_prices_pivot.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corr Heatmap between BTC and Cryptos for Crypto Crash 2018")
plt.show()

#results suggests high correlation of all cryptos with BTC, BTC market movements affcts highlt on other cryptos.
#Hence we can make interpretation of movements and trends of other cryptos using BTC with confidence
#for this period we will only take BTC and apply the insights and results to all other cryptos

#plotting indicators for the time frame for BTC-USD
crypto = "BTC-USD"
data = crypto_crash_2018[crypto_crash_2018["symbol"] == crypto]

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.title(f"{crypto} - Closing Price")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["SMA_7"], label="SMA 7", color="green")
plt.plot(data["dates"], data["SMA_50"], label="SMA 50", color="red")
plt.plot(data["dates"], data["SMA_200"], label="SMA 200", color="orange")
plt.title(f"{crypto} - Closing Price and SMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["EMA_9"], label="EMA 9", color="green")
plt.plot(data["dates"], data["EMA_21"], label="EMA 21", color="red")
plt.title(f"{crypto} - Closing Price and EMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["RSI_7"], label="RSI 7", color="green")
plt.plot(data["dates"], data["RSI_14"], label="RSI 14", color="red")
plt.title(f"{crypto} - RSI - Market Condition")
plt.xlabel("Dates")
plt.ylabel("RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(50, color='orange', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ADX_14"], label="ADX 14", color="blue")
plt.title(f"{crypto} - ADX 14 - Trend Strength")
plt.xlabel("Dates")
plt.ylabel("ADX")
plt.axhline(25, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ATR_PCT"], label="ATR 14", color="blue")
plt.title(f"{crypto} - ATR 14 - Volatility")
plt.xlabel("Dates")
plt.ylabel("ATR Percentage")
plt.axhline(5, color='red', linestyle='--')
plt.axhline(2, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["OBV"], label="OBV", color="blue")
plt.title(f"{crypto} - OBV - Volatility")
plt.xlabel("Dates")
plt.ylabel("OBV")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

def plot_categorical_label(data, col, title):
    color_map = {
    'Strong Bullish': '#006400',          # Dark Green
    'Weak Bullish': '#90ee90',            # Light Green
    'Neutral Bullish': '#228B22',         # Forest Green
    'Bullish Momentum': '#32CD32',        # Lime Green

    'Strong Bearish': '#8B0000',          # Dark Red
    'Weak Bearish': '#f08080',            # Light Coral
    'Neutral Bearish': '#DC143C',         # Crimson
    'Oversold': '#FF4500',                # Orange Red

    'Strong Sideways': '#1e90ff',         # Dodger Blue
    'Weak Sideways': '#add8e6',           # Light Blue
    'Neutral Sideways': '#4682b4',        # Steel Blue

    'Neutral': '#808080',                 # Gray

    'Overbaught': '#800080',              # Purple
    'Low': '#a0522d',                     # Sienna
    'Moderate': '#ffa500',               # Orange
    'High': '#00008b',                    # Dark Blue
    }

    plot_data = data[["dates", col]].dropna()
    plt.figure(figsize=(12, 4))
    label_levels = {label: i for i, label in enumerate(plot_data[col].unique())}
    for label, level in label_levels.items():
        label_dates = plot_data[plot_data[col] == label]["dates"]
        plt.scatter(label_dates, [level] * len(label_dates), label=label, color=color_map.get(label, "black"), s=10)
    plt.title(title)
    plt.xlabel("Dates")
    plt.yticks(list(label_levels.values()), list(label_levels.keys()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

btc_data = crypto_crash_2018[crypto_crash_2018["symbol"] == "BTC-USD"]
plot_categorical_label(btc_data, "market_label", "BTC-Market Trend Label")
plot_categorical_label(btc_data, "rsi_condition", "BTC-Market RSI Condition")
plot_categorical_label(btc_data, "volatality_label", "BTC-Market Volatality Label")

#BTC-USD Candlestick
btc_crypto = crypto_crash_2018[crypto_crash_2018["symbol"] == "BTC-USD"]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=btc_crypto["dates"], open=btc_crypto["open"], high=btc_crypto["high"],
                                                  low=btc_crypto["low"], close=btc_crypto["close"],
                                                  name="Candlesticks"))

fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_50"], mode="lines", name="SMA_50", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_200"], mode="lines", name="SMA_200", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_21"], mode="lines", name="EMA_21", line=dict(color="green")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_9"], mode="lines", name="EMA_9", line=dict(color="red")))

fib_levels = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
fib_colors = {
    "P": "black", "R1": "green", "R2": "darkgreen", "R3": "lightgreen",
    "S1": "red", "S2": "darkred", "S3": "salmon"
}

for level in fib_levels:
    fig.add_trace(go.Scatter(
        x=btc_crypto["dates"], y=btc_crypto[level],
        mode="lines", name=f"Fibonacci {level}",
        line=dict(dash="dot", color=fib_colors[level])
    ))

fig.update_layout(title="BTC-USD Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
fig.show()

fig, ax = plt.subplots(5, 1, figsize=(12, 8))
ax[0].plot(btc_crypto["dates"], btc_crypto["RSI_14"], label="RSI_14", color="brown")
ax[0].set_title('RSI 14 (Relative Strength Index)')
ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
ax[0].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[0].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[0].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[0].legend()

ax[1].plot(btc_crypto["dates"], btc_crypto["RSI_7"], label="RSI_7", color="orange")
ax[1].set_title('RSI 7 (Relative Strength Index)')
ax[1].axhline(70, color='red', linestyle='--', label='Overbought')
ax[1].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[1].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[1].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[1].legend()

ax[2].plot(btc_crypto["dates"], btc_crypto["OBV"], label="OBV", color="blue")
ax[2].set_title('OBV (On-Balance Volume)')

ax[3].plot(btc_crypto["dates"], btc_crypto["ATR_PCT"], label="ATR_PCT", color="green")
ax[3].set_title('ATR (Average True Range)')
ax[3].axhline(5, color='red', linestyle='--', label='High')
ax[3].axhline(2, color='darkgreen', linestyle='--', label='Moderate')
ax[3].axhline(1.9, color='green', linestyle='--', label='Low')

ax[4].plot(btc_crypto["dates"], btc_crypto["ADX_14"], label="ADX_14", color="orange")
ax[4].set_title('ADX 14 (Average Directional Strength)')
ax[4].axhline(25, color='red', linestyle='--', label="Strong")
ax[4].axhline(20, color='green', linestyle='--', label="Weak")
ax[4].legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

#almost stable and gradual increasing market from Jan 2019 to Mar 2020

#4. COVID-19 Bull Run, Bitcoin Halving of 2020 (After Mar 2020) - Dataframe: covid_run_2020 - Duration: Apr 2020 – May 2021
covid_run_2020 = crypto_prices_final[(crypto_prices_final["month"] >= "2020-04") & (crypto_prices_final["month"] <= "2021-05")]
cols_needed = [
    "dates", "rolling_pct_change", "ATR_PCT", "ATR_14", "S3", "S2", "S1", "R3", "R2", "R1", "P",
    "ADX_14", "RSI_14", "RSI_7", "EMA_21", "EMA_9", "SMA_200", "SMA_50", "SMA_7"
]
covid_run_2020 = covid_run_2020.dropna(subset=cols_needed).copy()
print(covid_run_2020[cols_needed].isna().sum())
print("Covid Bull Run 2020 DataFrame 4\n")
print(covid_run_2020.reset_index(drop=True))

#Plotting Correlation of BTC-USD with other cryptos for covid_run_2020
crypto_prices_pivot = covid_run_2020.pivot(index='dates', columns='symbol', values='close')
cryptos = ['NMC-USD', 'PPC-USD', 'LTC-USD', 'FTC-USD', 'PXC-USD']
BTC_Symbol = "BTC-USD"

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 15), sharex=True)
for i, crypto in enumerate(cryptos):
    rolling_corr = crypto_prices_pivot[crypto].rolling(window=30).corr(crypto_prices_pivot[BTC_Symbol])
    axes[i].plot(crypto_prices_pivot.index, rolling_corr, label=f"{crypto} vs BTC", color='blue')
    axes[i].axhline(0.7, color='red', linestyle='dashed')
    axes[i].axhline(0.5, color='blue', linestyle='dashed')
    axes[i].axhline(0, color='black', linestyle='dashed')
    axes[i].axhline(-0.5, color='blue', linestyle='dashed')
    axes[i].axhline(-0.7, color='red', linestyle='dashed')
    axes[i].set_ylabel("Correlation")
    axes[i].legend()
axes[-1].set_xlabel("Date")
plt.title(f"30 days Rolling Correlation: BTC and Cryptos for COVID-19 Bull Run 2020")
plt.tight_layout()
plt.show()

sns.heatmap(crypto_prices_pivot.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corr Heatmap between BTC and Cryptos for COVID-19 Bull Run 2020")
plt.show()

#results suggests high correlation of most of cryptos with BTC, there does not seems to have any affect of BTC market movements on other cryptos.
#PPC is the only coin with low (0.61) correlation with BTC. Hence we can somewhat make interpretation of movements and trends of PPC using BTC but cannot have strong confidence in results
#so for this period we will choose BTC, PPC as they are least correlated and apply the results and insights of BTC with other cryptos

#plotting indicators for the time frame for BTC-USD
crypto = "BTC-USD"
data = covid_run_2020[covid_run_2020["symbol"] == crypto]

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.title(f"{crypto} - Closing Price")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["SMA_7"], label="SMA 7", color="green")
plt.plot(data["dates"], data["SMA_50"], label="SMA 50", color="red")
plt.plot(data["dates"], data["SMA_200"], label="SMA 200", color="orange")
plt.title(f"{crypto} - Closing Price and SMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["EMA_9"], label="EMA 9", color="green")
plt.plot(data["dates"], data["EMA_21"], label="EMA 21", color="red")
plt.title(f"{crypto} - Closing Price and EMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["RSI_7"], label="RSI 7", color="green")
plt.plot(data["dates"], data["RSI_14"], label="RSI 14", color="red")
plt.title(f"{crypto} - RSI - Market Condition")
plt.xlabel("Dates")
plt.ylabel("RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(50, color='orange', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ADX_14"], label="ADX 14", color="blue")
plt.title(f"{crypto} - ADX 14 - Trend Strength")
plt.xlabel("Dates")
plt.ylabel("ADX")
plt.axhline(25, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ATR_PCT"], label="ATR 14", color="blue")
plt.title(f"{crypto} - ATR 14 - Volatility")
plt.xlabel("Dates")
plt.ylabel("ATR Percentage")
plt.axhline(5, color='red', linestyle='--')
plt.axhline(2, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["OBV"], label="OBV", color="blue")
plt.title(f"{crypto} - OBV - Volatility")
plt.xlabel("Dates")
plt.ylabel("OBV")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

def plot_categorical_label(data, col, title):
    color_map = {
    'Strong Bullish': '#006400',          # Dark Green
    'Weak Bullish': '#90ee90',            # Light Green
    'Neutral Bullish': '#228B22',         # Forest Green
    'Bullish Momentum': '#32CD32',        # Lime Green

    'Strong Bearish': '#8B0000',          # Dark Red
    'Weak Bearish': '#f08080',            # Light Coral
    'Neutral Bearish': '#DC143C',         # Crimson
    'Oversold': '#FF4500',                # Orange Red

    'Strong Sideways': '#1e90ff',         # Dodger Blue
    'Weak Sideways': '#add8e6',           # Light Blue
    'Neutral Sideways': '#4682b4',        # Steel Blue

    'Neutral': '#808080',                 # Gray

    'Overbaught': '#800080',              # Purple
    'Low': '#a0522d',                     # Sienna
    'Moderate': '#ffa500',               # Orange
    'High': '#00008b',                    # Dark Blue
    }

    plot_data = data[["dates", col]].dropna()
    plt.figure(figsize=(12, 4))
    label_levels = {label: i for i, label in enumerate(plot_data[col].unique())}
    for label, level in label_levels.items():
        label_dates = plot_data[plot_data[col] == label]["dates"]
        plt.scatter(label_dates, [level] * len(label_dates), label=label, color=color_map.get(label, "black"), s=10)
    plt.title(title)
    plt.xlabel("Dates")
    plt.yticks(list(label_levels.values()), list(label_levels.keys()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

btc_data = covid_run_2020[covid_run_2020["symbol"] == "BTC-USD"]
plot_categorical_label(btc_data, "market_label", "BTC-Market Trend Label")
plot_categorical_label(btc_data, "rsi_condition", "BTC-Market RSI Condition")
plot_categorical_label(btc_data, "volatality_label", "BTC-Market Volatality Label")

#BTC-USD Candlestick
btc_crypto = covid_run_2020[covid_run_2020["symbol"] == "BTC-USD"]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=btc_crypto["dates"], open=btc_crypto["open"], high=btc_crypto["high"],
                                                  low=btc_crypto["low"], close=btc_crypto["close"],
                                                  name="Candlesticks"))

fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_50"], mode="lines", name="SMA_50", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_200"], mode="lines", name="SMA_200", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_21"], mode="lines", name="EMA_21", line=dict(color="green")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_9"], mode="lines", name="EMA_9", line=dict(color="red")))

fib_levels = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
fib_colors = {
    "P": "black", "R1": "green", "R2": "darkgreen", "R3": "lightgreen",
    "S1": "red", "S2": "darkred", "S3": "salmon"
}

for level in fib_levels:
    fig.add_trace(go.Scatter(
        x=btc_crypto["dates"], y=btc_crypto[level],
        mode="lines", name=f"Fibonacci {level}",
        line=dict(dash="dot", color=fib_colors[level])
    ))

fig.update_layout(title="BTC-USD Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
fig.show()

fig, ax = plt.subplots(5, 1, figsize=(12, 8))
ax[0].plot(btc_crypto["dates"], btc_crypto["RSI_14"], label="RSI_14", color="brown")
ax[0].set_title('RSI 14 (Relative Strength Index)')
ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
ax[0].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[0].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[0].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[0].legend()

ax[1].plot(btc_crypto["dates"], btc_crypto["RSI_7"], label="RSI_7", color="orange")
ax[1].set_title('RSI 7 (Relative Strength Index)')
ax[1].axhline(70, color='red', linestyle='--', label='Overbought')
ax[1].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[1].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[1].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[1].legend()

ax[2].plot(btc_crypto["dates"], btc_crypto["OBV"], label="OBV", color="blue")
ax[2].set_title('OBV (On-Balance Volume)')

ax[3].plot(btc_crypto["dates"], btc_crypto["ATR_PCT"], label="ATR_PCT", color="green")
ax[3].set_title('ATR (Average True Range)')
ax[3].axhline(5, color='red', linestyle='--', label='High')
ax[3].axhline(2, color='darkgreen', linestyle='--', label='Moderate')
ax[3].axhline(1.9, color='green', linestyle='--', label='Low')

ax[4].plot(btc_crypto["dates"], btc_crypto["ADX_14"], label="ADX_14", color="orange")
ax[4].set_title('ADX 14 (Average Directional Strength)')
ax[4].axhline(25, color='red', linestyle='--', label="Strong")
ax[4].axhline(20, color='green', linestyle='--', label="Weak")
ax[4].legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

#plotting indicators for the time frame for PPC-USD
crypto = "PPC-USD"
data = covid_run_2020[covid_run_2020["symbol"] == crypto]

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.title(f"{crypto} - Closing Price")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["SMA_7"], label="SMA 7", color="green")
plt.plot(data["dates"], data["SMA_50"], label="SMA 50", color="red")
plt.plot(data["dates"], data["SMA_200"], label="SMA 200", color="orange")
plt.title(f"{crypto} - Closing Price and SMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["EMA_9"], label="EMA 9", color="green")
plt.plot(data["dates"], data["EMA_21"], label="EMA 21", color="red")
plt.title(f"{crypto} - Closing Price and EMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["RSI_7"], label="RSI 7", color="green")
plt.plot(data["dates"], data["RSI_14"], label="RSI 14", color="red")
plt.title(f"{crypto} - RSI - Market Condition")
plt.xlabel("Dates")
plt.ylabel("RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(50, color='orange', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ADX_14"], label="ADX 14", color="blue")
plt.title(f"{crypto} - ADX 14 - Trend Strength")
plt.xlabel("Dates")
plt.ylabel("ADX")
plt.axhline(25, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ATR_PCT"], label="ATR 14", color="blue")
plt.title(f"{crypto} - ATR 14 - Volatility")
plt.xlabel("Dates")
plt.ylabel("ATR Percentage")
plt.axhline(5, color='red', linestyle='--')
plt.axhline(2, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["OBV"], label="OBV", color="blue")
plt.title(f"{crypto} - OBV - Volatility")
plt.xlabel("Dates")
plt.ylabel("OBV")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

def plot_categorical_label(data, col, title):
    color_map = {
    'Strong Bullish': '#006400',          # Dark Green
    'Weak Bullish': '#90ee90',            # Light Green
    'Neutral Bullish': '#228B22',         # Forest Green
    'Bullish Momentum': '#32CD32',        # Lime Green

    'Strong Bearish': '#8B0000',          # Dark Red
    'Weak Bearish': '#f08080',            # Light Coral
    'Neutral Bearish': '#DC143C',         # Crimson
    'Oversold': '#FF4500',                # Orange Red

    'Strong Sideways': '#1e90ff',         # Dodger Blue
    'Weak Sideways': '#add8e6',           # Light Blue
    'Neutral Sideways': '#4682b4',        # Steel Blue

    'Neutral': '#808080',                 # Gray

    'Overbaught': '#800080',              # Purple
    'Low': '#a0522d',                     # Sienna
    'Moderate': '#ffa500',               # Orange
    'High': '#00008b',                    # Dark Blue
    }

    plot_data = data[["dates", col]].dropna()
    plt.figure(figsize=(12, 4))
    label_levels = {label: i for i, label in enumerate(plot_data[col].unique())}
    for label, level in label_levels.items():
        label_dates = plot_data[plot_data[col] == label]["dates"]
        plt.scatter(label_dates, [level] * len(label_dates), label=label, color=color_map.get(label, "black"), s=10)
    plt.title(title)
    plt.xlabel("Dates")
    plt.yticks(list(label_levels.values()), list(label_levels.keys()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

ppc_data = covid_run_2020[covid_run_2020["symbol"] == "PPC-USD"]
plot_categorical_label(ppc_data, "market_label", "PPC-Market Trend Label")
plot_categorical_label(ppc_data, "rsi_condition", "PPC-Market RSI Condition")
plot_categorical_label(ppc_data, "volatality_label", "PPC-Market Volatality Label")

#PPC-USD Candlestick
ppc_crypto = covid_run_2020[covid_run_2020["symbol"] == "PPC-USD"]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=ppc_crypto["dates"], open=ppc_crypto["open"], high=ppc_crypto["high"],
                                                  low=ppc_crypto["low"], close=ppc_crypto["close"],
                                                  name="Candlesticks"))

fig.add_trace(go.Scatter(x=ppc_crypto["dates"], y=ppc_crypto["SMA_50"], mode="lines", name="SMA_50", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=ppc_crypto["dates"], y=ppc_crypto["SMA_200"], mode="lines", name="SMA_200", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=ppc_crypto["dates"], y=ppc_crypto["EMA_21"], mode="lines", name="EMA_21", line=dict(color="green")))
fig.add_trace(go.Scatter(x=ppc_crypto["dates"], y=ppc_crypto["EMA_9"], mode="lines", name="EMA_9", line=dict(color="red")))

fib_levels = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
fib_colors = {
    "P": "black", "R1": "green", "R2": "darkgreen", "R3": "lightgreen",
    "S1": "red", "S2": "darkred", "S3": "salmon"
}

for level in fib_levels:
    fig.add_trace(go.Scatter(
        x=ppc_crypto["dates"], y=ppc_crypto[level],
        mode="lines", name=f"Fibonacci {level}",
        line=dict(dash="dot", color=fib_colors[level])
    ))

fig.update_layout(title="PPC-USD Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
fig.show()

fig, ax = plt.subplots(5, 1, figsize=(12, 8))
ax[0].plot(ppc_crypto["dates"], ppc_crypto["RSI_14"], label="RSI_14", color="brown")
ax[0].set_title('RSI 14 (Relative Strength Index)')
ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
ax[0].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[0].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[0].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[0].legend()

ax[1].plot(ppc_crypto["dates"], ppc_crypto["RSI_7"], label="RSI_7", color="orange")
ax[1].set_title('RSI 7 (Relative Strength Index)')
ax[1].axhline(70, color='red', linestyle='--', label='Overbought')
ax[1].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[1].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[1].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[1].legend()

ax[2].plot(ppc_crypto["dates"], ppc_crypto["OBV"], label="OBV", color="blue")
ax[2].set_title('OBV (On-Balance Volume)')

ax[3].plot(ppc_crypto["dates"], ppc_crypto["ATR_PCT"], label="ATR_PCT", color="green")
ax[3].set_title('ATR (Average True Range)')
ax[3].axhline(5, color='red', linestyle='--', label='High')
ax[3].axhline(2, color='darkgreen', linestyle='--', label='Moderate')
ax[3].axhline(1.9, color='green', linestyle='--', label='Low')

ax[4].plot(ppc_crypto["dates"], ppc_crypto["ADX_14"], label="ADX_14", color="orange")
ax[4].set_title('ADX 14 (Average Directional Strength)')
ax[4].axhline(25, color='red', linestyle='--', label="Strong")
ax[4].axhline(20, color='green', linestyle='--', label="Weak")
ax[4].legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

#5. SEC regulations, Terra/FTX Crash - Dataframe: crypto_crash_2021 - Duration: Jun 2021 – Dec 2022
crypto_crash_2021 = crypto_prices_final[(crypto_prices_final["month"] >= "2021-06") & (crypto_prices_final["month"] <= "2022-12")]
cols_needed = [
    "dates", "rolling_pct_change", "ATR_PCT", "ATR_14", "S3", "S2", "S1", "R3", "R2", "R1", "P",
    "ADX_14", "RSI_14", "RSI_7", "EMA_21", "EMA_9", "SMA_200", "SMA_50", "SMA_7"
]
crypto_crash_2021 = crypto_crash_2021.dropna(subset=cols_needed).copy()
print(crypto_crash_2021[cols_needed].isna().sum())
print("Crypto Crash 2021 DataFrame 5\n")
print(crypto_crash_2021.reset_index(drop=True))

#Plotting Correlation of BTC-USD with other cryptos for crypto_crash_2021
crypto_prices_pivot = crypto_crash_2021.pivot(index='dates', columns='symbol', values='close')
cryptos = ['NMC-USD', 'PPC-USD', 'LTC-USD', 'FTC-USD', 'PXC-USD']
BTC_Symbol = "BTC-USD"

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 15), sharex=True)
for i, crypto in enumerate(cryptos):
    rolling_corr = crypto_prices_pivot[crypto].rolling(window=30).corr(crypto_prices_pivot[BTC_Symbol])
    axes[i].plot(crypto_prices_pivot.index, rolling_corr, label=f"{crypto} vs BTC", color='blue')
    axes[i].axhline(0.7, color='red', linestyle='dashed')
    axes[i].axhline(0.5, color='blue', linestyle='dashed')
    axes[i].axhline(0, color='black', linestyle='dashed')
    axes[i].axhline(-0.5, color='blue', linestyle='dashed')
    axes[i].axhline(-0.7, color='red', linestyle='dashed')
    axes[i].set_ylabel("Correlation")
    axes[i].legend()
axes[-1].set_xlabel("Date")
plt.title(f"30 days Rolling Correlation: BTC and Cryptos for Crypto Crash 2021")
plt.tight_layout()
plt.show()

sns.heatmap(crypto_prices_pivot.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corr Heatmap between BTC and Cryptos for Crypto Crash 2021")
plt.show()

#results suggests low correlation of cryptos with BTC, there does not seems to have any affect of BTC market movements on other cryptos.
#LTC is the only coin with strong (0.92) correlation with BTC. Hence we can make interpretation of movements and trends of LTC using BTC
#So all cryptos needs to be analyzed separately for this duration
#for this period we will choose BTC, PXC (0.41) as they are least correlated

#plotting indicators for the time frame for BTC-USD
crypto = "BTC-USD"
data = crypto_crash_2021[crypto_crash_2021["symbol"] == crypto]

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.title(f"{crypto} - Closing Price")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["SMA_7"], label="SMA 7", color="green")
plt.plot(data["dates"], data["SMA_50"], label="SMA 50", color="red")
plt.plot(data["dates"], data["SMA_200"], label="SMA 200", color="orange")
plt.title(f"{crypto} - Closing Price and SMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["EMA_9"], label="EMA 9", color="green")
plt.plot(data["dates"], data["EMA_21"], label="EMA 21", color="red")
plt.title(f"{crypto} - Closing Price and EMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["RSI_7"], label="RSI 7", color="green")
plt.plot(data["dates"], data["RSI_14"], label="RSI 14", color="red")
plt.title(f"{crypto} - RSI - Market Condition")
plt.xlabel("Dates")
plt.ylabel("RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(50, color='orange', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ADX_14"], label="ADX 14", color="blue")
plt.title(f"{crypto} - ADX 14 - Trend Strength")
plt.xlabel("Dates")
plt.ylabel("ADX")
plt.axhline(25, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ATR_PCT"], label="ATR 14", color="blue")
plt.title(f"{crypto} - ATR 14 - Volatility")
plt.xlabel("Dates")
plt.ylabel("ATR Percentage")
plt.axhline(5, color='red', linestyle='--')
plt.axhline(2, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["OBV"], label="OBV", color="blue")
plt.title(f"{crypto} - OBV - Volatility")
plt.xlabel("Dates")
plt.ylabel("OBV")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

def plot_categorical_label(data, col, title):
    color_map = {
    'Strong Bullish': '#006400',          # Dark Green
    'Weak Bullish': '#90ee90',            # Light Green
    'Neutral Bullish': '#228B22',         # Forest Green
    'Bullish Momentum': '#32CD32',        # Lime Green

    'Strong Bearish': '#8B0000',          # Dark Red
    'Weak Bearish': '#f08080',            # Light Coral
    'Neutral Bearish': '#DC143C',         # Crimson
    'Oversold': '#FF4500',                # Orange Red

    'Strong Sideways': '#1e90ff',         # Dodger Blue
    'Weak Sideways': '#add8e6',           # Light Blue
    'Neutral Sideways': '#4682b4',        # Steel Blue

    'Neutral': '#808080',                 # Gray

    'Overbaught': '#800080',              # Purple
    'Low': '#a0522d',                     # Sienna
    'Moderate': '#ffa500',               # Orange
    'High': '#00008b',                    # Dark Blue
    }

    plot_data = data[["dates", col]].dropna()
    plt.figure(figsize=(12, 4))
    label_levels = {label: i for i, label in enumerate(plot_data[col].unique())}
    for label, level in label_levels.items():
        label_dates = plot_data[plot_data[col] == label]["dates"]
        plt.scatter(label_dates, [level] * len(label_dates), label=label, color=color_map.get(label, "black"), s=10)
    plt.title(title)
    plt.xlabel("Dates")
    plt.yticks(list(label_levels.values()), list(label_levels.keys()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

btc_data = crypto_crash_2021[crypto_crash_2021["symbol"] == "BTC-USD"]
plot_categorical_label(btc_data, "market_label", "BTC-Market Trend Label")
plot_categorical_label(btc_data, "rsi_condition", "BTC-Market RSI Condition")
plot_categorical_label(btc_data, "volatality_label", "BTC-Market Volatality Label")

#BTC-USD Candlestick
btc_crypto = crypto_crash_2021[crypto_crash_2021["symbol"] == "BTC-USD"]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=btc_crypto["dates"], open=btc_crypto["open"], high=btc_crypto["high"],
                                                  low=btc_crypto["low"], close=btc_crypto["close"],
                                                  name="Candlesticks"))

fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_50"], mode="lines", name="SMA_50", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_200"], mode="lines", name="SMA_200", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_21"], mode="lines", name="EMA_21", line=dict(color="green")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_9"], mode="lines", name="EMA_9", line=dict(color="red")))

fib_levels = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
fib_colors = {
    "P": "black", "R1": "green", "R2": "darkgreen", "R3": "lightgreen",
    "S1": "red", "S2": "darkred", "S3": "salmon"
}

for level in fib_levels:
    fig.add_trace(go.Scatter(
        x=btc_crypto["dates"], y=btc_crypto[level],
        mode="lines", name=f"Fibonacci {level}",
        line=dict(dash="dot", color=fib_colors[level])
    ))

fig.update_layout(title="BTC-USD Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
fig.show()

fig, ax = plt.subplots(5, 1, figsize=(12, 8))
ax[0].plot(btc_crypto["dates"], btc_crypto["RSI_14"], label="RSI_14", color="brown")
ax[0].set_title('RSI 14 (Relative Strength Index)')
ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
ax[0].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[0].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[0].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[0].legend()

ax[1].plot(btc_crypto["dates"], btc_crypto["RSI_7"], label="RSI_7", color="orange")
ax[1].set_title('RSI 7 (Relative Strength Index)')
ax[1].axhline(70, color='red', linestyle='--', label='Overbought')
ax[1].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[1].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[1].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[1].legend()

ax[2].plot(btc_crypto["dates"], btc_crypto["OBV"], label="OBV", color="blue")
ax[2].set_title('OBV (On-Balance Volume)')

ax[3].plot(btc_crypto["dates"], btc_crypto["ATR_PCT"], label="ATR_PCT", color="green")
ax[3].set_title('ATR (Average True Range)')
ax[3].axhline(5, color='red', linestyle='--', label='High')
ax[3].axhline(2, color='darkgreen', linestyle='--', label='Moderate')
ax[3].axhline(1.9, color='green', linestyle='--', label='Low')

ax[4].plot(btc_crypto["dates"], btc_crypto["ADX_14"], label="ADX_14", color="orange")
ax[4].set_title('ADX 14 (Average Directional Strength)')
ax[4].axhline(25, color='red', linestyle='--', label="Strong")
ax[4].axhline(20, color='green', linestyle='--', label="Weak")
ax[4].legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

#plotting indicators for the time frame for PXC-USD
crypto = "PXC-USD"
data = crypto_crash_2021[crypto_crash_2021["symbol"] == crypto]

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.title(f"{crypto} - Closing Price")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["SMA_7"], label="SMA 7", color="green")
plt.plot(data["dates"], data["SMA_50"], label="SMA 50", color="red")
plt.plot(data["dates"], data["SMA_200"], label="SMA 200", color="orange")
plt.title(f"{crypto} - Closing Price and SMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["EMA_9"], label="EMA 9", color="green")
plt.plot(data["dates"], data["EMA_21"], label="EMA 21", color="red")
plt.title(f"{crypto} - Closing Price and EMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["RSI_7"], label="RSI 7", color="green")
plt.plot(data["dates"], data["RSI_14"], label="RSI 14", color="red")
plt.title(f"{crypto} - RSI - Market Condition")
plt.xlabel("Dates")
plt.ylabel("RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(50, color='orange', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ADX_14"], label="ADX 14", color="blue")
plt.title(f"{crypto} - ADX 14 - Trend Strength")
plt.xlabel("Dates")
plt.ylabel("ADX")
plt.axhline(25, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ATR_PCT"], label="ATR 14", color="blue")
plt.title(f"{crypto} - ATR 14 - Volatility")
plt.xlabel("Dates")
plt.ylabel("ATR Percentage")
plt.axhline(5, color='red', linestyle='--')
plt.axhline(2, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["OBV"], label="OBV", color="blue")
plt.title(f"{crypto} - OBV - Volatility")
plt.xlabel("Dates")
plt.ylabel("OBV")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

def plot_categorical_label(data, col, title):
    color_map = {
    'Strong Bullish': '#006400',          # Dark Green
    'Weak Bullish': '#90ee90',            # Light Green
    'Neutral Bullish': '#228B22',         # Forest Green
    'Bullish Momentum': '#32CD32',        # Lime Green

    'Strong Bearish': '#8B0000',          # Dark Red
    'Weak Bearish': '#f08080',            # Light Coral
    'Neutral Bearish': '#DC143C',         # Crimson
    'Oversold': '#FF4500',                # Orange Red

    'Strong Sideways': '#1e90ff',         # Dodger Blue
    'Weak Sideways': '#add8e6',           # Light Blue
    'Neutral Sideways': '#4682b4',        # Steel Blue

    'Neutral': '#808080',                 # Gray

    'Overbaught': '#800080',              # Purple
    'Low': '#a0522d',                     # Sienna
    'Moderate': '#ffa500',               # Orange
    'High': '#00008b',                    # Dark Blue
    }

    plot_data = data[["dates", col]].dropna()
    plt.figure(figsize=(12, 4))
    label_levels = {label: i for i, label in enumerate(plot_data[col].unique())}
    for label, level in label_levels.items():
        label_dates = plot_data[plot_data[col] == label]["dates"]
        plt.scatter(label_dates, [level] * len(label_dates), label=label, color=color_map.get(label, "black"), s=10)
    plt.title(title)
    plt.xlabel("Dates")
    plt.yticks(list(label_levels.values()), list(label_levels.keys()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

pxc_data = crypto_crash_2021[crypto_crash_2021["symbol"] == "PXC-USD"]
plot_categorical_label(pxc_data, "market_label", "PXC-Market Trend Label")
plot_categorical_label(pxc_data, "rsi_condition", "PXC-Market RSI Condition")
plot_categorical_label(pxc_data, "volatality_label", "PXC-Market Volatality Label")

#PXC-USD Candlestick
pxc_crypto = crypto_crash_2021[crypto_crash_2021["symbol"] == "PXC-USD"]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=pxc_crypto["dates"], open=pxc_crypto["open"], high=pxc_crypto["high"],
                                                  low=pxc_crypto["low"], close=pxc_crypto["close"],
                                                  name="Candlesticks"))

fig.add_trace(go.Scatter(x=pxc_crypto["dates"], y=pxc_crypto["SMA_50"], mode="lines", name="SMA_50", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=pxc_crypto["dates"], y=pxc_crypto["SMA_200"], mode="lines", name="SMA_200", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=pxc_crypto["dates"], y=pxc_crypto["EMA_21"], mode="lines", name="EMA_21", line=dict(color="green")))
fig.add_trace(go.Scatter(x=pxc_crypto["dates"], y=pxc_crypto["EMA_9"], mode="lines", name="EMA_9", line=dict(color="red")))

fib_levels = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
fib_colors = {
    "P": "black", "R1": "green", "R2": "darkgreen", "R3": "lightgreen",
    "S1": "red", "S2": "darkred", "S3": "salmon"
}

for level in fib_levels:
    fig.add_trace(go.Scatter(
        x=pxc_crypto["dates"], y=pxc_crypto[level],
        mode="lines", name=f"Fibonacci {level}",
        line=dict(dash="dot", color=fib_colors[level])
    ))

fig.update_layout(title="PXC-USD Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
fig.show()

fig, ax = plt.subplots(5, 1, figsize=(12, 8))
ax[0].plot(pxc_crypto["dates"], pxc_crypto["RSI_14"], label="RSI_14", color="brown")
ax[0].set_title('RSI 14 (Relative Strength Index)')
ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
ax[0].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[0].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[0].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[0].legend()

ax[1].plot(pxc_crypto["dates"], pxc_crypto["RSI_7"], label="RSI_7", color="orange")
ax[1].set_title('RSI 7 (Relative Strength Index)')
ax[1].axhline(70, color='red', linestyle='--', label='Overbought')
ax[1].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[1].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[1].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[1].legend()

ax[2].plot(pxc_crypto["dates"], pxc_crypto["OBV"], label="OBV", color="blue")
ax[2].set_title('OBV (On-Balance Volume)')

ax[3].plot(pxc_crypto["dates"], pxc_crypto["ATR_PCT"], label="ATR_PCT", color="green")
ax[3].set_title('ATR (Average True Range)')
ax[3].axhline(5, color='red', linestyle='--', label='High')
ax[3].axhline(2, color='darkgreen', linestyle='--', label='Moderate')
ax[3].axhline(1.9, color='green', linestyle='--', label='Low')

ax[4].plot(pxc_crypto["dates"], pxc_crypto["ADX_14"], label="ADX_14", color="orange")
ax[4].set_title('ADX 14 (Average Directional Strength)')
ax[4].axhline(25, color='red', linestyle='--', label="Strong")
ax[4].axhline(20, color='green', linestyle='--', label="Weak")
ax[4].legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

#6. Bitcoin Halving of 2024 and market recorvery (Apr 2024) - Dataframe: btc_halving_2024 - Duration: Jan 2023 – Dec 2024
btc_halving_2024 = crypto_prices_final[(crypto_prices_final["month"] >= "2023-01") & (crypto_prices_final["month"] <= "2025-01")]
cols_needed = [
    "dates", "rolling_pct_change", "ATR_PCT", "ATR_14", "S3", "S2", "S1", "R3", "R2", "R1", "P",
    "ADX_14", "RSI_14", "RSI_7", "EMA_21", "EMA_9", "SMA_200", "SMA_50", "SMA_7"
]
btc_halving_2024 = btc_halving_2024.dropna(subset=cols_needed).copy()
print(btc_halving_2024[cols_needed].isna().sum())
print("Bitcoin Halving 2024 DataFrame 6\n")
print(btc_halving_2024.reset_index(drop=True))

#Plotting Correlation of BTC-USD with other cryptos for btc_halving_2024
crypto_prices_pivot = btc_halving_2024.pivot(index='dates', columns='symbol', values='close')
cryptos = ['NMC-USD', 'PPC-USD', 'LTC-USD', 'FTC-USD', 'PXC-USD']
BTC_Symbol = "BTC-USD"

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 15), sharex=True)
for i, crypto in enumerate(cryptos):
    rolling_corr = crypto_prices_pivot[crypto].rolling(window=30).corr(crypto_prices_pivot[BTC_Symbol])
    axes[i].plot(crypto_prices_pivot.index, rolling_corr, label=f"{crypto} vs BTC", color='blue')
    axes[i].axhline(0.7, color='red', linestyle='dashed')
    axes[i].axhline(0.5, color='blue', linestyle='dashed')
    axes[i].axhline(0, color='black', linestyle='dashed')
    axes[i].axhline(-0.5, color='blue', linestyle='dashed')
    axes[i].axhline(-0.7, color='red', linestyle='dashed')
    axes[i].set_ylabel("Correlation")
    axes[i].legend()
axes[-1].set_xlabel("Date")
plt.title(f"30 days Rolling Correlation: BTC and Cryptos for Bitcoin Halving 2024")
plt.tight_layout()
plt.show()

sns.heatmap(crypto_prices_pivot.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corr Heatmap between BTC and Cryptos for Bitcoin Halving 2024")
plt.show()

#results suggests low correlation of cryptos with BTC, there does not seems to have any affect of BTC market movements on other cryptos.
#FTC is the only coin with moderate (0.65) correlation with BTC. Hence we can somewhat make interpretation of movements and trends of FTC using BTC but cannot have strong confidence in results
#So all cryptos needs to be analyzed separately for this duration
#for this period we will choose BTC, NMC (-0.61) as they are least correlated

#plotting indicators for the time frame for BTC-USD
crypto = "BTC-USD"
data = btc_halving_2024[btc_halving_2024["symbol"] == crypto]

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.title(f"{crypto} - Closing Price")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["SMA_7"], label="SMA 7", color="green")
plt.plot(data["dates"], data["SMA_50"], label="SMA 50", color="red")
plt.plot(data["dates"], data["SMA_200"], label="SMA 200", color="orange")
plt.title(f"{crypto} - Closing Price and SMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["EMA_9"], label="EMA 9", color="green")
plt.plot(data["dates"], data["EMA_21"], label="EMA 21", color="red")
plt.title(f"{crypto} - Closing Price and EMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["RSI_7"], label="RSI 7", color="green")
plt.plot(data["dates"], data["RSI_14"], label="RSI 14", color="red")
plt.title(f"{crypto} - RSI - Market Condition")
plt.xlabel("Dates")
plt.ylabel("RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(50, color='orange', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ADX_14"], label="ADX 14", color="blue")
plt.title(f"{crypto} - ADX 14 - Trend Strength")
plt.xlabel("Dates")
plt.ylabel("ADX")
plt.axhline(25, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ATR_PCT"], label="ATR 14", color="blue")
plt.title(f"{crypto} - ATR 14 - Volatility")
plt.xlabel("Dates")
plt.ylabel("ATR Percentage")
plt.axhline(5, color='red', linestyle='--')
plt.axhline(2, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["OBV"], label="OBV", color="blue")
plt.title(f"{crypto} - OBV - Volatility")
plt.xlabel("Dates")
plt.ylabel("OBV")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

def plot_categorical_label(data, col, title):
    color_map = {
    'Strong Bullish': '#006400',          # Dark Green
    'Weak Bullish': '#90ee90',            # Light Green
    'Neutral Bullish': '#228B22',         # Forest Green
    'Bullish Momentum': '#32CD32',        # Lime Green

    'Strong Bearish': '#8B0000',          # Dark Red
    'Weak Bearish': '#f08080',            # Light Coral
    'Neutral Bearish': '#DC143C',         # Crimson
    'Oversold': '#FF4500',                # Orange Red

    'Strong Sideways': '#1e90ff',         # Dodger Blue
    'Weak Sideways': '#add8e6',           # Light Blue
    'Neutral Sideways': '#4682b4',        # Steel Blue

    'Neutral': '#808080',                 # Gray

    'Overbaught': '#800080',              # Purple
    'Low': '#a0522d',                     # Sienna
    'Moderate': '#ffa500',               # Orange
    'High': '#00008b',                    # Dark Blue
    }

    plot_data = data[["dates", col]].dropna()
    plt.figure(figsize=(12, 4))
    label_levels = {label: i for i, label in enumerate(plot_data[col].unique())}
    for label, level in label_levels.items():
        label_dates = plot_data[plot_data[col] == label]["dates"]
        plt.scatter(label_dates, [level] * len(label_dates), label=label, color=color_map.get(label, "black"), s=10)
    plt.title(title)
    plt.xlabel("Dates")
    plt.yticks(list(label_levels.values()), list(label_levels.keys()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

btc_data = btc_halving_2024[btc_halving_2024["symbol"] == "BTC-USD"]
plot_categorical_label(btc_data, "market_label", "BTC-Market Trend Label")
plot_categorical_label(btc_data, "rsi_condition", "BTC-Market RSI Condition")
plot_categorical_label(btc_data, "volatality_label", "BTC-Market Volatality Label")

#BTC-USD Candlestick
btc_crypto = btc_halving_2024[btc_halving_2024["symbol"] == "BTC-USD"]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=btc_crypto["dates"], open=btc_crypto["open"], high=btc_crypto["high"],
                                                  low=btc_crypto["low"], close=btc_crypto["close"],
                                                  name="Candlesticks"))

fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_50"], mode="lines", name="SMA_50", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["SMA_200"], mode="lines", name="SMA_200", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_21"], mode="lines", name="EMA_21", line=dict(color="green")))
fig.add_trace(go.Scatter(x=btc_crypto["dates"], y=btc_crypto["EMA_9"], mode="lines", name="EMA_9", line=dict(color="red")))

fib_levels = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
fib_colors = {
    "P": "black", "R1": "green", "R2": "darkgreen", "R3": "lightgreen",
    "S1": "red", "S2": "darkred", "S3": "salmon"
}

for level in fib_levels:
    fig.add_trace(go.Scatter(
        x=btc_crypto["dates"], y=btc_crypto[level],
        mode="lines", name=f"Fibonacci {level}",
        line=dict(dash="dot", color=fib_colors[level])
    ))

fig.update_layout(title="BTC-USD Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
fig.show()

fig, ax = plt.subplots(5, 1, figsize=(12, 8))
ax[0].plot(btc_crypto["dates"], btc_crypto["RSI_14"], label="RSI_14", color="brown")
ax[0].set_title('RSI 14 (Relative Strength Index)')
ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
ax[0].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[0].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[0].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[0].legend()

ax[1].plot(btc_crypto["dates"], btc_crypto["RSI_7"], label="RSI_7", color="orange")
ax[1].set_title('RSI 7 (Relative Strength Index)')
ax[1].axhline(70, color='red', linestyle='--', label='Overbought')
ax[1].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[1].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[1].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[1].legend()

ax[2].plot(btc_crypto["dates"], btc_crypto["OBV"], label="OBV", color="blue")
ax[2].set_title('OBV (On-Balance Volume)')

ax[3].plot(btc_crypto["dates"], btc_crypto["ATR_PCT"], label="ATR_PCT", color="green")
ax[3].set_title('ATR (Average True Range)')
ax[3].axhline(5, color='red', linestyle='--', label='High')
ax[3].axhline(2, color='darkgreen', linestyle='--', label='Moderate')
ax[3].axhline(1.9, color='green', linestyle='--', label='Low')

ax[4].plot(btc_crypto["dates"], btc_crypto["ADX_14"], label="ADX_14", color="orange")
ax[4].set_title('ADX 14 (Average Directional Strength)')
ax[4].axhline(25, color='red', linestyle='--', label="Strong")
ax[4].axhline(20, color='green', linestyle='--', label="Weak")
ax[4].legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

#plotting indicators for the time frame for NMC-USD
crypto = "NMC-USD"
data = btc_halving_2024[btc_halving_2024["symbol"] == crypto]

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.title(f"{crypto} - Closing Price")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["SMA_7"], label="SMA 7", color="green")
plt.plot(data["dates"], data["SMA_50"], label="SMA 50", color="red")
plt.plot(data["dates"], data["SMA_200"], label="SMA 200", color="orange")
plt.title(f"{crypto} - Closing Price and SMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["close"], label="Closing Price", color="blue")
plt.plot(data["dates"], data["EMA_9"], label="EMA 9", color="green")
plt.plot(data["dates"], data["EMA_21"], label="EMA 21", color="red")
plt.title(f"{crypto} - Closing Price and EMA")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["RSI_7"], label="RSI 7", color="green")
plt.plot(data["dates"], data["RSI_14"], label="RSI 14", color="red")
plt.title(f"{crypto} - RSI - Market Condition")
plt.xlabel("Dates")
plt.ylabel("RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(50, color='orange', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ADX_14"], label="ADX 14", color="blue")
plt.title(f"{crypto} - ADX 14 - Trend Strength")
plt.xlabel("Dates")
plt.ylabel("ADX")
plt.axhline(25, color='red', linestyle='--')
plt.axhline(20, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["ATR_PCT"], label="ATR 14", color="blue")
plt.title(f"{crypto} - ATR 14 - Volatility")
plt.xlabel("Dates")
plt.ylabel("ATR Percentage")
plt.axhline(5, color='red', linestyle='--')
plt.axhline(2, color='green', linestyle='--')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 6))
plt.plot(data["dates"], data["OBV"], label="OBV", color="blue")
plt.title(f"{crypto} - OBV - Volatility")
plt.xlabel("Dates")
plt.ylabel("OBV")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

def plot_categorical_label(data, col, title):
    color_map = {
    'Strong Bullish': '#006400',          # Dark Green
    'Weak Bullish': '#90ee90',            # Light Green
    'Neutral Bullish': '#228B22',         # Forest Green
    'Bullish Momentum': '#32CD32',        # Lime Green

    'Strong Bearish': '#8B0000',          # Dark Red
    'Weak Bearish': '#f08080',            # Light Coral
    'Neutral Bearish': '#DC143C',         # Crimson
    'Oversold': '#FF4500',                # Orange Red

    'Strong Sideways': '#1e90ff',         # Dodger Blue
    'Weak Sideways': '#add8e6',           # Light Blue
    'Neutral Sideways': '#4682b4',        # Steel Blue

    'Neutral': '#808080',                 # Gray

    'Overbaught': '#800080',              # Purple
    'Low': '#a0522d',                     # Sienna
    'Moderate': '#ffa500',               # Orange
    'High': '#00008b',                    # Dark Blue
    }

    plot_data = data[["dates", col]].dropna()
    plt.figure(figsize=(12, 4))
    label_levels = {label: i for i, label in enumerate(plot_data[col].unique())}
    for label, level in label_levels.items():
        label_dates = plot_data[plot_data[col] == label]["dates"]
        plt.scatter(label_dates, [level] * len(label_dates), label=label, color=color_map.get(label, "black"), s=10)
    plt.title(title)
    plt.xlabel("Dates")
    plt.yticks(list(label_levels.values()), list(label_levels.keys()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

nmc_data = btc_halving_2024[btc_halving_2024["symbol"] == "NMC-USD"]
plot_categorical_label(nmc_data, "market_label", "NMC-Market Trend Label")
plot_categorical_label(nmc_data, "rsi_condition", "NMC-Market RSI Condition")
plot_categorical_label(nmc_data, "volatality_label", "NMC-Market Volatality Label")

#BTC-USD Candlestick
nmc_crypto = btc_halving_2024[btc_halving_2024["symbol"] == "NMC-USD"]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=nmc_crypto["dates"], open=nmc_crypto["open"], high=nmc_crypto["high"],
                                                  low=nmc_crypto["low"], close=nmc_crypto["close"],
                                                  name="Candlesticks"))

fig.add_trace(go.Scatter(x=nmc_crypto["dates"], y=nmc_crypto["SMA_50"], mode="lines", name="SMA_50", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=nmc_crypto["dates"], y=nmc_crypto["SMA_200"], mode="lines", name="SMA_200", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=nmc_crypto["dates"], y=nmc_crypto["EMA_21"], mode="lines", name="EMA_21", line=dict(color="green")))
fig.add_trace(go.Scatter(x=nmc_crypto["dates"], y=nmc_crypto["EMA_9"], mode="lines", name="EMA_9", line=dict(color="red")))

fib_levels = ["P", "R1", "R2", "R3", "S1", "S2", "S3"]
fib_colors = {
    "P": "black", "R1": "green", "R2": "darkgreen", "R3": "lightgreen",
    "S1": "red", "S2": "darkred", "S3": "salmon"
}

for level in fib_levels:
    fig.add_trace(go.Scatter(
        x=nmc_crypto["dates"], y=nmc_crypto[level],
        mode="lines", name=f"Fibonacci {level}",
        line=dict(dash="dot", color=fib_colors[level])
    ))

fig.update_layout(title="NMC-USD Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
fig.show()

fig, ax = plt.subplots(5, 1, figsize=(12, 8))
ax[0].plot(nmc_crypto["dates"], nmc_crypto["RSI_14"], label="RSI_14", color="brown")
ax[0].set_title('RSI 14 (Relative Strength Index)')
ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
ax[0].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[0].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[0].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[0].legend()

ax[1].plot(nmc_crypto["dates"], nmc_crypto["RSI_7"], label="RSI_7", color="orange")
ax[1].set_title('RSI 7 (Relative Strength Index)')
ax[1].axhline(70, color='red', linestyle='--', label='Overbought')
ax[1].axhline(51, color='green', linestyle='--', label='Bullish Momentum')
ax[1].axhline(49, color='lightgreen', linestyle='--', label='Bearish Momentum')
ax[1].axhline(30, color='darkgreen', linestyle='--', label='Oversold')
ax[1].legend()

ax[2].plot(nmc_crypto["dates"], nmc_crypto["OBV"], label="OBV", color="blue")
ax[2].set_title('OBV (On-Balance Volume)')

ax[3].plot(nmc_crypto["dates"], nmc_crypto["ATR_PCT"], label="ATR_PCT", color="green")
ax[3].set_title('ATR (Average True Range)')
ax[3].axhline(5, color='red', linestyle='--', label='High')
ax[3].axhline(2, color='darkgreen', linestyle='--', label='Moderate')
ax[3].axhline(1.9, color='green', linestyle='--', label='Low')

ax[4].plot(nmc_crypto["dates"], nmc_crypto["ADX_14"], label="ADX_14", color="orange")
ax[4].set_title('ADX 14 (Average Directional Strength)')
ax[4].axhline(25, color='red', linestyle='--', label="Strong")
ax[4].axhline(20, color='green', linestyle='--', label="Weak")
ax[4].legend()

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()