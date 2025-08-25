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

print(crypto_prices_6.columns)
print(crypto_prices_6.isnull().sum())
cols_needed = [
    "dates", "rolling_pct_change", "ATR_PCT", "ATR_14", "S3", "S2", "S1", "R3", "R2", "R1", "P",
    "ADX_14", "RSI_14", "RSI_7", "EMA_21", "EMA_9", "SMA_200", "SMA_50", "SMA_7"
]
crypto_prices_6 = crypto_prices_6.dropna(subset=cols_needed)
print(crypto_prices_6.isnull().sum())
print(crypto_prices_6)
crypto_prices_6.to_csv("Data\\Crypto File with 6 Cryptos and Indicators.csv", index=False)
cols_to_drop = ['market_trend_percent_change',
       'trend_SMA_crossover', 'trend_EMA_crossover', 'market_trend_direction',
       'trend_strength_ADX', 'market_label', 'rsi_condition', 'volatality_label']
crypto_prices_6 = crypto_prices_6.drop(columns=cols_to_drop).reset_index(drop=True)
print(crypto_prices_6.columns)
print(crypto_prices_6)

#splitting data before creating labels to avoid data leackage
train_df = crypto_prices_6[(crypto_prices_6['dates'] >= '2015-07-19') & (crypto_prices_6['dates'] <= '2021-12-31')].reset_index(drop=True)
print(train_df)

val_df = crypto_prices_6[(crypto_prices_6['dates'] >= '2022-01-01') & (crypto_prices_6['dates'] <= '2023-12-31')].reset_index(drop=True)
print(val_df)

test_df = crypto_prices_6[(crypto_prices_6['dates'] >= '2024-01-01') & (crypto_prices_6['dates'] <= '2025-02-11')].reset_index(drop=True)
print(test_df)

#creating labels using rolling percentage and ADX values and calculating next day percentage change column for train_df
train_df['next_day_pct_change'] = train_df.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)

def classify_market(pct_change):
    if pct_change > bullish_threshold:
        return "Bullish"
    elif pct_change < bearish_threshold:
        return "Bearish"
    else:
        return "Sideways"

train_df["market_trend_label"] = train_df["rolling_pct_change"].apply(classify_market)
train_df = train_df.reset_index(drop=True)
print(train_df.columns)
print(train_df)

#creating labels using rolling percentage and ADX values and calculating next day percentage change column for val_df
val_df['next_day_pct_change'] = val_df.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)

def classify_market(pct_change):
    if pct_change > bullish_threshold:
        return "Bullish"
    elif pct_change < bearish_threshold:
        return "Bearish"
    else:
        return "Sideways"

val_df["market_trend_label"] = val_df["rolling_pct_change"].apply(classify_market)
val_df = val_df.reset_index(drop=True)
print(val_df.columns)
print(val_df)

#creating labels using rolling percentage and ADX values and calculating next day percentage change column for test_df
test_df['next_day_pct_change'] = test_df.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)

def classify_market(pct_change):
    if pct_change > bullish_threshold:
        return "Bullish"
    elif pct_change < bearish_threshold:
        return "Bearish"
    else:
        return "Sideways"

test_df["market_trend_label"] = test_df["rolling_pct_change"].apply(classify_market)
test_df = test_df.reset_index(drop=True)
print(test_df.columns)
print(test_df)

#features
features = ['open', 'close', 'high', 'low', 'volume', 'SMA_7', 'SMA_50', 'SMA_200', 
            'EMA_9', 'EMA_21', 'RSI_7', 'RSI_14', 'ATR_14', 'OBV', "ADX_14"]
target = "market_trend_label"

#encoding target variable
label_encoder = LabelEncoder()

#train_df
train_df[target] = label_encoder.fit_transform(train_df[target])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)

#val_df
val_df[target] = label_encoder.fit_transform(val_df[target])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)

#test_df
test_df[target] = label_encoder.fit_transform(test_df[target])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)

#splitting data in X and y for train, validate, test
X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]

X_test = test_df[features]
y_test = test_df[target]

#XGBoost Classification
#Train XGB model
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train, y_train)

#Train Randomg forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature Importances
xgb_imp = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False)
rf_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

# Plot
xgb_imp.plot(kind='bar', title='XGBoost Feature Importance (All Features)', color='darkorange')
plt.tight_layout()
plt.show()

rf_imp.plot(kind='bar', title='Random Forest Feature Importance (All Features)', color='forestgreen')
plt.tight_layout()
plt.show()

#Select Top features
top_n = 7
top_features_xgb = set(xgb_imp.head(top_n).index)
top_features_rf = set(rf_imp.head(top_n).index)

#using intersection of permutation features
refined_features = list(top_features_xgb.intersection(top_features_rf))
print(f"\nRefined Feature Set (Top-{top_n} Intersection): {refined_features}")

X_train_refined = train_df[refined_features]

#Training XGB and RF Classification models for train_df
xgb_refined = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
rf_refined = RandomForestClassifier(random_state=42)

xgb_param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

rf_param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro'
}

xgb_random_search = RandomizedSearchCV(
    estimator=xgb_refined,
    param_distributions=xgb_param_dist,
    n_iter=30,
    scoring=scoring,
    refit='accuracy',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
xgb_random_search.fit(X_train_refined, y_train)

rf_random_search = RandomizedSearchCV(
    estimator=rf_refined,
    param_distributions=rf_param_dist,
    n_iter=30,
    scoring=scoring,
    refit='accuracy',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
rf_random_search.fit(X_train_refined, y_train)

best_xgb = xgb_random_search.best_estimator_
joblib.dump(best_xgb, 'best_xgb_model.pkl')
best_rf = rf_random_search.best_estimator_
joblib.dump(best_rf, 'best_rf_model.pkl')

#Validating XGB and RF Classification models for val_df
#loading trained models
best_xgb = joblib.load('best_xgb_model.pkl')
best_rf = joblib.load('best_rf_model.pkl')

X_val_refined = val_df[refined_features]
print(X_val_refined.columns)

y_val_pred_xgb = best_xgb.predict(X_val_refined)
acc = accuracy_score(y_val, y_val_pred_xgb)
print(f"\n Validation XGBoost Evaluation:")
print("Accuracy:", round(acc, 4))
print("Classification Report:\n", classification_report(y_val, y_val_pred_xgb))
cm = confusion_matrix(y_val, y_val_pred_xgb)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Oranges')
plt.title(f'XGBoost Confusion Matrix')
plt.grid(False)
plt.show()

y_val_pred_rf = best_rf.predict(X_val_refined)
acc = accuracy_score(y_val, y_val_pred_rf)
print(f"\n Validation Random Forest Evaluation:")
print("Accuracy:", round(acc, 4))
print("Classification Report:\n", classification_report(y_val, y_val_pred_rf))
cm = confusion_matrix(y_val, y_val_pred_rf)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Oranges')
plt.title(f'Random Forest Confusion Matrix')
plt.grid(False)
plt.show()

#using voting classifier ensemble method on XGB and RF
ensemble = VotingClassifier(estimators=[
    ('xgb', best_xgb),
    ('rf', best_rf)
], voting='soft')  # or 'hard'

ensemble.fit(X_train_refined, y_train)
joblib.dump(ensemble, 'ensemble_model.pkl')
ensemble = joblib.load('ensemble_model.pkl')
X_val_refined = val_df[refined_features]
y_val_pred_ens = ensemble.predict(X_val_refined)

print("Validation Ensemble Evaluation:")
print("Accuracy:", accuracy_score(y_val, y_val_pred_ens))
print("Classification Report:\n", classification_report(y_val, y_val_pred_ens))
cm = confusion_matrix(y_val, y_val_pred_ens)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Oranges')
plt.title(f'Ensemble Confusion Matrix')
plt.grid(False)
plt.show()

#Testing XGB and RF Classification and Ensemble models for test_df
#loading trained models
best_xgb = joblib.load('best_xgb_model.pkl')
best_rf = joblib.load('best_rf_model.pkl')

X_test_refined = test_df[refined_features]
print(X_test_refined.columns)

y_test_pred_xgb = best_xgb.predict(X_test_refined)
acc = accuracy_score(y_test, y_test_pred_xgb)
print(f"\n Testing XGBoost Evaluation:")
print("Accuracy:", round(acc, 4))
print("Classification Report:\n", classification_report(y_test, y_test_pred_xgb))
cm = confusion_matrix(y_test, y_test_pred_xgb)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Oranges')
plt.title(f'XGBoost Confusion Matrix')
plt.grid(False)
plt.show()

y_test_pred_rf= best_rf.predict(X_test_refined)
acc = accuracy_score(y_test, y_test_pred_rf)
print(f"\n Testing Random Forest Evaluation:")
print("Accuracy:", round(acc, 4))
print("Classification Report:\n", classification_report(y_test, y_test_pred_rf))
cm = confusion_matrix(y_test, y_test_pred_rf)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Oranges')
plt.title(f'Random Forest Confusion Matrix')
plt.grid(False)
plt.show()

ensemble = joblib.load('ensemble_model.pkl')
X_test_refined = test_df[refined_features]
y_test_pred_ens = ensemble.predict(X_test_refined)

print("Testing Ensemble Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_test_pred_ens))
print("Classification Report:\n", classification_report(y_test, y_test_pred_ens))
cm = confusion_matrix(y_test, y_test_pred_ens)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Oranges')
plt.title(f'Ensemble Confusion Matrix')
plt.grid(False)
plt.show()