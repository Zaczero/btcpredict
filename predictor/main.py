import json
import os
import random
from collections import deque
from copy import copy
from typing import Tuple
import fire
import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression, SelectFwe, VarianceThreshold
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LassoLarsCV, RidgeCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, MinMaxScaler, Binarizer, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, Normalizer, PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from tpot import TPOTRegressor
from tpot.builtins import StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))


def fetch_price_data() -> pd.DataFrame:
    r = requests.get('https://api.blockchair.com/bitcoin/blocks', {
        'a': 'date,price(btc_usd)',
    }, timeout=30)
    r.raise_for_status()

    json = r.json()

    date_list = deque()
    price_list = deque()

    for data in json['data']:
        date_list.append(pd.to_datetime(data['date'], format='%Y-%m-%d').replace(hour=0, minute=0, second=0, microsecond=0))
        price_list.append(data['price(btc_usd)'])

    df = pd.DataFrame({
        'Date': pd.to_datetime(list(date_list)),
        'Price': price_list
    })

    df = df[df['Date'] >= '2011-01-01'].reset_index(drop=True)
    df.loc[df['Price'] <= 0.01, 'Price'] = np.NAN
    df.interpolate(inplace=True)

    df['PriceLog'] = np.log(df['Price'] + 1)
    return df


def mark_top_and_bottom(df: pd.DataFrame) -> pd.DataFrame:
    then_forward_window_size = 365 * 2
    ignore_past_n_days = 90

    df['IsTop'] = 0
    df['IsBottom'] = 0

    current_forward_window_size = 365
    current_index = 0
    searching_top = True

    while True:
        window = df.loc[current_index:current_index+current_forward_window_size-1]

        if window.shape[0] == 0:
            break

        if searching_top:
            new_index = window['Price'].idxmax()
            if new_index == current_index:
                df.loc[current_index, 'IsTop'] = 1
                current_forward_window_size = then_forward_window_size
                current_index += 1
                searching_top = False
                continue
        else:
            new_index = window['Price'].idxmin()
            if new_index == current_index:
                df.loc[current_index, 'IsBottom'] = 1
                current_forward_window_size = then_forward_window_size
                current_index += 1
                searching_top = True
                continue

        current_index = new_index

    df.loc[df.shape[0]-ignore_past_n_days:, 'IsTop'] = 0
    df.loc[df.shape[0]-ignore_past_n_days:, 'IsBottom'] = 0
    return df


def fetch_block_data(df: pd.DataFrame) -> pd.DataFrame:
    blocks_per_day = 6 * 24
    halving_block = 210000

    df['IsHalving'] = 0
    df['Cycle'] = 0
    df['CyclePow'] = 0
    df['CoinIssuance'] = 0
    df['MarketCap'] = 0

    total_market_cap = 0
    current_block = halving_block
    current_cycle = 1
    current_issuance = 50

    while True:
        r = requests.get(f'https://api.blockchair.com/bitcoin/dashboards/block/{current_block}', timeout=30)
        r.raise_for_status()

        json = r.json()
        current_daily_issuance = blocks_per_day * current_issuance

        if str(current_block) not in json['data']:
            df.loc[df['Cycle'] == 0, 'Cycle'] = current_cycle
            df.loc[df['CoinIssuance'] == 0, 'CoinIssuance'] = current_daily_issuance

            first_empty_market_cap_index = df.loc[df['MarketCap'] == 0].index[0]
            last_empty_market_cap_index = df.loc[df['MarketCap'] == 0].index[-1]
            num = last_empty_market_cap_index - first_empty_market_cap_index + 1
            current_market_cap_increase = (num - 1) * current_daily_issuance

            df.loc[first_empty_market_cap_index:last_empty_market_cap_index, 'MarketCap'] = \
                np.linspace(total_market_cap, total_market_cap + current_market_cap_increase, num)
            break

        current_halving_block_market_cap = halving_block * current_issuance
        current_halving_block_date = json['data'][str(current_block)]['block']['date']
        current_halving_block_row = df[df['Date'] == current_halving_block_date]

        if current_halving_block_row.shape[0] == 1:
            current_halving_block_index = current_halving_block_row.index[0]

            df.loc[current_halving_block_index, 'IsHalving'] = 1
            df.loc[(df['Date'] < current_halving_block_date) & (df['Cycle'] == 0), 'Cycle'] = current_cycle
            df.loc[(df['Date'] < current_halving_block_date) & (df['CoinIssuance'] == 0), 'CoinIssuance'] = current_daily_issuance

            first_empty_market_cap_index = df.loc[df['MarketCap'] == 0].index[0]

            num = current_halving_block_index - first_empty_market_cap_index
            start_offset = 0

            if first_empty_market_cap_index == 0:
                expected_num = halving_block / blocks_per_day
                start_offset = (expected_num - num) * current_daily_issuance

            df.loc[first_empty_market_cap_index:current_halving_block_index-1, 'MarketCap'] = \
                np.linspace(total_market_cap + start_offset, total_market_cap + current_halving_block_market_cap - current_daily_issuance, num)

        total_market_cap += current_halving_block_market_cap
        current_block += halving_block
        current_cycle += 1
        current_issuance /= 2

    df['CyclePow'] = np.power(2, df['Cycle'])
    df['CoinIssuanceUSD'] = df.apply(lambda row: row['Price'] * row['CoinIssuance'], axis=1)
    df['CoinIssuanceUSDLog'] = np.log(df['CoinIssuanceUSD'])
    df['MarketCapUSD'] = df.apply(lambda row: row['Price'] * row['MarketCap'], axis=1)
    df['MarketCapUSDLog'] = np.log(df['MarketCapUSD'])
    return df


def mark_days_since(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        indexes = df.loc[df[col] == 1].index
        df[f'DaysSince{col}'] = df.index.to_series().apply(lambda v: min([v-index if index <= v else np.NaN for index in indexes]))

    return df


def mark_bottom_price(df: pd.DataFrame) -> pd.DataFrame:
    df['BottomPrice'] = np.NaN

    bottom_indexes = df[df['IsBottom'] == 1].index

    for current_index, next_index in zip(bottom_indexes, bottom_indexes[1:]):
        df.loc[current_index:next_index-1, 'BottomPrice'] = df.loc[current_index, 'Price']

    df.loc[bottom_indexes[-1]:, 'BottomPrice'] = df.loc[bottom_indexes[-1], 'Price']
    return df


def impute_days_since(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        indexes = df.loc[df[col] == 1].index
        max_values = df.loc[indexes - 1, f'DaysSince{col}'][1:]
        avg_value = np.round(np.mean(max_values))
        df.loc[:indexes[0]-1, f'DaysSince{col}'] = np.linspace(avg_value - indexes[0] + 1, avg_value, indexes[0])

    return df


def mark_price_increase(df: pd.DataFrame) -> pd.DataFrame:
    df['PriceIncrease'] = (df['Price'] - df['BottomPrice']) / df['BottomPrice']
    df['PriceIncreaseLog'] = np.log(df['PriceIncrease'] + 1)

    df['PriceIncreaseCycle'] = df['PriceIncrease'] * df['CyclePow']
    df['PriceIncreaseCycleLog'] = np.log(df['PriceIncreaseCycle'] + 1)
    return df


def mark_top_percentage(df: pd.DataFrame) -> pd.DataFrame:
    df['TopDateBasedPercentage'] = np.NaN
    df['TopPriceBasedPercentage'] = np.NaN

    current_top_index = min(df[df['IsTop'] == 1].index)
    current_top_price = df.loc[current_top_index, 'Price']
    current_bottom_index = min(df[df['IsBottom'] == 1].index)
    current_bottom_price = df.loc[current_bottom_index, 'Price']

    while True:
        # decreasing value
        if current_top_index < current_bottom_index:
            df.loc[current_top_index:current_bottom_index, 'TopDateBasedPercentage'] = np.linspace(1, 0, current_bottom_index - current_top_index + 1)
            df.loc[current_top_index:current_bottom_index, 'TopPriceBasedPercentage'] = \
                df.loc[current_top_index:current_bottom_index, 'Price'].apply(lambda v: (v - current_bottom_price) / (current_top_price - current_bottom_price))

            mask = (df['IsTop'] == 1) & (df.index > current_top_index)
            if sum(mask) == 0:
                break

            current_top_index = min(df[mask].index)
            current_top_price = df.loc[current_top_index, 'Price']
        # increasing value
        else:
            df.loc[current_bottom_index:current_top_index, 'TopDateBasedPercentage'] = np.linspace(0, 1, current_top_index - current_bottom_index + 1)
            df.loc[current_bottom_index:current_top_index, 'TopPriceBasedPercentage'] = \
                df.loc[current_bottom_index:current_top_index, 'Price'].apply(lambda v: (v - current_bottom_price) / (current_top_price - current_bottom_price))

            mask = (df['IsBottom'] == 1) & (df.index > current_bottom_index)
            if sum(mask) == 0:
                break

            current_bottom_index = min(df[mask].index)
            current_bottom_price = df.loc[current_bottom_index, 'Price']

    df['TopDateBasedPercentage'] *= 100
    df['TopPriceBasedPercentage'] *= 100
    return df


def mark_2yma(df: pd.DataFrame) -> pd.DataFrame:
    df['2YMA'] = df['Price'].rolling(365 * 2).mean()
    df['2YMAx5'] = df['2YMA'] * 5
    df['2YMAPercentage'] = df.apply(lambda row: (row['Price'] - row['2YMA']) / (row['2YMAx5'] - row['2YMA']), axis=1)

    df['2YMAPercentageCycle'] = df['2YMAPercentage'] * df['CyclePow']
    return df


def mark_pi_cycle(df: pd.DataFrame) -> pd.DataFrame:
    df['111MA'] = df['Price'].rolling(111).mean()
    df['350MAx2'] = df['Price'].rolling(350).mean() * 2
    df['PiDifference'] = np.abs(df['111MA'] - df['350MAx2'])
    df['PiDifferenceLog'] = np.log(df['PiDifference'] + 1)

    df['111MALog'] = np.log(df['111MA'])
    df['350MAx2Log'] = np.log(df['350MAx2'])
    df['PiLogDifference'] = np.abs(df['111MALog'] - df['350MAx2Log'])
    df['PiLogDifferenceLog'] = np.log(df['PiLogDifference'] + 1)

    df['PiLogDifferenceCycle'] = df['PiLogDifferenceLog'] * df['CyclePow']
    return df


def mark_golden_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['GoldenRatio'] = df.apply(lambda row: (row['DaysSinceIsHalving'] / row['DaysSinceIsBottom']) if row['DaysSinceIsHalving'] < row['DaysSinceIsBottom'] else 0, axis=1)
    df['GoldenRatio'] = df['GoldenRatio'] / 0.49
    df['GoldenRatio'].replace(0, np.NAN, inplace=True)
    df['GoldenRatioAbs'] = 1 - np.abs(1 - df['GoldenRatio'])
    return df


def mark_puell_multiple(df: pd.DataFrame) -> pd.DataFrame:
    df['365MA-CoinIssuanceUSD'] = df['CoinIssuanceUSD'].rolling(365).mean()
    df['PuellMultiple'] = df['CoinIssuanceUSD'] / df['365MA-CoinIssuanceUSD']

    df['PuellMultipleCycle'] = df['PuellMultiple'] * df['CyclePow']
    return df


def mark_stock_to_flow(df: pd.DataFrame) -> pd.DataFrame:
    df['SF10'] = df['MarketCap'] / (df['CoinIssuance'].rolling(10).mean() * 365)
    df['SF463'] = df['MarketCap'] / (df['CoinIssuance'].rolling(463).mean() * 365)
    df['SFDifference'] = np.log(np.abs(df['SF10'] - df['SF463']) + 1)
    return df


def init_dataframe(cache_path: str, force_cache_update: bool) -> pd.DataFrame:
    if os.path.exists(cache_path) and not force_cache_update:
        return pd.read_csv(cache_path, parse_dates=['Date'])

    df = fetch_price_data()
    df = mark_top_and_bottom(df)
    df = fetch_block_data(df)
    df = mark_days_since(df, ['IsTop', 'IsBottom', 'IsHalving'])
    df = mark_bottom_price(df)
    df = impute_days_since(df, ['IsTop', 'IsBottom', 'IsHalving'])
    df = mark_price_increase(df)
    df = mark_top_percentage(df)
    df = mark_2yma(df)
    df = mark_pi_cycle(df)
    df = mark_golden_ratio(df)
    df = mark_puell_multiple(df)
    df = mark_stock_to_flow(df)
    df.to_csv(cache_path)
    return df


def drop_repeated_bins(df: pd.DataFrame, col: str, bins: list, max_repeat: int) -> pd.DataFrame:
    vals = df[col].values
    vals_bin = np.digitize(vals, bins)
    result_indexes = []

    for y_train_bin_id in np.unique(vals_bin):
        bin_indexes = list(np.argwhere(vals_bin == y_train_bin_id).flatten())

        if len(bin_indexes) > max_repeat:
            result_indexes += random.sample(list(bin_indexes), max_repeat)
        else:
            result_indexes += bin_indexes

    return df.iloc[result_indexes]


def get_date_pipeline() -> Tuple[float, Pipeline]:
    mean_error = np.sqrt(0.03735920619021114)
    pipeline = make_pipeline(
        SelectPercentile(score_func=f_regression, percentile=20),
        RBFSampler(gamma=0.75),
        LassoLarsCV(normalize=True)
    )

    set_param_recursive(pipeline.steps, 'random_state', 45)
    return mean_error, pipeline


def get_price_pipeline() -> Tuple[float, Pipeline]:
    mean_error = np.sqrt(2.6124858264243187)
    pipeline = make_pipeline(
        StackingEstimator(estimator=RidgeCV()),
        MinMaxScaler(),
        StackingEstimator(estimator=LinearSVR(C=0.001, dual=False, epsilon=0.1, loss="squared_epsilon_insensitive", tol=0.1)),
        SelectPercentile(score_func=f_regression, percentile=96),
        KNeighborsRegressor(n_neighbors=4, p=2, weights="distance")
    )

    set_param_recursive(pipeline.steps, 'random_state', 45)
    return mean_error, pipeline


def predict(file: str, force_cache_update: bool = False) -> None:
    random.seed(42)
    df = init_dataframe(f'{PROJECT_DIR}/cache/dataframe.csv', force_cache_update)

    date_mean_error, date_pipeline = get_date_pipeline()
    price_mean_error, price_pipeline = get_price_pipeline()

    # select only pre-top rows (inclusive)
    df = df[(df['DaysSinceIsBottom'] < df['DaysSinceIsTop']) | (df['DaysSinceIsTop'] == 0)]
    df_train = df.dropna()

    x_date_cols = [
        '2YMAPercentageCycle',
        'PiLogDifferenceCycle',
        'GoldenRatioAbs',
        'PuellMultipleCycle',
    ]
    y_date_col = 'TopDateBasedPercentage'
    df_date_predict = df.dropna(subset=x_date_cols).loc[df[y_date_col].isna()]

    # tpot = TPOTRegressor(
    #     generations=1000,
    #     population_size=100,
    #     n_jobs=15,
    #     random_state=45,
    #     config_dict='TPOT light',
    #     memory='auto',
    #     periodic_checkpoint_folder='cp_date',
    #     verbosity=2,
    # )
    #
    # tpot.fit(df_train[x_date_cols], df_train[y_date_col])
    # exit(0)

    date_pipeline.fit(df_train[x_date_cols], df_train[y_date_col])
    date_predict = date_pipeline.predict(df_date_predict[x_date_cols])
    date_predict_dev = date_predict - date_mean_error
    date_predict_value = df_date_predict['DaysSinceIsBottom'] / (date_predict / 100) - df_date_predict['DaysSinceIsBottom']
    date_predict_value_dev = np.abs(date_predict_value - (df_date_predict['DaysSinceIsBottom'] / (date_predict_dev / 100) - df_date_predict['DaysSinceIsBottom']))

    date_predict_window_half_size = 5
    date_predict_value_mean = date_predict_value.tail(date_predict_window_half_size * 2 + 1).mean() - date_predict_window_half_size
    date_predict_value_dev_mean = date_predict_value_dev.tail(date_predict_window_half_size * 2 + 1).mean()

    x_price_cols = x_date_cols + [y_date_col]
    y_price_col = 'TopPriceBasedPercentage'
    df_price_predict = df.dropna(subset=x_date_cols).loc[df[y_date_col].isna()]
    df_price_predict.loc[df[y_date_col].isna(), y_date_col] = date_predict

    price_pipeline.fit(df_train[x_price_cols], df_train[y_price_col])
    price_predict = price_pipeline.predict(df_price_predict[x_price_cols])
    price_predict_dev = price_predict - price_mean_error
    price_predict_value = (df_price_predict['Price'] - df_price_predict['BottomPrice']) / (price_predict / 100)
    price_predict_value_dev = np.abs(price_predict_value - (df_price_predict['Price'] - df_price_predict['BottomPrice']) / (price_predict_dev / 100))

    current_price = df.tail(1)['Price'].values[0]
    current_date = df.tail(1)['Date'].values[0]
    current_timestamp = int(current_date.astype('uint64') / 1e9)

    result = {
        'current_price': current_price,
        'predicted_price': round(price_predict_value.values[-1]),
        'predicted_price_dev': round(price_predict_value_dev.values[-1]),
        'current_date': current_timestamp,
        'predicted_date': round(date_predict_value_mean),
        'predicted_date_dev': round(date_predict_value_dev_mean),
    }

    with open(file, 'w+') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    fire.Fire(predict)
