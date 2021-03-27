import json
import os
import random
from collections import deque
from copy import copy
import fire
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LassoLarsCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MaxAbsScaler, Normalizer, PolynomialFeatures
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

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

    df['CyclePow'] = np.power(4, df['Cycle'])
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


def predict(file: str, force_cache_update: bool = False) -> None:
    random.seed(42)
    df = init_dataframe(f'{PROJECT_DIR}/cache/dataframe.csv', force_cache_update)

    x_cols = [
        # 'PriceIncreaseCycle', 'PriceIncreaseCycleLog',
        '2YMAPercentage',
        'PiLogDifference',
        'GoldenRatioAbs',
        'PuellMultiple',
        # 'SFDifference'
    ]
    y_date_col = 'TopDateBasedPercentage'
    y_price_col = 'TopPriceBasedPercentage'

    # Select only pre-top rows (inclusive)
    df = df[(df['DaysSinceIsBottom'] < df['DaysSinceIsTop']) | (df['DaysSinceIsTop'] == 0)]

    df_train = df.dropna()
    df_predict = df.dropna(subset=x_cols)
    df_predict = df_predict.loc[df[y_date_col].isna()]

    # TODO: visualisations
    # sns.histplot(df_train, x=y_price_col, bins=50)
    # plt.show()

    df_train_price = drop_repeated_bins(df_train, y_price_col, np.arange(0, 100.1, 2.01), 10)

    # tpot = TPOTRegressor(
    #     generations=500,
    #     population_size=100,
    #     n_jobs=-1,
    #     random_state=43,
    #     periodic_checkpoint_folder='cp_price',
    #     verbosity=2,
    # )
    #
    # tpot.fit(df_train_price[x_cols], df_train_price[y_price_col])
    # exit(0)

    price_mean_error = np.sqrt(7.4994000810773445)
    price_pipeline = make_pipeline(
        make_union(
            make_pipeline(
                make_union(
                    FunctionTransformer(copy),
                    make_pipeline(
                        make_union(
                            FunctionTransformer(copy),
                            FunctionTransformer(copy)
                        ),
                        StackingEstimator(estimator=LinearSVR(C=0.5, dual=False, epsilon=0.01, loss="squared_epsilon_insensitive", tol=0.1)),
                        MaxAbsScaler(),
                        Normalizer(norm="max")
                    )
                ),
                Nystroem(gamma=0.15000000000000002, kernel="additive_chi2", n_components=10)
            ),
            Nystroem(gamma=0.30000000000000004, kernel="poly", n_components=9)
        ),
        LinearSVR(C=5.0, dual=True, epsilon=0.01, loss="epsilon_insensitive", tol=0.01)
    )

    date_mean_error = np.sqrt(0.03471098570562975)
    date_pipeline = make_pipeline(
        SelectPercentile(score_func=f_regression, percentile=16),
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        Nystroem(gamma=0.8500000000000001, kernel="sigmoid", n_components=10),
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.001, loss="huber", max_depth=5, max_features=0.2, min_samples_leaf=10, min_samples_split=11, n_estimators=100,
                                                              subsample=0.6500000000000001)),
        LassoLarsCV(normalize=True)
    )

    set_param_recursive(price_pipeline.steps, 'random_state', 43)
    set_param_recursive(date_pipeline.steps, 'random_state', 42)

    price_pipeline.fit(df_train[x_cols], df_train[y_price_col])
    date_pipeline.fit(df_train[x_cols], df_train[y_date_col])

    price_predict = price_pipeline.predict(df_predict[x_cols])
    price_predict_dev = price_predict - price_mean_error
    price_predict_value = (df_predict['Price'] - df_predict['BottomPrice']) / (price_predict / 100)
    price_predict_value_dev = np.abs(price_predict_value - (df_predict['Price'] - df_predict['BottomPrice']) / (price_predict_dev / 100))

    date_predict = date_pipeline.predict(df_predict[x_cols])
    date_predict_dev = date_predict - date_mean_error
    date_predict_value = df_predict['DaysSinceIsBottom'] / (date_predict / 100) - df_predict['DaysSinceIsBottom']
    date_predict_value_dev = np.abs(date_predict_value - (df_predict['DaysSinceIsBottom'] / (date_predict_dev / 100) - df_predict['DaysSinceIsBottom']))

    date_predict_window_half_size = 5
    date_predict_value_mean = date_predict_value.tail(date_predict_window_half_size * 2 + 1).mean() - date_predict_window_half_size
    date_predict_value_dev_mean = date_predict_value_dev.tail(date_predict_window_half_size * 2 + 1).mean()

    current_price = df_predict.tail(1)['Price'].values[0]
    current_date = df_predict.tail(1)['Date'].values[0]
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
