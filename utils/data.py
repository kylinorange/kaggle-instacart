import numpy as np
np.random.seed(1019)

import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split

import os

root_paths = [
    "/data/kaggle-instacart",
    "/Users/jiayou/Dropbox/珺珺的程序/Kaggle/Instacart",
    "/Users/jiayou/Dropbox/Documents/珺珺的程序/Kaggle/Instacart"
]
root = None
for p in root_paths:
    if os.path.exists(p):
        root = p
        break

schema = {
    'product_id': np.int32,
    'up_order_count': np.int16,
    'up_first_order_number': np.int16,
    'up_last_order_number': np.int16,
    'up_average_cart_position': np.float32,
    'up_days_since_last_order': np.float32,
    'prod_total_cnt': np.int32,
    'prod_reorder_total_cnt': np.float32,
    'prod_user_cnt': np.int32,
    'prod_return_user_cnt': np.int32,
    'prod_user_reorder_ratio': np.float32,
    'prod_product_reorder_ratio': np.float32,
    'user_total_orders': np.int16,
    'user_sum_days_since_prior_order': np.float32,
    'user_mean_days_since_prior_order': np.float32,
    'user_reorder_ratio': np.float32,
    'user_total_products': np.int16,
    'user_distinct_products': np.int16,
    'user_average_basket': np.float32,
    'order_id': np.int32,
    'eval_set': str,
    'days_since_prior_order': np.float32,
    'cat_total_bought_cnt': np.int32,
    'cat_reorder_total_cnt': np.float32,
    'cat_user_cnt': np.int32,
    'cat_return_user_cnt': np.int32,
    'cat_user_reorder_ratio': np.float32,
    'cat_product_reorder_ratio': np.float32,
    'cat_num_of_prods_a_user_buys_in_this_cat_mean': np.float32,
    'cat_num_of_prods_a_user_buys_in_this_cat_std': np.float32,
    'cat_num_of_prods_a_user_buys_in_this_cat_max': np.int16,
    'up_order_rate': np.float32,
    'up_order_since_last_order': np.int16,
    'up_order_rate_since_first_order': np.float32,
    'reordered': np.float32,
    'prod_market_share_hod': np.float32,
    'prod_market_share_dow': np.float32
}

class Data:
    @staticmethod
    def train_aug0(down_sample=None):
        dfs = []
        for s in range(4):
            for ms in range(52):
                df = pd.read_csv(
                    os.path.join(root, 'abt', 'abt_train.aug{}-{}.csv'.format(s, ms)),
                    dtype=schema)
                if down_sample is not None:
                    df = df[df.order_id % down_sample == 0]
                df.loc[:, 'reordered'] = df.reordered.fillna(0)
                dfs.append(df)
        return pd.concat(dfs)

    @staticmethod
    def dtrain0(down_sample=None, test_size=0.2, aug = False):
        train = pd.read_csv(
            os.path.join(root, 'abt', 'abt_train.csv'),
            dtype=schema)

        if down_sample is not None:
            train = train[train.order_id % down_sample == 0]

        train.loc[:, 'reordered'] = train.reordered.fillna(0)

        X_train, X_val, y_train, y_val = train_test_split(
            train.drop(['eval_set', 'product_id', 'order_id', 'reordered'], axis=1),
            train.reordered,
            test_size=test_size, random_state=1019)

        if aug:
            train_aug = Data.train_aug0(down_sample=down_sample)
            X_train = pd.concat([X_train, train_aug.drop(['eval_set', 'product_id', 'order_id', 'reordered'], axis=1)])
            y_train = pd.concat([y_train, train_aug.reordered])

        Data.random_feature(X_train)
        Data.random_feature(X_val)

        X_train.sort_index(axis=1, inplace=True)
        X_val.sort_index(axis=1, inplace=True)

        dtrain = xgboost.DMatrix(X_train, y_train)
        dval = xgboost.DMatrix(X_val, y_val)

        return (dtrain, dval)

    @staticmethod
    def train_aug(down_sample=None):
        dfs = []
        for s in range(4):
            for ms in range(52):
                df = pd.read_csv(
                    os.path.join(root, 'abt', 'abt_train.aug{}-{}.csv'.format(s, ms)),
                    dtype=schema)
                if down_sample is not None:
                    df = df[df.order_id % down_sample == 0]
                dfs.append(df)
        return dfs

    @staticmethod
    def random_feature(data):
        n = data.shape[0]
        data['rand_uniform'] = np.random.uniform(0, 1, n).astype(np.float16)
        data['rand_normal'] = np.random.normal(0, 1, n).astype(np.float16)

    @staticmethod
    def dtrain(down_sample=None, test_size=0.2, aug = False):
        train = pd.read_csv(
            os.path.join(root, 'abt', 'abt_train.csv'),
            dtype=schema)

        if down_sample is not None:
            train = train[train.order_id % down_sample == 0]

        if aug:
            train_aug = Data.train_aug(down_sample=down_sample)
            train = pd.concat([train] + train_aug)

        train.loc[:, 'reordered'] = train.reordered.fillna(0)
        Data.random_feature(train)

        X_train, X_val, y_train, y_val = train_test_split(
            train.drop(['eval_set', 'product_id', 'order_id', 'reordered'], axis=1),
            train.reordered,
            test_size=test_size, random_state=1019)

        dtrain = xgboost.DMatrix(X_train, y_train)
        dval = xgboost.DMatrix(X_val, y_val)

        return (dtrain, dval)

    @staticmethod
    def test(down_sample=None):
        data = pd.read_csv(
            os.path.join(root, 'abt_test.csv'),
            dtype=schema)

        Data.random_feature(data)

        test = data

        if down_sample is not None:
            test = test[test.order_id % down_sample == 0]

        return test
