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
    'order_id': np.int32,
    'eval_set': str,
    'reordered': np.float32,
    'up_order_count': np.float32,
    'up_first_order_number': np.float32,
    'up_last_order_number': np.float32,
    'up_average_cart_position': np.float32,
    'up_days_since_last_order': np.float32,
    'prod_total_cnt': np.float32,
    'prod_reorder_total_cnt': np.float32,
    'prod_user_cnt': np.float32,
    'prod_return_user_cnt': np.float32,
    'prod_user_reorder_ratio': np.float32,
    'prod_product_reorder_ratio': np.float32,
    'user_total_orders': np.float32,
    'user_sum_days_since_prior_order': np.float32,
    'user_mean_days_since_prior_order': np.float32,
    'user_reorder_ratio': np.float32,
    'user_total_products': np.float32,
    'user_distinct_products': np.float32,
    'user_average_basket': np.float32,
    'days_since_prior_order': np.float32,
    'cat_total_bought_cnt': np.float32,
    'cat_reorder_total_cnt': np.float32,
    'cat_user_cnt': np.float32,
    'cat_return_user_cnt': np.float32,
    'cat_user_reorder_ratio': np.float32,
    'cat_product_reorder_ratio': np.float32,
    'cat_num_of_prods_a_user_buys_in_this_cat_mean': np.float32,
    'cat_num_of_prods_a_user_buys_in_this_cat_std': np.float32,
    'cat_num_of_prods_a_user_buys_in_this_cat_max': np.float32,
    'up_order_rate': np.float32,
    'up_order_since_last_order': np.float32,
    'up_order_rate_since_first_order': np.float32,
    'prod_market_share_hod': np.float32,
    'prod_market_share_dow': np.float32,
    'up_days_since_last_not_order': np.float32,
    'up_order_since_last_not_order': np.float32
}

pe = pd.read_pickle(os.path.join(root, 'abt-share', 'feature.product_embeddings.pkl'))
pe = pe[['product_id'] + [i for i in range(32)]]
pe.columns = [str(c) for c in pe.columns]

def product_embeddings():
    global pe
    return pe

sb_features = [
    'user_product_reordered_ratio',
    'reordered_sum',
    'add_to_cart_order_inverted_mean',
    'add_to_cart_order_relative_mean',
    'dep_reordered_ratio',
    'aisle_reordered_ratio',
    'aisle_products',
    'aisle_reordered',
    'dep_products',
    'dep_reordered',
    'order_dow', 'order_hour_of_day',
    'user_reorder_ratio',
    'days_since_prior_order_mean',
    'order_dow_mean',
    'order_hour_of_day_mean',
] + [str(i) for i in range(32)] + ['last', 'prev1', 'prev2', 'median', 'mean']

def up_interval_stats(augname=None):
    if augname is None:
        dfs = []
        for sid in range(64):
            dfs.append(pd.read_pickle(os.path.join(root, 'abt-share', 'feature.up_interval_stat.orig{}.pkl'.format(sid))))
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_pickle(os.path.join(root, 'abt', 'feature.up_interval_stat.{}.pkl'.format(augname)))
    return df

def sb_prune(df):
    df = df[sb_features + ['order_id', 'product_id']]

    for col in sb_features:
        df[col] = df[col].astype(np.float32)

    names = {}
    for old_name in sb_features:
        names[old_name] = "sb_{}".format(old_name)
    df.rename(columns=names, inplace=True)

    return df

def sb_train(augname=None):
    if augname is None:
        dfs = []
        for sid in range(64):
            df = pd.read_pickle(os.path.join(root, 'abt-share', 'abt_train.sb.orig{}.pkl'.format(sid)))
            dfs.append(df)
        train = pd.concat(dfs, ignore_index=True)
    else:
        train = pd.read_pickle(os.path.join(root, 'abt', 'abt_train.sb.{}.pkl'.format(augname)))

    train = train.merge(up_interval_stats(augname), on=['product_id', 'user_id'], how='left')
    train = train.merge(product_embeddings(), on='product_id', how='left')
    return sb_prune(train)

def sb_test():
    dfs = []
    for sid in range(64):
        df = pd.read_pickle(os.path.join(root, 'abt-share', 'abt_test.sb.orig{}.pkl'.format(sid)))
        dfs.append(df)
    test = pd.concat(dfs, ignore_index=True)
    test = test.merge(up_interval_stats(), on=['product_id', 'user_id'], how='left')
    test = test.merge(product_embeddings(), on='product_id', how='left')
    return sb_prune(test)

class Data:
    @staticmethod
    def train_aug(down_sample=None):
        dfs = []
        for s in range(4):
            for ms in range(52):
                augname = 'aug{}-{}'.format(s, ms)
                df = pd.read_csv(
                    os.path.join(root, 'abt', 'abt_train.{}.csv'.format(augname)),
                    dtype=schema)
                if down_sample is not None:
                    df = df[df.order_id % down_sample == 0]
                df = df.merge(sb_train(augname=augname), how='left', on=['order_id', 'product_id'])
                dfs.append(df)
        return dfs

    @staticmethod
    def train(down_sample=None, aug=False):
        train = pd.read_csv(
            os.path.join(root, 'abt-share', 'abt_train.csv'),
            dtype=schema)
        if down_sample is not None:
            train = train[train.order_id % down_sample == 0]
        train = train.merge(sb_train(augname=None), how='left', on=['order_id', 'product_id'])

        if aug:
            train_aug = Data.train_aug(down_sample=down_sample)
            train = pd.concat([train] + train_aug)

        train.loc[:, 'reordered'] = train.reordered.fillna(0)
        train.sort_index(axis=1, inplace=True)

        return train

    @staticmethod
    def train_val(down_sample=None, test_size=0.2, aug=False):
        train = Data.train(down_sample=down_sample, aug=aug)

        X_train, X_val, y_train, y_val = train_test_split(
            train.drop(['eval_set', 'product_id', 'order_id', 'reordered'], axis=1),
            train.reordered,
            test_size=test_size, random_state=1019)

        return (X_train, X_val, y_train, y_val)

    @staticmethod
    def dtrain_dval(down_sample=None, test_size=0.2, aug=False):
        X_train, X_val, y_train, y_val = Data.train_val(down_sample=down_sample, test_size=test_size, aug=aug)

        feature_names = X_train.columns.format()

        PANDAS_DTYPE_MAPPER = {'int8': 'int', 'int16': 'int', 'int32': 'int', 'int64': 'int',
                       'uint8': 'int', 'uint16': 'int', 'uint32': 'int', 'uint64': 'int',
                       'float16': 'float', 'float32': 'float', 'float64': 'float',
                       'bool': 'i'}
        feature_types = [PANDAS_DTYPE_MAPPER[dtype.name] for dtype in X_train.dtypes]

        X_train = X_train.values.astype(np.float32)
        X_val = X_val.values.astype(np.float32)
        gc.collect()

        dtrain = xgboost.DMatrix(
            data=X_train, label=y_train,
            feature_names=feature_names,
            feature_types=feature_types)

        dval = xgboost.DMatrix(
            data=X_val, label=y_val,
            feature_names=feature_names,
            feature_types=feature_types)

        gc.collect()
        return (dtrain, dval)

    @staticmethod
    def test(down_sample=None):
        test = pd.read_csv(
            os.path.join(root, 'abt-share', 'abt_test.csv'),
            dtype=schema)

        if down_sample is not None:
            test = test[test.order_id % down_sample == 0]

        test = test.merge(sb_test(), how='left', on=['order_id', 'product_id'])
        test.sort_index(axis=1, inplace=True)

        return test
