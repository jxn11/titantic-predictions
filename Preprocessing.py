import numpy as np
import pandas as pd
import torch

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

class Preprocessor():

    def __init__(self, df, cat_cols, num_cols, label_col=None):
        clean_cats = self.preproc_cats(df, cat_cols)
        norm_nums = self.preproc_nums(df, num_cols)
        all_data = self.combine(clean_cats, norm_nums)

        self.X = self.impute_nans(all_data)
        self.y = self.set_y(df, label_col)

        self.pos_weight = self.calc_weight(df, label_col)
    
    def preproc_cats(self, df, cat_cols):
        clean_cats = pd.DataFrame()
        for col in cat_cols:
            new_col = self._gen_one_hot_var(df, col)
            clean_cats = pd.concat([clean_cats, new_col], axis=1)
        return clean_cats

    def preproc_nums(self, df, num_cols):
        norm_nums = pd.DataFrame()
        for col in num_cols:
            new_col = self._normalize(df, col)
            norm_nums = pd.concat([norm_nums, new_col], axis=1)
        return norm_nums

    def impute_nans(self, df):
        imputer = KNNImputer()
        clean_data = pd.DataFrame(columns=df.columns,
                                  data=imputer.fit_transform(df))
        return clean_data

    def combine(self, clean_cats, norm_nums):
        return pd.concat([clean_cats, norm_nums], axis=1)

    def set_y(self, df, label_col):
        return df[[label_col]] if label_col else None

    def calc_weight(self, df, label_col):
        if not label_col:
            return None
        else:
            num_pos = np.sum(df[label_col])
            num_neg = df.shape[0] - num_pos
            return torch.Tensor([(num_neg/num_pos)])

    def get_splits(self, ts=0.2, rs=13):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=ts,
                                                            random_state=rs)
        return X_train, X_test, y_train, y_test


    def _gen_one_hot_var(self, df, col):
        return pd.get_dummies(df[col])

    def _normalize(self, df, col):
        return (df[col] - np.mean(df[col])) / np.std(df[col])