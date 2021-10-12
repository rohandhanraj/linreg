import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import logging

def data_preprocessing(df, target, cols, random_state=1, test_size=0.33):
    X = df.drop([*cols, target], axis=1)
    y = df[target]
    logging.info(f'X:\n{X.head()}')
    logging.info(f'y:\n{y.head()}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
    logging.info(f'X_train:\n{X_train.head()}')
    logging.info(f'y_train:\n{y_train.head()}')
    logging.info(f'X_test:\n{X_test.head()}')
    logging.info(f'y_test:\n{y_test.head()}')

    return (X_train, y_train), (X_test, y_test)