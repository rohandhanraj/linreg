from src.model import LinearRegressor
from src.data_mgmt import *
import os
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir,'running_logs.log'), level=logging.INFO, format=logging_str)


df = pd.read_csv('https://raw.githubusercontent.com/rohan-dhanraj/Datasets/main/Advertising.csv')

train, test = data_preprocessing(df, target='sales', random_state=42, test_size=0.2, cols=('Unnamed: 0',))

model = LinearRegressor(learning_rate=0.1, n_iter=1000)
model.fit(train[0], train[1])
y_pred = model.predict(test[0])
rmse = model.rmse(y_pred, test[1])
r2 = model.score(test[0], test[1])
adj_r2 = model.adj_r2(train[0], train[1])

logging.info(f'RMSE: {rmse}')
logging.info(f'R-squared: {r2}')
logging.info(f'ADjusted R-square: {adj_r2}')
