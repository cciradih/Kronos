import json
import asyncio
import logging
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
import matplotlib.pyplot as plt
from model import Kronos, KronosTokenizer, KronosPredictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KalmanFilter:
    def __init__(
        self, initial_state, initial_covariance, process_noise, measurement_noise
    ):
        self.x = np.array([[initial_state], [0]])
        self.P = np.array([[initial_covariance, 0], [0, initial_covariance]])
        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.array([[1, 0]])
        self.R = np.array([[measurement_noise]])
        self.Q = np.array([[process_noise, 0], [0, process_noise * 0.1]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0]

    def update(self, measurement, dynamic_R=None):
        if dynamic_R is not None:
            r_val = max(dynamic_R, 1e-6)
            self.R = np.array([[r_val]])

        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return self.x[0, 0]


class Bot:
    def __init__(self, config):
        self.exchange = ccxt.okx(config["ccxt"])
        self.bot = config["bot"]
        self.predictor = None

    def init(self):
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        self.predictor = KronosPredictor(model, tokenizer, max_context=512)

    async def fetch_ohlcv(self):
        all_ohlcv = []
        since = None
        for _ in range(2):
            ohlcv = []
            try:
                ohlcv = await self.exchange.fetch_ohlcv(
                    self.bot["symbol"],
                    self.bot["timeframe"],
                    since=since,
                    limit=200,
                )
            except Exception as e:
                logger.error(f"fetch_ohlcv: {e}")
            if len(ohlcv) > 0:
                all_ohlcv += ohlcv
                since = ohlcv[0][0]
        return all_ohlcv

    def generate_future_timestamps(self, start_ts, count, timeframe):
        freq_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
        }
        freq = freq_map.get(timeframe, "1min")
        ts_index = pd.date_range(start=start_ts, periods=count + 1, freq=freq)[1:]
        return pd.Series(ts_index)

    def plot_prediction(self, history_df, pred_df_list, kalman_series):
        width_px, height_px = 1920, 1080
        pixel_dpi = 100

        plt.style.use("dark_background")

        plt.figure(figsize=(width_px / pixel_dpi, height_px / pixel_dpi), dpi=pixel_dpi)

        plt.plot(
            history_df["timestamp"],
            history_df["close"],
            label="History",
            color="#d500f9",
        )

        last_history_ts = history_df["timestamp"].iloc[-1]
        last_history_close = history_df["close"].iloc[-1]

        for pdf in pred_df_list:
            combined_pred_ts = [last_history_ts] + pdf.index.tolist()
            combined_pred_val = [last_history_close] + pdf["close"].tolist()

            plt.plot(
                combined_pred_ts,
                combined_pred_val,
                label="Prediction",
                color="#00bce5",
                linestyle="--",
                alpha=0.25,
            )

        if kalman_series is not None:
            kf_ts = [last_history_ts] + pred_df_list[0].index.tolist()
            kf_val = [last_history_close] + kalman_series.tolist()
            plt.plot(
                kf_ts,
                kf_val,
                label="Kalman Consensus",
                color="#00bce5",
            )

        plt.ylabel("Close")
        plt.title(f"{self.bot['symbol']} {self.bot['timeframe']}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig("prediction.png", dpi=pixel_dpi)
        plt.close()

    def apply_kalman_ensemble(self, pred_df_list, initial_price):
        data_matrix = np.zeros((len(pred_df_list[0]), len(pred_df_list)))

        for i, df in enumerate(pred_df_list):
            data_matrix[:, i] = df["close"].values

        ensemble_mean = np.mean(data_matrix, axis=1)
        ensemble_var = np.var(data_matrix, axis=1)

        kf = KalmanFilter(
            initial_state=initial_price,
            initial_covariance=1.0,
            process_noise=0.01,
            measurement_noise=1.0,
        )

        filtered_results = []

        for t in range(len(ensemble_mean)):
            kf.predict()
            smoothed_val = kf.update(
                measurement=ensemble_mean[t], dynamic_R=ensemble_var[t]
            )
            filtered_results.append(smoothed_val)

        return pd.Series(filtered_results)

    async def loop(self):
        self.init()
        while True:
            ohlcv = await self.fetch_ohlcv()
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df = df.drop_duplicates(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            lookback = 400
            pred_len = 120

            x_df = df.tail(lookback)[["open", "high", "low", "close", "volume"]]
            x_timestamp = df.tail(lookback)["timestamp"]

            last_ts = x_timestamp.iloc[-1]
            last_close = x_df["close"].iloc[-1]
            y_timestamp = self.generate_future_timestamps(
                last_ts, pred_len, self.bot["timeframe"]
            )

            df_list = []
            x_timestamp_list = []
            y_timestamp_list = []
            for _ in range(5):
                df_list.append(x_df)
                x_timestamp_list.append(x_timestamp)
                y_timestamp_list.append(y_timestamp)

            # temperature   1.2-1.5
            # topP          0.95-1.0
            # sampleCount   2-3
            pred_df_list = self.predictor.predict_batch(
                df_list=df_list,
                x_timestamp_list=x_timestamp_list,
                y_timestamp_list=y_timestamp_list,
                pred_len=pred_len,
                T=self.bot["temperature"],
                top_p=self.bot["topP"],
                sample_count=self.bot["sampleCount"],
                verbose=False,
            )

            for pdf in pred_df_list:
                pdf.index = y_timestamp

            kalman_series = self.apply_kalman_ensemble(pred_df_list, last_close)

            self.plot_prediction(df, pred_df_list, kalman_series)

            await asyncio.sleep(5)


async def main():
    try:
        with open(".env", "r") as f:
            config = json.load(f)
            bot = Bot(config)
            await bot.loop()
    except Exception as e:
        logger.error(f"main: {e}")
    finally:
        await bot.exchange.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
