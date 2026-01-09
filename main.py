import sys
import json
import aiohttp
import asyncio
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from model import Kronos, KronosTokenizer, KronosPredictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Messenger:
    @staticmethod
    async def send_feishu(msg, feishu_url):
        logger.info(msg)
        if len(feishu_url) == 0:
            return
        payload = {"msg_type": "text", "content": {"text": msg}}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(feishu_url, json=payload, timeout=5) as resp:
                    if resp.status != 200:
                        logger.error(f"send_feishu: {resp.status}")
        except Exception as e:
            logger.error(f"send_feishu: {e}")


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

    async def init(self):
        # model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        # tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        # self.predictor = KronosPredictor(model, tokenizer, max_context=512)

        try:
            await self.exchange.set_leverage(
                self.bot["leverage"], self.bot["symbol"], {"marginMode": "cross"}
            )
            await self.exchange.set_position_mode(True)
        except Exception as e:
            logger.error(f"init: {e}")

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

    def plot_prediction(
        self,
        df,
        pred_df_list,
        kalman_series,
        angle,
        short_poc_price,
        long_poc_price,
        high_atr,
        low_atr,
        ema,
    ):
        history_df = df.tail(300)

        width_px, height_px = 1920, 1080
        pixel_dpi = 100

        plt.style.use("dark_background")

        fig, ax = plt.subplots(
            figsize=(width_px / pixel_dpi, height_px / pixel_dpi), dpi=pixel_dpi
        )

        ax.plot(
            history_df["timestamp"],
            history_df["close"],
            color="#d500f9",
        )

        last_history_ts = history_df["timestamp"].iloc[-1]
        # last_history_close = history_df["close"].iloc[-1]

        # for pdf in pred_df_list:
        #     combined_pred_ts = [last_history_ts] + pdf.index.tolist()
        #     combined_pred_val = [last_history_close] + pdf["close"].tolist()

        #     ax.plot(
        #         combined_pred_ts,
        #         combined_pred_val,
        #         color="#00bce5",
        #         linestyle="--",
        #         alpha=0.25,
        #     )

        # kf_ts = [last_history_ts] + pred_df_list[0].index.tolist()
        # kf_val = [last_history_close] + kalman_series.tolist()
        # ax.plot(
        #     kf_ts,
        #     kf_val,
        #     color="#00bce5",
        # )

        ax.axhline(y=short_poc_price, color="#DBDBDB", linestyle="--", alpha=0.8)
        ax.text(
            last_history_ts,
            short_poc_price,
            f"short_poc_price: {short_poc_price:.2f}",
            color="#DBDBDB",
            va="bottom",
            ha="right",
            fontsize=12,
        )

        ax.axhline(y=long_poc_price, color="#DBDBDB", linestyle="--", alpha=0.8)
        ax.text(
            last_history_ts,
            long_poc_price,
            f"long_poc_price: {long_poc_price:.2f}",
            color="#DBDBDB",
            va="bottom",
            ha="right",
            fontsize=12,
        )

        ax.axhline(y=high_atr, color="#25a750", linestyle="--", alpha=0.8)
        ax.text(
            last_history_ts,
            high_atr,
            f"high_atr: {high_atr:.2f}",
            color="#25a750",
            va="bottom",
            ha="right",
            fontsize=12,
        )

        ax.axhline(y=low_atr, color="#ca3f64", linestyle="--", alpha=0.8)
        ax.text(
            last_history_ts,
            low_atr,
            f"low_atr: {low_atr:.2f}",
            color="#ca3f64",
            va="bottom",
            ha="right",
            fontsize=12,
        )

        ax.axhline(y=ema, color="#2196f3", linestyle="--", alpha=0.8)
        ax.text(
            last_history_ts,
            ema,
            f"ema: {ema:.2f}",
            color="#2196f3",
            va="bottom",
            ha="right",
            fontsize=12,
        )

        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)

        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_formatter(formatter)

        fig.autofmt_xdate()

        ax.set_ylabel("Close")
        # ax.set_title(
        #     f"{self.bot['symbol']} {self.bot['timeframe']} {angle:.2f} {short_poc_price:.2f} {long_poc_price:.2f} {high_atr:.2f} {low_atr:.2f} {ema:.2f}"
        # )
        ax.set_title(
            f"{self.bot['symbol']} {self.bot['timeframe']} {short_poc_price:.2f} {long_poc_price:.2f} {high_atr:.2f} {low_atr:.2f} {ema:.2f}"
        )
        ax.grid(True, alpha=0.15)

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

    def apply_linear_regression(self, kalman_series):
        try:
            y = np.array(kalman_series)
            y_min = np.min(y)
            y_max = np.max(y)
            y_norm = (y - y_min) / (y_max - y_min)

            x = np.arange(len(kalman_series))
            x_norm = x / len(x)

            slope, _ = np.polyfit(x_norm, y_norm, 1)
            return np.degrees(np.arctan(slope))
        except Exception as e:
            logger.error(f"apply_linear_regression: {e}")
            return 0.0

    def apply_vp(self, df):
        try:
            short_df = df.tail(300)
            short_vp = ta.vp(
                close=short_df["close"], volume=short_df["volume"], width=24
            )
            short_poc_price = short_vp.loc[short_vp["total_volume"].idxmax()][
                "mean_close"
            ]

            long_vp = ta.vp(close=df["close"], volume=df["volume"], width=24)
            long_poc_price = long_vp.loc[long_vp["total_volume"].idxmax()]["mean_close"]

            return short_poc_price, long_poc_price
        except Exception as e:
            logger.error(f"apply_vp: {e}")
            return 0.0

    def apply_atr(self, df):
        try:
            atr = ta.atr(df["high"], df["low"], df["close"]).iloc[-1]
            price = df.iloc[-1]["close"]
            atr_multiplier = self.bot["atrMultiplier"]
            return price, price + atr_multiplier * atr, price - atr_multiplier * atr
        except Exception as e:
            logger.error(f"apply_atr: {e}")
            return 0.0, 0.0

    def apply_ema(self, df):
        try:
            return ta.ema(df["close"], 144).iloc[-1]
        except Exception as e:
            logger.error(f"apply_ema: {e}")
            return 0.0, 0.0

    async def fetch_ohlcv(self):
        try:
            return await self.exchange.fetch_ohlcv(
                self.bot["symbol"],
                self.bot["timeframe"],
                params={
                    "paginate": True,
                    "paginationCalls": 4,
                },
            )
        except Exception as e:
            logger.error(f"fetch_ohlcv: {e}")
            return None

    async def fetch_open_orders(self):
        try:
            return await self.exchange.fetch_open_orders(
                self.bot["symbol"], params={"ordType": "post_only"}
            )
        except Exception as e:
            logger.error(f"fetch_open_order: {e}")
            return None

    async def cancel_order(self, order_id):
        try:
            await self.exchange.cancel_order(order_id, self.bot["symbol"])
        except Exception as e:
            logger.error(f"cancel_order: {e}")
            return None

    async def fetch_position(self):
        try:
            return await self.exchange.fetch_position(self.bot["symbol"])
        except Exception as e:
            logger.error(f"fetch_position: {e}")
            return None

    async def fetch_balance(self):
        try:
            return await self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"fetch_balance: {e}")
            return None

    async def create_order(self, side, amount, price, take_profit, stop_loss):
        try:
            return await self.exchange.create_order(
                self.bot["symbol"],
                "limit",
                side,
                amount,
                price,
                {
                    "positionSide": "long" if side == "buy" else "short",
                    "postOnly": True,
                    "takeProfit": {
                        "triggerPrice": take_profit,
                        "price": take_profit,
                        "type": "limit",
                    },
                    "stopLoss": {
                        "triggerPrice": stop_loss,
                        "type": "market",
                    },
                },
            )
        except Exception as e:
            logger.error(f"create_order: {e}")
            return None

    def idle(self, now):
        hours = [0, 8, 16]
        for h in hours:
            base_time = now.replace(hour=h, minute=0, second=0, microsecond=0)
            before_10m = base_time - timedelta(minutes=10)
            after_10m = base_time + timedelta(minutes=10)
            if before_10m < now < after_10m:
                return True
        return False

    async def loop(self):
        await self.init()

        while True:
            now = datetime.now()

            if self.idle(now):
                continue

            next_run = now.replace(second=0, microsecond=0) + timedelta(
                minutes=1 - (now.minute % 1)
            )
            if next_run < now:
                next_run += timedelta(minutes=1)

            try:
                ohlcv = await self.fetch_ohlcv()
                if len(ohlcv) != 800:
                    continue
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df = df[-601:-1]
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                # lookback = 570
                # pred_len = self.bot["predictionLength"]

                # x_df = df.tail(lookback)[["open", "high", "low", "close", "volume"]]
                # x_timestamp = df.tail(lookback)["timestamp"]

                # last_ts = x_timestamp.iloc[-1]
                # last_close = x_df["close"].iloc[-1]
                # y_timestamp = self.generate_future_timestamps(
                #     last_ts, pred_len, self.bot["timeframe"]
                # )

                # df_list = []
                # x_timestamp_list = []
                # y_timestamp_list = []

                # for _ in range(self.bot["batch"]):
                #     df_list.append(x_df)
                #     x_timestamp_list.append(x_timestamp)
                #     y_timestamp_list.append(y_timestamp)

                # pred_df_list = self.predictor.predict_batch(
                #     df_list=df_list,
                #     x_timestamp_list=x_timestamp_list,
                #     y_timestamp_list=y_timestamp_list,
                #     pred_len=pred_len,
                #     T=self.bot["temperature"],
                #     top_p=self.bot["topP"],
                #     sample_count=self.bot["sampleCount"],
                #     verbose=True,
                # )
                pred_df_list = None

                # for pdf in pred_df_list:
                #     pdf.index = y_timestamp

                # kalman_series = self.apply_kalman_ensemble(pred_df_list, last_close)
                kalman_series = None
                # angle = self.apply_linear_regression(kalman_series)
                angle = None

                short_poc_price, long_poc_price = self.apply_vp(df)
                price, high_atr, low_atr = self.apply_atr(df)
                ema = self.apply_ema(df)

                self.plot_prediction(
                    df,
                    pred_df_list,
                    kalman_series,
                    angle,
                    short_poc_price,
                    long_poc_price,
                    high_atr,
                    low_atr,
                    ema,
                )

                open_orders = await self.fetch_open_orders()
                if open_orders:
                    logger.info(f"open_orders: {len(open_orders)}")
                    await asyncio.sleep(self.bot["orderTimeout"])
                    for order in open_orders:
                        order_id = order["id"]
                        await self.cancel_order(order["id"])
                        logger.info(f"cancel_order: {order_id}")
                    now_after_run = datetime.now()
                    sleep_seconds = (next_run - now_after_run).total_seconds()
                    if sleep_seconds > 0:
                        await asyncio.sleep(sleep_seconds)
                    continue

                position = await self.fetch_position()
                in_position = position is not None and position["contracts"] > 0
                if in_position:
                    logger.info(f"in_position")
                    now_after_run = datetime.now()
                    sleep_seconds = (next_run - now_after_run).total_seconds()
                    if sleep_seconds > 0:
                        await asyncio.sleep(sleep_seconds)
                    continue

                side = "buy" if price > ema else "sell"
                trend_threshold = self.bot["trendThreshold"]
                if side == "buy":
                    # if angle < trend_threshold:
                    #     logger.info(f"angle < {trend_threshold}")
                    #     now_after_run = datetime.now()
                    #     sleep_seconds = (next_run - now_after_run).total_seconds()
                    #     if sleep_seconds > 0:
                    #         await asyncio.sleep(sleep_seconds)
                    #     continue
                    # if price < ema:
                    #     logger.info(f"price < ema")
                    #     now_after_run = datetime.now()
                    #     sleep_seconds = (next_run - now_after_run).total_seconds()
                    #     if sleep_seconds > 0:
                    #         await asyncio.sleep(sleep_seconds)
                    #     continue
                    if long_poc_price < high_atr:
                        logger.info(f"long_poc_price < high_atr")
                        now_after_run = datetime.now()
                        sleep_seconds = (next_run - now_after_run).total_seconds()
                        if sleep_seconds > 0:
                            await asyncio.sleep(sleep_seconds)
                        continue
                    if short_poc_price < high_atr:
                        logger.info(f"short_poc_price < high_atr")
                        now_after_run = datetime.now()
                        sleep_seconds = (next_run - now_after_run).total_seconds()
                        if sleep_seconds > 0:
                            await asyncio.sleep(sleep_seconds)
                        continue
                elif side == "sell":
                    # if angle > -trend_threshold:
                    #     logger.info(f"angle > -{trend_threshold}")
                    #     now_after_run = datetime.now()
                    #     sleep_seconds = (next_run - now_after_run).total_seconds()
                    #     if sleep_seconds > 0:
                    #         await asyncio.sleep(sleep_seconds)
                    #     continue
                    # if price > ema:
                    #     logger.info(f"price > ema")
                    #     now_after_run = datetime.now()
                    #     sleep_seconds = (next_run - now_after_run).total_seconds()
                    #     if sleep_seconds > 0:
                    #         await asyncio.sleep(sleep_seconds)
                    #     continue
                    if long_poc_price > low_atr:
                        logger.info(f"long_poc_price > low_atr")
                        now_after_run = datetime.now()
                        sleep_seconds = (next_run - now_after_run).total_seconds()
                        if sleep_seconds > 0:
                            await asyncio.sleep(sleep_seconds)
                        continue
                    if short_poc_price > low_atr:
                        logger.info(f"short_poc_price > low_atr")
                        now_after_run = datetime.now()
                        sleep_seconds = (next_run - now_after_run).total_seconds()
                        if sleep_seconds > 0:
                            await asyncio.sleep(sleep_seconds)
                        continue
                else:
                    now_after_run = datetime.now()
                    sleep_seconds = (next_run - now_after_run).total_seconds()
                    if sleep_seconds > 0:
                        await asyncio.sleep(sleep_seconds)
                    continue

                leverage = self.bot["leverage"]

                balance = await self.fetch_balance()
                free_usdt = balance["USDT"]["free"]
                use_usdt = free_usdt * self.bot["useRate"]

                symbol = self.bot["symbol"]

                amount = self.exchange.amount_to_precision(
                    symbol, use_usdt * leverage / price
                )

                take_profit = self.exchange.price_to_precision(
                    symbol, high_atr if side == "buy" else low_atr
                )
                stop_loss = self.exchange.price_to_precision(
                    symbol, high_atr if side == "sell" else low_atr
                )

                order = await self.create_order(
                    side, amount, price, take_profit, stop_loss
                )

                msg = f"{order["id"]} {side} {amount} {price} {take_profit} {stop_loss}"
                await Messenger.send_feishu(msg, self.bot["feishuUrl"])
            except Exception as e:
                logger.error(f"loop: {e}")

            now_after_run = datetime.now()
            sleep_seconds = (next_run - now_after_run).total_seconds()
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)


async def main():
    config_file = ".config.json" if "--production" in sys.argv else ".config.local.json"
    try:
        with open(config_file, "r") as f:
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
