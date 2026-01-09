import sys
import json
import aiohttp
import asyncio
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
from datetime import datetime, timedelta

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


class Bot:
    def __init__(self, config):
        self.exchange = ccxt.okx(config["ccxt"])
        self.bot = config["bot"]
        self.predictor = None

    async def init_exchange(self):
        try:
            await self.exchange.load_markets()
            await self.exchange.set_leverage(
                self.bot["leverage"], self.bot["symbol"], {"marginMode": "cross"}
            )
            try:
                await self.exchange.set_position_mode(False)
            except Exception:
                pass

            logger.info("Initialized.")
        except Exception as e:
            logger.error(f"init_exchange: {e}")
            sys.exit(1)

    def calculate_indicators(self, df):
        try:
            short_df = df.tail(300).copy()
            short_vp = ta.vp(
                close=short_df["close"], volume=short_df["volume"], width=24
            )
            short_poc = short_vp.loc[short_vp["total_volume"].idxmax()]["mean_close"]

            long_vp = ta.vp(close=df["close"], volume=df["volume"], width=24)
            long_poc = long_vp.loc[long_vp["total_volume"].idxmax()]["mean_close"]

            atr = ta.atr(df["high"], df["low"], df["close"]).iloc[-1]

            ema = ta.ema(df["close"], length=144).iloc[-1]

            close = df["close"].iloc[-1]

            return {
                "short_poc": short_poc,
                "long_poc": long_poc,
                "atr": atr,
                "ema": ema,
                "close": close,
            }
        except Exception as e:
            logger.error(f"calculate_indicators: {e}")
            return None

    async def fetch_data(self):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                self.bot["symbol"],
                self.bot["timeframe"],
                params={
                    "paginate": True,
                    "paginationCalls": 4,
                },
            )
            if not ohlcv or len(ohlcv) != 800:
                return None
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df[-601:-1]
        except Exception as e:
            logger.error(f"fetch_data: {e}")
            return None

    async def fetch_ticker(self):
        try:
            ticker = await self.exchange.fetch_ticker(self.bot["symbol"])
            return ticker["bid"], ticker["ask"]
        except Exception as e:
            logger.error(f"fetch_ticker: {e}")
            return None, None, None

    async def manage_orders(self):
        try:
            open_orders = await self.exchange.fetch_open_orders(self.bot["symbol"])
            if open_orders:
                logger.info(f"open_orders: {len(open_orders)}")
                for order in open_orders:
                    await self.exchange.cancel_order(order["id"], self.bot["symbol"])
                return True
            return False
        except Exception as e:
            logger.error(f"manage_orders: {e}")
            return False

    async def fetch_position(self):
        try:
            positions = await self.exchange.fetch_positions([self.bot["symbol"]])
            for position in positions:
                if position["contracts"] > 0:
                    return position
            return None
        except Exception as e:
            logger.error(f"fetch_position: {e}")
            return None

    async def create_order(self, side, price, amount, tp_price, sl_price):
        try:
            symbol = self.bot["symbol"]
            price = self.exchange.price_to_precision(symbol, price)
            amount = self.exchange.amount_to_precision(symbol, amount)
            tp_price = self.exchange.price_to_precision(symbol, tp_price)
            sl_price = self.exchange.price_to_precision(symbol, sl_price)

            params = {
                "postOnly": True,
                "takeProfit": {
                    "triggerPrice": tp_price,
                    "price": tp_price,
                    "type": "limit",
                },
                "stopLoss": {
                    "triggerPrice": sl_price,
                    "type": "market",
                },
            }

            return await self.exchange.create_order(
                symbol, "limit", side, amount, price, params
            )
        except Exception as e:
            logger.error(f"create_order: {e}")
            return None

    def is_funding_time(self):
        now = datetime.now()
        if now.minute > 50 or now.minute < 10:
            if now.hour in [0, 8, 16, 24]:
                return True
        return False

    async def fetch_balance(self):
        try:
            return await self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"fetch_balance: {e}")
            return None

    async def sleep_until_next_candle(self):
        now = datetime.now()
        next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        sleep_seconds = (next_run - now).total_seconds()
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)

    async def loop(self):
        await self.init_exchange()

        while True:
            try:
                if self.is_funding_time():
                    await asyncio.sleep(60)
                    continue

                has_cancelled = await self.manage_orders()
                if has_cancelled:
                    continue

                position = await self.fetch_position()
                if position:
                    continue

                df = await self.fetch_data()
                if df is None:
                    continue

                indicators = self.calculate_indicators(df)
                if indicators is None:
                    continue

                close = indicators["close"]
                ema = indicators["ema"]
                atr = indicators["atr"]
                long_poc = indicators["long_poc"]
                short_poc = indicators["short_poc"]

                atr_multiplier = self.bot["atrMultiplier"]
                high_target = close + atr * atr_multiplier
                low_target = close - atr * atr_multiplier

                signal_side = None

                if close > ema:
                    if long_poc < high_target and short_poc < high_target:
                        signal_side = None
                    else:
                        signal_side = "buy"
                else:
                    if long_poc > low_target and short_poc > low_target:
                        signal_side = None
                    else:
                        signal_side = "sell"

                if signal_side is None:
                    logger.info("No signal generated.")
                    await self.sleep_until_next_candle()
                    continue

                bid, ask = await self.fetch_ticker()
                entry_price = bid if signal_side == "buy" else ask

                tp_price = (
                    entry_price + (atr * atr_multiplier)
                    if signal_side == "buy"
                    else entry_price - (atr * atr_multiplier)
                )
                sl_price = (
                    entry_price - (atr * atr_multiplier)
                    if signal_side == "buy"
                    else entry_price + (atr * atr_multiplier)
                )

                balance = await self.exchange.fetch_balance()
                free_usdt = balance["USDT"]["free"]
                use_usdt = free_usdt * self.bot["useRate"]

                amount = (use_usdt * self.bot["leverage"]) / entry_price

                # order = await self.create_order(
                #     signal_side, entry_price, amount, tp_price, sl_price
                # )
                # if order is not None:
                #     msg = f"{order["id"]}: {signal_side} | {entry_price} | {amount:.2f}"
                # await Messenger.send_feishu(msg, self.bot["feishuUrl"])

                await self.sleep_until_next_candle()
            except Exception as e:
                logger.error(f"loop: {e}")


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
