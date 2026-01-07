# Ish QT
基于 [Kronos](https://github.com/shiyu-coder/Kronos) 的量化交易程序。

## 配置
将配置文件模板 `copy.env.local` 重命名为 `.env.local` （模拟交易）和 `.env`（实盘交易）。

```json
{
    // ccxt 的相关配置。
    "ccxt": {
        "apiKey": "",
        "secret": "",
        "password": "",
        "enableRateLimit": true,
        "options": {
            "defaultType": "swap"
        },
        "sandbox": true,
        "httpsProxy": "",
        "wssProxy": ""
    },
    "bot": {
        // 市场。
        "symbol": "DOGE/USDT:USDT",
        // 周期。
        "timeframe": "1h",
        // 推荐 1.2-1.5。
        "temperature": 1.2,
        // 推荐 0.95-1.0。
        "topP": 0.95,
        // 推荐 2-3。
        "sampleCount": 2,
        // 批量预测的数量，越多越好。
        // 经测试，GTX 1050 Ti 4G 显存最大可支持 15，RTX 4070 8G 显存最大可支持 22。
        "batch": 22
    },
    // 发送通知到飞书群聊机器人的 Webhook 地址。
    "feishuUrl": ""
}
```

## 安装
推荐使用 Python 3.12。

```bash
# 建立虚拟环境。
python -m venv .venv
# 激活虚拟环境。
.venv\Scripts\Activate.ps1
# GTX 1050 Ti 使用 cu126。
# pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
# RTX 4070 使用 cu130。
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
# 安装其它依赖。
pip install -r requirements.txt
```

## 运行
默认进行模拟交易，使用 `--production` 进行实盘交易。

```bash
# python main.py --production
python main.py
```
