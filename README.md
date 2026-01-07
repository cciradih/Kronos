# Ish QT 📈
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

[![Framework](https://img.shields.io/badge/Based%20on-Kronos-orange)](https://github.com/shiyu-coder/Kronos)

**Ish QT** 是一款基于 **Kronos** 时间序列大模型的智能化量化交易程序。它结合了深度学习（Transformer）、统计学滤波与经典技术指标，旨在通过 AI 预测与严谨的数学模型捕捉市场趋势，并实现自动化交易管理。

---

## ✨ 核心特性
本项目不仅依赖 AI 的原始输出，还通过一套多层过滤机制来确保决策的鲁棒性：

*   **🧠 深度推理 (Kronos AI)**：利用 [Kronos Base](https://huggingface.co/NeoQuasar/Kronos-base) 模型回顾历史 **570** 根 K 线，对未来价格走势进行多样本批量生成。
*   **🛡️ 最优估计 (Kalman Filter)**：采用 **卡尔曼滤波** 算法处理批量生成的推理数据，有效滤除随机噪声，提取出最接近真实状态的预测曲线。
*   **📊 趋势量化 (Linear Regression)**：对滤波后的结果进行 **线性回归** 分析，将回归斜率转化为角度，精准判断趋势的强弱。
*   **🎯 精准止盈 (Volume Profile)**：计算成交量分布得到的 **POC (Point of Control)**，以此作为市场公认的价值中心，确定科学的止盈目标位。
*   **🚧 风险控制 (Average True Range)**：基于 **平均真实波幅 (ATR)** 动态计算市场的波动剧烈程度，并以此设定智能止损位。

---

## 🛠️ 环境准备

### 推荐环境
- **Python**: 3.12
- **GPU**: NVIDIA RTX 4070 (8G 显存) 或更高

### 安装步骤

1.  **克隆仓库并进入目录**
    ```bash
    git clone https://github.com/your-username/Ish-QT.git
    cd Ish-QT
    ```

2.  **创建虚拟环境**
    ```bash
    python -m venv .venv
    # Windows (PowerShell)
    .venv\Scripts\Activate.ps1
    # Linux/macOS
    source .venv/bin/activate
    ```

3.  **安装依赖**
    ```bash
    # 安装支持 CUDA 13.0 的 PyTorch (根据你的显卡驱动调整)
    pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130

    # 安装其他必要依赖
    pip install -r requirements.txt
    ```

---

## ⚙️ 配置指南
将配置文件模板 `copy.config.json` 重命名为：
* `.config.local.json`：用于**模拟交易**
* `.config.json`：用于**实盘交易**

### 配置参数详解（暂时仅支持 OKX）

```json5
{
    "ccxt": {
        "apiKey": "YOUR_API_KEY",
        "secret": "YOUR_SECRET",
        "password": "YOUR_PASSWORD",
        "enableRateLimit": true,
        "options": { "defaultType": "swap" }, // 默认合约交易
        "sandbox": true,                     // 是否开启沙盒模拟模式
        "httpsProxy": "",                    // 代理设置
        "wssProxy": ""
    },
    "bot": {
        "symbol": "SOL/USDT:USDT",           // 交易对
        "timeframe": "5m",                   // K 线周期
        "predictionLength": 24,              // 预测步长 (如 5min * 24 = 2小时)
        "temperature": 1.2,                  // 采样温度 (1.2-1.5)
        "topP": 0.95,                        // 核心采样阈值 (0.95-1.0)
        "sampleCount": 2,                    // 采样次数
        "batch": 40,                         // 批处理大小 (RTX 4070 建议 40)
        "feishuUrl": "",                     // 飞书机器人通知地址
        "atrMultiplier": 2.5                 // 止损 ATR 倍率
    }
}
```

---

## 🚀 运行
程序支持模拟与实盘两种模式切换：

```bash
# 模拟交易 (默认加载 .config.local.json)
python main.py

# 实盘交易 (使用 --production 参数加载 .config.json)
python main.py --production
```

---

## 📈 策略分析说明
程序运行后会生成预测图表 `prediction.png`：

![prediction.png]

### 图表元素对照表
| 元素 | 说明 |
| :--- | :--- |
| **紫色实线** | 历史 570 根 K 线走势 |
| **蓝色虚线区** | 模型的批量原始推理曲线 |
| **蓝色实线** | 经过卡尔曼滤波后的平滑预测曲线 |
| **白色水平线** | POC (Point of Control)，视为目标止盈位 |
| **红/绿水平线** | 基于 ATR 计算的动态止损位 |
| **标题信息** | 显示当前市场、周期、回归角度及 ATR |

### 入场逻辑
*   **📈 做多 (Long)**
    *   **趋势条件**：预测回归角度为正（向上）。
    *   **空间条件**：POC (止盈位) 位于当前价格上方。
    *   **风控**：以低位 ATR 线作为止损，POC 作为第一止盈。

*   **📉 做空 (Short)**
    *   **趋势条件**：预测回归角度为负（向下）。
    *   **空间条件**：POC (止盈位) 位于当前价格下方。
    *   **风控**：以高位 ATR 线作为止损，POC 作为第一止盈。

---

## ⚠️ 免责声明

本程序仅用于量化交易研究与技术演示。
*   **市场有风险，入市需谨慎**。
*   使用者因使用本程序产生的任何经济损失，作者不承担任何法律责任。
*   在实盘交易前，请务必在模拟盘（Sandbox）进行充分测试。

---
*Powered by [Ish](https://github.com/shiyu-coder/Ish-QT) & [Kronos Model](https://github.com/shiyu-coder/Kronos)*
