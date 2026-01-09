# Ish QT 📈
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Model-Kronos-orange)](https://github.com/shiyu-coder/Kronos)

**Ish QT** 是一款基于 **Kronos** 时间序列大模型的智能化量化交易引擎。它将前沿的 Transformer 深度学习预测与经典统计学滤波器、技术指标深度融合，构建了一套从“市场预测”到“风险管理”的闭环自动化交易系统。

---

## ✨ 核心特性
本项目不仅依赖 AI 的原始输出，还通过一套多层过滤机制来确保决策的鲁棒性：

*   **🧠 深度推理 (Kronos AI)**：集成 [Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base) 模型，回顾 **570** 根历史 K 线，实现多样本并行推理，捕捉非线性市场波动。
*   **🛡️ 统计降噪 (Kalman Filter)**：创新性引入卡尔曼滤波处理模型输出，消除 AI 推理的随机抖动，提取高置信度的价格轨迹。
*   **📊 趋势量化 (Linear Regression)**：通过线性回归将滤波曲线转化为动态角度，量化趋势强度，有效过滤震荡行情。
*   **🎯 科学止盈 (Volume Profile)**：计算成交量分布密集区 **POC (Point of Control)**，以此作为市场价值共识点，动态设定止盈目标。
*   **🚧 波动风控 (Average True Range)**：利用 **ATR (平均真实波幅)** 动态衡量市场波动率，实现“宽波动宽止损、窄波动窄止损”的智能仓位保护。

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
        "options": { "defaultType": "swap" },   // 默认合约交易
        "sandbox": true,                        // 是否开启沙盒模拟模式
        "httpsProxy": "",                       // 代理设置
        "wssProxy": ""
    },
    "bot": {
        "symbol": "SOL/USDT:USDT",              // 交易对
        "timeframe": "5m",                      // K 线周期
        "predictionLength": 24,                 // 预测步长 (如 5min * 24 = 2小时)
        "temperature": 1.2,                     // 采样温度 (1.2-1.5)
        "topP": 0.95,                           // 核心采样阈值 (0.95-1.0)
        "sampleCount": 2,                       // 采样次数
        "batch": 40,                            // 批处理大小 (RTX 4070 建议 40)
        "feishuUrl": "",                        // 飞书机器人通知地址
        "atrMultiplier": 2.5,                   // 止损 ATR 倍率
        "trendThreshold": 40,                   // 入场角度阈值
        "usePercent": 0.01,                     // 资金动用百分比
        "leverage": 10                          // 杠杆倍率
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
程序运行后会实时更新 `prediction.png`，辅助人工复核：

![hpcl.png](https://github.com/cciradih/Ish-QT/blob/master/hpcl.png?raw=true)

### 📊 视觉元素解析
| 元素 | 名称 |  交易含义 |
| :--- | :--- | :--- |
| **紫色实线** | 历史走势 | 基础市场状态 |
| **蓝色虚线区** | 原始推理 | AI 预测的原始分布 |
| **蓝色实线** | 滤波曲线 | **核心决策依据**：平滑后的预测方向 |
| **白色水平线** | POC 轴心 | 预期的价值回归目标 (止盈位) |
| **红/绿水平线** | ATR 边界 | 动态止损线 |

### ⚖️ 入场判定逻辑
*   **📈 做多 (Long)**: `角度 > trendThreshold` & `价格 < POC` & `High ATR < POC`
*   **📉 做空 (Short)**: `角度 < -trendThreshold` & `价格 > POC` & `Low ATR > POC`

---

## ⚠️ 免责声明
本程序仅供技术研究使用，不构成任何投资建议。

*   **市场有风险，入市需谨慎**。
*   作者不对因模型误差、网络延迟或程序 Bug 导致的任何资金损失负责。
*   在实盘交易前，请务必在模拟盘（Sandbox）进行充分测试。

---

*Made with ❤️ by [Ish](https://github.com/cciradih/Ish-QT)*

*Powered by [Kronos Model](https://github.com/shiyu-coder/Kronos)*
