# SmartCrypto · Bitget 合约自动交易机器人

自动在 Bitget USDT 永续合约（默认 BTC、ETH）上实时进场、出场，重点是**控制亏损**。

> ⚠️ 没有任何程序能保证赚钱或不亏钱。100 倍杠杆用逐仓时价格反向约 0.5% 就会爆仓，所以默认用全仓、同时只开一单。
> 请按顺序：回测 → 本地模拟 → Bitget 模拟盘，结果满意后再自己决定是否用真钱。

---

## 1. 它怎么交易

| 项目 | 规则 |
|---|---|
| 大方向 | 1 小时 K 线：价格在 EMA50 之上且 EMA50 向上才做多，反过来才做空，方向不明不交易 |
| 进场 | 5 分钟 K 线收盘突破前 20 根的最高点（做多）/ 最低点（做空），并且成交量大于平均 1.2 倍 |
| 仓位 | 100 倍全仓，同时只开一单；控制面板里设每单固定保证金（5–15 USDT） |
| 止损 | 按 ATR 算，限制在 0.3%–1% 之间（ETH 2400 时约 7–24 点）；超过 1% 说明波动太大，这单不做 |
| 仓位方式 | 一次开满，一个止盈（2 倍止损距离）。视频的分批进场、补仓、分批止盈、避开整数关口还在程序里，配置里打开就能用 |
| 移动止损 | 赚到 1 倍止损距离后止损移到成本；之后用 1.5 倍 ATR 跟着价格移动锁利润 |
| 最长持仓 | 4 小时（48 根 5 分钟 K 线）还没保本就平仓；已经保本的单交给移动止损 |

**止损和止盈在开仓时就挂在 Bitget 上**，就算你的电脑断网或程序崩溃，交易所也会帮你止损。

## 2. 风控（减少亏损）

- **按 100 美金算**：`capital_usd: 100`，就算模拟盘里有几万假钱，机器人也当作只有 100 美金（加上它自己赚亏的）来执行所有风控
- **每日亏损上限**：当天亏到开始资金的 30% → 立刻全部平仓，当天不再开单（按 UTC+8 00:00 算新的一天）
- **单笔亏损上限**：止损 + 手续费最多亏 10 USDT，超过会自动缩小仓位
- **爆仓保护**：止损价必须比估算爆仓价早至少 0.08%；开仓后再读交易所真实爆仓价，如果止损在爆仓价之后就马上平仓
- **连亏暂停**：连续亏 4 单暂停 2 小时；某个币止损后，这个币要等 30 分钟再开单
- **资金底线**：账户总资金低于 50 USDT，机器人完全停止
- **同时只开一单**，用全仓（整个账户一起扛，爆仓价远，止损才放得宽）
- **每天最多开几单**：可选（高频模式默认 40 单）

所有数字都在 `config.yaml` 里，有中文说明，可以自己改。

## 3. 安装（Windows）

1. 安装 Python 3.11 以上：https://www.python.org/downloads/ （安装时勾选 "Add Python to PATH"）
2. 把整个 `bitget-futures-bot` 文件夹复制到电脑上，在文件夹里打开 PowerShell
3. 安装依赖（Windows 上用 `py`，不是 `python` 或 `pip`）：
   ```powershell
   py -m pip install -r requirements.txt
   ```

## 4. 使用步骤

### 第一步：下载历史数据并回测（不需要 API key）
```powershell
py main.py download --days 180
py main.py backtest
```
如果 Bitget 只给了最近一段数据，可以用币安的历史数据回测（价格走势几乎一样）：
```powershell
py main.py download --days 365 --exchange binanceusdm
py main.py backtest --exchange binanceusdm
```
结果会显示收益、胜率、最大回撤、每天亏到 30% 的次数等，每笔交易在 `data/backtest_trades.csv`。

### 第二步：本地模拟（真实行情，不下单，不需要 API key）
把 `config.yaml` 里的 `mode` 改成 `paper`：
```powershell
py main.py run
```

### 第三步：Bitget 模拟盘（假钱）
1. 登录 Bitget → 切换到「模拟交易」→ 在模拟账户里创建 API key（权限选「合约交易」，**不要开提现**）
2. 在 PowerShell 设置环境变量（**不要把 key 写进任何文件，也不要发给别人**）：
   ```powershell
   $env:BITGET_API_KEY="你的key"
   $env:BITGET_API_SECRET="你的secret"
   $env:BITGET_API_PASSPHRASE="你的passphrase"
   ```
3. `config.yaml` 里 `mode: demo`，然后：
   ```powershell
   py main.py run
   ```
建议至少跑 3–5 天，看 `data/bot.log` 里的每一笔开仓、平仓。

### 第四步：真钱（只在你确定之后）
需要真实账户的 API key，`mode: live`，并且额外设置：
```powershell
$env:I_UNDERSTAND_REAL_MONEY="yes"
```
没有这一行，程序拒绝用真钱运行。

### 停止
按 `Ctrl + C`。已经开着的单仍然有交易所上的止损/止盈保护。
重新启动时程序会读取 `data/state.json`，接着管理之前的单，并记得当天已经亏了多少。

## 4.4 控制面板（推荐）

双击 **`SmartCrypto.bat`** 打开控制面板：

- 选模式（高频 / 普通），点「开始」「停止」
- 勾选要开的币（BTC、ETH、SOL、XRP、DOGE），设每单固定保证金（5–15 USDT，每单都一样）；机器人运行时不能改，停止后改完再开始
- 第一次点「设置 API key」填三项，会保存在这台电脑上
- 显示今天开单数、平仓数、胜率、今天赚亏和累计赚亏
- 运行记录用中文显示（勾「显示原始记录」可看英文原文）
- 「回测比较」：下载最近 30 天 Bitget 真实 1 分钟 K 线（不需要 API key），用你勾的币和保证金，把几种信号（1 分钟、1 分钟更严、1 分钟更严 + 只做趋势明确、5 分钟、5 分钟只做趋势明确）各跑一遍，按净赚排名。第一次要下载几分钟，之后只补新的 K 线。机器人运行时也可以用
- 关掉面板时会问你：机器人要一起停止，还是留在后台继续跑

想放到桌面：右键 `SmartCrypto.bat` →「发送到」→「桌面快捷方式」。

## 4.5 双击启动（不用开 PowerShell）

文件夹里有 4 个小程序，双击就能用：

| 文件 | 作用 |
|---|---|
| `set_api_key.bat` | 设置 API key，只要做一次（换 key 时再做） |
| `start_bot.bat` | 在后台启动机器人（高频模式），没有窗口 |
| `view_log.bat` | 打开一个窗口看机器人在做什么，关掉它不会停止机器人 |
| `stop_bot.bat` | 停止机器人，已经开着的单在 Bitget 上仍有止损保护 |

第一次用：先双击 `set_api_key.bat` 设好 key，再双击 `start_bot.bat`。
电脑重启后要再双击一次 `start_bot.bat`。

## 5. 高频模式（一天开很多单）

`config_hf.yaml` 是高频版本，和普通模式的区别：

| | 普通模式 `config.yaml` | 高频模式 `config_hf.yaml` |
|---|---|---|
| 币种 | BTC、ETH | BTC、ETH、SOL、XRP、DOGE |
| 进场 K 线 | 5 分钟 | 1 分钟 |
| 大方向 | 1 小时 | 15 分钟 |
| 进场方式 | 市价单（手续费 0.06%） | 市价单（马上成交；配置里可改成挂单 maker） |
| 止盈 | 2 倍止损距离 | 1.5 倍止损距离 |
| 最长持仓 | 4 小时 | 30 分钟 |
| 同时最多 | 1 单 | 1 单 |
| 每天最多 | 不限 | 40 单（防止手续费失控） |

风控完全一样：100 倍全仓、同时一单、每单 5–15 USDT 保证金、一天亏 30% 停止。

**先回测对比两种模式，再决定用哪个：**
```powershell
py main.py download --days 60 --config config_hf.yaml
py main.py download --days 180
py main.py compare
```
对比表里要特别看 `fees_paid`（总手续费）和 `return_pct`（扣完手续费后的收益）。
单多手续费就多，高频模式只有在每单平均赚得比手续费多时才划算。

运行高频模式：所有命令加 `--config config_hf.yaml`，例如
```powershell
py main.py run --config config_hf.yaml
```
两种模式的状态和日志分开保存，可以分别测试。
回测时挂单要价格"穿过"挂单价才算成交，成交那根 K 线只检查止损不算止盈，偏保守。

## 6. 文件说明

| 文件 | 作用 |
|---|---|
| `config.yaml` | 普通模式参数 |
| `config_hf.yaml` | 高频模式参数 |
| `gui.py` / `SmartCrypto.bat` | 控制面板 |
| `main.py` | 启动入口 |
| `bot/strategy.py` | 进出场信号 |
| `bot/risk.py` | 风控：仓位、止损、每日亏损上限、爆仓保护 |
| `bot/exchange.py` | Bitget 下单（模拟盘/实盘）和本地模拟 |
| `bot/engine.py` | 实时循环 |
| `bot/backtest.py` | 回测 |
| `tests/` | 自动测试：`py -m pytest tests` |

## 7. 需要知道的限制

- 手续费影响很大：1500 USDT 仓位开平一次约 1.8 USDT，止损又很近，所以策略需要赚得比手续费多才有用。回测已经算进手续费和滑点。
- 回测没有算资金费率（每 8 小时一次，金额通常很小）。
- 模拟盘的成交和真实市场会有差别，尤其是行情剧烈时的滑点。
- 交易所上挂的止损是按标记价格触发的，极端行情可能成交在比止损更差的价格。
