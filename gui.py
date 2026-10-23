"""SmartCrypto control panel: start/stop the bot, see today's trades and the live log.

Double-click SmartCrypto.bat (Windows) or run: py -3.13 gui.py
The bot itself runs as a separate background process, so closing this window can leave it
running; the panel asks what to do.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import yaml

ROOT = Path(__file__).resolve().parent
IS_WIN = sys.platform == "win32"
NO_WINDOW = 0x08000000 if IS_WIN else 0  # CREATE_NO_WINDOW

MODES = {"高频模式": "config_hf.yaml", "普通模式": "config.yaml"}
KEY_VARS = ("BITGET_API_KEY", "BITGET_API_SECRET", "BITGET_API_PASSPHRASE")
MODE_NAMES = {"paper": "本地模拟", "demo": "Bitget 模拟盘", "live": "真钱实盘"}
COINS = ("BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "PEPE", "BGB")
SETTINGS_FILE = ROOT / "data" / "panel_settings.json"
MARGIN_MIN, MARGIN_MAX = 5, 15

PNL_RE = re.compile(r"(?:pnl≈|approx pnl )(-?\d+(?:\.\d+)?)")
LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2}")


# ---------------- helpers (no GUI, unit-tested) ----------------
def read_paths(config_file: str) -> tuple[Path, Path, str]:
    """(log file, pid file, mode) for a config."""
    cfg = yaml.safe_load((ROOT / config_file).read_text(encoding="utf-8")) or {}
    state = ROOT / cfg.get("state_file", "data/state.json")
    return ROOT / cfg.get("log_file", "data/bot.log"), state.with_suffix(".pid"), cfg.get("mode", "demo")


def load_settings(path: Path = SETTINGS_FILE) -> dict:
    """Coins ticked and fixed margin per trade, remembered between panel sessions."""
    st = {"coins": list(COINS), "margin": 10.0}
    try:
        st.update(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    st["coins"] = [c for c in st["coins"] if c in COINS] or list(COINS)
    st["margin"] = min(MARGIN_MAX, max(MARGIN_MIN, float(st["margin"])))
    return st


def save_settings(st: dict, path: Path = SETTINGS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")


def run_command(config_file: str, coins: list[str], margin: float) -> list[str]:
    return [sys.executable, str(ROOT / "main.py"), "run", "--config", config_file,
            "--coins", ",".join(coins), "--margin", f"{margin:g}"]


def tune_command(config_file: str, coins: list[str], margin: float, days: int = 90) -> list[str]:
    return [sys.executable, str(ROOT / "main.py"), "tune", "--config", config_file,
            "--coins", ",".join(coins), "--margin", f"{margin:g}", "--days", str(days)]


def flowtest_command() -> list[str]:
    return [sys.executable, str(ROOT / "main.py"), "flowtest"]


def record_command(coins: list[str]) -> list[str]:
    return [sys.executable, str(ROOT / "main.py"), "record", "--coins", ",".join(coins)]


RECORD_PID = ROOT / "data" / "record.pid"


def selftest_command(config_file: str, coin: str, margin: float) -> list[str]:
    return [sys.executable, str(ROOT / "main.py"), "selftest", "--config", config_file,
            "--coins", coin, "--margin", f"{margin:g}"]


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if IS_WIN:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def running_pid(pid_file: Path) -> int | None:
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def stop_pid(pid: int) -> None:
    if IS_WIN:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True,
                       creationflags=NO_WINDOW)
    else:
        os.kill(pid, signal.SIGTERM)


@dataclass
class Stats:
    opens_today: int = 0
    closes_today: int = 0
    wins_today: int = 0
    pnl_today: float = 0.0
    pnl_total: float = 0.0
    closes_total: int = 0

    @property
    def win_rate(self) -> str:
        return f"{self.wins_today / self.closes_today:.0%}" if self.closes_today else "-"


def parse_stats(lines: list[str], today: str) -> Stats:
    st = Stats()
    for line in lines:
        m = LINE_RE.match(line)
        if not m:
            continue
        is_today = m.group(1) == today
        if " OPEN " in line or " FILLED " in line:
            st.opens_today += is_today
            continue
        if " CLOSE " in line or "position gone" in line:
            pm = PNL_RE.search(line)
            if not pm:
                continue
            pnl = float(pm.group(1))
            st.pnl_total += pnl
            st.closes_total += 1
            if is_today:
                st.pnl_today += pnl
                st.closes_today += 1
                st.wins_today += pnl > 0
    return st


REASONS = {"managed stop hit": "跟踪止损", "max hold time": "持仓超时", "risk stop": "风控全部平仓",
           "liq safety": "爆仓保护", "replaced by new signal": "换成新信号",
           "take-profit": "止盈"}
SKIPS = {"daily loss limit": "今天已亏到上限", "consecutive losses": "连亏暂停中",
         "cooling down": "刚止损，冷却中", "max open positions": "同时持仓已满",
         "loss budget": "手上几单的止损加起来已到今天亏损上限",
         "trades per day": "今天单数已满", "price already ran": "价格已经跑远",
         "too close to liquidation": "止损太靠近爆仓价", "wider than max": "波动太大",
         "closer than min": "止损太近", "below min": "仓位太小", "exchange minimum": "低于交易所最小下单量"}


def coin(sym: str) -> str:
    return sym.split("/")[0]


def friendly(line: str) -> str:
    """Turn a log line into a short Chinese sentence; unknown lines are returned as-is."""
    out = _friendly(line)
    if out is line:
        return line
    # BTC/ETH-size prices without decimals, the way traders read them (83528.96218 -> 83529)
    return re.sub(r"\b\d{4,}\.\d+", lambda m: f"{float(m[0]):.0f}", out)


def _friendly(line: str) -> str:
    m = re.match(r"^\d{4}-\d{2}-\d{2} (\d{2}:\d{2}):\d{2},\d+ \w+ [\w.]+: (.*)$", line)
    if not m:
        return line
    t, msg = m.groups()
    side = lambda x: "做多" if x == "long" else "做空"
    if r := re.match(r"MAKER (\S+) (long|short) qty=\S+ @ (\S+) stop=(\S+) tp=(\S+) margin≈(\S+)", msg):
        return (f"{t}  {coin(r[1])} 挂单{side(r[2])} @ {r[3]}，保证金 {r[6]}，"
                f"止损 {r[4]}，止盈 {r[5]}")
    if r := re.match(r"OPEN (\S+) (long|short) qty=\S+ @ (\S+) stop=(\S+) tp=(\S+) margin≈(\S+)", msg):
        return (f"{t}  {coin(r[1])} 市价开仓{side(r[2])} @ {r[3]}，保证金 {r[6]}，"
                f"止损 {r[4]}，止盈 {r[5]}")
    if r := re.match(r"ADD (\S+) (long|short) qty=\S+ @ (\S+) -> total qty=\S+ avg=(\S+)", msg):
        return f"{t}  {coin(r[1])} 补仓{side(r[2])} @ {r[3]}，新的平均成本 {r[4]}"
    if r := re.match(r"REDUCE (\S+) \S+ qty=\S+ @ (\S+) back at cost, gain (\S+)", msg):
        return f"{t}  {coin(r[1])} 回到成本，平掉补仓的部分 @ {r[2]}，约 {float(r[3]):+.2f} USDT"
    if r := re.match(r"PARTIAL (\S+) \S+ qty=\S+ @ (\S+) \(take-profit \S+\) gain (\S+), stop now (\S+)", msg):
        return (f"{t}  {coin(r[1])} 分批止盈，平掉一部分 @ {r[2]}，约 {float(r[3]):+.2f} USDT，"
                f"止损移到 {r[4]}")
    if r := re.match(r"PLAN (\S+) add=(\S+) @ (\S+) take-profits=(\S+)", msg):
        add = f"到 {r[3]} 补仓，" if float(r[2]) > 0 else ""
        return f"{t}  {coin(r[1])} 计划：{add}分批止盈 {r[4].replace('/', ' / ')}"
    if r := re.match(r"(\S+) new signal on (\S+): closing", msg):
        return f"{t}  {coin(r[2])} 出新信号，{coin(r[1])} 还没保本，平掉换新单"
    if r := re.match(r"SCAN (\d+) coins up=(\d+) down=(\d+) flat=(\d+) signals=(\d+)", msg):
        found = f"{r[5]} 个信号" if r[5] != "0" else "没有信号"
        return (f"{t}  扫描 {r[1]} 个币：大方向向上 {r[2]}、向下 {r[3]}、不明 {r[4]}，{found}")
    if r := re.match(r"FLOW (\S+) (long|short) taker (\d+)% book (\S+) (ok|against)", msg):
        verdict = "盘口确认，开单" if r[5] == "ok" else "盘口不支持，不开"
        return (f"{t}  {coin(r[1])} {side(r[2])}信号：主动{'买' if r[2] == 'long' else '卖'}占 {r[3]}%，"
                f"挂单厚度 {float(r[4]):+.2f}，{verdict}")
    if r := re.match(r"close (\S+): position already closed by the exchange", msg):
        return f"{t}  {coin(r[1])} 交易所已先平掉（止损/止盈），不用再平"
    if r := re.match(r"(\S+) maker entry partly filled (\S+)/(\S+) before cancel", msg):
        return f"{t}  {coin(r[1])} 挂单只成交了一部分（{r[2]}/{r[3]}），剩下的已撤"
    if r := re.match(r"TODAY (\d+) trades, pnl (\S+)", msg):
        return f"{t}  今天已平 {r[1]} 单，合计 {float(r[2]):+.2f} USDT"
    if r := re.match(r"kind:(\S+) last (\d+) trades (\S+): half size", msg):
        kinds = {"breakout": "突破", "bounce": "反弹被打回", "pullback": "回踩", "reversion": "超买超卖"}
        return (f"{t}  「{kinds.get(r[1], r[1])}」类信号最近 {r[2]} 单合计 {float(r[3]):+.2f}，"
                f"这单保证金减半")
    if r := re.match(r"(\S+) last (\d+) trades (\S+): half size", msg):
        return f"{t}  {coin(r[1])} 最近 {r[2]} 单合计 {float(r[3]):+.2f}，这单保证金减半"
    if r := re.match(r"no futures market for (\S+) on Bitget", msg):
        return f"{t}  {coin(r[1])} Bitget 没有合约，跳过"
    if r := re.match(r"cancel \S+ on (\S+): already gone", msg):
        return f"{t}  {coin(r[1])} 止盈挂单已不在（成交或随平仓撤掉）"
    if r := re.match(r"TP-LIMIT (\S+) \S+ qty=\S+ @ (\S+)", msg):
        return f"{t}  {coin(r[1])} 止盈挂单已挂好 @ {r[2]}"
    if r := re.match(r"(\S+) limit take-profit not placed .* close at (\S+) itself", msg):
        return f"{t}  {coin(r[1])} 止盈挂单没挂上，到 {r[2]} 机器人自己平仓"
    if r := re.match(r"FILLED (\S+) (long|short) qty=\S+ @ (\S+)", msg):
        return f"{t}  {coin(r[1])} 挂单成交，开仓{side(r[2])} @ {r[3]}"
    if r := re.match(r"(\S+) maker entry \S+ (not filled|cancelled)", msg):
        return f"{t}  {coin(r[1])} 挂单没成交，已撤单"
    if r := re.match(r"(\S+) position gone .*approx pnl (\S+)", msg):
        return f"{t}  {coin(r[1])} 被交易所止损/止盈平仓，约 {float(r[2]):+.2f} USDT"
    if r := re.match(r"CLOSE (\S+) \S+ qty=\S+ @ (\S+) \((.+)\) pnl≈(\S+)", msg):
        return (f"{t}  {coin(r[1])} 平仓 @ {r[2]}（{REASONS.get(r[3], r[3])}），"
                f"约 {float(r[4]):+.2f} USDT")
    if r := re.match(r"(\S+) signal skipped: (.*)", msg):
        if m := re.match(r"BTC moving against it \((\S+)% in (\d+) bars\)", r[2]):
            return (f"{t}  {coin(r[1])} 有信号但没开单：BTC 最近 {m[2]} 分钟 {float(m[1]):+.2f}%，"
                    f"大盘在反方向走")
        if m := re.match(r"free balance (\S+) too small", r[2]):
            return f"{t}  {coin(r[1])} 有信号但没开单：可用余额只剩 {m[1]}"
        if m := re.match(r"loss budget left (\S+), open trades risk (\S+)", r[2]):
            return (f"{t}  {coin(r[1])} 有信号但没开单：今天离 -30% 只剩 {m[1]}，"
                    f"手上单的止损占 {m[2]}")
        why = next((v for k, v in SKIPS.items() if k in r[2]), r[2])
        return f"{t}  {coin(r[1])} 有信号但没开单：{why}"
    if msg.startswith("RISK STOP"):
        return f"{t}  ⚠ 今天亏到上限，已全部平仓，明天再开始"
    if msg.startswith("started:"):
        return f"{t}  机器人启动"
    if r := re.search(r"real equity (\S+) taken as baseline", msg):
        return f"{t}  读到账户余额 {r[1]}，按 100 USDT 计算风控"
    if r := re.match(r"(\S+) max leverage is below \d+x, using (\d+)x", msg):
        return f"{t}  {coin(r[1])} 最高只能 {r[2]} 倍，已自动改用"
    if r := re.match(r"(\S+) leverage (\d+)x", msg):
        return f"{t}  {coin(r[1])} 杠杆 {r[2]} 倍"
    if "unified/UTA account" in msg:
        return f"{t}  已连接 Bitget 统一账户"
    if "classic futures account" in msg:
        return f"{t}  已连接 Bitget 合约账户"
    if r := re.match(r"(\S+) adopted unknown position, managed stop (\S+)", msg):
        return f"{t}  {coin(r[1])} 发现不是机器人开的仓位，已接管，止损 {r[2]}"
    if msg.startswith("account balance reads as 0"):
        return f"{t}  账户余额读到 0"
    if "账户余额读到 0" in msg:
        return f"{t}  ⚠ 账户里没有 USDT，机器人不会开单"
    if msg.startswith("network error"):
        return f"{t}  网络断了一下，自动重试中"
    if "loop error" in msg:
        return f"{t}  ⚠ 出错了，会自动重试（看原始记录了解详情）"
    return line


def tail(path: Path, max_bytes: int = 200_000) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = data.splitlines()
    return lines[1:] if size > max_bytes else lines


def save_keys(key: str, secret: str, passphrase: str) -> None:
    for name, val in zip(KEY_VARS, (key, secret, passphrase)):
        os.environ[name] = val
        if IS_WIN:  # keep for future sessions too
            subprocess.run(["setx", name, val], capture_output=True, creationflags=NO_WINDOW)


# ---------------- GUI ----------------
BG, CARD, FG, MUTED = "#0f1720", "#18222e", "#e6edf3", "#8b98a5"
GREEN, RED, AMBER, BLUE = "#22c55e", "#ef4444", "#f59e0b", "#0ea5e9"


class Panel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SmartCrypto")
        self.configure(bg=BG)
        self.geometry("980x660")
        self.minsize(720, 520)
        self.font = "Microsoft YaHei UI" if IS_WIN else "WenQuanYi Zen Hei"
        self.proc: subprocess.Popen | None = None
        self.tune_proc: subprocess.Popen | None = None
        self.test_proc: subprocess.Popen | None = None
        self.mode = tk.StringVar(value="高频模式")
        self.raw = tk.BooleanVar(value=False)
        st = load_settings()
        self.coin_vars = {c: tk.BooleanVar(value=c in st["coins"]) for c in COINS}
        self.margin = tk.StringVar(value=f"{st['margin']:g}")
        self._log_sig = None
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh()

    # ----- layout -----
    def _label(self, parent, text="", size=11, color=FG, bold=False, bg=None):
        return tk.Label(parent, text=text, fg=color, bg=bg or parent["bg"],
                        font=(self.font, size, "bold" if bold else "normal"))

    def _build(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=16, pady=(14, 6))
        logo = ROOT / "docs" / "logo.png"
        if logo.exists():
            try:
                img = tk.PhotoImage(file=str(logo))
                factor = max(1, img.height() // 56)
                self.logo = img.subsample(factor, factor)
                tk.Label(head, image=self.logo, bg=BG).pack(side="left", padx=(0, 10))
            except tk.TclError:
                pass
        titles = tk.Frame(head, bg=BG)
        titles.pack(side="left")
        self._label(titles, "SmartCrypto", 18, bold=True).pack(anchor="w")
        self.sub = self._label(titles, "", 10, MUTED)
        self.sub.pack(anchor="w")
        self.status = self._label(head, "● 已停止", 13, MUTED, bold=True)
        self.status.pack(side="right")

        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(fill="x", padx=16, pady=6)
        self._label(ctrl, "模式", 11, MUTED).pack(side="left")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=FG,
                        arrowcolor=FG)
        style.map("TCombobox", fieldbackground=[("readonly", CARD)], foreground=[("readonly", FG)],
                  selectbackground=[("readonly", CARD)], selectforeground=[("readonly", FG)])
        box = ttk.Combobox(ctrl, textvariable=self.mode, values=list(MODES), width=10,
                           state="readonly", font=(self.font, 11))
        box.pack(side="left", padx=(6, 16))
        box.bind("<<ComboboxSelected>>", lambda e: self.refresh(force_log=True))
        self.start_btn = self._button(ctrl, "▶  开始", GREEN, self.start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = self._button(ctrl, "■  停止", RED, self.stop)
        self.stop_btn.pack(side="left", padx=4)
        self._button(ctrl, "设置 API key", "#334155", self.key_dialog).pack(side="right")
        self.key_label = self._label(ctrl, "", 10, MUTED)
        self.key_label.pack(side="right", padx=8)

        # research tools on their own row so they never get pushed off a narrow window
        tools = tk.Frame(self, bg=BG)
        tools.pack(fill="x", padx=16, pady=(2, 4))
        self._label(tools, "工具", 11, MUTED).pack(side="left")
        self.tune_btn = self._button(tools, "回测比较", BLUE, self.tune)
        self.tune_btn.pack(side="left", padx=(6, 8))
        self.rec_btn = self._button(tools, "记录盘口", BLUE, self.toggle_record)
        self.rec_btn.pack(side="left", padx=(0, 8))
        self.flow_btn = self._button(tools, "分析盘口", BLUE, self.flowtest)
        self.flow_btn.pack(side="left", padx=(0, 8))
        self.test_btn = self._button(tools, "测试下单", BLUE, self.selftest)
        self.test_btn.pack(side="left")

        pick = tk.Frame(self, bg=BG)
        pick.pack(fill="x", padx=16, pady=(2, 4))
        self._label(pick, "币种", 11, MUTED).pack(side="left")
        self.inputs = []
        for c in COINS:
            cb = tk.Checkbutton(pick, text=c, variable=self.coin_vars[c], bg=BG, fg=FG,
                                selectcolor=CARD, activebackground=BG, activeforeground=FG,
                                font=(self.font, 10), command=self.save_choices)
            cb.pack(side="left", padx=(6, 0))
            self.inputs.append(cb)
        self._label(pick, "每单保证金", 11, MUTED).pack(side="left", padx=(20, 6))
        sp = tk.Spinbox(pick, from_=MARGIN_MIN, to=MARGIN_MAX, increment=1, width=4,
                        textvariable=self.margin, bg=CARD, fg=FG, buttonbackground=CARD,
                        insertbackground=FG, relief="flat", font=(self.font, 11),
                        command=self.save_choices)
        sp.bind("<FocusOut>", lambda e: self.save_choices())
        sp.pack(side="left")
        self.inputs.append(sp)
        self._label(pick, "USDT（每单都一样）", 10, MUTED).pack(side="left", padx=4)

        cards = tk.Frame(self, bg=BG)
        cards.pack(fill="x", padx=12, pady=8)
        self.card_vals = {}
        for key, title in (("opens", "今天开单"), ("closes", "今天平仓"), ("win", "今天胜率"),
                           ("pnl", "今天赚亏 USDT"), ("total", "累计赚亏 USDT")):
            c = tk.Frame(cards, bg=CARD, padx=12, pady=10)
            c.pack(side="left", expand=True, fill="x", padx=4)
            self._label(c, title, 10, MUTED).pack(anchor="w")
            v = self._label(c, "-", 20, bold=True)
            v.pack(anchor="w")
            self.card_vals[key] = v

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=12, pady=(6, 0))
        self._label(bar, "  运行记录", 10, MUTED).pack(side="left")
        tk.Checkbutton(bar, text="显示原始记录", variable=self.raw, bg=BG, fg=MUTED,
                       selectcolor=CARD, activebackground=BG, activeforeground=FG,
                       font=(self.font, 9), command=lambda: self.refresh_log()).pack(side="right")
        box = tk.Frame(self, bg=CARD)
        box.pack(fill="both", expand=True, padx=16, pady=(2, 8))
        self.log = tk.Text(box, bg=CARD, fg=FG, insertbackground=FG, relief="flat", wrap="word",
                           font=("Consolas" if IS_WIN else "DejaVu Sans Mono", 10), padx=8, pady=6)
        sb = tk.Scrollbar(box, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set, state="disabled")
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        for tag, color in (("good", GREEN), ("bad", RED), ("warn", AMBER), ("info", BLUE),
                           ("muted", MUTED)):
            self.log.tag_configure(tag, foreground=color)

        self._label(self, "止损和止盈挂在 Bitget 上：就算这个程序关掉或断网，已经开着的单也有保护。",
                    9, MUTED).pack(anchor="w", padx=16, pady=(0, 10))

    def _button(self, parent, text, color, cmd):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="white", relief="flat",
                         activebackground=color, activeforeground="white", padx=14, pady=4,
                         font=(self.font, 11, "bold"), cursor="hand2", bd=0)

    # ----- state -----
    def paths(self):
        return read_paths(MODES[self.mode.get()])

    def other_running(self) -> str | None:
        for name, cfg in MODES.items():
            if name != self.mode.get() and running_pid(read_paths(cfg)[1]):
                return name
        return None

    def refresh(self, force_log: bool = False):
        log_file, pid_file, mode = self.paths()
        pid = running_pid(pid_file)
        starting = self.proc is not None and self.proc.poll() is None and pid is None
        if pid or starting:
            self.status.configure(text="● 运行中" if pid else "● 启动中…", fg=GREEN)
            self.start_btn.configure(state="disabled", bg="#334155")
            self.stop_btn.configure(state="normal", bg=RED)
        else:
            self.status.configure(text="● 已停止", fg=MUTED)
            self.start_btn.configure(state="normal", bg=GREEN)
            self.stop_btn.configure(state="disabled", bg="#334155")
        for w in self.inputs:  # choices apply when the bot starts, so lock them while it runs
            w.configure(state="disabled" if pid or starting else "normal")
        has_keys = all(os.environ.get(k) for k in KEY_VARS)
        self.key_label.configure(text="API key 已设置" if has_keys else "API key 未设置",
                                 fg=GREEN if has_keys else AMBER)
        self.sub.configure(text=f"{MODES[self.mode.get()]}  ·  {MODE_NAMES.get(mode, mode)}",
                           fg=RED if mode == "live" else MUTED)

        lines = tail(log_file)
        st = parse_stats(lines, datetime.now().strftime("%Y-%m-%d"))
        self.card_vals["opens"].configure(text=str(st.opens_today))
        self.card_vals["closes"].configure(text=str(st.closes_today))
        self.card_vals["win"].configure(text=st.win_rate)
        for key, val in (("pnl", st.pnl_today), ("total", st.pnl_total)):
            self.card_vals[key].configure(text=f"{val:+.2f}",
                                          fg=GREEN if val > 0 else RED if val < 0 else FG)
        sig = (str(log_file), len(lines), lines[-1] if lines else "", self.raw.get())
        if force_log or sig != self._log_sig:
            self._log_sig = sig
            self.show_log(lines[-300:])
        if getattr(self, "_job", None):
            self.after_cancel(self._job)
        self.check_tune()
        self.check_record()
        self.check_selftest()
        self._job = self.after(1500, self.refresh)

    def refresh_log(self):
        self.show_log(tail(self.paths()[0])[-300:])

    def show_log(self, lines):
        at_bottom = self.log.yview()[1] > 0.98
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        for line in lines:
            tag = "muted"
            if any(k in line for k in (" OPEN ", " FILLED ", " MAKER ", " ADD ", " PLAN ",
                                        " REDUCE ", " PARTIAL ", " TP-LIMIT ")):
                tag = "info"
            pm = PNL_RE.search(line)
            if pm:
                tag = "good" if float(pm.group(1)) > 0 else "bad"
            if " WARNING " in line:
                tag = "warn"
            if " ERROR " in line or "Traceback" in line:
                tag = "bad"
            self.log.insert("end", (line if self.raw.get() else friendly(line)) + "\n", tag)
        self.log.configure(state="disabled")
        if at_bottom or not lines:
            self.log.see("end")

    # ----- actions -----
    def choices(self) -> tuple[list[str], float | None]:
        coins = [c for c in COINS if self.coin_vars[c].get()]
        try:
            margin = float(self.margin.get())
        except ValueError:
            return coins, None
        return coins, margin if MARGIN_MIN <= margin <= MARGIN_MAX else None

    def save_choices(self):
        coins, margin = self.choices()
        if coins and margin:
            save_settings({"coins": coins, "margin": margin})

    def start(self):
        other = self.other_running()
        if other:
            messagebox.showwarning("SmartCrypto", f"{other}正在运行。同一个账户一次只跑一种模式，请先停止它。")
            return
        log_file, pid_file, mode = self.paths()
        if mode in ("demo", "live") and not all(os.environ.get(k) for k in KEY_VARS):
            messagebox.showwarning("SmartCrypto", "请先点「设置 API key」。")
            return
        coins, margin = self.choices()
        if not coins:
            messagebox.showwarning("SmartCrypto", "至少勾选一个币。")
            return
        if margin is None:
            messagebox.showwarning("SmartCrypto", f"每单保证金要在 {MARGIN_MIN}–{MARGIN_MAX} USDT 之间。")
            return
        self.save_choices()
        if mode == "live":
            if not messagebox.askyesno("真钱实盘", "这个配置是真钱实盘（mode: live）。确定要用真钱开始吗？"):
                return
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        if mode == "live":
            env["I_UNDERSTAND_REAL_MONEY"] = "yes"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        err = open(log_file.parent / "panel_errors.log", "a", encoding="utf-8")
        self.proc = subprocess.Popen(
            run_command(MODES[self.mode.get()], coins, margin),
            cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=err,
            creationflags=NO_WINDOW | (0x00000200 if IS_WIN else 0))  # + NEW_PROCESS_GROUP
        self.after(4000, self.check_started)
        self.refresh(force_log=True)

    def check_started(self):
        if self.proc is not None and self.proc.poll() is not None:
            errs = tail(ROOT / "data" / "panel_errors.log")[-8:]
            messagebox.showerror("启动失败", "机器人没能启动：\n\n" + "\n".join(errs[-8:]))

    def stop(self):
        _, pid_file, _ = self.paths()
        pid = running_pid(pid_file)
        if pid:
            stop_pid(pid)
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None  # stopped on purpose: check_started must not report a failure
        pid_file.unlink(missing_ok=True)
        self.after(500, lambda: self.refresh(force_log=True))

    def tune(self):
        """回测比较: download recent real candles and compare a few ways of trading."""
        if self.tune_proc is not None:
            return
        coins, margin = self.choices()
        if not coins or margin is None:
            messagebox.showwarning("SmartCrypto", "先勾选币种，并把每单保证金设在 5–15 之间。")
            return
        data = ROOT / "data"
        data.mkdir(exist_ok=True)
        (data / "tune_progress.txt").write_text("开始…", encoding="utf-8")
        err = open(data / "tune_errors.log", "w", encoding="utf-8")
        self.tune_proc = subprocess.Popen(
            tune_command(MODES[self.mode.get()], coins, margin), cwd=str(ROOT),
            env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
            stdout=subprocess.DEVNULL, stderr=err, creationflags=NO_WINDOW)

    def check_tune(self):
        if self.tune_proc is None:
            self.tune_btn.configure(text="回测比较", state="normal")
            return
        if self.tune_proc.poll() is None:
            try:
                msg = (ROOT / "data" / "tune_progress.txt").read_text(encoding="utf-8")
            except OSError:
                msg = "回测中…"
            self.tune_btn.configure(text=msg[:28], state="disabled")
            return
        code, self.tune_proc = self.tune_proc.returncode, None
        self.tune_btn.configure(text="回测比较", state="normal")
        if code != 0:
            errs = tail(ROOT / "data" / "tune_errors.log")[-8:]
            messagebox.showerror("回测失败", "回测没有完成：\n\n" + "\n".join(errs))
            return
        report = (ROOT / "data" / "tune_report.txt").read_text(encoding="utf-8")
        win = tk.Toplevel(self, bg=BG)
        win.title("回测比较")
        win.geometry("620x560")
        box = tk.Text(win, bg=CARD, fg=FG, relief="flat", wrap="word", padx=12, pady=10,
                      font=(self.font, 11))
        box.insert("1.0", report)
        box.configure(state="disabled")
        box.pack(fill="both", expand=True, padx=12, pady=12)

    def flowtest(self):
        """分析盘口: does the recorded order book say anything a round trip's cost can't eat?"""
        data = ROOT / "data"
        data.mkdir(exist_ok=True)
        self.flow_btn.configure(text="分析中…", state="disabled")
        self.update_idletasks()
        try:
            proc = subprocess.run(flowtest_command(), cwd=str(ROOT), capture_output=True,
                                  env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                                  creationflags=NO_WINDOW, timeout=1800)
        finally:
            self.flow_btn.configure(text="分析盘口", state="normal")
        path = data / "flowtest_report.txt"
        if proc.returncode != 0 or not path.exists():
            err = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-8:]
            messagebox.showerror("分析失败", "分析没有完成：\n\n" + "\n".join(err))
            return
        self._report_window("分析盘口", path.read_text(encoding="utf-8"))

    def _report_window(self, title: str, text: str):
        win = tk.Toplevel(self, bg=BG)
        win.title(title)
        win.geometry("660x560")
        box = tk.Text(win, bg=CARD, fg=FG, relief="flat", wrap="word", padx=12, pady=10,
                      font=(self.font, 11))
        box.insert("1.0", text)
        box.configure(state="disabled")
        box.pack(fill="both", expand=True, padx=12, pady=12)

    def selftest(self):
        """测试下单: one pass of every order type on the demo account, about a minute."""
        if self.test_proc is not None:
            return
        coins, margin = self.choices()
        if not coins or margin is None:
            messagebox.showwarning("SmartCrypto", "先勾选币种，并把每单保证金设在 5–15 之间。")
            return
        cfg_file = MODES[self.mode.get()]
        _, pid_file, mode = read_paths(cfg_file)
        if running_pid(pid_file) or self.other_running():
            messagebox.showwarning("SmartCrypto", "机器人在运行，先停止再测试下单。")
            return
        if mode != "demo":
            messagebox.showwarning("SmartCrypto", "测试下单只在 Bitget 模拟盘上跑。")
            return
        if not all(os.environ.get(k) for k in KEY_VARS):
            messagebox.showwarning("SmartCrypto", "先设置 API key。")
            return
        if not messagebox.askyesno("测试下单", f"在模拟盘用 {coins[0]} 开一张小单、挂止损止盈、"
                                   "再马上平掉，大约一分钟。继续？"):
            return
        data = ROOT / "data"
        data.mkdir(exist_ok=True)
        err = open(data / "selftest_errors.log", "w", encoding="utf-8")
        self.test_proc = subprocess.Popen(
            selftest_command(cfg_file, coins[0], margin), cwd=str(ROOT),
            env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
            stdout=subprocess.DEVNULL, stderr=err, creationflags=NO_WINDOW)
        self.test_btn.configure(text="测试中…", state="disabled")

    def check_selftest(self):
        if self.test_proc is None or self.test_proc.poll() is None:
            return
        code, self.test_proc = self.test_proc.returncode, None
        self.test_btn.configure(text="测试下单", state="normal")
        try:
            text = (ROOT / "data" / "selftest_report.txt").read_text(encoding="utf-8")
        except OSError:
            text = ""
        if code != 0 or not text:
            text = "测试没有跑完：\n\n" + "\n".join(tail(ROOT / "data" / "selftest_errors.log")[-8:])
        win = tk.Toplevel(self, bg=BG)
        win.title("测试下单")
        win.geometry("620x360")
        box = tk.Text(win, bg=CARD, fg=FG, relief="flat", wrap="word", padx=12, pady=10,
                      font=(self.font, 11))
        box.insert("1.0", text)
        box.configure(state="disabled")
        box.pack(fill="both", expand=True, padx=12, pady=12)

    def toggle_record(self):
        """记录盘口: record order book + trades in the background for short-term research."""
        pid = running_pid(RECORD_PID)
        if pid:
            stop_pid(pid)
            RECORD_PID.unlink(missing_ok=True)
            self.after(500, self.check_record)
            return
        coins, _ = self.choices()
        if not coins:
            messagebox.showwarning("SmartCrypto", "先勾选要记录的币。")
            return
        data = ROOT / "data"
        data.mkdir(exist_ok=True)
        err = open(data / "record_errors.log", "w", encoding="utf-8")
        subprocess.Popen(record_command(coins), cwd=str(ROOT),
                         env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                         stdout=subprocess.DEVNULL, stderr=err, creationflags=NO_WINDOW)
        self.after(3000, self.check_record)

    def check_record(self):
        if running_pid(RECORD_PID):
            try:
                msg = (ROOT / "data" / "record_status.txt").read_text(encoding="utf-8")
            except OSError:
                msg = "记录盘口中…"
            self.rec_btn.configure(text="停止记录（" + msg.replace("记录盘口中 ", "")[:16] + "）")
        else:
            self.rec_btn.configure(text="记录盘口")

    def key_dialog(self):
        d = tk.Toplevel(self, bg=BG, padx=16, pady=12)
        d.title("设置 Bitget API key")
        d.transient(self)
        d.grab_set()
        self._label(d, "只保存在这台电脑上，不要发给任何人。", 10, MUTED).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        entries = []
        for i, (label, secret) in enumerate((("API Key", False), ("Secret Key", True),
                                              ("密码短语 Passphrase", True)), start=1):
            self._label(d, label, 10).grid(row=i, column=0, sticky="w", pady=4)
            e = tk.Entry(d, width=46, show="•" if secret else "", bg=CARD, fg=FG,
                         insertbackground=FG, relief="flat", font=(self.font, 10))
            e.grid(row=i, column=1, pady=4, padx=(8, 0))
            entries.append(e)

        def ok():
            vals = [e.get().strip() for e in entries]
            if not all(vals):
                messagebox.showwarning("SmartCrypto", "三项都要填。", parent=d)
                return
            save_keys(*vals)
            d.destroy()
            self.refresh()
        self._button(d, "保存", GREEN, ok).grid(row=4, column=1, sticky="e", pady=(10, 0))

    def on_close(self):
        _, pid_file, _ = self.paths()
        if running_pid(pid_file):
            ans = messagebox.askyesnocancel(
                "SmartCrypto", "机器人还在运行。\n\n是：一起停止\n否：让它在后台继续跑\n取消：不关窗口")
            if ans is None:
                return
            if ans:
                self.stop()
        self.destroy()


if __name__ == "__main__":
    Panel().mainloop()
