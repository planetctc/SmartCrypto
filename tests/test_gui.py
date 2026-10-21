import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("tkinter")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gui  # noqa: E402

D = "2026-10-21"
LOG = [
    f"{D} 15:12:06,720 INFO bot.engine: MAKER XRP/USDT:USDT long qty=595.0 @ 1.3360 stop=1.3340 tp=1.3390 margin≈7.96 strength=0.30 (x)",
    f"{D} 15:12:34,622 INFO bot.engine: FILLED XRP/USDT:USDT long qty=595.0 @ 1.3360 stop=1.3340 tp=1.3390",
    f"{D} 15:16:49,871 INFO bot.engine: XRP/USDT:USDT position gone (exchange stop/TP), approx pnl -1.49",
    f"{D} 15:44:40,553 INFO bot.engine: CLOSE ETH/USDT:USDT long qty=0.31 @ 3970.90 (managed stop hit) pnl≈2.31",
    "2026-10-20 10:00:00,000 INFO bot.engine: CLOSE BTC/USDT:USDT short qty=0.01 @ 1 (max hold time) pnl≈-3.00",
]


def test_parse_stats_counts_today_and_total():
    st = gui.parse_stats(LOG, D)
    assert (st.opens_today, st.closes_today, st.wins_today) == (1, 2, 1)
    assert st.pnl_today == pytest.approx(0.82) and st.pnl_total == pytest.approx(-2.18)
    assert st.win_rate == "50%"


def test_friendly_lines():
    assert gui.friendly(LOG[0]).startswith("15:12  XRP 挂单做多 @ 1.3360")
    assert "约 -1.49 USDT" in gui.friendly(LOG[2])
    assert "跟踪止损" in gui.friendly(LOG[3])
    assert gui.friendly("random text") == "random text"


def test_pid_file_detection(tmp_path):
    f = tmp_path / "x.pid"
    f.write_text(str(os.getpid()))
    assert gui.running_pid(f) == os.getpid()
    f.write_text("999999")
    assert gui.running_pid(f) is None


def test_read_paths_for_both_modes():
    log, pid, mode = gui.read_paths("config_hf.yaml")
    assert log.name == "bot_hf.log" and pid.name == "state_hf.pid" and mode in ("paper", "demo", "live")
    assert gui.read_paths("config.yaml")[1].name == "state.pid"


def test_settings_roundtrip_and_run_command(tmp_path):
    f = tmp_path / "s.json"
    assert gui.load_settings(f) == {"coins": list(gui.COINS), "margin": 10.0}
    gui.save_settings({"coins": ["BTC", "SOL"], "margin": 12}, f)
    assert gui.load_settings(f) == {"coins": ["BTC", "SOL"], "margin": 12.0}
    f.write_text('{"coins": ["XXX"], "margin": 99}')
    assert gui.load_settings(f) == {"coins": list(gui.COINS), "margin": 15.0}
    cmd = gui.run_command("config_hf.yaml", ["BTC", "SOL"], 12.0)
    assert cmd[-6:] == ["--config", "config_hf.yaml", "--coins", "BTC,SOL", "--margin", "12"]
