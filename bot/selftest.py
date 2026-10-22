"""测试下单: walk every order type the bot uses through the Bitget DEMO account in under a
minute, instead of waiting hours for a signal to find out whether Bitget accepts them."""
from __future__ import annotations

import time


def run_selftest(broker, symbol: str, margin: float, pause: float = 2.0) -> list[tuple[str, bool, str]]:
    """Returns (step, ok, detail) for each step. Always leaves no order or position behind."""
    out: list[tuple[str, bool, str]] = []

    def step(name, fn):
        try:
            detail = fn()
            out.append((name, True, detail or ""))
            return True
        except Exception as e:  # noqa: BLE001 - report every failure, keep cleaning up
            out.append((name, False, f"{type(e).__name__}: {e}"))
            return False

    if symbol in broker.positions():
        return [("开始前检查", False, f"{symbol} 已经有持仓，先平掉或停止机器人再测")]
    broker.setup_symbol(symbol)
    price = broker.price(symbol)
    lev = broker.leverage_for(symbol) if hasattr(broker, "leverage_for") else 100
    qty = broker.round_qty(symbol, margin * lev / price)
    if qty <= 0:
        return [("计算数量", False, f"保证金 {margin} 低于 {symbol} 的最小下单量")]
    entry_id = {}

    def maker_entry():
        limit = price * 0.995  # well below the market: rests, never fills
        entry_id["id"] = broker.open_maker(symbol, "long", qty, limit, price * 0.99, 0.0)
        time.sleep(pause)
        status, _, _ = broker.order_status(symbol, entry_id["id"])
        if status != "open":
            raise RuntimeError(f"挂单状态是 {status}，不是挂着")
        return f"挂单买入 {qty} @ {limit:.10g}，带止损"

    def cancel_entry():
        broker.cancel_order(symbol, entry_id["id"])
        time.sleep(pause)
        status, _, _ = broker.order_status(symbol, entry_id["id"])
        if status == "open":
            raise RuntimeError("撤单后还挂着")
        return "已撤"

    step("1. 挂单进场（只做 maker）+ 止损", maker_entry)
    if entry_id:
        step("2. 撤掉没成交的挂单", cancel_entry)

    opened = step("3. 市价开仓 + 交易所止损", lambda: (
        f"开多 {qty} @ {broker.open(symbol, 'long', qty, price * 0.994, 0.0):.10g}"))
    tp_id = {}
    try:
        if opened:
            time.sleep(pause)
            if symbol not in broker.positions():
                out.append(("3b. 查到持仓", False, "下单后查不到持仓"))

            def place_tp():
                tp = price * 1.004
                tp_id["id"] = broker.place_tp(symbol, "long", qty, tp)
                time.sleep(pause)
                status, _, _ = broker.order_status(symbol, tp_id["id"])
                if status != "open":
                    raise RuntimeError(f"止盈单状态是 {status}")
                return f"只减仓限价止盈 @ {tp:.10g} 已挂"

            if step("4. 挂单止盈（只减仓、只做 maker）", place_tp):
                step("5. 撤掉止盈单", lambda: broker.cancel_order(symbol, tp_id["id"]) or "已撤")
    finally:
        pos = broker.positions().get(symbol)
        if pos is not None:
            step("6. 市价平仓 + 撤掉止损", lambda: f"平仓 @ {broker.close(symbol, pos):.10g}")
        elif opened:
            out.append(("6. 市价平仓", False, "持仓已经不在了"))
        if tp_id:
            try:
                broker.cancel_order(symbol, tp_id["id"])
            except Exception:  # noqa: BLE001 - already gone is fine
                pass
    return out


def report(symbol: str, results: list[tuple[str, bool, str]]) -> str:
    lines = [f"测试下单：{symbol.split('/')[0]}，Bitget 模拟盘", ""]
    for name, ok, detail in results:
        lines.append(f"{'✓' if ok else '✗'} {name}" + (f"：{detail}" if detail else ""))
    bad = [r for r in results if not r[1]]
    lines += ["", "全部通过，机器人用到的下单方式 Bitget 都收。" if not bad else
              f"有 {len(bad)} 步没通过，把这个窗口截图发给 Claude。"]
    return "\n".join(lines)
