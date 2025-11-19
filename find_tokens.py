# find_tokens.py
import json
from pathlib import Path
p_map = Path("data/kite_instruments_map.json")
p_list = Path("data/kite_instruments_list.json")
if not p_list.exists():
    print("Instrument list not present. Start app once to build instrument list (it will create data/kite_instruments_list.json).")
    raise SystemExit(1)
instruments = json.loads(p_list.read_text(encoding="utf8"))
def search(q):
    q = q.strip().upper()
    out = []
    for it in instruments:
        ts = (it.get("tradingsymbol") or "").upper()
        name = (it.get("name") or "").upper()
        exch = (it.get("exchange") or "").upper()
        token = it.get("instrument_token")
        if q in ts or q in name:
            out.append((ts, name, exch, token))
    return out

queries = ["TATAMOTORS", "TATAMTR", "TATA MOTORS", "NIFTY", "NSEI", "NIFTY 50"]
for q in queries:
    print("----", q, "----")
    for r in search(q)[:15]:
        print(r)
    print()
