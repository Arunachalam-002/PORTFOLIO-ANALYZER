import json
from pathlib import Path

p = Path("data/kite_instruments_list.json")

if not p.exists():
    print("Instrument list missing at:", p)
    raise SystemExit(1)

instruments = json.loads(p.read_text(encoding="utf8"))

def search(query):
    q = query.upper()
    matches = []
    for it in instruments:
        ts = (it.get("tradingsymbol") or "").upper()
        name = (it.get("name") or "").upper()
        exch = (it.get("exchange") or "").upper()
        token = it.get("instrument_token")
        if q in ts or q in name:
            matches.append((ts, name, exch, token))
    return matches

queries = ["TATAMOTORS", "TATAMTR", "TATA MOTORS", "TATAM"]
for q in queries:
    print("-----------", q, "-----------")
    res = search(q)
    if not res:
        print("No matches")
    else:
        for ts, name, exch, token in res[:20]:
            print(ts, "|", name, "|", exch, "| token:", token)
    print()
