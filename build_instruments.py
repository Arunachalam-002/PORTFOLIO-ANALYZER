# build_instruments.py
from dotenv import load_dotenv
load_dotenv()

# call the analyzer helper to build instrument list
from utils import analyzer

print("Building Kite instrument map/list (this uses your .env KITE keys)...")
analyzer._build_instrument_map(force_refresh=True)
print("Done. Files should be in data/kite_instruments_list.json and data/kite_instruments_map.json")
