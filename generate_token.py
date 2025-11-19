from kiteconnect import KiteConnect
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("KITE_API_KEY")
api_secret = os.getenv("KITE_API_SECRET")

kite = KiteConnect(api_key=api_key)

print("Login URL:")
print(kite.login_url())

request_token = input("Paste request_token from redirect URL: ").strip()
data = kite.generate_session(request_token, api_secret=api_secret)

print("ACCESS TOKEN:", data["access_token"])
