# capture_token.py — run this, then open Kite login URL and complete login/authorize
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import sys

PORT = 5000

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        req_token = qs.get("request_token", [""])[0]
        # Print info to terminal
        print("----- Incoming redirect -----")
        print("Path:", self.path)
        print("Parsed query:", qs)
        if req_token:
            print("\nREQUEST_TOKEN:", req_token)
        else:
            print("\nNo request_token found in URL.")
        # Respond with a friendly page
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        html = "<html><body><h2>Got it — you can close this tab.</h2><pre>{}</pre></body></html>".format(self.path)
        self.wfile.write(html.encode("utf-8"))

if __name__ == "__main__":
    try:
        server = HTTPServer(("127.0.0.1", PORT), Handler)
        print(f"Listening on http://127.0.0.1:{PORT}/ — waiting for Kite redirect...")
        server.handle_request()  # handle a single request then exit
    except OSError as e:
        print("Failed to start server:", e)
        sys.exit(1)
