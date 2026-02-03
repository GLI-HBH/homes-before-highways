import os
import time
import requests

URL = os.environ["STREAMLIT_APP_URL"]
TIMEOUT = int(os.getenv("PING_TIMEOUT", "20"))
TRIES = int(os.getenv("PING_TRIES", "2"))

def ping_once():
    r = requests.get(URL, timeout=TIMEOUT, headers={"User-Agent": "keepalive-bot"})
    return r.status_code

if __name__ == "__main__":
    for i in range(TRIES):
        try:
            code = ping_once()
            print(f"Ping {i+1}/{TRIES}: {code}")
        except Exception as e:
            print(f"Ping {i+1}/{TRIES} failed: {e}")
        time.sleep(2)