"""Developer tool: posts fake wristband readings to /api/wristband/data, as the ESP32 will.

Get a device ID and API key from the patient's Wristband page ("Generate Pairing Code"), then:
    venv\\Scripts\\python scripts\\wristband_simulate.py --device-id ABC123DEF456 --api-key <key>

Sends one reading every 5 seconds for a minute by default. Not a route; never runs on the server.
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST',
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, res.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device-id', required=True)
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:5000/api/wristband/data')
    parser.add_argument('--interval', type=float, default=5, help='seconds between readings (default 5)')
    parser.add_argument('--duration', type=float, default=60, help='total seconds to run (default 60)')
    args = parser.parse_args()

    # A gentle random walk, so the readings look like one person rather than noise
    gsr, heart_rate, temperature = 450.0, 78.0, 36.6
    start = time.time()
    count = int(args.duration // args.interval) + 1   # e.g. 60s every 5s: readings at 0, 5, ... 60
    sent = 0
    for i in range(count):
        # Fixed timetable, so slow responses don't stretch the interval
        time.sleep(max(0, start + i * args.interval - time.time()))
        gsr = min(max(gsr + random.uniform(-25, 25), 150), 900)
        heart_rate = min(max(heart_rate + random.uniform(-3, 3), 55), 130)
        temperature = min(max(temperature + random.uniform(-0.1, 0.1), 35.5), 37.8)
        payload = {'device_id': args.device_id, 'api_key': args.api_key, 'gsr': round(gsr),
                   'heart_rate': round(heart_rate), 'temperature': round(temperature, 1),
                   'timestamp': int(time.time())}
        try:
            status, body = post(args.url, payload)
        except urllib.error.URLError as e:
            print(f'Could not reach {args.url}: {e.reason}. Is the Flask app running?')
            return
        sent += 1
        print(f"#{sent} GSR {payload['gsr']}  HR {payload['heart_rate']}  temp {payload['temperature']} C  -> {status} {body.strip()}")
        if status == 401:
            print('Device ID or API key was rejected. Generate new credentials on the Wristband page.')
            return
    print(f'Done: {sent} reading(s) sent.')


if __name__ == '__main__':
    main()
