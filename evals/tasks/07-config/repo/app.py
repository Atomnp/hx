import json

cfg = json.load(open("config.json"))
print(f"serving on {cfg['host']}:{cfg['port']}")
