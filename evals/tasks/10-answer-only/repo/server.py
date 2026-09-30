import os

DEFAULT_PORT = 8000

def port():
    return int(os.environ.get("APP_PORT", "8443"))
