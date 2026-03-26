import os
import ssl
import urllib3
from dotenv import load_dotenv

load_dotenv()

# Disable SSL verification for corporate proxy/firewall
os.environ["GRPC_SSL_CIPHER_SUITES"] = "HIGH"
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
from requests.adapters import HTTPAdapter

API_KEY = os.getenv("GOOGLE_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"


def read_logs():
    with open("logs/application.log", "r") as file:
        logs = file.read()
    return logs


def analyze_logs(log_text):
    prompt = f"""
You are an expert DevOps engineer.
Analyze the following system logs and return:
1. A short summary of the problem
2. List of detected errors
3. Possible root cause
4. Suggested troubleshooting step
Return output in structured format.

Logs:
{log_text}
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(API_URL, json=payload, verify=False)
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


if __name__ == "__main__":
    logs = read_logs()
    print("\nAnalyzing Logs...\n")
    analysis = analyze_logs(logs)
    print("Log Analysis Result:\n")
    print(analysis)
