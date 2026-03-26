import os
import httpx
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={"api_version": "v1beta"},
    httpx_client=httpx.Client(verify=False)
)


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
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text


if __name__ == "__main__":
    logs = read_logs()
    print("\nAnalyzing Logs...\n")
    analysis = analyze_logs(logs)
    print("Log Analysis Result:\n")
    print(analysis)
