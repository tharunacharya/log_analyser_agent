import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    logs = read_logs()
    print("\nAnalyzing Logs...\n")
    analysis = analyze_logs(logs)
    print("Log Analysis Result:\n")
    print(analysis)
