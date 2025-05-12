import csv
import json
import requests

API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
SYSTEM_MSG = {
  "role":"system",
  "content":
    ""
}

MAX_TOKENS = 128
TEMPERATURE = 0.5

def format_record(row: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in row.items() if k != " Label")

def classify_row(row: dict):
    payload = {
        "model": MODEL,
        "messages": [
            SYSTEM_MSG,
            {"role": "user", "content": f"{format_record(row)}"}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE, 
    }
    # print(format_record(row))
    # print("\n")
    # print("\n")
    resp = requests.post(API_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def loadDataset(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)

def evaluate(csv_path):
    data = loadDataset(csv_path)
    correct = 0
    total = 0
    for row in data:
        total += 1
        prediction = classify_row(row)
        label = row[' Label']
        if label.lower() in prediction.lower():
            correct += 1
            print(f"Correctly classified: {label} -> {prediction}")
        else:
            print(f"Incorrectly classified: {label} -> {prediction}")

        print(f"Total: {correct}/{total}\n ------\n")


if __name__ == "__main__":
    evaluate("Friday-WorkingHours-Morning.pcap_ISCX.csv")

