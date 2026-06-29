import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
CSV_CANDIDATES = [
    ROOT / "support_tickets.csv",
    ROOT / "support_ticket.csv",
    ROOT / "data" / "support_tickets.csv",
]

for candidate in CSV_CANDIDATES:
    if candidate.exists():
        CSV_PATH = candidate
        break
else:
    raise FileNotFoundError("Could not find support_tickets.csv in the workspace.")

LABEL2ID = {
    "Support general": 0,
    "Fileservice": 1,
    "O365": 2,
    "EOL": 3,
    "Software": 4,
    "Active Directory": 5,
    "Computer-Services": 6,
}

SYSTEM_PROMPT = (
    "You are an IT helpdesk ticket routing assistant. "
    "Given a support ticket, respond with exactly one of the following categories: "
    "Support general, Fileservice, O365, EOL, Software, Active Directory, Computer-Services."
)

LLAMA_DATA_DIR = ROOT / "LLaMA-Factory" / "data"
LLAMA_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_JSON_PATH = LLAMA_DATA_DIR / "TRAIN.json"
DATASET_INFO_PATH = LLAMA_DATA_DIR / "dataset_info.json"
VAL_CSV_PATH = ROOT / "val_split.csv"

print(f"Using CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH).rename(columns={"category_truth": "label"})
df = df[df["label"].isin(LABEL2ID)].sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Loaded {len(df):,} rows")
print(df["label"].value_counts())

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print(f"\nTrain: {len(train_df):,} rows | Val (held-out): {len(val_df):,} rows")

sharegpt_records = [
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Support Ticket: {row['text']}"},
            {"role": "assistant", "content": row["label"]},
        ]
    }
    for _, row in train_df.iterrows()
]

TRAIN_JSON_PATH.write_text(json.dumps(sharegpt_records, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nWritten {len(sharegpt_records):,} records → {TRAIN_JSON_PATH}")

if DATASET_INFO_PATH.exists():
    with DATASET_INFO_PATH.open("r", encoding="utf-8") as f:
        info = json.load(f)
else:
    info = {}

info["support_tickets"] = {
    "file_name": "TRAIN.json",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    },
}

DATASET_INFO_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Registered 'support_tickets' in {DATASET_INFO_PATH}")

val_df.to_csv(VAL_CSV_PATH, index=False)
print(f"Val split saved → {VAL_CSV_PATH} ({len(val_df):,} rows)")
