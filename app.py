import os
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path(os.getenv("MODEL_PATH", "./qwen3_merged"))
SYSTEM_PROMPT = (
    "You are an IT helpdesk ticket routing assistant. "
    "Given a support ticket, respond with exactly one of the following categories: "
    "Support general, Fileservice, O365, EOL, Software, Active Directory, Computer-Services."
)
LABEL_TOKENS: List[str] = [
    "Support general",
    "Fileservice",
    "O365",
    "EOL",
    "Software",
    "Active Directory",
    "Computer-Services",
]

TEAM_BY_CATEGORY = {
    "Support general": "General Support",
    "Fileservice": "Fileservice Team",
    "O365": "O365 Team",
    "EOL": "EOL Team",
    "Software": "Software Support",
    "Active Directory": "Identity & Access",
    "Computer-Services": "Hardware Support",
}

PRIORITY_BY_CATEGORY = {
    "Support general": "Low",
    "Fileservice": "Medium",
    "O365": "Medium",
    "EOL": "High",
    "Software": "Medium",
    "Active Directory": "High",
    "Computer-Services": "Medium",
}

app = FastAPI(
    title="Support Ticket Router",
    description="A minimal FastAPI service for classifying IT helpdesk tickets using a merged Qwen3 model.",
)

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def build_prompt(ticket_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Support Ticket: {ticket_text}"},
    ]
    if hasattr(_tokenizer, "apply_chat_template"):
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])


def classify_ticket_text(ticket_text: str, compute_confidence: bool = True):
    prompt = build_prompt(ticket_text)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model_device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    generated = _tokenizer.decode(
        out.sequences[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    matched = next(
        (label for label in LABEL_TOKENS if generated.lower().startswith(label.lower())),
        None,
    )
    if matched is None:
        matched = "Support general"

    if not compute_confidence or not out.scores:
        return matched, 1.0

    first_scores = out.scores[0][0]
    label_first_ids = [
        _tokenizer.encode(label, add_special_tokens=False)[0] for label in LABEL_TOKENS
    ]
    label_probs = torch.softmax(first_scores[label_first_ids], dim=-1).cpu().tolist()
    return matched, label_probs[LABEL_TOKENS.index(matched)]


class TicketRequest(BaseModel):
    text: str


class TicketResponse(BaseModel):
    category: str
    team: str
    priority: str
    confidence: float
    model_path: str
    device: str


class BatchTicketRequest(BaseModel):
    texts: List[str]


@app.on_event("startup")
def startup_event():
    global _tokenizer, _model, _model_device

    _tokenizer, _model, _model_device = load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH.resolve()} on {_model_device}")


@app.get("/")
def read_root():
    return {
        "service": "Support Ticket Router",
        "model_path": str(MODEL_PATH.resolve()),
        "device": str(_model_device),
        "labels": LABEL_TOKENS,
    }


@app.get("/preview", response_class=HTMLResponse)
def preview_page():
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Support Ticket Router Preview</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; line-height: 1.6; padding: 0 1rem; }
    textarea { width: 100%; min-height: 140px; font-size: 1rem; padding: 0.75rem; }
    pre { background: #f4f4f4; padding: 1rem; border-radius: 6px; white-space: pre-wrap; word-break: break-word; }
    button { padding: 0.75rem 1.2rem; font-size: 1rem; margin-top: 0.75rem; }
  </style>
</head>
<body>
  <h1>Support Ticket Router Preview</h1>
  <p>Enter one or more support ticket texts below, one per line. Submit to get predicted categories and confidence scores.</p>
  <textarea id="tickets" placeholder="Enter ticket text, one per line..."></textarea>
  <br />
  <button onclick="sendBatch()">Classify Batch</button>
  <h2>Results</h2>
  <pre id="output">No results yet.</pre>
  <script>
    async function sendBatch() {
      const textarea = document.getElementById('tickets');
      const output = document.getElementById('output');
      const lines = textarea.value.split('\n').map(l => l.trim()).filter(Boolean);
      if (!lines.length) {
        output.textContent = 'Please add one or more ticket texts.';
        return;
      }
      output.textContent = 'Calling /batch_classify...';
      try {
        const res = await fetch('/batch_classify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ texts: lines }),
        });
        const data = await res.json();
        if (!res.ok) throw data;
        output.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        output.textContent = 'Error:\n' + JSON.stringify(err, null, 2);
      }
    }
  </script>
</body>
</html>"""
    return HTMLResponse(content=html, status_code=200)


@app.post("/batch_classify", response_model=List[TicketResponse])
def batch_classify(request: BatchTicketRequest):
    if not request.texts or any(not text.strip() for text in request.texts):
        raise HTTPException(status_code=400, detail="Ticket texts must be a non-empty list of strings.")

    results = []
    for text in request.texts:
        category, confidence = classify_ticket_text(text)
        results.append(
            TicketResponse(
                category=category,
                team=TEAM_BY_CATEGORY.get(category, "General Support"),
                priority=PRIORITY_BY_CATEGORY.get(category, "Low"),
                confidence=float(confidence),
                model_path=str(MODEL_PATH.resolve()),
                device=str(_model_device),
            )
        )
    return results


@app.post("/classify", response_model=TicketResponse)
def classify_ticket(request: TicketRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Ticket text must be non-empty.")

    category, confidence = classify_ticket_text(request.text)
    return TicketResponse(
        category=category,
        team=TEAM_BY_CATEGORY.get(category, "General Support"),
        priority=PRIORITY_BY_CATEGORY.get(category, "Low"),
        confidence=float(confidence),
        model_path=str(MODEL_PATH.resolve()),
        device=str(_model_device),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080)
