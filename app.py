import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import json

# ==============================================================
# Hugging Face Spaces — BiLSTM + Attention MCQ Solver (From Scratch)
# ==============================================================
# Files required in this Space:
#   1. app.py              (this file)
#   2. requirements.txt    (torch, gradio)
#   3. bilstm_scratch_fold2_best.pt   (model weights from Kaggle)
#   4. vocab.json          (tokenizer vocabulary from Kaggle)
# ==============================================================

WEIGHTS_PATH = "bilstm_scratch_fold2_best.pt"
VOCAB_PATH   = "vocab.json"
MAX_LEN      = 256
OPTION_LABELS = ['A', 'B', 'C', 'D', 'E']

device = torch.device("cpu")  # Free Spaces don't have GPUs


# ── Model Architecture (must match training code exactly) ────

class SelfAttention(nn.Module):
    """Single-head additive attention for pooling variable-length sequences."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, x, mask=None):
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class BiLSTMForMCQ(nn.Module):
    """
    Built-from-scratch MCQ model:
      Embedding → BiLSTM → Self-Attention → MLP → 5-way score
    """
    def __init__(
        self,
        vocab_size    : int,
        embed_dim     : int   = 128,
        hidden_dim    : int   = 256,
        num_layers    : int   = 2,
        dropout       : float = 0.3,
        num_options   : int   = 5,
    ):
        super().__init__()
        self.num_options = num_options
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim // 2,
            num_layers    = num_layers,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            batch_first   = True,
        )
        self.attention = SelfAttention(hidden_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        B, C, L = input_ids.shape
        ids  = input_ids.view(B * C, L)
        mask = attention_mask.view(B * C, L) if attention_mask is not None else None
        emb    = self.embedding(ids)
        out, _ = self.lstm(emb)
        pooled = self.attention(out, mask)
        scores = self.head(pooled).squeeze(-1)
        logits = scores.view(B, C)
        return logits


# ── Simple Tokenizer (loads saved vocabulary) ────────────────

class SimpleTokenizer:
    def __init__(self, vocab_path: str, max_len: int = 256):
        with open(vocab_path, 'r') as f:
            self.word2idx = json.load(f)
        self.max_len = max_len
        print(f"Loaded vocabulary: {len(self.word2idx)} words")

    def encode(self, text: str):
        tokens = str(text).lower().split()[:self.max_len]
        ids = [self.word2idx.get(w, 1) for w in tokens]  # 1 = [UNK]
        ids = ids + [0] * (self.max_len - len(ids))       # 0 = [PAD]
        return ids

    @property
    def vocab_size(self):
        return len(self.word2idx)


# ── Load Model and Tokenizer ────────────────────────────────

print("Loading vocabulary...")
tokenizer = SimpleTokenizer(VOCAB_PATH, max_len=MAX_LEN)

print("Building model...")
model = BiLSTMForMCQ(
    vocab_size = tokenizer.vocab_size,
    embed_dim  = 128,
    hidden_dim = 256,
    num_layers = 2,
    dropout    = 0.3,
)

print("Loading trained weights...")
try:
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)

    # Strip 'module.' prefix if model was saved with DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print("Weights loaded successfully! ✅")
except Exception as e:
    print(f"ERROR loading weights: {e}")

model.to(device)
model.eval()


# ── Prediction Function ─────────────────────────────────────

def predict_mcq(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    if not prompt or not all([opt_a, opt_b, opt_c, opt_d, opt_e]):
        return {"Error": 1.0}

    options = [opt_a, opt_b, opt_c, opt_d, opt_e]

    all_ids  = []
    all_mask = []
    for opt in options:
        text = prompt + ' [SEP] ' + opt
        ids  = tokenizer.encode(text)
        mask = [1 if t != 0 else 0 for t in ids]
        all_ids.append(ids)
        all_mask.append(mask)

    input_ids      = torch.tensor([all_ids],  dtype=torch.long).to(device)   # [1, 5, L]
    attention_mask = torch.tensor([all_mask], dtype=torch.long).to(device)   # [1, 5, L]

    with torch.no_grad():
        logits = model(input_ids, attention_mask)  # [1, 5]
        probs  = torch.nn.functional.softmax(logits[0], dim=0).numpy()

    results = {OPTION_LABELS[i]: float(probs[i]) for i in range(5)}
    return results


# ── Gradio Interface ─────────────────────────────────────────

demo = gr.Interface(
    fn=predict_mcq,
    title="🧠 Smart MCQ Solver (BiLSTM + Attention)",
    description=(
        "A multiple-choice question solver built entirely from scratch using PyTorch.\n\n"
        "**Architecture:** Embedding → Bidirectional LSTM → Self-Attention → MLP\n\n"
        "Enter a question and 5 options below. The model will rank the most likely correct answers."
    ),
    inputs=[
        gr.Textbox(label="Question / Prompt", lines=4, placeholder="Type your question here..."),
        gr.Textbox(label="Option A", placeholder="First option"),
        gr.Textbox(label="Option B", placeholder="Second option"),
        gr.Textbox(label="Option C", placeholder="Third option"),
        gr.Textbox(label="Option D", placeholder="Fourth option"),
        gr.Textbox(label="Option E", placeholder="Fifth option"),
    ],
    outputs=gr.Label(num_top_classes=3, label="Top 3 Predictions"),
    theme=gr.themes.Soft(),
    examples=[
        [
            "Which of the following sorting algorithms has the best worst-case time complexity?",
            "Quick Sort",
            "Merge Sort",
            "Bubble Sort",
            "Insertion Sort",
            "Selection Sort",
        ],
        [
            "What is the primary function of mitochondria in a cell?",
            "Protein synthesis",
            "Energy production (ATP)",
            "Cell division",
            "DNA replication",
            "Waste removal",
        ],
    ],
)

if __name__ == "__main__":
    demo.launch()