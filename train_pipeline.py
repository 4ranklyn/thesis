import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')

# Seed global (Wajib untuk reproduktibilitas hasil skripsi)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class Config:
    # Sesuaikan path jika dijalankan lokal atau Kaggle
    IMAGE_DIR = "./dataset/Images"
    CSV_FILE = "./dataset/metadata.csv"

    # Global Hyperparameters (Sama untuk semua model)
    BATCH_SIZE = 8
    EPOCHS = 7
    LEARNING_RATE = 5e-5
    MAX_LENGTH = 64
    FP16 = True
    WEIGHT_DECAY = 0.01

    # Folder Branching
    BASE_EXPORT_DIR = "./final_thesis_models"

MODEL_DIRS = {
    "vit_indobert": os.path.join(Config.BASE_EXPORT_DIR, "vit_indobert"),
    "vit_gpt2": os.path.join(Config.BASE_EXPORT_DIR, "vit_gpt2"),
    "swin_indobartv2": os.path.join(Config.BASE_EXPORT_DIR, "swin_indobartv2"),
    "vit_indobartv2": os.path.join(Config.BASE_EXPORT_DIR, "vit_indobartv2"),
    "swin_gpt2": os.path.join(Config.BASE_EXPORT_DIR, "swin_gpt2"),
    "swin_indobert": os.path.join(Config.BASE_EXPORT_DIR, "swin_indobert")
}

for path in MODEL_DIRS.values():
    os.makedirs(path, exist_ok=True)

print(f"✅ Setup direktori & seed selesai. Branching models akan disimpan di: {Config.BASE_EXPORT_DIR}")

import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split

print("🚀 Memulai Statistical Noise Reduction...")
df = pd.read_csv(Config.CSV_FILE)

# --- ALGORITMA STATISTICAL SPELL CHECKER ---
all_text = " ".join(df['text'].astype(str).tolist()).lower()
words = re.findall(r'\w+', all_text)
WORD_COUNTS = Counter(words)
TOTAL_WORDS = sum(WORD_COUNTS.values())

def probability(word): return WORD_COUNTS[word] / TOTAL_WORDS

def known(words): return set(w for w in words if w in WORD_COUNTS and WORD_COUNTS[w] > 2)

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def correction(word):
    if WORD_COUNTS[word] > 2: return word
    candidates = (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
    return max(candidates, key=probability)

def clean_and_correct(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    corrected = [correction(w) for w in text.split()]
    return " ".join(corrected)

# ====== PROSES METRIK UNTUK VISUALISASI ======
raw_texts = df['text'].astype(str).tolist()
cleaned_texts = []
typo_map = {}
typo_count = Counter()

print("Memproses dataset & mengumpulkan metrik typo (ini mungkin butuh beberapa detik)...")
for text in raw_texts:
    text_lower = str(text).lower()
    text_clean = re.sub(r'[^a-z0-9\s]', '', text_lower)
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    corrected_words = []
    for w in text_clean.split():
        c = correction(w)
        corrected_words.append(c)
        if c != w:
            typo_map[w] = c
            typo_count[w] += 1
    cleaned_texts.append(" ".join(corrected_words))

df['text'] = cleaned_texts

# Simpan metrik perbandingan (diakses oleh cell visualisasi berikutnya)
raw_words = re.findall(r'\w+', ' '.join(raw_texts).lower())
clean_words = re.findall(r'\w+', ' '.join(cleaned_texts).lower())
raw_freq = Counter(raw_words).most_common(15)
clean_freq = Counter(clean_words).most_common(15)
top_typos = typo_count.most_common(10)
# ===============================================

df = df.dropna(subset=['text'])
print(f"✅ Data bersih siap digunakan! Total sampel: {len(df)}")

# --- FIX DATA LEAKAGE: SPLIT PER IMAGE (70:15:15) ---
# Flickr8k has ~5 captions per image. Splitting on rows causes leakage.
# Instead, split on unique file_name so all captions for one image stay together.
unique_images = df['file_name'].unique()
print(f"\n📌 Jumlah gambar unik: {len(unique_images)}")

train_imgs, temp_imgs = train_test_split(unique_images, test_size=0.3, random_state=42)
val_imgs, test_imgs   = train_test_split(temp_imgs,   test_size=0.5, random_state=42)

train_df = df[df['file_name'].isin(train_imgs)].reset_index(drop=True)
val_df   = df[df['file_name'].isin(val_imgs)].reset_index(drop=True)
test_df  = df[df['file_name'].isin(test_imgs)].reset_index(drop=True)

print("📊 Distribusi Dataset Final (split per gambar — TANPA data leakage):")
print(f"Train (70%): {len(train_df)} rows ({len(train_imgs)} images)")
print(f"Val   (15%): {len(val_df)} rows ({len(val_imgs)} images)")
print(f"Test  (15%): {len(test_df)} rows ({len(test_imgs)} images)")

# Verifikasi: pastikan tidak ada overlap
assert len(set(train_imgs) & set(val_imgs)) == 0, "LEAKAGE: train ∩ val"
assert len(set(train_imgs) & set(test_imgs)) == 0, "LEAKAGE: train ∩ test"
assert len(set(val_imgs) & set(test_imgs)) == 0, "LEAKAGE: val ∩ test"
print("✅ Verifikasi: Tidak ada overlap gambar antar set — data leakage FIXED.")

import os
import matplotlib.pyplot as plt
import seaborn as sns

palette_gray = sns.color_palette('Greys_r', 15)

# ==========================================
# GAMBAR 1: Sebelum Pembersihan
# ==========================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
words_b, counts_b = zip(*raw_freq)
sns.barplot(x=list(counts_b), y=list(words_b), hue=list(words_b), palette=palette_gray, ax=ax1, orient='h', legend=False)
ax1.set_title('Top 15 Kata — Sebelum Pembersihan', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Frekuensi', fontsize=11)
ax1.set_ylabel('')
for i, v in enumerate(counts_b):
    ax1.text(v + max(counts_b)*0.01, i, str(v), va='center', fontsize=9, color='#333')

# Keterangan di bawah plot
fig1.text(0.5, -0.05, "(a) Distribusi frekuensi top 15 kata (sebelum pembersihan).", ha='center', fontsize=11, style='italic')

plt.tight_layout()
try:
    plt.savefig(os.path.join(Config.BASE_EXPORT_DIR, 'frekuensi_kata_sebelum.png'), dpi=150, bbox_inches='tight', facecolor='white')
except Exception as e:
    print(f'Savefig warning: {e}')

plt.close(fig1)

# ==========================================
# GAMBAR 2: Sesudah Pembersihan
# ==========================================
fig2, ax2 = plt.subplots(figsize=(10, 6))
words_a, counts_a = zip(*clean_freq)
sns.barplot(x=list(counts_a), y=list(words_a), hue=list(words_a), palette=palette_gray, ax=ax2, orient='h', legend=False)
ax2.set_title('Top 15 Kata — Sesudah Pembersihan', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Frekuensi', fontsize=11)
ax2.set_ylabel('')
for i, v in enumerate(counts_a):
    ax2.text(v + max(counts_a)*0.01, i, str(v), va='center', fontsize=9, color='#333')

# Keterangan di bawah plot
fig2.text(0.5, -0.05, "(b) Distribusi frekuensi top 15 kata (sesudah pembersihan/noise reduction).", ha='center', fontsize=11, style='italic')

plt.tight_layout()
try:
    plt.savefig(os.path.join(Config.BASE_EXPORT_DIR, 'frekuensi_kata_sesudah.png'), dpi=150, bbox_inches='tight', facecolor='white')
except Exception as e:
    print(f'Savefig warning: {e}')

plt.close(fig2)

# ==========================================
# TABEL: Anomali/Typo (Pandas DataFrame)
# ==========================================
import pandas as pd
from IPython.display import display, HTML

print('\n➤ Top 10 Kata Anomali/Typo dan Hasil Koreksinya:')
if top_typos:
    table_data = []
    for rank, (typo, count) in enumerate(top_typos, 1):
        table_data.append([rank, typo, typo_map.get(typo, '-'), count])

    typo_df = pd.DataFrame(table_data, columns=['No.', 'Kata Asli (Typo)', 'Hasil Koreksi', 'Frekuensi'])
    typo_df.set_index('No.', inplace=True)

    # Tampilkan dataframe di output cell
    print(typo_df)

    # Simpan sebagai CSV (menggantikan gambar)
    try:
        typo_df.to_csv(os.path.join(Config.BASE_EXPORT_DIR, 'tabel_koreksi_typo.csv'))
    except Exception as e:
        print(f'Save CSV warning: {e}')
else:
    print('Tidak ditemukan anomali/typo.')

print(f'\nTotal kata unik (raw) : {len(set(raw_words))}')
print(f'Total kata unik (clean): {len(set(clean_words))}')
print(f'Total typo terdeteksi  : {len(typo_map)} jenis unik ({sum(typo_count.values())} kemunculan)')


from torch.utils.data import Dataset
from PIL import Image
import evaluate
import matplotlib.pyplot as plt


# --- CLASS DATASET ---
class FlickrIndoDataset(Dataset):
    def __init__(self, df, root_dir, processor, tokenizer, max_target_length=Config.MAX_LENGTH):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_target_length

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.loc[idx, 'file_name'])
        if "/" in img_name: img_name = img_name.split("/")[-1]
        image_path = os.path.join(self.root_dir, img_name)

        pixel_values = None
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
            except: pass
        if pixel_values is None:
            pixel_values = torch.zeros((3, 224, 224))

        caption_text = self.df.loc[idx, 'text']
        caption_text = f"<s> {caption_text} </s>"

        labels = self.tokenizer(
            caption_text, padding="max_length", truncation=True,
            max_length=self.max_length, add_special_tokens=True
        ).input_ids

        labels = [l if l != self.tokenizer.pad_token_id else -100 for l in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

# --- CUSTOM COLLATOR ---
class SmartDataCollator:
    def __init__(self, model):
        self.model = model

    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
        return {"pixel_values": pixel_values, "labels": labels, "decoder_input_ids": decoder_input_ids}


# Load metrik untuk training evaluation (BLEU)
bleu_metric = evaluate.load("bleu")

def compute_metrics_bleu(eval_pred, tokenizer):
    preds, labels = eval_pred

    # Handle jika preds berupa tuple (tergantung versi transformers HF)
    if isinstance(preds, tuple):
        preds = preds[0]

    # Masking nilai -100 menjadi pad_token_id untuk preds DAN labels
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode setelah aman dari integer negatif
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    # BLEU membutuhkan referensi dalam format list of lists
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": round(result["bleu"] * 100, 4)}


def plot_training_history(trainer, model_name):
    history = trainer.state.log_history

    # Ekstrak metrik
    train_steps = [x['step'] for x in history if 'loss' in x]
    train_loss = [x['loss'] for x in history if 'loss' in x]
    val_epochs = [x['epoch'] for x in history if 'eval_loss' in x]
    val_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    val_bleu = [x['eval_bleu'] for x in history if 'eval_bleu' in x]

    # --- Plot 1: Loss ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Step / Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(train_steps, train_loss, 'r--', alpha=0.7, label='Train Loss (per step)')
    if val_epochs and val_loss:
        # Konversi epoch ke step terdekat untuk alignment visual
        if train_steps:
            steps_per_epoch = train_steps[-1] / val_epochs[-1] if val_epochs[-1] > 0 else 1
            val_steps = [int(e * steps_per_epoch) for e in val_epochs]
        else:
            val_steps = list(range(1, len(val_loss) + 1))
        ax1.plot(val_steps, val_loss, 'b-o', label='Val Loss (per epoch)', linewidth=2)
    ax1.legend(loc='best')
    ax1.set_title(f'Loss per Step/Epoch: {model_name}')
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    plt.show()

    # --- Plot 2: BLEU ---
    if val_bleu:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        epochs_range = range(1, len(val_bleu) + 1)
        ax2.plot(epochs_range, val_bleu, 'g-o', label='Val BLEU', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('BLEU Score')
        ax2.set_title(f'BLEU Score per Epoch: {model_name}')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        plt.show()


import pandas as pd

# Parameter finetuning (shared across all models)
params_df = pd.DataFrame({
    'Parameter': [
        'Image Size',
        'Max Length (teks)',
        'Batch Size',
        'Learning Rate',
        'Epochs',
        'FP16 (Mixed Precision)',
        'Weight Decay',
        'Label Smoothing',
        'Num Beams (Generation)',
        'Early Stopping',
        'No Repeat N-gram Size',
        'Eval Strategy',
        'Save Strategy',
        'Metric for Best Model',
        'Encoder Freeze Strategy',
        'Logging Steps',
    ],
    'Nilai': [
        '224 x 224',
        str(Config.MAX_LENGTH),
        str(Config.BATCH_SIZE),
        str(Config.LEARNING_RATE),
        str(Config.EPOCHS),
        str(Config.FP16),
        str(Config.WEIGHT_DECAY),
        '0.1',
        '4',
        'True',
        '2',
        'epoch',
        'epoch (save_total_limit=2)',
        'BLEU',
        'Freeze all, unfreeze last layer',
        '50',
    ]
}).set_index('Parameter')

print('Hyperparameter Finetuning (berlaku untuk semua model)')
print(params_df)


import json, datetime
from transformers import EarlyStoppingCallback

# ======================================================================
# EXPERIMENT CONFIGURATION — Per-model hyperparameters
# Modify individual entries to tune each model independently.
# The dataset split is LOCKED and identical for all models (cell above).
# ======================================================================
EXPERIMENT_CONFIGS = {
    "vit_indobert": {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "weight_decay": 0.01,
        "num_train_epochs": 10,  # EarlyStoppingCallback will cut early
        "label_smoothing_factor": 0.1,
    },
    "vit_gpt2": {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "weight_decay": 0.01,
        "num_train_epochs": 10,
        "label_smoothing_factor": 0.1,
    },
    "swin_indobartv2": {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "weight_decay": 0.01,
        "num_train_epochs": 10,
        "label_smoothing_factor": 0.1,
    },
    "vit_indobartv2": {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "weight_decay": 0.01,
        "num_train_epochs": 10,
        "label_smoothing_factor": 0.1,
    },
    "swin_gpt2": {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "weight_decay": 0.01,
        "num_train_epochs": 10,
        "label_smoothing_factor": 0.1,
    },
    "swin_indobert": {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "weight_decay": 0.01,
        "num_train_epochs": 10,
        "label_smoothing_factor": 0.1,
    },
}

# Early Stopping patience (monitor val_loss / eval_loss)
EARLY_STOPPING_PATIENCE = 3

# ======================================================================
# EXPERIMENT LOGGING — Records hyperparams, stop-epoch, final metrics
# ======================================================================
experiment_log = []

def log_experiment(model_name, config, trainer):
    """Record hyperparams, early-stop epoch, and final eval metrics."""
    history = trainer.state.log_history
    eval_entries = [e for e in history if 'eval_loss' in e]
    final_eval = eval_entries[-1] if eval_entries else {}
    entry = {
        "model_name": model_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "hyperparameters": config,
        "total_epochs_trained": int(final_eval.get("epoch", 0)),
        "early_stopped_at_epoch": int(final_eval.get("epoch", 0)),
        "final_eval_loss": final_eval.get("eval_loss"),
        "final_bleu": final_eval.get("eval_bleu"),
    }
    experiment_log.append(entry)
    log_path = os.path.join(Config.BASE_EXPORT_DIR, "experiment_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(experiment_log, f, indent=2, ensure_ascii=False)
    print(f"📝 Experiment logged for {model_name}: epoch={entry['early_stopped_at_epoch']}, "
          f"eval_loss={entry['final_eval_loss']}, BLEU={entry['final_bleu']}")

# ======================================================================
# UNIFIED TOKENIZER — IndoNLG (native to IndoBARTv2), used for ALL models
# ======================================================================
from transformers import MBartTokenizer
UNIFIED_TOKENIZER = MBartTokenizer.from_pretrained("indobenchmark/indobart-v2")
print(f"✅ Unified IndoNLG tokenizer loaded. Vocab size: {len(UNIFIED_TOKENIZER)}")

from transformers import VisionEncoderDecoderModel, AutoImageProcessor, GenerationConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer

print("🚀 Memulai Training Model ViT + IndoBERT...")

# --- Unified tokenizer (IndoNLG) for ALL models ---
tokenizer = UNIFIED_TOKENIZER
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
cfg = EXPERIMENT_CONFIGS["vit_indobert"]

# 1. Construct VisionEncoderDecoder model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k",
    "indobenchmark/indobert-base-p1"
)

# PENTING: Resize embeddings decoder agar sesuai dengan vocab IndoNLG tokenizer
model.decoder.resize_token_embeddings(len(tokenizer))
print(f"  Token embeddings decoder resized to {len(tokenizer)}")

# 2. Token alignment (using unified IndoNLG tokenizer)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
if hasattr(model.config, 'decoder') and hasattr(model.config.decoder, 'vocab_size'):
    model.config.vocab_size = len(tokenizer)

legacy_keys = ['max_length', 'early_stopping', 'num_beams', 'length_penalty', 'no_repeat_ngram_size']
for key in legacy_keys:
    if hasattr(model.config, key): delattr(model.config, key)

_bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
_eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
model.generation_config = GenerationConfig(
    max_length=Config.MAX_LENGTH, early_stopping=True, num_beams=4,
    length_penalty=1.0, no_repeat_ngram_size=2,
    bos_token_id=_bos, eos_token_id=_eos,
    pad_token_id=tokenizer.pad_token_id, decoder_start_token_id=_bos,
)

# 3. Smart Freezing
for param in model.encoder.parameters(): param.requires_grad = False
for param in model.encoder.encoder.layer[-1].parameters(): param.requires_grad = True

# 4. Setup Dataset (dataset split is LOCKED — identical for all models)
train_data = FlickrIndoDataset(train_df, Config.IMAGE_DIR, processor, tokenizer)
val_data   = FlickrIndoDataset(val_df,   Config.IMAGE_DIR, processor, tokenizer)
collator   = SmartDataCollator(model=model)

# 5. Training Arguments (from per-model config)
args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(MODEL_DIRS["vit_indobert"], "checkpoints"),
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    per_device_eval_batch_size=cfg.get("per_device_train_batch_size", Config.BATCH_SIZE),
    predict_with_generate=True,
    logging_steps=50,
    learning_rate=cfg["learning_rate"],
    num_train_epochs=cfg["num_train_epochs"],
    fp16=Config.FP16,
    weight_decay=cfg["weight_decay"],
    label_smoothing_factor=cfg.get("label_smoothing_factor", 0.1),
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    report_to="none",
)

# 6. Trainer with EarlyStoppingCallback
trainer = Seq2SeqTrainer(
    model=model, processing_class=tokenizer, args=args,
    train_dataset=train_data, eval_dataset=val_data, data_collator=collator,
    compute_metrics=lambda eval_pred: compute_metrics_bleu(eval_pred, tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

trainer.train()
plot_training_history(trainer, "ViT + IndoBERT")

# 7. Log experiment
log_experiment("vit_indobert", cfg, trainer)

# 8. Export
trainer.save_model(MODEL_DIRS["vit_indobert"])
tokenizer.save_pretrained(MODEL_DIRS["vit_indobert"])
processor.save_pretrained(MODEL_DIRS["vit_indobert"])
print(f"✅ Model ViT + IndoBERT saved to {MODEL_DIRS['vit_indobert']}")

from transformers import VisionEncoderDecoderModel, AutoImageProcessor, GenerationConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer

print("🚀 Memulai Training Model ViT + GPT2...")

# --- Unified tokenizer (IndoNLG) for ALL models ---
tokenizer = UNIFIED_TOKENIZER
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
cfg = EXPERIMENT_CONFIGS["vit_gpt2"]

# GPT2 requires explicit cross-attention config injection
from transformers import AutoConfig
gpt2_cfg = AutoConfig.from_pretrained("cahya/gpt2-small-indonesian-522M")
gpt2_cfg.is_decoder = True
gpt2_cfg.add_cross_attention = True

# 1. Construct VisionEncoderDecoder model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k",
    "cahya/gpt2-small-indonesian-522M",
    decoder_config=gpt2_cfg
)

# PENTING: Resize embeddings decoder agar sesuai dengan vocab IndoNLG tokenizer
model.decoder.resize_token_embeddings(len(tokenizer))
print(f"  Token embeddings decoder resized to {len(tokenizer)}")

# 2. Token alignment (using unified IndoNLG tokenizer)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
if hasattr(model.config, 'decoder') and hasattr(model.config.decoder, 'vocab_size'):
    model.config.vocab_size = len(tokenizer)

legacy_keys = ['max_length', 'early_stopping', 'num_beams', 'length_penalty', 'no_repeat_ngram_size']
for key in legacy_keys:
    if hasattr(model.config, key): delattr(model.config, key)

_bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
_eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
model.generation_config = GenerationConfig(
    max_length=Config.MAX_LENGTH, early_stopping=True, num_beams=4,
    length_penalty=1.0, no_repeat_ngram_size=2,
    bos_token_id=_bos, eos_token_id=_eos,
    pad_token_id=tokenizer.pad_token_id, decoder_start_token_id=_bos,
)

# 3. Smart Freezing
for param in model.encoder.parameters(): param.requires_grad = False
for param in model.encoder.encoder.layer[-1].parameters(): param.requires_grad = True

# 4. Setup Dataset (dataset split is LOCKED — identical for all models)
train_data = FlickrIndoDataset(train_df, Config.IMAGE_DIR, processor, tokenizer)
val_data   = FlickrIndoDataset(val_df,   Config.IMAGE_DIR, processor, tokenizer)
collator   = SmartDataCollator(model=model)

# 5. Training Arguments (from per-model config)
args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(MODEL_DIRS["vit_gpt2"], "checkpoints"),
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    per_device_eval_batch_size=cfg.get("per_device_train_batch_size", Config.BATCH_SIZE),
    predict_with_generate=True,
    logging_steps=50,
    learning_rate=cfg["learning_rate"],
    num_train_epochs=cfg["num_train_epochs"],
    fp16=Config.FP16,
    weight_decay=cfg["weight_decay"],
    label_smoothing_factor=cfg.get("label_smoothing_factor", 0.1),
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    report_to="none",
)

# 6. Trainer with EarlyStoppingCallback
trainer = Seq2SeqTrainer(
    model=model, processing_class=tokenizer, args=args,
    train_dataset=train_data, eval_dataset=val_data, data_collator=collator,
    compute_metrics=lambda eval_pred: compute_metrics_bleu(eval_pred, tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

trainer.train()
plot_training_history(trainer, "ViT + GPT2")

# 7. Log experiment
log_experiment("vit_gpt2", cfg, trainer)

# 8. Export
trainer.save_model(MODEL_DIRS["vit_gpt2"])
tokenizer.save_pretrained(MODEL_DIRS["vit_gpt2"])
processor.save_pretrained(MODEL_DIRS["vit_gpt2"])
print(f"✅ Model ViT + GPT2 saved to {MODEL_DIRS['vit_gpt2']}")

from transformers import VisionEncoderDecoderModel, AutoImageProcessor, GenerationConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer

print("🚀 Memulai Training Model Swin + IndoBARTv2...")

# --- Unified tokenizer (IndoNLG) for ALL models ---
tokenizer = UNIFIED_TOKENIZER
processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
cfg = EXPERIMENT_CONFIGS["swin_indobartv2"]

# 1. Construct VisionEncoderDecoder model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "microsoft/swin-base-patch4-window7-224-in22k",
    "indobenchmark/indobart-v2"
)


# PENTING: Resize embeddings decoder agar sesuai dengan vocab IndoNLG tokenizer
model.decoder.resize_token_embeddings(len(tokenizer))
print(f"  Token embeddings decoder resized to {len(tokenizer)}")

# 2. Token alignment (using unified IndoNLG tokenizer)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
if hasattr(model.config, 'decoder') and hasattr(model.config.decoder, 'vocab_size'):
    model.config.vocab_size = len(tokenizer)

legacy_keys = ['max_length', 'early_stopping', 'num_beams', 'length_penalty', 'no_repeat_ngram_size']
for key in legacy_keys:
    if hasattr(model.config, key): delattr(model.config, key)

_bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
_eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
model.generation_config = GenerationConfig(
    max_length=Config.MAX_LENGTH, early_stopping=True, num_beams=4,
    length_penalty=1.0, no_repeat_ngram_size=2,
    bos_token_id=_bos, eos_token_id=_eos,
    pad_token_id=tokenizer.pad_token_id, decoder_start_token_id=_bos,
)

# 3. Smart Freezing
for param in model.encoder.parameters(): param.requires_grad = False
for param in model.encoder.encoder.layers[-1].parameters(): param.requires_grad = True

# 4. Setup Dataset (dataset split is LOCKED — identical for all models)
train_data = FlickrIndoDataset(train_df, Config.IMAGE_DIR, processor, tokenizer)
val_data   = FlickrIndoDataset(val_df,   Config.IMAGE_DIR, processor, tokenizer)
collator   = SmartDataCollator(model=model)

# 5. Training Arguments (from per-model config)
args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(MODEL_DIRS["swin_indobartv2"], "checkpoints"),
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    per_device_eval_batch_size=cfg.get("per_device_train_batch_size", Config.BATCH_SIZE),
    predict_with_generate=True,
    logging_steps=50,
    learning_rate=cfg["learning_rate"],
    num_train_epochs=cfg["num_train_epochs"],
    fp16=Config.FP16,
    weight_decay=cfg["weight_decay"],
    label_smoothing_factor=cfg.get("label_smoothing_factor", 0.1),
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    report_to="none",
)

# 6. Trainer with EarlyStoppingCallback
trainer = Seq2SeqTrainer(
    model=model, processing_class=tokenizer, args=args,
    train_dataset=train_data, eval_dataset=val_data, data_collator=collator,
    compute_metrics=lambda eval_pred: compute_metrics_bleu(eval_pred, tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

trainer.train()
plot_training_history(trainer, "Swin + IndoBARTv2")

# 7. Log experiment
log_experiment("swin_indobartv2", cfg, trainer)

# 8. Export
trainer.save_model(MODEL_DIRS["swin_indobartv2"])
tokenizer.save_pretrained(MODEL_DIRS["swin_indobartv2"])
processor.save_pretrained(MODEL_DIRS["swin_indobartv2"])
print(f"✅ Model Swin + IndoBARTv2 saved to {MODEL_DIRS['swin_indobartv2']}")

from transformers import VisionEncoderDecoderModel, AutoImageProcessor, GenerationConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer

print("🚀 Memulai Training Model ViT + IndoBARTv2...")

# --- Unified tokenizer (IndoNLG) for ALL models ---
tokenizer = UNIFIED_TOKENIZER
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
cfg = EXPERIMENT_CONFIGS["vit_indobartv2"]

# 1. Construct VisionEncoderDecoder model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k",
    "indobenchmark/indobart-v2"
)


# PENTING: Resize embeddings decoder agar sesuai dengan vocab IndoNLG tokenizer
model.decoder.resize_token_embeddings(len(tokenizer))
print(f"  Token embeddings decoder resized to {len(tokenizer)}")

# 2. Token alignment (using unified IndoNLG tokenizer)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
if hasattr(model.config, 'decoder') and hasattr(model.config.decoder, 'vocab_size'):
    model.config.vocab_size = len(tokenizer)

legacy_keys = ['max_length', 'early_stopping', 'num_beams', 'length_penalty', 'no_repeat_ngram_size']
for key in legacy_keys:
    if hasattr(model.config, key): delattr(model.config, key)

_bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
_eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
model.generation_config = GenerationConfig(
    max_length=Config.MAX_LENGTH, early_stopping=True, num_beams=4,
    length_penalty=1.0, no_repeat_ngram_size=2,
    bos_token_id=_bos, eos_token_id=_eos,
    pad_token_id=tokenizer.pad_token_id, decoder_start_token_id=_bos,
)

# 3. Smart Freezing
for param in model.encoder.parameters(): param.requires_grad = False
for param in model.encoder.encoder.layer[-1].parameters(): param.requires_grad = True

# 4. Setup Dataset (dataset split is LOCKED — identical for all models)
train_data = FlickrIndoDataset(train_df, Config.IMAGE_DIR, processor, tokenizer)
val_data   = FlickrIndoDataset(val_df,   Config.IMAGE_DIR, processor, tokenizer)
collator   = SmartDataCollator(model=model)

# 5. Training Arguments (from per-model config)
args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(MODEL_DIRS["vit_indobartv2"], "checkpoints"),
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    per_device_eval_batch_size=cfg.get("per_device_train_batch_size", Config.BATCH_SIZE),
    predict_with_generate=True,
    logging_steps=50,
    learning_rate=cfg["learning_rate"],
    num_train_epochs=cfg["num_train_epochs"],
    fp16=Config.FP16,
    weight_decay=cfg["weight_decay"],
    label_smoothing_factor=cfg.get("label_smoothing_factor", 0.1),
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    report_to="none",
)

# 6. Trainer with EarlyStoppingCallback
trainer = Seq2SeqTrainer(
    model=model, processing_class=tokenizer, args=args,
    train_dataset=train_data, eval_dataset=val_data, data_collator=collator,
    compute_metrics=lambda eval_pred: compute_metrics_bleu(eval_pred, tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

trainer.train()
plot_training_history(trainer, "ViT + IndoBARTv2")

# 7. Log experiment
log_experiment("vit_indobartv2", cfg, trainer)

# 8. Export
trainer.save_model(MODEL_DIRS["vit_indobartv2"])
tokenizer.save_pretrained(MODEL_DIRS["vit_indobartv2"])
processor.save_pretrained(MODEL_DIRS["vit_indobartv2"])
print(f"✅ Model ViT + IndoBARTv2 saved to {MODEL_DIRS['vit_indobartv2']}")

from transformers import VisionEncoderDecoderModel, AutoImageProcessor, GenerationConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer

print("🚀 Memulai Training Model Swin + GPT2...")

# --- Unified tokenizer (IndoNLG) for ALL models ---
tokenizer = UNIFIED_TOKENIZER
processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
cfg = EXPERIMENT_CONFIGS["swin_gpt2"]

# GPT2 requires explicit cross-attention config injection
from transformers import AutoConfig
gpt2_cfg = AutoConfig.from_pretrained("cahya/gpt2-small-indonesian-522M")
gpt2_cfg.is_decoder = True
gpt2_cfg.add_cross_attention = True

# 1. Construct VisionEncoderDecoder model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "microsoft/swin-base-patch4-window7-224-in22k",
    "cahya/gpt2-small-indonesian-522M",
    decoder_config=gpt2_cfg
)

# PENTING: Resize embeddings decoder agar sesuai dengan vocab IndoNLG tokenizer
model.decoder.resize_token_embeddings(len(tokenizer))
print(f"  Token embeddings decoder resized to {len(tokenizer)}")

# 2. Token alignment (using unified IndoNLG tokenizer)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
if hasattr(model.config, 'decoder') and hasattr(model.config.decoder, 'vocab_size'):
    model.config.vocab_size = len(tokenizer)

legacy_keys = ['max_length', 'early_stopping', 'num_beams', 'length_penalty', 'no_repeat_ngram_size']
for key in legacy_keys:
    if hasattr(model.config, key): delattr(model.config, key)

_bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
_eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
model.generation_config = GenerationConfig(
    max_length=Config.MAX_LENGTH, early_stopping=True, num_beams=4,
    length_penalty=1.0, no_repeat_ngram_size=2,
    bos_token_id=_bos, eos_token_id=_eos,
    pad_token_id=tokenizer.pad_token_id, decoder_start_token_id=_bos,
)

# 3. Smart Freezing
for param in model.encoder.parameters(): param.requires_grad = False
for param in model.encoder.encoder.layers[-1].parameters(): param.requires_grad = True

# 4. Setup Dataset (dataset split is LOCKED — identical for all models)
train_data = FlickrIndoDataset(train_df, Config.IMAGE_DIR, processor, tokenizer)
val_data   = FlickrIndoDataset(val_df,   Config.IMAGE_DIR, processor, tokenizer)
collator   = SmartDataCollator(model=model)

# 5. Training Arguments (from per-model config)
args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(MODEL_DIRS["swin_gpt2"], "checkpoints"),
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    per_device_eval_batch_size=cfg.get("per_device_train_batch_size", Config.BATCH_SIZE),
    predict_with_generate=True,
    logging_steps=50,
    learning_rate=cfg["learning_rate"],
    num_train_epochs=cfg["num_train_epochs"],
    fp16=Config.FP16,
    weight_decay=cfg["weight_decay"],
    label_smoothing_factor=cfg.get("label_smoothing_factor", 0.1),
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    report_to="none",
)

# 6. Trainer with EarlyStoppingCallback
trainer = Seq2SeqTrainer(
    model=model, processing_class=tokenizer, args=args,
    train_dataset=train_data, eval_dataset=val_data, data_collator=collator,
    compute_metrics=lambda eval_pred: compute_metrics_bleu(eval_pred, tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

trainer.train()
plot_training_history(trainer, "Swin + GPT2")

# 7. Log experiment
log_experiment("swin_gpt2", cfg, trainer)

# 8. Export
trainer.save_model(MODEL_DIRS["swin_gpt2"])
tokenizer.save_pretrained(MODEL_DIRS["swin_gpt2"])
processor.save_pretrained(MODEL_DIRS["swin_gpt2"])
print(f"✅ Model Swin + GPT2 saved to {MODEL_DIRS['swin_gpt2']}")

from transformers import VisionEncoderDecoderModel, AutoImageProcessor, GenerationConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer

print("🚀 Memulai Training Model Swin + IndoBERT...")

# --- Unified tokenizer (IndoNLG) for ALL models ---
tokenizer = UNIFIED_TOKENIZER
processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
cfg = EXPERIMENT_CONFIGS["swin_indobert"]

# 1. Construct VisionEncoderDecoder model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "microsoft/swin-base-patch4-window7-224-in22k",
    "indobenchmark/indobert-base-p1"
)

# PENTING: Resize embeddings decoder agar sesuai dengan vocab IndoNLG tokenizer
model.decoder.resize_token_embeddings(len(tokenizer))
print(f"  Token embeddings decoder resized to {len(tokenizer)}")

# 2. Token alignment (using unified IndoNLG tokenizer)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
if hasattr(model.config, 'decoder') and hasattr(model.config.decoder, 'vocab_size'):
    model.config.vocab_size = len(tokenizer)

legacy_keys = ['max_length', 'early_stopping', 'num_beams', 'length_penalty', 'no_repeat_ngram_size']
for key in legacy_keys:
    if hasattr(model.config, key): delattr(model.config, key)

_bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
_eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
model.generation_config = GenerationConfig(
    max_length=Config.MAX_LENGTH, early_stopping=True, num_beams=4,
    length_penalty=1.0, no_repeat_ngram_size=2,
    bos_token_id=_bos, eos_token_id=_eos,
    pad_token_id=tokenizer.pad_token_id, decoder_start_token_id=_bos,
)

# 3. Smart Freezing
for param in model.encoder.parameters(): param.requires_grad = False
for param in model.encoder.encoder.layers[-1].parameters(): param.requires_grad = True

# 4. Setup Dataset (dataset split is LOCKED — identical for all models)
train_data = FlickrIndoDataset(train_df, Config.IMAGE_DIR, processor, tokenizer)
val_data   = FlickrIndoDataset(val_df,   Config.IMAGE_DIR, processor, tokenizer)
collator   = SmartDataCollator(model=model)

# 5. Training Arguments (from per-model config)
args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(MODEL_DIRS["swin_indobert"], "checkpoints"),
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    per_device_eval_batch_size=cfg.get("per_device_train_batch_size", Config.BATCH_SIZE),
    predict_with_generate=True,
    logging_steps=50,
    learning_rate=cfg["learning_rate"],
    num_train_epochs=cfg["num_train_epochs"],
    fp16=Config.FP16,
    weight_decay=cfg["weight_decay"],
    label_smoothing_factor=cfg.get("label_smoothing_factor", 0.1),
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    report_to="none",
)

# 6. Trainer with EarlyStoppingCallback
trainer = Seq2SeqTrainer(
    model=model, processing_class=tokenizer, args=args,
    train_dataset=train_data, eval_dataset=val_data, data_collator=collator,
    compute_metrics=lambda eval_pred: compute_metrics_bleu(eval_pred, tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

trainer.train()
plot_training_history(trainer, "Swin + IndoBERT")

# 7. Log experiment
log_experiment("swin_indobert", cfg, trainer)

# 8. Export
trainer.save_model(MODEL_DIRS["swin_indobert"])
tokenizer.save_pretrained(MODEL_DIRS["swin_indobert"])
processor.save_pretrained(MODEL_DIRS["swin_indobert"])
print(f"✅ Model Swin + IndoBERT saved to {MODEL_DIRS['swin_indobert']}")

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import VisionEncoderDecoderModel, BertTokenizer, AutoTokenizer, AutoImageProcessor
import os
import torch
from safetensors.torch import load_file

# Import Evaluation Metrics dari pycocoevalcap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

print("🚀 Memulai Pengujian Terpadu dengan COCO Evaluation Suite (15% Unseen Data)...")

def compute_coco_metrics(gts, res):
    """
    gts: Ground Truth (Dictionary dict[image_id] = [caption1, caption2, ...])
    res: Results/Prediksi (Dictionary dict[image_id] = [pred_caption])
    """
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    eval_results = {}
    for scorer, method in scorers:
        print(f"Mengalkulasi {method}...")
        try:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    eval_results[m] = round(sc * 100, 3)
            else:
                eval_results[method] = round(score * 100, 3)
        except Exception as e:
            print(f"⚠️ Gagal menghitung {method}. Pastikan Java terinstal untuk SPICE/METEOR. Error: {e}")
            eval_results[method] = None

    return eval_results



def evaluate_model_comprehensive(model_name, model_dir, df, processor_class, tokenizer_class):
    print(f"\n🔍 Menjalankan Inferensi: {model_name}")

    processor = processor_class.from_pretrained(model_dir)
    tokenizer = tokenizer_class.from_pretrained(model_dir)

    # --- AUTO-CHECKPOINT FINDER ---
    import os
    weight_dir = model_dir
    weight_files = ["model.safetensors", "pytorch_model.bin"]
    has_weights = any(os.path.exists(os.path.join(model_dir, f)) for f in weight_files)
    if not has_weights:
        checkpoints_dir = os.path.join(model_dir, "checkpoints")
        if os.path.exists(checkpoints_dir):
            checkpoints = [d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                weight_dir = os.path.join(checkpoints_dir, latest_checkpoint)
                print(f"⚠️ Menggunakan bobot dari checkpoint terakhir: {latest_checkpoint}")

    # --- HANDLING ISU HUGGING FACE CONFIG ---
    try:
        # Coba load normal terlebih dahulu
        model = VisionEncoderDecoderModel.from_pretrained(weight_dir)
    except ValueError:
        print(f"⚠️ Peringatan: Config {model_name} tidak terbaca sempurna. Melakukan rekonstruksi arsitektur dan direct-weight loading...")

        # 1. Rebuild base architecture
        if "Swin" in model_name:
            model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                "microsoft/swin-base-patch4-window7-224-in22k",
                "indobenchmark/indobart-v2"
            )
        elif "GPT2" in model_name:
            model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                "google/vit-base-patch16-224-in21k",
                "cahya/gpt2-small-indonesian-522M"
            )
        else:
            model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                "google/vit-base-patch16-224-in21k",
                "indobenchmark/indobert-base-p1"
            )

        # 2. Load trained weights secara manual
        weight_safetensors = os.path.join(weight_dir, "model.safetensors")
        weight_bin = os.path.join(weight_dir, "pytorch_model.bin")

        if os.path.exists(weight_safetensors):
            model.load_state_dict(load_file(weight_safetensors))
        elif os.path.exists(weight_bin):
            model.load_state_dict(torch.load(weight_bin, map_location="cpu"))
        else:
            raise FileNotFoundError(f"Bobot model tidak ditemukan di {weight_dir}")

        # 3. Align special tokens untuk generation
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.decoder_start_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else tokenizer.bos_token_id
    # ----------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    test_dataset = FlickrIndoDataset(df, Config.IMAGE_DIR, processor, tokenizer)
    collator = SmartDataCollator(model=model)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collator)

    gts = {}
    res = {}

    img_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Generating ({model_name})"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            generated_ids = model.generate(
                pixel_values,
                max_length=Config.MAX_LENGTH,
                num_beams=4,
                early_stopping=True
            )

            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = np.where(labels.numpy() != -100, labels.numpy(), tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, label in zip(decoded_preds, decoded_labels):
                img_id = f"img_{img_idx}"
                res[img_id] = [pred.strip()]
                gts[img_id] = [label.strip()]
                img_idx += 1

    metrics_dict = compute_coco_metrics(gts, res)
    metrics_dict["Model"] = model_name

    return metrics_dict

results = []

results.append(evaluate_model_comprehensive(
    "ViT + IndoBERT", MODEL_DIRS["vit_indobert"], test_df, AutoImageProcessor, BertTokenizer
))

results.append(evaluate_model_comprehensive(
    "ViT + GPT2", MODEL_DIRS["vit_gpt2"], test_df, AutoImageProcessor, AutoTokenizer
))

results.append(evaluate_model_comprehensive(
    "Swin + IndoBARTv2", MODEL_DIRS["swin_indobartv2"], test_df, AutoImageProcessor, BertTokenizer
))


results.append(evaluate_model_comprehensive(
    "ViT + IndoBARTv2", MODEL_DIRS["vit_indobartv2"], test_df, AutoImageProcessor, BertTokenizer
))

results.append(evaluate_model_comprehensive(
    "Swin + GPT2", MODEL_DIRS["swin_gpt2"], test_df, AutoImageProcessor, AutoTokenizer
))

results.append(evaluate_model_comprehensive(
    "Swin + IndoBERT", MODEL_DIRS["swin_indobert"], test_df, AutoImageProcessor, BertTokenizer
))

# --- KOMPARASI FINAL ---
print("\n📊 HASIL KOMPARASI METRIK FINAL PADA TEST SET:")
# Reorder kolom agar 'Model' ada di depan
cols = ['Model', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
comparison_df = pd.DataFrame(results)[cols].set_index("Model")

print(comparison_df)

# Export hasil evaluasi
export_path = os.path.join(Config.BASE_EXPORT_DIR, "comprehensive_metrics_comparison.csv")
comparison_df.to_csv(export_path)
print(f"✅ Laporan evaluasi komprehensif diekspor ke: {export_path}")

import os
import torch
from PIL import Image
from tqdm import tqdm # Menggunakan versi console agar progress bar lebih rapi
import pandas as pd
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor, BertTokenizer

# 1. Konfigurasi Perangkat dan Direktori
# Memastikan penggunaan GPU RTX A4000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Menjalankan inferensi pada perangkat: {device}")

# Mengarahkan path model ke direktori penyimpanan akhirmu
# Ganti "best_model_checkpoint" dengan nama folder spesifik jika ada
model_path = "./final_thesis_models/swin_indobartv2"

# 2. Muat Model, Tokenizer, dan Processor
print("Memuat arsitektur Swin-IndoBARTv2...")
try:
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
except ValueError:
    print("Config incomplete, rebuilding architecture and loading weights...")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "microsoft/swin-base-patch4-window7-224-in22k",
        "indobenchmark/indobart-v2"
    )
    from safetensors.torch import load_file
    import os
    weight_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(weight_file):
        model.load_state_dict(load_file(weight_file))
    else:
        weight_bin = os.path.join(model_path, "pytorch_model.bin")
        model.load_state_dict(torch.load(weight_bin, map_location='cpu'))
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")

# 3. Proses Inferensi (Beam Search)
hasil_prediksi = []
model.eval()

print("Memulai generasi caption...")
# Asumsi dataframe pengujianmu bernama test_df.
# Sesuaikan 'image_file' dengan nama kolom yang menyimpan nama file gambar di metadata.csv milikmu
with torch.no_grad():
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inferensi Test Set"):
        # Menggabungkan IMAGE_DIR dari class Config dengan nama file gambar
        image_name = row['file_name'] # UBAH INI jika nama kolom di CSV-mu adalah 'image_name', 'file_name', dll
        full_image_path = os.path.join(Config.IMAGE_DIR, image_name)

        try:
            # Buka gambar dan konversi ke RGB
            image = Image.open(full_image_path).convert("RGB")

            # Ekstraksi fitur visual
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

            # Generasi teks dengan hiperparameter pencegah overfitting/pengulangan
            output_ids = model.generate(
                pixel_values,
                max_length=40,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id
            )

            # Dekode output menjadi string
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            hasil_prediksi.append(caption)

        except Exception as e:
            print(f"⚠️ Gagal memproses {image_name}: {e}")
            hasil_prediksi.append("")

# 4. Menyimpan Hasil ke Dataframe
test_df['prediksi_swin_indobart'] = hasil_prediksi

# Menampilkan 5 hasil teratas untuk inspeksi visual (sanity check)
print("\n🔍 Sampel Hasil Inferensi:")
print(test_df[['file_name', 'prediksi_swin_indobart']].head())

# Menyimpan hasil ke file CSV baru agar bisa dievaluasi skor CIDEr/ROUGE-nya di cell terpisah
output_csv_path = os.path.join(os.path.dirname(Config.CSV_FILE), "hasil_inferensi_swin_indobart.csv")
test_df.to_csv(output_csv_path, index=False)
print(f"✅ Hasil prediksi berhasil disimpan di: {output_csv_path}")

import matplotlib.pyplot as plt
from PIL import Image
import os

# Menampilkan contoh hasil inferensi dengan gambar
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Contoh Hasil Inferensi Swin + IndoBARTv2', fontsize=16, fontweight='bold')

samples = test_df[test_df['prediksi_swin_indobart'] != ''].sample(n=2)

for ax, (idx, row) in zip(axes.flat, samples.iterrows()):
    img_path = os.path.join(Config.IMAGE_DIR, row['file_name'])
    img = Image.open(img_path).convert('RGB')
    ax.imshow(img)
    ax.set_title(f'Prediksi:', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel(row['prediksi_swin_indobart'], fontsize=10, wrap=True, labelpad=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # Tampilkan ground truth jika ada
    if 'text' in row.index:
        ax.set_title(f'GT: {row["text"]}', fontsize=9, style='italic', pad=10)
        ax.set_xlabel(f'Pred: {row["prediksi_swin_indobart"]}', fontsize=10, fontweight='bold', wrap=True, labelpad=10)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Ambil 1 sampel acak dari dataset
sample_idx = np.random.randint(0, len(train_df))
sample_row = train_df.iloc[sample_idx]
img_name = str(sample_row['file_name'])
if '/' in img_name: img_name = img_name.split('/')[-1]
img_path = os.path.join(Config.IMAGE_DIR, img_name)

# Load processor for Swin
swin_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")

# Load gambar asli
pil_image = Image.open(img_path).convert('RGB')
orig_w, orig_h = pil_image.size

# Proses melalui processor (menghasilkan tensor 3x224x224)
processed = swin_processor(images=pil_image, return_tensors='pt')
tensor_img = processed.pixel_values.squeeze(0)  # [3, 224, 224]

# Konversi tensor ke format displayable (H, W, C) dan normalize ke [0,1]
display_tensor = tensor_img.permute(1, 2, 0).numpy()
display_tensor = (display_tensor - display_tensor.min()) / (display_tensor.max() - display_tensor.min())

print(f'Sampel: {img_name}')
print(f'Dimensi asli: {orig_w}x{orig_h} -> Tensor: {tuple(tensor_img.shape)}')

# === Plot 1: Gambar Asli ===
fig1, ax1 = plt.subplots(figsize=(7, 7))
ax1.imshow(pil_image)
ax1.set_title(f'Gambar Asli ({orig_w}x{orig_h})', fontsize=14, fontweight='bold', pad=12)
ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ax1.spines.values(): spine.set_edgecolor('#ccc')
fig1.tight_layout()
plt.show()

# === Plot 2: Tensor 224x224 ===
fig2, ax2 = plt.subplots(figsize=(7, 7))
ax2.imshow(display_tensor)
ax2.set_title('Tensor 224x224', fontsize=14, fontweight='bold', pad=12)
ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ax2.spines.values(): spine.set_edgecolor('#ccc')
fig2.tight_layout()
plt.show()

# === Plot 3: Ilustrasi Patching Swin Transformer ===
fig3, ax3 = plt.subplots(figsize=(7, 7))
ax3.imshow(display_tensor)
patch_size = 32  # Swin: 4x4 pixel patch, window 7x7 = ~32px visual grid
for x in range(0, 224, patch_size):
    ax3.axvline(x=x, color='white', linewidth=0.8, alpha=0.75)
for y in range(0, 224, patch_size):
    ax3.axhline(y=y, color='white', linewidth=0.8, alpha=0.75)
ax3.set_title('Ilustrasi Patching Swin Transformer', fontsize=14, fontweight='bold', pad=12)
ax3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ax3.spines.values(): spine.set_edgecolor('#ccc')
fig3.tight_layout()
plt.show()


# === Plot Gabungan: Proses Preprocessing Swin Transformer ===
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Proses Preprocessing Gambar untuk Swin Transformer', fontsize=16, fontweight='bold', y=1.02)

# Gambar Asli
ax1.imshow(pil_image)
ax1.set_title(f'Gambar Asli ({orig_w}x{orig_h})', fontsize=13, fontweight='bold', pad=10)
ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ax1.spines.values(): spine.set_edgecolor('#ccc')

# Tensor 224x224
ax2.imshow(display_tensor)
ax2.set_title('Tensor 224x224', fontsize=13, fontweight='bold', pad=10)
ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ax2.spines.values(): spine.set_edgecolor('#ccc')

# Ilustrasi Patching
ax3.imshow(display_tensor)
patch_size = 32
for x in range(0, 224, patch_size):
    ax3.axvline(x=x, color='white', linewidth=0.8, alpha=0.75)
for y in range(0, 224, patch_size):
    ax3.axhline(y=y, color='white', linewidth=0.8, alpha=0.75)
ax3.set_title('Ilustrasi Patching Swin Transformer', fontsize=13, fontweight='bold', pad=10)
ax3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ax3.spines.values(): spine.set_edgecolor('#ccc')

# Panah antar subplot
plt.tight_layout()
plt.savefig(os.path.join(Config.BASE_EXPORT_DIR, 'preprocessing_swin.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

# Ambil 5 sampel caption
samples = train_df.sample(n=5)

rows = []
for _, row in samples.iterrows():
    text = row['text']
    encoded = tokenizer(text, padding=False, truncation=True, max_length=64)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    rows.append({
        'Teks Asli': text,
        'Tokens': ' | '.join(tokens),
        'Token IDs': str(encoded['input_ids']),
        'Jumlah Token': len(tokens)
    })

tok_df = pd.DataFrame(rows)

# Tampilkan tanpa terpotong
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(f'Vocab size: {tokenizer.vocab_size}')
print(f'Special tokens: {tokenizer.all_special_tokens}')
print()
print(tok_df)


import matplotlib.pyplot as plt
from PIL import Image
import os

# Visualisasi 3 Contoh Hasil Inferensi Swin + IndoBARTv2
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Contoh Hasil Inferensi: Swin + IndoBARTv2', fontsize=16, fontweight='bold')

samples = test_df.sample(3, random_state=42).reset_index(drop=True)

for i, (_, row) in enumerate(samples.iterrows()):
    img_name = row['file_name']
    img_path = os.path.join(Config.IMAGE_DIR, img_name)
    img = Image.open(img_path).convert('RGB')

    axes[i].imshow(img)
    axes[i].set_title(f'Prediksi:\n{row["prediksi_swin_indobart"]}', fontsize=11, wrap=True)
    axes[i].axis('off')

    # Tampilkan ground truth jika ada kolom 'text'
    if 'text' in row.index:
        axes[i].set_xlabel(f'GT: {row["text"]}', fontsize=9, color='gray', wrap=True)
        axes[i].xaxis.set_visible(True)
        axes[i].tick_params(bottom=False, labelbottom=False)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data metrik dari ke-6 model
epochs = [1, 2, 3, 4, 5, 6, 7]
data = {
    'VITINDOBERT': {
        'train': [2.868395, 2.660275, 2.493438, 2.382591, 2.208388, 2.081209, 2.007757],
        'val': [2.799762, 2.685239, 2.628790, 2.611897, 2.611417, 2.627953, 2.638078],
        'bleu': [20.388300, 21.194700, 23.090900, 22.614500, 20.347000, 23.811300, 23.419800]
    },
    'VITGPT2': {
        'train': [2.856282, 2.611651, 2.400330, 2.219770, 2.010982, 1.871428, 1.787975],
        'val': [2.786187, 2.685431, 2.662508, 2.694251, 2.730166, 2.777635, 2.806989],
        'bleu': [7.257400, 7.449200, 8.085400, 8.356500, 8.502000, 8.507400, 8.613400]
    },
    'SWIN INDOBARTv2': {
        'train': [2.899655, 2.612892, 2.403778, 2.247270, 2.062702, 1.919349, 1.846817],
        'val': [2.794381, 2.652794, 2.622409, 2.640168, 2.674439, 2.711144, 2.731598],
        'bleu': [25.352500, 24.899300, 25.066600, 26.342300, 26.219100, 25.772700, 25.453500]
    },
    'Vit indobartv2': {
        'train': [2.979102, 2.716476, 2.535868, 2.397891, 2.224069, 2.112375, 2.034604],
        'val': [2.882159, 2.736912, 2.685708, 2.673144, 2.676881, 2.690378, 2.697590],
        'bleu': [23.106000, 24.577500, 25.274000, 25.894300, 24.860400, 25.343200, 25.340100]
    },
    'Swin gpt2': {
        'train': [2.782082, 2.530639, 2.299360, 2.104151, 1.902651, 1.774283, 1.696049],
        'val': [2.707310, 2.613176, 2.602335, 2.658361, 2.719883, 2.782053, 2.826359],
        'bleu': [8.482900, 9.222600, 9.294300, 9.060400, 9.114000, 8.907900, 8.998700]
    },
    'Swin indobert': {
        'train': [2.798711, 2.570385, 2.381804, 2.241052, 2.048075, 1.922469, 1.839520],
        'val': [2.725833, 2.611060, 2.562458, 2.564471, 2.589592, 2.624428, 2.649807],
        'bleu': [21.048000, 22.078500, 24.612100, 23.031000, 18.395700, 21.194800, 21.341700]
    }
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ==========================================
# GRAFIK 1: Training vs Validation Loss
# ==========================================
num_models = len(data)
cols = min(3, num_models)
rows = max(1, int(np.ceil(num_models / cols)))

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]

for i, (model, metrics) in enumerate(data.items()):
    ax = axes[i]
    ax.plot(epochs, metrics['train'], linestyle='--', marker='o',
             color='grey', label='Training Loss', linewidth=2)
    ax.plot(epochs, metrics['val'], linestyle='-', marker='x',
             color='black', label='Validation Loss', linewidth=2)
    ax.set_title(f'Model: {model}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10)

for j in range(num_models, len(axes)):
    axes[j].axis('off')

fig.suptitle('Grafik Loss Evaluasi per Model', fontsize=16, fontweight='bold', y=1.02)
fig.tight_layout()

# Simpan Grafik 1
plt.savefig('grafik_loss_evaluasi.png', dpi=300, bbox_inches='tight')
plt.close(fig) # Tutup figure agar tidak menumpuk dengan grafik berikutnya



# ==========================================
# GRAFIK 2: BLEU Score
# ==========================================
plt.figure(figsize=(12, 6))

for i, (model, metrics) in enumerate(data.items()):
    color = colors[i]
    plt.plot(epochs, metrics['bleu'], linestyle='-', marker='s',
             color=color, label=model, linewidth=2)

plt.title('Grafik Perkembangan Skor BLEU', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('BLEU Score', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=11)

# Simpan Grafik 2
plt.savefig('grafik_bleu_evaluasi.png', dpi=300, bbox_inches='tight')
plt.close() # Tutup figure

print("Berhasil! Dua file gambar (grafik_loss_evaluasi.png & grafik_bleu_evaluasi.png) telah disimpan.")

# Summary Table Generation
import json

def generate_summary():
    log_path = os.path.join(Config.BASE_EXPORT_DIR, "experiment_log.json")
    if not os.path.exists(log_path):
        print("experiment_log.json not found. Skipping summary generation.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    summary_md = "# Training Summary\n\n"
    summary_md += "| Model Name | Early Stopping Epoch | Final Validation Loss | Final BLEU Score |\n"
    summary_md += "|---|---|---|---|\n"

    for entry in log_data:
        model_name = entry.get("model_name", "N/A")
        epoch = entry.get("early_stopped_at_epoch", "N/A")
        val_loss = entry.get("final_eval_loss", "N/A")
        if val_loss != "N/A":
            val_loss = f"{val_loss:.4f}"
        bleu = entry.get("final_bleu", "N/A")
        if bleu != "N/A":
            bleu = f"{bleu:.4f}"

        summary_md += f"| {model_name} | {epoch} | {val_loss} | {bleu} |\n"

    summary_path = "training_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"✅ Training summary generated at {summary_path}")

generate_summary()
