import pygetwindow as gw
import mss
import mss.tools
import easyocr
import pandas as pd
import torch
import pickle
import re
import random
import model  # For: MOLDING_TERMS, signal_handler, build_vocab, tokenize
from config import block_size
import logging

# CONFIG & LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_final.pkl"

# LOAD TRAINED MODEL

def load_model():
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU (slower).")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_obj = pickle.load(f)
        logger.info("Loaded model_final.pkl")
        return model_obj
    except FileNotFoundError:
        logger.error(f"{MODEL_PATH} not found. Run training4.py first.")
        return None

# SYNTHETIC DATA (for vocab building)
def generate_synthetic_for_vocab(num_samples=500):
    """Generate minimal synthetic data to ensure key terms are in vocab."""
    templates = ["{t}: {v}", "{t} {v}"]
    data = []
    for _ in range(num_samples):
        t = random.choice(model.MOLDING_TERMS)
        v = round(random.uniform(10, 2000), 2)
        data.append(random.choice(templates).format(t=t, v=v))
    return data

# BUILD VOCABULARY FROM OCR + SYNTHETIC
def build_vocab_from_ocr(ocr_text):
    """Build vocab from OCR + synthetic data (same as training4.py)."""
    logger.info("Building vocabulary from OCR + synthetic data...")
    
    # 1. Real OCR text
    ocr_lines = all_sentences
    
    # 2. Add synthetic examples
    synthetic = generate_synthetic_for_vocab(1000)
    
    # 3. Combine
    all_text = ocr_lines + synthetic
    
    # 4. Build vocab
    vocab = model.build_vocab(all_text)
    logger.info(f"Built runtime vocabulary: {len(vocab)} tokens")
    return vocab

# TOKENIZE INPUT FOR GPT
def tokenize_input(text, vocab):
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    words = re.findall(r'[a-z]+|[0-9]+|[.,]', text.lower())
    tokens = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in words]
    tokens = [word_to_idx['<SOS>']] + tokens[:block_size-50] + [word_to_idx['<EOS>']]
    tokens += [word_to_idx['<PAD>']] * (block_size - len(tokens))
    return tokens

# GPT INFERENCE: Extract Key-Value Pairs
def extract_with_gpt(ocr_text, gpt_model, vocab):
    logger.info("Running GPT extraction...")
    
    prompt = (
        "Extract all molding parameters from the following OCR text. "
        "Output only in format: TERM: VALUE\n"
        "Text:\n" + ocr_text[:1500] + "\nOutput:\n"
    )
    
    input_tokens = tokenize_input(prompt, vocab)
    input_ids = torch.tensor([input_tokens], dtype=torch.long).to(DEVICE)
    
    gpt_model.eval()
    generated_ids = input_ids.clone()
    with torch.no_grad():
        for _ in range(200):
            logits, _ = gpt_model(generated_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == vocab.index('<EOS>'):
                break
    
    # Decode
    word_to_idx = {i: w for i, w in enumerate(vocab)}
    output = ""
    for tid in generated_ids[0]:
        word = word_to_idx.get(tid.item(), "")
        if word in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']: continue
        output += word + " "
    output = output.strip()
    
    logger.info(f"GPT Output:\n{output}")
    
    # Parse into dict
    data = {}
    for line in output.split('\n'):
        if ':' not in line: continue
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        if key in model.MOLDING_TERMS:
            num = re.search(r'[\d.]+', value)
            if num:
                data[key] = float(num.group(0))
    return data

# CAPTURE WINDOW + OCR
def capture_and_ocr():
    windows = [t for t in gw.getAllTitles() if t.strip()]
    if not windows:
        logger.error("No windows found.")
        return None

    print("\nAVAILABLE WINDOWS:")
    for i, t in enumerate(windows, 1):
        print(f"  {i}. {t}")
    choice = int(input(f"\nSelect window (1-{len(windows)}): ")) - 1
    title = windows[choice]

    win = next((w for w in gw.getWindowsWithTitle(title) if w.title == title and w.visible), None)
    if not win:
        logger.error("Window not found.")
        return None

    with mss.mss() as sct:
        monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
        img = sct.grab(monitor)
        path = "window.png"
        mss.tools.to_png(img.rgb, img.size, output=path)
        logger.info(f"Screenshot: {path}")

    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    result = reader.readtext(path, detail=0)
    text = " ".join(result)
    logger.info(f"OCR extracted {len(result)} blocks")
    return text

# SAVE TO EXCEL
def save_to_excel(data, filename="Molding_Data.xlsx"):
    if not data:
        logger.warning("No data to save.")
        return
    df = pd.DataFrame([data])
    df = df.reindex(sorted(df.columns), axis=1)
    df.to_excel(filename, index=False)
    logger.info(f"Saved: {filename}")
    print(f"\nSUCCESS! Open '{filename}'")

# MAIN
if __name__ == "__main__":
    logger.info("AI Molding Data Extractor Starting...")

    # 1. Load model
    gpt_model = load_model()
    if not gpt_model:
        exit(1)

    # 2. Capture + OCR
    ocr_text = capture_and_ocr()
    if not ocr_text:
        exit(1)

    # 3. Build vocab from OCR
    vocab = build_vocab_from_ocr(ocr_text)

    # 4. Extract with GPT
    data = extract_with_gpt(ocr_text, gpt_model, vocab)

    # 5. Save
    if data:
        logger.info(f"Extracted {len(data)} parameters")
        for k, v in data.items():
            logger.info(f"  • {k}: {v}")
        save_to_excel(data)
    else:
        logger.warning("No parameters extracted.")
        print("Try another window or check OCR quality.")