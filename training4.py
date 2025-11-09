import logging  # Import logging module for logging messages
import sys  # Import sys module for system-specific parameters
import pickle  # Import pickle module for serializing and deserializing objects
import torch  # Import torch module for deep learning operations
import signal  # Import signal module for handling system signals
import datasets # Import required data sets for training
import model # Import functions from model module
import config  # Import config module
import random

# Configure logging to record events in a file and on the console
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define log format
    handlers=[  # Define handlers for logging
        logging.StreamHandler(sys.stdout)  # Log to standard output
    ]
)
logger = logging.getLogger(__name__)  # Get logger for current module
for handler in logger.handlers:  # Ensure logs are flushed immediately
    if isinstance(handler, logging.StreamHandler):
        handler.setStream(sys.stdout)
        handler.flush = sys.stdout.flush


# SYNTHETIC OCR TRAINING DATA
def generate_synthetic_ocr_data(num_samples=15000):
    """Generate noisy OCR text with clean ground truth."""
    templates = [
        "Inj Pres: {v1} psi  Cycle Time {v2}s",
        "Melt Temp {v1}°F  Pack Pressure: {v2}",
        "{t1}: {v1}  {t2} {v2}",
        "Cycle: {v1}s  Clamp Tonnage {v2}",
        "Nozzle Temp: {v1}  {t1} {v2}",
    ]
    data = []
    for _ in range(num_samples):
        t = random.choice(templates)
        term1 = random.choice(model.MOLDING_TERMS)
        term2 = random.choice(model.MOLDING_TERMS)
        v1 = round(random.uniform(10, 2000), 2)
        v2 = round(random.uniform(10, 2000), 2)

        ocr = t.format(t1=term1, t2=term2, v1=v1, v2=v2)
        # Add OCR noise
        if len(ocr) > 5 and random.random() < 0.7:
            pos = random.randint(0, len(ocr)-1)
            ocr = ocr[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + ocr[pos+1:]
        ocr = ocr.replace(" ", "  ", random.randint(0, 3))

        gt = f"{term1}: {v1}\n{term2}: {v2}"
        data.append((ocr, gt))
    return data

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    logger.info("Starting training4.py: OCR + Molding Key-Value Extraction...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

# Load previous model
    try:
        with open('model_updated2.pkl', 'rb') as f:
            gpt = pickle.load(f).to(device)
        logger.info("Loaded model_updated2.pkl")
    except FileNotFoundError:
        logger.error("Run training3.py first!")
        sys.exit(1)


    # Generate synthetic OCR data
    synthetic_pairs = generate_synthetic_ocr_data(15000)
    synthetic_inputs = [f"Extract parameters from OCR text:\n{x[0]}\nOutput:" for x in synthetic_pairs]
    synthetic_targets = [x[1] for x in synthetic_pairs]
    synthetic_full = [inp + " " + tgt for inp, tgt in zip(synthetic_inputs, synthetic_targets)]
    logger.info(f"Generated {len(synthetic_full)} synthetic OCR examples")

    # Build final vocabulary
    all_sentences = all_sentences + synthetic_full # Combine sentences
    vocab = model.build_vocab(all_sentences)
    logger.info(f"Final vocabulary: {len(vocab)} tokens")

    # Resize model
    gpt = model.resize_model(gpt, len(vocab), device)
    gpt.vocab = vocab  # Attach vocab for agent.py

    # Training data generator (infinite)
    def data_generator():
        combined = list(fineweb_data) + list(c4_data) + molding_data + list(funsd_data) + list(cord_data) + list(sroie_data) + synthetic_full
        while True:
            yield random.choice(combined)

    # Train
    gpt = model.train_model(
        model=gpt,
        sequence_generator=data_generator(),
        device=device,
        max_iters=4000,
        batch_size=8,
        grad_accum_steps=4
    )

    # Save final model
    with open('model_final.pkl', 'wb') as f:
        pickle.dump(gpt, f)

    logger.info("Saved model_final.pkl")
    print("\nFINAL MODEL READY! Use in agent.py")