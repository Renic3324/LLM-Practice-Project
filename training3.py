import pickle  # Import pickle module for serializing and deserializing Python objects
import torch  # Import torch module for deep learning operations
import signal  # Import signal module for handling system signals
import sys  # Import sys module for system-specific parameters and functions
import model # Import functions from model module
import config  # Import config module
import logging  # Import logging module for logging messages
import datasets # Import required data sets for training

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


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Set signal handler
    logger.info("Starting initial training...")  # Log start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device
    logger.info(f"Using device: {device}")  # Log device
    try:
        with open('model_updated.pkl', 'rb') as f:  # Open model file
            gpt = pickle.load(f).to(device)  # Load model
        logger.info(f"Loaded model")  # Log loaded model confirmation
    except FileNotFoundError:
        logger.error("model_updated.pkl not found. Run training2.py first.")  # Log error
        sys.exit(1)  # Exit program
    funsd_data = model.load_dataset_stream("nielsr/funsd", split="train", max_samples=5000)  # Load FUNSD data
    cord_data = model.load_dataset_stream("naver-clova-ix/cord-v2", 5000)  # Load CORD data
    sroie_data = model.load_dataset_stream("podbilabs/sroie-donut", 5000)  # Load SROIE data
    all_sentences = all_sentences + list(funsd_data) + list(cord_data) + list(sroie_data) # Combine sentences
    logger.info(f"Loaded {len(sentences)} sentences")  # Log loaded sentences
    vocab = model.build_vocab(all_sentences)  # Build vocabulary
    sequences = model.tokenize(all_sentences, vocab, block_size)  # Tokenize sentences
    gpt = model.resize_model(gpt, len(vocab), device)  # Resize model
    gpt = model.train_model(gpt, iter(sequences), device)   # Train model
    with open('model_updated2.pkl', 'wb') as f:  # Save updated model
        pickle.dump(gpt, f)  # Dump model
    logger.info("Saved model_updated2.pkl")  # Log saving
