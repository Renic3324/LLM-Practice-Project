import pickle  # Import pickle module for serializing and deserializing Python objects
import torch  # Import torch module for deep learning operations
import signal  # Import signal module for handling system signals
import sys  # Import sys module for system-specific parameters and functions
import model # Import functions from model module
import config  # Import config module
import logging  # Import logging module for logging messages

# Configure logging to record events in a file and on the console
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define log format
    handlers=[  # Define handlers for logging
        logging.StreamHandler(sys.stdout)  # Log to standard output
    ]
)
logger = logging.getLogger(__name__)  # Get logger for current module
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setStream(sys.stdout)
        handler.flush = sys.stdout.flush

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Set signal handler
    logger.info("Starting initial training...")  # Log start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device
    logger.info(f"Using device: {device}")  # Log device
    fineweb_data = model.load_dataset_stream("HuggingFaceFW/fineweb", "sample-10BT", 5000)  # Load Fineweb data
    c4_data = model.load_dataset_stream("c4", "en", 5000)  # Load c4 data
    all_sentences = list(fineweb_data) + list(c4_data)  # Combine sentences
    logger.info(f"Loaded {len(sentences)} sentences")  # Log loaded sentences
    vocab = model.build_vocab(all_sentences)  # Build vocabulary
    sequences = model.tokenize(all_sentences, vocab, block_size=block_size)  # Tokenize sentences
    gpt = model.GPTLanguageModel(vocab_size=len(vocab)).to(device)  # Initialize model
    logger.info(f"Model: ~{sum(p.numel() for p in gpt.parameters())/1e6:.1f}M params")  # Log model size
    gpt = model.train_model(gpt, iter(model.tokenize(sentences, vocab, block_size)), device) # Train model
    with open('model.pkl', 'wb') as f:  # Save model
        pickle.dump(gpt, f)
    logger.info("Saved model.pkl")  # Log saving
