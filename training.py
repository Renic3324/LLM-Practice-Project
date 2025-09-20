import pickle  # Import pickle module for serializing and deserializing Python objects
import os  # Import os module for operating system dependent functionality
import torch  # Import torch module for deep learning operations
import requests  # Import requests module for making HTTP requests
import zipfile  # Import zipfile module for handling zip files
import io  # Import io module for handling in-memory streams
import signal  # Import signal module for handling system signals
import sys  # Import sys module for system-specific parameters and functions
import re  # Import re module for regular expressions
from model import GPTLanguageModel, tokenize  # Import GPTLanguageModel and tokenize function from model module
from config import block_size  # Import block_size from config module
import logging  # Import logging module for logging messages
from datasets import load_dataset

# Configure logging to record events in a file and on the console
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define log format
    handlers=[  # Define handlers for logging
        logging.StreamHandler(sys.stdout)  # Log to standard output
    ]
)
logger = logging.getLogger(__name__)  # Get logger for current module
'''
def load_sms_data():
    """Function to load SMS spam dataset, caching locally."""
    logger.info("Loading SMS data...")  # Log loading start
    cache_file = 'sms_spam.csv'  # Define cache file name
    if os.path.exists(cache_file):  # Check if cache file exists
        try:
            with open(cache_file, 'r', encoding='utf-8', errors='replace') as f:  # Open cache file
                lines = [line.strip() for line in f if line.strip()]  # Read and strip lines
            logger.info(f"Loaded {len(lines)} SMS sentences from cache")  # Log loaded sentences
            return lines  # Return loaded sentences
        except Exception as e:
            logger.error(f"Error reading cached SMS data: {e}")  # Log error
            return []  # Return empty list on error
    try:
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'  # Define dataset URL
        response = requests.get(url, timeout=10)  # Make HTTP request
        response.raise_for_status()  # Raise error if request fails
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:  # Open zip file in memory
            with z.open('SMSSpamCollection') as f:  # Open file in zip
                lines = []  # Initialize list for sentences
                for line in f:  # Loop through lines
                    try:
                        text = line.decode('utf-8').strip().split('\t')[-1]  # Decode and extract text
                        if text:  # Check if text is not empty
                            lines.append(text)  # Append text to list
                    except UnicodeDecodeError:
                        logger.warning("Skipping invalid SMS line due to decode error")  # Log decode error
                        continue  # Skip line on error
        with open(cache_file, 'w', encoding='utf-8', errors='replace') as f:  # Open cache file for writing
            for line in lines:  # Loop through lines
                f.write(line + '\n')  # Write line to cache
        logger.info(f"Loaded {len(lines)} SMS sentences from UCI")  # Log loaded sentences
        return lines  # Return loaded sentences
    except Exception as e:
        logger.error(f"Failed to load SMS data: {e}")  # Log error
        return []  # Return empty list on error
'''
def load_pile_data():
    """Function to load Pile spam dataset, caching locally."""
    logger.info("Loading Pile data...")  # Log loading start
    dataset = load_dataset('the_pile', split='train') # Open "The Pile" data set
    with open('pile_data.txt', 'w', encoding='utf-8') as f: # Read, strip, and deduplicate lines
        for item in dataset: # Loop through items
            f.write(item['text'] + '\n') # Write item to cache
    logger.info(f"Loaded {len(lines)} Pile sentences from UCI")  # Log loaded sentences
    return item  # Return loaded sentences
'''
def load_molding_data():
    """Function to load molding terms."""
    logger.info("Loading molding terms...")  # Log loading start
    try:
        with open('molding_terms.txt', 'r', encoding='utf-8') as f:  # Open molding terms file
            lines = list(set(line.strip() for line in f if line.strip()))  # Read, strip, and deduplicate lines
        logger.info(f"Loaded {len(lines)} unique molding terms")  # Log loaded terms
        return lines  # Return loaded terms
    except Exception as e:
        logger.error(f"Error reading molding terms: {e}")  # Log error
        return []  # Return empty list on error
'''
def load_c4_data():
    """Function to load C4 spam dataset."""
    logger.info("Loading c4 data...")  # Log loading start
    dataset = load_dataset('c4', 'en', split='train') # Open "C4" data set
    with open('c4_data.txt', 'w', encoding='utf-8') as f: # Read, strip, and deduplicate lines
        for item in dataset: # Loop through items
            f.write(item['text'] + '\n') # Write item to cache
    logger.info(f"Loaded {len(lines)} C4 sentences from UCI")  # Log loaded sentences
    return item  # Return loaded sentences

def build_vocab(sentences):
    """Function to build vocabulary from sentences."""
    logger.info("Building vocabulary...")  # Log building start
    tokens = set()  # Initialize set for tokens
    for sentence in sentences:  # Loop through sentences
        if not isinstance(sentence, str):  # Check if sentence is string
            continue  # Skip if not string
        sentence = re.sub(r'\s+', ' ', sentence).strip().lower()  # Clean sentence
        words = re.findall(r'[a-z]+|[0-9]+|[.,]', sentence)  # Extract words
        tokens.update(words)  # Update tokens set
    vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + sorted(list(tokens))  # Create vocabulary list
    logger.info(f"Built vocabulary with {len(vocab)} words")  # Log vocabulary size
    return vocab  # Return vocabulary

def train_model(model, sequences, device, max_iters=2000, batch_size=2):
    """Function to train the GPT model."""
    logger.info("Starting training...")  # Log training start
    model.train()  # Set model to training mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # Initialize optimizer
    sequences = torch.tensor(sequences, dtype=torch.long).to(device)  # Convert sequences to tensor on device
    for i in range(max_iters):  # Loop through iterations
        idx = torch.randint(0, len(sequences), (batch_size,))  # Generate random indices
        batch = sequences[idx]  # Get batch
        inputs = batch[:, :-1]  # Get inputs
        targets = batch[:, 1:]  # Get targets
        logits, loss = model(inputs, targets)  # Forward pass
        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Optimizer step
        if (i + 1) % 500 == 0:  # Log every 500 iterations
            logger.info(f"Iteration {i+1}/{max_iters}, Loss: {loss.item():.4f}")  # Log loss
    logger.info("Training completed")  # Log training completion
    return model  # Return trained model

def signal_handler(sig, frame):
    """Function to handle interrupts by saving partial model."""
    logger.info("Saving partial model due to interruption...")  # Log saving
    if 'model' in globals():  # Check if model exists
        with open('partial_model.pkl', 'wb') as f:  # Open file for writing
            pickle.dump(model, f)  # Dump model
    sys.exit(0)  # Exit program

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Set signal handler
    logger.info("Starting initial training...")  # Log start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device
    logger.info(f"Using device: {device}")  # Log device

    pile_data = load_pile_data()  # Load Pile data
    c4_data = load_c4_data()  # Load c4 data
    all_sentences = pile_data + c4_data  # Combine sentences
    logger.info(f"Loaded {len(all_sentences)} sentences ({len(sms_data)} Pile + {len(molding_data)} C4)")  # Log loaded sentences

    vocab = build_vocab(all_sentences)  # Build vocabulary
    sequences = tokenize(all_sentences, vocab, block_size=block_size)  # Tokenize sentences

    model = GPTLanguageModel(vocab_size=len(vocab)).to(device)  # Initialize model
    logger.info(f"Created model with ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")  # Log model size

    model = train_model(model, sequences, device, max_iters=2000, batch_size=2)  # Train model

    with open('model.pkl', 'wb') as f:  # Save model
        pickle.dump(model, f)
    with open('vocab.pkl', 'wb') as f:  # Save vocabulary
        pickle.dump(vocab, f)
    logger.info("Saved model.pkl and vocab.pkl")  # Log saving
