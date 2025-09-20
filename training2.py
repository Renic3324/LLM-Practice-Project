import logging  # Import logging module for logging messages
import sys  # Import sys module for system-specific parameters
import os  # Import os module for operating system dependent functionality
import pickle  # Import pickle module for serializing and deserializing objects
import torch  # Import torch module for deep learning operations
import signal  # Import signal module for handling system signals
import re  # Import re module for regular expressions
from datasets import load_dataset  # Import load_dataset from datasets module for loading datasets
from model import GPTLanguageModel, tokenize  # Import GPTLanguageModel and tokenize from model module
from config import block_size  # Import block_size from config module
import psutil  # Import psutil module for system monitoring

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


def log_resources():
    """Function to log current system resources."""
    cpu_percent = psutil.cpu_percent()  # Get CPU percentage
    mem = psutil.virtual_memory()  # Get memory stats
    mem_used = mem.used / 1e9  # Convert memory to GB
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0  # Get GPU memory if available
    logger.info(f"Resources: CPU {cpu_percent:.1f}%, RAM {mem_used:.2f}GB, GPU {gpu_mem:.2f}GB")  # Log resources


def validate_pickle_file(file_path):
    """Function to validate pickle file integrity."""
    try:
        with open(file_path, 'rb') as f:  # Open file in binary read mode
            header = f.read(4)  # Read first 4 bytes
            valid_magic = header.startswith(b'\x80\x03') or header.startswith(b'\x80\x04')  # Check magic number
            logger.debug(f"File {file_path} magic number valid: {valid_magic}")  # Log validation result
            return valid_magic  # Return validation result
    except Exception as e:
        logger.error(f"Failed to validate {file_path}: {e}")  # Log error
        return False  # Return False on failure


def load_sciqa_data():
    """Function to load SciQA dataset."""
    logger.info("Loading SciQA dataset...")  # Log loading start
    try:
        dataset = load_dataset("allenai/qasc", split="train")  # Load dataset
        logger.info("Loaded SciQA dataset")  # Log loading success
        sentences = []  # Initialize list for sentences
        for item in dataset:  # Loop through dataset items
            question = item.get("question", "").strip()  # Get question
            answer = item.get("answer", {}).get("text", "").strip()  # Get answer
            if question and answer:  # Check if both are present
                sentences.append(f"Q: {question} A: {answer}")  # Append formatted sentence
        logger.info(f"Loaded {len(sentences)} sentences")  # Log number of sentences
        log_resources()  # Log resources
        return sentences  # Return sentences
    except Exception as e:
        logger.error(f"Failed to load SciQA data: {e}")  # Log error
        return []  # Return empty list on failure

'''
def load_molding_data():
    """Function to load molding terms."""
    logger.info("Loading molding terms...")  # Log loading start
    try:
        with open('molding_terms.txt', 'r', encoding='utf-8') as f:  # Open file
            lines = list(set(line.strip() for line in f if line.strip()))  # Read and deduplicate lines
        logger.info(f"Loaded {len(lines)} unique molding terms")  # Log number of terms
        log_resources()  # Log resources
        return lines  # Return terms
    except Exception as e:
        logger.error(f"Error reading molding terms: {e}")  # Log error
        return []  # Return empty list on failure
'''

def load_molding_data():
    """Function to load molding terms."""
    logger.info("Loading molding terms...")  # Log loading start
    dataset = load_dataset('scientific_papers', 'arxiv', split='train') # Loads scientific data sets
    keywords = ['plastic', 'molding', 'injection', 'thermoplastic', 'polymer'] # Specifies using molding based data
    plastics_data = [item['text'] for item in dataset if any(kw in item['text'].lower() for kw in keywords)] # Loop through items
    with open('plastics_papers.txt', 'w', encoding='utf-8') as f: # Read, strip, and deduplicate lines
        f.write('\n'.join(plastics_data)) # Write item to cache
    return item  # Return terms

def build_vocab(sentences):
    """Function to build vocabulary from sentences."""
    logger.info("Building vocabulary...")  # Log building start
    tokens = set()  # Initialize set for tokens
    for sentence in sentences:  # Loop through sentences
        sentence = re.sub(r'\s+', ' ', sentence).strip().lower()  # Clean sentence
        words = re.findall(r'[a-z]+|[0-9]+|[.,]', sentence)  # Extract words
        tokens.update(words)  # Update tokens set
    vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + sorted(list(tokens))  # Create vocabulary list
    logger.info(f"Built vocabulary with {len(vocab)} words")  # Log vocabulary size
    return vocab  # Return vocabulary


def resize_model(model, new_vocab_size, device):
    """Function to resize model for new vocabulary size."""
    logger.info(f"Resizing model from vocab size {model.vocab_size} to {new_vocab_size}")  # Log resizing
    new_model = GPTLanguageModel(vocab_size=new_vocab_size).to(device)  # Create new model
    new_state_dict = new_model.state_dict()  # Get new state dict
    old_state_dict = model.state_dict()  # Get old state dict

    for key in old_state_dict:  # Loop through old state dict
        if key not in ['token_embedding.weight', 'head.weight', 'head.bias']:  # Skip embedding and head
            new_state_dict[key].copy_(old_state_dict[key])  # Copy parameter

    old_embedding = old_state_dict['token_embedding.weight']  # Get old embedding
    new_embedding = torch.zeros((new_vocab_size, old_embedding.size(1)), device=device)  # Create new embedding
    copy_size = min(model.vocab_size, new_vocab_size)  # Determine copy size
    new_embedding[:copy_size] = old_embedding[:copy_size]  # Copy old embedding
    if new_vocab_size > model.vocab_size:  # If new size is larger
        new_embedding[copy_size:] = torch.randn(new_vocab_size - copy_size,
                                                old_embedding.size(1)) * 0.02  # Initialize new entries
    new_state_dict['token_embedding.weight'].copy_(new_embedding)  # Update new state dict

    old_head_weight = old_state_dict['head.weight']  # Get old head weight
    new_head_weight = torch.zeros((new_vocab_size, old_head_weight.size(1)), device=device)  # Create new head weight
    new_head_weight[:copy_size] = old_head_weight[:copy_size]  # Copy old head weight
    if new_vocab_size > model.vocab_size:  # If new size is larger
        new_head_weight[copy_size:] = torch.randn(new_vocab_size - copy_size,
                                                  old_head_weight.size(1)) * 0.02  # Initialize new entries
    new_state_dict['head.weight'].copy_(new_head_weight)  # Update new state dict

    if 'head.bias' in old_state_dict:  # Check if head bias exists
        old_head_bias = old_state_dict['head.bias']  # Get old head bias
        new_head_bias = torch.zeros(new_vocab_size, device=device)  # Create new head bias
        new_head_bias[:copy_size] = old_head_bias[:copy_size]  # Copy old head bias
        if new_vocab_size > model.vocab_size:  # If new size is larger
            new_head_bias[copy_size:] = torch.randn(new_vocab_size - copy_size) * 0.02  # Initialize new entries
        new_state_dict['head.bias'].copy_(new_head_bias)  # Update new state dict

    new_model.load_state_dict(new_state_dict)  # Load updated state dict into new model
    logger.info("Model resized")  # Log resizing completion
    return new_model  # Return resized model


def train_model(model, sequences, device, max_iters=1000, batch_size=2):
    """Function to train the GPT model."""
    logger.info("Starting fine-tuning...")  # Log training start
    model.train()  # Set model to training mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Initialize optimizer
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
            logger.info(f"Iteration {i + 1}/{max_iters}, Loss: {loss.item():.4f}")  # Log loss
    logger.info("Fine-tuning completed")  # Log training completion
    log_resources()  # Log resources
    return model  # Return trained model


def signal_handler(sig, frame):
    """Function to handle interrupts."""
    logger.info("Saving partial model due to interruption...")  # Log saving
    if 'model' in globals():  # Check if model exists
        with open('model_funsd_partial.pkl', 'wb') as f:  # Open file for writing
            pickle.dump(model, f)  # Dump model
    sys.exit(0)  # Exit program


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Set signal handler
    logger.info("Starting FUNSD fine-tuning...")  # Log start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device
    logger.info(f"Using device: {device}")  # Log device
    log_resources()  # Log resources

    try:
        with open('model.pkl', 'rb') as f:  # Open model file
            model = pickle.load(f).to(device)  # Load model
        with open('vocab.pkl', 'rb') as f:  # Open vocab file
            vocab = pickle.load(f)  # Load vocab
        logger.info(f"Loaded model and vocab with {len(vocab)} words")  # Log loaded vocab size
        log_resources()  # Log resources
    except FileNotFoundError:
        logger.error("model.pkl or vocab.pkl not found. Run Training.py first.")  # Log error
        sys.exit(1)  # Exit program

    funsd_data = load_funsd_data()  # Load FUNSD data
    molding_data = load_molding_data()  # Load molding data
    all_sentences = funsd_data + molding_data  # Combine sentences
    logger.info(
        f"Loaded {len(all_sentences)} sentences ({len(funsd_data)} FUNSD + {len(molding_data)} molding)")  # Log loaded sentences

    vocab = update_vocab(all_sentences, vocab)  # Update vocabulary
    sequences = tokenize(all_sentences, vocab, block_size=block_size)  # Tokenize sentences

    if len(vocab) != model.vocab_size:  # Check if vocab size changed
        model = resize_model(model, len(vocab), device)  # Resize model

    model = train_model(model, sequences, device, max_iters=1000, batch_size=2)  # Train model

    with open('model_updated.pkl', 'wb') as f:  # Save updated model
        pickle.dump(model, f)  # Dump model
    with open('vocab_updated.pkl', 'wb') as f:  # Save updated vocab
        pickle.dump(vocab, f)  # Dump vocab
    logger.info("Saved model_updated.pkl and vocab_updated.pkl")  # Log saving
    log_resources()  # Log resources
