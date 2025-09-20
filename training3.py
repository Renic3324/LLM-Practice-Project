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
import gc  # Import gc module for garbage collection

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


def load_tableqa_data():
    """Function to load WikiTableQuestions dataset."""
    logger.info("Loading WikiTableQuestions data...")  # Log loading start
    try:
        dataset = load_dataset("wikitablequestions", split="train")  # Load dataset
        logger.info("Loaded WikiTableQuestions dataset")  # Log loading success
        sentences = []  # Initialize list for sentences
        for item in dataset:  # Loop through dataset items
            question = item.get("question", "").strip()  # Get question
            answer = item.get("answer", [])  # Get answer
            if isinstance(answer, list):  # Check if answer is list
                answer_text = " ".join(str(a) for a in answer if str(a).strip()).strip()  # Join answer list
            else:
                answer_text = str(answer).strip()  # Convert answer to string
            if question and answer_text:  # Check if both are present
                sentences.append(f"Q: {question} A: {answer_text}")  # Append formatted sentence
        logger.info(f"Loaded {len(sentences)} sentences")  # Log number of sentences
        log_resources()  # Log resources
        return sentences  # Return sentences
    except Exception as e:
        logger.error(f"Failed to load WikiTableQuestions data: {e}")  # Log error
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


def train_model(model, sequences, device, max_iters=100, batch_size=8, grad_accum_steps=8):
    """Function to fine-tune the GPT model with gradient accumulation."""
    logger.info("Starting fine-tuning...")  # Log training start
    model.train()  # Set model to training mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Initialize optimizer
    sequences = torch.tensor(sequences, dtype=torch.long).to(device)  # Convert sequences to tensor on device
    for i in range(max_iters):  # Loop through iterations
        optimizer.zero_grad()  # Zero gradients
        loss_accum = 0.0  # Initialize loss accumulator
        for _ in range(grad_accum_steps):  # Loop through gradient accumulation steps
            idx = torch.randint(0, len(sequences), (batch_size,))  # Generate random indices
            batch = sequences[idx]  # Get batch
            inputs = batch[:, :-1]  # Get inputs
            targets = batch[:, 1:]  # Get targets
            logits, loss = model(inputs, targets)  # Forward pass
            (loss / grad_accum_steps).backward()  # Backward pass with accumulation
            loss_accum += loss.item() / grad_accum_steps  # Accumulate loss
            del inputs, targets, logits, loss  # Delete tensors to free memory
            gc.collect()  # Collect garbage
            if device.type == 'cuda':  # If using CUDA
                torch.cuda.empty_cache()  # Clear CUDA cache
        optimizer.step()  # Optimizer step
        if (i + 1) % 50 == 0:  # Log every 50 iterations
            logger.info(f"Iteration {i + 1}/{max_iters}, Avg Loss: {loss_accum:.4f}")  # Log average loss
    logger.info("Fine-tuning completed")  # Log training completion
    log_resources()  # Log resources
    return model  # Return trained model


def signal_handler(sig, frame):
    """Function to handle interrupts."""
    logger.info("Saving partial model due to interruption...")  # Log saving
    if 'model' in globals():  # Check if model exists
        with open('model_tableqa_partial.pkl', 'wb') as f:  # Open file for writing
            pickle.dump(model, f, protocol=4)  # Dump model with protocol 4
    sys.exit(1)  # Exit program with error code 1


if __name__ == "__main__":
    logger.info("Starting table QA fine-tuning...")  # Log start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device
    logger.info(f"Using device: {device}")  # Log device
    log_resources()  # Log resources

    signal.signal(signal.SIGINT, signal_handler)  # Set signal handler

    try:
        if not os.path.exists("model_updated.pkl") or not os.path.exists("vocab_updated.pkl"):  # Check for files
            logger.error("model_updated.pkl or vocab_updated.pkl not found. Run Training2.py first.")  # Log error
            sys.exit(1)  # Exit program
        if not validate_pickle_file("model_updated.pkl"):  # Validate model file
            logger.error("model_updated.pkl is corrupted. Rerun Training2.py.")  # Log error
            sys.exit(1)  # Exit program
        if not validate_pickle_file("vocab_updated.pkl"):  # Validate vocab file
            logger.error("vocab_updated.pkl is corrupted. Rerun Training2.py.")  # Log error
            sys.exit(1)  # Exit program
        with open("model_updated.pkl", "rb") as f:  # Open model file
            model = pickle.load(f).to(device)  # Load model
        with open("vocab_updated.pkl", "rb") as f:  # Open vocab file
            vocab = pickle.load(f)  # Load vocab
        logger.info(f"Loaded model and vocab with {len(vocab)} words")  # Log loaded vocab size
        log_resources()  # Log resources
    except Exception as e:
        logger.error(f"Error loading model/vocab: {e}")  # Log error
        sys.exit(1)  # Exit program

    tableqa_data = load_tableqa_data()  # Load table QA data
    #molding_data = load_molding_data()  # Load molding data
    all_sentences = tableqa_data #+ molding_data  # Combine sentences
    logger.info(f"Loaded {len(all_sentences)} sentences")  # Log loaded sentences

    new_vocab = build_vocab(all_sentences)  # Build new vocabulary
    vocab = list(set(vocab + new_vocab))  # Update vocabulary
    sequences = tokenize(all_sentences, vocab, block_size=block_size)  # Tokenize sentences

    if len(vocab) != model.vocab_size:  # Check if vocab size changed
        model = resize_model(model, len(vocab), device)  # Resize model

    model = train_model(model, sequences, device)  # Train model

    with open('model_updated_tableqa.pkl', 'wb') as f:  # Save updated model
        pickle.dump(model, f, protocol=4)  # Dump model with protocol 4
    with open('vocab_tableqa.pkl', 'wb') as f:  # Save updated vocab
        pickle.dump(vocab, f, protocol=4)  # Dump vocab with protocol 4
    logger.info("Saved model_updated_tableqa.pkl and vocab_tableqa.pkl")  # Log saving
    log_resources()  # Log resources
