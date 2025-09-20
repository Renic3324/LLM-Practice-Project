import time  # Import the time module to measure execution time and add delays where necessary, such as for rate limiting in web requests
import logging  # Import the logging module to handle logging of informational, debug, and error messages throughout the script
import sys  # Import the sys module to interact with system-specific parameters, such as exiting the program or accessing standard output
import os  # Import the os module to handle file system operations, such as checking file existence, removing files, or getting absolute paths
import pickle  # Import the pickle module to serialize and deserialize Python objects, used for loading saved models and vocabularies
import torch  # Import the torch module, which is the core library for PyTorch, used for tensor operations and model inference on CPU or GPU
import csv  # Import the csv module to handle CSV file operations, such as writing question-answer pairs to a file
import re  # Import the re module for regular expression operations, used for pattern matching like extracting URLs from prompts
from datasets import load_dataset  # Import the load_dataset function from the datasets library to load pre-built datasets like UltraChat for training or fine-tuning
from model import GPTLanguageModel, tokenize  # Import the GPTLanguageModel class and the tokenize function from the model.py script, where the model architecture is defined
from config import block_size  # Import the block_size variable from the config.py script, which defines the maximum sequence length for tokenization and model input
import psutil  # Import psutil module for system monitoring
import signal  # Import the signal module to handle system signals like interrupts (e.g., Ctrl+C)
import gc  # Import gc module for garbage collection

# Configure the logging system to capture and output messages at various levels (INFO, DEBUG, ERROR)
# This setup ensures logs are written to a file for persistent storage and also printed to the console for real-time monitoring
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level to INFO, meaning INFO and higher severity levels (WARNING, ERROR) will be logged
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the format of each log message to include timestamp, log level, and the message itself
    handlers=[  # Specify the destinations for log messages
        logging.StreamHandler(sys.stdout)  # Handler to write logs to the standard output (console) for immediate visibility during execution
    ]
)
logger = logging.getLogger(__name__)  # Create a logger instance named after the current module for organized logging
# Ensure that logs are flushed immediately to the console by setting the flush method for the stream handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setStream(sys.stdout)  # Set the stream to standard output
        handler.flush = sys.stdout.flush  # Ensure that logs are flushed right away to avoid buffering delays

def log_resources():
    """Function to log the current usage of system resources, including CPU, RAM, and GPU memory."""
    cpu_percent = psutil.cpu_percent()  # Retrieve the current CPU utilization as a percentage using psutil
    mem = psutil.virtual_memory()  # Retrieve virtual memory statistics using psutil
    mem_used = mem.used / 1e9  # Convert the used memory from bytes to gigabytes for easier readability
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0  # Retrieve allocated GPU memory in gigabytes if CUDA is available, otherwise set to 0
    # Log the resource usage in a formatted string for monitoring system performance during execution
    logger.info(f"Resources: CPU {cpu_percent:.1f}%, RAM {mem_used:.2f}GB, GPU {gpu_mem:.2f}GB")

def validate_pickle_file(file_path):
    """Function to validate the integrity of a pickle file by checking its magic number."""
    try:
        with open(file_path, 'rb') as f:  # Open the file in binary read mode to access its contents
            header = f.read(4)  # Read the first 4 bytes of the file, which contain the pickle magic number
            # Check if the header starts with the valid pickle magic numbers for protocol versions 3 or 4
            valid_magic = header.startswith(b'\x80\x03') or header.startswith(b'\x80\x04')
            logger.debug(f"File {file_path} magic number valid: {valid_magic}")  # Log the validation result at debug level
            return valid_magic  # Return True if valid, False otherwise
    except Exception as e:
        logger.error(f"Failed to validate {file_path}: {e}")  # Log any error that occurs during validation
        return False  # Return False if validation fails due to an exception

def load_ultrachat_data(max_samples=5000):
    """Function to load a subset of the UltraChat dataset, with caching to avoid repeated downloads."""
    logger.info("Loading UltraChat dataset...")  # Log the start of the dataset loading process
    cache_file = 'ultrachat_cache.pt'  # Define the name of the cache file for storing the loaded sentences
    if os.path.exists(cache_file):  # Check if the cache file already exists
        try:
            sentences = torch.load(cache_file, map_location='cpu', weights_only=False)  # Load the sentences from the cache file using torch.load
            logger.info(f"Loaded {len(sentences)} sentences from cache")  # Log the number of sentences loaded from cache
            return sentences[:max_samples]  # Return a subset of the sentences up to the maximum specified
        except Exception as e:
            logger.error(f"Error loading cached UltraChat data: {e}")  # Log any error that occurs during cache loading
            return []  # Return an empty list if loading fails
    try:
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{max_samples}]")  # Load the UltraChat dataset from Hugging Face, limiting to max_samples
        sentences = []  # Initialize an empty list to store the sentences
        for item in dataset:  # Loop through each item in the dataset
            messages = item.get("messages", [])  # Get the messages from the item, default to empty list if not present
            for i in range(len(messages) - 1):  # Loop through messages to find user-assistant pairs
                if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":  # Check for user-assistant pair
                    instruction = messages[i]["content"].strip()  # Get and strip the user instruction
                    response = messages[i + 1]["content"].strip()  # Get and strip the assistant response
                    if instruction and response and len(instruction) > 10 and len(response) > 20:  # Check if both are valid and sufficiently long
                        sentences.append(f"{instruction} {response}")  # Append the combined instruction and response as a sentence
        torch.save(sentences, cache_file)  # Save the sentences to the cache file using torch.save for future use
        logger.info(f"Loaded and cached {len(sentences)} sentences")  # Log the number of sentences loaded and cached
        log_resources()  # Call function to log current system resources
        return sentences  # Return the list of sentences
    except Exception as e:
        logger.error(f"Failed to load UltraChat data: {e}")  # Log any error that occurs during dataset loading
        return []  # Return an empty list if loading fails
'''
def load_molding_data():
    """Function to load molding terms from a file, deduplicating and stripping lines."""
    logger.info("Loading molding terms...")  # Log the start of loading molding terms
    try:
        with open('molding_terms.txt', 'r', encoding='utf-8') as f:  # Open the molding_terms.txt file in read mode with UTF-8 encoding
            lines = list(set(line.strip() for line in f if line.strip()))  # Read lines, strip whitespace, remove empty lines, and deduplicate using a set converted back to list
        logger.info(f"Loaded {len(lines)} unique molding terms")  # Log the number of unique terms loaded
        log_resources()  # Call function to log current system resources
        return lines  # Return the list of unique molding terms
    except Exception as e:
        logger.error(f"Error reading molding terms: {e}")  # Log any error that occurs during file reading
        return []  # Return an empty list if an error occurs
'''
def update_vocab(sentences, existing_vocab):
    """Function to update the existing vocabulary with new tokens from sentences."""
    logger.info("Updating vocabulary...")  # Log the start of vocabulary update
    tokens = set()  # Initialize an empty set to store new tokens
    for sentence in sentences:  # Loop through each sentence in the list of sentences
        sentence = re.sub(r'\s+', ' ', sentence).strip().lower()  # Clean the sentence by replacing multiple spaces with single space, stripping whitespace, and converting to lowercase
        words = re.findall(r'[a-z]+|[0-9]+|[.,]', sentence)  # Extract words using regular expression: lowercase letters, numbers, or punctuation
        tokens.update(words)  # Add the extracted words to the set of tokens
    vocab = list(set(existing_vocab + sorted(list(tokens))))  # Combine existing vocabulary with new tokens, sort, and convert to list to remove duplicates
    logger.info(f"Updated vocabulary to {len(vocab)} words")  # Log the updated vocabulary size
    log_resources()  # Call function to log current system resources
    return vocab  # Return the updated vocabulary list

def resize_model(model, new_vocab_size, device):
    """Function to resize the model when the vocabulary size changes."""
    logger.info(f"Resizing model from vocab size {model.vocab_size} to {new_vocab_size}")  # Log the resizing process start
    new_model = GPTLanguageModel(vocab_size=new_vocab_size).to(device)  # Create a new model instance with the new vocabulary size and move it to the specified device
    new_state_dict = new_model.state_dict()  # Get the state dictionary of the new model
    old_state_dict = model.state_dict()  # Get the state dictionary of the old model

    for key in old_state_dict:  # Loop through each key in the old state dictionary
        if key not in ['token_embedding.weight', 'head.weight', 'head.bias']:  # Skip keys related to embedding and head layers that need resizing
            new_state_dict[key].copy_(old_state_dict[key])  # Copy parameters from old to new for non-resizable layers

    # Resize token embedding layer
    old_embedding = old_state_dict['token_embedding.weight']  # Get the old token embedding weights
    new_embedding = torch.zeros((new_vocab_size, old_embedding.size(1)), device=device)  # Create a new tensor for embedding weights with the new size
    copy_size = min(model.vocab_size, new_vocab_size)  # Determine how many rows to copy from old to new
    new_embedding[:copy_size] = old_embedding[:copy_size]  # Copy the common part from old to new embedding
    if new_vocab_size > model.vocab_size:  # If the new vocabulary is larger
        new_embedding[copy_size:] = torch.randn(new_vocab_size - copy_size, old_embedding.size(1)) * 0.02  # Initialize new entries with small random values
    new_state_dict['token_embedding.weight'].copy_(new_embedding)  # Update the new state dictionary with the resized embedding

    # Resize head weight
    old_head_weight = old_state_dict['head.weight']  # Get the old head weight
    new_head_weight = torch.zeros((new_vocab_size, old_head_weight.size(1)), device=device)  # Create a new tensor for head weight with the new size
    new_head_weight[:copy_size] = old_head_weight[:copy_size]  # Copy the common part from old to new head weight
    if new_vocab_size > model.vocab_size:  # If the new vocabulary is larger
        new_head_weight[copy_size:] = torch.randn(new_vocab_size - copy_size, old_head_weight.size(1)) * 0.02  # Initialize new entries with small random values
    new_state_dict['head.weight'].copy_(new_head_weight)  # Update the new state dictionary with the resized head weight

    # Resize head bias if it exists
    if 'head.bias' in old_state_dict:  # Check if head bias is present in the old model
        old_head_bias = old_state_dict['head.bias']  # Get the old head bias
        new_head_bias = torch.zeros(new_vocab_size, device=device)  # Create a new tensor for head bias with the new size
        new_head_bias[:copy_size] = old_head_bias[:copy_size]  # Copy the common part from old to new head bias
        if new_vocab_size > model.vocab_size:  # If the new vocabulary is larger
            new_head_bias[copy_size:] = torch.randn(new_vocab_size - copy_size) * 0.02  # Initialize new entries with small random values
        new_state_dict['head.bias'].copy_(new_head_bias)  # Update the new state dictionary with the resized head bias

    new_model.load_state_dict(new_state_dict)  # Load the updated state dictionary into the new model
    logger.info("Model resized")  # Log the completion of resizing
    return new_model  # Return the resized model

def train_model(model, sequences, device, max_iters=500, batch_size=8, grad_accum_steps=4):
    """Function to fine-tune the GPT model with mixed precision and gradient accumulation."""
    logger.info("Starting fine-tuning...")  # Log the start of fine-tuning
    model.train()  # Set the model to training mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Initialize the optimizer with AdamW algorithm and learning rate
    sequences = torch.tensor(sequences, dtype=torch.long).to(device)  # Convert the sequences to a tensor and move to the device
    for i in range(max_iters):  # Loop through the specified number of iterations
        optimizer.zero_grad()  # Zero out the gradients from the previous step
        loss_accum = 0.0  # Initialize a loss accumulator for gradient accumulation
        for _ in range(grad_accum_steps):  # Loop through the gradient accumulation steps
            idx = torch.randint(0, len(sequences), (batch_size,))  # Generate random indices for batch selection
            batch = sequences[idx]  # Select the batch from the sequences
            inputs = batch[:, :-1]  # Extract inputs from the batch (all but the last token)
            targets = batch[:, 1:]  # Extract targets from the batch (all but the first token)
            logits, loss = model(inputs, targets)  # Perform forward pass to get logits and loss
            (loss / grad_accum_steps).backward()  # Perform backward pass, scaling the loss for accumulation
            loss_accum += loss.item() / grad_accum_steps  # Accumulate the scaled loss value
            del inputs, targets, logits, loss  # Delete variables to free memory
            gc.collect()  # Explicitly collect garbage to free memory
            if device.type == 'cuda':  # Check if using CUDA device
                torch.cuda.empty_cache()  # Clear the CUDA memory cache
        optimizer.step()  # Perform the optimizer step to update model parameters
        if (i + 1) % 50 == 0:  # Check if it's time to log the progress (every 50 iterations)
            logger.info(f"Iteration {i + 1}/{max_iters}, Avg Loss: {loss_accum:.4f}")  # Log the iteration and average loss
    logger.info("Fine-tuning completed")  # Log the completion of fine-tuning
    log_resources()  # Log current system resources
    return model  # Return the fine-tuned model

def signal_handler(sig, frame):
    """Function to handle interrupts by saving partial model."""
    logger.info("Saving partial model due to interruption...")  # Log the saving process
    if 'model' in globals():  # Check if the model variable exists in the global scope
        with open('model_dolly_partial.pkl', 'wb') as f:  # Open a file for writing the partial model
            pickle.dump(model, f, protocol=4)  # Dump the model to the file using pickle protocol 4
    sys.exit(1)  # Exit the program with exit code 1 indicating an error

if __name__ == "__main__":
    logger.info("Starting UltraChat fine-tuning...")  # Log the start of the script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Determine the device to use (GPU if available, else CPU)
    logger.info(f"Using device: {device}")  # Log the device being used
    log_resources()  # Log initial system resources

    signal.signal(signal.SIGINT, signal_handler)  # Set the signal handler for interrupt signals (e.g., Ctrl+C)

    try:
        if not os.path.exists("model_updated_tableqa.pkl") or not os.path.exists("vocab_tableqa.pkl"):  # Check if required files exist
            logger.error("model_updated_tableqa.pkl or vocab_tableqa.pkl not found. Run Training3.py first.")  # Log error if files are missing
            sys.exit(1)  # Exit the program if files are missing
        if not validate_pickle_file("model_updated_tableqa.pkl"):  # Validate the model file
            logger.error("model_updated_tableqa.pkl is corrupted. Rerun Training3.py.")  # Log error if corrupted
            sys.exit(1)  # Exit the program
        if not validate_pickle_file("vocab_tableqa.pkl"):  # Validate the vocab file
            logger.error("vocab_tableqa.pkl is corrupted. Rerun Training3.py.")  # Log error if corrupted
            sys.exit(1)  # Exit the program
        with open("model_updated_tableqa.pkl", "rb") as f:  # Open the model file
            model = pickle.load(f).to(device)  # Load and move the model to the device
        with open("vocab_tableqa.pkl", "rb") as f:  # Open the vocab file
            vocab = pickle.load(f)  # Load the vocabulary
        logger.info(f"Loaded model and vocab with {len(vocab)} words")  # Log the loaded vocabulary size
        log_resources()  # Log resources after loading
    except Exception as e:
        logger.error(f"Error loading model/vocab: {e}")  # Log any loading error
        sys.exit(1)  # Exit the program

    ultrachat_data = load_ultrachat_data(max_samples=5000)  # Load UltraChat data with max samples
    #molding_data = load_molding_data()  # Load molding data
    all_sentences = ultrachat_data + #molding_data  # Combine the data
    logger.info(f"Loaded {len(all_sentences)} sentences")  # Log the number of combined sentences

    vocab = update_vocab(all_sentences, vocab)  # Update the vocabulary with new data
    sequences = tokenize(all_sentences, vocab, block_size=block_size)  # Tokenize the combined sentences

    if len(vocab) != model.vocab_size:  # Check if the vocabulary size has changed
        model = resize_model(model, len(vocab), device)  # Resize the model if necessary

    model = train_model(model, sequences, device)  # Fine-tune the model

    with open('model_updated_dolly.pkl', 'wb') as f:  # Open file for saving the updated model
        pickle.dump(model, f, protocol=4)  # Save the model using pickle protocol 4
    with open('vocab_dolly.pkl', 'wb') as f:  # Open file for saving the updated vocabulary
        pickle.dump(vocab, f, protocol=4)  # Save the vocabulary using pickle protocol 4
    logger.info("Saved model_updated_dolly.pkl and vocab_dolly.pkl")  # Log the saving of files
    log_resources()  # Log final resources
