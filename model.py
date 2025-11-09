import torch  # Import torch module for deep learning operations
import re  # Import re module for regular expressions
from torch import nn  # Import nn module from torch for neural network components
from torch.nn import functional as F  # Import functional from nn for functional operations
from config import block_size, n_embd, n_head, n_layer, dropout  # Import config parameters
import logging  # Import logging module for logging messages
import sys # System-specific parameters (e.g., stdout, exit)
import psutil  # Required for log_resources
import datasets  # Required for streaming

# Configure logger
logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    """Function to handle interrupts."""
    logger.info("Saving partial model due to interruption...")  # Log saving
    if 'model' in globals():  # Check if model exists
        with open('model_dolly_partial.pkl', 'wb') as f:  # Open file for writing
            pickle.dump(model, f, protocol=4)  # Dump model with protocol 4
    sys.exit(1)  # Exit program with error code 1

# List of molding terms to be used in training
molding_terms = [
    "Acetal",
    "Acrylic",
    "Acrylonitrile Butadiene Styrene (ABS)",
    "Additive",
    "Adhesion",
    "Air Entrapment",
    "Alloy",
    "Amorphous",
    "Annealing",
    "Antioxidant",
    "Antistatic Agent",
    "Aspect Ratio",
    "Back Pressure",
    "Back Rake",
    "Baffle",
    "Barrel",
    "Barrel Temperature",
    "Belt Sander",
    "Bead",
    "Bead Blasting",
    "Bending",
    "Billet",
    "Blanking",
    "Blister",
    "Blow Molding",
    "Blow Pin",
    "Blow Pressure",
    "Blow Rate",
    "Blow-Up Ratio",
    "Blown Film",
    "Bond Strength",
    "Boss",
    "Bottle",
    "Brittleness",
    "Bubbler",
    "Bulk Density",
    "Burn Mark",
    "Cavity",
    "Cavity Pressure",
    "Charge",
    "Chiller",
    "Clamping Force",
    "Clamping Plate",
    "Clamping Pressure",
    "Clamp Tonnage",
    "Clearance",
    "Co-extrusion",
    "Coefficient of Thermal Expansion (CTE)",
    "Cold Slug",
    "Cold Slug Well",
    "Colorant",
    "Compression Molding",
    "Compression Ratio",
    "Compressive Strength",
    "Conditioning",
    "Cooling Channels",
    "Cooling Fixture",
    "Core",
    "Core Pin",
    "Core-Cavity",
    "Corrosion Resistance",
    "Crazing",
    "Creep",
    "Cross-linking",
    "Crystallinity",
    "Cure",
    "Curing Temperature",
    "Cycle Time",
    "Daylight Opening",
    "Deflashing",
    "Degassing",
    "Degradation",
    "Delamination",
    "Density",
    "Design for Manufacturability (DFM)",
    "Dessicant",
    "Die",
    "Die Gap",
    "Die Swell",
    "Dimensional Stability",
    "Draft",
    "Draft Angle",
    "Drying",
    "Durometer",
    "Dwell",
    "Ejector Pin",
    "Ejector Plate",
    "Ejector Rod",
    "Ejector Sleeve",
    "Elastomer",
    "Electrical Discharge Machining (EDM)",
    "Embossing",
    "Encapsulating",
    "End Cap",
    "Environmental Stress Cracking (ESC)",
    "Extrusion",
    "Family Mold",
    "Fan Gate",
    "Feed Throat",
    "Fiber Orientation",
    "Filling",
    "Fill Time",
    "Film",
    "Finish",
    "Flame Retardant",
    "Flash",
    "Flash Gate",
    "Flexural Modulus",
    "Flexural Strength",
    "Flow Lines",
    "Flow Marks",
    "Flow Rate",
    "Foam Molding",
    "Gate",
    "Gate Blush",
    "Gate Seal",
    "Gate Vestige",
    "Glass Transition Temperature (Tg)",
    "Gloss",
    "Granule",
    "Guide Pins",
    "Hardness",
    "Heat Deflection Temperature (HDT)",
    "Heat Stake",
    "Heater Bands",
    "High-Density Polyethylene (HDPE)",
    "Hopper",
    "Hot Runner",
    "Impact Strength",
    "Injection Blow Molding",
    "Injection Molding",
    "Injection Pressure",
    "Injection Speed",
    "Insert Molding",
    "Insulated Runner",
    "Isotropic",
    "Jetting",
    "Knit Line",
    "L/D Ratio",
    "Laminar Flow",
    "Laminating",
    "Land",
    "LCP (Liquid Crystal Polymer)",
    "Leader Pins",
    "Light-Curable Resins",
    "Linear Low-Density Polyethylene (LLDPE)",
    "Living Hinge",
    "Low-Density Polyethylene (LDPE)",
    "Lubricant",
    "Machine Shot Capacity",
    "Melt Flow Index (MFI)",
    "Melt Temperature",
    "Melting Point",
    "Mold",
    "Mold Base",
    "Mold Cavity",
    "Mold Release Agent",
    "Mold Shrinkage",
    "Mold Temperature",
    "Mold Venting",
    "Molding Cycle",
    "Molding Pressure",
    "Mold-Tight",
    "Multi-Cavity Mold",
    "Multi-Shot Molding",
    "Nesting",
    "Nozzle",
    "Nozzle Temperature",
    "Nucleating Agent",
    "Overmolding",
    "Pack Pressure",
    "Packing",
    "Parison",
    "Parting Line",
    "Pin Gate",
    "Pitch",
    "Plastic",
    "Plastic Deformation",
    "Plasticizer",
    "Platen",
    "Plunger",
    "Polyamide (Nylon)",
    "Polycarbonate (PC)",
    "Polyethylene (PE)",
    "Polyethylene Terephthalate (PET)",
    "Polymer",
    "Polypropylene (PP)",
    "Polystyrene (PS)",
    "Polyurethane (PU)",
    "Polyvinyl Chloride (PVC)",
    "Porosity",
    "Post-Mold Shrinkage",
    "Pressure",
    "Purging",
    "PVC",
    "Regrind",
    "Reinforcement",
    "Release Agent",
    "Residence Time",
    "Resin",
    "Rheology",
    "Ribs",
    "Rotational Molding",
    "Runner",
    "Runnerless Molding",
    "Screw",
    "Screw Speed",
    "Shear",
    "Shear Rate",
    "Shear Stress",
    "Short Shot",
    "Shot",
    "Shot Size",
    "Shrinkage",
    "Side Action",
    "Sink Mark",
    "Slide",
    "Specific Gravity",
    "Spiral Flow",
    "Sprue",
    "Sprue Bushing",
    "Stack Mold",
    "Staging",
    "Stress",
    "Stress Cracking",
    "Structural Foam Molding",
    "Support Pillars",
    "Surface Finish",
    "Surface Treatment",
    "Tab Gate",
    "Tensile Strength",
    "Texture",
    "Thermal Conductivity",
    "Thermal Expansion",
    "Thermoplastic",
    "Thermoset",
    "Tie Bar",
    "Tolerance",
    "Tooling",
    "Transfer Molding",
    "Tunnel Gate",
    "Two-Plate Mold",
    "Two-Shot Molding",
    "Ultimate Tensile Strength",
    "Undercut",
    "Vent",
    "Venting",
    "Viscosity",
    "Void",
    "Wall Thickness",
    "Warp",
    "Warpage",
    "Water Absorption",
    "Wear Resistance",
    "Weld Line",
    "Weld Strength"
]

class Head(nn.Module):
    """Class for a single attention head."""
    def __init__(self, head_size):
        super().__init__()  # Call super class initializer
        self.key = nn.Linear(n_embd, head_size, bias=False)  # Linear layer for key
        self.query = nn.Linear(n_embd, head_size, bias=False)  # Linear layer for query
        self.value = nn.Linear(n_embd, head_size, bias=False)  # Linear layer for value
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # Register tril buffer for masking
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x):
        """Forward pass for attention head."""
        B, T, C = x.shape  # Get batch, time, channel dimensions
        k = self.key(x)  # Compute key
        q = self.query(x)  # Compute query
        wei = q @ k.transpose(-2, -1) * C**-0.5  # Compute attention weights
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Mask future tokens
        wei = torch.softmax(wei, dim=-1)  # Softmax for probabilities
        wei = self.dropout(wei)  # Apply dropout
        v = self.value(x)  # Compute value
        out = wei @ v  # Weighted sum
        return out  # Return output

class MultiHeadAttention(nn.Module):
    """Class for multi-head attention."""
    def __init__(self, num_heads, head_size):
        super().__init__()  # Call super class initializer
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # List of attention heads
        self.proj = nn.Linear(n_embd, n_embd)  # Projection layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x):
        """Forward pass for multi-head attention."""
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate head outputs
        out = self.dropout(self.proj(out))  # Project and apply dropout
        return out  # Return output

class FeedForward(nn.Module):
    """Class for feed-forward network."""
    def __init__(self, n_embd):
        super().__init__()  # Call super class initializer
        self.net = nn.Sequential(  # Sequential network
            nn.Linear(n_embd, 4 * n_embd),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(4 * n_embd, n_embd),  # Linear layer
            nn.Dropout(dropout),  # Dropout layer
        )

    def forward(self, x):
        """Forward pass for feed-forward network."""
        return self.net(x)  # Pass input through network

class Block(nn.Module):
    """Class for transformer block."""
    def __init__(self, n_embd, n_head):
        super().__init__()  # Call super class initializer
        head_size = n_embd // n_head  # Compute head size
        self.sa = MultiHeadAttention(n_head, head_size)  # Self-attention layer
        self.ffwd = FeedForward(n_embd)  # Feed-forward layer
        self.ln1 = nn.LayerNorm(n_embd)  # Layer norm 1
        self.ln2 = nn.LayerNorm(n_embd)  # Layer norm 2

    def forward(self, x):
        """Forward pass for transformer block."""
        x = x + self.sa(self.ln1(x))  # Self-attention residual
        x = x + self.ffwd(self.ln2(x))  # Feed-forward residual
        return x  # Return output

class GPTLanguageModel(nn.Module):
    """Class for GPT language model."""
    def __init__(self, vocab_size):
        super().__init__()  # Call super class initializer
        self.vocab_size = vocab_size  # Set vocabulary size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)  # Token embedding layer
        self.position_embedding = nn.Embedding(block_size, n_embd)  # Position embedding layer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])  # Transformer blocks
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm
        self.head = nn.Linear(n_embd, vocab_size)  # Output head
        self.apply(self._init_weights)  # Apply weight initialization

    def _init_weights(self, module):
        """Function to initialize weights."""
        if isinstance(module, nn.Linear):  # Check if linear layer
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize weights
            if module.bias is not None:  # Check if bias exists
                torch.nn.init.zeros_(module.bias)  # Initialize bias to zero
        elif isinstance(module, nn.Embedding):  # Check if embedding layer
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize weights

    def forward(self, idx, targets=None):
        """Forward pass for GPT language model."""
        B, T = idx.shape  # Get batch and time dimensions
        tok_emb = self.token_embedding(idx)  # Token embeddings
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # Position embeddings
        x = tok_emb + pos_emb  # Combine embeddings
        x = self.blocks(x)  # Pass through transformer blocks
        x = self.ln_f(x)  # Final layer norm
        logits = self.head(x)  # Output logits
        if targets is None:  # If no targets
            loss = None  # No loss
        else:
            B, T, C = logits.shape  # Get dimensions
            logits = logits.view(B * T, C)  # Reshape logits
            targets = targets.view(B * T)  # Reshape targets
            loss = torch.nn.functional.cross_entropy(logits, targets)  # Compute loss
        return logits, loss  # Return logits and loss

def tokenize(sentences, vocab, block_size=256):
    """Function to tokenize sentences."""
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}  # Create word to index dictionary
    sequences = []  # Initialize list for sequences
    for sentence in sentences:  # Loop through sentences
        sentence = re.sub(r'\s+', ' ', sentence).strip().lower()  # Clean sentence
        words = re.findall(r'[a-z]+|[0-9]+|[.,]', sentence)  # Extract words
        tokens = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]  # Get tokens
        tokens = [word_to_idx['<SOS>']] + tokens + [word_to_idx['<EOS>']]  # Add SOS and EOS
        if len(tokens) > block_size:  # Check length
            tokens = tokens[:block_size]  # Truncate if too long
        else:
            tokens += [word_to_idx['<PAD>']] * (block_size - len(tokens))  # Pad if too short
        sequences.append(tokens)  # Append sequence
    return sequences  # Return sequences

def train_model(model, sequence_generator, device, max_iters=2000, batch_size=8, grad_accum_steps = 4):
    """Function to train the GPT model."""
    logger.info("Starting training...")  # Log training start
    model.train()  # Set model to training mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # Initialize optimizer
    for i in range(max_iters):  # Loop through iterations
        optimizer.zero_grad()  # Zero gradients
        loss_accum = 0.0 # Initialize loss accumulator
        for _ in range(grad_accum_steps):
            # Sample a random batch from sentences
            batch_sentences = [next(sequence_generator) for _ in range(batch_size)]
            # Tokenize batch on-the-fly (assumes tokenize from model.py)
            batch_tokens = tokenize(batch_sentences, vocab, block_size=block_size)
            batch_tokens = torch.tensor(batch_tokens, dtype=torch.long).to(device)
            inputs = batch_tokens[:, :-1]  # Get inputs
            targets = batch_tokens[:, 1:]  # Get targets
            inputs = inputs.to(device) # Ensure CUDA and GPU is used in creation of model
            targets = targets.to(device) # Ensure CUDA and GPU is used in creation of model
            logits, loss = model(inputs, targets)  # Forward pass
            (loss / grad_accum_steps).backward()  # Backward pass
            loss_accum += loss.item() / grad_accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # Optimizer step
        if (i + 1) % 500 == 0:  # Log every 500 iterations
            logger.info(f"Iteration {i+1}/{max_iters}, Loss: {loss.item():.4f}")  # Log loss
    logger.info("Training completed")  # Log training completion
    return model  # Return trained model

def resize_model(model, new_vocab_size, device):
    """Function to resize model for new vocabulary size."""
    logger.info(f"Resizing model from vocab size {model.vocab_size} to {new_vocab_size}")  # Log resizing
    new_model = GPTLanguageModel(vocab_size=new_vocab_size).to(device)  # Create new model
    new_state_dict = new_model.state_dict()  # Get new state dict
    old_state_dict = model.state_dict()  # Get old state dict

    # Copy non-embedding layers
    for key in old_state_dict:  # Loop through old state dict
        if key not in ['token_embedding.weight', 'head.weight', 'head.bias']:  # Skip embedding and head
            new_state_dict[key].copy_(old_state_dict[key])  # Copy parameter

    # Resize token embeddings
    old_embedding = old_state_dict['token_embedding.weight']  # Get old embedding
    new_embedding = torch.zeros((new_vocab_size, old_embedding.size(1)), device=device)  # Create new embedding
    copy_size = min(model.vocab_size, new_vocab_size)  # Determine copy size
    new_embedding[:copy_size] = old_embedding[:copy_size]  # Copy old embedding
    if new_vocab_size > model.vocab_size:  # If new size is larger
        new_embedding[copy_size:] = torch.randn(new_vocab_size - copy_size, old_embedding.size(1)) * 0.02  # Initialize new entries
    new_state_dict['token_embedding.weight'].copy_(new_embedding)  # Update new state dict

    # Resize output head weight
    old_head_weight = old_state_dict['head.weight']  # Get old head weight
    new_head_weight = torch.zeros((new_vocab_size, old_head_weight.size(1)), device=device)  # Create new head weight
    new_head_weight[:copy_size] = old_head_weight[:copy_size]  # Copy old head weight
    if new_vocab_size > model.vocab_size:  # If new size is larger
        new_head_weight[copy_size:] = torch.randn(new_vocab_size - copy_size, old_head_weight.size(1)) * 0.02  # Initialize new entries
    new_state_dict['head.weight'].copy_(new_head_weight)  # Update new state dict

    # Resize output head bias
    if 'head.bias' in old_state_dict:  # Check if head bias exists
        old_head_bias = old_state_dict['head.bias']  # Get old head bias
        new_head_bias = torch.zeros((new_vocab_size,), device=device)  # Create new head bias
        new_head_bias[:copy_size] = old_head_bias[:copy_size]  # Copy old head bias
        if new_vocab_size > model.vocab_size:  # If new size is larger
            new_head_bias[copy_size:] = torch.randn(new_vocab_size - copy_size) * 0.02  # Initialize new entries
        new_state_dict['head.bias'].copy_(new_head_bias)  # Update new state dict

    new_model.load_state_dict(new_state_dict)  # Load updated state dict into new model
    logger.info("Model resized")  # Log resizing completion
    return new_model  # Return resized model

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
    # FORCE INCLUDE MOLDING TERMS
    for term in molding_terms:
        clean_term = re.sub(r'[^a-z0-9 ]', '', term.lower())
        term_words = clean_term.split()
        tokens.update(term_words)
    vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + sorted(list(tokens))  # Create vocabulary list
    logger.info(f"Built vocabulary with {len(vocab)} words")  # Log vocabulary size
    return vocab  # Return vocabulary

def load_dataset_stream(dataset_name, config=None, split="train", max_samples=5000):
    """Stream a dataset from Hugging Face."""
    logger.info(f"Streaming {dataset_name} data...") # Log loading start
    try:
        dataset = load_dataset(dataset_name, config, split="train", streaming=True) # Open dataset
        count = 0
        for item in dataset: # Sets up to go through each item in dataset
            if count >= max_samples: # Stops the process once all of the samples have been processed
                break
            text = item.get('text') or " ".join(item.get('words', [])) # Collect item data without trailing whitespace characters
            if text and text.strip():
                yield text.strip() # Send text data to rest of code
            count += 1
    except Exception as e: # Send error if stream does not work
        logger.error(f"Failed to stream {dataset_name} data: {e}")

# RESOURCE LOGGING
def log_resources():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().used / 1e9
    gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    logger.info(f"Resources: CPU {cpu:.1f}%, RAM {mem:.2f}GB, GPU {gpu:.2f}GB")