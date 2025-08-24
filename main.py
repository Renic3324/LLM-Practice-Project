import subprocess  # Import the subprocess module to run external commands and manage processes
import logging  # Import the logging module to handle logging of messages for debugging and information
import sys  # Import the sys module to interact with system-specific parameters and functions
import os  # Import the os module to interact with the operating system, such as file paths and directory management
import torch  # Import the torch module for PyTorch, a deep learning framework used for GPU operations
import psutil  # Import the psutil module to monitor system resources like CPU, memory, and GPU usage
import signal  # Import the signal module to handle system signals like interrupts (e.g., Ctrl+C)
import time  # Import the time module to measure execution time and add delays if needed

# Configure logging to record pipeline events both in a file and on the console
# This setup ensures that all info-level messages and above are logged with timestamps and levels
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO, capturing informational messages and higher severity levels
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format including time, level, and message
    handlers=[  # Specify handlers for where logs are sent
        logging.FileHandler('logs/main.log', encoding='utf-8'),  # Log to a file named main.log in the logs directory using UTF-8 encoding
        logging.StreamHandler(sys.stdout)  # Also log to the standard output (console) for real-time viewing
    ]
)
logger = logging.getLogger(__name__)  # Create a logger instance for this module
for handler in logger.handlers:  # Loop through all handlers attached to the logger
    if isinstance(handler, logging.StreamHandler):  # Check if the handler is for streaming to console
        handler.setStream(sys.stdout)  # Set the stream to standard output
        handler.flush = sys.stdout.flush  # Ensure logs are flushed immediately to the console

def log_resources():
    """Function to log current system resources including CPU, RAM, and GPU usage."""
    cpu_percent = psutil.cpu_percent()  # Get the current CPU utilization as a percentage
    mem = psutil.virtual_memory()  # Get virtual memory statistics
    mem_used = mem.used / 1e9  # Convert used memory from bytes to gigabytes
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0  # Get allocated GPU memory in gigabytes if CUDA is available, else 0
    logger.info(f"Resources: CPU {cpu_percent:.1f}%, RAM {mem_used:.2f}GB, GPU {gpu_mem:.2f}GB")  # Log the resource usage in a formatted string

def cleanup():
    """Function to clear GPU memory if CUDA is available."""
    if torch.cuda.is_available():  # Check if CUDA (GPU support) is available
        torch.cuda.empty_cache()  # Clear the GPU memory cache to free up unused memory
    logger.info("Cleared GPU memory")  # Log that GPU memory has been cleared

def run_script(script_name):
    """Function to run a specified Python script and log its execution time and output."""
    logger.info(f"Running {script_name}...")  # Log that the script is starting to run
    start_time = time.time()  # Record the start time for measuring execution duration
    try:
        result = subprocess.run(  # Run the script using subprocess.run
            ['python', script_name],  # Command to execute: python followed by the script name
            check=True,  # Raise an exception if the script returns a non-zero exit code
            capture_output=True,  # Capture the standard output and error
            text=True  # Treat output as text (strings) rather than bytes
        )
        elapsed_time = time.time() - start_time  # Calculate the elapsed time
        logger.info(f"{script_name} completed in {elapsed_time:.2f}s")  # Log the completion time
        logger.debug(f"{script_name} stdout: {result.stdout}")  # Log the standard output at debug level
        logger.debug(f"{script_name} stderr: {result.stderr}")  # Log the standard error at debug level
        return True  # Return True if successful
    except subprocess.CalledProcessError as e:  # Catch errors if the script fails
        logger.error(f"Error running {script_name}: {e}")  # Log the error
        logger.error(f"Stdout: {e.stdout}")  # Log the captured stdout
        logger.error(f"Stderr: {e.stderr}")  # Log the captured stderr
        return False  # Return False on failure
    except Exception as e:  # Catch any other exceptions
        logger.error(f"Unexpected error running {script_name}: {e}")  # Log the unexpected error
        return False  # Return False on failure

def signal_handler(sig, frame):
    """Function to handle system signals like interrupts (e.g., Ctrl+C)."""
    logger.info("Pipeline interrupted. Cleaning up...")  # Log that the pipeline is being interrupted
    cleanup()  # Call the cleanup function to clear GPU memory
    sys.exit(1)  # Exit the program with exit code 1 indicating an error

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Set up signal handler for interrupt signals
    logger.info("Starting AI pipeline...")  # Log the start of the pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Determine if CUDA is available and set the device accordingly
    logger.info(f"Using device: {device}")  # Log the device being used
    log_resources()  # Log the initial resource usage

    scripts = ['spider.py', 'Training.py', 'Training2.py', 'Training3.py', 'Training4.py', 'Agent.py']  # List of scripts to run in order
    expected_outputs = [  # List of expected output files for each script
        'molding_terms.txt',  # Output for spider.py
        ['model.pkl', 'vocab.pkl'],  # Outputs for Training.py
        ['model_updated.pkl', 'vocab_updated.pkl'],  # Outputs for Training2.py
        ['model_updated_tableqa.pkl', 'vocab_tableqa.pkl'],  # Outputs for Training3.py
        ['model_updated_dolly.pkl', 'vocab_dolly.pkl'],  # Outputs for Training4.py
        'agent_responses.csv'  # Output for Agent.py
    ]

    for script, outputs in zip(scripts, expected_outputs):  # Loop through each script and its expected outputs
        if isinstance(outputs, list):  # If outputs are a list (multiple files)
            if all(os.path.exists(output) for output in outputs):  # Check if all output files exist
                logger.info(f"Outputs {outputs} exist, skipping {script}")  # Log skipping the script
                continue  # Skip to the next script
        elif os.path.exists(outputs):  # If a single output file exists
            logger.info(f"Output {outputs} exists, skipping {script}")  # Log skipping the script
            continue  # Skip to the next script
        if not run_script(script):  # Run the script and check if it succeeds
            logger.error(f"Pipeline stopped due to {script} failure")  # Log the failure
            cleanup()  # Clean up GPU memory
            sys.exit(1)  # Exit the program with error code 1
        if isinstance(outputs, list):  # Check if all output files were created
            if not all(os.path.exists(output) for output in outputs):
                logger.error(f"One or more outputs {outputs} not found after {script}")  # Log missing outputs
                cleanup()  # Clean up GPU memory
                sys.exit(1)  # Exit the program with error code 1
        elif not os.path.exists(outputs):  # Check if the single output file was created
            logger.error(f"Output {outputs} not found after {script}")  # Log missing output
            cleanup()  # Clean up GPU memory
            sys.exit(1)  # Exit the program with error code 1

    cleanup()  # Clean up GPU memory after all scripts complete
    logger.info("Pipeline completed successfully!")  # Log successful completion of the pipeline
