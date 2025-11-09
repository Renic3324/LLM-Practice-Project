import subprocess  # Run external Python scripts as subprocesses
import logging     # Structured logging for debugging and monitoring
import sys         # System-specific parameters (e.g., stdout, exit)
import os          # File/directory operations (check if files exist)
import torch       # PyTorch: GPU detection and memory management
import psutil      # System monitoring: CPU, RAM usage
import signal      # Handle OS signals (e.g., Ctrl+C)
import time        # Measure execution time of each script

# =============================================================================
# 0. GLOBAL VARIABLES
# =============================================================================
all_sentences = []


# =============================================================================
# 1. LOGGING CONFIGURATION
# =============================================================================
# Log to console with timestamps, level, and message.
# Ensures real-time visibility during long training runs.
logging.basicConfig(
    level=logging.INFO,  # Capture INFO and higher (WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format: "2025-04-05 10:23:11,123 - INFO - Starting..."
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output logs to terminal
    ]
)
logger = logging.getLogger(__name__)  # Logger for this module

# Ensure immediate flush to console (critical for long-running scripts)
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setStream(sys.stdout)
        handler.flush = sys.stdout.flush

# =============================================================================
# 2. RESOURCE MONITORING
# =============================================================================
def log_resources():
    """Log current CPU, RAM, and GPU memory usage."""
    cpu_percent = psutil.cpu_percent(interval=1)  # % CPU usage over 1-second sample
    mem = psutil.virtual_memory()                 # Total system memory stats
    mem_used_gb = mem.used / 1e9                  # Convert bytes → GB
    gpu_mem_gb = (torch.cuda.memory_allocated() / 1e9) if torch.cuda.is_available() else 0  # Allocated GPU VRAM
    logger.info(f"Resources: CPU {cpu_percent:.1f}%, RAM {mem_used_gb:.2f}GB, GPU {mem_used_gb:.2f}GB")

# =============================================================================
# 3. GPU MEMORY CLEANUP
# =============================================================================
def cleanup():
    """Free GPU memory and log cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release all unused cached memory
        torch.cuda.synchronize() # Ensure all GPU ops are complete
    logger.info("GPU memory cleared")

# =============================================================================
# 4. RUN EXTERNAL SCRIPT
# =============================================================================
def run_script(script_name):
    """
    Execute a Python script via subprocess.
    Captures stdout/stderr, measures time, logs success/failure.
    """
    logger.info(f"Executing: {script_name}")
    start_time = time.time()  # Start timer

    try:
        # Run script: ['python', 'training.py']
        result = subprocess.run(
            ['python', script_name],
            check=True,           # Raise exception on non-zero exit code
            capture_output=True,  # Capture stdout and stderr
            text=True             # Return as string (not bytes)
        )
        elapsed = time.time() - start_time
        logger.info(f"{script_name} completed in {elapsed:.2f}s")
        logger.debug(f"{script_name} stdout:\n{result.stdout}")
        logger.debug(f"{script_name} stderr:\n{result.stderr}")
        return True

    except subprocess.CalledProcessError as e:
        # Script failed (non-zero exit)
        elapsed = time.time() - start_time
        logger.error(f"{script_name} failed after {elapsed:.2f}s")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"stdout:\n{e.stdout}")
        logger.error(f"stderr:\n{e.stderr}")
        return False

    except FileNotFoundError:
        logger.error(f"Script not found: {script_name}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error running {script_name}: {e}")
        return False

# =============================================================================
# 5. SIGNAL HANDLER (Ctrl+C)
# =============================================================================
def signal_handler(sig, frame):
    """Handle interruption (Ctrl+C) gracefully."""
    logger.info("Pipeline interrupted by user (Ctrl+C)")
    cleanup()           # Free GPU memory
    sys.exit(1)         # Exit with error code

# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================
if __name__ == "__main__":
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("AI Molding Data Pipeline Starting...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    log_resources()  # Log initial system state

    # =============================================================================
    # 7. PIPELINE STAGES: Script + Expected Output(s)
    # =============================================================================
    pipeline = [
        # (script, [expected_output_files])
        ('training.py',     ['model.pkl']),
        ('training2.py',    ['model_updated.pkl']),
        ('training3.py',    ['model_updated2.pkl']),
        ('training4.py',    ['model_final.pkl']),
        ('agent.py',        'Imported_Window.xlsx')                   # ← agent.py saves Excel
    ]

    # =============================================================================
    # 8. EXECUTE PIPELINE WITH SKIP LOGIC
    # =============================================================================
    for script_name, expected_outputs in pipeline:
        # Convert single string to list for uniform handling
        outputs_list = expected_outputs if isinstance(expected_outputs, list) else [expected_outputs]

        # Skip if all outputs already exist
        if all(os.path.exists(out) for out in outputs_list):
            logger.info(f"Outputs {outputs_list} exist → skipping {script_name}")
            continue

        # Run the script
        if not run_script(script_name):
            logger.error(f"Pipeline halted due to failure in {script_name}")
            cleanup()
            sys.exit(1)

        # Verify outputs were created
        missing = [out for out in outputs_list if not os.path.exists(out)]
        if missing:
            logger.error(f"Missing outputs after {script_name}: {missing}")
            cleanup()
            sys.exit(1)
        else:
            logger.info(f"Outputs created: {outputs_list}")

    # =============================================================================
    # 9. FINAL CLEANUP & SUCCESS
    # =============================================================================
    cleanup()
    log_resources()  # Log final resource usage
    logger.info("AI PIPELINE COMPLETED SUCCESSFULLY!")
    print("\n" + "="*60)
    print("   MOLDING DATA EXTRACTED TO: Imported_Window.xlsx")
    print("   OPEN THE FILE TO VIEW RESULTS!")
    print("="*60 + "\n")