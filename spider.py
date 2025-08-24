import logging  # Import logging module for logging messages
import sys  # Import sys module for system-specific parameters and functions
import os  # Import os module for operating system dependent functionality
import scrapy  # Import scrapy module for web scraping
from scrapy.crawler import CrawlerProcess  # Import CrawlerProcess to run the spider
import re  # Import re module for regular expressions
import psutil  # Import psutil module for system monitoring

# Configure logging to record events both in a file and on the console
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define log format
    handlers=[  # Define handlers for logging
        logging.FileHandler('logs/spider.log', encoding='utf-8'),  # Log to file
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
    logger.info(f"Resources: CPU {cpu_percent:.1f}%, RAM {mem_used:.2f}GB, GPU 0.00GB")  # Log resources

class MoldingSpider(scrapy.Spider):
    name = 'molding_spider'  # Name of the spider
    start_urls = [  # List of URLs to scrape
        'https://plasticsguy.com/glossary-page/',
        'https://teampti.com/glossary-of-terms/',
        'https://www.midcontinentplastics.com/glossary.html'
    ]
    custom_settings = {  # Custom settings for the spider
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',  # User agent string
        'DOWNLOAD_DELAY': 2,  # Delay between requests
        'CONCURRENT_REQUESTS': 1,  # Number of concurrent requests
        'DOWNLOAD_TIMEOUT': 15,  # Timeout for downloads
        'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7',  # Fingerprint implementation
        'LOG_LEVEL': 'INFO'  # Log level for Scrapy
    }

    def parse(self, response):
        """Parse the response from a URL."""
        logger.info(f"Scraping {response.url}")  # Log the URL being scraped
        terms = []  # List to store terms
        keywords = [  # Keywords to filter terms
            'plastic', 'molding', 'injection', 'extrusion', 'thermoplastic',
            'tensile', 'polymer', 'resin', 'polycarbonate', 'polypropylene',
            'ABS', 'clamping', 'cycle time', 'shrinkage', 'die', 'melt flow'
        ]

        for element in response.css('p, li, td, dt, dd'):  # Select elements to extract text from
            text = element.css('::text').get(default='').strip().lower()  # Get text from element
            if text and any(keyword in text for keyword in keywords):  # Check if text contains keywords
                cleaned_text = re.sub(r'\s+', ' ', text).strip()  # Clean text
                if len(cleaned_text) > 10:  # Check length
                    terms.append(cleaned_text)  # Add to terms list

        for dl in response.css('dl'):  # Select description lists
            term = dl.css('dt::text').get(default='').strip().lower()  # Get term
            definition = dl.css('dd::text').get(default='').strip().lower()  # Get definition
            if term and definition and any(keyword in term + definition for keyword in keywords):  # Check keywords
                terms.append(f"{term}: {definition}")  # Add term: definition

        logger.info(f"Extracted {len(terms)} terms from {response.url}")  # Log number of terms
        if terms:  # If terms were extracted
            try:
                with open('molding_terms.txt', 'a', encoding='utf-8') as f:  # Open file in append mode
                    for term in terms:
                        f.write(term + '\n')  # Write term to file
                logger.info(f"Appended {len(terms)} terms to molding_terms.txt")  # Log append operation
            except Exception as e:
                logger.error(f"Error writing to molding_terms.txt: {e}")  # Log error if writing fails

if __name__ == "__main__":
    logger.info("Starting spider...")  # Log start of spider
    log_resources()  # Log initial resources
    if os.path.exists('molding_terms.txt'):  # Check if file exists
        os.remove('molding_terms.txt')  # Remove existing file
        logger.info("Cleared existing molding_terms.txt")  # Log file removal
    process = CrawlerProcess()  # Create CrawlerProcess instance
    process.crawl(MoldingSpider)  # Crawl with the spider
    process.start()  # Start the process
    log_resources()  # Log final resources
