# Import packages
import sys
import os
import tiktoken

from openai import OpenAI

# Add the project directory to sys.path to ensure function imports
project_dir = os.getcwd() # Get the current working directory (project root)
if project_dir not in sys.path: 
    sys.path.append(project_dir)

# Import functions
from data_extract.extract import fetch_pubmed_articles


# Define parameters
openai = OpenAI()
query = "artificial intelligence AND Health technology Assessment"
retmax = 10  # Maximum number of articles to retrieve.
email = "bjboverhof@gmail.com"

# Fetch articles and process results
articles = fetch_pubmed_articles(query=query, email=email, retmax=retmax)
articles_keys = list(articles.keys())  # Extract PMIDs from the results.
len(articles_keys)  # Count the number of retrieved articles.



# Choose the model's tokenizer
model = "gpt-3.5-turbo"  # Replace with your model
encoding = tiktoken.encoding_for_model(model)

# Your text input
text = "Hello, this is an example to count tokens."

# Count tokens
tokens = encoding.encode(text)
token_count = len(tokens)

print(f"Number of tokens: {token_count}")

