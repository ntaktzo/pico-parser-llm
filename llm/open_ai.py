import tiktoken 
import openai
import json
from typing import Dict, Any
from dotenv import load_dotenv
import os

# Define a function to count tokens in a given text for a specified model.
def count_tokens(text, model="gpt-4o-mini"):
    # Get the encoding object for the specified model. This object determines how text is tokenized.
    encoding = tiktoken.encoding_for_model(model)
    
    # Use the encoding object to tokenize the input text. This converts the text into a list of token IDs.
    tokens = encoding.encode(text)

    print(f"Number of tokens: {len(tokens)}")

    # Return the number of tokens in the text.
    return len(tokens)



def validate_api_key(dotenv_path='.env'):
    """
    Validates the presence and format of the OPENAI_API_KEY from the .env file.

    Parameters:
        dotenv_path (str): Path to the .env file. Defaults to '.env'.

    Returns:
        str: Message indicating the status of the API key.
    """
    # Load the .env file
    load_dotenv(dotenv_path=dotenv_path, override=True)

    # Check if the API key is loaded
    api_key = os.getenv('OPENAI_API_KEY')

    # Validate the API key
    if not api_key:
        return "No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!"
    elif not api_key.startswith("sk-proj-"):
        return "An API key was found, but it doesn't start with 'sk-proj-'; please check you're using the right key."
    elif api_key.strip() != api_key:
        return "An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them."
    else:
        return "API key found and looks good so far!"
    


def create_batched_prompts(articles: dict, batch_size: int) -> list:
    """
    Given a dictionary of articles, produce a list of batched prompts suitable for the OpenAI API.
    
    Each prompt in the list contains up to `batch_size` articles.
    
    Returns:
        List of prompts (strings).
    """
    batched_prompts = []
    batch = []
    delimiter = "\n\n---\n\n"  # Unique delimiter unlikely to appear in abstracts

    for pmid, data in articles.items():
        title = data.get("Title", "").replace('\n', ' ').strip()
        abstract = data.get("Abstract", "").replace('\n', ' ').strip()
        article_text = f"PMID: {pmid}\nTitle: {title}\nAbstract:\n{abstract}"
        batch.append(article_text)
        
        if len(batch) == batch_size:
            batched_prompt = delimiter.join(batch)
            batched_prompts.append(batched_prompt)
            batch = []
    
    # Add any remaining articles as the last batch
    if batch:
        batched_prompt = delimiter.join(batch)
        batched_prompts.append(batched_prompt)
    
    return batched_prompts



def query_gpt_api_batched(batched_prompts: list, system_prompt: str, model: str) -> Dict[str, Any]:
    """
    Query the OpenAI API with batched prompts and parse the responses.
    
    Args:
        batched_prompts: List of batched prompt strings.
        system_prompt: The system-level instructions for the model.
        model: The OpenAI model to use.
    
    Returns:
        Dictionary mapping PMID to their respective PICO extraction.
    """
    results = {}
    
    for batch_num, user_prompt in enumerate(batched_prompts, start=1):
        print(f"Processing Batch {batch_num}")
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=3000,  # Adjust based on expected response size
                temperature=0.0,
                n=1,  # Number of responses to generate
                stop=None  # Define stop sequences if needed
            )
            
            assistant_reply = response.choices[0].message.content.strip()
            
            # Attempt to parse the JSON response
            batch_results = json.loads(assistant_reply)
            
            # Validate the structure of the parsed JSON
            if isinstance(batch_results, dict):
                results.update(batch_results)
            else:
                print(f"Unexpected response format in batch {batch_num}. Response was not a JSON object.")
                print("Assistant response was:")
                print(assistant_reply)
        
        except json.JSONDecodeError as e:
            print(f"JSON decode error for batch {batch_num}: {e}")
            print("Assistant response was:")
            print(assistant_reply)

    
    return results