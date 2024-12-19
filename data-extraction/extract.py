"""
This script interacts with the PubMed database to retrieve and process scientific articles based on a specified search query. 
It uses the Biopython library to perform the search, fetch detailed article information in MEDLINE format, 
and extract key details such as the title, abstract, DOI, and authors. The results are stored in a dictionary for further analysis.
"""

from Bio import Entrez, Medline  # Importing necessary modules for PubMed access and parsing.

def fetch_pubmed_articles(query, email, retmax=10):
    """
    Searches PubMed for articles matching the given query and fetches detailed information.
    
    Parameters:
        query (str): The search query string to find relevant articles.
        email (str): The user's email address, required by NCBI for API access.
        retmax (int): The maximum number of articles to retrieve (default is 10).
    
    Returns:
        dict: A dictionary where each key is a PubMed ID (PMID) and the value is another dictionary 
              containing details about the corresponding article (title, abstract, DOI, authors).
    """
    Entrez.email = email  # Set email for NCBI API access.

    # Search PubMed with the query and retrieve up to `retmax` results.
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    search_record = Entrez.read(handle)  # Parse search results.
    handle.close()

    # Get the list of PubMed IDs from the search results.
    pubmed_ids = search_record["IdList"]

    # Fetch detailed article records in MEDLINE format for the retrieved IDs.
    handle = Entrez.efetch(db="pubmed", id=pubmed_ids, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))  # Parse MEDLINE records into a list of dictionaries.
    handle.close()

    # Create a dictionary with PMIDs as keys and selected article details as values.
    articles_dict = {
        record["PMID"]: {
            "PMID": record["PMID"],
            "Title": record.get("TI", ""),  # Title of the article.
            "Abstract": record.get("AB", ""),  # Abstract text.
            "DOI": record.get("LID", ""),  # Digital Object Identifier.
            "Authors": record.get("AU", [])  # List of authors.
        }
        for record in records if "PMID" in record  # Ensure 'PMID' is present in the record.
    }

    return articles_dict

# Query parameters
query = "artificial intelligence AND Health technology Assessment"
retmax = 10000  # Maximum number of articles to retrieve.
email = "bjboverhof@gmail.com"

# Fetch articles and process results
articles = fetch_pubmed_articles(query=query, email=email, retmax=retmax)
articles_keys = list(articles.keys())  # Extract PMIDs from the results.
len(articles_keys)  # Count the number of retrieved articles.

# Print the abstract of the second article (index 1).
articles[articles_keys[1]]["Abstract"]
