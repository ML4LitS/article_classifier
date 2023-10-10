import requests
import csv
from tqdm import tqdm


# Function to read filenames.txt and return unique PMCIDs
def get_unique_pmids(filename):
    with open(filename, "r") as file:
        return list(set([line.strip() for line in file]))


# Function to yield chunks from a list
def chunked(data, chunk_size):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


url = "https://www.ebi.ac.uk/europepmc/webservices/rest/searchPOST"

pmids = get_unique_pmids("metagenomics_pmcids.txt")

with open("abstracts.csv", "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["PMCID", "abstract"])  # Header of CSV

    # Wrap the chunk iteration with tqdm for progress bar
    for chunk in tqdm(chunked(pmids, 5), desc="Processing PMCIDs"):
        # Create the query string using the chunked PMCID list
        query = " OR ".join([f"PMC:'{pmcid}'" for pmcid in chunk])

        query_params = {
            "query": query,
            "resultType": "core",
            "pageSize": 10,
            "format": "json"
        }

        response = requests.post(url, data=query_params)
        response.raise_for_status()  # Check for any errors in the response

        # Process the response data
        data = response.json()

        # Write to CSV
        for article in data["resultList"]["result"]:
            abstract = article.get("abstractText", "")
            csv_writer.writerow([article["id"], abstract])
