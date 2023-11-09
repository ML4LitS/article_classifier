import csv
import pandas as pd

# Step 1: Process accession_data.csv
processed_data = []

with open('/home/stirunag/work/github/article_classifier/data/accession_data.csv', 'r') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        if row['accession_type'] != 'MED':
            if row['source'] == 'MED':
                processed_data.append({'accession_type': row['accession_type'], 'source': 'MED', 'id': row['extid']})
            elif row['source'] == 'PMC':
                if row['pmcid'].startswith('PMC'):
                    processed_data.append({'accession_type': row['accession_type'], 'source': 'PMC', 'id': row['pmcid']})

# Step 2: Process metagenomics_pmcids.txt
with open('/home/stirunag/work/github/article_classifier/data/metagenomics_pmcids.txt', 'r') as f:
    for line in f:
        id_ = line.strip()
        if id_.startswith('PMC'):
            processed_data.append({'accession_type': 'metagenomics', 'source': 'PMC', 'id': id_})
        elif id_.isdigit():
            processed_data.append({'accession_type': 'metagenomics', 'source': 'MED', 'id': id_})

# Step 3: Process protein_structures.txt
with open('/home/stirunag/work/github/article_classifier/data/protein_structures.txt', 'r') as f:
    for line in f:
        id_ = line.strip()
        if id_.startswith('PMC'):
            processed_data.append({'accession_type': 'protein_structures', 'source': 'PMC', 'id': id_})
        elif id_.isdigit():
            processed_data.append({'accession_type': 'protein_structures', 'source': 'MED', 'id': id_})

# Step 4: Write the final data to output.csv
with open('/home/stirunag/work/github/article_classifier/data/sources_output.csv', 'w', newline='') as csvfile:
    fieldnames = ['accession_type', 'source', 'id']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in processed_data:
        writer.writerow(row)

print("Data has been written to output.csv")


# Generate sample dataset
# Load the data
df = pd.read_csv("/home/stirunag/work/github/article_classifier/data/sources_output.csv")

# Group by accession_type and source, and sample 100 rows for each group
sampled_dfs = []
for (accession, source), group in df.groupby(['accession_type', 'source']):
    sampled_dfs.append(group.sample(n=min(100, len(group)), random_state=42))

# Concatenate the results
sampled_df = pd.concat(sampled_dfs, axis=0)

# Write to a new CSV
sampled_df.to_csv("/home/stirunag/work/github/article_classifier/data/sample_abstracts.csv", index=False)

print("Sampled data has been written to sample_abstracts.csv")
