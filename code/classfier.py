import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load spaCy model (assuming you're using 'en_core_web_md' for embeddings)
nlp = spacy.load("/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model")

# Read the CSV
data = pd.read_csv('../data/metagenomics_abstracts.csv')

# Drop rows with missing abstracts
data.dropna(subset=['abstract'], inplace=True)

# Split the data
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

def get_average_embeddings_batched(terms, model):
    embeddings = []
    for doc in tqdm(model.pipe(terms), total=len(terms), desc="Generating Embeddings"):
        valid_vectors = [token.vector for token in doc if token.has_vector and token.vector.shape[0] == 300]
        if len(valid_vectors) == 0:
            embeddings.append(np.zeros((300,)))
        else:
            average_embedding = np.mean(valid_vectors, axis=0)
            embeddings.append(average_embedding)
    return embeddings

# Get embeddings for train data
train_embeddings = get_average_embeddings_batched(train_data['abstract'], nlp)

# Average all the train embeddings to get class_embedding
class_embedding = np.mean(train_embeddings, axis=0)

# Get embeddings for test data
test_embeddings = get_average_embeddings_batched(test_data['abstract'], nlp)

# Check similarity and assign labels
labels = []
for embedding in tqdm(test_embeddings, desc="Calculating Similarities"):
    similarity = np.dot(embedding, class_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(class_embedding))
    labels.append(1 if similarity > 0.7 else 0)

# Assuming there's no actual ground truth for classification; creating a dummy ground truth of all ones
# If you have a ground truth, replace the line below
ground_truth = [1] * len(test_data)

# Calculate similarity scores for the test embeddings against class_embedding
similarity_scores = [np.dot(embedding, class_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(class_embedding))
                     for embedding in test_embeddings]

# Define the thresholds
thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

results = []

for threshold in thresholds:
    # Predicted labels based on threshold
    predicted_labels = [1 if score >= threshold else 0 for score in similarity_scores]

    # Calculate the number of misclassifications
    misclassifications = sum([1 for true, pred in zip(ground_truth, predicted_labels) if true != pred])

    # Compute the misclassification rate
    misclassification_rate = misclassifications / len(ground_truth)

    results.append((threshold, misclassification_rate))

# Print the results
for threshold, mr in results:
    print(f"At {threshold * 100:.0f}% similarity, Misclassification Rate: {mr:.2%}")


# # Get classification report
# report = classification_report(ground_truth, labels)
# print(report)
#
#
# # Calculate the number of misclassifications
# misclassifications = sum([1 for true, pred in zip(ground_truth, labels) if true != pred])
#
# # Compute the misclassification rate
# misclassification_rate = misclassifications / len(ground_truth)
#
# # Compute the accuracy
# accuracy = 1 - misclassification_rate
#
# print(f"Misclassification Rate: {misclassification_rate:.2%}")
# print(f"Accuracy: {accuracy:.2%}")