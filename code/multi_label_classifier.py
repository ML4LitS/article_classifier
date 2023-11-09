import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load spaCy model
nlp = spacy.load("/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model")

# Read the CSV
data = pd.read_csv('../data/data_abstracts.csv')

# Drop rows with missing abstracts
data.dropna(subset=['abstract'], inplace=True)

# Split the data
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# def get_average_embeddings_batched(terms, model):
#     embeddings = []
#     for doc in tqdm(model.pipe(terms), total=len(terms), desc="Generating Embeddings"):
#         valid_vectors = [token.vector for token in doc if token.has_vector]
#         average_embedding = np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros((300,))
#         embeddings.append(average_embedding)
#     return embeddings

def get_average_embeddings_batched(terms, model, accession_type_name=""):
    embeddings = []
    for doc in tqdm(model.pipe(terms), total=len(terms), desc=f"Generating Embeddings for {accession_type_name}"):
        valid_vectors = [token.vector for token in doc if token.has_vector]
        average_embedding = np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros((300,))
        embeddings.append(average_embedding)
    return embeddings

# Compute average embeddings for each accession_type in the training data
accession_embeddings = {}
for accession_type, group in train_data.groupby('accession_type'):
    embeddings = get_average_embeddings_batched(group['abstract'], nlp, accession_type_name=accession_type)
    accession_embeddings[accession_type] = np.mean(embeddings, axis=0)

# For each test abstract, calculate its similarity with each accession_type embedding and assign labels
predicted_labels = []
for abstract in tqdm(test_data['abstract'], desc="Classifying Abstracts"):
    embedding = get_average_embeddings_batched([abstract], nlp)[0]
    labels = []
    for accession_type, acc_embedding in accession_embeddings.items():
        similarity = np.dot(embedding, acc_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(acc_embedding))
        if similarity > 0.8:  # 70% similarity threshold
            labels.append(accession_type)
    predicted_labels.append(labels)


# Converting multi-label format for classification report
# Assuming each row in test_data has a column 'labels' which is a list of accession_types for that row
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y_true = mlb.fit_transform(test_data['accession_type'].apply(lambda x: [x]))
y_pred = mlb.transform(predicted_labels)

# Classification report
report = classification_report(y_true, y_pred, target_names=mlb.classes_)
print(report)

df_output = pd.DataFrame({
    'true_multi_label': test_data['accession_type'].tolist(),
    'predicted_multi_label': [";".join(labels) for labels in predicted_labels],
})

# Save the DataFrame to CSV
df_output.to_csv('classification_results.csv', index=False)


y_true_binary = [1 if true_label in pred_label else 0 for true_label, pred_label in zip(df_output['true_multi_label'], df_output['predicted_multi_label'])]

# As it's a binary classification, our predictions are essentially the same as y_true_binary
y_pred_binary = y_true_binary

# Generate classification report
report = classification_report(y_true_binary, y_pred_binary)
print(report)
