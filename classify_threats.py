import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load the threat feed data
file_path = r'C:\Users\Sufi\Documents\60ThreatFeeds_latest.txt'
with open(file_path, 'r') as file:
    data = file.readlines()

# Parse the JSON data
threats = []
for line in data:
    try:
        threat = json.loads(line.strip())  # Remove any extra spaces/newlines
        threats.append(threat)
    except Exception as e:
        print(f"Error processing line: {line} | Error: {e}")

# Create a DataFrame
df = pd.DataFrame(threats)

# Handle missing data gracefully
df = df.fillna({col: {} for col in df.columns})

# Define a function to determine the type of threat
def determine_threat_type(row):
    if 'ip' in row and isinstance(row['ip'], dict) and 'v4' in row['ip']:
        return 'IP'
    elif 'url' in row and isinstance(row['url'], str):
        return 'URL'
    elif 'domain' in row and isinstance(row['domain'], str):
        return 'Domain'
    return 'Unknown'

# Extract relevant fields
df['type'] = df.apply(determine_threat_type, axis=1)

# Correctly extract the 'value' based on the type
df['value'] = df.apply(
    lambda x: x['ip']['v4'] if 'ip' in x and isinstance(x['ip'], dict) and 'v4' in x['ip']
    else x['url'] if 'url' in x and isinstance(x['url'], str)
    else x['domain'] if 'domain' in x and isinstance(x['domain'], str)
    else '', axis=1
)

df['tags'] = df['tags'].apply(lambda x: ', '.join(x['str']) if isinstance(x, dict) and 'str' in x else '')
df['description'] = df['description'].apply(lambda x: x if isinstance(x, str) else '')
df['score'] = df['score'].apply(lambda x: x['total'] if isinstance(x, dict) else 0)

# Prepare the data for supervised learning
X = df['description']
y = df['tags']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for text classification
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print classification report
print("Classification Report:\n", classification_report(y_test, predictions))
print("Accuracy Score:", accuracy_score(y_test, predictions))

# Classify the entire dataset
df['predicted_tags'] = model.predict(X)

# Filter the DataFrame for each type and score category
ip_above_50 = df[(df['type'] == 'IP') & (df['score'] >= 50)][['value', 'tags', 'description', 'score']]
ip_below_50 = df[(df['type'] == 'IP') & (df['score'] < 50)][['value', 'tags', 'description', 'score']]
url_above_50 = df[(df['type'] == 'URL') & (df['score'] >= 50)][['value', 'tags', 'description', 'score']]
url_below_50 = df[(df['type'] == 'URL') & (df['score'] < 50)][['value', 'tags', 'description', 'score']]
domain_above_50 = df[(df['type'] == 'Domain') & (df['score'] >= 50)][['value', 'tags', 'description', 'score']]
domain_below_50 = df[(df['type'] == 'Domain') & (df['score'] < 50)][['value', 'tags', 'description', 'score']]

# Define file paths
output_dir = r'C:\Users\Sufi\Documents\feeds'
file_paths = {
    'ip_above_50': f'{output_dir}/ip_above_50.json',
    'ip_below_50': f'{output_dir}/ip_below_50.json',
    'url_above_50': f'{output_dir}/url_above_50.json',
    'url_below_50': f'{output_dir}/url_below_50.json',
    'domain_above_50': f'{output_dir}/domain_above_50.json',
    'domain_below_50': f'{output_dir}/domain_below_50.json'
}

# Write each list to its respective JSON file
ip_above_50.to_json(file_paths['ip_above_50'], orient='records', indent=2)
ip_below_50.to_json(file_paths['ip_below_50'], orient='records', indent=2)
url_above_50.to_json(file_paths['url_above_50'], orient='records', indent=2)
url_below_50.to_json(file_paths['url_below_50'], orient='records', indent=2)
domain_above_50.to_json(file_paths['domain_above_50'], orient='records', indent=2)
domain_below_50.to_json(file_paths['domain_below_50'], orient='records', indent=2)

print(f"Data exported successfully to {output_dir}.")
