import json

# Load JSON file
with open('dataset.json', 'r') as f:
    data = json.load(f)

# Save as JSONL
with open('datasetfinal.jsonl', 'w') as f:
    for entry in data:
        f.write(json.dumps(entry) + '\n')
