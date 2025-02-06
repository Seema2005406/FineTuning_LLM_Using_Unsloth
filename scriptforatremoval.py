import json

# Input and output file names
input_file = "carB.jsonl"
output_file = "carB_new.jsonl"

# Open the input file and process each line
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)  # Parse JSON
        
        # Remove '@' from all string values in the JSON object
        def remove_at_symbols(obj):
            if isinstance(obj, str):
                return obj.replace("@", "")
            elif isinstance(obj, list):
                return [remove_at_symbols(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: remove_at_symbols(value) for key, value in obj.items()}
            return obj
        
        cleaned_data = remove_at_symbols(data)
        
        # Write cleaned JSON object back to the output file
        outfile.write(json.dumps(cleaned_data) + "\n")

print(f"Processing complete. Cleaned data saved to {output_file}")
