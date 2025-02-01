import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    """
    Converts a CSV file to a JSON file.

    Args:
        csv_file_path (str): Path to the input CSV file.
        json_file_path (str): Path to the output JSON file.
    """
    try:
        # Read the CSV file
        with open("charging_bidding_dataset_with_winner.csv", mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            # Convert rows to a list of dictionaries
            data = [row for row in csv_reader]

        # Write the data to a JSON file
        with open(json_file_path, mode='w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)

        print(f"CSV data has been successfully converted to JSON and saved to {json_file_path}.")
    except FileNotFoundError:
        print("Error: The specified CSV file does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    csv_file = "input.csv"  # Replace with your CSV file path
    json_file = "output.json"  # Replace with your desired JSON file path
    csv_to_json(csv_file, json_file)
