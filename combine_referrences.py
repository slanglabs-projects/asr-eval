import csv
import json

references = []

# Open the CSV file
with open('reference_audio_transcripts.csv', 'r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)
    
    # Skip the header row
    next(csv_reader)
    
    # Open the JSON file
    with open('payments_reference_data.json', 'r') as json_file:
        # Load the JSON data
        json_data = json.load(json_file)
        
        # Loop over both datasets simultaneously using zip()
        for csv_row, json_row in zip(csv_reader, json_data):
            # Access data from the current row of each dataset
            referrence = {
                    "references": [],
                    "file_name": None,
                    "language": None
                }

            referrence["references"] = [csv_row[1], json_row["text"]]
            referrence["file_name"] = csv_row[0]
            referrence["language"] = csv_row[-1]
            references.append(referrence)

with open('payments_transcript__data_en.json', 'w') as json_file:
    # Write the list to the JSON file
    json.dump(references, json_file, indent=4)

