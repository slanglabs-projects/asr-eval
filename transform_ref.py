import json
import re

with open('payments_transcript__data_en.json', 'r') as f:
    data_list = json.load(f)

replacements = [
        ("ipo", "i p o"), 
        ("adity" , "aditi"), 
        ("qr", "q r"), 
        ("ankitha", "ankita"),
        ("Mom", "mummy"),
        ("sharda", "shraddha"),
        ("Dr", "doctor")
    ]

for data in data_list:
    new_references = []
    for item in data['references']:
        new_item = item
        new_references.append(new_item)
        for old, new in replacements:
            new_item = re.sub(rf'\b{old}\b', new, new_item, flags=re.IGNORECASE)
        if new_item not in new_references:
            new_references.append(new_item)
    data['references'] = new_references

with open('latest_payments_transcript_data_en.json', 'w') as f:
    json.dump(data_list, f, indent=4)
