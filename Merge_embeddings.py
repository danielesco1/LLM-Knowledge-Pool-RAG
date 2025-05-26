import json
import os

def merge_json_files(directory):
    merged_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json") and filename != "merged.json":
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding='utf-8') as file:
                data = json.load(file)
                # Add source filename to each entry
                for entry in data:
                    entry['source_file'] = filename
                merged_data.extend(data)
    return merged_data

merged_json = merge_json_files("knowledge_pool")

with open("knowledge_pool/merged.json", "w", encoding='utf-8') as output_file:
    json.dump(merged_json, output_file, indent=4, ensure_ascii=False)

print("Merged JSON saved to knowledge_pool/merged.json")