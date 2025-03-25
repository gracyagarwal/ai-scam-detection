import json

# File paths
file1_path = "C:\\Users\\Gracy\\OneDrive\\Desktop\\vit\\24-25winter\\project-2\\scam calls\\datasets\\fraud_call_cleaned.json"
file2_path = "C:\\Users\\Gracy\\OneDrive\\Desktop\\vit\\24-25winter\\project-2\\scam calls\\datasets\\robocall_fraud_dataset.json"

output_file = "merged_dataset.json"

# Load first JSON file
with open(file1_path, "r", encoding="utf-8") as f:
    data1 = json.load(f)

# Load second JSON file
with open(file2_path, "r", encoding="utf-8") as f:
    data2 = json.load(f)

# Merge both datasets
merged_data = data1 + data2  # Concatenating the lists

# Remove duplicates (optional)
seen = set()
cleaned_data = []
for entry in merged_data:
    text = entry["transcript"]  # Assuming 'transcript' is the message text
    if text not in seen:
        seen.add(text)
        cleaned_data.append(entry)

# Save merged JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=4)

print(f"✅ Merged dataset saved as {output_file}")
