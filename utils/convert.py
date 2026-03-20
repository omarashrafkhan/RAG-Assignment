import json
import os

# Set up the exact paths
script_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(script_directory, "urdu_medical_rag_FINAL.json")
output_folder = os.path.join(script_directory, "urdu_health_corpus")

# Ensure the output folder exists (it will use the existing one, or create it if missing)
os.makedirs(output_folder, exist_ok=True)

print("Loading JSON dataset...")

# Load the JSON data
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        medical_articles = json.load(f)
except FileNotFoundError:
    print(f"❌ ERROR: Could not find '{json_file_path}'. Make sure the extraction script finished running!")
    exit()

print(f"Found {len(medical_articles)} articles. Generating text files...")

# Loop through each article and create a .txt file
for index, article in enumerate(medical_articles, start=1):
    title = article.get("title", "Unknown Title")
    category = article.get("category", "Unknown Category")
    text = article.get("text", "")
    
    # Generate the filename (e.g., wiki_health_001.txt, wiki_health_024.txt)
    filename = f"wiki_health_{index:03d}.txt"
    file_path = os.path.join(output_folder, filename)
    
    # Write the content in the exact requested format
    with open(file_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(f"{title}\n")
        txt_file.write(f"{category}\n")
        txt_file.write(f"{text}\n")
        
    print(f"Saved: {filename} -> {title}")

print(f"\n🎉 Success! All {len(medical_articles)} articles have been added to the '{output_folder}' folder.")