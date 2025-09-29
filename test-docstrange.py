import os
from docstrange import DocumentExtractor

# Initialize extractor (cloud mode by default)
extractor = DocumentExtractor(api_key="02f2e6a4-9539-11f0-8cc6-a275236cdc86")

# Directory paths
input_dir = "files"
output_dir = "nanonet-markdown"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of files in input directory
files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# Process each file
for file in files:
    input_path = os.path.join(input_dir, file)
    # Convert any document to clean markdown
    result = extractor.extract(input_path)
    markdown = result.extract_markdown()

    # Create output filename (replace extension with .md)
    base_name = os.path.splitext(file)[0]
    output_path = os.path.join(output_dir, f"{base_name}_ocr.md")

    # Save markdown to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f"Processed {file} -> {output_path}")