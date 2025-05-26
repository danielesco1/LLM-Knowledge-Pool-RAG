# This script needs a llama-parse key setup in the keys.py script to run.
from llama_parse import LlamaParse
import os
from config import *

# Parser parameters
parser = LlamaParse(
    api_key=LLAMAPARSE_API_KEY, 
    result_type="markdown",  # "markdown" or "text"
    num_workers=4,
    verbose=True,
    language="en",
)


def process_pdfs_in_directory(directory="knowledge_pool"):
    """Process all PDF files in directory and convert to text files."""
    
    for filename in os.listdir(directory):
        if not filename.endswith(".pdf"):
            continue
            
        file_path = os.path.join(directory, filename)
        extra_info = {"file_name": file_path}
        
        try:
            # Parse PDF
            with open(file_path, "rb") as pdf_file:
                documents = parser.load_data(pdf_file, extra_info=extra_info)
                print(f"Parsed {len(documents)} documents from {filename}")
            
            # Create output path
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(directory, output_filename)
            
            # Write to text file
            with open(output_path, 'w', encoding='utf-8', errors='replace') as txt_file:
                for doc in documents:
                    if doc.text.strip():  # Skip empty documents
                        txt_file.write(doc.text)
            
            print(f"Finished processing {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Run the function
process_pdfs_in_directory()

"""
for document in os.listdir("knowledge_pool"):
    #Iterate through the pdfs
    if document.endswith(".pdf"):
        file_name = os.path.join("knowledge_pool", document)

        # file_name = "knowledge_pool\LEED_v4.1_Residential_BD_C_Multifamily_Homes_RS_211025_clean.pdf"
        extra_info = {"file_name": file_name}

        with open(f"{file_name}", "rb") as f:
            # must provide extra_info with file_name key with passing file object
            documents = parser.load_data(f, extra_info=extra_info)

            print(f"Parsed {len(documents)} documents from {file_name}")
            # Iterate through the parsed documents
            # Save to a txt file
            basename = os.path.basename(file_name)
            print(f"Basename: {basename}")
            output_filename = os.path.splitext(basename)[0]
            output_path = os.path.join("knowledge_pool", f"{output_filename}.txt")

            with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
                for document in documents:
                    text = document.text
                    # If the text is empty, skip writing to file
                    if not text.strip():
                        continue
                    # Write the text to the file
                    f.write(text)
                
            print(f"Finished parsing {document}")
"""