import os
import glob
from ingestion import extract_document, clean_document, TextChunker, validate_provenance

def assess_corpus():
    files = glob.glob('data/hr/*') + glob.glob('data/technical/*')
    chunker = TextChunker()
    
    print("================================================================")
    print("PHASE 3.8: MILESTONE 1 KNOWLEDGE BASE INGESTION ASSESSMENT")
    print("================================================================")
    
    for file_path in files:
        if not os.path.isfile(file_path):
            continue
            
        print(f"\nProcessing File: {file_path}")
        
        # 1. Extract
        try:
            doc = extract_document(file_path)
            # 2. Clean
            cleaned_doc = clean_document(doc)
            # 3. Chunk
            chunks = chunker.chunk_document(cleaned_doc)
            # 4. Validate Provenance
            validate_provenance(chunks, cleaned_doc)
            
            # Print Summary
            print(f"  - Filename:        {cleaned_doc.filename}")
            print(f"  - Type:            {cleaned_doc.file_type}")
            print(f"  - Document ID:     {cleaned_doc.document_id}")
            print(f"  - Text Length:     {len(cleaned_doc.text)} chars")
            print(f"  - Chunk Count:     {len(chunks)}")
            print(f"  - Metadata:")
            for k, v in cleaned_doc.metadata.items():
                print(f"      {k}: {v}")
            print(f"  - Validation:      PASSED (Provenance intact for all chunks)")
            
        except Exception as e:
            print(f"  - Validation:      FAILED ({str(e)})")

if __name__ == "__main__":
    assess_corpus()
