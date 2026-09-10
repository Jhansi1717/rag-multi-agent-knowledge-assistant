import os
import glob
import traceback

from ingestion import extract_document, clean_document, TextChunker, validate_provenance

def main():
    print("================================================================")
    print("PHASE 3.9: END-TO-END INGESTION VALIDATION")
    print("================================================================")
    
    files = []
    files.extend(glob.glob("data/hr/*.*"))
    files.extend(glob.glob("data/technical/*.*"))
    files = [f for f in files if os.path.isfile(f) and not f.endswith('.gitkeep')]
    
    chunker = TextChunker()
    
    passed = 0
    failed = 0
    total_chunks = 0
    
    print(f"Discovering supported files... Found {len(files)} files.\n")
    
    for file_path in files:
        print(f"--- Processing: {file_path} ---")
        try:
            # 1. Extraction
            doc = extract_document(file_path)
            
            # 2. Cleaning
            cleaned_doc = clean_document(doc)
            
            # 3. Chunking
            chunks = chunker.chunk_document(cleaned_doc)
            
            # 4. Metadata validation
            validate_provenance(chunks, cleaned_doc)
            
            print(f"Filename:               {cleaned_doc.filename}")
            print(f"Type:                   {cleaned_doc.file_type}")
            print(f"Document ID:            {cleaned_doc.document_id}")
            print(f"Extracted Text Length:  {len(cleaned_doc.text)} chars")
            print(f"Number of Chunks:       {len(chunks)}")
            if chunks:
                first_chunk = chunks[0].text
                preview = first_chunk[:100].replace("\n", " ") + ("..." if len(first_chunk) > 100 else "")
                print(f"First Chunk Preview:    {preview}")
            print(f"Validation Status:      PASSED")
            print("")
            passed += 1
            total_chunks += len(chunks)
        except Exception as e:
            print(f"Validation Status:      FAILED ({type(e).__name__}: {str(e)})")
            print("")
            failed += 1

    print("================================================================")
    print("AGGREGATE REPORT")
    print("================================================================")
    print(f"Files Processed: {len(files)}")
    print(f"Files Passed:    {passed}")
    print(f"Files Failed:    {failed}")
    print(f"Total Chunks:    {total_chunks}")
    
if __name__ == "__main__":
    main()
