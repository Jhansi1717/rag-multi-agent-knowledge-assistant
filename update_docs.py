import os
import glob

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace specific strings
    content = content.replace("~500 characters per chunk, no overlap", "Token-aware, ~600 tokens per chunk with 80 tokens overlap (baseline experiment)")
    content = content.replace("~500 characters (fixed)", "600 tokens (baseline) with 80 tokens overlap")
    content = content.replace("fixed-size, ~500 chars", "token-aware, 600/80 token baseline")
    content = content.replace("fixed-size chunks of approximately 500 characters", "token-aware chunks of approximately 600 tokens with 80 tokens overlap (baseline experiment)")
    content = content.replace("(≈500 chars)", "(≈600 tokens)")
    content = content.replace("Fixed-Size Chunking (~500 chars)", "Token-Aware Chunking (600/80 tokens baseline)")
    content = content.replace("fixed-size chunking of approximately 500 characters", "token-aware chunking of approximately 600 tokens (80 token overlap)")
    content = content.replace("~500 characters", "600 tokens (baseline)")
    content = content.replace("500-char fixed-size", "600-token baseline")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    files = glob.glob('**/*.md', recursive=True)
    for file in files:
        update_file(file)
    print("Documentation updated.")

if __name__ == "__main__":
    main()
