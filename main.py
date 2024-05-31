import json
from eda import EDA
from tqdm import tqdm

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(file_path, data, mode='a'):
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    count = 0
    input_file = 'metadata.jsonl'
    output_file = 'metadata_eda.jsonl'
    
    # Read samples from the input JSONL file
    samples = list(read_jsonl(input_file))
    if not samples:
        print("No data found in the input file.")
        return
    
    # Process each sample with a progress bar
    augmented_data = []
    for sample in tqdm(samples, desc="Processing samples"):
        count += 1
        if count == 3:
            break
        original_text = sample['text']
        
        # Generate augmented sentences using EDA
        augmented_sentences = EDA(original_text, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.5, p_rd=0.5, num_aug=3)
        
        # Prepare augmented data for each sample
        for aug_text in augmented_sentences:
            augmented_data.append({
                "file_name": sample['file_name'],
                "text": aug_text
            })
    
    # Write augmented data to the output JSONL file
    write_jsonl(output_file, augmented_data, mode='a')
    
    print(f"Augmented data has been written to {output_file}")

if __name__ == "__main__":
    main()
