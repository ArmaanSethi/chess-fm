import os
from datasets import load_dataset

# Dataset Map
DATASETS = {
    "reasoning": "multimodal-reasoning-lab/chess",
    #"mate": "MATE-benchmark/MATE", # Hypothetical ID, need to verify if public
    "lichees_games": "Lichess/chess-games", # Example, need valid ID
    "chessgpt": "laion/strategic-game-chess", # Check validity
}

def download_sample(dataset_name, split="train", num_samples=100):
    print(f"Downloading sample from {dataset_name}...")
    try:
        # Load streaming to avoid full download
        ds = load_dataset(dataset_name, split=split, streaming=True)
        iterator = iter(ds)
        
        samples = []
        for _ in range(num_samples):
            try:
                samples.append(next(iterator))
            except StopIteration:
                break
                
        # Save to data/
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../data")
        os.makedirs(data_dir, exist_ok=True)
        
        safe_name = dataset_name.replace("/", "_")
        output_file = os.path.join(data_dir, f"{safe_name}_sample.txt")
        
        with open(output_file, "w") as f:
            for s in samples:
                f.write(str(s) + "\n")
                
        print(f"Saved {len(samples)} samples to {output_file}")
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")

if __name__ == "__main__":
    # Check if datasets is installed
    try:
        import datasets
    except ImportError:
        print("Please install 'datasets': pip install datasets")
        exit(1)
        
    # specific confirmed ones
    # Note: Using valid IDs found in search or educated guesses. 
    # Search said "multimodal-reasoning-lab/chess".
    
    targets = [
        "multimodal-reasoning-lab/chess",
        # "Isotonic/chess-puzzles", # Example high quality
    ]
    
    for t in targets:
        download_sample(t)
