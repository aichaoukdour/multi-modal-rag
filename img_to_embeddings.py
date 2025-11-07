import torch
from PIL import Image
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from pathlib import Path
import numpy as np
import json

img_dir = Path("data/imgs")
pdf_img_paths = sorted(list(img_dir.glob("hai_ai*.jpg")))

model = ColIdefics3.from_pretrained(
        "vidore/colSmol-256M",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    ).eval()
processor = ColIdefics3Processor.from_pretrained("vidore/colSmol-256M")

# Batch processing parameters
batch_size = 4
image_embeddings = []
image_paths = []  # Track which embeddings correspond to which images

# Outer loop for batch processing
for i in range(0, len(pdf_img_paths), batch_size):
    print(f"Processing batch {1 + i//4} of {1 + len(pdf_img_paths) // batch_size}")
    batch_paths = pdf_img_paths[i:i + batch_size]
    batch_images = []

    # Load images for current batch
    for img_path in batch_paths:
        with Image.open(img_path) as img:
            batch_images.append(img.convert("RGB"))

    # Process current batch
    processed_batch = processor.process_images(batch_images).to(model.device)
    with torch.no_grad():
        batch_embeddings = model(**processed_batch)
        # Convert to numpy immediately and append
        for emb in batch_embeddings:
            image_embeddings.append(emb.cpu().to(torch.float32).numpy())
        image_paths.extend([str(path) for path in batch_paths])

# Convert list of numpy arrays to single array
embeddings_array = np.array(image_embeddings)

# Create output directory
output_dir = Path("data/embeddings")
output_dir.mkdir(exist_ok=True)

# Save embeddings as numpy array
np.save(output_dir / "image_embeddings.npy", embeddings_array)

# Save metadata as JSON
metadata = {
    "image_paths": image_paths,
    "embedding_shape": embeddings_array.shape,
    "model_name": "vidore/colSmol-256M",
    "batch_size": batch_size,
    "num_images": len(image_paths)
}
with open(output_dir / "embeddings_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved {len(image_embeddings)} embeddings with shape {embeddings_array.shape}")
print(f"Files saved to: {output_dir}")
print("- image_embeddings.npy (embeddings)")
print("- embeddings_metadata.json (metadata)")
