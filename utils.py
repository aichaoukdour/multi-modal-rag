import os
import torch
from PIL import Image
from colpali_engine.models import ColIdefics3, ColIdefics3Processor


def load_model():
    """
    Load the colSmol-256M model and processor from Hugging Face.
    Uses a local cache directory to avoid re-downloading.
    """
    # ✅ Optional: silence TensorFlow CUDA noise
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # ✅ Use a local cache directory for model weights
    cache_dir = "./hf_model_cache"

    # ✅ Load model & processor from local (downloaded once)
    model = ColIdefics3.from_pretrained(
        "vidore/colSmol-256M",
        cache_dir=cache_dir,
        torch_dtype=torch.float32,  # use float32 for CPU
        device_map="cpu",           # explicitly use CPU
    ).eval()

    processor = ColIdefics3Processor.from_pretrained(
        "vidore/colSmol-256M",
        cache_dir=cache_dir
    )

    # ✅ Optional: improve CPU speed via token pooling + lower resolution
    processor.image_processor.token_pooling = True
    processor.image_processor.size = {"longest_edge": 224}
    return processor, model


def embed_text(text: str) -> list[list[float]]:
    processor, model = load_model()
    batch_queries = processor.process_queries([text]).to("cpu")

    with torch.no_grad():
        query_embeddings = model(**batch_queries)

    query_embedding = query_embeddings[0].to("cpu").tolist()
    return query_embedding
