from pathlib import Path
from typing import List, Dict, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def evaluate_prompt_set_for_classes(
    model: CLIPModel,
    processor: CLIPProcessor,
    class_map: Dict[str, List[Path]],
    prompts: List[str],
    device: str = "cuda",
) -> Tuple[List[int], List[int]]:
    """

    Parameters
    ----------
    model
        A CLIPModel that will be doing the inference
    processor
        The associated CLIPProcessor that will preprocess the inputs
    class_map
        A dictionary of class name to a list of file paths for the class images
    prompts
        A list of prompts, one per class
    device
        Inference device, default cuda

    Returns
    -------
    predictions, labels
        The predicted and actual class labels
    """
    preds = []
    labels = []
    with torch.no_grad():
        for i, (cls, files) in enumerate(class_map.items()):
            model.eval()
            images = [Image.open(p) for p in files]
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1).cpu().numpy()
            preds.extend(list(probs.argmax(1)))
            labels.extend(len(files) * [i])
    return preds, labels
