from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
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


def get_embeddings_per_class(
        model: CLIPModel,
        processor: CLIPProcessor,
        class_map: Dict[str, List[Path]],
        device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Get embeddings of all the images in a class map

    Parameters
    ----------
    model
        A CLIPModel that will be doing the inference
    processor
        The associated CLIPProcessor that will preprocess the inputs
    class_map
        A dictionary of class name to a list of file paths for the class images
    device
        Inference device, default cuda


    Returns
    -------
    embeddings, labels
        ndarray [n_filesx512], ndarray [n_files]
        labels is the per embedding class index

    """
    embeddings = []
    labels = []
    with torch.no_grad():
        model.eval()
        for i, (cls, files) in enumerate(class_map.items()):
            images = [Image.open(p) for p in files]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            embedding = model.get_image_features(**inputs).cpu().numpy()
            embeddings.append(embedding)
            labels.append(len(files) * [i])
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    return embeddings, labels
