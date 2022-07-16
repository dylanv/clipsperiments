import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

project_dir = Path(__file__).parent.parent


def get_yoga_dataset() -> Dict[str, List[Path]]:
    """Load the yoga pose dataset from kaggle

    https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification

    Returns
    -------
    class_map
        A dictionary mapping class name to a list of class image paths

    """
    data_dir = project_dir / "data" / "yoga"
    classes = ["Downdog", "Warrior2", "Tree", "Plank", "Goddess"]
    extension = "jpg"

    print(f"Checking for yoga data in {data_dir}.")
    # Download the kaggle dataset
    if not data_dir.exists():
        try:
            from kaggle import KaggleApi

            api = KaggleApi()
            api.authenticate()
        except OSError as e:
            print(f"You need to setup up your kaggle api keys.")
            raise e
        api.dataset_download_files("ujjwalchowdhury/yoga-pose-classification", path=data_dir, quiet=False, unzip=True)
    # Move the class folders out of the downloaded folder and delete
    download_dir = data_dir / "YogaPoses"
    if download_dir.exists():
        for folder in [f for f in download_dir.glob("*") if f.is_dir()]:
            shutil.move(folder, data_dir / folder.name)
        shutil.rmtree(download_dir, ignore_errors=True)
    # Check the data and build our class map
    class_map = {}
    for c in classes:
        if not (data_dir / c).exists():
            raise ValueError(f"Oops. Something went wrong. Missing class {c}")
        cls_files = list((data_dir / c).glob(f"*.{extension}"))
        print(f"Got {len(cls_files)} files for class {c}.")
        class_map[c] = cls_files

    return class_map


def get_intel_scene_dataset(split: str = "train") -> Dict[str, List[Path]]:
    """Load the Intel scene dataset from kaggle

    https://www.kaggle.com/datasets/puneet6060/intel-image-classification
    Originally from a intel sponsored hackathon on https://datahack.analyticsvidhya.com/

    Returns
    -------
    class_map
        A dictionary mapping class name to a list of class image paths

    """
    data_dir = project_dir / "data" / "intel"
    classes = ["buildings", "sea", "street", "mountain", "glacier", "forest"]
    extension = "jpg"

    print(f"Checking for scene data in {data_dir}.")
    # Download the kaggle dataset
    if not data_dir.exists():
        try:
            from kaggle import KaggleApi

            api = KaggleApi()
            api.authenticate()
        except OSError as e:
            print(f"You need to setup up your kaggle api keys.")
            raise e
        api.dataset_download_files("puneet6060/intel-image-classification", path=data_dir, quiet=False, unzip=True)
    # Check the data and build our class map
    class_map = {}
    folder = f"seg_{split}"
    split_folder = data_dir / folder / folder
    for c in classes:
        class_folder = split_folder / c
        if not class_folder.exists():
            raise ValueError(f"Oops. Something went wrong. Missing class {c}")
        cls_files = list(class_folder.glob(f"*.{extension}"))
        print(f"Got {len(cls_files)} files for class {c}.")
        class_map[c] = cls_files

    return class_map


def get_fruits_dataset(split: str = "train") -> Dict[str, List[Path]]:
    """Load the fruits dataset from kaggle

    https://www.kaggle.com/datasets/moltean/fruits

    Returns
    -------
    class_map
        A dictionary mapping class name to a list of class image paths

    """
    data_dir = project_dir / "data" / "fruits"
    classes = [
        "cucumber",
        "zucchini",
        "apple_red_delicios",
        "apple_braeburn",
        "pear",
        "carrot",
        "cabbage_white",
        "apple_granny_smith",
        "apple_golden",
        "apple_crimson_snow",
        "apple",
        "zucchini_dark",
        "apple_red_yellow",
        "eggplant_violet",
        "apple_red",
        "apple_pink_lady",
        "apple_hit",
        "apple_rotten",
    ]
    extension = "jpg"

    print(f"Checking for scene data in {data_dir}.")
    # Download the kaggle dataset
    if not data_dir.exists():
        try:
            from kaggle import KaggleApi

            api = KaggleApi()
            api.authenticate()
        except OSError as e:
            print(f"You need to setup up your kaggle api keys.")
            raise e
        api.dataset_download_files("moltean/fruits", path=data_dir, quiet=False, unzip=True, force=True)
        # Merge fruits and clean up structure
        fruit_root_folder = data_dir / "fruits-360-original-size" / "fruits-360-original-size"
        for folder in ["Training", "Validation", "Test"]:
            for sub_folder in (fruit_root_folder / folder).glob("*"):
                cls = "_".join(sub_folder.stem.split("_")[:-1])
                print(f"Moving {sub_folder.stem} to {cls}")
                os.makedirs(data_dir / folder / cls, exist_ok=True)
                shutil.move(sub_folder, data_dir / folder / cls / sub_folder.stem)
    # Check the data and build our class map
    class_map = {}
    split_folder = data_dir / split
    for c in sorted(classes):
        class_folder = split_folder / c
        if not class_folder.exists():
            raise ValueError(f"Oops. Something went wrong. Missing class {c}")
        # Need to recursive search now because we have subfolders
        cls_files = list(class_folder.glob(f"**/*.{extension}"))
        print(f"Got {len(cls_files)} files for class {c}.")
        class_map[c] = cls_files

    return class_map
