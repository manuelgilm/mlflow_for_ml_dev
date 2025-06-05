from pathlib import Path
from typing import Optional
from typing import Union
import pandas as pd
from typing import Optional
from typing import Dict
from typing import Any
import yaml
import pkgutil


def get_root_dir() -> Path:
    """
    Get the root directory of the project.

    :return root_dir: Path to the root directory.
    """
    return Path(__file__).parents[2]


def read_csv(file_path: Path):
    """
    Read a CSV file and return a DataFrame.

    :param file_path: Path to the CSV file.
    :return df: DataFrame containing the CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def read_yaml(file_path: Path) -> Dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.

    :param file_path: Path to the YAML file.
    :return data: Dictionary containing the YAML file contents.
    """
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def reader(extension: str) -> Optional[callable]:
    """
    Get the appropriate reader function based on the file extension.

    :param extension: File extension (e.g., '.csv').
    :return reader_func: Function to read the file, or None if no reader is found.
    """
    readers = {
        ".csv": read_csv,
        ".yaml": read_yaml,
        ".yml": read_yaml,
    }

    return readers.get(extension, None)


def read_file(path: Union[str, Path]) -> str:
    """
    Read a file and return its contents as a string.

    :param path: Path to the file.
    :return contents: Contents of the file.
    """
    if not isinstance(path, Path):
        path = Path(path)

    extension = path.suffix

    reader_func = reader(extension)

    if reader_func:
        return reader_func(path)
