from pathlib import Path
from typing import Optional
from typing import Union
import pandas as pd 

def read_csv(file_path:Path):
    """
    Read a CSV file and return a DataFrame.

    :param file_path: Path to the CSV file.
    :return df: DataFrame containing the CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def reader(extension: str) -> Optional[callable]:
    """
    Get the appropriate reader function based on the file extension.
    """
    readers = {
        ".csv": read_csv,
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
    