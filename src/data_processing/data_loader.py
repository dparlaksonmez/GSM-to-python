import pandas as pd

def load_input_file(input_file_name, *, separator: str = ","):
    """
    Load the input file from the data folder.

    Tries UTF-8 first, then falls back to latin-1 if a UnicodeDecodeError
    is raised (some GEO datasets contain non-ASCII probe annotations).

    Args:
        input_file_name (str): The name of the input file.
        separator (str): Column separator used in the input file.

    Returns:
        pd.DataFrame: The input file data.
    """
    try:
        data = pd.read_csv(input_file_name, sep=separator)
    except UnicodeDecodeError:
        data = pd.read_csv(input_file_name, sep=separator, encoding="latin-1")
    return data

def load_group_file(group_file_name, *, separator: str = "\t"):
    """
    Load the group file from the data folder.

    Args:
        project_folder (pathlib.Path): The project folder.
        group_file_name (str): The name of the group file.
        separator (str): Column separator used in the grouping file.

    Returns:
        pd.DataFrame: The group file data.
    """
    group = pd.read_csv(group_file_name, sep=separator)
    return group