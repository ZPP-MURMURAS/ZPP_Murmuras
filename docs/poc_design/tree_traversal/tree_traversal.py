import pandas as pd
import sys
from typing import Any

def validate_input(file_path: str, row_index: Any) -> tuple:
    if not file_path.endswith('.csv'):
        exit("File must be a CSV file.")

    try:
        row_index = int(row_index) - 2
        if row_index < 0:
            exit("Row index must be an integer greater than 1.")
    except ValueError:
        exit("Row index must be an integer.")
    
    return file_path, row_index 

def find_parent(file_path: Any, index: Any) -> pd.Series:
    """
    Find the parent of the given row index.
    """

    file_path, index = validate_input(file_path, index)
    df = to_df(file_path)
    current_depth = df.loc[index, 'view_depth']

    for i in range(index - 1, -1, -1):
        if df.loc[i, 'view_depth'] < current_depth:
            return df.loc[i]

    return None

def find_children(file_path: str, index: int) -> list:
    """
    Find the children of the given row index.
    """
    
    file_path, index = validate_input(file_path, index)
    df = to_df(file_path)

    current_depth = df.loc[index, 'view_depth']
    children = []

    for i in range(index + 1, len(df)):
        if df.loc[i, 'view_depth'] == current_depth + 1:
            children.append(df.loc[i])
        elif df.loc[i, 'view_depth'] <= current_depth:
            break

    return children if children else None

def to_df(file_path):
    """
    Load the CSV file and return the DataFrame with only the necessary columns.
    """
    df = pd.read_csv(file_path)
    if 'view_depth' not in df.columns or 'view_class_name' not in df.columns:
        exit("CSV file must contain 'view_depth' and 'view_class_name' columns.")
        
    df = df[['view_depth', 'view_class_name']]

    return df

if __name__ == "__main__":
    '''
    The script takes in a CSV file path and a row index as command line arguments.
    The row index must be an integer greater than 1.
    The script then finds the parent and children of the row at the given index.
    '''

    if len(sys.argv) != 3:
        print("Usage: python3 file_name.py csv_file_name row_index")
        sys.exit(1)

    parent = find_parent(sys.argv[1], sys.argv[2])
    if parent is not None:
        print("Parent:", parent.to_dict())
    else:
        print("No parent found.")

    children = find_children(sys.argv[1], sys.argv[2])
    if children is not None:
        print("Children:")
        for child in children:
            print(child.to_dict())
    else:
        print("No children found.")
