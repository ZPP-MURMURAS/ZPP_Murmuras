import pandas as pd
import sys
from typing import Any


'''
Args: index: Any
Validates the index and returns the index adjusted into a 0-based index.
'''
def validate_index(index: Any) -> int:
    try:
        index = int(index) - 2
        if index < 0:
            exit("Row index must be an integer greater than 1.")
    except ValueError:
        exit("Row index must be an integer.")
    
    return index


'''
Args: file_path: Any, row_index: Any
row_index is the index of the row in the CSV file.
Returns: tuple containing the file path and the row index adjusted into a 0-based index.
'''
def validate_input(file_path: Any, row_index: Any) -> tuple:
    if not file_path.endswith('.csv'):
        exit("File must be a CSV file.")

    row_index = validate_index(row_index)
    return file_path, row_index 


'''
Args: file_path: Any, index: Any
Validates the input and returns the DataFrame of the parent to the given row index.
You can pass either the file path or the DataFrame.
'''
def find_parent(df: Any, index: Any) -> Any:
    value_at_row = df.loc[index, 'view_depth']

    above_indices = df.loc[:index].iloc[:-1]  # Use loc to slice by index, then exclude the current row
    smaller_rows = above_indices[above_indices['view_depth'] < value_at_row]
    if smaller_rows.empty:
        return None
    return smaller_rows.index[-1]


'''
Args: file_path: Any, index: Any
Finds the children of the given row index. 
You can pass either the file path or the DataFrame.
'''
def find_children(file_path_or_df: Any, index: Any) -> list:
    """
    Find the children of the given row index.
    """

    if isinstance(file_path_or_df, pd.DataFrame):
        df = file_path_or_df
        index = validate_index(index)
    else:
        file_path, index = validate_input(file_path_or_df, index)
        df = to_df(file_path)

    current_depth = df.loc[index, 'view_depth']
    children = []

    for i in range(index + 1, len(df)):
        if df.loc[i, 'view_depth'] == current_depth + 1:
            children.append(df.loc[i])
        elif df.loc[i, 'view_depth'] <= current_depth:
            break

    return children if children else None


def get_children_counts(frame: pd.DataFrame) -> dict:
    """
    returns map from node index to number of children of this node
    """
    depth_id_stack = []
    child_no = {}
    for ix, row in frame.iterrows():
        child_no[ix] = 0
        d = row['view_depth']
        while depth_id_stack and depth_id_stack[-1][0] >= d:
            depth_id_stack.pop()
        if depth_id_stack:
            child_no[depth_id_stack[-1][1]] += 1
        depth_id_stack.append([d, ix])
    return child_no


def get_leafs(frame: pd.DataFrame) -> list:
    """
    returns indexes of the leaf nodes.
    """
    series = frame['view_depth']
    indexes = series[:-1].index[series[:-1].reset_index(drop=True) >= series[1:].reset_index(drop=True)]
    return indexes.astype(int).tolist()

'''
Args: file_path: Any, args: Any
Loads the CSV file and returns the DataFrame with only the necessary columns.
Args define the columns that must be present in the CSV file.
'''
def to_df(file_path, columns=['view_depth', 'view_class_name']):
    """
    Load the CSV file and return the DataFrame with only the necessary columns.
    """
    df = pd.read_csv(file_path)
    if 'view_depth' not in df.columns or 'view_class_name' not in df.columns:
        exit("CSV file must contain 'view_depth' and 'view_class_name' columns.")
        
    df = df[columns]

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
