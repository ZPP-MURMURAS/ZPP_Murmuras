from tree_traversal import find_parent, find_children, to_df


row_index = 48
parent = find_parent("example1.csv", row_index)
children = find_children("example1.csv", row_index)

print("Parent:", parent.to_dict() if parent is not None else "No parent")
print("Children:", [child.to_dict() for child in children] if children else "No children")

df = to_df("example1.csv", ['view_depth', 'view_class_name'])

parent = find_parent(df, row_index)
children = find_children(df, row_index)

print("Parent:", parent.to_dict() if parent is not None else "No parent")
print("Children:", [child.to_dict() for child in children] if children else "No children")
