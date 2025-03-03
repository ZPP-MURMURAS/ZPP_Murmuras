import pandas as pd
import xml.etree.ElementTree as ET


def prepare_content_generic(content_generic_df):
    df = content_generic_df.copy()
    df = df[df['is_visible'] != 0]
    df = df[df['text'].notna()]
    df = df.drop(columns=['CAST(id, \'String\')', 'id', 'user_id', 'time', 'i', 'language', 'application_name', 'package_name', 'class_name', 'context', 'view_id', 'view_class_name', 'is_visible', 'x_1', 'y_1', 'x_2', 'y_2'])
    return df


def content_generic_2_xml(content_generic_df):
    df = prepare_content_generic(content_generic_df)
    xml = ET.Element('root')
    
    if df.empty:
        return xml

    timestamp = df['seen_timestamp'].iloc[0]
    timestamp_element = ET.SubElement(xml, 'view')
    element_stack = [(-1, timestamp_element)]

    for index, row in df.iterrows():
        if row['seen_timestamp'] != timestamp:
            timestamp = row['seen_timestamp']
            timestamp_element = ET.SubElement(xml, 'view')
            element_stack = [(-1, timestamp_element)]

        while row['view_depth'] <= element_stack[-1][0]:
            element_stack.pop()

        text_element = ET.SubElement(element_stack[-1][1], 'text')
        text_element.text = str(row['text'])

        element_stack.append((row['view_depth'], text_element))

    return xml


input_csv = 'dm_content_generic_complete_from_2024.csv'
df = pd.read_csv(input_csv)
xml = content_generic_2_xml(df)
tree = ET.ElementTree(xml)
tree.write('out.xml')
