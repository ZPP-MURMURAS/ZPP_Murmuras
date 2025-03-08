import sys
import pandas as pd
from io import StringIO
import xml.etree.ElementTree as ET
from scrapegraphai.graphs import XMLScraperGraph


def prepare_content_generic(content_generic_df):
    df = content_generic_df.copy()
    df = df[df['text'].notna()]
    df = df[df['seen_timestamp'] != 0]
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


def run_scrape_graph_ai(xml_string):
    graph_config = {
       'llm': {
          'model': 'ollama/llama3.2:1b',
          'temperature': 0.0,
          'format': 'json',
          'model_tokens': 2048,
          'base_url': 'http://localhost:11434',
        }
    }

    scraper_graph = XMLScraperGraph(
        prompt='A coupon consists of a name, a description, and a discount. Extract all coupons from the given phone screen views.',
        source=xml_string,
        config=graph_config,
    )

    return scraper_graph.run()


if __name__ == '__main__':
    input_csv_str = sys.stdin.read()
    df = pd.read_csv(StringIO(input_csv_str), usecols=['seen_timestamp', 'view_depth', 'text'])
    xml = content_generic_2_xml(df)
    xml_string = ET.tostring(xml, encoding='utf-8').decode('utf-8')
    output = run_scrape_graph_ai(xml_string)
    print(output)
