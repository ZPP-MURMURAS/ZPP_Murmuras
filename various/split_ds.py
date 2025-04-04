from datasets import load_dataset, DatasetDict, Dataset

data = load_dataset('zpp-murmuras/bert-extraction-ds-rev2', token='')

print(data)

new_datasets = {}
for name, ds in data.items():
    train_split = int(len(ds) * 0.8)
    train_texts = ds['texts'][:train_split]
    train_labels = ds['labels'][:train_split]
    test_texts = ds['texts'][train_split:]
    test_labels = ds['labels'][train_split:]

    new_datasets[name + '_train'] = Dataset.from_dict({'texts': train_texts, 'labels': train_labels}, features=ds.features)
    new_datasets[name + '_test'] = Dataset.from_dict({'texts': test_texts, 'labels': test_labels}, features=ds.features)

final_ds = DatasetDict(new_datasets)

print(final_ds)

final_ds.push_to_hub('zpp-murmuras/bert-extraction-ds-rev3', token='')
