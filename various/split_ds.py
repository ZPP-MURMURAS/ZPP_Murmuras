from datasets import load_dataset, DatasetDict, Dataset

TYPE = 'llama'

assert TYPE in ['llama', 'bert']

data = load_dataset('zpp-murmuras/llama-ds-w-rev4', token='')

print(data)

if TYPE == 'llama':
    new_datasets = {}
    for name, ds in data.items():
        train_split = int(len(ds) * 0.8)
        train = ds.select(range(train_split))
        val = ds.select(range(train_split, len(ds)))

        new_datasets[name + '_train'] = train
        new_datasets[name + '_test'] = val

    final_ds = DatasetDict(new_datasets)

    print(final_ds)

    final_ds.push_to_hub('zpp-murmuras/llama-ds-w-rev5', token='')

elif TYPE == 'bert':
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
