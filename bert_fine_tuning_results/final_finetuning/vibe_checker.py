from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from datasets import load_dataset

def __align_labels_with_tokens(labels: list, word_ids: list, bi_split: bool) -> list[int]:
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            if not bi_split:
                if label > 0: # If the label is not O or N/A
                    label = 2*label - 1 # Change it to B-XXX
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else: # Same word as previous token
            label = labels[word_id]
            if not bi_split:
                label = 2*label
            else: # Adjust so that one word has only one B-X token.
                if label%2 == 1: # I-X are even; Unknown is even also. B-X is odd.
                    label += 1 # from (2*label)-1 to 2*label
            new_labels.append(label)

    return new_labels

model = AutoModelForTokenClassification.from_pretrained("GustawB/zpp-murmuras", token='')
dataset = load_dataset("zpp-murmuras/coupon_select_big_plain_rev2", token='')
tokenizer = AutoTokenizer.from_pretrained("GustawB/zpp-murmuras", token='')

sample = dataset['dm']
text_input = sample["texts"][11]
label_input = sample["labels"][11]

import torch
tokenized_input = tokenizer(text_input, is_split_into_words=True, return_tensors="pt")
word_ids = tokenized_input.word_ids(batch_index=0)
prepared_labels = __align_labels_with_tokens(label_input, word_ids, True)
pred2 = model(**tokenized_input)
logits = pred2.logits
predicted_labels = torch.argmax(logits, axis=2)
for a, b in zip(prepared_labels, predicted_labels[0]):
    print(f"Truth: {a}; Pred: {b}")


