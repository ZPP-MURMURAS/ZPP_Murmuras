# spaCy
### Overview
spaCy is a library for advanced Natural Language Processing (NLP). It provides [pretrained pipelines](https://spacy.io/models) of different sizes and purposes "e.g. ... for general-purpose pipeline with tagging, parsing, lemmatization and named entity recognition, or [...] for only tagging, parsing and lemmatization". These pipelines are trained on different types of data such as the news or the web. 

[spaCy](https://spacy.io/) provides support for 75+ languages, pretrained transformers such as BERT, support for custom models in PyTorch and TensorFlow, visualizers for syntax and NER, as well as the option to customize and extend existing solutions to suit one's needs. 

Some examples of spaCy's capabilities can be seen in the `spaCy_research.ipynb` file that is found in this directory. 

### Saving and loading your model
From my [research](https://spacy.io/usage/saving-loading), I found that you can do the following steps to save your model and then use it with spaCy. 

```python
nlp.to_disk("/path/to/model") # saving the model to disk
nlp = spacy.load("/path/to/model") # loading the model from disk
```

You can make a package from your model
```bash
python -m spacy package /path/to/model /output/path 
```

The provided link has all the relevant information. 

### spacy-transformers
[spacy-transformers](https://github.com/explosion/spacy-transformers) is a package that "provides spaCy components and architectures to use transformer models via Hugging Face's transformers in spaCy." This allows for using models from HF such as BERT in a project. The installation process and guides are provided in the link. It does have some limitations such as not supporting task-specific heads like token or text classification. 

### spacy-huggingface-pipelines 
[spacy-huggingface-pipelines](https://github.com/explosion/spacy-huggingface-pipelines) - "This package provides spaCy components to use pretrained Hugging Face Transformers pipelines for inference only."

### Conclusions
First of all, spaCy's documentation is well-written and comprehensive, making it easier to work with. Furthermore, spaCy is highly customizable as it enables the integration of custom models and pipelines as well as fine-tuning existing ones to some extent; it is [optimized](https://www.seaflux.tech/blogs/NLP-libraries-spaCy-NLTK-differences) for performance, seen as it is implemented in cython, so it is appropriate for processing large quantities of data. 

### Other interesting things
1. [spaCy Layout](https://github.com/explosion/spacy-layout) is a plugin that processes structured documents such as PDFs, Word documents (and in our case XML trees), and outputs spaCy's Doc objects with labels. This workflow facilitates applying inter alia, linguistic analysis, NER, text classification to documents. 
2. According to [this](https://medium.com/%40prabhuss73/spacy-vs-nltk-a-comprehensive-comparison-of-two-popular-nlp-libraries-in-python-b66dc477a689) article, spaCy is not as flexible as NLTK. Perhaps we should look into it. 
3. [BERT](https://spacy.io/universe/project/bertopic) and spaCy
4. [spaCy projects](https://spacy.io/usage/projects) - allows for uploading projects and sharing them, could be useful for uploading our own custom projects, etc.
5. [Saving progress](https://spacy.io/usage/saving-loading)

### Sources
1. spaCy's official website: https://spacy.io/ 
2. spaCy's Github: https://github.com/explosion/spaCy?tab=readme-ov-file 
3. Introduction to spaCy: https://spacy.io/usage/facts-figures
4. spaCy tutorial: https://spacy.io/usage/spacy-101 
5. Installation: https://spacy.io/usage 

### Loading custom models
#### Useful sources
1. https://stackoverflow.com/questions/68542743/load-custom-trained-spacy-model  
