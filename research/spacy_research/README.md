# spaCy
### Overview
spaCy is a library for advanced Natural Language Processing (NLP). It provides [pretrained pipelines](https://spacy.io/models) of different sizes and purposes "e.g. ... for general-purpose pipeline with tagging, parsing, lemmatization and named entity recognition, or [...] for only tagging, parsing and lemmatization". These pipelines are trained on different types of data such as the news or the web. 

[spaCy](https://spacy.io/) provides support for 75+ languages, pretrained transformers such as BERT, support for custom models in PyTorch and TensorFlow, visualizers for syntax and NER, as well as the option to customize and extend existing solutions to suit one's needs. 

Some examples of spaCy's capabilities can be seen in the `spaCy_research.ipynb` file that is found in this directory. 

### Conclusions
First of all, spaCy's documentation is well-written and comprehensive, making it easier to work with. Furthermore, spaCy is highly customizable as it enables the integration of custom models and pipelines as well as fine-tuning existing ones to some extent; it is [optimized](https://www.seaflux.tech/blogs/NLP-libraries-spaCy-NLTK-differences) for performance, seen as it is implemented in cython, so it is appropriate for processing large quantities of data. 

### Other interesting things
1. [spaCy Layout](https://github.com/explosion/spacy-layout) is a plugin that processes structured documents such as PDFs, Word documents (and in our case XML trees), and outputs spaCy's Doc objects with labels. This workflow facilitates applying inter alia, linguistic analysis, NER, text classification to documents. 
2. According to [this](https://medium.com/%40prabhuss73/spacy-vs-nltk-a-comprehensive-comparison-of-two-popular-nlp-libraries-in-python-b66dc477a689) article, spaCy is not as flexible as NLTK. Perhaps we should look into it. 

### Sources
1. spaCy's official website: https://spacy.io/ 
2. spaCy's Github: https://github.com/explosion/spaCy?tab=readme-ov-file 
3. Introduction to spaCy: https://spacy.io/usage/facts-figures
4. spaCy tutorial: https://spacy.io/usage/spacy-101 
5. Installation: https://spacy.io/usage 