# Knowledge Graphs Datasets

A repository for KG Datasets.
Includes a new WikiData Infobox dataset with 2+ millions training examples

Features:
- Code to clean raw [RDF Triple](https://en.wikipedia.org/wiki/Semantic_triple) data
- Code to generate negative examples
- Code to analyze the datasets (relation histograms, distance of test triples in training graph, etc)
- Datasets available to download

## Negative Examples Files

Files such as `infobox_en_neg_train.ttl` contains negative examples to train models that need them.
Each line of these files have `n` wrong target, separated with a space, 
that correspond to the head and relation **on the same line in the corresponding file**.  
In our example that would be `infobox_en_train.ttl`.
