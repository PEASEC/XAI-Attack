# XAI-Attack

This is the repository of the adversarial example generation method XAI-Attack.
Disclaimer: This repository contains a experimental version of the code and may therefore contain bugs or unintended behaviour. We are also working to further improve the user experience and add all features.

The adversarial examples (No Filtering, Count Filtering, Indicator Filtering, SHAP, Random Insert, BERT-Attack, BAE) of the XAI-Attack-Paper are available here: https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4323/Adversarial%20Examples.zip?sequence=3&isAllowed=y or https://anonymfile.com/yp0g/adversarial-examples.zip.


## Contact and Support

If you have any questions, need access to datasets or the complete research data, or if you encounter any bugs, please feel free to contact me!


## Getting started
The gerneration of adversarial examples is done by running the `main.py` script. The script takes the following arguments:
- `--dataset`: The dataset to use. Currently supported for adversarial example creation are all the GLUE datasets.
- `--model`: The model to use. Currently supported are all Huggingface models. We tested it with the distilBERT-base-uncased model.
- `--wandb_logging`: Whether to use wandb or local logging. For wandb logging please set wandb variables in the script.
- `--filtering`: Which filtering method to use (please refer to the paper): `none`, `count` or `indicator_words`.
- `--random_insertion`: Whether to use random insertion instead of prefixing of the adversarial words.

The script will create a folder in the `results` directory with the name of the dataset and the model. In this folder the adversarial examples will be saved.


## Experiments
The experiments of the adversarial examples are done running the `adversarial_testing.py`, `adversarial_training.py`, or the `adversarial_transfer.py` (for detailed experiment descriptions and settings please have a look at the paper).


The `adversarial_testing.py` script trains one model on the given GLUE task and one model on the given GLUE task and the adversarial examples for this tasks (IMPORTANT: These have to be created by the `main.py` script before or inserted into the results directory) and evaluates them on the dev set of the __Adversarial GLUE__ task. The script takes the following arguments:
- `--dataset`: The dataset to use. Currently supported for adversarial example experiments are `mnli`, `sst2`, `rte`, `qnli`, and `qqp`.
- `--model`: The model to use. Currently supported are all Huggingface models. We tested it with the distilBERT-base-uncased model.
- `--wandb_logging`: Whether to use wandb or local logging. For wandb logging please set wandb variables in the script.

The `adversarial_training.py` script trains one model on the given GLUE task and one model on the given GLUE task and the adversarial examples for this tasks (IMPORTANT: These have to be created by the `main.py` script before or inserted into the results directory) and evaluates them on the dev set of the GLUE task. The script takes the following arguments:
- `--dataset`: The dataset to use. Currently supported for adversarial example experiments are `mnli`, `sst2`, `rte`, `qnli`, and `qqp`.
- `--model`: The model to use. Currently supported are all Huggingface models. We tested it with the distilBERT-base-uncased model.
- `--wandb_logging`: Whether to use wandb or local logging. For wandb logging please set wandb variables in the script.

The `adversarial_transfer.py` script evaluates the adversarial examples of the *basemodel* (IMPORTANT: These have to be created by the `main.py` script before or inserted into the results directory) and evaluates the *transfermodel* on them. The script takes the following arguments:
- `--dataset`: The dataset to use. Currently supported for adversarial example experiments are `mnli`, `sst2`, `rte`, `qnli`, and `qqp`.
- `--basemodel`: The model to use. Currently supported are all Huggingface models. We tested it with the distilBERT-base-uncased model.
- `--transfermodel`: The model to use. Currently supported are all Huggingface models. We tested it with the distilBERT-base-uncased, bert-base-uncased and roberta-base model.
- `--wandb_logging`: Whether to use wandb or local logging. For wandb logging please set wandb variables in the script.


## Datasets


Currently fully supported for adversarial creation and the evaluation experiments are `mnli`, `sst2`, `rte`, `qnli`, and `qqp`.
Adversarial creation also works with all GLUE datasets.

If you want to add a new dataset, please have a look at the `data/data_reader.py`.

The adversarial examples (No Filtering, Count Filtering, Indicator Filtering, SHAP, Random Insert, BERT-Attack, BAE) of the XAI-Attack-Paper are available here: https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4323/Adversarial%20Examples.zip?sequence=3&isAllowed=y or https://anonymfile.com/yp0g/adversarial-examples.zip


## Requirements 

Please run the requirements.txt file to install all the necessary packages.

## Cluster
The code was tested on a cluster. For following the same procedure, please have a look at the `jobscript` folder. 

## Acknowledgements
The repository contains adjusted code from the LIME repository.

## Authors

Markus Bayer, Markus Neiczer, Maximilian Samsinger, Björn Buchhold, Christian Reuter
