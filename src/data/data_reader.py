from sklearn.model_selection import train_test_split
from datasets import load_dataset
import os

dir = os.path.dirname(os.path.abspath(__file__))
import pickle


def read_adv_sst2():
    dataset = load_dataset("adv_glue", "adv_sst2")

    return dataset["validation"]["sentence"], dataset["validation"]["label"]


def read_glue(task, adversarial):
    if not adversarial:
        if task == "mnli_matched":
            dataset = load_dataset("glue", "mnli")

            train = dataset["train"]
            test = dataset["test_matched"]
            dev = dataset["validation_matched"]
        elif task == "mnli_mismatched":
            dataset = load_dataset("glue", "mnli")

            train = dataset["train"]
            test = dataset["test_mismatched"]
            dev = dataset["validation_mismatched"]
        else:
            dataset = load_dataset("glue", task)

            train = dataset["train"]
            test = dataset["test"]
            dev = dataset["validation"]

        keys = list(train.column_names)
        if len(keys) == 3:
            key = keys[0]
            x_train = train[key]
            y_train = train["label"]

            x_test = test[key]
            y_test = test["label"]

            x_dev = dev[key]
            y_dev = dev["label"]

        elif len(keys) == 4:
            key_1, key_2 = keys[0], keys[1]
            x_train = (train[key_1], train[key_2])
            y_train = train["label"]

            x_test = (test[key_1], test[key_2])
            y_test = test["label"]

            x_dev = (dev[key_1], dev[key_2])
            y_dev = dev["label"]

        return x_train, y_train, x_test, y_test, x_dev, y_dev
    else:
        if task == "mnli_matched":
            task = "mnli"
        dataset = load_dataset("adv_glue", "adv_" + task)
        dev = dataset["validation"]

        keys = list(dev.column_names)
        if len(keys) == 3:
            key = keys[0]
            x_dev = dev[key]
            y_dev = dev["label"]

        elif len(keys) == 4:
            key_1, key_2 = keys[0], keys[1]
            x_dev = (dev[key_1], dev[key_2])
            y_dev = dev["label"]

        return x_dev, y_dev


def read_data(dataset_name, adversarial):
    return read_glue(dataset_name, adversarial)


def read_adv_examples(dataset_name, model_name, is_sentence_pair_task):
    # reading in the adversarial examples for each file in the folder
    # the files follow the structure: "../../results/adversarial_examples/" + dataset_name + "/" + model_name + "/adversarial_examples_class_" + class_number + ".txt"

    # first get all the files in the folder
    files = os.listdir(
        os.path.join(
            dir, "../results/adversarial_examples/" + dataset_name + "/" + model_name
        )
    )

    adversarial_examples = []
    i = 0
    while "adversarial_examples_class_" + str(i) + ".txt" in files:
        with open(
            os.path.join(
                dir,
                "../results/adversarial_examples/"
                + dataset_name
                + "/"
                + model_name
                + "/adversarial_examples_class_"
                + str(i)
                + ".txt",
            ),
            "r",
            encoding="utf-8",
        ) as f:
            adversarial_examples.append(f.read().splitlines())
        i += 1

    if is_sentence_pair_task:
        for i, adversarial_examples_class in enumerate(adversarial_examples):
            adversarial_examples_class_list = []
            [
                adversarial_examples_class_list.append(adv.split("_SEP_"))
                for adv in adversarial_examples_class
            ]
            adversarial_examples[i] = list(
                map(list, zip(*adversarial_examples_class_list))
            )

    # return the adversarial examples flattend and a list of labels
    if is_sentence_pair_task:
        adversarial_examples_appended = [[], []]
    else:
        adversarial_examples_appended = []
    labels = []
    for i, adversarial_examples_class in enumerate(adversarial_examples):
        if is_sentence_pair_task:
            adversarial_examples_appended[0] += adversarial_examples_class[0]
            adversarial_examples_appended[1] += adversarial_examples_class[1]
            labels += [i] * len(adversarial_examples_class[0])
        else:
            adversarial_examples_appended += adversarial_examples_class
            labels += [i] * len(adversarial_examples_class)

    return adversarial_examples_appended, labels


def split_train(x_train, y_train):
    x_train, x_dev, y_train, y_dev = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )
    return x_train, y_train, x_dev, y_dev
