import data.data_reader as data_reader
from modelclass import ModelClass
import logging
import datetime
import os

dir = os.path.dirname(os.path.abspath(__file__))
import argparse

# create parser
parser = argparse.ArgumentParser(description="Adversarial training")
parser.add_argument("--dataset", type=str, help="Dataset name", required=True)
parser.add_argument("--model", type=str, help="Model name", required=True)
parser.add_argument(
    "--wandb_logging",
    action=argparse.BooleanOptionalAction,
    help="Whether to log to wandb",
    default=False,
)

# parse arguments
args = parser.parse_args()
DATASET_NAME = args.dataset
MODEL_NAME = args.model
WANDB_LOGGING = args.wandb_logging

if WANDB_LOGGING:
    import wandb

    # login to wandb with key
    wandb.login(key="KEY")
    wandb.init(
        project="XAI-Attack",
        entity="ENTITY",
        group="adversarial_example_creation",
        name=DATASET_NAME
        + "_"
        + MODEL_NAME
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

logging_path = os.path.join(
    dir,
    "../logs/evaluation/adversarial_training/" + DATASET_NAME + "/" + MODEL_NAME,
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log",
)
os.makedirs(os.path.dirname(logging_path), exist_ok=True)
# setting up logger to log to file in logs folder with name created from current time and date
logging.basicConfig(
    filename=logging_path,
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

x_train, y_train, x_test, y_test, x_dev, y_dev = data_reader.read_data(
    DATASET_NAME, adversarial=False
)
is_sentence_pair_task = type(x_train) is tuple
x_train_adv, y_train_adv = data_reader.read_adv_examples(
    DATASET_NAME, MODEL_NAME, is_sentence_pair_task
)
num_labels = len(set(y_train))

model_baseline = ModelClass(MODEL_NAME, False, None, num_labels)
model_baseline.train(x_train + x_dev, y_train + y_dev, x_test, y_test)
model_baseline.save(
    os.path.join(dir, "../model/" + DATASET_NAME + "/" + MODEL_NAME + "_baseline")
)

model_adv = ModelClass(MODEL_NAME, False, None, num_labels)
model_adv.train(
    x_train + x_train_adv + x_dev, y_train + y_train_adv + y_dev, x_test, y_test
)
model_adv.save(
    os.path.join(dir, "../model/" + DATASET_NAME + "/" + MODEL_NAME + "_adv")
)
