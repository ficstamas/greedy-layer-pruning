#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import itertools
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import pandas as pd

import numpy as np
from datasets import load_dataset, load_metric, Features, ClassLabel, Sequence

import transformers
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from torch import Tensor
from typing import Dict

# Custom pruning models
from models.model_factory import create_model
from disable_checkpoint_handler import DisableCheckpointCallbackHandler
import wandb

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "ner": ("tokens", None),
    "pos": ("tokens", None),
}

task_to_metric = {
    "cola": "matthews_correlation",
    "mrpc": "f1",
    "rte": "accuracy",
    "sst2": "accuracy",
    "ner": "f1",
    "pos": "f1",
}

logger = logging.getLogger(__name__)


def dynamic_padding_sequence_classification(batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
    features = ['input_ids', 'attention_mask', 'token_type_ids']
    max_len = max([len(b['input_ids']) for b in batch])

    batch = {key: [torch.tensor(batch[i][key]) for i in range(len(batch))] for key in batch[0].keys()}

    b = {}
    for f in features:
        try:
            b[f] = torch.stack([torch.nn.functional.pad(t, (0, max_len - t.shape[-1]), value=0) for t in batch[f]])
        except KeyError:
            continue
    b['labels'] = torch.stack(batch['label'])
    return b


def dynamic_padding_token_classification(batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
    features = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    max_len = max([len(b['input_ids']) for b in batch])

    batch = {key: [torch.tensor(batch[i][key]) for i in range(len(batch))] for key in batch[0].keys()}
    b = {}
    for f in features:
        try:
            if f != 'label':
                b[f] = torch.stack(
                    [torch.nn.functional.pad(t, (0, max_len - t.shape[-1]), value=0) for t in batch[f]])
            else:
                b[f] = torch.stack(
                    [torch.nn.functional.pad(t, (0, max_len - t.shape[-1]), value=-100) for t in batch[f]])
        except KeyError:
            continue
    return b


# See https://github1s.com/huggingface/transformers/blob/HEAD/src/transformers/training_args.py
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default="glue",
        metadata={"help": "The name of the dataset to train on: " + ", ".join(task_to_keys.keys())},
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    task_type: Optional[str] = field(
        default="sequence",
        metadata={"help": "The type of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    prune_n_layers: int = field(
        default=6,
        metadata={
            "help": "The maximum number of layers that is pruned afterwards. "
        },
    ),
    prune_all_but_one: bool = field(
        default=True,
        metadata={
            "help": "Prune all but one layer"
        },
    )
    prune_method: Optional[str] = field(
        default="greedy", metadata={"help": "Prune greedy in O(n^2) or perfect in O(n!)."}
    )
    wandb_project: Optional[str] = field(
        default="greedy-layer-pruning", metadata={"help": "Wandb project name"}
    )
    wandb_entity: Optional[str] = field(
        default="szegedai-semantics", metadata={"help": "Wandb entitry name"}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    if not os.path.exists("experiments/layer_files"):
        os.makedirs("experiments/layer_files")

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.task_type = data_args.task_type

    training_args.group_by_length = True

    wandb.init(project=data_args.wandb_project, entity=data_args.wandb_entity, config={
        "model_name": model_args.model_name_or_path,
        "seed": training_args.seed,
        "dataset_name": data_args.dataset_name,
        "subset_name": data_args.task_name,
        "task_type": data_args.task_type,
    })
    # Detecting last checkpoint.
    last_checkpoint = None
    #training_args.output_dir = f"{training_args.output_dir}/{data_args.task_name}/{model_args.model_name_or_path}/{model_args.prune_method}/{str(model_args.prune_n_layers)}/{str(training_args.seed)}"
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.dataset_name == "conll2003":
            datasets = load_dataset(data_args.dataset_name)
            for splits in datasets.keys():
                ds = datasets[splits].add_column(name="label", column=datasets[splits][f"{data_args.task_name}_tags"])
                feature_dict = {}
                for key, item in ds.features.items():
                    feature_dict[key] = item

                feature_dict["label"] = Sequence(ClassLabel(
                    num_classes=datasets[splits].features[f"{data_args.task_name}_tags"].feature.num_classes,
                    names=datasets[splits].features[f"{data_args.task_name}_tags"].feature.names
                ))
                ds = ds.cast(Features(feature_dict))
                datasets[splits] = ds
        elif data_args.dataset_name == "glue":
            datasets = load_dataset(data_args.dataset_name, data_args.task_name)
            val = datasets['validation'].train_test_split(train_size=0.5, seed=0)
            datasets['validation'] = val['train']
            datasets['test'] = val['test']
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            if data_args.dataset_name == "conll2003":
                label_list = datasets["train"].features["label"].feature.names
            else:
                label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None
    )

    optional_tokenizer_kwargs = {} if "roberta" not in model_args.tokenizer_name else {"add_prefix_space": True}
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        **optional_tokenizer_kwargs
    )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}
    config.label2id = label_to_id

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=max_seq_length, truncation=True)
        result["length"] = [len(x) for x in result['input_ids']]
        # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
        #     result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    def preprocess_function_conll(examples):
        kwargs = {
            "padding": False,
            "max_length": max_seq_length,
            "truncation": True
        }
        if "is_split_into_words" not in kwargs:
            kwargs["is_split_into_words"] = True
        tokenized_inputs = tokenizer(
            examples["tokens"], **kwargs
        )

        """
        wikiann: O (0), B-PER (1), I-PER (2), B-ORG (3), I-ORG (4), B-LOC (5), I-LOC (6).
        conll:   O (0), B-PER (1), I-PER (2), B-ORG (3)  I-ORG (4), B-LOC (5), I-LOC (6), B-MISC (7), I-MISC (8)
        model:   O (0), B-PER (3), I-PER (4), B-ORG (5), I-ORG (6), B-LOC (7), I-LOC (8), B-MISC (1), I-MISC (2).
        """

        labels = []
        task = data_args.task_name

        id2label = {i: v for i,v in enumerate(datasets['train'].features[f'{task}_tags'].feature.names)}
        label2id = {v: k for k,v in id2label.items()}
        if task == "ner":
            if "B-PER" not in label2id:
                label2id["B-PER"] = label2id["I-PER"]

        data = [
            [label2id[id2label[x]] for x in y]
            for y in examples[f'label']
        ]

        for i, label in enumerate(data):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label
                # to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either
                # the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if False else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["length"] = [len(x) for x in tokenized_inputs['input_ids']]
        return tokenized_inputs

    datasets = datasets.map(
        preprocess_function if data_args.task_type == "sequence" else preprocess_function_conll,
        batched=True, load_from_cache_file=not data_args.overwrite_cache,
        remove_columns=["label"] if data_args.task_type == "token" else []
    )

    # If the following line is uncommented, the dev set is used. Otherwise a 15% split of the training set is used.
    # eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    if data_args.task_name is not None or data_args.test_file is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        if data_args.dataset_name == "glue":
            metric = load_metric("glue", data_args.task_name)
        elif data_args.dataset_name == "conll2003":
            metric = load_metric("seqeval")
        else:
            metric = load_metric("accuracy")
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    def compute_metrics_conll(p):
        id2label = config.id2label
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        return results

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        if data_args.task_type == "sequence":
            data_collator = dynamic_padding_sequence_classification  # default_data_collator
        else:
            data_collator = dynamic_padding_token_classification  # DataCollatorForTokenClassification(tokenizer=tokenizer)
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Create pruning strategy
    prune_fn = prune_greedy if data_args.prune_method == "greedy" else prune_optimal
    prune_fn(config, data_args, model_args, training_args, train_dataset,
             eval_dataset, test_dataset, compute_metrics if data_args.task_type == "sequence" else compute_metrics_conll,
             tokenizer, data_collator, datasets)


def prune_optimal(config, data_args, model_args, training_args, train_dataset,
                  eval_dataset, compute_metrics, tokenizer, data_collator, datasets):
    num_layers = data_args.prune_n_layers
    all_layers = [i for i in range(config.num_hidden_layers)]
    file_rows = []

    for num_layers_to_prune in range(1, data_args.prune_n_layers+1):
        all_combinations = [i for i in itertools.combinations(all_layers, num_layers_to_prune)]
        cache_dict = {}

        print("\n#\n# TRAINING %d NETWORKS!\n#" % len(all_combinations))
        for pruned_layers in all_combinations:
            pruning_id = "-".join(map(str, pruned_layers))
            loss = evaluate_model(cache_dict, pruning_id, pruned_layers, config, model_args, data_args,
                    training_args, train_dataset, eval_dataset, compute_metrics, tokenizer, data_collator, datasets)

            # Log result for later analysis
            with open(f'experiments/layer_files/{model_args.model_name_or_path}_{data_args.task_name}_perfect_log.txt', 'a') as f:
                f.writelines(f"{pruning_id};{loss[pruning_id]}\n")

        layer_to_prune = None
        for key in cache_dict.keys():
            if(layer_to_prune == None or cache_dict[key] >= cache_dict[layer_to_prune]):
                layer_to_prune = key
        file_rows.append(layer_to_prune)


    # DONE - store into file
    with open(f'experiments/layer_files/{model_args.model_name_or_path}_{data_args.task_name}_perfect.txt', 'w') as f:
        f.writelines("%s\n" % l for l in file_rows)


def prune_greedy(config, data_args, model_args, training_args, train_dataset,
                 eval_dataset, test_dataset, compute_metrics, tokenizer, data_collator, datasets):

    finally_pruned_layers = []
    cache_dict = {}
    num_iterations = min(data_args.prune_n_layers, config.num_hidden_layers-1) if not data_args.prune_all_but_one else config.num_hidden_layers - 1
    table = {"dev": [], "test": [], "pruned": []}
    while len(finally_pruned_layers) < num_iterations:
        layer_ids = [i for i in range(config.num_hidden_layers) if i not in finally_pruned_layers]
        lower_layer = 0
        upper_layer = len(layer_ids)-1
        middle_layer = int(upper_layer / 2)

        for i in range(0, len(layer_ids)):
            print("\n####")
            cache_dict = evaluate_model(
                cache_dict, layer_ids[i], finally_pruned_layers, config, model_args, data_args,
                training_args, train_dataset, eval_dataset, test_dataset, compute_metrics, tokenizer, data_collator, datasets
            )
            print(cache_dict)

        # Log result for later analysis
        with open(f'experiments/layer_files/{model_args.model_name_or_path}_{data_args.task_name}_greedy_log.txt', 'a') as f:
            f.writelines(f"{str(len(finally_pruned_layers))};{str(cache_dict)}\n")

        layer_to_prune = -1
        for key in cache_dict.keys():
            if layer_to_prune < 0 or cache_dict[key][0] >= cache_dict[layer_to_prune][0]:
                layer_to_prune = key

        eval_score, test_score = cache_dict[layer_to_prune]

        finally_pruned_layers.append(layer_to_prune)
        wandb.log({
            f"progress_eval": eval_score,
            "num_layers": config.num_hidden_layers - len(finally_pruned_layers)
        })
        wandb.log({
            f"progress_test": test_score,
            "num_layers": config.num_hidden_layers - len(finally_pruned_layers)
        })
        table["dev"].append(eval_score)
        table["test"].append(test_score)
        table["pruned"].append(layer_to_prune)

        cache_dict = {}
        print("PRUNED LAYER %s" % finally_pruned_layers)

    wandb.log({"progress_table": wandb.Table(dataframe=pd.DataFrame(data=table))})
    # DONE - store into file
    with open(f'experiments/layer_files/{model_args.model_name_or_path}_{data_args.task_name}_greedy.txt', 'w') as f:
        f.writelines("%s\n" % l for l in finally_pruned_layers)


def evaluate_model(cache_dict, layer_id, finally_pruned_layers, config, model_args, data_args,
                   training_args, train_dataset, eval_dataset, test_dataset, compute_metrics, tokenizer, data_collator,
                   datasets):

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if layer_id in cache_dict:
        print("Layer %d from cache: %.4f" % (layer_id, cache_dict[layer_id]))
        return cache_dict[layer_id]

    print(f"Calculate layer {str(layer_id)}")
    model = create_model(config, model_args)

    model.prune_layers(finally_pruned_layers)
    if isinstance(layer_id, int):
        model.prune_layers([layer_id])

    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    trainer.add_callback(
        DisableCheckpointCallbackHandler()
    )

    # Training
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics

    # Evaluation
    eval_results = {}
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        eval_mismatch = datasets["validation_mismatched"]
        eval_datasets.append(eval_mismatch)

    for eval_dataset, task in zip(eval_datasets, tasks):
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        eval_results.update(eval_result)

    # res = eval_results.get("eval_loss", None)
    res = None
    res = res or eval_results.get("eval_f1", None)
    res = res or eval_results.get("eval_spearmanr", None)
    res = res or eval_results.get("eval_matthews_correlation", None)
    res = res or eval_results.get("eval_accuracy", None)

    if res is None and "eval_matthews_correlation" in eval_results:
        res = eval_results.get("eval_matthews_correlation", None)

    if res is None:
        raise Exception("New performance metric found!", eval_results)

    res = round(res, 3)

    # Evaluation
    test_results = {}
    logger.info("*** Evaluate Test set ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    test_datasets = [test_dataset]

    for ts, task in zip(test_datasets, tasks):
        test_result = trainer.evaluate(eval_dataset=ts)
        test_results.update(test_result)

    # res = eval_results.get("eval_loss", None)
    test = None
    test = test or test_results.get("eval_f1", None)
    test = test or test_results.get("eval_spearmanr", None)
    test = test or test_results.get("eval_matthews_correlation", None)
    test = test or test_results.get("eval_accuracy", None)

    if test is None and "eval_matthews_correlation" in test_results:
        test = eval_results.get("eval_matthews_correlation", None)

    test = round(test, 3)

    cache_dict[layer_id] = [res, test]
    return cache_dict


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()



if __name__ == "__main__":
    main()
