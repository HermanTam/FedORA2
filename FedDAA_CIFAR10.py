
from sklearn import cluster
from utils.utils import *
from utils.constants import *
from utils.args import *
from utils.experiment_logging import ExperimentLogger
from utils.metrics_utils import sample_stats
from utils.experiment_helpers import (
    parse_seed_argument,
    evaluate_iterator_accuracy,
    build_bar_plot,
    aggregate_summary,
)

from torch.utils.tensorboard import SummaryWriter

import copy
import numpy as np
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import silhouette_score


def _assign_objectives_random_ratio(clients, ratio_spec: str, seed: int):
    """
    Assign objectives 'G'/'P' to clients according to a ratio specification,
    e.g. ratio_spec = 'G:70,P:30'.
    """
    if not clients:
        return
    # Parse ratios
    ratios = {"G": 0.0, "P": 0.0}
    for part in ratio_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        key = key.strip().upper()
        if key not in ratios:
            continue
        try:
            ratios[key] += float(val.strip())
        except ValueError:
            continue
    total = ratios["G"] + ratios["P"]
    if total <= 0:
        # Fallback: 50/50
        ratios = {"G": 1.0, "P": 1.0}
        total = 2.0
    n = len(clients)
    n_G = int(round(n * ratios["G"] / total))
    n_P = n - n_G
    assignment = ["G"] * n_G + ["P"] * n_P
    rng = np.random.RandomState(seed)
    rng.shuffle(assignment)
    for client, obj in zip(clients, assignment):
        setattr(client, "objective", obj)


def assign_objectives(clients, assignment_str: str, seed: int):
    """
    Lightweight objective assignment hook.

    This only sets client.objective and does NOT change training logic.
    Supported patterns:
      - 'all:G' or 'all:P'
      - 'random_ratio:G:70,P:30'
    """
    if not assignment_str:
        return
    assignment_str = assignment_str.strip()
    if not assignment_str:
        return

    # All clients same objective
    if assignment_str.startswith("all:"):
        obj = assignment_str.split(":", 1)[1].strip().upper()
        if obj not in ("G", "P"):
            return
        for client in clients:
            setattr(client, "objective", obj)
        return

    # Random ratio G/P
    if assignment_str.startswith("random_ratio:"):
        ratio_spec = assignment_str[len("random_ratio:") :]
        _assign_objectives_random_ratio(clients, ratio_spec, seed=seed)
        return

    # Fallback: do nothing for unsupported patterns (keep behaviour unchanged)
    return

def init_clients(args_, root_path, logs_dir):
    """
    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation 
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 

        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)


    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        if args_.split:

            learners_ensemble =\
            get_split_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                domain_disc=args_.domain_disc,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary
            )
        else:

            learners_ensemble =\
                get_learners_ensemble(
                    n_learners=args_.n_learners,
                    client_type=CLIENT_TYPE[args_.method],
                    name=args_.experiment,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.lr,
                    input_dim=args_.input_dimension,
                    output_dim=args_.output_dimension,
                    n_rounds=args_.n_rounds,
                    seed=args_.seed,
                    mu=args_.mu,
                    n_gmm=args_.n_gmm,
                    embedding_dim=args_.embedding_dimension,
                    hard_cluster=args_.hard_cluster,
                    binary=args_.binary,
                    phi_model=args.phi_model
                )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        if CLIENT_TYPE[args_.method] == "conceptEM_tune" and "train" in logs_dir:

            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=True,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )
        else:
            
            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=args_.locally_tune_clients,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )

        clients_.append(client) 

    return clients_

def get_data_iterator(args_, root_path, logs_dir):
    """
    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_concept_drift(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_concept_drift(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation 
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 
       
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)



    return train_iterators, val_iterators, test_iterators, client_types, feature_types

def get_data_iterator_for_store_data(args_,data_indexes,root_path, logs_dir):
    """

    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    data_indexes = data_indexes,
                    test = True,
                    test_num = 3
                )
        else: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes=data_indexes
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 
       
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)



    return train_iterators, val_iterators, test_iterators, client_types, feature_types

def get_data_iterator_for_store_data_rotate_images(args_,data_indexes,root_path,rotate_degrees, logs_dir):
    """

    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    data_indexes = data_indexes,
                    rotate_degrees=rotate_degrees,
                    test = True,
                    test_num = 3
                )
        else: 
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    data_indexes=data_indexes,
                    rotate_degrees=rotate_degrees
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 

        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)



    return train_iterators, val_iterators, test_iterators, client_types, feature_types



def init_clients_for_store_history(args_, last_data_indexes, current_data_indexes,root_path, logs_dir):
    """
    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")


    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
       
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )


            last_train_iterators = train_iterators
            last_val_iterators = val_iterators
            last_test_iterators = test_iterators

        else: 

            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes = current_data_indexes
                )
            last_train_iterators, last_val_iterators, last_test_iterators, last_client_types, last_feature_types =\
                get_cifar10C_loaders_for_store_history(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes = last_data_indexes
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 
        
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)


    print("===> Initializing clients..")
    clients_ = []
    
    for task_id, (train_iterator, val_iterator, test_iterator,last_train_iterator, last_val_iterator, last_test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators,last_train_iterators, last_val_iterators, last_test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        if args_.split:

            learners_ensemble =\
            get_split_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                domain_disc=args_.domain_disc,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary
            )
        else:

            learners_ensemble =\
                get_learners_ensemble(
                    n_learners=args_.n_learners,
                    client_type=CLIENT_TYPE[args_.method],
                    name=args_.experiment,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.lr,
                    input_dim=args_.input_dimension,
                    output_dim=args_.output_dimension,
                    n_rounds=args_.n_rounds,
                    seed=args_.seed,
                    mu=args_.mu,
                    n_gmm=args_.n_gmm,
                    embedding_dim=args_.embedding_dimension,
                    hard_cluster=args_.hard_cluster,
                    binary=args_.binary,
                    phi_model=args.phi_model
                )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        if CLIENT_TYPE[args_.method] == "conceptEM_tune" and "train" in logs_dir:
          
            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=True,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )
        else:

            client = get_client_for_store_history(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator, 
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                last_train_iterator=last_train_iterator, 
                last_val_iterator=last_val_iterator,
                last_test_iterator=last_test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=args_.locally_tune_clients,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )

        clients_.append(client) 

    return clients_


def rot_120deg_init_50_client_store_history_simple2_iid_auto_clst_num\
                (args_, last_data_indexes, current_data_indexes,rotate_degrees,root_path,logs_dir,data_root_path,cluster_num):
    """

    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")


    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = cluster_num,
                    rotate_degrees=rotate_degrees
                )

            
            last_train_iterators = train_iterators
            last_val_iterators = val_iterators
            last_test_iterators = test_iterators
            last_client_types = client_types
            last_feature_types = feature_types

        else: 

            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes = current_data_indexes,
                    rotate_degrees = rotate_degrees
                )
            last_train_iterators, last_val_iterators, last_test_iterators, last_client_types, last_feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes = last_data_indexes,
                    rotate_degrees= rotate_degrees-120
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 
        
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)


    print("===> Initializing clients..")
    clients_ = []

    for task_id, (train_iterator, val_iterator, test_iterator,last_train_iterator, last_val_iterator, last_test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators,last_train_iterators, last_val_iterators, last_test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        if args_.split:

            learners_ensemble =\
            get_split_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                domain_disc=args_.domain_disc,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary
            )
        else:
 
            learners_ensemble =\
                get_learners_ensemble(
                    n_learners=cluster_num,
                    client_type=CLIENT_TYPE[args_.method],
                    name=args_.experiment,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.lr,
                    input_dim=args_.input_dimension,
                    output_dim=args_.output_dimension,
                    n_rounds=args_.n_rounds,
                    seed=args_.seed,
                    mu=args_.mu,
                    n_gmm=args_.n_gmm,
                    embedding_dim=args_.embedding_dimension,
                    hard_cluster=args_.hard_cluster,
                    binary=args_.binary,
                    phi_model=args.phi_model
                )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        if CLIENT_TYPE[args_.method] == "conceptEM_tune" and "train" in logs_dir:
           
            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=True,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )
        else:

            client = get_client_for_store_history_version_2(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator, 
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                last_train_iterator=last_train_iterator, 
                last_val_iterator=last_val_iterator,
                last_test_iterator=last_test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=args_.locally_tune_clients,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                last_data_type = last_client_types[task_id],
                last_feature_type= last_feature_types[task_id],
                class_number = class_number
            )

        clients_.append(client) 

    return clients_


def rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept\
                (args_, last_data_indexes, current_data_indexes,rotate_degrees,root_path,logs_dir,data_root_path,cluster_num,test_num):
    """

    initialize clients from data folders

    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")


    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]

    if LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path: 
            
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = test_num,
                    rotate_degrees=rotate_degrees
                )


            last_train_iterators = train_iterators
            last_val_iterators = val_iterators
            last_test_iterators = test_iterators
            last_client_types = client_types
            last_feature_types = feature_types

        else: 



            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    data_indexes = current_data_indexes,
                    rotate_degrees = rotate_degrees
                )
            last_train_iterators, last_val_iterators, last_test_iterators, last_client_types, last_feature_types =\
                get_cifar10C_loaders_for_store_history_rotate_images(
                    root_path=data_root_path,
                    batch_size=args_.bz,
                    is_validation=args_.validation, 
                    data_indexes = last_data_indexes,
                    rotate_degrees= rotate_degrees-120
                )
    elif LOADER_TYPE[args_.experiment] == 'tiny-imagenet-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_imagenetC_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/tiny-imagenet-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'cifar100-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 1
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar100-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'fmnist-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_fmnistC_loaders(
                    root_path='./data/fmnist-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'airline':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/airline/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'elec':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/elec/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    elif LOADER_TYPE[args_.experiment] == 'powersupply':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_airline_loaders(
                    root_path='./data/powersupply/all_data',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )
    else: 
        
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_loaders(
                type_=LOADER_TYPE[args_.experiment],
                root_path=root_path,
                batch_size=args_.bz,
                is_validation=args_.validation
            )
        client_types = [0] * len(train_iterators)


    print("===> Initializing clients..")
    clients_ = []

    for task_id, (train_iterator, val_iterator, test_iterator,last_train_iterator, last_val_iterator, last_test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators,last_train_iterators, last_val_iterators, last_test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        if args_.split:

            learners_ensemble =\
            get_split_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                domain_disc=args_.domain_disc,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary
            )
        else:

            learners_ensemble =\
                get_learners_ensemble(
                    n_learners=cluster_num,
                    client_type=CLIENT_TYPE[args_.method],
                    name=args_.experiment,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.lr,
                    input_dim=args_.input_dimension,
                    output_dim=args_.output_dimension,
                    n_rounds=args_.n_rounds,
                    seed=args_.seed,
                    mu=args_.mu,
                    n_gmm=args_.n_gmm,
                    embedding_dim=args_.embedding_dimension,
                    hard_cluster=args_.hard_cluster,
                    binary=args_.binary,
                    phi_model=args.phi_model
                )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        if CLIENT_TYPE[args_.method] == "conceptEM_tune" and "train" in logs_dir:
           
            client = get_client(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator,
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=True,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                class_number = class_number
            )
        else:
  
            client = get_client_for_store_history_version_2(
                client_type=CLIENT_TYPE[args_.method],
                learners_ensemble=learners_ensemble,
                q=args_.q,
                train_iterator=train_iterator, 
                val_iterator=val_iterator,
                test_iterator=test_iterator,
                last_train_iterator=last_train_iterator,
                last_val_iterator=last_val_iterator,
                last_test_iterator=last_test_iterator,
                logger=logger,
                local_steps=args_.local_steps,
                tune_locally=args_.locally_tune_clients,
                data_type = client_types[task_id],
                feature_type = feature_types[task_id],
                last_data_type = last_client_types[task_id],
                last_feature_type= last_feature_types[task_id],
                class_number = class_number
            )

        clients_.append(client) 

    return clients_


def check_whether_clients_concept_shifts(clients):
    """

    Parameters
    ----------
    clients

    Returns
    -------
    shift_set,clean_set
    """
    LID_list = []
    for client in clients:
        output = client.get_output() 
 
        LID = client.get_LID(output, output)
        LID_list.append(LID)



    gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=args.seed).fit(
        np.array(LID_list).reshape(-1,1))  
    labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(LID_list).reshape(-1, 1))

    clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]


    shift_set = np.where(labels_LID_accumulative != clean_label)[0]

    clean_set = np.where(labels_LID_accumulative == clean_label)[0]
    return shift_set,clean_set

def check_whether_clients_concept_shifts_accuracy_based(clients):
    """

    Parameters
    ----------
    clients

    Returns
    -------
    shift_set,clean_set
    """
    shift_set = []
    clean_set = []
    for i, client in enumerate(clients):
        last_data_losses, last_data_accuracies = client.get_last_data_all_models_loss_and_accuracy()
        current_data_losses, current_data_accuracies = client.get_current_data_all_models_loss_and_accuracy()

        last_data_minimal_loss_model_index = np.argmin(last_data_losses)
        current_data_minimal_loss_model_index = np.argmin(current_data_losses)
        
        last_data_maximal_accuracy_model_index = np.argmax(last_data_accuracies)
        current_data_maximal_accuracy_model_index = np.argmax(current_data_accuracies)

 

        if last_data_minimal_loss_model_index == current_data_minimal_loss_model_index \
                and last_data_maximal_accuracy_model_index == current_data_maximal_accuracy_model_index:
           
            clean_set.append(i)
        else:
            
            shift_set.append(i)
    return shift_set, clean_set

def check_whether_clients_concept_shifts_cluster_center_based(clients,cluster_centers):
    """

    Parameters
    ----------
    clients

    Returns
    -------
    shift_set,clean_set
    """
    shift_set = []
    clean_set = []
    for i, client in enumerate(clients):
      
        last_prototype = client.get_last_val_iterator_output_prototype()
        current_prototype = client.get_current_val_iterator_output_prototype()

  
        last_prototype = np.array(last_prototype)
        current_prototype = np.array(current_prototype)


        if last_prototype.ndim >1:
          
            last_prototype = last_prototype.flatten()

        if current_prototype.ndim >1:
         
            current_prototype = current_prototype.flatten()
      
        last_predicted_cluster = predict_cluster(last_prototype, cluster_centers)
        current_predicted_cluster = predict_cluster(current_prototype, cluster_centers)

        print("client:",i)
        print("last_predicted_cluster",last_predicted_cluster)
        print("current_predicted_cluster",current_predicted_cluster)
        if last_predicted_cluster == current_predicted_cluster :
            clean_set.append(i)
        else:
           
            shift_set.append(i)
    return shift_set, clean_set


def get_real_shift_clients_set(clients):
    shift_label_list = []
    for client in clients:
        shift_label = client.get_real_shift_label()
        shift_label_list.append(shift_label)


    shift_labels = np.array(shift_label_list)


    real_clean_set = np.where(shift_labels == 0)[0]
    real_shift_set = np.where(shift_labels == 1)[0]

    return  real_shift_set.tolist(),real_clean_set.tolist()

def get_shift_clients_prediction_accuracy(shift_set, clean_set, real_shift_set, real_clean_set):


    correct_predictions = (
        len(set(shift_set) & set(real_shift_set)) +  
        len(set(clean_set) & set(real_clean_set))    
    )


    total_clients = len(shift_set) + len(clean_set)


    prediction_accuracy = correct_predictions / total_clients if total_clients > 0 else 0.0

    return prediction_accuracy

def get_shift_clients_precision(shift_set, real_shift_set):


    true_positive = (
        len(set(shift_set) & set(real_shift_set))   
    )


    true_positive_plus_false_positive = len(real_shift_set)

    precision = true_positive / true_positive_plus_false_positive if true_positive_plus_false_positive > 0 else 0.0
    return precision

def get_shift_clients_recall(shift_set, real_shift_set):

    true_positive = (
        len(set(shift_set) & set(real_shift_set))   
    )


    true_positive_plus_false_nagative = len(shift_set)


    recall = true_positive / true_positive_plus_false_nagative if true_positive_plus_false_nagative > 0 else 0.0
    return recall

def set_clients_concept_shift_flag(clients,shift_set):
    for i,client in enumerate(clients):
        if i in shift_set:
            client.set_concept_shift_flag(1)
        else:
            client.set_concept_shift_flag(0)

def update_clients_train_iterator_and_other_attribute(clients):
    for client in clients:
        if client.concept_shift_flag==0:
          
            merged_train_iterator, merged_val_iterator, merged_test_iterator = \
                client.get_merge_last_and_current_train_iterators()
            client.update_train_iterator_and_other_attributes(merged_train_iterator, merged_val_iterator,
                                                              merged_test_iterator)
        else:
      
            client.update_train_iterator_and_other_attributes(client.current_train_iterator, client.current_val_iterator,
                                                              client.current_test_iterator)


def rotate_120degree_update_clients_train_iterator_and_other_attribute(clients,rotate_degrees):
    """

    Parameters
    ----------
    clients
    rotate_degrees

    Returns
    -------

    """
    for client in clients:
        if client.concept_shift_flag == 0:
           
            merged_train_iterator, merged_val_iterator, merged_test_iterator = \
                client.rotate_120_get_merge_last_and_current_train_iterators(rotate_degrees=rotate_degrees)
            client.update_train_iterator_and_other_attributes(merged_train_iterator, merged_val_iterator,
                                                              merged_test_iterator)
        else:   
    
            client.update_train_iterator_and_other_attributes(client.current_train_iterator, client.current_val_iterator,
                                                              client.current_test_iterator)

def get_clients_output_prototype_version1(clients):

    clients_output_prototype = []
    for client in clients:
        output_prototype = client.get_current_val_iterator_output_prototype()
        clients_output_prototype.append(output_prototype)

    return clients_output_prototype

def determine_cluster_number(clients_output_prototype):

    X = np.array(clients_output_prototype)

    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)  
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    cluster_number = silhouette_scores.index(max(silhouette_scores))
    cluster_number += 2 
    return cluster_number,silhouette_scores


def determine_cluster_number_return_centers(clients_output_prototype):

    X = np.array(clients_output_prototype)


    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1) 

    silhouette_scores = []
    best_kmeans = None  

    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)


        if best_kmeans is None or score > max(silhouette_scores[:-1]):
            best_kmeans = kmeans

    cluster_number = silhouette_scores.index(max(silhouette_scores)) + 2  
    cluster_centers = best_kmeans.cluster_centers_  

    return cluster_number, silhouette_scores, cluster_centers

def predict_cluster(client_prototype, cluster_centers):




    distances = np.linalg.norm(cluster_centers - client_prototype, axis=1)


    cluster_index = np.argmin(distances)

    return cluster_index

def run_experiment(args_):
    torch.manual_seed(args_.seed)



    data_dir = get_data_dir('cifar10-c-60_client-simple2-iid-4concept-change-name-version2')
    data_root_path = './data/cifar10-c-60_client-simple2-iid-4concept-change-name-version2'

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")

    train_num = 60  

  
    numbers = list(range(30)) 

    numbers = numbers * 2

    random.shuffle(numbers)

    initial_data_indexes = numbers  
    current_data_indexes = copy.deepcopy(initial_data_indexes)


    cluster_num = 2

    clients = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_,
                                                                                    last_data_indexes=initial_data_indexes,
                                                                                    current_data_indexes=initial_data_indexes,
                                                                                    rotate_degrees=0,
                                                                                    root_path=os.path.join(data_dir,
                                                                                                           "train"),
                                                                                    logs_dir=os.path.join(logs_dir,
                                                                                                          "train"),
                                                                                    data_root_path=data_root_path,
                                                                                    cluster_num=cluster_num,
                                                                                    test_num=4
                                                                                                )


    print("==> Test Clients initialization..")

    test_clients_0degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_,
                                            last_data_indexes=initial_data_indexes,
                                             current_data_indexes=initial_data_indexes,
                                             root_path=os.path.join(data_dir, "test"),
                                             logs_dir=os.path.join(logs_dir, "test"),
                                             rotate_degrees=0,
                                             data_root_path=data_root_path,
                                             cluster_num=cluster_num,
                                             test_num=4
                                             )


    test_clients_120degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_, last_data_indexes=initial_data_indexes,
                                             current_data_indexes=initial_data_indexes,
                                             root_path=os.path.join(data_dir, "test"),
                                             logs_dir=os.path.join(logs_dir, "test"),
                                             rotate_degrees=120,
                                             data_root_path=data_root_path,
                                             cluster_num=cluster_num,
                                             test_num=4
                                             )

    test_clients_240degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_, last_data_indexes=initial_data_indexes,
                                             current_data_indexes=initial_data_indexes,
                                             root_path=os.path.join(data_dir, "test"),
                                             logs_dir=os.path.join(logs_dir, "test"),
                                             rotate_degrees=240,
                                             data_root_path=data_root_path,
                                             cluster_num=cluster_num,
                                             test_num=4
                                             )

    test_clients = test_clients_0degree+test_clients_120degree+test_clients_240degree

    # Initialize per-client test history (slot concepts) if requested
    if getattr(args_, "eval_all_past_concepts", False):
        for c in clients:
            try:
                c._test_history = [copy.deepcopy(c.test_iterator)]
            except Exception:
                c._test_history = []

    # Optional objective assignment (does not change training logic by default)
    if hasattr(args_, "objective_assignment") and args_.objective_assignment:
        assign_objectives(clients, args_.objective_assignment, seed=args_.seed)
        assign_objectives(test_clients, args_.objective_assignment, seed=args_.seed)

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    # Optional fixed 0° global test iterator (evaluation only)
    global_test_iterator = None
    if getattr(args_, "global_eval", False):
        try:
            from torch.utils.data import ConcatDataset
            from torch.utils.data import DataLoader as TorchDataLoader
            global_test_dataset = ConcatDataset(
                [copy.deepcopy(c.test_iterator.dataset) for c in test_clients_0degree]
            )
            global_test_iterator = TorchDataLoader(
                global_test_dataset,
                batch_size=args_.bz,
                shuffle=False,
                drop_last=False,
            )
            print("==> Global test iterator initialized (fixed baseline, 0° rotation)")
        except Exception as e:
            print(f"[WARN] Failed to build global test iterator: {e}")
            global_test_iterator = None


    if args_.split:
 
        global_learners_ensemble = \
        get_split_learners_ensemble(
            n_learners=args_.n_learners,
            client_type=CLIENT_TYPE[args_.method],
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            embedding_dim=args_.embedding_dimension,
            n_gmm=args_.n_gmm,
            domain_disc=args_.domain_disc,
            hard_cluster=args_.hard_cluster,
            binary=args_.binary
        )
    else:
        global_learners_ensemble = \
            get_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                embedding_dim=args_.embedding_dimension,
                n_gmm=args_.n_gmm,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary,
                phi_model=args.phi_model
            )


    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]


    T = args_.T
    prediction_accuracy_list = []
    precision_list = []
    recall_list = []
    for t in range(T):
        print("==> time slot {} starts..".format(t))
        if t!=0:
            print("==> Clients initialization..")

            
            if t % 3 == 0:
                rotate_degrees = 0
                cluster_num = 4
                
                torch.manual_seed(torch.seed())
                last_data_indexes = copy.deepcopy(current_data_indexes)  
                current_data_indexes = torch.randperm(train_num)  
                torch.manual_seed(args_.seed)

            elif t % 3 == 1:
                rotate_degrees = 120

                if t == 1:
                    cluster_num = 3
                    torch.manual_seed(torch.seed())
                    last_data_indexes = copy.deepcopy(current_data_indexes)  
                    range1 = list(range(0, 15)) 
                    range2 = list(range(15, 30))  
                    range3 = list(range(30, 45)) 
                    
                    numbers = []
                    numbers.extend(range1)  
                    numbers.extend(range2)  
                    numbers.extend(range3)  
                    
                    numbers.extend(random.choices(range1, k=20 - len(range1)))  
                    numbers.extend(random.choices(range2, k=20 - len(range2)))  
                    numbers.extend(random.choices(range3, k=20 - len(range3)))  
                    
                    random.shuffle(numbers)
                    current_data_indexes = numbers  
                    torch.manual_seed(args_.seed)

                else:
                    cluster_num = 4
                    torch.manual_seed(torch.seed())
                    last_data_indexes = copy.deepcopy(current_data_indexes)  
                    current_data_indexes = torch.randperm(train_num)  
                    torch.manual_seed(args_.seed)

            elif t % 3 == 2:
                rotate_degrees = 240
                cluster_num = 4
                torch.manual_seed(torch.seed())
                last_data_indexes = copy.deepcopy(current_data_indexes)  
                current_data_indexes = torch.randperm(train_num)  
                torch.manual_seed(args_.seed)

            clients = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_,
                                                                            last_data_indexes=last_data_indexes,
                                                                            current_data_indexes=current_data_indexes,
                                                                            rotate_degrees=rotate_degrees,
                                                                            root_path=os.path.join(data_dir, "train"),
                                                                            logs_dir=os.path.join(logs_dir, "train"),
                                                                            data_root_path=data_root_path,
                                                                            cluster_num=cluster_num,
                                                                            test_num=4)


            for client in clients:
                for learner_id, learner in enumerate(global_learners_ensemble): 
                    copy_model(client.learners_ensemble[learner_id].model, learner.model)



            print("==> Test Clients initialization..")
            test_clients_0degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_,
                                                                                                         last_data_indexes=initial_data_indexes,
                                                                                                         current_data_indexes=initial_data_indexes,
                                                                                                         root_path=os.path.join(data_dir,"test"),
                                                                                                         logs_dir=os.path.join(logs_dir,"test"),
                                                                                                         rotate_degrees=0,
                                                                                                         data_root_path=data_root_path,
                                                                                                         cluster_num=cluster_num,
                                                                                                         test_num = 4)

            test_clients_120degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_,
                                                                                                           last_data_indexes=initial_data_indexes,
                                                                                                           current_data_indexes=initial_data_indexes,
                                                                                                           root_path=os.path.join(data_dir,"test"),
                                                                                                           logs_dir=os.path.join(logs_dir,"test"),
                                                                                                           rotate_degrees=120,
                                                                                                           data_root_path=data_root_path,
                                                                                                           cluster_num=cluster_num,
                                                                                                           test_num = 4)

            test_clients_240degree = rot_120deg_init_60_client_store_history_simple2_iid_auto_clst_num_balance_concept(args_,
                                                                                                           last_data_indexes=initial_data_indexes,
                                                                                                           current_data_indexes=initial_data_indexes,
                                                                                                           root_path=os.path.join(data_dir,"test"),
                                                                                                           logs_dir=os.path.join(logs_dir,"test"),
                                                                                                           rotate_degrees=240,
                                                                                                           data_root_path=data_root_path,
                                                                                                           cluster_num=cluster_num,
                                                                                                           test_num = 4)

            test_clients = test_clients_0degree + test_clients_120degree + test_clients_240degree


            for client in test_clients:
                for learner_id, learner in enumerate(global_learners_ensemble):  
                    copy_model(client.learners_ensemble[learner_id].model, learner.model)

            
            if args_.split:

                new_global_learners_ensemble = \
                    get_split_learners_ensemble(
                        n_learners=cluster_num,
                        client_type=CLIENT_TYPE[args_.method],
                        name=args_.experiment,
                        device=args_.device,
                        optimizer_name=args_.optimizer,
                        scheduler_name=args_.lr_scheduler,
                        initial_lr=args_.lr,
                        input_dim=args_.input_dimension,
                        output_dim=args_.output_dimension,
                        n_rounds=args_.n_rounds,
                        seed=args_.seed,
                        mu=args_.mu,
                        embedding_dim=args_.embedding_dimension,
                        n_gmm=args_.n_gmm,
                        domain_disc=args_.domain_disc,
                        hard_cluster=args_.hard_cluster,
                        binary=args_.binary
                    )
            else:
                new_global_learners_ensemble = \
                    get_learners_ensemble(
                        n_learners=cluster_num,
                        client_type=CLIENT_TYPE[args_.method],
                        name=args_.experiment,
                        device=args_.device,
                        optimizer_name=args_.optimizer,
                        scheduler_name=args_.lr_scheduler,
                        initial_lr=args_.lr,
                        input_dim=args_.input_dimension,
                        output_dim=args_.output_dimension,
                        n_rounds=args_.n_rounds,
                        seed=args_.seed,
                        mu=args_.mu,
                        embedding_dim=args_.embedding_dimension,
                        n_gmm=args_.n_gmm,
                        hard_cluster=args_.hard_cluster,
                        binary=args_.binary,
                        phi_model=args.phi_model
                    )
            
            for learner_id, learner in enumerate(global_learners_ensemble):  
                copy_model(new_global_learners_ensemble[learner_id].model, learner.model)

            global_learners_ensemble = new_global_learners_ensemble

            
            clients_output_prototype = get_clients_output_prototype_version1(clients)
            cluster_num, silhouette_scores,cluster_centers = \
                determine_cluster_number_return_centers(clients_output_prototype)

            # print("Cluster Number:", cluster_num)
            # print("silhouette_scores", silhouette_scores)
            
            with open('./logs/{}/determine_cluster_number-{}-{}-{}.txt'.format(args_.experiment, args_.method, args_.gamma, args_.suffix), 'a+') as f:

                f.write('{}'.format(cluster_num))
                f.write('\n')
                f.write('{}'.format(silhouette_scores))
                f.write('\n')


            shift_set, clean_set = check_whether_clients_concept_shifts_cluster_center_based(clients, cluster_centers)
            # print("shift_set:", shift_set)
            # print("clean_set:", clean_set)


            set_clients_concept_shift_flag(clients, shift_set)


            real_shift_set, real_clean_set = get_real_shift_clients_set(clients)
            # print("real_shift_set:",real_shift_set)
            # print("real_clean_set:",real_clean_set)

            # Optional objective-aware action logging (Stage 2: evaluation only).
            # This records, per client and time slot, whether the original
            # pipeline treated the client as "reset" or "merge", grouped by
            # objective label. It does NOT change any routing behavior.
            if getattr(args_, "objective_aware", False):
                try:
                    obj_logs_dir = os.path.join("logs", args_.experiment)
                    os.makedirs(obj_logs_dir, exist_ok=True)
                    objective_log_path = os.path.join(
                        obj_logs_dir,
                        f"objective-actions-{args_.method}-{args_.gamma}-{args_.suffix}.txt",
                    )
                    with open(objective_log_path, "a+") as fobj:
                        fobj.write(f"time_slot={t}\n")
                        for idx, client in enumerate(clients):
                            # In the original pipeline, concept_shift_flag drives
                            # whether the client is effectively "reset" or "merge".
                            action = "reset" if getattr(client, "concept_shift_flag", 0) else "merge"
                            objective = getattr(client, "objective", "N/A")
                            # We do not log drift types here to keep Stage 1 logic unchanged.
                            fobj.write(f"{idx},{objective},{action}\n")
                except Exception:
                    pass

            rotate_120degree_update_clients_train_iterator_and_other_attribute(clients,rotate_degrees=rotate_degrees)

            # Append current test iterator to history if requested (evaluation only)
            if getattr(args_, "eval_all_past_concepts", False):
                for c in clients:
                    try:
                        c._test_history.append(copy.deepcopy(c.test_iterator))
                    except Exception:
                        pass

            prediction_accuracy = get_shift_clients_prediction_accuracy(shift_set=shift_set,clean_set=clean_set,
                                                                        real_shift_set=real_shift_set,real_clean_set=real_clean_set)
            prediction_accuracy_list.append(prediction_accuracy)
            # print("shift client prediction accuracy:",prediction_accuracy)

            precision = get_shift_clients_precision(shift_set=shift_set,real_shift_set=real_shift_set)
            precision_list.append(precision)
            # print("precision:",precision)

            recall = get_shift_clients_recall(shift_set=shift_set,real_shift_set=real_shift_set)
            recall_list.append(recall)
            # print("recall:",recall)

            # Lightweight per-slot summary (Stage 1, objective-agnostic).
            # This mirrors the modified repo's slot-summary file but only
            # logs non-objective-aware quantities.
            try:
                slot_logs_dir = os.path.join("logs", args_.experiment)
                os.makedirs(slot_logs_dir, exist_ok=True)
                slot_summary_path = os.path.join(
                    slot_logs_dir,
                    f"slot-summary-{args_.method}-{args_.gamma}-{args_.suffix}.txt",
                )
                with open(slot_summary_path, "a+") as fsum:
                    fsum.write(f"time_slot={t}\n")
                    fsum.write(f"rotate_degrees={rotate_degrees}\n")
                    fsum.write(f"cluster_num={cluster_num}\n")
                    fsum.write(f"n_clients={len(clients)}\n")
                    fsum.write(f"n_shift={len(shift_set)}\n")
                    fsum.write(f"n_clean={len(clean_set)}\n")
                    fsum.write(f"n_real_shift={len(real_shift_set)}\n")
                    fsum.write(f"n_real_clean={len(real_clean_set)}\n")
                    fsum.write(f"prediction_accuracy={float(prediction_accuracy):.4f}\n")
                    fsum.write(f"precision={float(precision):.4f}\n")
                    fsum.write(f"recall={float(recall):.4f}\n")
                    fsum.write("-" * 80 + "\n")
            except Exception:
                # Slot summary logging is best-effort only.
                pass




        aggregator =\
            get_aggregator(
                aggregator_type=aggregator_type,
                clients=clients,
                global_learners_ensemble=global_learners_ensemble,
                lr_lambda=args_.lr_lambda,
                lr=args_.lr,
                q=args_.q,
                mu=args_.mu,
                communication_probability=args_.communication_probability,
                sampling_rate=args_.sampling_rate, 
                log_freq=args_.log_freq,
                global_train_logger=global_train_logger,
                global_test_logger=global_test_logger,
                test_clients=test_clients,
                verbose=args_.verbose,
                seed=args_.seed,
                experiment = args_.experiment,
                method = args_.method,
                suffix = args_.suffix,
                split = args_.split,
                domain_disc=args_.domain_disc,
                em_step=args_.em_step
            )

        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        pre_action = 0

        while current_round <= args_.n_rounds:

      
            if pre_action == 0:
                aggregator.mix(diverse=False)
            else:
                aggregator.mix(diverse=False)

            C = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]
            n_learner = aggregator.n_learners 
            cluster_label_weights = [[0] * C for _ in range(n_learner)]
            cluster_weights = [0 for _ in range(n_learner)]
            global_flags = [[] for _ in range(n_learner)]


            if 'shakespeare' not in args_.experiment:
                with open('./logs/{}/sample-weight-{}-{}.txt'.format(args_.experiment, args_.method, args_.suffix), 'w') as f:
                    for client_index, client in enumerate(clients):
           
                        for i in range(len(client.train_iterator.dataset.targets)):
                            if args_.method == 'FedSoft':
                                f.write('{},{},{}, {}\n'.format(client.data_type, client.train_iterator.dataset.targets[i], client.feature_types[i], aggregator.clusters_weights[client_index]))
                            else:

                                f.write('{},{},{}, {}\n'.format(client.data_type, client.train_iterator.dataset.targets[i], client.feature_types[i], client.samples_weights.T[i]))

                            for j in range(len(cluster_label_weights)):
                                cluster_weights[j] += client.samples_weights[j][i]
                        f.write('\n')
            else:
                for client_index, client in enumerate(clients):
                    for i in range(len(client.train_iterator.dataset.targets)): 
                        for j in range(len(cluster_label_weights)): 
                                cluster_weights[j] += client.samples_weights[j][i] 
 
            with open('./logs/{}/mean-I-{}-{}-{}.txt'.format(args_.experiment, args_.method, args_.gamma, args_.suffix), 'a+') as f:
                mean_Is = torch.zeros((len(clients),))
                clusters = torch.zeros((len(clients),))
                client_types = torch.zeros((len(clients),))
                for i, client in enumerate(clients):
                    
                    mean_Is[i] = client.mean_I
                    client_types[i] = client.data_type
                    
                f.write('{}'.format(mean_Is))
                f.write('\n')
            with open('./logs/{}/cluster-weights-{}-{}-{}.txt'.format(args_.experiment, args_.method, args_.gamma, args_.suffix), 'a+') as f:
                
                f.write('{}'.format(cluster_weights))
                f.write('\n')

            for client in clients:
                client_labels_learner_weights = client.labels_learner_weights
                for j in range(len(cluster_label_weights)):
                    for k in range(C):
                        
                        cluster_label_weights[j][k] += client_labels_learner_weights[j][k]
            for j in range(len(cluster_label_weights)):
                for i in range(len(cluster_label_weights[j])):
                    if cluster_label_weights[j][i] < 1e-8:
                        cluster_label_weights[j][i] = 1e-8
                cluster_label_weights[j] = [i / sum(cluster_label_weights[j]) for i in cluster_label_weights[j]]

          
            for client in clients:
                client.update_labels_weights(cluster_label_weights)
    
            for client in test_clients:
                print(client.mean_I, client.cluster, torch.nonzero(client.cluster==torch.max(client.cluster)).squeeze())

            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round

        
        if "save_dir" in args_:
            save_dir = os.path.join(args_.save_dir)

            os.makedirs(save_dir, exist_ok=True)
            aggregator.save_state(save_dir)

        with open('./logs/{}/drift-prediction-result-{}-{}-{}.txt'.format(
                args_.experiment, args_.method, args_.gamma, args_.suffix), 'a+') as f:
            f.write("prediction_accuracy_list:")
            f.write('{}'.format(prediction_accuracy_list))
            f.write('\n')
            f.write("precision_list:")
            f.write('{}'.format(precision_list))
            f.write('\n')
            f.write("recall_list:")
            f.write('{}'.format(recall_list))
            f.write('\n')

        # Additional evaluation metrics (evaluation only; no effect on training)
        eval_metrics = {}

        # Drift detection summary across slots
        eval_metrics["detection_stats"] = {
            "accuracy": sample_stats(prediction_accuracy_list),
            "precision": sample_stats(precision_list),
            "recall": sample_stats(recall_list),
        }

        # Global evaluation on fixed 0° test iterator
        if 'global_test_iterator' in locals() and global_test_iterator is not None:
            global_accs = []
            for c in clients:
                acc = evaluate_iterator_accuracy(c.learners_ensemble, global_test_iterator)
                if acc is not None:
                    global_accs.append(acc)
            if global_accs:
                eval_metrics["global_eval_mean"] = float(np.mean(global_accs))
                eval_metrics["global_eval_std"] = float(np.std(global_accs))

        # All past concepts: per-client mean over test history (final slot)
        if getattr(args_, "eval_all_past_concepts", False):
            per_client_means = []
            for c in clients:
                history = getattr(c, "_test_history", None)
                if not history:
                    continue
                accs = []
                for it in history:
                    a = evaluate_iterator_accuracy(c.learners_ensemble, it)
                    if a is not None:
                        accs.append(a)
                if accs:
                    per_client_means.append(float(np.mean(accs)))
            if per_client_means:
                eval_metrics["all_past_concepts_mean"] = float(np.mean(per_client_means))
                eval_metrics["all_past_concepts_std"] = float(np.std(per_client_means))

        # TASKS@FINAL (Rotation): mean accuracy at final slot per rotation
        try:
            def _mean_acc_over(test_list):
                vals = []
                for tc in test_list:
                    a = evaluate_iterator_accuracy(global_learners_ensemble, tc.test_iterator)
                    if a is not None:
                        vals.append(a)
                return float(np.mean(vals)) if vals else float("nan")

            rot0 = _mean_acc_over(test_clients_0degree)
            rot120 = _mean_acc_over(test_clients_120degree)
            rot240 = _mean_acc_over(test_clients_240degree)
            eval_metrics["tasks_final_rotation"] = {
                "rot_0": rot0,
                "rot_120": rot120,
                "rot_240": rot240,
            }
        except Exception:
            pass

        # Return basic drift metrics plus extra evaluation summaries so
        # the multi-seed wrapper can aggregate across seeds.
        eval_metrics.update({
            "prediction_accuracy_list": prediction_accuracy_list,
            "precision_list": precision_list,
            "recall_list": recall_list,
        })

        # Objective-grouped evaluation metrics and grouped accuracy.
        # These are computed purely from final-slot models and do NOT change training,
        # routing, or drift detection behavior. They are active whenever objectives
        # have been assigned (objective_assignment), regardless of whether the
        # routing itself is objective-aware.
        if getattr(args_, "objective_assignment", None):
            try:
                # Per-client final accuracies on local current, old (if available), and global.
                p_curr, g_curr, all_curr = [], [], []
                p_global, g_global, all_global = [], [], []
                p_old, g_old = [], []

                # Evaluate on fixed global iterator if available.
                def _eval_global(client):
                    if 'global_test_iterator' in locals() and global_test_iterator is not None:
                        return evaluate_iterator_accuracy(client.learners_ensemble, global_test_iterator)
                    return None

                for c in clients:
                    obj = getattr(c, "objective", "G")
                    acc_curr = evaluate_iterator_accuracy(c.learners_ensemble, c.test_iterator)
                    acc_old = evaluate_iterator_accuracy(
                        c.learners_ensemble,
                        getattr(c, "last_test_iterator", None),
                    )
                    acc_global = _eval_global(c)

                    if acc_curr is not None:
                        all_curr.append(acc_curr)
                        if obj == "P":
                            p_curr.append(acc_curr)
                        else:
                            g_curr.append(acc_curr)
                    if acc_old is not None:
                        if obj == "P":
                            p_old.append(acc_old)
                        else:
                            g_old.append(acc_old)
                    if acc_global is not None:
                        all_global.append(acc_global)
                        if obj == "P":
                            p_global.append(acc_global)
                        else:
                            g_global.append(acc_global)

                # Define objective-aware scores:
                #  - P: current local accuracy.
                #  - G: global test accuracy (baseline).
                p_scores = p_curr
                g_scores = g_global

                # Stats for JSON metrics.
                stats_block = eval_metrics.setdefault("stats", {})
                if p_scores:
                    stats_block["p_scores"] = sample_stats(p_scores)
                if g_scores:
                    stats_block["g_scores"] = sample_stats(g_scores)
                if all_curr:
                    stats_block["overall_current"] = sample_stats(all_curr)
                if all_global:
                    stats_block["overall_global"] = sample_stats(all_global)
                if g_old:
                    # G-clients' old local accuracy (useful for forgetting),
                    # but we do NOT compute the mixed "Objective-aware OLD" metric.
                    stats_block["G_old_local"] = sample_stats(g_old)

                eval_metrics["p_scores"] = p_scores
                eval_metrics["g_scores"] = g_scores

                # Human-readable grouped accuracy summary (final slot).
                logs_base_path = os.path.join("logs", args_.experiment)
                os.makedirs(logs_base_path, exist_ok=True)
                objective_summary_path = os.path.join(
                    logs_base_path,
                    f"objective-grouped-accuracy-{args_.method}-{args_.suffix}.txt",
                )
                with open(objective_summary_path, "w") as f:
                    f.write("=" * 80 + "\n")
                    f.write("LEGEND\n")
                    f.write("-" * 80 + "\n")
                    f.write("Personalization (P) Clients: P-clients on current local test (final slot).\n")
                    f.write("Generalization (G) Clients: G-clients on previous (old) local test (final slot; if available).\n")
                    f.write("Overall (All Clients): All clients on current local test (final slot).\n")
                    f.write("GLOBAL TEST (Global P/G): P/G-clients on the fixed global test set (baseline).\n")
                    f.write("P_on_local / G_on_local: P/G-clients on current local test.\n")
                    f.write("P_on_global / G_on_global: P/G-clients on the fixed global test set.\n")
                    f.write("Overall Global (All Clients): All clients on the fixed global test set.\n")
                    f.write("TASKS@FINAL (Rotation): Mean accuracy at final slot on rotation-stratified test sets (0°, 120°, 240°).\n")
                    f.write("ALL PAST CONCEPTS @ FINAL: Mean accuracy across all stored past test iterators.\n")
                    f.write("Objective-aware OLD (P:local, G:old) is intentionally omitted here.\n")
                    f.write("=" * 80 + "\n\n")
                    f.write("FINAL TEST ACCURACY GROUPED BY OBJECTIVE\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"n_clients={len(clients)}\n")
                    nP = sum(1 for c in clients if getattr(c, "objective", "G") == "P")
                    nG = sum(1 for c in clients if getattr(c, "objective", "G") == "G")
                    f.write(f"nP={nP}, nG={nG}\n\n")

                    def _write_block(title, vals):
                        if not vals:
                            return
                        s = sample_stats(vals)
                        f.write(f"{title}\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"  Mean:   {s['mean']:.4f}\n")
                        f.write(f"  Std:    {s['std']:.4f}\n")
                        f.write(f"  Min:    {s['min']:.4f}\n")
                        f.write(f"  Median: {s['median']:.4f}\n")
                        f.write(f"  Max:    {s['max']:.4f}\n\n")

                    _write_block("P Clients (current local)", p_curr)
                    _write_block("G Clients (current local)", g_curr)
                    _write_block("G Clients (old local)", g_old)
                    _write_block("All Clients (current local)", all_curr)
                    _write_block("P Clients (global test)", p_global)
                    _write_block("G Clients (global test)", g_global)
                    _write_block("All Clients (global test)", all_global)

                    # TASKS@FINAL (Rotation) using rotation-specific test clients
                    try:
                        def _mean_acc_over(test_list):
                            vals = []
                            for tc in test_list:
                                a = evaluate_iterator_accuracy(global_learners_ensemble, tc.test_iterator)
                                if a is not None:
                                    vals.append(a)
                            return float(np.mean(vals)) if vals else float("nan")

                        rot0 = _mean_acc_over(test_clients_0degree)
                        rot120 = _mean_acc_over(test_clients_120degree)
                        rot240 = _mean_acc_over(test_clients_240degree)
                        f.write("TASKS@FINAL (Rotation)\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"(rotation=0°)   Mean: {rot0:.4f}\n")
                        f.write(f"(rotation=120°) Mean: {rot120:.4f}\n")
                        f.write(f"(rotation=240°) Mean: {rot240:.4f}\n\n")
                    except Exception:
                        pass

                    # ALL PAST CONCEPTS @ FINAL (if histories were stored)
                    if getattr(args_, "eval_all_past_concepts", False):
                        all_hist, p_hist, g_hist = [], [], []
                        for c in clients:
                            history = getattr(c, "_test_history", None)
                            if not history:
                                continue
                            accs = []
                            for it in history:
                                a = evaluate_iterator_accuracy(c.learners_ensemble, it)
                                if a is not None:
                                    accs.append(a)
                            if not accs:
                                continue
                            mean_acc = float(np.mean(accs))
                            all_hist.append(mean_acc)
                            if getattr(c, "objective", "G") == "P":
                                p_hist.append(mean_acc)
                            else:
                                g_hist.append(mean_acc)

                        if all_hist:
                            f.write("ALL PAST CONCEPTS @ FINAL\n")
                            f.write("-" * 80 + "\n")
                            _write_block("All clients", all_hist)
                            _write_block("P clients", p_hist)
                            _write_block("G clients", g_hist)

            except Exception:
                # Objective-aware evaluation is best-effort and must never
                # affect the core training or baseline metrics.
                pass

        return eval_metrics


def _copy_text_logs_to_seed_dirs(args, seed_logs_dir):
    """
    Convenience helper: copy original TXT logs from ./logs/<experiment>/*
    into the structured experiments/.../seed_x/logs directory.
    This does not affect training or metrics; it only mirrors outputs.
    """
    root_logs_dir = os.path.join("logs", args.experiment)
    if not os.path.isdir(root_logs_dir):
        return

    patterns = [
        # Global / task-level summaries
        f"results-{args.method}-{args.suffix}.txt",
        f"test-results-{args.method}-{args.suffix}.txt",
        # Drift detection summary
        f"drift-prediction-result-{args.method}-{args.gamma}-{args.suffix}.txt",
        # Cluster / weighting diagnostics (classic FedDAA logs)
        f"sample-weight-{args.method}-{args.suffix}.txt",
        f"mean-I-{args.method}-{args.gamma}-{args.suffix}.txt",
        f"cluster-weights-{args.method}-{args.gamma}-{args.suffix}.txt",
        f"determine_cluster_number-{args.method}-{args.gamma}-{args.suffix}.txt",
        # Per-slot summary (Stage 1, objective-agnostic)
        f"slot-summary-{args.method}-{args.gamma}-{args.suffix}.txt",
        # Objective-aware logs (Stage 2, evaluation only)
        f"objective-actions-{args.method}-{args.gamma}-{args.suffix}.txt",
        f"objective-grouped-accuracy-{args.method}-{args.suffix}.txt",
    ]
    for name in patterns:
        src = os.path.join(root_logs_dir, name)
        if os.path.exists(src):
            dst = os.path.join(seed_logs_dir, name)
            try:
                shutil.copy2(src, dst)
            except Exception:
                # Mirroring is best-effort; ignore copy failures.
                pass


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    seeds = parse_seed_argument(getattr(args, "seeds", None), args.seed)
    logger = ExperimentLogger(args, seeds)
    seed_metrics = []
    for seed in seeds:
        seed_args = copy.deepcopy(args)
        seed_args.seed = seed
        # Redirect logs into experiments/.../seed_x/logs
        seed_logs_dir = logger.seed_logs_dir(seed)
        seed_args.logs_dir = str(seed_logs_dir)
        metrics = run_experiment(seed_args)
        seed_metrics.append(metrics or {})

        # Persist per-seed metrics JSON under experiments/.../seed_x/metrics
        logger.write_metrics(seed, metrics or {})

        # Mirror original TXT logs into experiments/.../seed_x/logs
        _copy_text_logs_to_seed_dirs(seed_args, seed_logs_dir)

        # Simple drift-detection diagnostics plot (per seed, optional)
        try:
            if metrics:
                acc_list = metrics.get("prediction_accuracy_list", []) or []
                prec_list = metrics.get("precision_list", []) or []
                rec_list = metrics.get("recall_list", []) or []
                if any(len(lst) > 0 for lst in (acc_list, prec_list, rec_list)):
                    x = list(range(len(acc_list or prec_list or rec_list)))
                    fig, ax = plt.subplots(figsize=(6, 4))
                    if acc_list:
                        ax.plot(x, acc_list, marker="o", label="accuracy")
                    if prec_list:
                        ax.plot(x, prec_list, marker="s", label="precision")
                    if rec_list:
                        ax.plot(x, rec_list, marker="^", label="recall")
                    ax.set_xlabel("Time slot t")
                    ax.set_ylabel("Score")
                    ax.set_title("Drift detection over time")
                    ax.set_ylim(0.0, 1.0)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.legend()
                    logger.write_plot(seed, "drift_detection.png", fig)
                    plt.close(fig)
        except Exception:
            # Plotting is best-effort only
            pass

    aggregate = aggregate_summary(seed_metrics)

    logger.save_summary({
        "per_seed": seed_metrics,
        "aggregate": aggregate,
    })

    print("\n=== FedORA Summary ===")
    print(f"Seeds: {seeds}")
    if 'p_scores' in aggregate:
        print(f"P-Score mean±std: {aggregate['p_scores']['mean']:.4f} ± {aggregate['p_scores']['std']:.4f}")
    if 'g_scores' in aggregate:
        print(f"G-Score mean±std: {aggregate['g_scores']['mean']:.4f} ± {aggregate['g_scores']['std']:.4f}")
    if 'forgetting' in aggregate:
        print(f"Forgetting ratio mean±std: {aggregate['forgetting']['mean']:.4f} ± {aggregate['forgetting']['std']:.4f}")
