"""
LIP predictor

This script allows the user to train and/or test a classifier
of Linear Interacting Peptides in protein structures (PDBs).
"""

import argparse
from os import mkdir, path
from os.path import isdir, isfile, abspath
import logging
import configparser
import pandas as pd
import numpy as np

from lip_feature_extractor import LipFeatureExtractor as FeatureExtractor
from lip_classifier import LipClassifier as Classifier
from residue_interactions_extractor import ResidueInteractionsExtractor as ring


MIN_RESIDUE_INTERACTION_DISTANCE_THRESHOLD = 4.
MAX_RESIDUE_INTERACTION_DISTANCE_THRESHOLD = 10.
MIN_WINDOW_SIZE = 30
MAX_WINDOW_SIZE = 500
MIN_REGULARIZATION_STRENGTH = 10
MAX_REGULARIZATION_STRENGTH = 1e8
SELECTED_FEATURES = {'relative_asa', 'secondary_structure', 'inter_intra_chain_interaction_ratio', 'energy',
                     'residue_interaction_distance'}
KEY_COLUMNS = ['structure_id', 'model_id', 'chain_id', 'residue_position', 'insertion_code', 'residue_name']


def main():
    seed = 3

    _set_up_logger(level=logging.INFO)

    args = _parse_arguments()

    _check_arguments(args)

    if args.configuration:  # an existent configuration is provided
        classifier = _load_configuration(args.configuration, random_state=seed)
    else:
        if args.training_set:  # a training set is provided
            training_set = _prepare_data_for_training(args)
            classifier = _fit_classifier(training_set, random_state=seed)
            _save_configuration(classifier, args)

    if args.pdb_file and args.edges_file:  # a pdb file is provided
        test_set = _prepare_data_for_prediction(classifier, args)
        predictions = classifier.predict(test_set)
        _save_predictions(predictions, args)


def _check_arguments(args):
    if args.configuration:
        if not isfile(args.configuration):
            raise Exception("Attention: '{}' file does not exist!".format(abspath(args.configuration)))
        if not args.pdb_file:
            if not args.edges_file:
                raise Exception("Attention: both PDB and edges files are not provided, "
                                "and they are needed for prediction!")
            else:
                raise Exception("Attention: a PDB file is not provided, "
                                "and it is needed for prediction!")
        else:
            if not args.edges_file:
                raise Exception("Attention: an edges file is not provided, "
                                "and it is needed for prediction!")
            else:
                if not isfile(args.pdb_file):
                    raise Exception("Attention: '{}' file does not exist!".format(abspath(args.pdb_file)))
                if not isfile(args.edges_file):
                    raise Exception("Attention: '{}' file does not exist!".format(abspath(args.edges_file)))
    else:
        if args.training_set:
            if not isfile(args.training_set):
                raise Exception("Attention: '{}' file does not exist!".format(abspath(args.training_set)))
            if args.training_set_type != "AUGMENTED":
                if not args.pdb_dir:
                    if not args.edges_dir:
                        raise Exception("Attention: both PDB and edges directories are not provided, "
                                        "and they are needed for training with a basic dataset!")
                    else:
                        raise Exception("Attention: a PDB directory is not provided, "
                                        "and it is needed for training with a basic dataset!")
                else:
                    if not args.edges_dir:
                        raise Exception("Attention: an edges directory is not provided, "
                                        "and it is needed for training with a basic dataset!")
                    else:
                        if not isdir(args.pdb_dir):
                            raise Exception("Attention: '{}' directory does not exist!".format(abspath(args.pdb_dir)))
                        if not isdir(args.edges_dir):
                            raise Exception("Attention: '{}' directory does not exist!".format(abspath(args.edges_dir)))
        else:
            raise Exception("Attention: neither a configuration file nor a training set are provided, "
                            "so no operations can be executed!")


def _set_up_logger(level):
    logging.basicConfig(format='%(asctime)s %(message)s', level=level)


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pdb_file", help="specify the path to a pdb file for prediction")
    parser.add_argument("-pdb_dir", help="specify the path to a directory with pdb files for training")
    parser.add_argument("-configuration", help="specify the path to a configuration file")
    parser.add_argument("-out_dir", help="specify the path to an output directory to storing predictions "
                                         "and/or a new configuration")
    parser.add_argument("-edges_file", help="specify the path to an edges file")
    parser.add_argument("-edges_dir", help="specify the path to a directory with edges files")
    parser.add_argument("-training_set", help="specify the path to a training set file")
    parser.add_argument("-training_set_type", help="specify a training set type")
    parser.add_argument("-store_augmented_dataset", help="specify that an augmented dataset has to be stored to disk",
                        action="store_true")
    return parser.parse_args()


def _load_configuration(config_file_path, random_state):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return Classifier.from_params(config, key_columns=KEY_COLUMNS, random_state=random_state)


def _prepare_data_for_training(args):
    training_set = _get_training_set(args)
    return _extract_features_for_training(training_set)


def _get_training_set(args):
    if args.training_set_type and args.training_set_type.upper() == "AUGMENTED":
        training_set = pd.read_csv(args.training_set)
    else:  # a BASIC training set is provided
        training_set = _augment_training_set(args)
    return training_set


def _extract_features_for_training(training_set):
    target_column = 'lip'
    other_columns = ['residue_interaction_distance', 'interaction_type', 'location', 'other_chain_id']
    columns = list(SELECTED_FEATURES.union(KEY_COLUMNS)
                                    .union([target_column] + other_columns)
                                    .intersection(training_set.columns))
    training_set = training_set.loc[:, columns]
    training_set = FeatureExtractor.calc_interaction_occurrences(training_set)
    training_set['residue_interaction_distance'] = training_set['residue_interaction_distance'].fillna(0)
    training_set = training_set.dropna()
    return FeatureExtractor.binarize_secondary_structure(training_set)


def _augment_training_set(args):
    basic_dataset = _read_basic_training_set(args.training_set)
    pdb_dir = _get_pdb_directory(args)
    structure_ids = basic_dataset['structure_id'].unique()
    edges = _acquire_edges(structure_ids, args, training=True)
    augmented_dataset = _combine_data(basic_dataset, pdb_dir, edges)
    if args.store_augmented_dataset:
        output_path_dir = _get_output_directory(args)
        augmented_dataset_path = path.join(output_path_dir, "lips_augmented_dataset.csv")
        augmented_dataset.to_csv(augmented_dataset_path, index=False)
        logging.info("Saved augmented dataset at '{}'".format(abspath(augmented_dataset_path)))
    return augmented_dataset


def _read_basic_training_set(training_set_file_path):
    training_set = pd.read_csv(training_set_file_path, sep='\t', header=0) \
                     .rename({'pdb': 'structure_id', 'chain': 'chain_id',
                              'start': 'start_residue_position', 'end': 'end_residue_position'}, axis=1)
    training_set.loc[training_set['start_residue_position'] == 'neg', 'start_residue_position'] = np.nan
    training_set.loc[training_set['end_residue_position'] == 'neg', 'end_residue_position'] = np.nan
    training_set['start_residue_position'] = training_set['start_residue_position'].astype(np.float)
    training_set['end_residue_position'] = training_set['end_residue_position'].astype(np.float)
    training_set['lip'] = ~training_set['start_residue_position'].isna()
    return training_set


def _get_pdb_directory(args):
    if args.pdb_dir:
        pdb_dir = args.pdb_dir
    else:
        pdb_dir = None
    return pdb_dir


def _acquire_edges(structure_ids, args, training=False):
    logging.info("Searching for interactions between residues")

    if training:  # training mode
        if args.edges_dir:  # an edges directory path is provided
            edges_file_suffix = "_edges.txt"
            edges_file_paths = [path.join(args.edges_dir, structure_id + edges_file_suffix)
                                for structure_id in structure_ids]
            return ring.load_residue_interactions_of_multiple_pdbs(structure_ids, *edges_file_paths)
    else:  # prediction mode
        if args.edges_file:  # an edges file path is provided
            return ring.load_residue_interactions(structure_ids[0], args.edges_file)


def _combine_data(basic_dataset, pdb_dir, edges):
    labeled_data = FeatureExtractor.extract_features_from_pdb_set(basic_dataset, pdb_dir)
    key_columns = list(set(KEY_COLUMNS).intersection(labeled_data.columns).intersection(edges.columns))
    data = labeled_data.set_index(key_columns, drop=True) \
        .join(edges.set_index(key_columns), how="left", sort=False) \
        .reset_index(drop=False)
    return _drop_irrelevant_columns(data)


def _drop_irrelevant_columns(data):
    data = data.dropna(how='all', axis=1)
    n_unique_values = data.apply(pd.Series.nunique)
    static_columns = set(n_unique_values[n_unique_values == 1].index)
    undroppable_columns = {'structure_id', 'model_id', 'chain_id', 'residue_position', 'insertion_code',
                           'residue_name', 'lip'}
    data = data.drop(columns=list(static_columns.difference(undroppable_columns)))
    return data


def _get_output_directory(args):
    if args.out_dir:
        output_path_dir = args.out_dir
        if not isdir(output_path_dir):
            mkdir(output_path_dir)
    else:
        output_path_dir = '.'
    return output_path_dir


def _fit_classifier(training_set, random_state):
    classifier = Classifier(KEY_COLUMNS, SELECTED_FEATURES,
                            MIN_RESIDUE_INTERACTION_DISTANCE_THRESHOLD, MAX_RESIDUE_INTERACTION_DISTANCE_THRESHOLD,
                            MIN_WINDOW_SIZE, MAX_WINDOW_SIZE, MIN_REGULARIZATION_STRENGTH, MAX_REGULARIZATION_STRENGTH,
                            random_state=random_state)
    return classifier.fit(training_set)


def _save_configuration(classifier, args):
    config = configparser.ConfigParser()

    base_learners = classifier.base_learners
    meta_learner = classifier.meta_learner
    feature_extractor = classifier.feature_extractor

    learned_params = {
        base_learner.get_feature_name(0): {
            'base_coefficient': base_learner.get_coefficient(0),
            'base_intercept': base_learner.intercept,
            'ensemble_coefficient': ensemble_coefficient
        }
        for base_learner, ensemble_coefficient in zip(base_learners,
                                                      meta_learner.coefficients)
    }
    learned_params['GLOBAL'] = {
        'ensemble_intercept': meta_learner.intercept,
        'window_size': feature_extractor.window_size,
        'residue_interaction_distance_threshold': feature_extractor.residue_interaction_distance_threshold
    }

    config.read_dict(learned_params)
    output_path_dir = _get_output_directory(args)
    parameters_file_path = path.join(output_path_dir, "parameters.ini")
    with open(parameters_file_path, 'w') as configfile:
        config.write(configfile)
    logging.info("Saved learned parameters at '{}'".format(abspath(parameters_file_path)))


def _prepare_data_for_prediction(classifier, args):
    pdb_file_path = args.pdb_file
    structure_id = _extract_file_root_from_path(pdb_file_path)
    edges = _acquire_edges([structure_id], args, training=False)
    return _extract_features_for_prediction(edges, pdb_file_path, structure_id, classifier)


def _extract_features_for_prediction(edges, pdb_file_path, structure_id, classifier):
    logging.info("Extracting residues features from PDB")
    feature_extractor = classifier.feature_extractor
    pdb_data = feature_extractor.extract_features_from_pdb(structure_id, pdb_file_path)
    key_columns = list(set(KEY_COLUMNS).intersection(pdb_data.columns).intersection(edges.columns))
    data = pdb_data.set_index(key_columns, drop=True) \
                   .join(edges.set_index(key_columns), how="left", sort=False) \
                   .reset_index(drop=False)
    other_features = ['residue_interaction_distance', 'interaction_type', 'location', 'other_chain_id']
    if classifier.features_subset is None:
        features_subset = set(KEY_COLUMNS).union(other_features)
    else:
        features_subset = list(classifier.features_subset.union(KEY_COLUMNS)
                                                         .union(other_features)
                                                         .intersection(data.columns))
    data = data[features_subset]
    data = feature_extractor.calc_interaction_occurrences(data)
    data = feature_extractor.binarize_secondary_structure(data)
    data = feature_extractor.aggregate_interactions_by_residue(data)
    data = feature_extractor.calc_inter_intra_chain_interactions_ratios(data)
    return feature_extractor.rolling_windows(data)


def _extract_file_root_from_path(file_path):
    pdb_file_name = path.basename(file_path)
    structure_id = path.splitext(pdb_file_name)[0]
    return structure_id


def _save_predictions(predictions, args):
    structure_id = _extract_file_root_from_path(args.pdb_file)
    output_path_dir = _get_output_directory(args)
    predictions_file_suffix = "_pred.txt"
    predictions_file_path = path.join(output_path_dir, structure_id + predictions_file_suffix)
    predictions.to_csv(predictions_file_path, sep=' ', header=['>' + structure_id, '', ''],
                       float_format='%.3f', index=False)
    logging.info("Saved predictions at '{}'".format(abspath(predictions_file_path)))


if __name__ == '__main__':
    main()
