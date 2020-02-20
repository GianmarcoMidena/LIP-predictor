import numpy as np
import pandas as pd
import logging
import math
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from Bio.SeqUtils import seq1
from multiprocessing import Pool
from functools import partial
from configparser import ConfigParser

from lip_feature_extractor import LipFeatureExtractor as FeatureExtractor
from stacking_classifier import StackingClassifier
from logistic_regression import LogisticRegression

pd.set_option('mode.chained_assignment', 'raise')


class LipClassifier:
    """
    Classifier of Linear Interacting Peptides in protein structures (PDBs).

    Parameters
    ----------
    key_columns : list of str
        Ordered list of key columns that identifies records in data.
    features_subset : set of str, optional
        The subset of features the classifier has to take account of.
    min_residue_interaction_distance_threshold, max_residue_interaction_distance_threshold : float, optional
        Minimum and maximum thresholds for distance between residues.
    min_window_size, max_window_size : int, optional
        Minimum and maximum sizes for moving average windows.
    min_regularization_strength, max_regularization_strength: int, optional
        Minimum and maximum regularization strengths for logistic regression models.
    random_state : int, optional
        Seed for the random number generator.

    Attributes
    ----------
    _model : object
        Fitted classifier.
    _feature_estractor : object
        Optimized feature extractor.
    """

    def __init__(self, key_columns, features_subset=None, min_residue_interaction_distance_threshold=None,
                 max_residue_interaction_distance_threshold=None,
                 min_window_size=None, max_window_size=None, min_regularization_strength=None,
                 max_regularization_strength=None, random_state=None):
        self._model = None
        self._feature_estractor = None
        self._key_columns = key_columns
        self._features_subset = features_subset
        self._min_residue_interaction_distance_threshold = min_residue_interaction_distance_threshold
        self._max_residue_interaction_distance_threshold = max_residue_interaction_distance_threshold
        self._min_window_size = min_window_size
        self._max_window_size = max_window_size
        self._min_regularization_strength = min_regularization_strength
        self._max_regularization_strength = max_regularization_strength
        self._random_state = random_state

    @property
    def base_learners(self):
        """Get the fitted base learners."""
        return self._model.base_learners

    @property
    def meta_learner(self):
        """Get the fitted meta learner."""
        return self._model.meta_learner

    @property
    def feature_extractor(self):
        """Get the optimized feature estractor"""
        return self._feature_estractor

    @property
    def features_subset(self):
        """Get the subset of features the classifier takes account of."""
        return self._features_subset

    @staticmethod
    def from_params(params, key_columns, random_state=None):
        """Create classifier from an existent configuration.

        Loads all parameters of base learners, meta learner and feature extractor.

        Parameters
        ----------
        params : ConfigParser
        key_columns : list of str
            Ordered list of key columns that identifies records in data.
        random_state : int, optional
            Seed for the random number generator.

        Returns
        -------
        self : object
        """
        logging.info("Loading learned parameters")

        classifier = LipClassifier(key_columns=key_columns, random_state=random_state)
        ensemble_model = StackingClassifier()
        meta_learner = classifier._create_meta_learner()
        classifier._features_subset = set(params.sections()).difference(['GLOBAL'])
        for column in classifier._features_subset:
            base_learner = LogisticRegression(features_subset=[column])
            base_learner.intercept = float(params[column]['base_intercept'])
            base_learner.coefficients = [float(params[column]['base_coefficient'])]
            base_learner.set_classes([0, 1], np.int64)
            ensemble_model.add_base_learner(base_learner)
            meta_learner.add_coefficient(float(params[column]['ensemble_coefficient']))
        meta_learner.intercept = float(params['GLOBAL']['ensemble_intercept'])
        meta_learner.set_classes([0, 1], np.int64)
        ensemble_model.meta_learner = meta_learner
        classifier._model = ensemble_model

        window_size = int(params['GLOBAL']['window_size'])
        residue_interaction_distance_threshold = float(params['GLOBAL']
                                                       ['residue_interaction_distance_threshold'])
        classifier._feature_estractor = FeatureExtractor(window_size, residue_interaction_distance_threshold,
                                                         classifier._key_columns, classifier._features_subset)
        return classifier

    def fit(self, training_set):
        """Train an ensemble classifier of LIPs in PDB structures.

        Tunes hyperparameters (window size for moving average,
        residue interactions distance threshold, regularization strength)
        using cross validation.

        At the end, it train the ensemble with best hyperparameters found
        on the full dataset.

        Parameters
        ----------
        training_set : pd.DataFrame, shape = [n_samples, n_features + 1]
            Training vector, where n_samples is the number of samples,
            n_features is the number of PDB features and 1 is the number of labels.

        Returns
        -------
        self : object
        """
        logging.info("Training parameters")

        best_model = self._tune_hyperparameters(training_set)

        training_set = self._extract_features(training_set, self._feature_estractor)
        X, y = training_set.drop(columns='lip'), training_set['lip']
        best_model.fit(X, y)
        self._model = best_model
        return self

    def predict(self, X):
        """Predict LIPs in PDB structures using a trained ensemble classifier.

        Parameters
        ----------
        X : pd.DataFrame(), shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of PDB features.

        Returns
        -------
        predictions : pd.DataFrame
        """
        logging.info("Predicting LIPs")

        predictions = pd.DataFrame()
        key_columns = list(filter(lambda key: key in X.columns, self._key_columns))
        X = X.sort_values(key_columns, axis=0, ascending=True)
        X.loc[X['insertion_code'] == '-', 'insertion_code'] = ''
        X['residue_name'] = X['residue_name'].apply(seq1)
        predictions['residue_id'] = (X['model_id'].astype(str) + '/' + X['chain_id'].astype(str) + '/'
                                     + X['residue_position'].astype(str) + '/' + X['insertion_code'].astype(str)
                                     + '/' + X['residue_name'].astype(str))
        pred_class, pred_proba = self._model.predict(X)
        predictions['lip_score'] = pred_proba
        predictions['lip_class'] = pred_class
        return predictions

    def _tune_hyperparameters(self, dataset_original):
        best_model = None
        best_score = 0
        best_configuration = None
        best_feature_extractor = FeatureExtractor(self._min_window_size,
                                                  self._min_residue_interaction_distance_threshold,
                                                  key_columns=self._key_columns, features_subset=self._features_subset)

        n_residue_interaction_distance_thresholds = len(self._get_residue_interaction_distance_thresholds())
        n_window_sizes = len(self._get_window_sizes())
        n_regularization_strengths_base = n_regularization_strengths_meta = len(self._get_regularization_strengths())
        scores_history = np.zeros(shape=(n_residue_interaction_distance_thresholds, n_window_sizes,
                                         n_regularization_strengths_base, n_regularization_strengths_meta))
        for d, residue_interaction_distance_threshold in enumerate(self._get_residue_interaction_distance_thresholds()):
            for w, window_size in enumerate(self._get_window_sizes()):
                feature_extractor = FeatureExtractor(window_size, residue_interaction_distance_threshold,
                                                     self._key_columns, self._features_subset)
                dataset = self._extract_features(dataset_original.copy(), feature_extractor)
                for r1, regularization_strength_base in enumerate(self._get_regularization_strengths()):
                    for r2, regularization_strength_meta in enumerate(self._get_regularization_strengths()):
                        if self._features_subset is None:
                            features_set = dataset.columns
                        else:
                            features_set = list(self._features_subset.intersection(dataset.columns))
                            if len(features_set) == 0:
                                features_set = dataset.columns
                        model = self._create_ensemble_model(features_set,
                                                            regularization_strength_base=regularization_strength_base,
                                                            regularization_strength_meta=regularization_strength_meta)
                        f_score, recall, precision, roc_auc = self._evaluate_ensemble(model, dataset)
                        logging.info("residue_interaction_distance_threshold={}, window_size={}, "
                                     "regularization_strength_base={:.1E}, regularization_strength_meta={:.1E}, "
                                     "f_Score={:.3}, recall={:.3}, precision={:.3}, roc_auc={:.3}"
                                     .format(feature_extractor.residue_interaction_distance_threshold,
                                             feature_extractor.window_size,
                                             regularization_strength_base, regularization_strength_meta,
                                             f_score, recall, precision, roc_auc))
                        scores_history[d, w, r1, r2] = f_score

                        if f_score > best_score:
                            best_score = f_score
                            best_configuration = \
                                {'residue_interaction_distance_threshold': residue_interaction_distance_threshold,
                                 'window_size': window_size,
                                 'regularization_strength_base': regularization_strength_base,
                                 'regularization_strength_meta': regularization_strength_meta,
                                 'f_score': f_score, 'recall': recall, 'precision': precision, 'roc_auc': roc_auc}
                            best_model = model
                            best_feature_extractor = feature_extractor
                        else:
                            if r2 > 0 and scores_history[d, w, r1, r2] <= scores_history[d, w, r1, r2-1]:
                                # no improvements in the last increment of regularization strength for meta learner
                                # w.r.t. the previous one
                                break
                    if r1 > 0 and scores_history[:, :, r1].max() <= scores_history[:, :, r1-1].max():
                        # no improvements in the last increment of regularization strength for base learners
                        # w.r.t. the previous one
                        break
                if w > 0 and scores_history[:, w].max() <= scores_history[:, w-1].max():
                    # no improvements in the last 'window size' increment w.r.t. the previous one
                    break
            if d > 0 and scores_history[d].max() <= scores_history[d-1].max():
                # no improvements in the last distance threshold increment w.r.t. the previous one
                break
        if best_configuration is not None:
            logging.info("BEST PERFORMANCE: residue_interaction_distance_threshold={}, window_size={}, "
                         "regularization_strength_base={:.1E}, regularization_strength_meta={:.1E}, "
                         "f_Score={:.3}, recall={:.3}, precision={:.3}, roc_auc={:.3}"
                         .format(best_configuration['residue_interaction_distance_threshold'],
                                 best_configuration['window_size'],
                                 best_configuration['regularization_strength_base'],
                                 best_configuration['regularization_strength_meta'],
                                 best_configuration['f_score'], best_configuration['recall'],
                                 best_configuration['precision'], best_configuration['roc_auc']))
        self._feature_estractor = best_feature_extractor
        return best_model

    def _get_residue_interaction_distance_thresholds(self):
        return np.arange(start=self._min_residue_interaction_distance_threshold,
                         stop=self._max_residue_interaction_distance_threshold + .5,
                         step=.5)

    def _get_window_sizes(self):
        step = 10
        return range(self._min_window_size, self._max_window_size + 1, step)

    def _get_regularization_strengths(self):
        n_samples = 1 + math.log10(self._max_regularization_strength) - math.log10(self._min_regularization_strength)
        return np.geomspace(start=self._min_regularization_strength, stop=self._max_regularization_strength,
                            num=n_samples)\
                 .tolist()

    def _evaluate_ensemble(self, model, dataset):
        f_scores = []
        recall_scores = []
        precision_scores = []
        roc_auc_scores = []
        X, y = dataset.drop(columns='lip'), dataset['lip']
        with Pool() as pool:
            ys = pool.starmap_async(partial(self._evaluate_ensemble_in_split, X=X, y=y, model=model),
                                    self._k_fold(X, y, n_splits=10))
            for y_pred, y_test in ys.get():
                f_scores += [f1_score(y_test, y_pred)]
                recall_scores += [recall_score(y_test, y_pred)]
                precision_scores += [precision_score(y_test, y_pred)]
                roc_auc_scores += [roc_auc_score(y_test, y_pred)]
        recall = np.array(recall_scores).mean()
        precision = np.array(precision_scores).mean()
        f_score = np.array(f_scores).mean()
        roc_auc = np.array(roc_auc_scores).mean()
        return f_score, recall, precision, roc_auc

    @staticmethod
    def _evaluate_ensemble_in_split(train, test, X, y, model):
        X_train, y_train = X.iloc[train], y.iloc[train]
        X_test, y_test = X.iloc[test], y.iloc[test]
        model.fit(X_train, y_train)
        y_pred, _ = model.predict(X_test)
        return y_pred, y_test

    def _create_base_learners(self, columns, regularization_strength=1e3):
        return [LogisticRegression(regularization_strength=regularization_strength, features_subset=[column],
                                   class_weight='balanced', random_state=self._random_state)
                for column in columns]

    def _create_meta_learner(self, regularization_strength=1e3):
        return LogisticRegression(regularization_strength=regularization_strength, class_weight='balanced',
                                  random_state=self._random_state)

    def _create_ensemble_model(self, columns, regularization_strength_base, regularization_strength_meta):
        splitter = GroupKFold(n_splits=10)
        return StackingClassifier(self._create_base_learners(columns, regularization_strength_base),
                                  meta_classifier=self._create_meta_learner(regularization_strength_meta),
                                  cv_splitter=splitter)

    def _k_fold(self, X, y, n_splits):
        groups = self._calc_groups(X)
        splitter = GroupKFold(n_splits=n_splits)
        return splitter.split(X, y, groups=groups)

    def _leave_one_out(self, X, y):
        groups = self._calc_groups(X)
        splitter = LeaveOneGroupOut()
        return splitter.split(X, y, groups=groups)

    @staticmethod
    def _create_ensemble_model_empty():
        splitter = GroupKFold(n_splits=10)
        return StackingClassifier(cv_splitter=splitter)

    @staticmethod
    def _calc_groups(data):
        return pd.Categorical(data['structure_id']).codes

    @staticmethod
    def _extract_features(data, feature_extractor):
        data = feature_extractor.aggregate_interactions_by_residue(data)
        data = feature_extractor.calc_inter_intra_chain_interactions_ratios(data)
        return feature_extractor.rolling_windows(data)
