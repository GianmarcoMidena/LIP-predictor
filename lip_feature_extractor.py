import numpy as np
import pandas as pd
import logging
from os import path
from Bio.PDB import PDBParser, DSSP


class LipFeatureExtractor:
    """
    Feature extractor for identification of Linear Interacting Peptides in protein structures (PDBs).

    Parameters
    ----------
    window_size : int
        Size for moving average windows.
    residue_interaction_distance_threshold : float
        Threshold for distance between residues.
    key_columns : list of str
        Ordered list of key columns that identifies records in data.
    features_subset : set of str, optional
        The subset of features the feature extractor has to take account of.
    """

    INTERACTION_OCCURRENCES_COLUMN_SUFFIX = "_chain_interaction"
    INTER_INTRA_CHAIN_INTERACTION_OCCURRENCES_RATIO_COLUMN_SUFFIX = "inter_intra_chain_interaction_ratio"
    INTERACTION_TYPE_LABELS_BY_KEY = {
        'HBOND': 'hydrogen',
        'VDW': 'van_der_waals',
        'SSBOND': 'disulfide',
        'IONIC': 'salt',
        'PIPISTACK': 'pi_pi_stacking',
        'PICATION': 'pi_cation'
    }
    INTERACTION_SUBTYPE_LABELS_BY_KEY = {
        'MC': 'main',
        'SC': 'side',
        'LIG': 'ligand'
    }

    def __init__(self, window_size, residue_interaction_distance_threshold, key_columns, features_subset=None):
        self._window_size = window_size
        self._residue_interaction_distance_threshold = residue_interaction_distance_threshold
        self._key_columns = key_columns
        self._features_subset = features_subset

    @property
    def window_size(self):
        """Get size for moving average windows."""
        return self._window_size

    @property
    def residue_interaction_distance_threshold(self):
        """Get threshold for distance between residues."""
        return self._residue_interaction_distance_threshold

    def aggregate_interactions_by_residue(self, data):
        """Aggregates interactions by residue.

        Before aggregation, it filters out residue interactions with distance greater than
        ``self._residue_interaction_distance_threshold``.

        For each residue, it calculates the mean distance and energy of its interactions,
        and counts interactions by type (e.g. intra-chain, inter-chain, ...).

        Parameters
        ----------
        data : pd.DataFrame
            A dataset of residue interactions.

        Returns
        -------
        pd.DataFrame
            Returns a dataset of residues.
        """
        original_data = data.copy()
        all_interaction_features = list(filter(lambda d: self.INTERACTION_OCCURRENCES_COLUMN_SUFFIX in d, data.columns))
        if self._features_subset is None:
            features_subset = set(all_interaction_features + ['residue_interaction_distance', 'energy'])
            interaction_features_subset = all_interaction_features
        else:
            interaction_features_subset = list(self._features_subset.intersection(all_interaction_features))
            if len(interaction_features_subset) == 0:
                features_subset = set(all_interaction_features + ['residue_interaction_distance', 'energy'])
                interaction_features_subset = all_interaction_features

        aggregation = {}
        for c in interaction_features_subset:
            aggregation[c] = 'sum'
        for c in features_subset.intersection(['residue_interaction_distance', 'energy']):
            aggregation[c] = 'mean'

        key_columns = list(
            self._features_subset.intersection(['relative_asa', 'secondary_structure'])
                                 .union(self._key_columns + ['lip']).intersection(data.columns))

        new_data = data.loc[data['residue_interaction_distance'] <= self._residue_interaction_distance_threshold]\
                       .set_index(key_columns, drop=True) \
                       .groupby(level=key_columns, sort=False)\
                       .agg(aggregation)\
                       .reset_index(drop=False)

        data = original_data[key_columns].drop_duplicates()\
                                         .set_index(key_columns)\
                                         .join(new_data.set_index(key_columns), sort=False, how='left')\
                                         .reset_index(drop=False)
        data[interaction_features_subset] = data[interaction_features_subset].fillna(0)
        return data

    @classmethod
    def calc_inter_intra_chain_interactions_ratios(cls, data):
        """Calculates different types of inter-/intra-chain residue interactions ratios.

        Parameters
        ----------
        data : pd.DataFrame
            A dataset that contains different types of residue interaction counts.

        Returns
        -------
        pd.DataFrame
            Returns a dataset that contains residue interactions ratios.
        """
        interaction_occurrences_column_names = list(
            filter(lambda c: c.endswith(cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX), data.columns))
        interaction_occurrences_column_name_prefixes = {
            c.replace('intra' + cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX, '')
             .replace('inter' + cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX, '')
            for c in interaction_occurrences_column_names}
        new_cols = pd.DataFrame({
            (c + cls.INTER_INTRA_CHAIN_INTERACTION_OCCURRENCES_RATIO_COLUMN_SUFFIX):
                cls._calc_inter_intra_chain_interaction_occurrences_ratio(c, data)
            for c in interaction_occurrences_column_name_prefixes
        })
        return pd.concat([data, new_cols], axis=1, sort=False)

    def rolling_windows(self, data):
        """Computes a moving average`.

        It uses a window size equals to ``self._window_size`.

        Parameters
        ----------
        data : pd.DataFrame
            A dataset of residues.

        Returns
        -------
        pd.DataFrame
            Returns a dataset of residues.
        """
        old_index = data.index
        data = data.sort_values(['structure_id', 'chain_id', 'residue_position'], ascending=True)
        interaction_occurrences_ratio_columns = \
            list(filter(lambda c: c.endswith(self.INTER_INTRA_CHAIN_INTERACTION_OCCURRENCES_RATIO_COLUMN_SUFFIX),
                        data.columns))
        interaction_occurrences_columns = list(filter(lambda c: c.endswith(self.INTERACTION_OCCURRENCES_COLUMN_SUFFIX),
                                                      data.columns))
        sequential_features_columns = ['secondary_structure'] + interaction_occurrences_ratio_columns + \
                                      interaction_occurrences_columns
        if self._features_subset is not None:
            sequential_features_columns = list(self._features_subset.intersection(sequential_features_columns))
            if len(sequential_features_columns) == 0:
                return data

        data[sequential_features_columns] = \
            data.set_index(['structure_id', 'chain_id'], drop=True) \
                .groupby(level=['structure_id', 'chain_id'], sort=True)[sequential_features_columns] \
                .rolling(window=self._window_size, min_periods=1, center=True).mean().values
        data = self._scale(data, interaction_occurrences_ratio_columns + interaction_occurrences_columns)
        return data.loc[old_index]

    @classmethod
    def extract_features_from_pdb_set(cls, data, pdb_dir):
        """Extract features from a set of pdb files.

        Parameters
        ----------
        data : pd.DataFrame
            Set of pdb residue identifiers.
        pdb_dir : str
            Path to a directory of pdb files.

        Returns
        -------
        pd.DataFrame
            Returns a dataset with pdb features.
        """
        logging.info("Extracting residues features from PDB set")

        unlabeled_data = pd.DataFrame()
        structure_ids = data['structure_id'].unique()
        tot_structures = structure_ids.shape[0]
        for i, structure_id in enumerate(structure_ids):
            chain_ids = data.loc[data['structure_id'] == structure_id, 'chain_id'].unique()
            pdb_file_path = cls._generate_pdb_file_path(structure_id, pdb_dir)
            data_from_single_pdb = cls.extract_features_from_pdb(structure_id, pdb_file_path,
                                                                 chain_ids_subset=chain_ids)
            unlabeled_data = unlabeled_data.append(data_from_single_pdb, ignore_index=True, sort=False)
            if (((i + 1) % 5) == 0) or ((i + 1) == tot_structures):
                logging.info("{}/{} structures processed".format(i + 1, tot_structures))

        labeled_data = cls._label_data(unlabeled_data, data)
        return labeled_data

    @classmethod
    def extract_features_from_pdb(cls, structure_id, pdb_file_path, chain_ids_subset=None):
        """Extract features from a pdb file.

        Parameters
        ----------
        structure_id : str
            A protein structure identifier.
        pdb_file_path : str
            Path to a pdb file.
        chain_ids_subset : list of str, optional
            Set of chain identifiers.

        Returns
        -------
        pd.DataFrame
            Returns a dataset with pdb features.
        """
        dataset = pd.DataFrame()
        model = cls._load_model(structure_id, pdb_file_path)
        dssp = cls._load_DSSP(model, pdb_file_path)

        all_chain_ids = map(lambda chain: chain.get_id(), model.get_chains())
        if chain_ids_subset is None:
            chain_ids = all_chain_ids
        else:
            chain_ids = filter(lambda c: c in chain_ids_subset, all_chain_ids)

        for chain_id in chain_ids:
            for residue in model[chain_id]:
                residue_id = residue.get_id()
                position = residue_id[1]
                insertion_code = residue_id[2] if residue_id[2] != ' ' else '-'
                hetero_field = residue_id[0] if residue_id[0] != ' ' else '-'
                residue_name = residue.get_resname()
                chain_residue_key = (chain_id, residue_id)
                if chain_residue_key in dssp:
                    dssp_info = dssp[chain_residue_key]
                    secondary_structure = dssp_info[2]
                    relative_asa = dssp_info[3]
                else:
                    secondary_structure = np.nan
                    relative_asa = np.nan

                feature_record = {
                    'structure_id': structure_id,
                    'model_id': model.get_id(),
                    'chain_id': chain_id,
                    'residue_position': position,
                    'insertion_code': insertion_code,
                    'hetero_field': hetero_field,
                    'residue_name': residue_name,
                    'secondary_structure': secondary_structure,
                    'relative_asa': relative_asa
                }

                dataset = dataset.append(feature_record, ignore_index=True, sort=False)

        dataset['model_id'] = dataset['model_id'].astype('int')
        dataset['residue_position'] = dataset['residue_position'].astype('int')
        return dataset

    @staticmethod
    def binarize_secondary_structure(data):
        """Transforms ``secondary structure`` feature from categorical to binary.

        It puts a 1 where a secondary structure type is present,
        puts a 0 where a secondary structure type is not present.

        It leaves null values unchanged.

        Parameters
        ----------
        data : pd.DataFrame
            A datset with at least a categorical ``secondary structure`` column.

        Returns
        -------
        pd.DataFrame
            Returns a datset with  at least a binary ``secondary structure`` column.
        """
        if 'secondary_structure' in data:
            data.loc[(~data['secondary_structure'].isna()) & (data['secondary_structure'] != '-'),
                     'secondary_structure'] = 1.
            data.loc[data['secondary_structure'] == '-', 'secondary_structure'] = 0.
        return data

    @classmethod
    def calc_interaction_occurrences(cls, data):
        """Counts different types of residue interactions.

        It counts intra and inter chain interactions by residue.

        It counts hydrogen, van_der_waals, disulfide, salt,
        pi_pi_stacking and pi_cation interactions by residue,
        distinguishing between intra and inter chain interactions.

        It counts main, side chain and ligand interactions by residue,
        distinguishing between intra and inter chain interactions.

        It counts hydrogen, van_der_waals, disulfide, salt,
        pi_pi_stacking and pi_cation interactions by residue,
        distinguishing both between main, side chain and ligand
        and between intra and inter chain interactions.

        Parameters
        ----------
        data : pd.DataFrame
            A datset with at least a categorical ``secondary structure`` column.

        Returns
        -------
        pd.DataFrame
            Returns a datset with  at least a binary ``secondary structure`` column.
        """
        data = cls._calc_intra_chain_interaction_occurrences(data)
        data = cls._calc_inter_chain_interaction_occurrences(data)
        data = cls._calc_interaction_occurrences_by_type(data)
        data = cls._calc_interaction_occurrences_by_location(data)
        data = cls._calc_interaction_occurrences_by_type_and_location(data)
        return data.drop(columns=['interaction_type', 'location', 'other_chain_id'])

    @classmethod
    def _calc_inter_intra_chain_interaction_occurrences_ratio(cls, column, data):
        inter_chain_interaction_occurrences = data[column + 'inter' + cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX]
        intra_chain_interaction_occurrences = data[column + 'intra' + cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX]
        ratio = inter_chain_interaction_occurrences / np.maximum(1, intra_chain_interaction_occurrences)
        return np.minimum(1, ratio)

    @staticmethod
    def _scale(data, columns):
        values = data[columns] - data[columns].min()
        denominator = values.max()
        denominator[denominator == 0] = 1
        data[columns] = values.div(denominator)
        return data

    @classmethod
    def _calc_intra_chain_interaction_occurrences(cls, data):
        not_null_chains = ~data['chain_id'].isna() & ~data['other_chain_id'].isna()
        same_chains = (data['chain_id'] == data['other_chain_id'])
        data['intra' + cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX] = 1 * (not_null_chains & same_chains)
        return data

    @classmethod
    def _calc_inter_chain_interaction_occurrences(cls, data):
        not_null_chains = ~data['chain_id'].isna() & ~data['other_chain_id'].isna()
        different_chains = (data['chain_id'] != data['other_chain_id'])
        data['inter' + cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX] = 1 * (not_null_chains & different_chains)
        return data

    @classmethod
    def _calc_interaction_occurrences_by_type(cls, data):
        interaction_occurrences_by_type = pd.DataFrame({
            "{}_{}{}".format(type_val, inter_intra, cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX):
                1 * ((data['interaction_type'] == type_key) &
                     data[inter_intra + cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX])
            for type_key, type_val in cls.INTERACTION_TYPE_LABELS_BY_KEY.items()
            for inter_intra in ['inter', 'intra']
        })
        return pd.concat([data, interaction_occurrences_by_type], sort=False, axis=1)

    @classmethod
    def _calc_interaction_occurrences_by_location(cls, data):
        interaction_occurrences_by_location = pd.DataFrame({
            "{}_{}{}".format(loc_val, inter_intra, cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX):
                1 * ((data['location'] == loc_key) &
                     data[inter_intra + cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX])
            for loc_key, loc_val in cls.INTERACTION_SUBTYPE_LABELS_BY_KEY.items()
            for inter_intra in ['inter', 'intra']
        })
        return pd.concat([data, interaction_occurrences_by_location], sort=False, axis=1)

    @classmethod
    def _calc_interaction_occurrences_by_type_and_location(cls, data):
        interaction_occurrences_by_type_and_location = pd.DataFrame({
            "{}_{}_{}{}".format(type_val, loc_val, inter_intra, cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX):
                1 * ((data['interaction_type'] == type_key) &
                     (data['location'] == loc_key) &
                     data[inter_intra + cls.INTERACTION_OCCURRENCES_COLUMN_SUFFIX])
            for type_key, type_val in cls.INTERACTION_TYPE_LABELS_BY_KEY.items()
            for loc_key, loc_val in cls.INTERACTION_SUBTYPE_LABELS_BY_KEY.items()
            for inter_intra in ['inter', 'intra']
        })
        return pd.concat([data, interaction_occurrences_by_type_and_location], sort=False, axis=1)

    @staticmethod
    def _label_data(unlabeled_data, dataset):
        labeled_data = unlabeled_data.assign(lip=0)
        chain_data = dataset.loc[~dataset['start_residue_position'].isna(), ['structure_id', 'chain_id',
                                                                             'start_residue_position',
                                                                             'end_residue_position']]
        for _, structure_id, chain, start_position, end_position in chain_data.itertuples():
            lip_data = ((labeled_data['structure_id'] == structure_id) &
                        (labeled_data['chain_id'] == chain) &
                        labeled_data['residue_position'].ge(start_position) &
                        labeled_data['residue_position'].le(end_position))
            labeled_data.loc[lip_data, 'lip'] = 1
        return labeled_data

    @staticmethod
    def _generate_pdb_file_path(structure_id, pdb_dir):
        if pdb_dir is None:
            pdb_file_path = None
        else:
            pdb_file_path = path.join(pdb_dir, structure_id + '.pdb')
        return pdb_file_path

    @staticmethod
    def _load_model(structure_id, pdb_file_path):
        parser = PDBParser(PERMISSIVE=1, QUIET=True)
        structure = parser.get_structure(structure_id, pdb_file_path)
        model = structure[0]
        return model

    @staticmethod
    def _load_DSSP(model, pdb_file_path):
        return DSSP(model, pdb_file_path)
