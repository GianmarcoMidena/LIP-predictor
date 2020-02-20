import pandas as pd
import numpy as np


class ResidueInteractionsExtractor:
    """
    Extractor of all types of non-covalent interactions at atomic level in protein structures (PDBs).
    """

    def __init__(self):
        pass

    @classmethod
    def load_residue_interactions_of_multiple_pdbs(cls, pdb_names, *edges_filepaths_or_buffers):
        """
        Load interactions between residues of one or more protein structures (PDBs).


        Parameters
        ----------
        pdb_names: str
            name of a protein structure
        edges_filepaths_or_buffers: list of str, pathlib.Path, py._path.local.LocalPath or any \
        object with a read() method (such as a file handle or StringIO)
            The string could be a URL. Valid URL schemes include http, ftp, s3, and
            file. For file URLs, a host is expected. For instance, a local file could
            be file://localhost/path/to/table.csv

        Returns
        -------
        pd.DataFrame
            a dataset of residue interactions
        """
        edges = pd.DataFrame()
        for structure_id, edges_file_path in zip(pdb_names, edges_filepaths_or_buffers):
            edges = edges.append(cls.load_residue_interactions(structure_id, edges_file_path), sort=False)
        return edges

    @classmethod
    def load_residue_interactions(cls, pdb_name, edges_filepath_or_buffer):
        """
        Load interactions between residues of a protein structures (PDBs).


        Parameters
        ----------
        pdb_name: str
            name of a protein structure
        edges_filepath_or_buffer: str, pathlib.Path, py._path.local.LocalPath or any \
        object with a read() method (such as a file handle or StringIO)
            The string could be a URL. Valid URL schemes include http, ftp, s3, and
            file. For file URLs, a host is expected. For instance, a local file could
            be file://localhost/path/to/table.csv

        Returns
        -------
        pd.DataFrame
            a dataset of residue interactions
        """
        edges = pd.read_csv(edges_filepath_or_buffer, sep='\t', header=0) \
            .dropna(axis=1, how='all') \
            .rename(columns={'NodeId1': 'residue', 'NodeId2': 'other_residue',
                             'Distance': 'residue_interaction_distance',
                             'Angle': 'angle',
                             'Energy': 'energy',
                             'Donor': 'donor',
                             'Orientation': 'orientation',
                             'Atom1': 'atom', 'Atom2': 'other_atom',
                             'Positive': 'positive',
                             'Cation': 'cation'})

        edges = cls._split('Interaction', to_columns=['interaction_type', 'location'], data=edges, sep=':')
        edges.loc[edges['interaction_type'].isin(['VDW', 'IAC']), 'angle'] = np.nan
        edges = cls._split('location', to_columns=['location', 'other_location'], data=edges, sep='_')

        edges = cls._mirror_edges(edges)

        edges = cls._split('residue', to_columns=['chain_id', 'residue_position', 'insertion_code', 'residue_name'],
                       data=edges, sep=':')
        edges = cls._split('other_residue', to_columns=['other_chain_id', 'other_residue_position', 'other_insertion_code',
                                                    'other_residue_name'], data=edges, sep=':')
        edges.loc[edges['insertion_code'] == '_', 'insertion_code'] = '-'
        edges.loc[edges['other_insertion_code'] == '_', 'other_insertion_code'] = '-'
        edges['residue_position'] = edges['residue_position'].astype('int64')
        edges['other_residue_position'] = edges['other_residue_position'].astype('int64')
        edges['structure_id'] = pdb_name
        return edges

    @staticmethod
    def _mirror_edges(edges):
        edges2 = edges.rename(columns={'residue': 'other_residue', 'other_residue': 'residue',
                                       'atom': 'other_atom', 'other_atom': 'atom',
                                       'location': 'other_location', 'other_location': 'location'})
        return edges.append(edges2, sort=False)

    @staticmethod
    def _split(from_column, to_columns, data, sep):
        swap_name = '_' + from_column + '_'
        data = data.rename(columns={from_column: swap_name})
        data[to_columns] = data[swap_name].str.split(sep, expand=True)
        return data.drop(columns=swap_name)
