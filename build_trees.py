import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

from src.utils import tree_portfolio, read_rename_df, rows_to_quantiles, unstack_df
from src.constants import Columns, Chars, DataPaths, Parameters


if __name__ == '__main__':
    chars = Chars()
    paths = DataPaths()

    logging.info(f"Loading base characteristics")
    raw_lme_df = read_rename_df(paths.input_data / f"{chars.lme}.csv")
    ret_df = unstack_df(read_rename_df(paths.input_data / f"{paths.returns_file_name}.csv"), paths.returns_file_name)
    rf_factor_df = pd.read_csv(paths.input_data / f"{paths.rf_factor_file_name}.csv", names=[Columns.returns_col])
    rf_factor_df[Columns.returns_col] = rf_factor_df[Columns.returns_col].apply(lambda x: x / 100)

    logging.info(f"Transform base Size feature into quantiles")
    quantile_lme_df = rows_to_quantiles(raw_lme_df.copy())

    logging.info(f"Stack raw Size and Returns variables together")
    merged_df = pd.concat([
        unstack_df(raw_lme_df, Columns.size_col),
        unstack_df(quantile_lme_df, chars.lme),
        ret_df], axis=1)

    logging.info(f"Start building the AP trees given the combinations of features")
    for char_comb in tqdm(chars.combinations_of_chars(k=Parameters.n_chars, exclude_chars=[chars.lme, chars.returns])):
        feature_sequence = [chars.lme] + list(char_comb)
        output_file_name = f"{paths.sep}".join(feature_sequence)
        comb_df = pd.concat([unstack_df(
            rows_to_quantiles(
                read_rename_df(paths.input_data / f"{c}.csv")
            ), name=c
        ) for c in char_comb], axis=1)
        comb_df = pd.concat([merged_df, comb_df], axis=1)
        comb_df = comb_df[~(comb_df.isna().sum(axis=1).astype(bool))]
        comb_df.reset_index(inplace=True)

        # In the original implementation, the year start from 1964, thus excluding 1963 from the dataset
        comb_df[Columns.date_col] = pd.to_datetime(comb_df[Columns.date_col])
        comb_df = comb_df[comb_df[Columns.date_col].apply(lambda x: x.year != 1963)]

        # Start building the tree portfolios
        portfolio = tree_portfolio(comb_df, feature_sequence,
                                   n_split=Parameters.n_splits, tree_depth=Parameters.tree_depth)
        portfolio = pd.concat(portfolio, axis=1).T.drop_duplicates().T
        portfolio.index.names = [Columns.date_col, Columns.features_col]
        portfolio.columns.names = [Columns.comb_col, Columns.port_col, Columns.node_col]

        # Get returns excess variable
        ret_mask = portfolio.index.get_level_values(Columns.features_col) == Columns.w_returns_col
        portfolio.iloc[ret_mask, :] = portfolio.iloc[ret_mask, :].sub(
            rf_factor_df[Columns.returns_col].tolist(), axis=0)

        # Remove the trees that aare solely based on the single characteristics
        # (all combinations in max port are the same)
        mask_one = (
                portfolio.columns.get_level_values(Columns.port_col) ==
                f"{Columns.port_col}{Columns.col_sep}{Parameters.tree_depth}")
        mask_two = portfolio.columns.get_level_values(Columns.comb_col).isin(
            ['_'.join([v] * Parameters.tree_depth) for v in feature_sequence])
        mask = np.logical_and(mask_one, mask_two)
        portfolio = portfolio.iloc[:, ~mask]
        portfolio.to_pickle(paths.processed_data / f"{output_file_name}.pkl")
