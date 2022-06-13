import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.constants import Chars, DataPaths, Parameters
from src.model import TreeElastic


if __name__ == '__main__':
    chars = Chars()
    paths = DataPaths()
    for tree_file_path in tqdm(paths.processed_data.iterdir()):
        if paths.model_suffix in tree_file_path:
            continue
        tree_portolio = pd.read_pickle(tree_file_path)
        param_grid = {
            'mean_shrinkage': Parameters.mean_shrinkage,
            'ridge_lambda': Parameters.ridge_lambda
        }
        tscv = TimeSeriesSplit(n_splits=3)
        tree_model = TreeElastic(k_min=5, k_max=50)
        cv_tree_model = GridSearchCV(estimator=tree_model, param_grid=param_grid, verbose=1, cv=tscv)
        cv_tree_model.fit(tree_portolio)
        model_output_name = f"{tree_portolio.with_suffix('').name}{paths.sep}{paths.model_suffix}"
        with open(paths.processed_data / model_output_name, 'wb') as f:
            pickle.dump(cv_tree_model, f)
