# Adapted from https://github.com/SquareResearchCenter-AI/BEExAI
import os
import openml
import yaml

smallest_datasets = [ # smaller size datasets (10)
    "analcatdata_authorship",  #8
    "balance-scale",  #1
    "car",  #3
    "cmc",  #2
    "eucalyptus",  #6
    "mfeat-morphological",  #7
    "segment",  #9
    "steel-plates-fault",  #10
    "vehicle",  #4
    "vowel",  #5
]
middle_datasets = [ # middle size datasets (10)
    "dna", #20
    "jungle_chess_2pcs_raw_endgame_complete", #12
    "letter", #13
    "mfeat-pixel", #17
    "mfeat-zernike", #19
    "MiceProtein", #18
    "pendigits", #16
    "semeion", #15
    "splice", #11
    "wall-robot-navigation" #14
]
largest_datasets = [ #larger size datasets (9)
    "cnae-9", #25
    "connect-4", #28
    "first-order-theorem-proving", #26
    "GesturePhaseSegmentationProcessed", #27
    "mfeat-factors", #22
    "mfeat-karhunen", #21
    "mnist_784", #29
    "optdigits", #24
    "texture", #23
]
tabular_multiclass = smallest_datasets #+ middle_datasets + largest_datasets

save_path = "data/csv/"
config_path = "data/config/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(config_path):
    os.makedirs(config_path)
benchmark_suite = openml.study.get_suite(suite_id=99)
for id in benchmark_suite.data:
    dataset = openml.datasets.get_dataset(
        id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    name = dataset.name
    if name in tabular_multiclass:

        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )

        data = X.copy()

        data[dataset.default_target_attribute] = y

        data.to_csv(f"{save_path}/{name}.csv", index=False)

        print(f"Dataset {name} has {X.shape[0]} samples, {X.shape[1]} features and {y.nunique()} classes.")

        config = {
            "path": f"{save_path}/{name}.csv",
            "target_col": dataset.default_target_attribute,
            "cleaned_data_path": f"{save_path}/{name}.csv",
            "task": "classification"
        }

        with open(f"{config_path}/{name}.yaml", "w") as file:
            yaml.dump(config, file)