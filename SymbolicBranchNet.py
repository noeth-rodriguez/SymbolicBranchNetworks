import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
import numpy as np

class SymbolicBranchNet(nn.Module):
    def __init__(
            self,
            teacher_model=None,
            training_w1=True,
            training_w2=False,
            device=None,
            dtype=torch.float,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.hidden_neurons = 0
        self.branch_info = []
        self.in_features = None
        self.out_features = None
        self.bn0 = None
        self.w1 = None
        self.m1 = None
        self.bn1 = None
        self.bn2 = None
        self.w2 = None
        self.m2 = None
        self.training_w1 = training_w1
        self.training_w2 = training_w2
        self.teacher_model = teacher_model

    def forward(self, input: Tensor) -> Tensor:
        x = self.bn0(input)
        if self.training and self.training_w1:
            x = F.linear(x, self.r1 * self.w1 * self.m1)
        else:
            x = F.linear(x, self.r1 * self.w1)
        x = self.bn1(x)
        x = F.silu(x, inplace=True)
        x = self.bn2(x)
        if self.training and self.training_w2:
            x = F.linear(x, self.r2 * self.w2 * self.m2)
        else:
            x = F.linear(x, self.r2 * self.w2)
        return x

    def build_model_from_teacher(self) -> None:
        """
        gets all the necessary info from the tree ensemble
        """
        tree_ensemble = self.teacher_model
        self.hidden_neurons = 0
        self.branch_info = []
        def bf_search(index, path, conditions, is_leaf, tree_index):

            if index == -1 or is_leaf[index]:
                return

            left_i = tree.children_left[index]
            right_i = tree.children_right[index]

            has_left_leaf = left_i != -1 and is_leaf[left_i]
            has_right_leaf = right_i != -1 and is_leaf[right_i]

            new_path = path[:]
            new_conditions = conditions[:]

            feature = tree.feature[index]
            threshold = tree.threshold[index]

            if feature >= 0:
                new_path.append(feature)

            # This parent node defines a branch-neuron
            if has_left_leaf or has_right_leaf:
                n_samples = tree.n_node_samples[0]
                node_samples = tree.n_node_samples[index]

                parents.append(index)
                path_to_parent.append(new_path)

                branch_info.append({
                    "tree_index": tree_index,
                    "parent_node": int(index),
                    "path_features": [int(f) for f in new_path],
                    "conditions": new_conditions,
                    # Split performed at the parent node itself
                    "parent_split": {
                        "feature": int(feature),
                        "threshold": float(threshold),
                        "left_leaf": int(left_i) if has_left_leaf else None,
                        "right_leaf": int(right_i) if has_right_leaf else None,
                        "left_majority_class": (
                            int(np.argmax(tree.value[left_i][0]))
                            if has_left_leaf else None
                        ),
                        "right_majority_class": (
                            int(np.argmax(tree.value[right_i][0]))
                            if has_right_leaf else None
                        ),
                    }
                })

                factor = node_samples / n_samples
                dist = factor * tree.value[index][0]
                class_proportion.append(dist)

            # Continue to non-leaf children
            if not has_left_leaf:
                bf_search(
                    left_i,
                    new_path,
                    new_conditions + [{
                        "feature": int(feature),
                        "threshold": float(threshold),
                        "direction": "left"
                    }],
                    is_leaf,
                    tree_index
                )

            if not has_right_leaf:
                bf_search(
                    right_i,
                    new_path,
                    new_conditions + [{
                        "feature": int(feature),
                        "threshold": float(threshold),
                        "direction": "right"
                    }],
                    is_leaf,
                    tree_index
                )

        def get_w1(size, device, dtype):
            w1 = torch.zeros(size, dtype=dtype)
            i = 0
            for t, paths_in_tree in enumerate(all_path_to_parent):
                for path in paths_in_tree:
                    w1[i][path] = feature_importance[t][path]
                    i += 1
            return w1.to(device)

        def get_w2(size, device, dtype):
            w2 = torch.zeros(size, dtype=dtype)
            i = 0
            for t, proportions_in_tree in enumerate(all_proportions):
                for p, classes_involved_in_branch in enumerate(proportions_in_tree):
                    w2[:, i] = torch.from_numpy(classes_involved_in_branch)
                    i += 1
            return w2.to(device)

        self.in_features = tree_ensemble.n_features_in_
        self.out_features = tree_ensemble.n_classes_

        all_path_to_parent = []
        feature_importance = []
        all_proportions = []

        estimators = tree_ensemble.estimators_
        for t, estimator in enumerate(estimators):
            branch_info = []
            tree = estimator.tree_
            is_leaf = (tree.children_left == -1) & (tree.children_right == -1)
            path_to_parent = []
            parents = []
            class_proportion = []
            bf_search(
                0,
                [],
                [],
                is_leaf,
                t
            )
            all_path_to_parent.append(path_to_parent)
            self.hidden_neurons += len(parents)
            all_proportions.append(class_proportion)
            importance = torch.zeros(self.in_features).float()
            for path in path_to_parent:
                for feat in path:
                    importance[feat] += 1
            feature_importance.append(importance / importance.max())
            self.branch_info.extend(branch_info)

        for neuron_idx, info in enumerate(self.branch_info):
            info["neuron"] = neuron_idx
        assert len(self.branch_info) == self.hidden_neurons

        for j, branch in enumerate(self.branch_info):
            assert branch["neuron"] == j

        self.bn0 = nn.BatchNorm1d(self.in_features, device=self.device)
        w1 = get_w1((self.hidden_neurons, self.in_features), device=self.device, dtype=self.dtype)
        self.w1 = nn.Parameter(w1, requires_grad=self.training_w1)
        self.m1 = (w1 != 0)

        for j, branch in enumerate(self.branch_info):

            for feature_idx in set(branch["path_features"]):
                assert self.m1[j, feature_idx].item(), (
                    f"Mismatch: neuron {j}, "
                    f"feature {feature_idx}, "
                    f"path={branch['path_features']}"
                )

        self.r1 = nn.Parameter(torch.tensor(1 / np.sqrt(self.in_features), dtype=self.dtype, device=self.device))
        self.r2 = nn.Parameter(torch.tensor(1 / np.sqrt(self.in_features), dtype=self.dtype, device=self.device))
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        w2 = get_w2((self.out_features, self.hidden_neurons), device=self.device, dtype=self.dtype)
        self.m2 = (w2 != 0)
        self.w2 = nn.Parameter(w2, requires_grad=self.training_w2)

    def build_from_dict(self, fn_dict) -> None:
        self.w1 = nn.Parameter(fn_dict['w1'])
        self.m1 = (self.w1 != 0)
        self.r1 = nn.Parameter(fn_dict['r1'])
        self.hidden_neurons = self.w1.shape[0]
        self.w2 = nn.Parameter(fn_dict['w2'])
        self.m2 = (self.w2 != 0)
        self.r2 = nn.Parameter(fn_dict['r2'])
        self.out_features = self.w2.shape[0]
        self.in_features = self.w1.shape[1]
        self.bn0 = nn.BatchNorm1d(self.in_features, device=self.device)
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)



