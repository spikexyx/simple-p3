import torch, dgl
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import time
import csv
from dataclasses import dataclass
from dgl import create_block
import os
# For visualization
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

def partition_ids(rank: int, world_size: int, nids: torch.Tensor) -> torch.Tensor:
    step = int(nids.shape[0] / world_size)
    start_idx = rank * step
    end_idx = start_idx + step
    loc_ids = nids[start_idx : end_idx]
    return loc_ids.to(rank)

def print_model_weights(model: torch.nn.Module):
    for name, weight in model.named_parameters():
        if weight.requires_grad:
            print(name, weight, weight.shape, "\ngrad:", weight.grad)
        else:
            print(name, weight, weight.shape)


def get_size(tensor: torch.Tensor) -> int:
    shape = tensor.shape
    size = 1
    if torch.float32 == tensor.dtype or torch.int32 == tensor.dtype:
        size *= 4
    elif torch.float64 == tensor.dtype or torch.int64 == tensor.dtype:
        size *= 8
    for dim in shape:
        size *= dim
    return size

def get_size_str(tensor: torch.Tensor) -> str:
    size = get_size(tensor)
    if size < 1e3:
        return f"{round(size / 1000.0)} KB"
    elif size < 1e6:
        return f"{round(size / 1000.0)} KB"
    elif size < 1e9:
        return f"{round(size / 1000000.0)} MB"
    else:
        return f"{round(size / 1000000000.0)} GB"
    


# This function split the feature data horizontally
# each node's data is partitioned into 'world_size' chunks
# return the partition corresponding to the 'rank'
# Input args:
# rank: [0, world_size - 1]
# Output: feat
def get_local_feat(rank: int, world_size:int, feat: torch.Tensor, padding=True) -> torch.Tensor:
    org_feat_width = feat.shape[1]
    if padding and org_feat_width % world_size != 0:
        step = int(org_feat_width / world_size)
        pad = world_size - org_feat_width + step * world_size
        padded_width = org_feat_width + pad
        assert(padded_width % world_size == 0)
        step = int(padded_width / world_size)
        start_idx = rank * step
        end_idx = start_idx + step
        local_feat = None
        if rank == world_size - 1:
            # padding is required for P3 to work correctly
            local_feat = feat[:, start_idx : org_feat_width]
            zeros = torch.zeros((local_feat.shape[0], pad), dtype=local_feat.dtype)
            local_feat = torch.concatenate([local_feat, zeros], dim=1)
        else:
            local_feat = feat[:, start_idx : end_idx]
        return local_feat
    else:
        step = int(feat.shape[1] / world_size)
        start_idx = rank * step
        end_idx = min(start_idx + step, feat.shape[1])
        if rank == world_size - 1:
            end_idx = feat.shape[1]
        local_feat = feat[:, start_idx : end_idx]
        return local_feat
    
class TrainProfiler:
    def __init__(self, filepath: str) -> None:
        self.items = []
        self.path = filepath
        self.fields = ["epoch", "val_acc", "epoch_time", "forward", "backward", "feat", "sample", "other"]        
    
    def log_step_dict(self, item: dict):
        for k, v in item.items():
            if (type(v) == float):
                item[k] = round(v, 5)
        self.items.append(item)
        self.fields = list(item.keys())
        
    def log_step(self, 
                epoch: int, 
                val_acc: float,
                epoch_time: float,
                forward: float,
                backward: float,
                feat: float,
                sample: float) -> dict:
        
        other = epoch_time - forward - backward - feat - sample
        item = {
            "epoch": epoch,
            "val_acc": val_acc,
            "epoch_time": epoch_time,
            "forward": forward,
            "backward": backward,
            "feat": feat,
            "sample": sample,
            "other": other
        }

        for k, v in item.items():
            if (type(v) == type(1.0)):
                item[k] = round(v, 5)
        self.items.append(item)
        return item
    
    def avg_epoch(self) -> float:
        if (len(self.items) <= 1):
            return 0
        avg_epoch_time = 0.0
        epoch = 0
        for idx, item in enumerate(self.items):
            if idx != 0:
                avg_epoch_time += item["epoch_time"]
                epoch += 1
        return avg_epoch_time / epoch
    
    
    def saveToDisk(self):
        print("AVERAGE EPOCH TIME: ", round(self.avg_epoch(), 4))
        with open(self.path, "w+") as file:
            writer = csv.DictWriter(file, self.fields)
            writer.writeheader()
            for idx, item in enumerate(self.items):
                if idx > 0:
                    writer.writerow(item)


def visualize_feature_splits(local_feats, save_path="feature_splits.png"):
    num_gpus = len(local_feats)
    num_nodes = local_feats[0].shape[0]
    
    # Concatenate all local features to show the complete feature space
    combined_feats = np.hstack([local_feat.cpu().numpy() for local_feat in local_feats])

    plt.figure(figsize=(20, 5))
    plt.imshow(combined_feats[:100, :], aspect='auto', cmap='viridis')  # Display only the first 100 nodes' features
    plt.colorbar()
    plt.title('Feature Partitions Across GPUs')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Node Index')
    
    # Add vertical lines to separate GPU partitions
    for i in range(num_gpus):
        if i > 0:
            plt.axvline(x=i * local_feats[0].shape[1], color='red', linestyle='--')
        midpoint = (i * local_feats[0].shape[1] + (i + 1) * local_feats[0].shape[1]) / 2
        plt.text(midpoint, -10, f'GPU {i}', ha='center', va='center', color='red', fontsize=12, fontweight='bold')
    
    plt.savefig(save_path)
    plt.close()

def get_graph_statistics(graph, node_labels, features):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    num_classes = len(torch.unique(node_labels))
    num_features = features.shape[1]

    stats = {
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Number of Classes": num_classes,
        "Number of Features": num_features
    }

    return stats

def visualize_degree_distribution(graph, save_path="degree_distribution.png"):
    degrees = graph.out_degrees().numpy()
    plt.figure(figsize=(10, 6))
    sns.histplot(degrees, bins=50, kde=True)
    plt.yscale('log')
    plt.xlabel("Degree")
    plt.ylabel("Frequency (log scale)")
    plt.title("Node Degree Distribution")
    plt.savefig(save_path)
    plt.close()

def display_graph_statistics(stats, save_path="graph_statistics.csv"):
    df = pd.DataFrame(stats.items(), columns=["Statistic", "Value"])
    df.to_csv(save_path, index=False)
    print(df)

def display_feature_examples(features, num_examples=5, save_path="feature_examples.csv"):
    sample_indices = np.random.choice(features.shape[0], num_examples, replace=False)
    sample_features = features[sample_indices]
    df = pd.DataFrame(sample_features.numpy())
    df.to_csv(save_path, index=False)
    print(df)


@dataclass
class RunConfig:
    rank: int = 0
    world_size: int = 1
    topo: str = "gpu"
    feat: str = "gpu"
    global_in_feats: int = -1
    local_in_feats: int = -1
    hid_feats: int = 64
    num_classes: int = -1 # output feature size
    batch_size: int = 512
    total_epoch: int = 30
    save_every: int = 30
    fanouts: list[int] = None
    log_dir: str = ""
    graph_name: str = "ogbn-arxiv"
    log_path: str = "log.csv" # logging output path
    checkpt_path: str = "checkpt.pt" # checkpt path
    model: str = "sage" # model (sage or gat)
    num_heads: int = 3 # if use GAT, number of heads in the model
    mode: int = 3 # runner version
    def uva_sample(self) -> bool:
        return self.topo == 'uva'
    
    def uva_feat(self) -> bool:
        return self.feat == 'uva'
    
    def set_logpath(self):
        feat_setting = f"{self.feat.lower()}feat"
        topo_setting = f"{self.topo.lower()}topo"
        self.log_path = os.path.join(self.log_dir, f"{self.graph_name}_v{self.mode}_w{self.world_size}_{feat_setting}_{topo_setting}_h{self.hid_feats}_b{self.batch_size}.csv")

