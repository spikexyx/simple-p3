# Train the model with mutiple gpus
# Topology graph is duplicated across all the GPUs
# Feature data is horizontally partitioned across all the GPUs
# Each GPU has a feature size with: [#Nodes, Origin Feature Size / Total GPUs]
# Before every minibatch, feature is fetched from other GPUs for aggregation
# Model is duplicated across all the GPUs
# Use Pytorch DDP to synchronize model parameters / gradients across GPUs
# Use NCCL as the backend for communication
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from models.sage import Sage, create_sage_p3
from models.gat import Gat, create_gat_p3
from p3_trainer import P3Trainer
import gc
from utils import *
from torch.distributed import init_process_group, destroy_process_group, barrier
import os
import torch.multiprocessing as mp
from dgl.utils import pin_memory_inplace

def get_dgl_dataloader(config: RunConfig,
                         sampler: dgl.dataloading.NeighborSampler, 
                         graph: dgl.DGLGraph, 
                         train_nids: torch.Tensor,
                         use_dpp=True,
                         use_uva=False) -> dgl.dataloading.dataloader.DataLoader:
    device = torch.device(f"cuda:{config.rank}")
    graph = graph.to(device)
    dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph=graph,              # The graph
        indices=train_nids.to(device),         # The node IDs to iterate over in minibatches
        graph_sampler=sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on GPU
        use_ddp=use_dpp, # enable ddp if using mutiple gpus
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=config.batch_size,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=True,    # Whether to drop the last incomplete batch
        num_workers=0,       # Number of sampler processes
        use_uva=use_uva
    )
    return dataloader

def create_p3_model(config: RunConfig):
    if config.model == 'sage':
        return create_sage_p3(config.rank, config.local_in_feats, hid_feats=config.hid_feats, num_layers=len(config.fanouts), num_classes=config.num_classes)
    elif config.model == 'gat':
        return create_gat_p3(config.rank, config.local_in_feats, hid_feats=config.hid_feats, num_layers=len(config.fanouts), num_classes=config.num_classes, num_heads=config.num_heads)
    
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "63500"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def p3_train(rank:int, 
         world_size:int, 
         config: RunConfig,
         loc_feats: list[torch.Tensor],
         sampler: dgl.dataloading.NeighborSampler,
         node_labels: torch.Tensor, 
         idx_split):
    ddp_setup(rank, world_size)
    graph = dgl.hetero_from_shared_memory("dglgraph").formats("csc")
    node_labels = node_labels.to(rank)
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    loc_feat = loc_feats[rank].to(rank)
        
    config.rank = rank
    config.world_size = world_size
    config.mode = 3
    config.set_logpath()
    local_model, global_model = create_p3_model(config)                                           
    train_dataloader = get_dgl_dataloader(config, sampler, graph, train_nids, use_dpp=True)
    val_dataloader = get_dgl_dataloader(config, sampler, graph, valid_nids, use_dpp=True)
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3)
    local_optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
    trainer = P3Trainer(config, global_model, local_model, train_dataloader, val_dataloader, loc_feat, node_labels, global_optimizer, local_optimizer, nid_dtype=torch.int32)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple p3')
    parser.add_argument('--total_epochs', default=6, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=150, type=int, help='How often to save a snapshot')
    parser.add_argument('--hid_feats', default=256, type=int, help='Size of a hidden feature')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--nprocs', default=4, type=int, help='Number of GPUs / processes')
    parser.add_argument('--model', default="gat", type=str, help='Model type: sage or gat', choices=['sage', 'gat'])
    parser.add_argument('--num_heads', default=4, type=int, help='Number of heads for GAT model')
    parser.add_argument('--graph_name', default="ogbn-arxiv", type=str, help="Input graph name any of ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']", choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'])
    args = parser.parse_args()
    project_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(project_dir, "logs")
    data_dir = os.path.join(project_dir, "dataset")
    
    config = RunConfig()
    world_size = min(args.nprocs, torch.cuda.device_count())
    print(f"using {world_size} GPUs")
    print("start loading data")
    
    load_start = time.time()
    dataset = DglNodePropPredDataset(args.graph_name, root=data_dir)
    load_end = time.time()
    print(f"finish loading in {round(load_end - load_start, 1)}s")
    
    graph: dgl.DGLGraph = dataset[0][0]
    graph = dgl.add_self_loop(graph)
    node_labels: torch.Tensor = dataset[0][1]
    node_labels = node_labels.flatten().clone()
    torch.nan_to_num_(node_labels, nan=0.1)
    node_labels: torch.Tensor = node_labels.type(torch.int64)
    feat: torch.Tensor = graph.dstdata.pop("feat")

    # Get and display graph statistics
    stats = get_graph_statistics(graph, node_labels, feat)
    display_graph_statistics(stats, save_path=os.path.join(project_dir, "graph_statistics.csv"))

    # Display feature examples
    display_feature_examples(feat, num_examples=5, save_path=os.path.join(project_dir, "feature_examples.csv"))

    # Visualize degree distribution
    visualize_degree_distribution(graph, save_path=os.path.join(project_dir, "degree_distribution.png"))
     
    config.num_classes = dataset.num_classes
    config.batch_size = args.batch_size
    config.total_epoch = args.total_epochs
    config.hid_feats = args.hid_feats
    config.save_every = args.save_every
    config.graph_name = args.graph_name
    config.fanouts = [20, 20, 20]
    config.global_in_feats = feat.shape[1]
    config.model = args.model
    config.num_heads = args.num_heads
    idx_split = dataset.get_idx_split()
    config.log_dir = log_dir

    print("Global Feature Size: ", get_size_str(feat))    

    graph = graph.int()
    for key, nids in idx_split.items():
        idx_split[key] = nids.type(torch.int32)
    graph.create_formats_()
    print(f"using dgl sampler, graph formats created: {graph.formats()}")
    shared_graph = graph.shared_memory("dglgraph")
    sampler = dgl.dataloading.NeighborSampler(config.fanouts)
    del dataset, graph
    gc.collect()
    
    # Feature data is horizontally partitioned
    feats = [None] * world_size
    for i in range(world_size):
        feats[i] = get_local_feat(i, world_size, feat, padding=True).clone()
        if i == 0:
            config.global_in_feats = feats[i].shape[1] * world_size
            config.local_in_feats = feats[i].shape[1]
        assert(config.global_in_feats == feats[i].shape[1] * world_size)
        assert(config.local_in_feats == feats[i].shape[1])

    del feat
    gc.collect()

    # Visualize and save feature splits
    visualize_feature_splits(feats, save_path=os.path.join(project_dir, "feature_splits.png"))

    # P3 Data Vertical Split + Intra-Model Parallelism
    mp.spawn(p3_train, args=(world_size, config, feats, sampler, node_labels, idx_split), nprocs=world_size, daemon=True)            
