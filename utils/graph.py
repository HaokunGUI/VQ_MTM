import torch
import argparse
import pickle
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import os
import torch.sparse as sparse

def get_supports(args:argparse.Namespace, input: torch.Tensor):
    batchsize, num_node, _ = input.shape
    
    if args.graph_type == 'distance':
        path = args.marker_dir + '/electrode_graph/adj_mx_3d.pkl'
        if not os.path.exists(path):
            raise ValueError('adjacency matrix not found')
        with open(path, 'rb') as f:
            adj_mat = pickle.load(f)
        adj_mat = torch.FloatTensor(adj_mat[-1]).cuda()
        adj_mat = adj_mat.unsqueeze(dim=0).repeat(batchsize, 1, 1)
    elif args.graph_type == 'correlation':
        with torch.no_grad():
            adj_mat = input @ input.transpose(1, 2)
            if args.normalize:
                corr_x = torch.sum(input**2, axis=-1).reshape(batchsize, 1, num_node)
                corr_y = torch.sum(input**2, axis=-1).reshape(batchsize, num_node, 1)
                scale = torch.sqrt(corr_x * corr_y)
                adj_mat /= scale
            adj_mat = torch.abs(adj_mat)
            if args.top_k is not None:
                k = args.top_k
                adj_mat_no_self_edge = adj_mat.clone().cuda()
                diagnol = torch.eye(num_node).reshape(1, num_node, num_node).repeat(batchsize, 1, 1).cuda()
                adj_mat_no_self_edge -= diagnol * adj_mat_no_self_edge

                _, topk_idx = torch.topk(adj_mat_no_self_edge, k=k, dim=-1)
                mask = torch.zeros_like(adj_mat_no_self_edge).cuda()
                mask.scatter_(-1, topk_idx, 1)

                # Symmetric mask if not directed
                if not args.directed:
                    mask = torch.logical_or(mask, mask.transpose(1, 2))
                mask = torch.logical_or(mask, diagnol)

                # Apply the mask to the adjacency matrix
                adj_mat = adj_mat * mask
    else:
        raise ValueError('Unknown graph type')
    
    # Convert adjacency matrix to support matrix
    supports = []
    supports_mat = []
    if args.filter_type == "laplacian":  # ChebNet graph conv
        supports_mat.append(calculate_scaled_laplacian(adj_mat, lambda_max=None))
    elif args.filter_type == "random_walk":  # Forward random walk
        supports_mat.append(calculate_random_walk_matrix(adj_mat).transpose(1, 2))
    elif args.filter_type == "dual_random_walk":  # Bidirectional random walk
        supports_mat.append(calculate_random_walk_matrix(adj_mat).transpose(1, 2))
        supports_mat.append(calculate_random_walk_matrix(adj_mat.transpose(1, 2)).transpose(1, 2))
    else:
        supports_mat.append(calculate_scaled_laplacian(adj_mat))
    for support in supports_mat:
        supports.append(support)
    
    return adj_mat, supports

def calculate_normalized_laplacian(adj:torch.Tensor):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    batch_size = adj.shape[0]
    d = adj.sum(dim=-1)
    d_inv_sqrt = d.pow(-0.5).reshape(batch_size, -1)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt = torch.diag_embed(d_inv_sqrt).cuda()
    normalized_laplacian = -d_inv_sqrt.bmm(adj).to_sparse().bmm(d_inv_sqrt) + \
    torch.eye(adj.shape[1]).repeat(batch_size, 1, 1).to_sparse().cuda()

    return normalized_laplacian

def calculate_random_walk_matrix(adj_mx:torch.Tensor):
    """
    State transition matrix D_o^-1W in paper.
    """
    batch_size = adj_mx.shape[0]
    d = adj_mx.sum(dim=-1)
    d_inv = d.pow(-1).reshape(batch_size, -1)
    d_inv[torch.isinf(d_inv)] = 0.
    d_mat_inv = torch.diag_embed(d_inv)
    random_walk_mx = d_mat_inv.bmm(adj_mx)
    return random_walk_mx

def calculate_scaled_laplacian(adj_mx: torch.Tensor, lambda_max=2, undirected=True):
    """
    Scaled Laplacian for ChebNet graph convolution
    """
    if undirected:
        adj_mx, _ = torch.max(torch.stack([adj_mx, adj_mx.t()]), dim=0)
    L = calculate_normalized_laplacian(adj_mx)  
    if lambda_max is None:
        lambda_max, _ = torch.linalg.eigvalsh(L)
        lambda_max = lambda_max[-1]
    
    batch_size, node_num, _ = adj_mx
    I = torch.eye(node_num).repeat(batch_size, 1, 1).cuda()
    L = (2 / lambda_max * L) - I
    return L


        