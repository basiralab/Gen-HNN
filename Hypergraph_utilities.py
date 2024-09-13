# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np



def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat



def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def feature_concat_modified(*F_list, normal_col=False):
    features = None
    for f in F_list:
        if f is not None and f.size > 0:  # Check if tensor is not empty
            if len(f.shape) > 2:
                # Flatten the tensor along the last dimension
                f = f.reshape(-1, f.shape[-1])
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col and features is not None:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features

def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and len(h) > 0:  # Check if h is not None and not empty
            if H is None:
                H = h.copy() if isinstance(h, np.ndarray) else h
            else:
                if isinstance(h, np.ndarray):
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H

def construct_G(data):
    """
    Constructs a normalized hypergraph incidence matrix G from the input data.
    
    Args:
        data (np.ndarray): Input data of shape (n_ROIs, n_ROIs, n_views), where `n_views` is 4.
    
    Returns:
        G (np.ndarray or tensor): The normalized hypergraph incidence matrix.
    """
    
    # Construct hypergraph incidence matrices for each view using K-nearest neighbors
    H1 = construct_H_with_KNN(data[:, :, 0], K_neigs=3)
    H2 = construct_H_with_KNN(data[:, :, 1], K_neigs=3)
    H3 = construct_H_with_KNN(data[:, :, 2], K_neigs=3)
    H4 = construct_H_with_KNN(data[:, :, 3], K_neigs=3)

    # Ensure contiguous memory layout for efficiency
    H1 = np.ascontiguousarray(H1)
    H2 = np.ascontiguousarray(H2)
    H3 = np.ascontiguousarray(H3)
    H4 = np.ascontiguousarray(H4)

    # Concatenate the incidence matrices along the second axis
    H = np.concatenate((H1, H2, H3, H4), axis=1)

    # Generate and return the normalized hypergraph incidence matrix G
    G = generate_G_from_H(H)
    
    return G

def construct_hypergraph_tensor(data, K_neigs=5):
    """
    Converts a given dataset to a hypergraph tensor using the K-nearest neighbors algorithm.

    Args:
        data (np.ndarray): Input data of shape (N, 35, 35, 4), where N is the number of subjects/samples.
        K_neigs (int, optional): Number of nearest neighbors to use in the hypergraph construction. Default is 5.

    Returns:
        hypergraph_tensor (np.ndarray): Hypergraph tensor of shape (N, 35, 35, 4) representing the constructed hypergraphs.
    """
    N, n_ROIs, _, n_views = data.shape
    hypergraph_tensor = np.zeros(shape=(N, n_ROIs, n_ROIs, n_views))

    for i, mat in enumerate(data):
        for view in range(n_views):
            view_data = mat[:, :, view]
            H = construct_H_with_KNN(view_data, K_neigs=K_neigs)
            hypergraph_tensor[i, :, :, view] = H

    return hypergraph_tensor


    