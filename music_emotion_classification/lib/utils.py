import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

def normalize_A(A, symmetry=False):
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        I = torch.eye(D.size(0)).cuda()
        L = I - torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        I = torch.eye(D.size(0)).cuda()
        L = I - torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(A.shape[1]).cuda())
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support

# Metrics
def r_squared(y_pred, y_true):
    mean_true = torch.mean(y_true)
    total_sum_squares = torch.sum((y_true - mean_true) ** 2)
    sum_squares_residuals = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (sum_squares_residuals / total_sum_squares)
    return r2.item()

def kendall_tau(y_pred, y_true):
    tau, _ = kendalltau(y_pred.to("cpu"), y_true.to("cpu"))
    return tau

def mean_absolute_difference(y_pred, y_true):
    diff = np.abs(y_true.to("cpu") - y_pred.to("cpu"))
    mad = torch.mean(torch.squeeze(diff))
    return mad

def epsilon_insensitive_loss(y_pred, y_true, epsilon=0.1):
    diff = (y_pred - y_true).abs()
    return torch.where(diff < epsilon, torch.zeros_like(diff), diff - epsilon).mean()

def accuracy(y_pred, y_true):
    acc = (y_pred == y_true).sum().item() / len(y_pred)
    return acc

def precision(y_pred, y_true):
    TP = torch.sum((y_pred == 1) & (y_true == 1)).item()
    FP = torch.sum((y_pred == 1) & (y_true == 0)).item()
    precision = TP / (TP + FP)
    return precision

def recall(y_pred, y_true):
    TP = torch.sum((y_pred == 1) & (y_true == 1)).item()
    FN = torch.sum((y_pred == 0) & (y_true == 1)).item()
    recall = TP / (TP + FN)
    return recall

def f1_score(y_pred, y_true):
    precision_score = precision(y_pred, y_true)
    recall_score = recall(y_pred, y_true)
    f1_score = 2 * precision_score * recall_score / (precision_score + recall_score)
    return f1_score

def auc_score(y_pred, y_true):
    auc = roc_auc_score(y_true.to("cpu"), y_pred.to("cpu"))
    return auc

def confusion_matrix(y_pred, y_true):
    TP = torch.sum((y_pred == 1) & (y_true == 1)).item()
    TN = torch.sum((y_pred == 0) & (y_true == 0)).item()
    FP = torch.sum((y_pred == 1) & (y_true == 0)).item()
    FN = torch.sum((y_pred == 0) & (y_true == 1)).item()
    return np.array([[TP, FP], [FN, TN]])


# arg parse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_num", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--l2reg", type=float, default=0.0001, help="L2 regularization.")
    parser.add_argument("--decay_rate", type=float, default=0.96, help="Exponential decay rate.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size.")
    parser.add_argument("--model", type=str, choices=["RGCB","GCB","DGCNN","SVM"], default="RGCB", help="Model selection.")
    parser.add_argument("--evaluation_mode", type=str, choices=["a","v"], default="a", help="Evaluation mode.")
    
    parser.add_argument("--data_path_co", type=str, default="data/cochlegram.npy", help="Path to cochlegram data.")
    parser.add_argument("--data_path_mel", type=str, default="data/mel_spec.npy", help="Path to mel-spectrogram data.")
    parser.add_argument("--label_path_a", type=str, default="data/label_a.npy", help="Path to A label data.")
    parser.add_argument("--label_path_v", type=str, default="data/label_v.npy", help="Path to V label data.")
    parser.add_argument("--log_path", type=str, default="training_log_0422/", help="Path to log file.")
    
    return parser.parse_args()