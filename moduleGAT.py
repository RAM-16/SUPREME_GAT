import os
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv
from collections import Counter
# torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(torch.nn.Module):
    def __init__(self, in_size=16, hid_size=8, out_size=2, heads=4):
        super(Net, self).__init__()
        # Replace GCNConv with GATConv
        self.conv1 = GATConv(in_size, hid_size, heads=heads, concat=True)  # First GAT layer
        self.conv2 = GATConv(hid_size * heads, out_size, heads=1, concat=False)  # Output GAT layer

    def forward(self, data):
        data = data.to(device)
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)  # Pass edge weights if needed
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x, x_emb
