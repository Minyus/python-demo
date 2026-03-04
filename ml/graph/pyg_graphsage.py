import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv

torch.manual_seed(0)

# ----------------------------
# 1) Build a tiny directed graph with out-degree=3
# ----------------------------
N = 20
F_in = 8
x = torch.randn(N, F_in)  # [N, F_in]

src_list, dst_list = [], []
for i in range(N):
    for j in (1, 2, 3):
        src_list.append(i)
        dst_list.append((i + j) % N)

edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)  # [2, E]
data = Data(x=x, edge_index=edge_index)

# ----------------------------
# 2) NeighborSampler (block/bipartite sampling)
# ----------------------------
# sizes=[3,3] => 2 blocks (2 "hops") for a 2-layer model.
batch_size = 4
sizes = [3, 3]

# NeighborSampler yields:
#   batch_size (int), n_id (LongTensor [N_batch]), adjs (list of length L)
# where each adj is (edge_index, e_id, size), and size=(num_src, num_dst).  :contentReference[oaicite:1]{index=1}
sampler = NeighborSampler(
    data.edge_index,
    sizes=sizes,
    node_idx=torch.arange(N),
    batch_size=batch_size,
    shuffle=False,
)

bs, n_id, adjs = next(iter(sampler))
# n_id: [N_batch] node ids from the original graph (local-to-global mapping)
# adjs: list of length L, each element is a bipartite block

print("=== Sampled batch ===")
print(f"seed batch_size bs = {bs}")
print(f"n_id.shape = {tuple(n_id.shape)}  # [N_batch]")
print(f"num blocks = {len(adjs)} (should match num_layers=2)")
print()

# ----------------------------
# 3) Inspect blocks to prove what each layer will see
# ----------------------------
# Each block has:
#   edge_index: [2, E_block] (indices are LOCAL within the block)
#   size: (num_src, num_dst)
#
# Crucial property (PyG docs): target nodes are included at the beginning of the
# source-node list in each block, so dst nodes correspond to the first size[1]
# nodes in the src list. :contentReference[oaicite:2]{index=2}
#
# In a 2-hop sample:
#   Block 0: hop2 -> hop1   (dst = hop1)
#   Block 1: hop1 -> seeds  (dst = seeds)
#
# We'll verify this by printing size[1]. For the last block, size[1] should equal bs.
for li, (eidx, e_id, size) in enumerate(adjs):
    E = eidx.size(1)
    num_src, num_dst = size
    print(
        f"[Block {li}] edge_index.shape = {tuple(eidx.shape)}  # [2, E_block], E_block={E}"
    )
    print(f"[Block {li}] size = (num_src={num_src}, num_dst={num_dst})")
    # Sanity: dst indices are in [0, num_dst)
    assert int(eidx[1].max()) < num_dst
    # src indices are in [0, num_src)
    assert int(eidx[0].max()) < num_src
    print()

print(
    "NOTE: In typical 2-layer neighbor sampling, the LAST block's num_dst == bs (seed count)."
)
print(
    "      That last block is the only adjacency used by the FINAL layer that outputs seed embeddings."
)
print()


# ----------------------------
# 4) Block-style (bipartite) 2-layer GraphSAGE forward
# ----------------------------
class BlockSAGE(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        assert num_layers == 2
        self.convs = nn.ModuleList(
            [
                SAGEConv(in_dim, hid_dim),
                SAGEConv(hid_dim, out_dim),
            ]
        )

    def forward(self, x_all: torch.Tensor, n_id: torch.Tensor, adjs):
        # x_all: [N, F_in]
        # n_id:  [N_batch]
        x = x_all[
            n_id
        ]  # [N_batch, F_in]  local features for all nodes in this sampled computation graph

        for layer, (edge_index, e_id, size) in enumerate(adjs):
            num_src, num_dst = size

            # For this block:
            # - "src" nodes correspond to x[:num_src]
            # - "dst/target" nodes correspond to x[:num_dst]  (targets come first) :contentReference[oaicite:3]{index=3}
            x_src = x[:num_src]  # [num_src, D_in]
            x_dst = x[:num_dst]  # [num_dst, D_in]

            print(f"=== Layer {layer+1} ===")
            print(
                f"uses ONLY Block {layer}: size(src={num_src}, dst={num_dst}), E={edge_index.size(1)}"
            )
            print("=> This layer cannot see edges from any other block.")
            print()

            x_out = self.convs[layer]((x_src, x_dst), edge_index)  # [num_dst, D_out]

            if layer != len(adjs) - 1:
                x_out = F.relu(x_out)  # [num_dst, D_out]
            x = x_out  # [num_dst, D_out] becomes the node features for the next (closer-to-seed) block

        # After final layer, x corresponds to embeddings of the FINAL block's dst nodes (seeds).
        return x  # [bs, out_dim] typically


model = BlockSAGE(F_in, 16, 4, num_layers=2)
out = model(data.x, n_id, adjs)  # [bs, 4] if last block dst=seeds

print(f"out.shape = {tuple(out.shape)}  # [bs, out_dim]")
print(
    "If out.shape[0] == bs, then the model output corresponds exactly to seed node embeddings."
)
