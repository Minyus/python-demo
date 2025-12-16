# Contrastive Learning

## Angular Margin Loss in Metric Learning

ArcFace/CosFace,SphereFace, etc.

```mermaid
flowchart TD

image -- encoder --> raw_embedding -- L2-normalize --> 
L2-normalized_image_embedding -- dot_product --> cosine_similarity -- "add margin penalty on positive class" --> cosine_similarity_w_margin  -- temperature-scale --> logits -- "softmax" --> probabilities --> cross_entropy 
ground_truth --> cross_entropy

image["image (N,3,H,W)"]
raw_embedding["raw embedding (N,D)"]
L2-normalized_image_embedding["L2-normalized embedding (N,D)"]
cosine_similarity_w_margin["cosine similarity with margin penalty on positive class (N,C)"]
logits["logits (N,C)"]
probabilities["probabilities (N,C)"]
ground_truth["onehot ground truth (N,C)"]

raw_category_embedding -- L2-normalize --> L2-normalized_category_embedding -- dot_product --> cosine_similarity

raw_category_embedding["raw category embedding (C,D)"]
L2-normalized_category_embedding["L2-normalized category embedding (C,D)"]

cosine_similarity["cosine similarity (N,C)"]
```

Formulate as category classification,
num_classes = num of categories

num of cosine similarity values = batch size x num_classes

## NT-Xent or InfoNCE

NT-Xent (Normalized temperature-scaled cross entropy) a.k.a InfoNCE (Information Noise Contrastive Estimation)

InfoNCE may not include temperature-scaling.

CLIP

```mermaid
flowchart TD

image -- encoder --> raw_embedding -- L2-normalize --> 
L2-normalized_image_embedding -- dot_product --> cosine_similarity -- temperature-scale --> logits -- "softmax" --> probabilities --> cross_entropy 
identity_matrix --> cross_entropy

image["image (N,3,H,W)"]
raw_embedding["raw embedding (N,D)"]
L2-normalized_image_embedding["L2-normalized embedding (N,D)"]
logits["logits (N,C)"]
probabilities["probabilities (N,C)"]
identity_matrix["identity matrix (N,N)"]

text_tokens --> raw_text_embedding -- L2-normalize --> L2-normalized_text_embedding -- dot_product --> cosine_similarity

text_tokens["text tokens (N, Tokens)"]
raw_text_embedding["raw text embedding (N,D)"]
L2-normalized_text_embedding["L2-normalized text embedding (N,D)"]

cosine_similarity["cosine similarity (N,N)"]
```

Using in-batch negatives,
num_classes = batch size

num of cosine similarity values = batch size *num_classes = batch size x batch size
