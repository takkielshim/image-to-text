"""Compute Recall@K metrics for image-to-text and text-to-image retrieval
on the MS COCO 2017 validation set.
"""
import torch
from tqdm import tqdm

from dataset import get_dataloader
from model import ImageTextModel


def recall_at_k(sim_matrix: torch.Tensor, k: int) -> float:
    """Recall@K: fraction of queries whose ground-truth is in the top-K results.
    Assumes the correct pair for query i is candidate i (diagonal).
    """
    top_k_indices = torch.topk(sim_matrix, k, dim=1).indices        # (N, k)
    ground_truth = torch.arange(sim_matrix.size(0), device=sim_matrix.device).unsqueeze(1)
    hits = (top_k_indices == ground_truth).any(dim=1).float()
    return hits.mean().item()


def median_rank(sim_matrix: torch.Tensor) -> float:
    """Median rank of the ground-truth match for each query."""
    ranks = (sim_matrix.argsort(dim=1, descending=True) ==
             torch.arange(sim_matrix.size(0), device=sim_matrix.device).unsqueeze(1)).nonzero()
    return float(ranks[:, 1].float().median().item() + 1)  # 1-indexed


@torch.no_grad()
def evaluate(checkpoint: str = 'checkpoints/final_model.pt', embed_dim: int = 256) -> None:
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = ImageTextModel(embed_dim=embed_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    val_loader = get_dataloader(
        img_dir='val2017',
        ann_file='annotations/captions_val2017.json',
        batch_size=64,
        shuffle=False,
    )

    img_embs, txt_embs = [], []
    for batch in tqdm(val_loader, desc='Extracting embeddings'):
        img_embs.append(model.encode_image(batch['image'].to(device)).cpu())
        txt_embs.append(
            model.encode_text(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
            ).cpu()
        )

    I = torch.cat(img_embs)   # (N, D)
    T = torch.cat(txt_embs)   # (N, D)

    sim_i2t = I @ T.T
    sim_t2i = T @ I.T

    print('\n=== Image → Text Retrieval ===')
    for k in [1, 5, 10]:
        print(f'  R@{k:2d}: {recall_at_k(sim_i2t, k) * 100:.1f}%')
    print(f'  Median rank: {median_rank(sim_i2t):.1f}')

    print('\n=== Text → Image Retrieval ===')
    for k in [1, 5, 10]:
        print(f'  R@{k:2d}: {recall_at_k(sim_t2i, k) * 100:.1f}%')
    print(f'  Median rank: {median_rank(sim_t2i):.1f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/final_model.pt')
    parser.add_argument('--embed_dim', type=int, default=256)
    args = parser.parse_args()
    evaluate(checkpoint=args.checkpoint, embed_dim=args.embed_dim)
