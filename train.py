"""Full contrastive training loop with symmetric InfoNCE loss.
Trains ImageTextModel on MS COCO 2017 image-caption pairs.
"""
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloader
from model import ImageTextModel


def infonce_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    temp: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE loss. Correct pairs sit on the diagonal of the B×B logit matrix."""
    logits = (img_emb @ txt_emb.T) * temp          # (B, B)
    labels = torch.arange(len(img_emb), device=img_emb.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2.0


def train(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    embed_dim: int = 256,
    freeze_epochs: int = 2,
    save_dir: str = 'checkpoints',
    max_samples: int = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Training on: {device}')

    train_loader = get_dataloader(
        img_dir='train2017',
        ann_file='annotations/captions_train2017.json',
        batch_size=batch_size,
        max_samples=max_samples,
    )
    total = len(train_loader.dataset)
    print(f'Training samples: {total:,} ({total // batch_size} steps/epoch)')

    model = ImageTextModel(embed_dim=embed_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        # Freeze encoders for the first freeze_epochs to warm up the projection heads
        if epoch <= freeze_epochs:
            for p in model.image_encoder.features.parameters():
                p.requires_grad = False
            for p in model.text_encoder.bert.parameters():
                p.requires_grad = False
        else:
            for p in model.parameters():
                p.requires_grad = True

        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            img_emb, txt_emb, temp = model(image, input_ids, attention_mask)
            loss = infonce_loss(img_emb, txt_emb, temp)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % 200 == 0:
                print(f'Epoch {epoch}/{epochs}  step {step:5d}  loss={loss.item():.4f}')

        scheduler.step()
        avg = total_loss / len(train_loader)
        print(f'Epoch {epoch} complete — avg loss={avg:.4f}')
        torch.save(model.state_dict(), os.path.join(save_dir, f'epoch{epoch:02d}.pt'))

    final_path = os.path.join(save_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f'Training complete. Model saved to {final_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--batch_size',   type=int,   default=64)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--embed_dim',    type=int,   default=256)
    parser.add_argument('--freeze_epochs',type=int,   default=2)
    parser.add_argument('--save_dir',     default='checkpoints')
    parser.add_argument('--max_samples',  type=int, default=None,
                        help='Use only the first N samples (default: full dataset)')
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        freeze_epochs=args.freeze_epochs,
        save_dir=args.save_dir,
        max_samples=args.max_samples,
    )
