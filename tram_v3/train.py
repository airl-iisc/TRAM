"""Train TRAM V3 (V2 + CLAHE preprocessing).

The only difference from V2 training is the input transform: CLAHE is applied
on the L channel of LAB before the standard resize/normalize. This boosts
local contrast in low-contrast sonar imagery.

Example:
    python -m final.tram_v3.train \
        --data-root /path/to/SSS_OD-5 \
        --epochs 100 \
        --patience 20 \
        --output-dir runs/tram_v3
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from final.common.coco_eval import evaluate
from final.common.dataset import (
    DEFAULT_LABEL_MAP,
    SSSDataset,
    collate_fn,
    create_weighted_sampler,
    default_transform,
)
from final.common.seeding import seed_all, seed_worker
from final.common.transforms import CLAHETransform
from final.tram_v3.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TRAM V3 on SSS_OD")
    p.add_argument("--data-root", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--clahe-clip", type=float, default=2.0)
    p.add_argument("--clahe-grid", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="runs/tram_v3")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed Python/NumPy/Torch + cuDNN deterministic mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        seed_all(args.seed)
        print(f"Seeded everything to {args.seed} (cuDNN deterministic mode)")
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training.log")
    ckpt_path = os.path.join(args.output_dir, "tram_v3_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = DEFAULT_LABEL_MAP

    clahe = CLAHETransform(clipLimit=args.clahe_clip,
                           tileGridSize=(args.clahe_grid, args.clahe_grid))
    transform = default_transform(extra_first=clahe)

    train_ds = SSSDataset(
        os.path.join(args.data_root, "train", "images"),
        os.path.join(args.data_root, "train", "labels"),
        transform=transform,
        label_map=label_map,
    )
    val_ds = SSSDataset(
        os.path.join(args.data_root, "valid", "images"),
        os.path.join(args.data_root, "valid", "labels"),
        transform=transform,
        label_map=label_map,
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    sampler = create_weighted_sampler(train_ds)
    g = torch.Generator()
    if args.seed is not None:
        g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker if args.seed is not None else None,
        generator=g if args.seed is not None else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model(num_classes=len(label_map) + 1, pretrained=not args.no_pretrained)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_map = 0.0
    best_epoch = 0
    no_improve = 0

    with open(log_path, "w") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch}")
            for i, (images, targets) in pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in t.items()} for t in targets]
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                running += losses.item()
                if i % 20 == 0:
                    pbar.set_postfix(loss=losses.item())
            scheduler.step()
            avg = running / max(len(train_loader), 1)
            log_file.write(f"Epoch {epoch} avg loss {avg:.4f}\n")

            metrics = evaluate(model, val_loader, device, label_map)
            log_file.write(
                f"Epoch {epoch} | bbox mAP@0.5 {metrics['mAP_50_bbox']:.4f} "
                f"mAP@0.5:0.95 {metrics['mAP_50_95_bbox']:.4f} | "
                f"mask mAP@0.5 {metrics['mAP_50_mask']:.4f} "
                f"mAP@0.5:0.95 {metrics['mAP_50_95_mask']:.4f}\n"
            )
            log_file.flush()

            if metrics["mAP_50_95_bbox"] > best_map:
                best_map = metrics["mAP_50_95_bbox"]
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
                print(f"New best at epoch {epoch}: bbox mAP@0.5:0.95 {best_map:.4f}")
                log_file.write(f"New best at epoch {epoch}: {best_map:.4f}\n")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    msg = f"Early stop at epoch {epoch} (patience {args.patience})"
                    print(msg)
                    log_file.write(msg + "\n")
                    break

        log_file.write(f"Best: epoch {best_epoch} bbox mAP@0.5:0.95 {best_map:.4f}\n")
    if best_epoch > 0:
        print(f"Best model saved to {ckpt_path} (epoch {best_epoch}, bbox mAP@0.5:0.95 {best_map:.4f})")
    else:
        print("WARNING: no checkpoint saved — bbox mAP@0.5:0.95 never exceeded 0.0")


if __name__ == "__main__":
    main()
