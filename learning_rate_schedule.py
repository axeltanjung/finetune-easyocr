from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Option 1: Cosine Annealing with Warm Restarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10000,  # Restart setiap 10k iterations
    T_mult=2,    # Double period setiap restart
    eta_min=1e-6
)

# Option 2: One Cycle LR (recommended untuk convergence cepat)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=num_iterations,
    pct_start=0.3,  # 30% warmup
    anneal_strategy='cos'
)

# Update LR setiap iteration
for iteration in range(num_iterations):
    train_step()
    scheduler.step()
