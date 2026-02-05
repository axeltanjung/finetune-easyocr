from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for iteration in range(num_iterations):
    optimizer.zero_grad()

    with autocast():
        # Forward pass dengan mixed precision
        preds = model(images)
        loss = criterion(preds, labels)

    # Backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
