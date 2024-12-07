import torch


def train_model(model, dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, stage_name):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\n[{stage_name}] Starting Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        print(f"[{stage_name}] Epoch {epoch + 1}: Starting training phase")
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f"[{stage_name}] Epoch {epoch + 1}: Processing batch {batch_idx + 1}/{len(dataloader)}")
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            print(f"[{stage_name}] Epoch {epoch + 1}: Batch {batch_idx + 1}: Forward pass completed")

            # Compute loss
            loss = criterion(outputs, targets)
            print(f"[{stage_name}] Epoch {epoch + 1}: Batch {batch_idx + 1}: Loss computed: {loss.item()}")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            print(f"[{stage_name}] Epoch {epoch + 1}: Batch {batch_idx + 1}: Backward pass and optimization completed")

            running_loss += loss.item()

        avg_train_loss = running_loss / len(dataloader)
        train_losses.append(avg_train_loss)
        print(f"[{stage_name}] Epoch {epoch + 1}: Training phase completed. Avg Train Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        print(f"[{stage_name}] Epoch {epoch + 1}: Starting validation phase")
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_dataloader):
                print(f"[{stage_name}] Epoch {epoch + 1}: Validation batch {batch_idx + 1}/{len(val_dataloader)}")
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                print(f"[{stage_name}] Epoch {epoch + 1}: Validation batch {batch_idx + 1}: Forward pass completed")

                # Compute loss
                loss = criterion(outputs, targets)
                print(
                    f"[{stage_name}] Epoch {epoch + 1}: Validation batch {batch_idx + 1}: Loss computed: {loss.item()}")

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print(f"[{stage_name}] Epoch {epoch + 1}: Validation phase completed. Avg Val Loss: {avg_val_loss}")

        # Step the scheduler
        scheduler.step()
        print(f"[{stage_name}] Epoch {epoch + 1}: Scheduler step completed")

        # Epoch summary
        print(f"[{stage_name}] Epoch {epoch + 1} Summary: Train Loss = {avg_train_loss}, Val Loss = {avg_val_loss}")

    print(f"\n[{stage_name}] Training completed for all {num_epochs} epochs.")
    return train_losses, val_losses