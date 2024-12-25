import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from datetime import datetime
import os
from sft_data import SupervisedFineTuningDataset, collate_fn
from gpt import GPT, GPTConfig, transpose_specific_layers


def setup_logger():
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    log_file = f"training-{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(), timestamp


def estimate_val_loss(model, dataloader, device, logger, num_batches=10):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits, loss = model(
                input_ids, targets=target_ids, attention_mask=attention_mask
            )

            total_loss += loss.item()

    average_loss = total_loss / min(len(dataloader), num_batches)
    logger.info(f"Validation Subset Loss: {average_loss:.4f}")
    return average_loss


def calculate_val_loss(model, dataloader, device, logger):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits, loss = model(
                input_ids, targets=target_ids, attention_mask=attention_mask
            )

            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    logger.info(f"Full Validation Loss: {average_loss:.4f}")
    return average_loss


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    logger,
    timestamp,
    val_interval=10,
):

    weights_dir = f"weights_{timestamp}"
    os.makedirs(weights_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            start_time = time.time()

            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            logits, loss = model(
                input_ids, targets=target_ids, attention_mask=attention_mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            elapsed_time = time.time() - start_time

            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, Time: {elapsed_time:.2f}s"
            )

            if (batch_idx + 1) % val_interval == 0:
                logger.info(
                    f"Running validation subset at Epoch {epoch+1}, Batch {batch_idx+1}"
                )
                estimate_val_loss(model, val_dataloader, device, logger)

                weights_path = os.path.join(
                    weights_dir, f"gpt2_sft_{epoch+1}_{batch_idx+1}.pt"
                )
                torch.save(model.state_dict(), weights_path)
                logger.info(f"Saved model weights to {weights_path}")

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss / len(train_dataloader):.4f}"
        )

        logger.info(f"Starting full validation for Epoch {epoch+1}")
        calculate_val_loss(model, val_dataloader, device, logger)


def main():
    logger, timestamp = setup_logger()

    train_file = "train.jsonl"
    test_file = "test.jsonl"
    output_model_path = "gpt2_sft.pt"
    pretrained_weights = "gpt2.pt"
    block_size = 1024
    batch_size = 8
    learning_rate = 5e-5
    num_epochs = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_dataset = SupervisedFineTuningDataset(train_file, block_size=block_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_dataset = SupervisedFineTuningDataset(test_file, block_size=block_size)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    config = GPTConfig()
    model = GPT(config).to(device)

    if pretrained_weights:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        state_dict_transposed = transpose_specific_layers(state_dict)
        model.load_state_dict(state_dict_transposed, strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * num_epochs)

    train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        device,
        num_epochs,
        logger,
        timestamp,
    )

    torch.save(model.state_dict(), output_model_path)


if __name__ == "__main__":
    main()
