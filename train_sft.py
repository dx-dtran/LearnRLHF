import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sft_data import SupervisedFineTuningDataset, collate_fn
from gpt import GPT, GPTConfig


def train(model, dataloader, optimizer, scheduler, device, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
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

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss / len(dataloader):.4f}"
        )


def main():
    train_file = "train.jsonl"
    output_model_path = "gpt2_sft.pt"
    pretrained_weights = "gpt2.pt"
    block_size = 1024
    batch_size = 8
    learning_rate = 5e-5
    num_epochs = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SupervisedFineTuningDataset(train_file, block_size=block_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    config = GPTConfig()
    model = GPT(config).to(device)

    if pretrained_weights:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * num_epochs)

    train(model, dataloader, optimizer, scheduler, device, num_epochs)

    torch.save(model.state_dict(), output_model_path)


if __name__ == "__main__":
    main()