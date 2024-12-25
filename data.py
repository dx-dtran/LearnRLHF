import json
from datasets import load_dataset


# Function to split a multi-turn conversation into ChatML roles
def convert_multi_turn_to_chatml(dialogue):
    """
    Convert a multi-turn dialogue into ChatML format.
    Handles alternating 'Human:' and 'Assistant:' turns.
    """
    chatml_parts = []
    turns = dialogue.split("\n\n")  # Split on double newlines as turn separator

    for turn in turns:
        if turn.startswith("Human:"):
            user_message = turn[len("Human:") :].strip()
            chatml_parts.append(f"<|im_start|>user\n{user_message}\n<|im_end|>")
        elif turn.startswith("Assistant:"):
            assistant_message = turn[len("Assistant:") :].strip()
            chatml_parts.append(
                f"<|im_start|>assistant\n{assistant_message}\n<|im_end|>"
            )

    # Combine all turns into a single ChatML-formatted conversation
    return "\n".join(chatml_parts)


# Function to process dataset and convert entries to ChatML
def convert_dataset_to_chatml(data):
    """
    Process the dataset, converting each entry (chosen/rejected) to ChatML.
    Handles multi-turn conversations. Excludes the system prompt.
    """
    chatml_data = []
    for entry in data:
        chosen = entry.get("chosen", "").strip()
        rejected = entry.get("rejected", "").strip()

        # Convert multi-turn conversations into ChatML format
        chatml_chosen = convert_multi_turn_to_chatml(chosen)
        chatml_rejected = convert_multi_turn_to_chatml(rejected)

        # Append converted data as JSON object
        chatml_data.append({"chosen": chatml_chosen, "rejected": chatml_rejected})

    return chatml_data


# Main function to process the dataset and save train/test separately
def main():
    # Load the dataset
    print("Loading the HH RLHF dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf")

    # Process each subset separately
    for subset_name in dataset.keys():
        print(f"Processing subset: {subset_name}")
        subset_data = dataset[subset_name]

        # Convert to ChatML format
        chatml_data = convert_dataset_to_chatml(subset_data)

        # Save to file
        output_file = f"chatml_{subset_name}.jsonl"
        print(f"Saving ChatML data for {subset_name} to {output_file}...")
        with open(output_file, "w") as f:
            for entry in chatml_data:
                f.write(json.dumps(entry) + "\n")

    print("All subsets processed and saved.")


if __name__ == "__main__":
    main()
