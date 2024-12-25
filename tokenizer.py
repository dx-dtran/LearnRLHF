import json
import tiktoken


def tokenize_chatml_file(input_file, output_file, tokenizer):
    """
    Tokenizes the ChatML-formatted conversations in the given JSONL file
    and saves the tokenized output to a new JSONL file.
    """
    with open(input_file, "r") as f, open(output_file, "w") as out_f:
        for line in f:
            # Load the original JSON object
            conversation = json.loads(line)

            # Tokenize "chosen" and "rejected"
            chosen = conversation["chosen"]
            rejected = conversation["rejected"]

            # Tokenize using tiktoken
            chosen_tokens = tokenizer.encode(chosen)
            rejected_tokens = tokenizer.encode(rejected)

            # Save tokenized data to new JSON object
            tokenized_entry = {"chosen": chosen_tokens, "rejected": rejected_tokens}

            # Write the tokenized entry to the output file
            # Use separators to remove unnecessary whitespace
            out_f.write(json.dumps(tokenized_entry, separators=(",", ":")) + "\n")


def main():
    # Initialize the tiktoken tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")  # Replace with appropriate encoding

    # Input and output file paths
    input_files = ["chatml_train.jsonl", "chatml_test.jsonl"]
    output_files = ["train.jsonl", "test.jsonl"]

    # Tokenize each file
    for input_file, output_file in zip(input_files, output_files):
        print(f"Tokenizing {input_file}...")
        tokenize_chatml_file(input_file, output_file, tokenizer)
        print(f"Tokenized data saved to {output_file}")


if __name__ == "__main__":
    main()
