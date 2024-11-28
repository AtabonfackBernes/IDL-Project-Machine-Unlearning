import torch
import unlearn
import arg_parser

def load_checkpoint_and_print_metrics(checkpoint_path, device):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if the checkpoint contains evaluation results
    if "evaluation_result" in checkpoint:
        evaluation_result = checkpoint["evaluation_result"]
    else:
        print("No evaluation results found in the checkpoint.")
        return

    # Print the metrics
    for metric, value in evaluation_result.items():
        print(f"{metric}: {value}")

def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    # Path to the checkpoint file
    checkpoint_path = args.model_path

    # Load the checkpoint and print metrics
    load_checkpoint_and_print_metrics(checkpoint_path, device)

if __name__ == "__main__":
    main()