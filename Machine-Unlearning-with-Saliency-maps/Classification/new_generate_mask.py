import copy
import os
from collections import OrderedDict

import arg_parser
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils

def compute_integrated_gradients(model, inputs, targets, baseline=None, steps=50, batch_size=10):
    """
    Compute Integrated Gradients for a batch of inputs.
    
    Args:
        model: The PyTorch model.
        inputs: The input tensor (batch_size, ...).
        targets: The target labels for the inputs.
        baseline: The baseline tensor (same shape as inputs).
        steps: Number of interpolation steps.
        batch_size: Number of samples to process at a time.
        
    Returns:
        Integrated gradients tensor (same shape as inputs).
    """
    device = inputs.device
    
    # Ensure the baseline has the same shape as inputs
    if baseline is None:
        baseline = torch.zeros_like(inputs).to(device)
    
    # Create scaled inputs
    alphas = torch.linspace(0, 1, steps).to(device)  # Interpolation coefficients
    scaled_inputs = torch.stack([baseline + alpha * (inputs - baseline) for alpha in alphas])  # Shape: (steps, batch_size, ...)
    
    # Enable gradient computation
    scaled_inputs.requires_grad_(True)

    # Initialize integrated gradients
    integrated_gradients = torch.zeros_like(inputs).to(device)

    # Process in smaller batches
    for i in range(0, inputs.size(0), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        batch_scaled_inputs = scaled_inputs[:, i:i+batch_size]

        # Accumulate gradients for each step
        for j in range(steps):
            # Forward pass through the model
            outputs = model(batch_scaled_inputs[j])
            target_outputs = outputs.gather(1, batch_targets.unsqueeze(1)).squeeze()  # Select target class outputs

            # Compute gradients w.r.t. inputs
            grads = torch.autograd.grad(
                outputs=target_outputs.sum(), inputs=batch_scaled_inputs[j], create_graph=False, retain_graph=False, allow_unused=True
            )[0]  # Shape: (batch_size, ...)

            # Accumulate gradients if grads is not None
            if grads is not None:
                integrated_gradients[i:i+batch_size] += grads / steps

    # Scale integrated gradients by the difference between inputs and baseline
    integrated_gradients *= (inputs - baseline)
    
    return integrated_gradients


def save_gradient_ratio_with_ig(model, data_loaders, criterion, args, steps=50, batch_size=10):
    """
    Save gradient ratios using Integrated Gradients.
    
    Args:
        model: The PyTorch model.
        data_loaders: DataLoader for the 'forget' dataset.
        criterion: The loss function.
        steps: Number of steps for IG computation.
        batch_size: Number of samples to process at a time.
    """
    
    model.eval()
    device = next(model.parameters()).device
    forget_loader = data_loaders["forget"]
    gradient_dict = {name: torch.zeros_like(param).to(device) for name, param in model.named_parameters()}
    baseline = None  # Default baseline (zero tensor)

    for inputs, targets in forget_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Ensure inputs require gradients
        inputs.requires_grad = True
        
        # Compute Integrated Gradients for the batch
        integrated_grads = compute_integrated_gradients(model, inputs, targets, baseline, steps, batch_size)
        
        # Sum and store the gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Compute gradients w.r.t. model parameters using integrated gradients
                param_grads = torch.autograd.grad(outputs=integrated_grads.sum(), inputs=param, create_graph=False, retain_graph=False, allow_unused=True)[0]
                if param_grads is not None:
                    gradient_dict[name] += param_grads

    # Normalize gradients and apply thresholds as in the original function
    total_gradient = sum(torch.sum(g) for g in gradient_dict.values())
    for name, grad in gradient_dict.items():
        gradient_dict[name] /= total_gradient  # Normalize
    
    # Save thresholds and masks (same as the original function)
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        hard_dict = {}
        for name, grad in gradient_dict.items():
            mask = (grad > threshold).float()
            hard_dict[name] = mask
        torch.save(hard_dict, os.path.join(args.save_dir, f"with_{threshold:.1f}.pt"))
        
        
def main():
    torch.cuda.empty_cache()
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.to(device)

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except AttributeError:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except AttributeError:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except AttributeError:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except AttributeError:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )

    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except AttributeError:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
            model.load_state_dict(checkpoint, strict=False)

    save_gradient_ratio_with_ig(model, unlearn_data_loaders, criterion, args, steps=50, batch_size=10)


if __name__ == "__main__":
    main()
