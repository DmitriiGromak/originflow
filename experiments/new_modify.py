import torch
import os
from collections import OrderedDict

import torch
import os
from collections import OrderedDict

import torch
import os
from collections import OrderedDict


def modify_checkpoint_parameters(ckpt_path, output_path=None):
    """
    Load a PyTorch checkpoint, modify the ipa.c_s parameter to reference node_embed_size,
    and save it back.

    Args:
        ckpt_path (str): Path to the original checkpoint file
        output_path (str, optional): Path to save the modified checkpoint. If None, overwrites the original.

    Returns:
        str: Path to the saved modified checkpoint
    """
    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Print the top-level keys to understand the structure
    print("Top-level keys in the checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")

    # Check if the checkpoint has the expected structure
    if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
        cfg = checkpoint['hyper_parameters']['cfg']

        # Get the current values
        current_node_embed_size = cfg['model']['node_embed_size']
        current_ipa_c_s = cfg['model']['ipa']['c_s']

        print(f"Current values:")
        print(f"  - node_embed_size: {current_node_embed_size}")
        print(f"  - ipa.c_s: {current_ipa_c_s}")

        # Update ipa.c_s to reference node_embed_size using string interpolation
        cfg['model']['ipa']['c_s'] = "${model.node_embed_size}"

        print(f"Updated ipa.c_s to reference node_embed_size: {cfg['model']['ipa']['c_s']}")

        # Save the modified checkpoint
        if output_path is None:
            output_path = ckpt_path

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(checkpoint, output_path)
        print(f"\nSaved modified checkpoint to {output_path}")

        return output_path
    else:
        print("Error: Checkpoint does not have the expected structure")
        print(
            "Available keys in hyper_parameters:" if 'hyper_parameters' in checkpoint else "No hyper_parameters key found")
        if 'hyper_parameters' in checkpoint:
            for key in checkpoint['hyper_parameters'].keys():
                print(f"  - {key}")
        return None


if __name__ == "__main__":
    # Path to the checkpoint
    ckpt_path = "/home/junyu/project/Proflow/weight/sym.ckpt"

    # Path to save the modified checkpoint
    output_path = "/weight/sym.ckpt"

    # Modify the checkpoint
    modify_checkpoint_parameters(ckpt_path, output_path)

