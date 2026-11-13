import torch
import torch.nn as nn
from rsl_rl.modules.former import RTransformer
def count_parameters(model):
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_parameter_count_by_layer(model):
    """Print parameter count for each layer"""
    print("\nParameter count by layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,} parameters")

def test_rtransformer():
    # Define parameters
    batch_size = 32
    seq_length = 50
    obs_dim = 44
    act_dim = 12
    obs_act_dim = obs_dim + act_dim
    emb_dim = 64
    heads = 2
    depth = 2
    output_dim = 64

    # Create random input data
    obs_act_history = torch.randn(batch_size, seq_length, obs_act_dim)

    # Initialize transformer
    transformer = RTransformer(
        emb=emb_dim,
        heads=heads,
        depth=depth,
        seq_length=seq_length,
        obs_act_dim=obs_act_dim,
        output_dim=output_dim,
        input_hidden_dims=[512, 256], 
        output_hidden_dims=[256, 128], 
        input_output_activation=nn.ReLU,
    )
    print(f"Transformer: {transformer}")
    # Print model parameter statistics
    total_params, trainable_params = count_parameters(transformer)
    print(f"\nTransformer Parameter Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print detailed layer-by-layer breakdown
    print_parameter_count_by_layer(transformer)
    # Test forward pass
    try:
        output = transformer(obs_act_history)
        print(f"\nTest successful!")
        print(f"Input shape: {obs_act_history.shape}")
        print(f"Output shape: {output.shape}")
        
        # Additional shape checks
        assert output.shape == (batch_size, seq_length, output_dim), \
            f"Expected output shape {(batch_size, seq_length, output_dim)}, got {output.shape}"
        
        # Test device transfer
        transformer = transformer.cuda()
        obs_act_history = obs_act_history.cuda()
        output = transformer(obs_act_history)
        print("GPU test successful!")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    test_rtransformer()
    # import torch
    # Simple matrix multiplication on GPU
    a = torch.randn(2, 3, dtype=torch.float32).cuda()
    b = torch.randn(3, 2, dtype=torch.float32).cuda()
    c = torch.mm(a, b)  # Should execute without error
    print(c)