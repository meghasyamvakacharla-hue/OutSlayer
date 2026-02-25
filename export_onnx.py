import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

def export_to_onnx():
    print("Loading model...")
    num_classes = 6
    model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=num_classes)
    model.load_state_dict(torch.load("best_model_20_epochs.pth", map_location='cpu'))
    model.eval()

    print("Exporting to ONNX...")
    # Dummy input
    dummy_input = torch.randn(1, 3, 256, 256)
    
    torch.onnx.export(
        model,
        dummy_input,
        "best_model_20_epochs.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete: best_model_20_epochs.onnx")

if __name__ == "__main__":
    export_to_onnx()
