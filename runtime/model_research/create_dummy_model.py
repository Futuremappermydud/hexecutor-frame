import torch
import torch.nn as nn

class DummyStereoDepth(nn.Module):
    def __init__(self):
        super(DummyStereoDepth, self).__init__()
        # Simple convolutional layers to mock a depth network
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        # x is expected to be [B, 2, H, W] (Left and Right stacked)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

def main():
    model = DummyStereoDepth()
    model.eval()
    
    # The pipeline expects 616x616 rectified images
    dummy_input = torch.randn(1, 2, 616, 616)
    
    onnx_path = "hitnet_dummy.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=['input'], 
        output_names=['output'],
        opset_version=11
    )
    print(f"Successfully generated {onnx_path}")

if __name__ == "__main__":
    main()
