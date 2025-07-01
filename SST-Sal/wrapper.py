import torch
import torch.nn as nn
import cv2
import numpy as np
import onnxruntime as ort
from torch.amp import autocast
from utils.flow_viz import flow_to_image  # Make sure this is in your PYTHONPATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CombinedFlowSalWrapper(nn.Module):
    """
    Combined model wrapper that:
      1. Runs SEA-RAFT (via an ONNX session) to obtain optical flow.
      2. Converts the 2-channel flow to a 3-channel representation.
      3. Preprocesses the original image according to a configurable image size.
      4. Concatenates the preprocessed RGB image and flow (3 channels each) to form a 6-channel input.
      5. Repeats the input over T time steps for SST-SAL.
      6. Runs SST-SAL and averages the temporal predictions to produce the final saliency map.
      
    Args:
        session_raft: An ONNX inference session for SEA-RAFT.
        sst_sal_model: A PyTorch SST-SAL model.
        img_size: Tuple (width, height) used for input resizing (e.g., from config).
        flow_to_sal_fn: Optional function to convert a 2-channel flow to 3 channels.
                        If None, a simple default is used.
        T: Temporal length (number of repetitions) for SST-SAL input.
    """
    def __init__(self, session_raft, sst_sal_model, img_size=(320, 240), flow_to_sal_fn=None, T=20):
        super(CombinedFlowSalWrapper, self).__init__()
        self.session_raft = session_raft         # ONNX session for SEA-RAFT
        self.sst_sal_model = sst_sal_model         # SST-SAL model (PyTorch)
        self.img_size = img_size                   # (width, height) for resizing inputs
        self.T = T
        # Use provided function or default conversion
        self.flow_to_sal_fn = flow_to_sal_fn if flow_to_sal_fn is not None else self.simple_flow_to_sal

    def preprocess_image_for_sal(self, image):
        """
        Preprocess a BGR image for SST-SAL:
          - Resize to self.img_size
          - Convert BGR -> RGB
          - Normalize to [0, 1]
          - Convert to torch tensor with shape [1, 3, H, W]
        """
        resized = cv2.resize(image, self.img_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        return tensor

    def preprocess_flow_from_np(self, flow_color):
        """
        Preprocess a flow color image:
          - Resize to self.img_size
          - Normalize to [0, 1]
          - Convert to torch tensor with shape [1, 3, H, W]
        """
        resized = cv2.resize(flow_color, self.img_size)
        resized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(device)
        return tensor

    def simple_flow_to_sal(self, flow):
        """
        Convert a 2-channel flow tensor (shape [1, 2, H, W]) to a 3-channel representation:
          1. Convert the tensor to a numpy array.
          2. Use flow_to_image to get a color-coded (BGR) image.
          3. Preprocess that image to produce a tensor of shape [1, 3, H, W].
        """
        flow_np = flow[0].detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
        flow_color = flow_to_image(flow_np, convert_to_bgr=True)
        return self.preprocess_flow_from_np(flow_color)

    def forward(self, image1, image2):
        """
        Args:
            image1, image2: torch tensors of shape [1, 3, H, W] with values in [0, 255].
        Returns:
            saliency: Final saliency map from SST-SAL, shape [1, 1, H, W].
            flow: Optical flow from RAFT, shape [1, 2, H, W].
        """
        # --- Run SEA-RAFT via ONNX ---
        image1_np = image1.cpu().numpy()
        image2_np = image2.cpu().numpy()
        input_names = [inp.name for inp in self.session_raft.get_inputs()]
        raft_inputs = {input_names[0]: image1_np, input_names[1]: image2_np}
        outputs = self.session_raft.run(None, raft_inputs)
        flow = outputs[0]
        if flow.ndim == 5:
            flow = flow[-1]  # Use final iteration if multiple are returned
        flow_tensor = torch.tensor(flow).to(device)  # [1, 2, H, W]

        # --- Convert Flow to 3-Channel Representation ---
        flow_3ch = self.flow_to_sal_fn(flow_tensor)  # [1, 3, H, W]

        # --- Prepare SST-SAL Input ---
        # Convert image1 (tensor) to a numpy BGR image.
        image1_np_uint8 = image1[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        # Note: If your image is already in BGR, adjust accordingly.
        image_sal = self.preprocess_image_for_sal(cv2.cvtColor(image1_np_uint8, cv2.COLOR_RGB2BGR))
        # Concatenate image_sal and flow_3ch along channel dimension -> [1, 6, H, W]
        sst_input_single = torch.cat([image_sal, flow_3ch], dim=1)
        # Add temporal dimension and repeat T times -> [1, T, 6, H, W]
        sst_input = sst_input_single.unsqueeze(1).repeat(1, self.T, 1, 1, 1)

        # --- Run SST-SAL ---
        with torch.no_grad():
            with autocast(device_type='cuda', enabled=True):
                saliency_seq = self.sst_sal_model(sst_input)  # Expected shape: [1, T, 1, H, W]
        saliency = saliency_seq.mean(dim=1)  # Average temporal predictions

        return saliency, flow_tensor
