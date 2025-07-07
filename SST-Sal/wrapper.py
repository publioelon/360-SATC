import torch
import torch.nn as nn
import cv2
import numpy as np
import onnxruntime as ort
from torch.amp import autocast
from utils.flow_viz import flow_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CombinedFlowSalWrapper(nn.Module):
    def __init__(self, session_raft, sst_sal_model, img_size=(320, 240), flow_to_sal_fn=None, T=20):
        super(CombinedFlowSalWrapper, self).__init__()
        self.session_raft = session_raft
        self.sst_sal_model = sst_sal_model
        self.img_size = img_size
        self.T = T
        self.flow_to_sal_fn = flow_to_sal_fn if flow_to_sal_fn is not None else self.simple_flow_to_sal

    def preprocess_image_for_sal(self, image):
        resized = cv2.resize(image, self.img_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        return tensor

    def preprocess_flow_from_np(self, flow_color):
        resized = cv2.resize(flow_color, self.img_size)
        resized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(device)
        return tensor

    def simple_flow_to_sal(self, flow):
        flow_np = flow[0].detach().cpu().numpy().transpose(1, 2, 0)
        flow_color = flow_to_image(flow_np, convert_to_bgr=True)
        return self.preprocess_flow_from_np(flow_color)

    def forward(self, image1, image2):
        image1_np = image1.cpu().numpy()
        image2_np = image2.cpu().numpy()
        input_names = [inp.name for inp in self.session_raft.get_inputs()]
        raft_inputs = {input_names[0]: image1_np, input_names[1]: image2_np}
        outputs = self.session_raft.run(None, raft_inputs)
        flow = outputs[0]
        if flow.ndim == 5:
            flow = flow[-1]
        flow_tensor = torch.tensor(flow).to(device)

        flow_3ch = self.flow_to_sal_fn(flow_tensor)

        image1_np_uint8 = image1[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        image_sal = self.preprocess_image_for_sal(cv2.cvtColor(image1_np_uint8, cv2.COLOR_RGB2BGR))
        sst_input_single = torch.cat([image_sal, flow_3ch], dim=1)
        sst_input = sst_input_single.unsqueeze(1).repeat(1, self.T, 1, 1, 1)

        with torch.no_grad():
            with autocast(device_type='cuda', enabled=True):
                saliency_seq = self.sst_sal_model(sst_input)
        saliency = saliency_seq.mean(dim=1)
        return saliency, flow_tensor
