import numpy as np
from numpy import sin, cos, tan, pi, arcsin, arctan
from functools import lru_cache
import torch
from torch import nn
from torch.nn.parameter import Parameter


# Calculate kernels of SphereCNN
@lru_cache(None)
def get_xy(delta_phi, delta_theta):
    return np.array([
        [
            (-tan(delta_theta), 1/cos(delta_theta)*tan(delta_phi)),
            (0, tan(delta_phi)),
            (tan(delta_theta), 1/cos(delta_theta)*tan(delta_phi)),
        ],
        [
            (-tan(delta_theta), 0),
            (1, 1),
            (tan(delta_theta), 0),
        ],
        [
            (-tan(delta_theta), -1/cos(delta_theta)*tan(delta_phi)),
            (0, -tan(delta_phi)),
            (tan(delta_theta), -1/cos(delta_theta)*tan(delta_phi)),
        ]
    ])

@lru_cache(None)
def cal_index(h, w, img_r, img_c):
    '''
    Calculate Kernel Sampling Pattern for a 3x3 filter.
    Returns a (3, 3, 2) array of sampled coordinates.
    '''
    # pixel -> rad
    phi = -((img_r+0.5)/h*pi - pi/2)
    theta = (img_c+0.5)/w*2*pi - pi

    delta_phi = pi/h
    delta_theta = 2*pi/w

    xys = get_xy(delta_phi, delta_theta)
    x = xys[..., 0]
    y = xys[..., 1]
    rho = np.sqrt(x**2 + y**2)
    v = arctan(rho)

    new_phi = arcsin(cos(v)*sin(phi) + y*sin(v)*cos(phi)/rho)
    new_theta = theta + arctan(x*sin(v) / (rho*cos(phi)*cos(v) - y*sin(phi)*sin(v)))

    # rad -> pixel
    new_r = (-new_phi + pi/2)*h/pi - 0.5
    new_c = (new_theta + pi)*w/(2*pi) - 0.5

    # Wrap horizontally for equirectangular continuity
    new_c = (new_c + w) % w

    new_result = np.stack([new_r, new_c], axis=-1)
    # The center filter location remains the original (img_r, img_c)
    new_result[1, 1] = (img_r, img_c)

    return new_result


@lru_cache(None)
def _gen_filters_coordinates(h, w, stride):
    co = np.array([
        [cal_index(h, w, i, j) for j in range(0, w, stride)]
        for i in range(0, h, stride)
    ])
    # shape: (H/stride, W/stride, 3, 3, 2)
    # Transpose to (2, H/stride, W/stride, 3, 3)
    return np.ascontiguousarray(co.transpose([4, 0, 1, 2, 3]))


def gen_filters_coordinates(h, w, stride=1):
    '''
    Returns a NumPy array of shape (2, H/stride, W/stride, 3, 3).
    '''
    # Force h, w to be integers
    h = int(h)
    w = int(w)
    # Removed the assertion, since we are pinning to a fixed resolution
    return _gen_filters_coordinates(h, w, stride).copy()


def gen_grid_coordinates(h, w, stride=1):
    '''
    Generates a grid of normalized coordinates suitable for
    spherical sampling via grid_sample.
    '''
    # Force h, w to be integers
    h = int(h)
    w = int(w)

    coordinates = gen_filters_coordinates(h, w, stride).copy()
    # Rescale to [-1, 1]
    coordinates[0] = (coordinates[0] * 2 / h) - 1
    coordinates[1] = (coordinates[1] * 2 / w) - 1
    coordinates = coordinates[::-1]  # swap x <-> y dimension
    coordinates = coordinates.transpose(1, 3, 2, 4, 0)
    sz = coordinates.shape
    # Flatten the 2D kernel dimension into grid shape
    coordinates = coordinates.reshape(1, sz[0]*sz[1], sz[2]*sz[3], sz[4])

    return coordinates.copy()


class SphereConv2D(nn.Module):
    '''
    SphereConv2D
    A 3x3 "spherical" convolution that uses grid_sample
    to adapt the kernel to the sphere.
    '''
    def __init__(self, in_c, out_c, stride=1, bias=True, mode='bilinear'):
        super(SphereConv2D, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.mode = mode
        self.weight = Parameter(torch.Tensor(out_c, in_c, 3, 3))
        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.grid_shape = None
        self.grid = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        # x: (B, C, H, W)
        # If the shape changes (e.g., first run), compute the grid
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            # Force them to int (in case they're symbolic)
            h, w = int(x.shape[2]), int(x.shape[3])
            coordinates = gen_grid_coordinates(h, w, self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = False

        # Repeat the grid for the batch dimension
        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)

        # Spherical sampling
        x = nn.functional.grid_sample(x, grid, mode=self.mode, align_corners=False)
        # Ensure the input is cast to the same dtype as the weight (e.g., FP16)
        x = x.to(self.weight.dtype)
        # Now do a 3x3 conv with stride=3
        x = nn.functional.conv2d(x, self.weight, self.bias, stride=3)
        return x


class SphereMaxPool2D(nn.Module):
    '''
    SphereMaxPool2D
    A 3x3 spherical pooling operation using grid_sample
    and then standard MaxPool2d with stride=3.
    '''
    def __init__(self, stride=1, mode='bilinear'):
        super(SphereMaxPool2D, self).__init__()
        self.stride = stride
        self.mode = mode
        self.grid_shape = None
        self.grid = None
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

    def forward(self, x):
        # x: (B, C, H, W)
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            h, w = int(x.shape[2]), int(x.shape[3])
            coordinates = gen_grid_coordinates(h, w, self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = False

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)

        x = nn.functional.grid_sample(x, grid, mode=self.mode, align_corners=False)
        return self.pool(x)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Simple test
    print("[INFO] Testing SphereConv2D(3, 5, stride=1) on a random tensor")
    cnn = SphereConv2D(3, 5, 1)
    out = cnn(torch.randn(2, 3, 10, 10))
    print('Output shape: ', out.size())

    # Test pool
    print("[INFO] Testing SphereMaxPool2D(1) on a synthetic image")
    h, w = 100, 200
    img = np.ones([h, w, 3])
    for r in range(h):
        for c in range(w):
            img[r, c, 0] = img[r, c, 0] - r/h
            img[r, c, 1] = img[r, c, 1] - c/w

    plt.imsave('demo_original.png', img)
    img = img.transpose([2, 0, 1])  # (C, H, W)
    img = np.expand_dims(img, 0)    # (B, C, H, W)
    pool = SphereMaxPool2D(1)
    out = pool(torch.from_numpy(img).float())
    out = np.squeeze(out.numpy(), 0).transpose([1, 2, 0])
    plt.imsave('demo_pool_1.png', out)
    print('Saved image after pooling with stride=1 -> demo_pool_1.png')

    # Pool with stride=3
    pool = SphereMaxPool2D(3)
    img = np.expand_dims(img, 0)  # re-expand
    out = pool(torch.from_numpy(img[0]).float())
    out = np.squeeze(out.numpy(), 0).transpose([1, 2, 0])
    plt.imsave('demo_pool_3.png', out)
    print('Saved image after pooling with stride=3 -> demo_pool_3.png')
