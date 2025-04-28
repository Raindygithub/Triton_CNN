from torch import nn
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_HW': 32, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_HW': 64, 'BLOCK_SIZE_K': 32}, num_warps=8),
    ],
    key=['in_channels', 'out_channels', 'kernel_size'],
)


# 自定义 Triton 融合卷积+ReLU 内核
@triton.jit
def fused_conv_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr, in_channels, out_channels, kernel_size,
    input_height, input_width, stride, padding, BLOCK_SIZE: tl.constexpr
):
    # 获取当前线程块的输出位置
    pid = tl.program_id(0)
    oh = pid // input_width
    ow = pid % input_width
    
    # 计算输入窗口的起始位置
    ih_start = oh * stride - padding
    iw_start = ow * stride - padding
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # 卷积计算循环
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = ih_start + kh
                iw = iw_start + kw
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # 加载输入和权重
                    input_val = tl.load(input_ptr + [ic, ih, iw])
                    weight_val = tl.load(weight_ptr + [ic, kh, kw])
                    acc += input_val * weight_val
    
    # 添加偏置并应用 ReLU
    bias = tl.load(bias_ptr)
    result = tl.maximum(acc + bias, 0.0)
    
    # 存储结果
    tl.store(output_ptr + [oh, ow], result)

# 封装为 PyTorch 模块
class TritonConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        batch_size, _, h_in, w_in = x.shape
        h_out = (h_in + 2*self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2*self.padding - self.kernel_size) // self.stride + 1
        
        output = torch.empty((batch_size, self.out_channels, h_out, w_out), device=x.device)
        
        # 调用 Triton 内核
        grid = (batch_size * h_out * w_out,)
        fused_conv_relu_kernel[grid](
            x, self.weight, self.bias, output,
            self.in_channels, self.out_channels, self.kernel_size,
            h_in, w_in, self.stride, self.padding,
            BLOCK_SIZE=128
        )
        return output

# 优化后的模型
class OptimizedNet(nn.Module):
    def __init__(self):
        super(OptimizedNet, self).__init__()
        self.model = nn.Sequential(
            # 使用 Triton 融合层替换原始 Conv2d+ReLU
            TritonConvReLU(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            TritonConvReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            TritonConvReLU(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(4*4*64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


