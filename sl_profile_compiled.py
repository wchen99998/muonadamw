"""Profile the compiled training step to find actual bottlenecks."""
import torch
from sl_model import build_ours_model, build_ours_optimizer, generate_batches
from sl_train_step import train_step
from sl_constants import AUTOCAST_DTYPE, GRAD_CLIP_NORM

torch.set_float32_matmul_precision('high')
device = "cuda"
model = build_ours_model(device=device)
optimizer = build_ours_optimizer(model)
batches = generate_batches(device=device)

# Warmup (triggers compilation + CUDA graph capture)
for i in range(3):
    torch.compiler.cudagraph_mark_step_begin()
    train_step(model, batches[i], optimizer, AUTOCAST_DTYPE, GRAD_CLIP_NORM)
torch.cuda.synchronize()

# Profile
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for i in range(3, 6):
        torch.compiler.cudagraph_mark_step_begin()
        train_step(model, batches[i], optimizer, AUTOCAST_DTYPE, GRAD_CLIP_NORM)
    torch.cuda.synchronize()

# Print top kernels
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
