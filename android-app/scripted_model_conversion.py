import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

# Load TorchScript model
model = torch.jit.load('model_scripted.pt')
model.eval()

# Optimize the model
scripted_module = torch.jit.script(model)
optimized_scripted_module = optimize_for_mobile(scripted_module)

# Export full jit version model (not compatible with lite interpreter)
scripted_module.save("model_scripted.pt")
# Export lite interpreter version model (compatible with lite interpreter)
scripted_module._save_for_lite_interpreter("model_scripted.ptl")
# using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
optimized_scripted_module._save_for_lite_interpreter("model_scripted_optimized.ptl")
