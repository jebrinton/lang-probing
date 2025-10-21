import nnsight
import torch

TRACER_KWARGS = {'scan': False, 'validate': False}

llama = nnsight.LanguageModel("meta-llama/Llama-3.1-8B", torch_dtype=torch.float16, device_map="auto")

print(llama)

with llama.trace("Jungles become parks, but ") as tracer:
    output = llama.model.layers[4].output[0].save()

print(output)
print(output.shape)

