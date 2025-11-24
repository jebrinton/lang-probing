import torch
from nnsight import LanguageModel

model = LanguageModel("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="cuda")

test_sentences = ["Hello world", "This is a much longer sentence with more words"]

with model.trace(test_sentences, scan=False, validate=False) as tracer:
    input_data = model.inputs.save()
    acts_proxy = model.model.layers[16].output  # Don't call .save() yet
    
    # Try to use it directly - get the shape
    print(f"Acts proxy shape (inside trace): {acts_proxy.shape}")
    
    # Now save it
    acts = acts_proxy.save()

# Check shape after trace
print(f"Acts shape (after trace): {acts.shape}")