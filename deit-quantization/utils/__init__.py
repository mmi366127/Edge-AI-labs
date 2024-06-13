import torch
import os


def save_model(quantized_model, save_path):
    # save quantized model
    quantized_model.cpu()
    cpu_example_inputs = (torch.randn(1, 3, 224, 224), )
    quantized_ep = torch.export.export(
        quantized_model, 
        cpu_example_inputs
    )
    print(f"quantized model saved as {save_path}")
    torch.export.save(quantized_ep, save_path)


def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    model_size = os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return model_size