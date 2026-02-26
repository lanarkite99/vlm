from transformers import ViTImageProcessor,ViTModel
import torch
from PIL import Image

image_processor=Image.open("dataset/cc_images/00000/000001964.jpg")
print(image_processor.size)

model_name="google/vit-base-patch16-224"
feature_extractor=ViTImageProcessor.from_pretrained(model_name)
vit_model=ViTModel.from_pretrained(model_name)

print(vit_model.config)
inputs=feature_extractor(image_processor,return_tensors="pt")
print(inputs)
print(inputs.pixel_values.shape)

with torch.no_grad():
    outputs=vit_model(**inputs)
print('embeddings shape:')
print(outputs.last_hidden_state.shape)


