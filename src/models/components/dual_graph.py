import torch
import clip
from PIL import Image
import os
from torch import nn

class DualGraph(nn.Module):
    def __init__(self
                 ):
        super().__init__()
        self.model,self.preprocess = clip.load("ViT-B/32",device="cuda")
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier = classifier(input_size=512,num_class=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text,image_path):
        #text,image_path = batch
        text = clip.tokenize(text, context_length=77, truncate=True).to("cuda")
        text_features = self.model.encode_text(text)
        text_features = text_features.squeeze(0)
        if text_features.dim()==1:
            text_features = text_features.unsqueeze(0)
        # image_features_list = []
        # for image in image_path:
        #     image_features = self.preprocess(Image.open(image)).unsqueeze(0).to("cuda")
        #     #image_features = image_features.squeeze(0)
        #     image_features = torch.mean(image_features,dim=[3])
        #     image_features = image_features.view(1,-1)
        #     image_features_list.append(image_features)
        # # fusion all the image_features to one tensor
        # image_features_batch = torch.stack(image_features_list)
        # image_features_batch = image_features_batch.squeeze()
        # if image_features_batch.dim()==1:
        #     image_features_batch = image_features_batch.unsqueeze(0)
        # fused_features = torch.cat((text_features,image_features_batch),dim=1)
        
        
        outputs = self.classifier(text_features.to(torch.float32))
        logits = self.sigmoid(outputs)
        
        return logits
    
class classifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_class,
    ):
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(512,num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        #batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)
        return self.model(x)