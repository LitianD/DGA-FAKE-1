import torch
import clip
from PIL import Image
import os
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.nn import HypergraphConv
from torch_geometric.datasets import Planetoid
from src.models.components.diffusion import Diffusion
import json
from src.models.diffusion_module import DiffusionModule
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
import pickle


class GAT(torch.nn.Module):
    def __init__(self, in_channels=128, hidden_channels=128, out_channels=128, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class HyperGraph(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=128, out_channels=128, dropout=0.6):
        super(HyperGraph, self).__init__()
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels,out_channels)
        self.dropout_p = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class DualGraph(nn.Module):
    def __init__(self,embedding_checkpoint_path
                 ):
        super().__init__()
        self.model,self.preprocess = clip.load("ViT-B/32",device="cuda")
        # self.user_embedding = DiffusionModule()
        # self.embedding_checkpoint_path = '/data/zlt/python_code/fake-news-baselines/logs/train_diffusion_pol/runs/2024-01-13_11-54-10/checkpoints/last.ckpt'
        self.embedding_checkpoint_path = embedding_checkpoint_path
        # self.user_embedding.load_state_dict(torch.load(self.embedding_checkpoint_path))
        # user_embedding.eval()
        self.user_embedding = DiffusionModule.load_from_checkpoint(self.embedding_checkpoint_path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.gat = GAT()
        self.hypergraph = HyperGraph()
        self.classifier = classifier(input_size=1280,num_class=2)
        self.sigmoid = nn.Sigmoid()
        
    @torch.no_grad()
    def cal_user_embedding(self,seq, seq_len):
        seq = torch.Tensor(seq).to(text_features.device)
        seq_len = torch.Tensor(seq_len).to(text_features.device)
        user_feature = self.user_embedding.net.model.predict(seq.long(), seq_len.long())
        if user_feature.shape[0] == 128:
            return user_feature
        return torch.mean(user_feature,dim=0)
    
    @torch.no_grad()
    def trans_propogation_graph(self,seq, seq_len, mask=1435):
        node_list = [0]
        node_list_id = [0]
        edge_index = []
        edge_index_id = []
        node_id = {
            0:0
        }
        hyper_edge_node = []
        hyper_edge_id = []
        temp_edge_num = 0
        for item in seq:
            temp_hyper_node = []
            for i in range(len(item)):
                if item[i]==mask:
                    break
                else:
                    temp_hyper_node.append(item[i])
            hyper_edge_node+=temp_hyper_node
            hyper_edge_id=hyper_edge_id+[temp_edge_num]*len(temp_hyper_node)
            temp_edge_num+=1
            
            for i in range(len(item)):
                if item[i]==0:
                    continue
                if item[i]==mask:
                    break
                if item[i] not in node_list:
                    node_list.append(item[i])
                    node_id[item[i]] = len(node_id)
                    # node_list_id.append(len(node_list_id))
                if (item[i-1],item[i]) not in edge_index:
                    edge_index.append([item[i-1],item[i]])
                    edge_index_id.append((node_id[item[i-1]],node_id[item[i]]))
        transpose_edge_index_id = list(map(list, zip(*edge_index_id)))
        nodes = torch.Tensor(node_list).to('cuda')
        transpose_edge_index_id = torch.Tensor(transpose_edge_index_id).to('cuda')
        user_feature = self.user_embedding.net.model.item_embeddings(nodes.long())
        # user_feature = self.user_embedding.net.model.predict(seq.long(), seq_len.long())
        
        hyper_edge_node = [node_id[item] for item in hyper_edge_node]
        hyper_edge_index = torch.Tensor([hyper_edge_node,hyper_edge_id]).to('cuda')
        return user_feature, transpose_edge_index_id.to(torch.int), hyper_edge_index.to(torch.int64)

        
    def forward(self, text,image_path,user_path):
        #text,image_path = batch
        text = clip.tokenize(text, context_length=77, truncate=True).to("cuda")
        text_features = self.model.encode_text(text)
        text_features = text_features.squeeze(0)
        if text_features.dim()==1:
            text_features = text_features.unsqueeze(0)
            
        image_features_list = []
        for i in range(len(image_path)):
            image = image_path[i]
            if image == '-1':
                image_features = text_features[i]
            else: 
                try:
                    Image.open(image)
                except:
                    image_features = text_features[i]
                    image_features_list.append(image_features)
                    continue
                image_features = self.preprocess(Image.open(image)).unsqueeze(0).to("cuda")
                image_features = self.model.encode_image(image_features)
                image_features = image_features.squeeze(0)
            image_features_list.append(image_features)
        # fusion all the image_features to one tensor
        image_features_batch = torch.stack(image_features_list)
        image_features_batch = image_features_batch.squeeze()
        if image_features_batch.dim()==1:
            image_features_batch = image_features_batch.unsqueeze(0)

        
        directed_graph_features_list = []
        hyper_graph_features_list = []
        for i in range(len(user_path)):
            user = user_path[i]
            if user == '-1':
                seq  = [[0]+99*[1435]]
                seq_len = [1]
            else:
                with open(user, 'rb') as file:
                    user = pickle.load(file)
                seq, seq_len = user['seq'], user['seq_len']
            user_feature, edge_index, hyper_edge_index = self.trans_propogation_graph(seq,seq_len)
            # user_feature = self.cal_user_embedding(seq, seq_len)
            # user_features_list.append(user_feature)
            directed_graph_features_list.append(gData(x=user_feature, edge_index=edge_index))
            hyper_graph_features_list.append(gData(x=user_feature, edge_index=hyper_edge_index))
        # user_features_batch = torch.stack(user_features_list)
        # user_features_batch = user_features_batch.squeeze() 
        directed_batch_graph = Batch.from_data_list(directed_graph_features_list)  
        directed_x = self.gat(directed_batch_graph.x,directed_batch_graph.edge_index)
        directed_out,mask = to_dense_batch(directed_x, directed_batch_graph.batch)
        directed_graph_feature = directed_out.mean(dim=1)
                
        hypergraph_batch_graph = Batch.from_data_list(hyper_graph_features_list)  
        hypergraph_x = self.hypergraph(hypergraph_batch_graph.x,hypergraph_batch_graph.edge_index)
        hypergraph_out,mask = to_dense_batch(hypergraph_x, hypergraph_batch_graph.batch)
        hypergraph_features = hypergraph_out.mean(dim=1)
        
        
        fused_features = torch.cat((text_features,image_features_batch,directed_graph_feature,hypergraph_features),dim=1)   
        outputs = self.classifier(fused_features.to(torch.float32))
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
            nn.Linear(input_size, 512),
            nn.Linear(512,num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        #batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)
        return self.model(x)