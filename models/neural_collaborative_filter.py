import torch
from torch import nn
from .layers import EmbeddingLayer, MultiLayerPerceptron

class NCF(nn.Module):
    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout):
        super(NCF, self).__init__()
        self.task_num = task_num
        # self.num_embedding_gmf = nn.Embedding(num_users, embedding_size)
        self.cat_embedding_gmf = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.cat_embedding_mlp = EmbeddingLayer(categorical_field_dims, embed_dim)

        self.num_embedding_gmf = nn.Linear(numerical_num, embed_dim)
        self.num_embedding_mlp = nn.Linear(numerical_num, embed_dim)

        # self.fc1 = nn.Linear(embed_dim * 2, 128)
        # self.dropout1 = nn.Dropout(dropout)
        # self.act1 = nn.ReLU()
        # self.fc2 = nn.Linear(128, 64)
        # self.dropout2 = nn.Dropout(dropout)
        # self.act2 = nn.ReLU()
        # self.fc3 = nn.Linear(embed_dim + 64, 1)
        # self.act3 = nn.Sigmoid()
        # self.MLP = MultiLayerPerceptron(embed_dim * 2, tower_mlp_dims, dropout, output_layer=False)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.embed_output_dim_gmf = (len(categorical_field_dims)) * embed_dim
        self.bottom = MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False)
        self.tower = torch.nn.ModuleList(
            [MultiLayerPerceptron(bottom_mlp_dims[-1]+self.embed_output_dim_gmf, tower_mlp_dims, dropout) for i in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        categorical_x_gmf = self.cat_embedding_gmf(categorical_x)
        numerical_x_gmf = self.num_embedding_gmf(numerical_x).unsqueeze(1)
        # print(categorical_x_gmf.shape, numerical_x_gmf.shape)
        gmf_output = categorical_x_gmf * numerical_x_gmf
        gmf_output = gmf_output.view(-1, self.embed_output_dim_gmf)

        categorical_x_mlp = self.cat_embedding_mlp(categorical_x)
        numerical_x_mlp = self.num_embedding_mlp(numerical_x).unsqueeze(1)

        mlp_x = torch.cat([categorical_x_mlp, numerical_x_mlp], 1).view(-1, self.embed_output_dim)
        mlp_x = self.bottom(mlp_x)
        # mlp_x = self.dropout1(self.act1(self.fc1(mlp_x)))
        # mlp_output = self.dropout2(self.act2(self.fc2(mlp_x)))
        #
        concat_output = torch.cat([gmf_output, mlp_x], dim=-1)
        # prediction = self.act3(self.fc3(concat_output))
        prediction = [torch.sigmoid(self.tower[i](concat_output).squeeze(1)) for i in range(self.task_num)]

        return prediction