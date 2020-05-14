import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    '''
    带有attention计算的网络层
    '''


    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        '''
        参数：in_features 输入节点的特征数F
        参数：out_features 输出的节点的特征数F'
        参数：dropout
        参数：alpha LeakyRelu激活函数的斜率
        参数：concat
        '''
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.alpha = alpha  # 激活斜率 (LeakyReLU)的激活斜率
        self.concat = concat  # 用来判断是不是最有一个attention

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # 建立一个w权重，用于对特征数F进行线性变化
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 对权重矩阵进行初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))  # 计算函数α，输入是上一层两个输出的拼接，输出的是eij，a的size为(2*F',1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 对a进行初始化
        self.leakyrelu = nn.LeakyReLU(self.alpha)  # 激活层


    # 前向传播过程
    def forward(self, input, adj):
        '''
        参数input：表示输入的各个节点的特征矩阵
        参数adj ：表示邻接矩阵
        '''


        # 线性变化特征的过程,h的size为(N,F')，N表示节点的数量，F‘表示输出的节点的特征的数量
        h = torch.mm(input, self.W)
        # 获取当前的节点数量
        N = h.size()[0]
        # 下面是self-attention input ，构建自我的特征矩阵
        # 参数的计算过程如下：
        # h.repeat(1,N)将h的每一行按列扩展N次，扩展后的size为(N,F'*N)
        # .view(N*N,-1)对扩展后的矩阵进行重新排列，size为(N*N,F')每N行表示的都是同一个节点的N次重复的特征表示。
        # h.repeat(N,1)对当前的所有行重复N次，每N行表示N个节点的特征表示
        # torch.cat对view之后和repeat(N,1)的特征进行拼接，每N行表示一个节点的N次特征重复，分别和其他节点做拼接。size为(N*N,2*F')
        # .view(N,-1,2*self.out_features)表示将矩阵整理为(N,N,2*F')的形式。第一维度的每一个表示一个节点，第二个维度表示上一个节点对应的其他的所有节点，第三个节点表示特征拼接
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # 每一行是一个词与其他各个词的相关性值
        # matmul(a_input,self.a) 的size为(N,N,1)表示eij对应的数值，最后对第二个维度进行压缩
        # e的size为(N,N)，每一行表示一个节点，其他各个节点对该行的贡献度
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # 生成一个矩阵，size为(N,N)
        zero_vec = -9e15 * torch.ones_like(e)
        # 对于邻接矩阵中的元素，>0说明两种之间有变连接，就用e中的权值，否则表示没有变连接，就用一个默认值来表示
        attention = torch.where(adj > 0, e, zero_vec)
        # 做一个softmax，生成贡献度权重
        attention = F.softmax(attention, dim=1)
        # dropout操作
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 根据权重计算最终的特征输出。
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)  # 做一次激活
        else:
            return h_prime


    # 打印输出类名称，输入特征数量，输出特征数量
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer.
       对稀疏区域的反向传播函数
    """

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
