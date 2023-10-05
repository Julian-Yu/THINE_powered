import torch
import numpy as np
import random
import sys
from Model_Dataset import mtne_metapath_dataset
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from Evaluation import Evaluation, read_embedding
from Heterogeneous_Rec import H_Evaluation, H_read_embedding
from Node_Classification import do_node_classification


class metapath_mtne(torch.nn.Module):
    def __init__(self, args, metapath_data, train_edge):
        super(metapath_mtne, self).__init__()
        self.args = args

        self.DID = self.args.DID  # Device ID
        self.dataset_name = self.args.dataset_name  # 数据集名称, 'Aminer'或'Yelp'
        self.node_type = self.args.node_type  # 节点类型, ['author', 'paper', 'venue', 'term']
        self.node_dim = self.args.node_dim  # 节点维度, {node type: , ...}
        self.metapath_type = self.args.metapath_type  # 元路径类型数量, {metapath type: , ...}
        self.edge_type = self.args.edge_type  # 边类型, sorted edge []
        self.output_file = self.args.output_file
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.embedding_dim = self.args.dim
        self.learning_rate = self.args.initial_lr
        self.closest_metapath = self.args.closest_metapath  # 计算连接概率的最近元路径数量
        self.closest_edge = self.args.closest_edge  # 计算元路径强度的最近边数量
        self.node_similarity_method = self.args.node_similarity_method  # 节点相似度计算方法, True: 矩阵相似度, False: 欧氏距离
        self.metapath_influence_method = self.args.metapath_influence_method  # 元路径影响力计算方法, True: 加法, False: 乘法
        self.data = mtne_metapath_dataset(self.args, metapath_data, train_edge)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # region 节点嵌入以保证所有节点和边的特征都在同一维度
        if torch.cuda.is_available():
            with torch.cuda.device(self.DID):
                self.emb_end = '_emb'
                self.delta_end = '_delta'
                # 为每一类节点设置时间衰减系数
                for node_type_key in self.node_type:
                    # 节点嵌入系数矩阵，[节点数量, 嵌入维度]，(-1/√n, 1/√n)的均匀分布
                    temp = torch.FloatTensor(np.random.uniform(-1. / np.sqrt(self.node_dim[node_type_key]),
                                                               1. / np.sqrt(self.node_dim[node_type_key]),
                                                               (
                                                                   self.node_dim[node_type_key],
                                                                   self.embedding_dim))).cuda(
                        self.DID)
                    # 嵌入权重及偏置
                    self.register_parameter(node_type_key + self.emb_end, torch.nn.Parameter(temp))
                    temp = torch.FloatTensor(torch.zeros(self.node_dim[node_type_key]) + 1.).cuda(self.DID)
                    self.register_parameter(node_type_key + self.delta_end, torch.nn.Parameter(temp))

            for edge_type_key in self.edge_type:
                edge_type_key = edge_type_key[0] + '_' + edge_type_key[1]
                # 边嵌入系数矩阵，[嵌入维度, 嵌入维度]，(-1/√n, 1/√n)的均匀分布
                temp = torch.FloatTensor(np.random.uniform(-1. / self.embedding_dim, 1. / self.embedding_dim,
                                                           (self.embedding_dim, self.embedding_dim))).cuda(self.DID)
                self.register_parameter(edge_type_key, torch.nn.Parameter(temp))

            self.distance_att = torch.nn.Parameter(
                (torch.zeros(self.closest_edge) + 1.).type(torch.FloatTensor).cuda(self.DID)) # 距离权重，可学习
            self.metapath_att = torch.nn.Parameter(
                (torch.zeros(self.metapath_type) + 1.).type(torch.FloatTensor).cuda(self.DID)) # 元路径权重，可学习
        else:
            self.emb_end = '_emb'
            self.delta_end = '_delta'
            for node_type_key in self.node_type:
                temp = torch.FloatTensor(np.random.uniform(-1. / np.sqrt(self.node_dim[node_type_key]),
                                                           1. / np.sqrt(self.node_dim[node_type_key]),
                                                           (self.node_dim[node_type_key], self.embedding_dim)))
                self.register_parameter(node_type_key + self.emb_end, torch.nn.Parameter(temp))
                temp = torch.FloatTensor(torch.zeros(self.node_dim[node_type_key]) + 1.)
                self.register_parameter(node_type_key + self.delta_end, torch.nn.Parameter(temp))

            for edge_type_key in self.edge_type:
                edge_type_key = edge_type_key[0] + '_' + edge_type_key[1]
                temp = torch.FloatTensor(np.random.uniform(-1. / self.embedding_dim, 1. / self.embedding_dim,
                                                           (self.embedding_dim, self.embedding_dim)))
                self.register_parameter(edge_type_key, torch.nn.Parameter(temp))  # self.edge_type

            self.distance_att = torch.nn.Parameter((torch.zeros(self.closest_edge) + 1.).type(torch.FloatTensor))
            self.metapath_att = torch.nn.Parameter((torch.zeros(self.metapath_type) + 1.).type(torch.FloatTensor))
        # endregion

        self.opt = torch.optim.Adam(lr=self.learning_rate, params=self.parameters())
        self.loss = torch.FloatTensor()

    def edge_att_index(self, s_type, t_type):
        for i in range(len(self.edge_type)):
            if ([s_type, t_type] == self.edge_type[i]) or ([t_type, s_type] == self.edge_type[i]):
                return i

    def forward(self, s_node, t_node, s_t_time, metapath_s, neg_s_node, neg_s_metapath):
        # 计算节点相似度，η(x,y)
        s_t_similarity = self.nodes_similarity(s_node[0], s_node[1], t_node[0], t_node[1])
        dis_att = softmax(self.distance_att, dim=0)     # 公式(8)中θ(h(i,j))
        mp_type_att = softmax(self.metapath_att, dim=0)     #公式(6)中的ω(m)
        if self.metapath_influence_method:
            mp_influ = 0.
        else:
            mp_influ = 1.
        for metapath_temp in metapath_s:
            single_mp = 0.
            # 单个元路径的影响, η(m)(t)=∑(e(i,j))
            train_edge = self.single_metapath_influence(s_node[0], s_node[1], t_node[0], t_node[1], metapath_temp)
            for edge_temp in train_edge:
                single_temp = self.nodes_similarity(edge_temp[0], edge_temp[1], edge_temp[2], edge_temp[3])
                single_temp = single_temp * dis_att[int(edge_temp[5]) - 1]
                if torch.cuda.is_available():
                    delta_time = torch.abs(torch.FloatTensor([int(s_t_time) - int(edge_temp[4])])).cuda(self.DID)
                else:
                    delta_time = torch.abs(torch.FloatTensor([int(s_t_time) - int(edge_temp[4])]))
                delta_s = eval('self.' + s_node[1] + '_delta')[int(s_node[0]) - 1]
                delta_t = eval('self.' + t_node[1] + '_delta')[int(t_node[0]) - 1]
                single_temp = single_temp * torch.exp((delta_s * delta_t * delta_time).neg())
                single_mp = single_mp + single_temp
            mp_type = metapath_temp['type']
            single_mp = single_mp * mp_type_att[int(mp_type[0]) - 1]
            if self.metapath_influence_method:
                mp_influ = mp_influ + single_mp
            else:
                mp_influ = mp_influ * single_mp

        p_lambda = s_t_similarity + mp_influ

        if torch.cuda.is_available():
            n_lambda = torch.FloatTensor([0.]).cuda(self.DID)
        else:
            n_lambda = torch.FloatTensor([0.])
        for [neg_node, neg_node_type] in neg_s_node:
            s_neg_similarity = self.nodes_similarity(s_node[0], s_node[1], neg_node, neg_node_type)
            if self.metapath_influence_method:
                mp_influ = 0.
            else:
                mp_influ = 1.
            for mp_temp in neg_s_metapath[neg_node_type + neg_node]:
                single_mp = 0.
                train_edge = self.single_metapath_influence(s_node[0], s_node[1], neg_node[0], neg_node_type[0],
                                                            mp_temp)
                for edge_temp in train_edge:
                    # [node id, node type, node id, node type, time, distance]
                    single_temp = self.nodes_similarity(edge_temp[0], edge_temp[1], edge_temp[2], edge_temp[3])
                    single_temp = single_temp * dis_att[int(edge_temp[5]) - 1]
                    if torch.cuda.is_available():
                        delta_time = torch.abs(torch.FloatTensor([int(s_t_time[0]) - int(edge_temp[4])])).cuda(self.DID)
                    else:
                        delta_time = torch.abs(torch.FloatTensor([int(s_t_time[0]) - int(edge_temp[4])]))
                    delta_s = eval('self.' + s_node[1] + '_delta')[int(s_node[0]) - 1]
                    delta_neg = eval('self.' + neg_node_type + '_delta')[int(neg_node) - 1]
                    single_temp = single_temp * torch.exp((delta_s * delta_neg * delta_time).neg())
                    single_mp = single_mp + single_temp
                mp_type = mp_temp['type']
                single_mp = single_mp * mp_type_att[int(mp_type) - 1]
                if self.metapath_influence_method:
                    mp_influ = mp_influ + single_mp
                else:
                    mp_influ = mp_influ * single_mp

            n_lambda = n_lambda + torch.log((s_neg_similarity + mp_influ).neg().sigmoid() + 1e-6)

        loss = -torch.log(p_lambda.sigmoid() + 1e-6) - n_lambda
        return loss

    def single_metapath_influence(self, s_node, s_type, t_node, t_type, metapath):
        """
        单个元路径的影响, η(m)=
        :param s_node: 源节点特征
        :param s_type: 源节点类型
        :param t_node: 目标节点特征
        :param t_type: 目标节点类型
        :param metapath: 元路径
        :return:
        """
        result_edge = []  # [[node id, node type, node id, node type, time, distance], ...]
        # metapath {type:, edge:, node type:, time: }
        for index in range(len(metapath['edge']) - 1):
            if (
                    (metapath['edge'][index] == s_node and metapath['node_type'][index] == s_type) and
                    (metapath['edge'][index + 1] == t_node and metapath['node_type'][index + 1] == t_type)
                    #
            ):
                s_index = index
                t_index = s_index + 1
                if s_index < (self.closest_edge / 2):
                    for i in range(s_index):
                        edge_temp = []  # [node id, node type, node id, node type, time, distance]
                        edge_temp.extend([metapath['edge'][i], metapath['node_type'][i],
                                          metapath['edge'][i + 1], metapath['node_type'][i + 1],
                                          metapath['time'][i], s_index - i])
                        result_edge.append(edge_temp)

                    for i in range(self.closest_edge - s_index):
                        edge_temp = []
                        edge_temp.extend([metapath['edge'][t_index + i], metapath['node_type'][t_index + i],
                                          metapath['edge'][t_index + i + 1], metapath['node_type'][t_index + i + 1],
                                          metapath['time'][t_index + i], i + 1])
                        result_edge.append(edge_temp)

                elif (len(metapath['edge']) - t_index - 1) < (self.closest_edge / 2):
                    for i in range(len(metapath['edge']) - 1 - t_index):
                        edge_temp = []
                        edge_temp.extend([metapath['edge'][- 2 - i], metapath['node_type'][- 2 - i],
                                          metapath['edge'][- 1 - i], metapath['node_type'][- 1 - i],
                                          metapath['time'][-1], len(metapath['edge']) - 1 - t_index - i])
                        result_edge.append(edge_temp)

                    for i in range(self.closest_edge - (len(metapath['edge']) - 1 - t_index)):
                        edge_temp = []
                        edge_temp.extend([metapath['edge'][s_index - 1 - i], metapath['node_type'][s_index - 1 - i],
                                          metapath['edge'][s_index - i], metapath['node_type'][s_index - i],
                                          metapath['time'][s_index - 1 - i], i + 1])
                        result_edge.append(edge_temp)
                else:
                    for i in range(int(self.closest_edge / 2)):
                        edge_temp = []
                        edge_temp.extend([metapath['edge'][t_index + i], metapath['node_type'][t_index + i],
                                          metapath['edge'][t_index + i + 1], metapath['node_type'][t_index + i + 1],
                                          metapath['time'][t_index + i], i + 1])
                        result_edge.append(edge_temp)
                        edge_temp = []
                        edge_temp.extend([metapath['edge'][s_index - i - 1], metapath['node_type'][s_index - i - 1],
                                          metapath['edge'][s_index - i], metapath['node_type'][s_index - i],
                                          metapath['time'][s_index - i - 1], i + 1])

        return result_edge

    def nodes_similarity(self, s_node, s_type, t_node, t_type):
        """
        计算两个节点的相似度, η(x,y)=-║u(x)-u(y)║^2
        :param s_node: 源节点特征
        :param s_type: 源节点类型
        :param t_node: 目标节点特征
        :param t_type: 目标节点类型
        :return: 节点间相似度
        """
        # 计算源节点和目标节点的嵌入，不同种类的节点嵌入系数矩阵不同
        s_emb = eval('self.' + s_type + '_emb')[int(s_node) - 1]    # [嵌入维度]
        t_emb = eval('self.' + t_type + '_emb')[int(t_node) - 1]    # [嵌入维度]

        if self.node_similarity_method:
            # matrix
            # 边类型 = 源节点类型_目标节点类型
            edge_type = s_type + '_' + t_type
            if edge_type in self.state_dict().keys():
                # 相似度 = 源节点嵌入(1,嵌入维度) * 边类型嵌入(嵌入维度,嵌入维度) * 目标节点嵌入(嵌入维度,1)
                # 边类型嵌入为可学习方阵，(嵌入维度,嵌入维度)
                similarity = torch.mm(s_emb.unsqueeze(dim=0), eval('self.' + edge_type))
                similarity = torch.tanh(torch.mm(similarity, t_emb.unsqueeze(dim=1)))
            else:
                edge_type = t_type + '_' + s_type
                similarity = torch.mm(s_emb.unsqueeze(dim=0), eval('self.' + edge_type).t())
                similarity = torch.tanh(torch.mm(similarity, t_emb.unsqueeze(dim=1)))
        else:
            # Euclidean distance
            similarity = ((s_emb - t_emb) ** 2).sum(dim=0).neg()

        return similarity.squeeze()

    def save_embedding(self):
        emb_path = './output/'
        emb_end = '.emb'
        for node_type_temp in self.node_type:
            file_temp = emb_path + node_type_temp + emb_end
            if torch.cuda.is_available():
                embedding = eval('self.' + node_type_temp + '_emb').cpu().data.numpy()
            else:
                embedding = self.node_emb.data.numpy()
            with open(file_temp, 'wt', encoding='utf-8') as f:
                f.write('%d %d\n' % (self.node_dim[node_type_temp], self.embedding_dim))
                for n_idx in range(self.node_dim[node_type_temp]):
                    f.write(' '.join(str(d) for d in embedding[n_idx]) + '\n')


def custom_collate_fn(batch):
    item = batch[0]  # 由于batch_size为1，我们只取出第一个元素
    source_node = item['source_node']
    target_node = item['target_node']
    train_time = item['train_time']
    metapath_s = item['metapath_s']
    negative_s_node = item['negative_s_node']
    negative_s_metapath = item['negative_s_metapath']

    collated_item = {
        'source_node': source_node,
        'target_node': target_node,
        'train_time': train_time,
        'metapath_s': metapath_s,
        'negative_s_node': negative_s_node,
        'negative_s_metapath': negative_s_metapath
    }
    return collated_item


class metapath_mtne_Trainer(object):
    def __init__(self, args, metapath_data, train_edges):
        self.args = args
        self.DID = self.args.DID
        self.train_edges = train_edges
        self.metapath_data = metapath_data
        self.setup_model(metapath_data, train_edges)

    def setup_model(self, metapath_data, train_edges):
        if torch.cuda.is_available():
            self.model = metapath_mtne(self.args, metapath_data, train_edges).cuda(self.DID)
        else:
            self.model = metapath_mtne(self.args, metapath_data, train_edges)

    def loss_func(self, s_node, t_node, s_t_time, metapath_s, neg_s_node, neg_s_metapath):
        loss = self.model(s_node, t_node, s_t_time, metapath_s, neg_s_node, neg_s_metapath)
        return loss

    def fit(self):
        self.model.train()
        self.opt = torch.optim.Adam(lr=self.args.initial_lr, params=self.model.parameters())
        # self.opt = torch.optim.SGD(lr=self.args.initial_lr,
        #                             params=self.model.parameters(),
        #                             weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.args.decay_epoch, gamma=0.9)
        for epoch in range(self.args.epochs):
            print('Epoch : ', epoch)
            self.data = mtne_metapath_dataset(self.args, self.metapath_data, self.train_edges)
            self.dataloader = DataLoader(self.data, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
            self.opt.zero_grad()
            accumulated_losses = 0.
            for step, data_temp in enumerate(self.dataloader):
                if step % self.args.batch_size == 0 and step != 0:
                    accumulated_losses = accumulated_losses / self.args.batch_size
                    accumulated_losses.backward()
                    self.opt.step()
                    print('Loss per batch: ', round(accumulated_losses.item(), 4))
                    accumulated_losses = 0.
                    self.opt.zero_grad()

                loss = self.loss_func(
                    s_node=data_temp['source_node'],
                    t_node=data_temp['target_node'],
                    s_t_time=data_temp['train_time'],
                    metapath_s=data_temp['metapath_s'],
                    neg_s_node=data_temp['negative_s_node'],
                    neg_s_metapath=data_temp['negative_s_metapath']
                )
                accumulated_losses = accumulated_losses + loss

            accumulated_losses = accumulated_losses / (self.data.data_size % self.args.batch_size)
            accumulated_losses.backward()
            self.opt.step()
            print('Loss per batch: ', round(accumulated_losses.item(), 4))
            self.scheduler.step()
            self.model.save_embedding()
            self.score()
        print('finish')

    def score(self):
        self.model.eval()
        # do_node_classification()
        # emb = read_embedding()
        # evalu_temp = Evaluation(emb)
        # evalu_temp.link_recommendation(5)
        # evalu_temp.link_recommendation(10)
        # test_link_pre = Evaluation(emb)
        # test_link_pre.link_prediction_with_auc()
        s_emb = H_read_embedding('./output/user.emb')
        t_emb = H_read_embedding('./output/business.emb')
        evalu_temp = H_Evaluation(s_emb, t_emb)
        evalu_temp.link_recommendation(2)
        evalu_temp.link_recommendation(4)
        self.model.train()
