class SageLayer(nn.Module):
	"""
	一层SageLayer
	"""
	def __init__(self, input_size, out_size, gcn=False):
		super(SageLayer, self).__init__()
		self.input_size = input_size
		self.out_size = out_size
		self.gcn = gcn
		self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size)) #初始化权重参数w*input.T
		self.init_params() # 调整权重参数分布

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, self_feats, aggregate_feats, neighs=None):
		"""
		Parameters:
			self_feats:源节点的特征向量
			aggregate_feats:聚合后的邻居节点特征
		"""
		if not self.gcn: # 如果不是gcn的话就要进行concatenate
			combined = torch.cat([self_feats, aggregate_feats], dim=1)
		else:
			combined = aggregate_feats
		combined = F.relu(self.weight.mm(combined.t())).t()
		return combined

class GraphSage(nn.Module):
	"""定义一个GraphSage模型"""
	def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN'):
	    super(GraphSage, self).__init__()
		self.input_size = input_size
		self.out_size = out_size
		self.num_layers = num_layers # Graphsage的层数
		self.gcn = gcn
		self.device = device
		self.agg_func = agg_func
		self.raw_features = raw_features
		self.adj_lists = adj_lists
		# 定义每一层的输入和输出
		for index in range(1, num_layers+1):
			layer_size = out_size if index != 1 else input_size
			setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))#除了第1层的输入为input_size,其余层的输入和输出均为outsize

	def forward(self, nodes_batch):
		"""
		为一批节点生成嵌入表示
		Parameters:
			nodes_batch:目标批次的节点
		"""
		lower_layer_nodes = list(nodes_batch) # 初始化第一层节点
		nodes_batch_layers = [(lower_layer_nodes,)] # 存放每一层的节点信息
		for i in range(self.num_layers):
			lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes) # 根据当前层节点获得下一层节点
			nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

		assert len(nodes_batch_layers) == self.num_layers + 1

		pre_hidden_embs = self.raw_features # 初始化h0
		for index in range(1, self.num_layers+1):
			nb = nodes_batch_layers[index][0]  #所有邻居节点
			pre_neighs = nodes_batch_layers[index-1] # 上一层的邻居节点
			aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
			sage_layer = getattr(self, 'sage_layer'+str(index))
			if index > 1:
				nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
			# self.dc.logger.info('sage_layer.')
			cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
										aggregate_feats=aggregate_feats)
			pre_hidden_embs = cur_hidden_embs

		return pre_hidden_embs

	def _nodes_map(self, nodes, hidden_embs, neighs):
		layer_nodes, samp_neighs, layer_nodes_dict = neighs
		assert len(samp_neighs) == len(nodes)
		index = [layer_nodes_dict[x] for x in nodes]
		return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes] # 获取目标节点集的所有邻居节点[[v0的邻居],[v1的邻居],[v2的邻居]]
		if not num_sample is None: # 如果num_sample为实数的话
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs] # [set(随机采样的邻居集合),set(),set()]
            # 遍历所有邻居集合如果邻居节点数>=num_sample，就从邻居节点集中随机采样num_sample个邻居节点，否则直接把邻居节点集放进去
		else:
			samp_neighs = to_neighs
		samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)] # 把源节点也放进去
		_unique_nodes_list = list(set.union(*samp_neighs)) #展平
		i = list(range(len(_unique_nodes_list))) # 重新编号
		unique_nodes = dict(list(zip(_unique_nodes_list, i)))
		return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
    """聚合邻居节点信息
    Parameters:
        nodes:从最外层开始的节点集合
        pre_hidden_embs:上一层的节点嵌入
        pre_neighs:上一层的节点
    """
		unique_nodes_list, samp_neighs, unique_nodes = pre_neighs # 上一层的源节点，...,....,
		assert len(nodes) == len(samp_neighs)
		indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))] # 判断每个节点是否出现在邻居节点中
		assert (False not in indicator)
		if not self.gcn:
			# 如果不适用gcn就要把源节点去除
			samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]
		if len(pre_hidden_embs) == len(unique_nodes):
			embed_matrix = pre_hidden_embs
		else:
			embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
		# self.dc.logger.info('3')
		mask = torch.zeros(len(samp_neighs), len(unique_nodes))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1 # 每个源节点为一行，一行元素中1对应的就是邻居节点的位置

		if self.agg_func == 'MEAN':
			num_neigh = mask.sum(1, keepdim=True) # 计算每个源节点有多少个邻居节点
			mask = mask.div(num_neigh).to(embed_matrix.device) #
			aggregate_feats = mask.mm(embed_matrix)

		elif self.agg_func == 'MAX':
			# print(mask)
			indexs = [x.nonzero() for x in mask==1]
			aggregate_feats = []
			for feat in [embed_matrix[x.squeeze()] for x in indexs]:
				if len(feat.size()) == 1:
					aggregate_feats.append(feat.view(1, -1))
				else:
					aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
			aggregate_feats = torch.cat(aggregate_feats, 0)
        return aggregate_feats
