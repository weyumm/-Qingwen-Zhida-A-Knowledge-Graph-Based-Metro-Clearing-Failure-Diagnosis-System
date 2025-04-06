import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv
import torch
import pandas as pd
import numpy as np
import gradio as gr
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
import warnings
import traceback
import requests
import math
import json
import random
import re
from typing import List
import tiktoken
import networkx as nx
# 精确导入 GlobalSearch 类
sys.path.append("E:\\work_and_study\\path_same\\mini_adjust\\graphrag-local-ollama\\graphrag-local-ollama")
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.context_builder.builders import GlobalContextBuilder
from graphrag.query.structured_search.global_search.callbacks import GlobalSearchLLMCallback

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

torch.set_float32_matmul_precision('high')  # <-- 添加在程序启动时
SEED = 42
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
def set_deterministic(seed):
    # Python/Random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

set_deterministic(SEED)

# 初始化特征池大小
init_pool_size = 2000  # 定义一个初始值


def check_service():
    try:
        response = requests.get("http://localhost:11434/", timeout=5)
        return True
    except ConnectionError:
        print("❌ 无法连接到Ollama服务，请确认：")
        print("1. 已执行 'ollama serve' 启动服务")
        print("2. 防火墙允许11434端口通信")
        return False

def compute_loss(model, data, epoch, is_train=True):
    total_loss = 0
    edge_types = data.metadata()[1]
    
    # 单次前向传播（移除不必要的detach）
    h_dict, l2_reg = model(data.x_dict, data.edge_index_dict)
    
    for edge_type in edge_types:
        edge_index = data[edge_type].edge_index
        if edge_index.numel() == 0:
            continue
            
        # 正样本损失（增加有效性检查）
        pos_pred = model.predictor(edge_index, h_dict, edge_type)
        if torch.isnan(pos_pred).any():
            raise ValueError("正样本预测出现NaN值")
            
        # 负采样（增加采样比例）
        # 修改负采样比例为指数衰减
        if epoch < 35:
            neg_ratio = 20 * (0.97**epoch) + 5  # 初始更高采样率
        else:
            neg_ratio = 24 + 24/(1 + math.exp((epoch-100)/20))  # S型衰减曲线
        num_neg = int(edge_index.size(1) * neg_ratio)
        neg_edge = negative_sampling(
            edge_index, 
            num_neg_samples=num_neg,
            num_nodes=(data[edge_type[0]].num_nodes, data[edge_type[2]].num_nodes),
            force_undirected=False  # 避免生成重复负样本
        )
        
        # 损失计算（增加梯度检查）
        current_pos_weight = max(2.0, 5.0 * (0.995 ** epoch))  # 添加最低限制
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_pred, 
            torch.ones_like(pos_pred),
            pos_weight=torch.tensor([current_pos_weight])
        )

        neg_loss = F.binary_cross_entropy_with_logits(
            model.predictor(neg_edge, h_dict, edge_type),
            torch.zeros_like(neg_edge[0].float())
        )
        pos_weight = torch.tensor([3.0])  # 固定权重
        neg_weight = torch.tensor([0.8])
        total_loss += pos_loss * pos_weight + neg_loss * neg_weight

    # 正则项（仅作用于可训练参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # 在compute_loss中增加动态正则化系数
    l2_reg = sum(p.norm() for p in trainable_params)
    # 修改compute_loss中的正则化系数计算
    base_weight = 0.001
    if epoch < 100:  
        reg_weight = base_weight * 0.1
    else:
        reg_weight = base_weight * (1 - 0.99**epoch)  # 渐进增强
    # 添加特征余弦相似度正则
    cos_sim = F.cosine_similarity(h_dict['station'][:, None], h_dict['station'], dim=-1)
    sim_reg = torch.triu(cos_sim, diagonal=1).mean()
    total_loss += l2_reg * min(reg_weight, 0.005)  # 上限0.005
    total_loss += 0.01 * sim_reg  # 抑制过度相似
    
    return total_loss / len(edge_types) + l2_reg * 1e-4

def compute_single_loss(model, x_dict, edge_index_dict, edge_type, num_samples=10):
    """计算节点对的交叉熵损失（多次采样取平均）"""
    total_loss = 0.0
    valid_samples = 0
    
    # 强制模型进入评估模式
    model.eval()
    
    # 预检查边索引有效性
    edge_index = edge_index_dict.get(edge_type, None)
    if edge_index is None or edge_index.numel() == 0:
        raise ValueError(f"边类型 {edge_type} 不存在或为空")

    with torch.no_grad():  # 禁用梯度计算
        for _ in range(num_samples):
            try:
                # 每次前向传播都重新计算（确保Dropout等随机性生效）
                h_dict, _ = model(x_dict, edge_index_dict)
                
                # 维度验证
                src_type, _, dst_type = edge_type
                assert h_dict[src_type].shape[0] > edge_index[0].max(), "源节点索引越界"
                assert h_dict[dst_type].shape[0] > edge_index[1].max(), "目标节点索引越界"
                
                # 计算正样本损失
                pos_pred = model.predictor(edge_index, h_dict, edge_type)
                current_loss = F.binary_cross_entropy_with_logits(
                    pos_pred, 
                    torch.ones_like(pos_pred)
                ).item()
                
                total_loss += current_loss
                valid_samples += 1
            except Exception as e:
                print(f"第 {_+1} 次采样失败: {str(e)}")
                continue

    # 有效性检查
    if valid_samples == 0:
        raise RuntimeError("所有采样尝试均失败")
    
    return total_loss / valid_samples

# 改进的RGCN模型
class EnhancedRGCN(nn.Module):
    def __init__(self, dynamic_graph, in_channels=32, hidden_channels=32, out_channels=32, use_enhanced_predictor=True):  # 调整参数顺序
        super().__init__()
        # 添加维度断言
        sample_feat = next(iter(dynamic_graph.external_features.values()))
        assert sample_feat.shape[-1] == in_channels, "输入通道不匹配"
        
        self.dynamic_graph = dynamic_graph
        
        # 获取所有边类型的三元组表示
        edge_types = list(dynamic_graph.rel_map.keys())

        # 调整GAT层结构
        self.conv1 = HeteroConv({
            edge_type: GATConv(
                in_channels=in_channels,
                out_channels=16,  # 保持维度一致
                heads=2,
                negative_slope=0.01, 
                dropout=0.1,
                add_self_loops=False,
                concat=True  
            ) for edge_type in edge_types
        }, aggr='mean')

        # 添加批归一化层
        self.bn1 = nn.ModuleDict({
            node_type: nn.BatchNorm1d(
                32, 
                eps=1e-3,  # 防止除零
                momentum=0.3,
                affine=True,
                track_running_stats=True  # 禁用运行统计
            ) for node_type in dynamic_graph.type_nodes
        })

        self.conv2 = HeteroConv({
            edge_type: GATConv(
                in_channels=32,
                out_channels=16,
                heads=2,
                add_self_loops=False,  # 不添加自环边
                concat=True
            ) for edge_type in edge_types
        }, aggr='mean')

        self.bn2 = nn.ModuleDict({
            node_type: nn.BatchNorm1d(32, momentum=0.3)
            for node_type in dynamic_graph.type_nodes
        })

        # 添加第三层GAT
        self.conv3 = HeteroConv({
            edge_type: GATConv(
                in_channels=32,  # 输入通道
                out_channels=out_channels,
                heads=1,
                add_self_loops=False,  # 不添加自环边
                concat=False
            ) for edge_type in edge_types
        }, aggr='mean')
        
        # 使用Kaiming初始化
        for name, conv in self.conv1.convs.items():
            if hasattr(conv, 'lin_src') and hasattr(conv.lin_src, 'weight'):
                nn.init.kaiming_normal_(conv.lin_src.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(conv.lin_src.bias, 0.1)
            if hasattr(conv, 'lin_dst') and hasattr(conv.lin_dst, 'weight'):
                nn.init.kaiming_normal_(conv.lin_dst.weight, mode='fan_out', nonlinearity='leaky_relu')
            if hasattr(conv, 'att'):
                nn.init.normal_(conv.att, mean=0.0, std=0.05)  
                conv.att.requires_grad_(True)
            # 添加注意力偏置初始化
            if hasattr(conv, 'bias') and conv.bias is not None:
                nn.init.constant_(conv.bias, 0.1)

        for name, conv in self.conv2.convs.items():
            if hasattr(conv, 'lin_src') and hasattr(conv.lin_src, 'weight'):
                nn.init.kaiming_normal_(conv.lin_src.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(conv.lin_src.bias, 0.1)
            if hasattr(conv, 'lin_dst') and hasattr(conv.lin_dst, 'weight'):
                nn.init.kaiming_normal_(conv.lin_dst.weight, mode='fan_out', nonlinearity='leaky_relu')
            if hasattr(conv, 'att'):
                nn.init.normal_(conv.att, mean=0.0, std=0.05) 
                conv.att.requires_grad_(False)  # 新增冻结
            # 添加注意力偏置初始化
            if hasattr(conv, 'bias') and conv.bias is not None:
                nn.init.constant_(conv.bias, 0.1)

        for name, conv in self.conv3.convs.items():
            if hasattr(conv, 'lin_src') and hasattr(conv.lin_src, 'weight'):
                nn.init.xavier_uniform_(conv.lin_src.weight, gain=1.414)
                nn.init.kaiming_normal_(conv.lin_src.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(conv.lin_src.bias, 0.01)
            if hasattr(conv, 'lin_dst') and hasattr(conv.lin_dst, 'weight'):
                nn.init.kaiming_normal_(conv.lin_dst.weight, mode='fan_out', nonlinearity='leaky_relu')
            if hasattr(conv, 'att'):
                nn.init.normal_(conv.att, mean=0.0, std=0.05) 
                conv.att.requires_grad_(False)  # 新增冻结
            # 添加注意力偏置初始化
            if hasattr(conv, 'bias') and conv.bias is not None:
                nn.init.constant_(conv.bias, 0.1)
            
        # 添加Dropout层
        self.dropout = nn.Dropout(p=0.3)
        # 改进残差连接结构（替换原res_gate）
        self.res_gate = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(32, 32),
                nn.LayerNorm(32),
                nn.ELU(),
                nn.Linear(32, 32)
            ) for node_type in dynamic_graph.type_nodes
        })

        # 添加梯度监控与裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

        # 添加特征增强层
        self.feature_enhancer = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(32, 64),
                nn.GELU(),
                nn.Linear(64, 32)
            ) for node_type in dynamic_graph.type_nodes
        })

        # 可学习残差系数
        self.res_alpha = nn.Parameter(torch.tensor(0.7))

        # 添加边类型转字符串的辅助方法
        def _et_to_str(edge_type):
            """将边类型元组转换为合法字符串键"""
            return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"

        # 修改ParameterDict的键类型
        self.edge_attention = nn.ParameterDict({
            _et_to_str(et): nn.Parameter(torch.randn(1))  # 转换为字符串键
            for et in edge_types
        })

        # 替换归一化层为LayerNorm
        self.norm = nn.ModuleDict({
            node_type: nn.LayerNorm(32, elementwise_affine=True) 
            for node_type in dynamic_graph.type_nodes
        })

        # 添加边类型激活器
        self.edge_activator = nn.ParameterDict({
            f"{et[0]}__{et[1]}__{et[2]}": nn.Parameter(torch.ones(1))
            for et in edge_types
        })

        self.freeze_layers = False
        # 修改预测器选择逻辑
        if use_enhanced_predictor:
            self.predictor = EnhancedPredictor(edge_types=edge_types)
        else:
            self.predictor = HeteroDotProductPredictor(edge_types=edge_types)
        
        for node_type in dynamic_graph.type_nodes:
            nn.init.kaiming_normal_(
                self.res_gate[node_type][0].weight,
                mode='fan_in',
                nonlinearity='relu'
            )
            nn.init.zeros_(self.res_gate[node_type][0].bias)
            nn.init.normal_(
                self.res_gate[node_type][-1].weight,
                mean=0.0,
                std=0.01
            )

        print("\n=== 参数可训练状态 ===")
        for name, param in self.named_parameters():
            if "conv3" in name:
                print(f"{name}: requires_grad={param.requires_grad}, mean={param.data.mean():.4f}")

    def forward(self, x_dict, edge_index_dict):
        # 过滤空边索引
        valid_edge_types = [
            et for et, idx in edge_index_dict.items() 
            if idx is not None and idx.size(1) > 0
        ]

        # 打印边索引
        for k, v in edge_index_dict.items():
            print(f"边索引 - 边类型: {k}, 索引: {v[:5]}")
        
        # 修改前向传播
        print("\n=== GATConv Input ===")
        for node_type, x in x_dict.items():
            print(f"Node Type: {node_type}, First 5 Features: {x[:5] if x is not None else 'MISSING'}")

        if not x_dict or not edge_index_dict:
            raise ValueError("输入的特征字典或边索引字典为空！")

        # 修改第一层卷积层前向传播
        h1 = {}
        for edge_type in valid_edge_types:
            conv = self.conv1.convs[edge_type]
            src_type = edge_type[0]
            with torch.backends.cudnn.flags(enabled=False):
                h1[src_type] = conv(
                    x_dict[src_type],
                    edge_index_dict[edge_type]
                )
                et_str = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"  # 保持格式一致
                active_weight = torch.tanh(self.edge_activator[et_str])  # [-1,1]范围激活
                if active_weight < 0.1:  # 动态屏蔽无效边类型
                    continue
                h1[src_type] = h1[src_type] * active_weight

        # 梯度增强技巧（保持前向传播特性同时改善梯度流）
        h1 = {k: h + 0.05 * h.detach() for k, h in h1.items()}

        print("=== Before ELU ===")
        for node_type, h in h1.items():
            print(f"Node Type: {node_type}, Mean: {h.mean()}, Min: {h.min()}, Max: {h.max()}")

        if self.training:
            # 训练模式逻辑
            h1 = {k: F.gelu(self.bn1[k](h)) for k, h in h1.items()}  # 改用GELU激活
            h1 = {k: self.dropout(h) for k, h in h1.items()}  # 仅训练时启用dropout
        else:
            # 评估模式逻辑
            with torch.no_grad():
                h1 = {k: F.elu(h) for k, h in h1.items()}  # 评估模式跳过BN
        print("=== After elu ===")
        for node_type, h in h1.items():
            print(f"Node Type: {node_type}, Mean: {h.mean()}, Min: {h.min()}, Max: {h.max()}")
        # 添加特征裁剪
        h1 = {k: torch.clamp(h, min=-3.0, max=3.0) for k, h in h1.items()}  # 控制特征范围
        print("=== After Clamp ===")
        for node_type, h in h1.items():
            print(f"Node Type: {node_type}, Mean: {h.mean()}, Min: {h.min()}, Max: {h.max()}")
        print("\n=== After Conv1 ===")
        for node_type, x in h1.items():
            print(f"Node Type: {node_type}, First 5 Features: {x[:5] if x is not None else 'MISSING'}")

        # 保留未更新节点特征
        for node_type in x_dict:
            if node_type not in h1:
                h1[node_type] = x_dict[node_type]
                print(f"保留初始特征: {node_type}")
        
        # 修改后的残差连接部分
        for k in x_dict:
            if k in h1 and h1[k].shape == x_dict[k].shape:
                # 确保输入维度匹配
                if h1[k].size(1) != self.res_gate[k][0].in_features:
                    raise ValueError(f"特征维度不匹配！节点类型 {k} 需要维度 {self.res_gate[k][0].in_features}")
                # 增强特征差分项
                delta = x_dict[k] - h1[k].detach()
                gate = torch.sigmoid(self.res_gate[k](delta))
                h1[k] = h1[k] + 0.5 * gate * delta  # 动态调整残差强度

        # 添加特征噪声注入
        if h1[k].std() < 0.5:
            noise = torch.randn_like(h1[k]) * 0.5
            h1[k] = h1[k] + noise
            print(f"注入噪声到 {k} 节点特征")

        # 修改第二层卷积层前向传播
        h2_input = h1
        h2 = {}
        for edge_type, conv in self.conv2.convs.items():
            # 强制使用确定性的注意力计算
            with torch.backends.cudnn.flags(enabled=False):
                h2[edge_type[0]] = conv(
                    h2_input[edge_type[0]], 
                    edge_index_dict[edge_type]
                )
        h2 = {k: self.bn2[k](F.leaky_relu(h, 0.5)) for k, h in h2.items()}
        gate = {k: torch.sigmoid(self.res_gate[k](h2[k])) for k in h2}
        h2 = {k: gate[k] * h2[k] + (1 - gate[k]) * h1.get(k, torch.zeros_like(h2[k])) for k in h2}

        print("\n=== After Conv2 ===")
        for node_type, x in h2.items():
            print(f"Node Type: {node_type}, First 5 Features: {x[:5] if x is not None else 'MISSING'}")
        # 保留未更新节点特征
        for node_type in h1:
            if node_type not in h2:
                h2[node_type] = h1[node_type]
                print(f"保留第1层特征: {node_type}")
        
        # 修改第三层卷积层前向传播
        h3 = {}
        for edge_type, conv in self.conv3.convs.items():
            # 强制使用确定性的注意力计算
            with torch.backends.cudnn.flags(enabled=False):
                h3[edge_type[0]] = conv(
                    {k: self.dropout(v) for k, v in h2.items()}[edge_type[0]], 
                    edge_index_dict[edge_type]
                )
        # 增加跨层残差
        for node_type in h3:
            if node_type in x_dict:
                h3[node_type] = h3[node_type] + x_dict[node_type] * 0.5  # 可学习系数更佳
        # 最终特征保留
        for node_type in h2:
            if node_type not in h3:
                h3[node_type] = h2[node_type]
                print(f"保留第2层特征: {node_type}")
        print("\n=== After Conv3 ===")
        for node_type, x in h3.items():
            act_ratio = (x > 0).float().mean()
            print(f"{node_type}激活率：{act_ratio.item():.2%}")
            if act_ratio < 0.3:
                x = x + torch.randn_like(x) * 0.5  # 激活不足时添加噪声
            print(f"Node Type: {node_type}, First 5 Features: {x[:5] if x is not None else 'MISSING'}")

        # 添加边索引检查
        print("\n=== Edge Indexes ===")
        for et, idx in edge_index_dict.items():
            print(f"{et}: {idx.shape if idx is not None else 'MISSING'}")
            if idx is not None and idx.numel() > 0:
                src_type, _, dst_type = et
                print(f"  Source nodes: {src_type} (exists: {src_type in x_dict})")
                print(f"  Target nodes: {dst_type} (exists: {dst_type in x_dict})")
        # 消息传递验证
        for et, idx in edge_index_dict.items():
            if idx.numel() > 0:
                src_type, _, dst_type = et
                src_feat = x_dict.get(src_type)
                dst_feat = x_dict.get(dst_type)
                if src_feat is None or dst_feat is None:
                    warnings.warn(f"边类型 {et} 存在未初始化节点特征")
                else:
                    print(f"边类型 {et} 消息传递验证:")
                    print(f"  源特征范围: [{src_feat.min():.3f}, {src_feat.max():.3f}]")
                    print(f"  目标特征范围: [{dst_feat.min():.3f}, {dst_feat.max():.3f}]")

        # 添加特征差异检查
        for layer_name, h_dict in zip(['Conv1', 'Conv2', 'Conv3'], [h1, h2, h3]):
            for node_type, feat in h_dict.items():
                if feat.requires_grad:
                    feat_std = feat.std().item()
                    print(f"{layer_name} {node_type} 特征标准差: {feat_std:.4f}")
                    if feat_std < 0.5:
                        feat.data += torch.randn_like(feat) * 1  # 添加噪声

        # 添加梯度监控
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"梯度异常参数: {name}")
                param.grad[torch.isnan(param.grad)] = 0  # 梯度截断

        missing_types = []
        for node_type in x_dict:
            if node_type not in h3:
                missing_types.append(node_type)
                h3[node_type] = x_dict[node_type]  # 回退到初始特征
                print(f"[Final] Restored missing {node_type}")
        if missing_types:
            print(f"警告: 以下类型通过回退保留: {missing_types}")

        return h3, sum(p.norm() for p in self.parameters() if p.requires_grad)

# 增强的预测器
class EnhancedPredictor(nn.Module):
    def __init__(self, edge_types, feat_dim=32):
        super().__init__()
        # 新增关系类型注册方法
        # 使用OrderedDict保持顺序
        self.edge_type_map = nn.ParameterDict()
        self.feat_dim = feat_dim
        
        # 初始化已知关系类型
        for edge_type in edge_types:
            self.register_edge_type(edge_type)

        # 动态注册方法
        self.register_buffer('new_relations', torch.zeros(0))  # 占位符

        # 参数命名包含完整类型信息
        self.relation_weights = nn.ParameterDict({
            f"{src}_{rel}_{dst}": nn.Parameter(torch.randn(feat_dim))
            for (src, rel, dst) in edge_types
        })

        # 修改后的门控网络
        self.gate = nn.Sequential(
            nn.Linear(feat_dim*3, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def register_edge_type(self, edge_type):
        """动态注册新关系类型"""
        if str(edge_type) not in self.edge_type_map:
            # 生成标准化参数名（避免元组作为key的问题）
            param_name = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
            self.edge_type_map[param_name] = nn.Parameter(
                torch.randn(self.feat_dim) * (1.0 / math.sqrt(self.feat_dim))
            )

    def forward(self, edge_index, h_dict, edge_type):
        # 动态注册检查
        if str(edge_type) not in self.edge_type_map:
            self.register_edge_type(edge_type)

        src_type, rel_type, dst_type = edge_type
        
        # 新增索引范围校验
        max_src_idx = edge_index[0].max().item()
        if h_dict[src_type].shape[0] <= max_src_idx:
            raise RuntimeError(
                f"源节点索引越界: {src_type}类型只有{h_dict[src_type].shape[0]}个节点, "
                f"但存在索引{max_src_idx}"
            )

        max_dst_idx = edge_index[1].max().item()
        if h_dict[dst_type].shape[0] <= max_dst_idx:
            raise RuntimeError(
                f"目标节点索引越界: {dst_type}类型只有{h_dict[dst_type].shape[0]}个节点, "
                f"但存在索引{max_dst_idx}"
            )

        # 验证特征是否存在
        if src_type not in h_dict or dst_type not in h_dict:
            raise KeyError(f"节点类型 {src_type} 或 {dst_type} 不在特征字典中")
            
        # 获取节点特征
        src_features = h_dict[src_type][edge_index[0]]
        dst_features = h_dict[dst_type][edge_index[1]]
        
        param_name = f"{src_type}_{rel_type}_{dst_type}"
        
        # 获取关系权重（新增存在性检查）
        if param_name not in self.edge_type_map:
            raise KeyError(f"关系参数 {param_name} 未注册！")
            
        rel_weight = self.edge_type_map[param_name]
        
        interaction = torch.cat([src_features, dst_features, rel_weight.expand_as(src_features)], dim=-1)
        gate_value = self.gate(interaction)

        # 计算得分
        return (gate_value * (src_features * dst_features * rel_weight)).sum(dim=-1)
    
# 动态图管理器
class DynamicGraph(nn.Module):
    def __init__(self, max_nodes=5000):
        super().__init__()
        # 参数定义
        self.node_counter = 0  # 全局节点计数器
        self.node_feat_pool = nn.ParameterDict()  # 改为ParameterDict存储
        self.static_feat = nn.ParameterList()
        self.type_embeddings = nn.Embedding(10, 16)
        self.node_name_to_idx = {}  # 记录节点名到参数索引的映射
        self.max_nodes = max_nodes
        # 改为每个类型独立维护节点ID
        self.type_node_counters = defaultdict(int)  # 新增类型节点计数器
        self.node_map = {}  # 结构改为 {node_name: {'type': ..., 'type_id': ...}}
        self.type_nodes = defaultdict(list)  # 新增类型节点记录
        self.rel_map = {}  # 改为存储三元组关系
        self.node_counter = 0
        # 添加边索引字典初始化
        self.edge_index_dict = {}  # 新增此行
        # 使用ModuleDict管理节点特征
        self.external_features = nn.ParameterDict()  # 改为ParameterDict
        self.self_loops = defaultdict(list)  # 新增自环记录
        self.NODE_TYPE_MAP = {'network':0, 'factor':1,'station':2,'line':3} 
        self.REL_TYPE_MAP = {
            ('network', 'self_loop', 'network'): 0,
            ('network', 'has', 'line'): 1,
            ('line', 'self_loop', 'line'): 2,
            ('line', 'has', 'station'): 3,
            ('station', 'next_station', 'station'): 4,
            ('station', 'self_loop', 'station'): 5,
            ('line', 'etype', 'factor'): 6,
            ('factor', 'self_loop', 'factor'): 7,
            ('station', 'etype', 'factor'): 8,
            ('factor', 'influence', 'factor'): 9,
            ('station', 'influence', 'factor'): 10,
            ('line', 'influence', 'factor'): 11,
            ('station', 'transfer', 'network'): 12,
        }
        # 新增类型映射字典
        self.type_map = {
            'network': 0, 
            'factor': 1, 
            'station':2,
            "line":3
        }
        # 确保初始化所有可能的边类型为空的张量
        for edge_type in self.REL_TYPE_MAP.keys():
            self.edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long)
        # 在DynamicGraph初始化时检查
        all_rel_types = defaultdict(list)
        for et in self.rel_map.keys():
            all_rel_types[et[1]].append(et)  # et[1]是rel_type

        # 确保每个rel_type唯一对应一个三元组
        for rel, et_list in all_rel_types.items():
            if len(et_list) > 1:
                raise ValueError(f"关系类型'{rel}'存在歧义: {et_list}")


    def _get_or_create_index(self, node_name):
        """动态扩展特征池的核心方法"""
        if node_name not in self.node_name_to_idx:
            # 动态扩展特征池
            new_id = len(self.node_name_to_idx)
            # 动态添加特征
            self.node_feat_pool.append(nn.Parameter(torch.randn(6)*5, requires_grad=True))
            self.static_feat.append(nn.Parameter(torch.randn(2)*5, requires_grad=True))
            self.node_name_to_idx[node_name] = new_id
        return self.node_name_to_idx[node_name]
        
    def add_external_features(self, node_name, features):
        """添加外部特征（如客流量、设备状态等）"""
        self.external_features[node_name] = features
        
    def get_node_id(self, node_name, node_type):
        node_name = str(node_name).strip()
        if not node_name:  # 防止空字符串
            raise ValueError("Node name cannot be empty")
        if node_name not in self.node_map:
            # 使用类型专属的ID计数器
            type_id = len(self.type_nodes.get(node_type, []))
            self.node_map[node_name] = {
                'id': len(self.node_map),        # 全局唯一ID
                'type_id': type_id,              # 类型内部ID（从0开始）
                'type': node_type
            }
            if node_type not in self.type_nodes:
                self.type_nodes[node_type] = []
            self.type_nodes[node_type].append(node_name)
            print(f"新增节点: {node_name}({node_type}) | 全局ID={len(self.node_map)-1} 类型ID={type_id}")  # 调试输出
            self.type_node_counters[node_type] += 1  # 递增类型计数器
            self._init_node_features(node_name, node_type)
        return self.node_map[node_name]
    
    @property
    def device(self):
        """通过嵌入层自动获取设备信息"""
        return self.type_embeddings.weight.device

    def _init_node_features(self, node_name, node_type):
        """为每个节点生成唯一特征（修复张量类型问题）"""
        # 使用统一设备
        device = self.type_embeddings.weight.device
        # 确保节点已注册
        if node_name not in self.node_map:
            self.node_map[node_name] = {
                'type': node_type,
                'type_id': len(self.type_nodes[node_type])
            }
        
        # 添加类型相关初始化系数
        init_scale = {
            'network': 0.5,
            'factor': 1,
            'station':1,
            'line':0.6
        }.get(node_type, 0.5)

        # 生成随机特征
        learnable = nn.Parameter(torch.randn(6)) # 每次新建参数
        static = nn.Parameter(torch.randn(2))

        # 位置编码增加随机性
        pos = self.node_map[node_name]['type_id']
        
        # 位置编码参数化
        pos_enc = torch.zeros(8, device=device)
        # 在位置编码后添加高斯噪声
        pos_enc += torch.randn_like(pos_enc) * 0.3  # 新增噪声项
        for i in range(4):
            freq = 10000 ** (2*i / 8)
            pos_enc[2*i] = math.sin(pos / freq) + torch.randn(1)*0.1  # 添加噪声
            pos_enc[2*i+1] = math.cos(pos / freq) + torch.randn(1)*0.1

        # 类型嵌入（确保可训练）
        type_embed = self.type_embeddings(
            torch.tensor([self._get_type_id(node_type)], device=device)
        ).squeeze(0)
        
        # 组合特征（全部为张量操作）
        combined = torch.cat([
            learnable,
            self.type_embeddings(torch.tensor(self._get_type_id(node_type))),
            pos_enc,
            static
        ])

        self.external_features[node_name] = combined

    def _get_type_id(self, node_type):
        """获取类型ID"""
        # 使用类属性中的type_map
        return self.type_map.get(node_type, 9)  # 未知类型默认为9
    
    def register_relation(self, src_type, rel_type, dst_type):
        """注册关系三元组"""
        edge_type = (src_type, rel_type, dst_type)
        if edge_type not in self.rel_map:
            self.rel_map[edge_type] = len(self.rel_map)
        return edge_type
    
    def add_self_loops_for_all_nodes(self):
        """确保所有节点都有自环边"""
        for node_type in self.type_nodes:
            nodes = self.type_nodes[node_type]
            if not nodes:  # 处理空类型的情况
                continue
            edge_type = (node_type, "self_loop", node_type)
            self.register_relation(*edge_type)
            # 确保至少存在一个自环边
            if not self.self_loops[edge_type]:
                sample_id = self.node_map[nodes[0]]['type_id']
                self.self_loops[edge_type].append((sample_id, sample_id))
        # 更新边索引
        self.edge_index_dict.update({
            et: torch.tensor(edges).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)
            for et, edges in self.self_loops.items()
        })
        print(f"自环边更新完成，边类型: {list(self.self_loops.keys())}")

    def check_node_exists(self, node_name):
        return node_name in self.node_map

    def add_new_node(self, node_name, node_type, features=None):
        """添加新节点"""
        if self.check_node_exists(node_name):
            raise ValueError(f"节点 '{node_name}' 已存在")
        
        # 添加节点特征
        if features is not None:
            if not isinstance(features, nn.Parameter):
                features = nn.Parameter(features, requires_grad=True)
            self.external_features[node_name] = features
        
        # 更新节点映射
        self.node_map[node_name] = {
            'id': self.node_counter,
            'type': node_type,
            'type_id': len(self.type_nodes[node_type])
        }
        self.type_nodes[node_type].append(node_name)
        self.node_counter += 1
        
        # 移除默认特征设置，确保只使用_init_node_features生成的特征
        if node_name not in self.external_features:
            self._init_node_features(node_name, node_type)
        
        print(f"节点 '{node_name}' 添加成功")

    def add_edge(self, src_name, dst_name, rel_type):
        """添加边"""
        if not self.check_node_exists(src_name) or not self.check_node_exists(dst_name):
            raise ValueError("源节点或目标节点不存在")
        # 获取源节点和目标节点的类型内ID
        src_info = self.get_node_id(src_name, self.node_map[src_name]['type'])
        dst_info = self.get_node_id(dst_name, self.node_map[dst_name]['type'])
        # 获取节点类型
        src_type = src_info['type']
        dst_type = dst_info['type']
        # 验证索引范围
        src_id = src_info['type_id']
        dst_id = dst_info['type_id']
        print(f"添加边验证: {src_name}({src_type}-{src_id}) → {dst_name}({dst_type}-{dst_id})")
        print(f"{src_type}节点数: {len(self.type_nodes[src_type])} | {dst_type}节点数: {len(self.type_nodes[dst_type])}")  # 新增
        # 添加严格断言
        assert src_id < len(self.type_nodes[src_type]), \
            f"源ID越界: {src_id} >= {len(self.type_nodes[src_type])}"
        assert dst_id < len(self.type_nodes[dst_type]), \
            f"目标ID越界: {dst_id} >= {len(self.type_nodes[dst_type])}"
        # 新增索引范围校验
        if src_id >= len(self.type_nodes[src_type]):
            raise ValueError(f"源节点越界: {src_type}类型最大ID为{len(self.type_nodes[src_type])-1}, 当前{src_id}")
        if dst_id >= len(self.type_nodes[dst_type]):
            raise ValueError(f"目标节点越界: {dst_type}类型最大ID为{len(self.type_nodes[dst_type])-1}, 当前{dst_id}")

        # 构建边索引（使用正确的类型内ID）
        edge_type = (src_type, rel_type, dst_type)
        # 强制转换为LongTensor并验证维度
        new_edge = torch.tensor([[src_id], [dst_id]], dtype=torch.long)
        if self.edge_index_dict[edge_type].dim() != 2:
            self.edge_index_dict[edge_type] = torch.empty((2,0), dtype=torch.long)
            
        # 更新边索引字典
        if edge_type in self.edge_index_dict:
            self.edge_index_dict[edge_type] = torch.cat(
                [self.edge_index_dict[edge_type].long(), new_edge], dim=1
            )
        else:
            self.edge_index_dict[edge_type] = new_edge
            
        # 新增有效性检查
        print(f"当前边 {edge_type} 索引内容示例:", self.edge_index_dict[edge_type][:, :3])
        print(f"边 {edge_type} 索引更新: {self.edge_index_dict[edge_type].shape}")
        print(f"边 '{src_name} -> {dst_name}' 添加成功")
        


class HeteroDotProductPredictor(nn.Module):
    """
    轻量级点积关系预测器（兼容原始训练代码）
    """
    def forward(self, edge_index, h_dict, rel_type):
        src_type, _, dst_type = self.edge_type_map[rel_type]
        src_feat = h_dict[src_type][edge_index[0]]
        dst_feat = h_dict[dst_type][edge_index[1]]
        return (src_feat * dst_feat).sum(dim=1)

                            ###
# 添加模型代码路径
sys.path.append('E:\\work_and_study\\path_same\\mini_adjust')

# 常量定义
MODEL_PATH = 'E:\\work_and_study\\path_same\\mini_adjust\\qingfenzhidaRGCN_2.pth'
DATA_PATH = 'E:\\work_and_study\\path_same\\mini_adjust\\表5.20 数据预处理三元组结构_new.xlsx'

# 类型映射（与训练代码严格一致）
NODE_TYPE_MAP = {'network': 0, 'factor': 1,'station':2,"line":3}
REL_TYPE_MAP = {
            ('network', 'self_loop', 'network'): 0,
            ('network', 'has', 'line'): 1,
            ('line', 'self_loop', 'line'): 2,
            ('line', 'has', 'station'): 3,
            ('station', 'next_station', 'station'): 4,
            ('station', 'self_loop', 'station'): 5,
            ('line', 'etype', 'factor'): 6,
            ('factor', 'self_loop', 'factor'): 7,
            ('station', 'etype', 'factor'): 8,
            ('factor', 'influence', 'factor'): 9,
            ('station', 'influence', 'factor'): 10,
            ('line', 'influence', 'factor'): 11,
            ('station', 'transfer', 'network'):12
}

# 定义所有关系类型
all_edge_types = [
    ('network', 'self_loop', 'network'),
    ('network', 'has', 'line'),
    ('line', 'self_loop', 'line'),
    ('line', 'has', 'station'),
    ('station', 'next_station', 'station'),
    ('station', 'self_loop', 'station'),
    ('line', 'etype', 'factor'),
    ('factor', 'self_loop', 'factor'),
    ('station', 'etype', 'factor'),
    ('factor', 'influence', 'factor'),
    ('station', 'influence', 'factor'),
    ('line', 'influence', 'factor'),
    ('station', 'transfer', 'network')
]

class IncrementalTrainer:
    def __init__(self, model):
        self.model = model
        self.init_trainable_params()
        
        # 优化器配置
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=0.1,        # 提高初始学习率
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        self.grad_clip = 5.0  # 梯度裁剪阈值
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,        # 缩短周期
            T_mult=2,
            eta_min=1e-5
        )
    
    def init_trainable_params(self):
        """动态收集可训练参数"""
        self.trainable_params = []
        # 始终包含预测器和最后两层卷积
        for name, param in self.model.named_parameters():
            if 'predictor' in name or 'conv2' in name or 'conv3' in name:
                param.requires_grad = True
                self.trainable_params.append(param)
                print(f"可训练参数: {name}")
                
    def update_parameters(self):
        """当添加新关系后调用，更新优化器"""
        self.init_trainable_params()  # 重新收集参数
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=0.1,  # 可调整学习率
            weight_decay=1e-5
        )
        print(f"优化器已更新，当前参数数量: {len(self.trainable_params)}")
        
    def incremental_train(self, data, epochs=100):
        """增强的训练循环"""
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # 动态调整学习率
            curr_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | LR: {curr_lr:.6f}")
            
            # 前向传播
            h_dict, _ = self.model(data.x_dict, data.edge_index_dict)
            
            # 损失计算（需包含新关系）
            loss = compute_loss(self.model, data, epoch)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪与更新
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            
            # 监控新关系梯度
            for name, param in self.model.predictor.edge_type_map.items():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"关系 {name} 梯度范数: {grad_norm:.6f}")

class PredictionSystem:
    def __init__(self):
        self.dynamic_graph = DynamicGraph()
        self._load_base_data()

        # 在加载基础数据后调用
        self.dynamic_graph.add_self_loops_for_all_nodes()

        print("自环边示例:", self.dynamic_graph.self_loops[('network', 'self_loop', 'network')][:3])
        # 注册所有关系类型
        for edge_type in all_edge_types:
            self.dynamic_graph.register_relation(*edge_type)

        # 保持与训练完全一致的模型定义
        self.model = EnhancedRGCN(
            dynamic_graph=self.dynamic_graph,
            use_enhanced_predictor=True
        )
        
        self.trainer = IncrementalTrainer(self.model)  # 新增训练器

        # 加载预训练模型
        pretrained_dict = torch.load(MODEL_PATH)
        model_dict = self.model.state_dict()

        # 过滤掉与当前模型不匹配的参数
        pretrained_dict_filtered = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and 
            model_dict[k].size()[:len(v.size())] == v.size()
        }
        model_dict.update(pretrained_dict_filtered)

        # 加载模型时打印不匹配的键
        missing_keys, unexpected_keys = self.model.load_state_dict(model_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        # 添加新参数初始化
        for name, param in model_dict.items():
            if name not in pretrained_dict_filtered:
                if 'bn' in name:
                    if 'weight' in name:
                        nn.init.ones_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                elif 'conv' in name and param.dim() >= 2:
                    nn.init.kaiming_normal_(param)

        self.model.load_state_dict(model_dict, strict=False)  # 使用 strict=False 以避免形状不匹配的错误
        self.model.eval()

        for name, param in self.model.named_parameters():
            if name not in pretrained_dict:  # 新增加的层
                param.requires_grad = True
            else:  # 预训练参数
                param.requires_grad = True  # 保持可训练

    def generate_answer(self, question):
        # 使用预训练的问答模型生成回答
        inputs = self.qa_tokenizer(question, return_tensors="pt")
        outputs = self.qa_model.generate(**inputs, max_length=100)
        answer = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def _plot_confidence(self, confidence, input_data):
        """生成关系置信概率图"""
        # 示例：保留最近5次预测
        if not hasattr(self, 'confidence_history'):
            self.confidence_history = []
        self.confidence_history.append((input_data['subject_name'], input_data['object_name'], confidence))
        self.confidence_history = self.confidence_history[-5:]  # 只保留最新的五条记录

        plt.figure(figsize=(20,10))
        bars = plt.bar(range(len(self.confidence_history)), 
                    [x[2] for x in self.confidence_history], 
                    color='#3498db')
        plt.title("关系置信概率对比", fontsize=24)
        plt.ylabel("置信概率",fontsize=24)
        plt.xticks(range(len(self.confidence_history)), 
                [f"{x[0]} -> {x[1]}" for x in self.confidence_history],rotation=10, ha='right',
                fontsize=20)
        plt.yticks(fontsize=24)
        plt.grid(True, alpha=0.3)
        plt.legend(["置信概率"],fontsize=24)
        
        # 添加数值显示
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # x 坐标
                height,                             # y 坐标
                f"{height:.4f}",                    # 显示的文本
                ha='center',                        # 水平对齐方式
                va='bottom',                        # 垂直对齐方式
                fontsize=27                       # 设置字体大小
            )
        
        return plt.gcf()

    def _load_base_data(self):
        """加载预训练模型所需的基础图数据"""
        base_df = pd.read_excel(DATA_PATH)

        # 预处理数据并注册节点
        for _, row in base_df.iterrows():
            self.dynamic_graph.get_node_id(row['subject_name'], row['subject_type'])
            self.dynamic_graph.get_node_id(row['object_name'], row['object_type'])
        
        # 构建边索引
        self.base_edges = defaultdict(list)
        for _, row in base_df.iterrows():
            src_info = self.dynamic_graph.get_node_id(row['subject_name'], row['subject_type'])
            dst_info = self.dynamic_graph.get_node_id(row['object_name'], row['object_type'])
            src_id = src_info['type_id']
            dst_id = dst_info['type_id']
            et = (row['subject_type'], row['relation_type'], row['object_type'])
            self.base_edges[et].append([src_id, dst_id])

        # 确保所有预定义边类型都有初始化（新增代码）
        for et in all_edge_types:
            if et not in self.base_edges:
                # 填充虚拟边防止空索引
                self.base_edges[et] = [[0,0]] if et[0] == et[2] else []  # 自环边填充[0,0]

        # 更新图数据
        self.dynamic_graph.edge_index_dict = {
            et: torch.tensor(edges).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
            for et, edges in self.base_edges.items()
        }

    def _build_temp_data(self, df):
        """正确构建临时图数据"""
        # 注册所有节点
        for _, row in df.iterrows():
            self.dynamic_graph.get_node_id(row['subject_name'], row['subject_type'])
            self.dynamic_graph.get_node_id(row['object_name'], row['object_type'])
            # 打印节点映射
            print("Node map size:", len(self.dynamic_graph.node_map))
            # 打印类型节点
            print("Type nodes:", self.dynamic_graph.type_nodes)
        
        # 构建边索引
        edge_dict = defaultdict(list)
        for _, row in df.iterrows():
            src = self.dynamic_graph.get_node_id(row['subject_name'], row['subject_type'])['type_id']
            dst = self.dynamic_graph.get_node_id(row['object_name'], row['object_type'])['type_id']
            et = (row['subject_type'], row['relation_type'], row['object_type'])
            print(f"构建边: {et} {src}->{dst} | 源类型节点数={len(self.dynamic_graph.type_nodes[row['subject_type']])} 目标类型节点数={len(self.dynamic_graph.type_nodes[row['object_type']])}")  # 调试输出
            edge_dict[et].append([src, dst])
        
        # 转换为HeteroData
        data = HeteroData()
        for node_type in self.dynamic_graph.type_nodes:
            # 按type_id排序节点
            nodes = sorted(
                self.dynamic_graph.type_nodes[node_type],
                key=lambda x: self.dynamic_graph.node_map[x]['type_id']
            )
            print(f"特征矩阵构建: {node_type}类型节点数={len(nodes)}")
            # 验证特征存在性
            features = []
            for node in nodes:
                if node not in self.dynamic_graph.external_features:
                    raise KeyError(f"节点 {node} 缺少特征")
                features.append(self.dynamic_graph.external_features[node])
            
            data[node_type].x = torch.stack(features)
            print(f"Node Type: {node_type}, First 5 Features: {data[node_type].x[:5]}")
        
        # 特征矩阵维度验证
        for et in data.edge_types:
            src_type, _, dst_type = et
            edge_index = data[et].edge_index
            
            if edge_index.numel() > 0:
                max_src_idx = edge_index[0].max().item()
                max_dst_idx = edge_index[1].max().item()
                
                if data[src_type].x.shape[0] <= max_src_idx:
                    raise ValueError(
                        f"{src_type}特征矩阵维度不足: "
                        f"需要至少{max_src_idx+1}行, 实际{data[src_type].x.shape[0]}行"
                    )
                    
                if data[dst_type].x.shape[0] <= max_dst_idx:
                    raise ValueError(
                        f"{dst_type}特征矩阵维度不足: "
                        f"需要至少{max_dst_idx+1}行, 实际{data[dst_type].x.shape[0]}行"
                    )

        for et, edges in edge_dict.items():
            data[et].edge_index = torch.tensor(edges).t().contiguous()
        
        # 数据完整性验证
        validate_data_integrity(data)

                # 确保所有类型节点都有特征
        for node_type in self.dynamic_graph.type_nodes:
            nodes = self.dynamic_graph.type_nodes[node_type]
            if not nodes:
                print(f"警告: {node_type}类型无节点，填充虚拟特征")
                dummy_feat = torch.zeros(0, 32)  # 兼容维度
            else:
                dummy_feat = torch.stack([
                    self.dynamic_graph.external_features[n] 
                    for n in nodes
                ])
            data[node_type].x = dummy_feat
        
        # 边索引安全处理
        for et in all_edge_types:  # 使用预定义的所有边类型
            if et not in edge_dict:
                data[et].edge_index = torch.empty((2,0), dtype=torch.long)
        
        return data

    def _validate_input(self, input_data):
        """输入数据校验"""
        required = ['subject_name', 'subject_type', 'object_name', 'object_type', 'relation_type']
        missing = [col for col in required if col not in input_data]
        if missing:
            raise ValueError(f"缺少必要字段: {missing}")
        
        if input_data['subject_type'] not in NODE_TYPE_MAP:
            raise ValueError(f"非法主体类型: {input_data['subject_type']}")
        
        if input_data['object_type'] is None or input_data['object_type'] not in NODE_TYPE_MAP:
            raise ValueError(f"非法客体类型: {input_data['object_type']}")
        
        edge_type = (
            input_data['subject_type'],
            input_data['relation_type'],
            input_data['object_type']
        )
        if edge_type not in REL_TYPE_MAP:
            raise ValueError(f"非法关系组合: {edge_type}")

    def predict(self, input_data, feature_params=None):
        self.model.eval() 
        # 在预测前添加全局校验
        for edge_type, edge_index in self.dynamic_graph.edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
                
            src_type, _, dst_type = edge_type
            max_src = edge_index[0].max().item()
            max_dst = edge_index[1].max().item()
            
            print(f"全局校验 {edge_type}: max_src={max_src}(应有<{len(self.dynamic_graph.type_nodes[src_type])}) "
                  f"max_dst={max_dst}(应有<{len(self.dynamic_graph.type_nodes[dst_type])})")
            
            assert max_src < len(self.dynamic_graph.type_nodes[src_type]), \
                f"{edge_type}源索引越界 {max_src} >= {len(...)}"
            assert max_dst < len(self.dynamic_graph.type_nodes[dst_type]), \
                f"{edge_type}目标索引越界 {max_dst} >= {len(...)}"

        # 数据校验
        self._validate_input(input_data)
        # 添加参数有效性检查
        if feature_params is None:
            feature_params = {}
        # === 新增节点处理 ===
        subject_name = input_data['subject_name']
            # 统一处理subject和object节点
        for role in ['subject', 'object']:
            node_name = input_data[f'{role}_name']
            # 当节点不存在且未提供特征时抛出错误
            if not self.dynamic_graph.check_node_exists(node_name):
                if not feature_params.get(role):
                    raise ValueError(f"{role}节点'{node_name}'必须提供特征参数！")

                # 添加新节点
                self.dynamic_graph.add_new_node(
                    node_name=subject_name,
                    node_type=input_data['subject_type'],
                    features=feature_params
                )

        # 构建临时数据文件
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
        pd.DataFrame([input_data]).to_excel(temp_path, index=False)
        temp_df = pd.read_excel(temp_path)
        # 删除临时文件
        os.remove(temp_path)  # 添加此行    
        # 构建动态数据
        data = self._build_temp_data(temp_df)

        # 获取所有节点特征
        x_dict = {
            node_type: torch.stack([
                self.dynamic_graph.external_features[node].detach() 
                for node in nodes
            ])
            for node_type, nodes in self.dynamic_graph.type_nodes.items()
        }

        # 在预测前添加特征打印
        # 添加特征差异检查
        for node_type, feats in x_dict.items():
            std = feats.std(dim=0).mean().item()
            print(f"{node_type}特征标准差: {std:.4f}")
            if std < 1e-3:
                raise ValueError(f"{node_type}节点特征未正确区分！")

        # 执行预测
        edge_type = (input_data['subject_type'], 
                    input_data['relation_type'],
                    input_data['object_type'])

        # 获取关系类型字符串而非三元组
        rel_type_str = input_data['relation_type']  # 例如 "influence"



        with torch.no_grad(): 
            h_dict, _ = self.model(x_dict, data.edge_index_dict)
            pred = self.model.predictor(
                edge_index=data[edge_type].edge_index,  # 边索引
                h_dict=h_dict, 
                edge_type=edge_type  # 传递完整三元组
            )
        # 确保h_dict包含新节点特征
        print("New node features:", h_dict['network'][-1])  

        # 计算关系置信概率
        confidence = torch.sigmoid(pred).item()

        # 计算节点交叉熵
        loss = compute_single_loss(self.model, x_dict, data.edge_index_dict, edge_type)

        # 生成关系置信概率图
        confidence_fig = self._plot_confidence(confidence, input_data)

        # 生成节点交叉熵图
        cross_entropy_fig = self._plot_cross_entropy(loss, input_data)

        # 风险等级评估
        risk_assessment = self._assess_risk_level(confidence)

        return (
            confidence_fig,  # 关系置信概率图
            cross_entropy_fig,  # 节点交叉熵图
            risk_assessment  # 风险等级评估
        )

    def _calc_pair_entropy(self, src_node, dst_node):
        """计算指定节点对的交叉熵"""
        # 节点存在性检查
        if src_node not in self.dynamic_graph.node_map:
            raise ValueError(f"源节点 {src_node} 不存在")
        if dst_node not in self.dynamic_graph.node_map:
            raise ValueError(f"目标节点 {dst_node} 不存在")
        
        # 获取节点信息
        src_type = self.dynamic_graph.node_map[src_node]['type']
        dst_type = self.dynamic_graph.node_map[dst_node]['type']
        src_idx = self.dynamic_graph.node_map[src_node]['type_id']
        dst_idx = self.dynamic_graph.node_map[dst_node]['type_id']
        
        # 构建临时数据
        temp_data = HeteroData()
        for node_type in self.dynamic_graph.type_nodes:
            nodes = sorted(self.dynamic_graph.type_nodes[node_type],
                          key=lambda x: self.dynamic_graph.node_map[x]['type_id'])
            temp_data[node_type].x = torch.stack([
                self.dynamic_graph.external_features[n] 
                for n in nodes
            ])
        
        # 模型计算
        x_dict = {nt: temp_data[nt].x for nt in temp_data.node_types}
        h_dict, _ = self.model(x_dict, self.dynamic_graph.edge_index_dict)
        
        # 特征提取
        src_feat = h_dict[src_type][src_idx]
        dst_feat = h_dict[dst_type][dst_idx]
        
        # 维度验证
        assert src_feat.dim() == 1, f"源特征维度错误: {src_feat.shape}"
        assert dst_feat.dim() == 1, f"目标特征维度错误: {dst_feat.shape}"
        
        # 计算交叉熵
        return F.cross_entropy(
            src_feat.unsqueeze(0),  # 添加batch维度
            dst_feat.unsqueeze(0)
        ).item()
    
    # 修改后的 _plot_cross_entropy 方法
    def _plot_cross_entropy(self, cross_entropy, input_data):
        """生成节点交叉熵图"""
        # 保留最近5次记录
        if not hasattr(self, 'cross_entropy_history'):
            self.cross_entropy_history = []
        self.cross_entropy_history.append((input_data['subject_name'], input_data['object_name'], cross_entropy))
        self.cross_entropy_history = self.cross_entropy_history[-5:]  # 只保留最新的五条记录

        plt.figure(figsize=(20,10))
        bars = plt.bar(range(len(self.cross_entropy_history)), 
                    [x[2] for x in self.cross_entropy_history], 
                    color='#e74c3c')
        plt.title("节点交叉熵损失对比",fontsize=24)
        plt.ylabel("交叉熵损失值",fontsize=24)
        plt.xticks(range(len(self.cross_entropy_history)), 
                [f"{x[0]} -> {x[1]}" for x in self.cross_entropy_history],rotation=10, ha='right',
                fontsize=20)
        plt.yticks(fontsize=24)  # 设置 y 轴刻度值的字体大小
        plt.grid(True, alpha=0.3)
        plt.legend(["交叉熵损失"],fontsize=24)
        
        # 添加数值显示
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2,
                height,                             # y 坐标
                f"{height:.4f}",                    # 显示的文本
                ha='center',                        # 水平对齐方式
                va='bottom',                        # 垂直对齐方式
                fontsize=27                         # 设置字体大小
            )
        
        return plt.gcf()


    def _assess_risk_level(self, confidence):
        """风险评估逻辑"""
        if confidence < 0.2:
            return {"不准确": 1.0, "模糊": 0.0, "准确": 0.0}
        elif 0.2 <= confidence < 0.8:
            return {"不准确": 0.2, "模糊": 0.7, "准确": 0.1}
        else:
            return {"不准确": 0.0, "模糊": 0.2, "准确": 0.8}
        
    def _get_reference_groups(self, input_data):
        """改进的特征相似度计算方法"""
        current_feat = self.dynamic_graph.external_features[input_data['subject_name']]
        
        # 计算所有节点的余弦相似度
        similarities = []
        for node in self.dynamic_graph.node_map:
            if node == input_data['subject_name']:
                continue
            node_feat = self.dynamic_graph.external_features[node]
            sim = F.cosine_similarity(current_feat, node_feat, dim=0).item()
            similarities.append( (node, sim) )
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 选择预测组（前5）和参照组（后3）
        pred_group = [n for n, s in similarities[:5]]
        ref_group = [n for n, s in similarities[-3:]]
        
        return pred_group, ref_group
        
        

def validate_data_integrity(data):
    """数据完整性验证"""
    print("\n=== 数据完整性检查 ===")
    # 检查节点特征
    for node_type in data.node_types:
        feat = data[node_type].x
        print(f"{node_type}特征统计: mean={feat.mean():.2f}, std={feat.std():.2f}")
        if torch.isnan(feat).any():
            raise ValueError(f"{node_type}特征包含NaN值")
        
    # 新增特征存在性检查
    for node_type in data.node_types:
        if node_type not in data.x_dict:
            raise ValueError(f"缺失节点类型: {node_type}")
        if data[node_type].x is None:
            raise ValueError(f"{node_type} 特征为空")
        
    # 检查边索引
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"{edge_type}边索引形状错误: {edge_index.shape}")
        if edge_index.numel() > 0:
            max_idx = edge_index.max()
            num_nodes = len(data[edge_type[0]].x)  # 动态获取节点数量
            print(f"Edge type: {edge_type}, max_idx: {max_idx}, num_nodes: {num_nodes}")
            # 分别检查源节点和目标节点
            src_type, _, dst_type = edge_type
            max_src = edge_index[0].max().item()
            max_dst = edge_index[1].max().item()
            num_src = len(data[src_type].x)
            num_dst = len(data[dst_type].x)
            src_type = edge_type[0]
            dst_type = edge_type[2]
            
            if src_type not in data.node_types or dst_type not in data.node_types:
                raise ValueError(f"边类型 {edge_type} 的源类型或目标类型不在节点类型列表中")
            
            if max_src >= num_src:
                raise ValueError(f"源节点越界: {edge_type} max_src={max_src} >= {num_src}")
            if max_dst >= num_dst:
                raise ValueError(f"目标节点越界: {edge_type} max_dst={max_dst} >= {num_dst}")
            if max_idx >= num_nodes:
                raise ValueError(f"{edge_type}边索引越界: max_index={max_idx}, num_nodes={num_nodes}")
    print("数据完整性验证通过\n")
    
    # 新增特征存在性检查
    for node_type in data.node_types:
        if node_type not in data.x_dict:
            raise ValueError(f"缺失节点类型: {node_type}")
        if data[node_type].x is None:
            raise ValueError(f"{node_type} 特征为空")


def create_visualization(system):
    """创建带动态验证的可视化界面"""
    node_types = list(NODE_TYPE_MAP.keys())

    # 合法关系组合映射（前端显示用）
    RELATION_GUIDE = {
        "network": {
            "self_loop": ["network"],
            "has": ["line"]
        },
        "factor": {
            "influence": ["factor"],
            "self_loop": ["factor"]
        },
        "line": {
            "influence": ["factor"],
            "self_loop": ["line"],
            "has": ["station"],
            "etype": ["factor"]
        },
        "station": {
            "influence": ["factor"],
            "self_loop": ["station"],
            "next_station": ["station"],
            "etype": ["factor"],
            "transfer":["network"]
        },
    }
    
    # 生成可用的关系类型选项
    rel_types = sorted({k for guide in RELATION_GUIDE.values() for k in guide.keys()})


    def parse_embedding(emb_str: str) -> tuple[np.ndarray, bool]:
        """支持无逗号分隔的嵌入字符串解析"""
        try:
            # 清洗嵌入字符串：移除方括号、省略号、多余空格等
            cleaned = re.sub(r'[$$\.\.\.]', '', emb_str.strip())  # 移除方括号和省略号
            cleaned = re.sub(r'\s+', ' ', cleaned)  # 合并多个空格为单个空格
            
            # 提取所有合法的浮点数（包括负数和科学计数法）
            numbers = []
            for part in cleaned.split():
                try:
                    num = float(part)
                    numbers.append(num)
                except ValueError:
                    print(f"⚠️ 无效部分跳过: '{part}'")
                    continue
            
            # 维度校验与填充
            if len(numbers) < 1024:
                numbers += [0.0] * (1024 - len(numbers))  # 填充至1024维
            elif len(numbers) > 1024:
                numbers = numbers[:1024]  # 截断
            
            return np.array(numbers, dtype=np.float32), True
        except Exception as e:
            print(f"⚠️ 终极解析失败: {emb_str[:30]}... | 错误: {str(e)}")
            return np.zeros(1024), False

    def load_knowledge_data():
        """加载所有知识数据"""
        # 1. 原始实体
        entities = []
        with open("E:/work_and_study/path_same/mini_adjust/graphrag-local-ollama/graphrag-local-ollama/ragtest_4/output/20250401-204548/artifacts/raw_extracted_entities.json", "r") as f:
            for line in f:
                data = json.loads(line)
                entities.extend(data.get("entities", []))
            
        # 2. 最终实体与关系
        df_entities = pd.read_parquet("E:/work_and_study/path_same/mini_adjust/graphrag-local-ollama/graphrag-local-ollama/ragtest_4/output/20250401-204548/artifacts/create_final_entities.parquet")
        df_relations = pd.read_parquet("E:/work_and_study/path_same/mini_adjust/graphrag-local-ollama/graphrag-local-ollama/ragtest_4/output/20250401-204548/artifacts/create_final_relationships.parquet")

        # 3. 合并后的知识图谱
        merged_kg = nx.read_graphml("E:/work_and_study/path_same/mini_adjust/graphrag-local-ollama/graphrag-local-ollama/ragtest_4/output/20250401-204548/artifacts/merged_graph.graphml")
        # 转换为有向图（即使原始数据是无向的）
        merged_kg = merged_kg.to_directed()

        # 清洗实体名称（去除引号和首尾空格）
        df_entities['cleaned_name'] = df_entities['name'].str.strip('"').str.strip()
        # 清洗图谱节点名称（假设节点ID为名称）
        for node_id, data in merged_kg.nodes(data=True):
            data['cleaned_name'] = str(node_id).strip('"').strip()

        print("图谱类型:", type(merged_kg))  # 应显示 <class 'networkx.classes.digraph.DiGraph'>
        print("是否无向图:", merged_kg.is_directed()) 
        
        return {
            "raw_entities": entities,
            "clean_entities": df_entities,
            "relationships": df_relations,
            "knowledge_graph": merged_kg
        }

    # 在初始化 RAG 前加载数据
    knowledge_data = load_knowledge_data()

        # 验证数据完整性
    assert len(knowledge_data["clean_entities"]) > 0, "实体数据未加载！"
    assert isinstance(knowledge_data["knowledge_graph"], nx.Graph), "图谱格式错误！"
    
    # 调试输出
    print(f"✅ 已加载知识数据：{len(knowledge_data['clean_entities'])} 个实体，{knowledge_data['knowledge_graph'].number_of_edges()} 条关系")

    # ---------- 自定义上下文构建器 ----------
    class OllamaContextBuilder(GlobalContextBuilder):
        """基于 Ollama 本地服务的上下文构建器"""

        def __init__(self, knowledge_data):
            super().__init__()
            self.df_entities = knowledge_data["clean_entities"]
            self.kg = knowledge_data["knowledge_graph"]

            # ==== 核心映射逻辑 ====
            self.name_to_node = {}
            self.node_to_name = {}
            
            # 构建双向映射
            for node_id, data in self.kg.nodes(data=True):
                kg_name = data.get('cleaned_name', node_id)
                # 在实体表中查找匹配名称
                matched = self.df_entities[self.df_entities.cleaned_name == kg_name]
                if not matched.empty:
                    entity_id = matched.iloc[0]['id']
                    self.name_to_node[entity_id] = node_id
                    self.node_to_name[node_id] = entity_id

            print(f"名称映射成功率: {len(self.name_to_node)}/{len(self.df_entities)}")

            # 预处理嵌入数据（带异常处理）
            self.valid_embeddings = []
            self.valid_entity_indices = []
            for idx, row in self.df_entities.iterrows():
                # 强制转换为字符串并移除换行符
                emb_str = str(row["description_embedding"]).replace('\n', ' ')
                emb_array, is_valid = parse_embedding(emb_str)
                
                if is_valid:
                    self.valid_embeddings.append(emb_array)
                    self.valid_entity_indices.append(idx)
                else:
                    print(f"🚨 实体 '{row['name']}' 使用解析后的向量（非全零）")
                    self.valid_embeddings.append(emb_array)  # 即使无效也保留解析结果
                    
            self.valid_embeddings = np.stack(self.valid_embeddings) if self.valid_embeddings else np.zeros((0,1024))
            if self.valid_embeddings.shape[1] != 1024:
                print(f"⚠️ 实体嵌入维度异常 ({self.valid_embeddings.shape[1]}D)，将统一到1024维")
                # 截断或填充
                if self.valid_embeddings.shape[1] > 1024:
                    self.valid_embeddings = self.valid_embeddings[:, :1024]
                else:
                    pad_width = [(0,0), (0, 1024 - self.valid_embeddings.shape[1])]
                    self.valid_embeddings = np.pad(self.valid_embeddings, pad_width)
            # 在初始化后添加质量检查
            valid_count = len(self.valid_entity_indices)
            total_count = len(self.df_entities)
            print(f"""
            📊 嵌入数据质量报告：
            有效实体：{valid_count}/{total_count} ({valid_count/total_count:.1%})
            无效实体：{total_count - valid_count}
            典型问题样本：{self.df_entities.iloc[0]['description_embedding'][:100]}...
            """)

        def build_context(self, query: str, **kwargs):
            # 初始化 context_chunks 为空列表
            context_chunks = []
            
            try:
                # 空知识库保护
                if len(self.valid_embeddings) == 0:
                    return ["📢 知识库尚未初始化有效数据"], {}
                
                # 获取查询嵌入并强制转换
                query_embedding = self._get_ollama_embedding(query)
                # 确保二维形状 (1, 1024)
                if query_embedding.ndim == 1:
                    query_embedding = query_embedding.reshape(1, -1)
                elif query_embedding.ndim == 3:  # 处理意外三维情况
                    query_embedding = query_embedding.squeeze(axis=0)
                # 调试输出
                print(f"✅ 最终查询嵌入形状: {query_embedding.shape}")  # 应为 (1, 1024)

                # 语义检索实体（新增代码）
                similarities = np.dot(self.valid_embeddings, query_embedding.T).flatten()
                top_indices = np.argsort(similarities)[-3:][::-1]
                matched_entities = self.df_entities.iloc[top_indices]
                
                # ==== 关联路径提取 ====
                context_chunks = []
                related_edges = []
                # ==== 修改后的边遍历 ====
                for _, row in matched_entities.iterrows():
                    kg_node = self.name_to_node.get(row['id'])
                    if kg_node and self.kg.has_node(kg_node):
                        # 获取所有关联边（兼容有向/无向图）
                        edges = list(self.kg.edges(kg_node))  # 所有与节点相连的边
                        related_edges.extend(edges[:3])  # 最多取3条
                
                # ==== 关联路径提取 ====
                if related_edges:
                    context_chunks.append("\n🔗 关联路径：")
                    for u, v in related_edges:
                        edge_data = self.kg.get_edge_data(u, v)
                        # 处理有向图特性
                        direction = "→" if self.kg.is_directed() else "—"
                        rel_type = edge_data.get('relation_type', '关联')
                        context_chunks.append(f"{u} {direction} {rel_type} {direction} {v}")
                else:
                    # 方案1：显示同类实体
                    context_chunks.append("\n📌 同类实体参考：")
                    same_type = self.df_entities[self.df_entities.type == matched_entities.iloc[0]['type']].sample(3)
                    for _, row in same_type.iterrows():
                        context_chunks.append(f"- {row['name']}：{row['description'][:50]}...")

                # 在build_context末尾添加
                if not context_chunks:
                    context_chunks = [
                        "🏷️ 当前知识库中暂无直接相关数据，以下为通用知识：",
                        "轨道交通风险分析需考虑网络拓扑结构与客流时空分布",
                        "常见风险因素包括设备故障、客流突变、调度冲突等"
                    ]
                else:
                    context_chunks.insert(0, "📚 知识库检索结果：")

                print(f"匹配到{len(matched_entities)}个实体 | 首实体：{matched_entities.iloc[0]['name'] if not matched_entities.empty else '无'}")
                print(f"生成上下文块数量：{len(context_chunks)} | 示例：{context_chunks[:1]}")
                # 在返回前添加校验
                assert query_embedding.ndim == 2, f"查询嵌入应为二维，实际维度 {query_embedding.ndim}"
                assert query_embedding.shape == (1, 1024), f"形状异常: {query_embedding.shape}"
                return (
                    context_chunks,  # 确保返回列表
                    {"embedding": pd.DataFrame(query_embedding)}# 直接传入二维数组
                )
            except Exception as e:
                print(f"""
                    ❌ 关键错误详情：
                    查询内容: {query}
                    查询嵌入类型: {type(query_embedding) if 'query_embedding' in locals() else '未初始化'}
                    实体嵌入矩阵形状: {self.valid_embeddings.shape if hasattr(self, 'valid_embeddings') else '未加载'}
                    错误轨迹: {traceback.format_exc()}
                """)
                return (context_chunks, {})  # 返回已初始化的空列表

        def _get_ollama_embedding(self, text: str) -> np.ndarray:
            """确保返回NumPy数组"""
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "bge-m3:latest", "prompt": text},
                timeout=30
            )
            embedding = response.json()["embedding"]
            # 双重类型保障
            return np.array(embedding).astype(np.float32)  # 强制转换并指定精度

    # ---------- 模型调用适配器 ----------
    class OllamaModelAdapter:
        """将 Ollama API 适配为 GraphRAG 需要的 ChatModel 协议"""
        
        async def achat(self, prompt: str, history: List[dict], **kwargs) -> dict:
            """实现异步聊天接口"""
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": kwargs.get("model_parameters", {})
                }
            )
            return {"output": {"content": response.json()["response"]}}

    # ---------- 初始化 RAG 引擎 ----------
    def init_rag_engine():
        # 1. 初始化组件
        token_encoder = tiktoken.get_encoding("cl100k_base")
        model_adapter = OllamaModelAdapter()
        # 在初始化 GlobalSearch 前创建回调实例
        callbacks = GlobalSearchLLMCallback()
        knowledge_data = load_knowledge_data()

        # 2. 创建 GlobalSearch 实例
        return GlobalSearch(
            llm=model_adapter,
            context_builder=OllamaContextBuilder(knowledge_data),  # 关键修复：传入 knowledge_data
            token_encoder=token_encoder,
            allow_general_knowledge=True,
            callbacks=[callbacks],  # 关键修改：注入回调实例
            map_llm_params={"temperature": 0.5},
            reduce_llm_params={"max_tokens": 1000}
        )

    rag_engine = init_rag_engine()

    with gr.Blocks(title="清问智答系统", theme=gr.themes.Soft(primary_hue="blue")) as demo:
        # 头部区域
        gr.HTML("""<div style="text-align:center;padding:20px;background:linear-gradient(135deg,#2b5876,#4e4376);border-radius:8px;">
                    <h1 style="color:white;margin:0;">🚇 清问智答平台</h1>
                    <p style="color:#cfd7ff;">基于知识图谱的地铁清分失效诊断系统</p>
                  </div>""")
        
        # 操作指引
        with gr.Accordion("📌 操作指南", open=True):
            gr.Markdown("""
            <div style="padding:15px;background:#f8f9fa;border-radius:8px;">
            <p style="color:#2c3e50;font-weight:500;">✦ 交互逻辑说明：</p>
            <ul style="color:#4a5568;">
                <li>“智能关系预测”界面——如果输入知识图谱里不存在的节点名称，会出现错误。</li>
                <li>“智能关系预测”界面——节点交叉熵越小，关系置信概率越大，节点对间关系的存在性就越高</li>
                <li>“智能问答助手”界面——点击示例问题，能直接向大语言模型提问</li>
            </ul>
            </div>
            """)
        
        # 主功能区域
        with gr.Tabs(selected=0) as main_tabs:
            # 关系预测标签页
            with gr.Tab("🔮 智能关系预测", id=0):
                with gr.Row(variant="panel"):
                    # 输入列
                    with gr.Column(scale=2, min_width=400):
                        with gr.Blocks():
                            gr.HTML("<h3 style='margin-top: 0;'>预测参数配置</h3>")
                            with gr.Blocks():
                                gr.HTML("<h4 style='margin-top: 0;'>新增节点关联</h4>")
                                with gr.Row():
                                    new_subject = gr.Textbox(label="起始节点名称", 
                                                        placeholder="输入station/line节点...")
                                    new_subject_type = gr.Dropdown(
                                        label="起始类型",
                                        choices=["station", "line"],
                                        value="station"
                                    )
                                new_object = gr.Textbox(label="目标factor节点", 
                                                    placeholder="输入已有factor节点...")
                                add_result = gr.HTML()
                                add_edge_btn = gr.Button("✅ 确认添加etype关联", variant="primary")

                            with gr.Blocks():
                                gr.HTML("<h4 style='margin-top: 0;'>节点配置</h4>")
                                subject = gr.Textbox(label="起始节点", 
                                                    placeholder="输入节点名称...",
                                                    elem_classes="highlight-input")
                                subject_type = gr.Dropdown(
                                    label="起始类型",
                                    choices=node_types,
                                    value="station",
                                    interactive=True,
                                    container=True
                                )
                            
                            gr.HTML("<div class='divider'></div>")

                            with gr.Blocks():
                                gr.HTML("<h4 style='margin-top: 0;'>关系配置</h4>")
                                relation_type = gr.Dropdown(
                                    label="关系类型",
                                    choices=rel_types,
                                    value="influence",
                                    interactive=True,
                                    container=True
                                )
                                object_ = gr.Textbox(label="目标节点", 
                                                    placeholder="输入目标节点...",
                                                    elem_classes="highlight-input")
                                object_type = gr.Dropdown(
                                    label="目标类型",
                                    choices=[],
                                    interactive=True,
                                    container=True
                                )

                            status_indicator = gr.HighlightedText(
                                label="验证状态",
                                value=[("等待输入", "")],
                                show_legend=False,
                                visible=False
                            )

                        # 操作按钮组
                        with gr.Row():
                            submit_btn = gr.Button("🚀 执行预测", 
                                                variant="primary",
                                                scale=2,
                                                size="lg")
                            clear_btn = gr.Button("🔄 重置输入", 
                                                variant="secondary",
                                                scale=1)

                    # 输出列
                    with gr.Column(scale=3, min_width=600):
                        output_tabs = gr.Tabs()
                        with output_tabs:
                            with gr.Tab("📊 节点交叉熵图", id=0):
                                cross_entropy_plot = gr.Plot(label="节点交叉熵分布",
                                                        container=True,
                                                        show_label=False)
                            with gr.Tab("📈 置信概率趋势", id=1):
                                reference_plot = gr.Plot(label="置信概率曲线",
                                                            container=True,
                                                            show_label=False)
                        
                        # 模型评估面板
                        with gr.Row():
                            risk_level = gr.Label(label="预测准确度",
                                                value={"准确": 0.6, "模糊": 0.3, "不准确": 0.1},
                                                num_top_classes=3,
                                                show_label=False)

                # 交互逻辑保持不变
                def update_relations(subject_type):
                    available_relations = list(RELATION_GUIDE.get(subject_type, {}).keys())
                    return {
                        relation_type: gr.update(choices=available_relations, value=available_relations[0] if available_relations else None)
                    }

                subject_type.change(
                    fn=update_relations,
                    inputs=[subject_type],
                    outputs=[relation_type]
                )

                def add_new_edge(subject, subject_type, obj):
                    try:
                        # 验证节点存在性
                        if not system.dynamic_graph.check_node_exists(subject):
                            raise ValueError(f"起始节点'{subject}'不存在")
                        if not system.dynamic_graph.check_node_exists(obj):
                            raise ValueError(f"目标节点'{obj}'不存在")
                        
                        # 验证节点类型
                        if system.dynamic_graph.node_map[subject]['type'] != subject_type:
                            raise ValueError(f"起始节点类型应为{subject_type}，实际为{system.dynamic_graph.node_map[subject]['type']}")
                        if system.dynamic_graph.node_map[obj]['type'] != 'factor':
                            raise ValueError(f"目标节点类型必须为factor，实际为{system.dynamic_graph.node_map[obj]['type']}")
                        
                        # 添加边
                        system.dynamic_graph.add_edge(subject, obj, "etype")
                        # === 新增训练逻辑 ===
                        # 1. 注册新关系到预测器
                        edge_type = (subject_type, "etype", "factor")
                        system.model.predictor.register_edge_type(edge_type)
                        
                        # 2. 重新初始化训练器
                        system.trainer.init_trainable_params()
                        
                        # 3. 构建训练数据
                        temp_df = pd.DataFrame([{
                            'subject_name': subject,
                            'subject_type': subject_type,
                            'relation_type': 'etype',
                            'object_name': obj,
                            'object_type': 'factor'
                        }])
                        train_data = system._build_temp_data(temp_df)
                        
                        # 4. 执行快速训练
                        system.trainer.incremental_train(train_data, epochs=30)
                        return f"<div style='color:green'>✅ 成功添加关联：{subject} ({subject_type}) → {obj} (factor)</div>"
                    except Exception as e:
                        return f"<div style='color:red'>❌ 错误：{str(e)}</div>"

                def update_object_type(subject_type, relation_type):
                    if subject_type and relation_type:
                        allowed_objects = RELATION_GUIDE.get(subject_type, {}).get(relation_type, [])
                        return {
                            object_type: gr.update(choices=allowed_objects, value=allowed_objects[0] if allowed_objects else None)
                        }
                    return {
                        object_type: gr.update(choices=[], value=None)
                    }

                relation_type.change(
                    fn=lambda st, rt: update_object_type(st, rt),
                    inputs=[subject_type, relation_type],
                    outputs=object_type
                )

                def validate_combination(subject_type, relation_type, object_type):
                    if not all([subject_type, relation_type, object_type]):
                        return [("请完成所有选择", "error")], False
                    
                    edge_type = (subject_type, relation_type, object_type)
                    is_valid = edge_type in REL_TYPE_MAP
                    msg = "✅ 合法组合" if is_valid else f"❌ 非法组合: {edge_type}"
                    return [(msg, "valid" if is_valid else "invalid")], is_valid

                inputs = [subject_type, relation_type, object_type]
                for component in inputs:
                    component.change(
                        fn=validate_combination,
                        inputs=inputs,
                        outputs=[status_indicator, submit_btn]
                    )

                submit_btn.click(
                    fn=lambda s, st, r, o, ot: system.predict(
                        {'subject_name': s, 'subject_type': st,
                        'relation_type': r, 'object_name': o, 'object_type': ot}
                    ),
                    inputs=[subject, subject_type, relation_type, object_, object_type ],
                    outputs=[reference_plot, cross_entropy_plot, risk_level]
                )
                
                clear_btn.click(
                    lambda: [
                        None,  # 清空subject输入
                        "station",  # 重置subject_type
                        "influence",  # 重置relation_type
                        None,  # 清空object_
                        gr.update(choices=['factor'], value='factor'),  # 重置object_type
                        None,  # 新增：重置cross_entropy_plot
                        None,  # 新增：重置reference_plot
                        {"准确": 0.0, "模糊": 0.0, "不准确": 0.0}  # 新增：重置risk_level
                    ],
                    outputs=[
                        subject, 
                        subject_type, 
                        relation_type, 
                        object_, 
                        object_type,
                        cross_entropy_plot,  # 新增输出参数
                        reference_plot,  # 新增输出参数
                        risk_level  # 新增输出参数
                    ]
                )

                add_edge_btn.click(
                    fn=add_new_edge,
                    inputs=[new_subject, new_subject_type, new_object],
                    outputs=add_result
                )

            # AI问答标签页
            with gr.Tab("💬 智能问答助手", id=1):
                with gr.Row(variant="panel"):
                    # 聊天主界面
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(bubble_full_width=False,
                                        avatar_images=("E:/work_and_study/path_same/mini_adjust/user.png", "E:/work_and_study/path_same/mini_adjust/assistant.png"),
                                        height=500,
                                        type="messages" )
                        
                        # 示例问题按钮组（新增多按钮布局）
                        with gr.Column(elem_classes="example-group"):
                            gr.HTML("<div class='example-title'>常见分析场景：</div>")
                            with gr.Row():
                                example_btn1 = gr.Button(
                                    "海伦路站故障期间清分系统应急路径库未激活导致的周边站点分流失效评估",
                                    elem_classes="example-btn"
                                )
                                example_btn2 = gr.Button(
                                    "浦东机场站清分模型节假日参数组缺失，导致铁路联运客流预测失准的风险评估",
                                    elem_classes="example-btn"
                                )
                            with gr.Row():
                                example_btn3 = gr.Button(
                                    "徐家汇站15号线接入后清分系统OD反推模型未及时校准，导致早高峰断面预测持续偏离的失效路径分析",
                                    elem_classes="example-btn"
                                )
                                example_btn4 = gr.Button(
                                    "迪士尼站跨年夜清分系统反向阻抗系数未动态调整，导致11号线车次匹配失序的失效预测",
                                    elem_classes="example-btn"
                                )
                            with gr.Row():
                                example_btn5 = gr.Button(
                                    "虹桥枢纽清分路径特征库未融合新线参数引发的线通勤路径迁移失真预测",
                                    elem_classes="example-btn"
                                )
                                example_btn6 = gr.Button(
                                    "新生入学时，导致同济大学站清分系统失效的原因有哪些",
                                    elem_classes="example-btn"
                                )

                        question = gr.Textbox(placeholder="请输入您的问题...",
                                            lines=2,
                                            max_lines=5,
                                            autofocus=True)
                        
                        with gr.Row():
                            ask_button = gr.Button("📤 发送提问", variant="primary")
                            reset_button = gr.Button("🧹 清空对话", variant="stop")


                    # 参数控制面板
                    with gr.Column(scale=1):
                        with gr.Blocks():
                            gr.HTML("<h3 style='margin-top: 0;'>生成参数调节</h3>")
                            temperature = gr.Slider(label="创造性温度",
                                                minimum=0.1,
                                                maximum=2.0,
                                                value=1.0,
                                                step=0.1,
                                                info="值越大回答越多样")
                            top_k = gr.Slider(label="Top-K采样",
                                            minimum=1,
                                            maximum=50,
                                            value=10,
                                            step=1,
                                            info="考虑最可能的K个词")
                            top_p = gr.Slider(label="Top-P采样",
                                            minimum=0.1,
                                            maximum=1.0,
                                            value=0.9,
                                            step=0.1,
                                            info="累积概率阈值")


                # AI问答功能逻辑
                def build_prompt(history):
                    """增强版提示词构建（保留原对话逻辑）"""
                    # 1. 处理空历史场景
                    if not history:
                        current_query = "欢迎提问！请输入您的问题。"  # 默认查询
                        knowledge_context = "【初始知识库】城市轨道交通客流量分析基础概念"
                    else:
                        current_query = history[-1]["content"]
                        # 执行知识检索
                        context_chunks, _ = rag_engine.context_builder.build_context(current_query)
                        knowledge_context = "\n".join(context_chunks) if context_chunks else ""
                    
                    print(knowledge_context)

                    # 2. 原有系统提示
                    """构建包含完整对话历史的提示"""
                    system_prompt = f"""
                    【上海城市轨道交通清分系统逻辑诊断框架】
                    ##框架概述
                    你是由智答小队开发的基于知识图谱的地铁清分失效诊断系统————清问智答，你专注于上海城市轨道交通清分系统的诊断问题，不要回答突发事件的解决方案和应对举措，而是需要有效识别突发事件产生的清分系统潜在失效因子，并对其进行重点的、全面的分析，最终诊断突发事件是否会导致清分系统失效。以下是相关知识库储备，你需要充分利用知识库的知识。
                    注意：你的回答应尽可能地专业全面。当用户的问题完全与清分系统诊断问题完全无关时，请你礼貌地拒绝。
                    {knowledge_context}

                    ## 清分系统模型架构

                    1. 广义费用动态修正：
                        - 公式：GC_k = αT_k + βC_k + γR_k
                        - T_k：时变旅行时间（含可靠性修正）
                        - C_k：换乘感知成本（异质性系数δ_i）
                        - R_k：票价差异因子（多网融合场景激活）

                    2. 有效路径集动态生成：
                        - 公式：R_rs = （k | Z_k ≤ min(Z_rs)×(1+θ₂) ∧ Z_k ≤ Z_min+θ₃）
                        - 触发更新条件：网络拓扑变更 / 运营事件 / 参数漂移

                    3. 异质性响应机制：
                        - 公式：p_k^h = U_k^h / ∑U_m^h
                        - h∈（理性,有限理性,群体决策）
                        - U_k^h = exp[-(ΔC_k^h)²/2θ₁² + βS_k - γσ_t]

                    ## 三维致因体系升级

                    ### 一、模型理论层

                    1. 假设体系
                        - 理性人偏差 → 路径倒置（有限理性乘客）
                        - 小流量OD噪声 → 长尾累积误差
                        - 多网票价断层 → 效用失真（城轨/市域铁路混行）

                    2. 结构缺陷
                        - 时间可靠性盲区 → 高峰偏移（准点率优先）
                        - 换乘成本均质化 → 路径集失真（个体差异）

                    3. 参数漂移
                        - 选择概率扁平化
                        - 有效路径漏判

                    ### 二、运营网络层

                    1. 物理网络扰动
                        - 拓扑地图幻觉 → 选择悖论（地图vs实际）
                        - 特殊结构漩涡 → 等待黑洞（Y型线/环线）

                    2. 行车网络振荡
                        - 交路模式混沌 → 换乘涟漪（大小交路/快慢车）
                        - 末班车断层 → 夜间绕行潮
                        - 运营事件冲击波 → 路径重构（延误）

                    ### 三、乘客行为层

                    1. 个体异常
                        - 反向绕行脉冲 → 断面振荡（舒适度驱动）
                        - 同站进出噪声 → 数据污染（无效AFC记录）

                    2. 群体涌现
                        - 结伴绕行共振 → 断面偏移（社交路径）
                        - 留乘行为扩散 → 负载离散（主动/被动滞留）

                    ## 失效机理图谱

                    ### 空间维度偏差链：
                    - [票价差异/拓扑失真] → 有效路径集失配 → 选择概率倒置
                    ↖ 参数标定滞后 ↗

                    ### 时间维度偏移链：
                    - [交路混沌/留乘扩散] → 车次匹配失准 → 断面流量时移
                    ↖ 末班断层 ↗

                    ### 突变传导路径：
                    - 运营事件冲击 → 路径可用性骤变 → 绕行路径涌现
                    ↘ 车次可靠性下降 → 留乘雪崩

                    ## 诊断决策流增强

                    ### 空间异常溯源：
                    1. 检查有效路径集生成器
                        - 验证模型参数 vs 网络复杂度指数
                    2. 解析广义费用构成
                        - 核查票价因子激活状态 / 换乘分布

                    ### 时间异常溯源：
                    1. 行车网络解码
                        - 交路模式冲突检测 / 末班衔接间隙分析
                    2. 行为模式匹配
                        - 留乘特征提取（滞留次数）

                    ### 突发畸变响应：
                    1. 运营事件影响域计算
                        - 构建冲击传播有向图
                    2. 路径重构监测
                        - 实时有效路径集动态扩缩

                    ##逻辑思维链
                    1.问题分析：首先对问题中的事件或现象进行详细分析，明确其特征和表现形式。
                    2.致因定位：根据问题特征，定位与之相关的失效致因。
                    3.影响判断：分析这些失效致因对清分系统的具体影响。
                    4.失效诊断：综合考虑失效致因及其影响，判断清分系统是否失效。若问题中的事件或现象导致清分系统的预测结果与实际客流情况偏差较大，且这种偏差无法通过正常调整和修正消除，则可判定模型失效。
                    
                    """



                    query_terms = current_query.split()[:3]  # 取前3个词作为关键词
                    # 新增：添加关键关系提示（使用 query_terms）
                    if "knowledge_graph" in knowledge_data:
                        related_edges = []
                        for term in query_terms:
                            # 搜索包含关键词的节点
                            matched_nodes = [
                                node for node in knowledge_data["knowledge_graph"].nodes 
                                if term.lower() in node.lower()
                            ]
                            # 收集相关边
                            for node in matched_nodes:
                                related_edges.extend(knowledge_data["knowledge_graph"].edges(node))
                        
                        # 去重
                        unique_edges = list(set(related_edges))
                        if unique_edges:
                            system_prompt += "\n\n【知识图谱关联路径】\n" + "\n".join(
                                [f"{src} → {dst}" for (src, dst) in unique_edges[:5]]  # 最多显示5条
                            )

                    prompt = f"{system_prompt}\n\n当前对话上下文：\n"
                    for entry in history[-5:]:      #保留最后5条历史记录
                        if entry["role"] == "user":
                            prompt += f"用户: {entry['content']}\n"
                        elif entry["role"] == "assistant":
                            prompt += f"助理: {entry['content']}\n"
                    prompt += "助理: "  # 最后添加引导回答的标记
                    return prompt

                def process_response(raw_answer):
                    """智能声明管理方案"""
                    # 阶段1：内容标准化
                    normalized = re.sub(r'\u3000', ' ', raw_answer)  # 处理全角空格
                    
                    # 阶段2：声明检测与替换
                    declaration_pattern = r'※[\s\u3000]*本推理基于.*?现场验证[\s\u3000]*'
                    
                    # 保留最后一个声明
                    parts = re.split(declaration_pattern, normalized)
                    cleaned_content = ''.join(parts).strip()
                    
                    # 阶段3：标记关键假设
                    marked_content = re.sub(
                        r'(?<!⚠️)(假设|若)', 
                        '⚠️\g<1>', 
                        cleaned_content
                    )
                    
                    # 阶段4：智能添加声明
                    if len(parts) > 1 or not re.search(declaration_pattern, normalized):
                        final_output = f"{marked_content}\n\n※ 本推理基于行业通用分析框架，实际结论需现场验证"
                    else:
                        final_output = marked_content
                    
                    # 阶段5：格式优化
                    return re.sub(r'\n{3,}', '\n\n', final_output)

                # 添加状态存储
                chat_history = gr.State([])

                def generate_answer(question, chat_history, temp, top_k, top_p):
                    try:
                        # 临时保存原始历史用于错误恢复
                        original_history = chat_history.copy()
                        
                        # 1. 添加新用户消息到历史记录
                        updated_history = chat_history + [{"role": "user", "content": question}]
                        
                        # 2. 构建提示词（调试用打印）
                        full_prompt = build_prompt(updated_history)
                        print("DEBUG - 当前提示词:\n", full_prompt)  # 调试输出
                        
                        # 3. 请求模型生成
                        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": "deepseek-r1:8b",
                                "prompt": full_prompt,
                                "temperature": float(temp),
                                "top_k": int(top_k),
                                "top_p": float(top_p),
                                "repeat_penalty": 1.5,  # 降低重复内容
                                "stop": ["关键假设:", "验证路径:"],     # 防止截断
                                "stream": False
                            },
                            timeout=120
                        )
                        
                        # 4. 处理响应（关键修复点）
                        if response.status_code == 200:
                            raw_answer = response.json().get("response", "").strip()
                            # 新增内容清洗逻辑
                            answer = re.sub(r'<think>.*?</think>\s*', '', raw_answer, flags=re.DOTALL)
                            answer = re.sub(r'^\s*(助理|AI|机器人)[:：]?\s*', '', answer)
                            answer = process_response(answer)
                        else:
                            answer = f"请求失败（状态码 {response.status_code}）"
                        
                        # 5. 空回答保护
                        answer = answer.strip() or "（未获得有效响应，请尝试调整参数或重新提问）"
                        print("DEBUG - 生成回答:", answer)  # 调试输出
                        
                        # 6. 更新完整历史记录
                        final_history = updated_history + [{"role": "assistant", "content": answer}]
                        
                        # 7. 转换为Gradio兼容格式
                        formatted_messages = []
                        for msg in final_history:
                            # 严格匹配Gradio消息格式
                            formatted_msg = {
                                "role": "user" if msg["role"] == "user" else "assistant",
                                "content": msg["content"]
                            }
                            formatted_messages.append(formatted_msg)
                        
                        print("DEBUG - 格式化消息:", formatted_messages)  # 调试输出

                    except Exception as e:
                        # 错误处理流程
                        print(f"ERROR - 发生异常: {str(e)}")
                        formatted_messages = original_history.copy()
                        error_msg = f"系统错误: {str(e)}"
                        formatted_messages += [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": error_msg}
                        ]
                        final_history = formatted_messages
                    
                    # 确保返回最新状态
                    return "", formatted_messages, final_history

                ask_button.click(
                    generate_answer,
                    inputs=[question, chat_history, temperature, top_k, top_p],
                    outputs=[question, chatbot, chat_history]  # 三个输出对应三个返回值
                )

                # 增强的清空功能
                def clear_history():
                    return [], []  # 同时清空显示和历史记录

                reset_button.click(
                    clear_history,
                    outputs=[chatbot, chat_history],
                    queue=False
                )

                # 为所有示例按钮绑定事件（新增多个按钮处理）
                example_questions = [
                    example_btn1, example_btn2, example_btn3, example_btn4, example_btn5, example_btn6
                ]
                question_texts = [
                    "海伦路站故障期间清分系统应急路径库未激活导致的周边站点分流失效评估",
                    "浦东机场站清分模型节假日参数组缺失，导致铁路联运客流预测失准的风险评估",
                    "徐家汇站15号线接入后清分系统OD反推模型未及时校准，导致早高峰断面预测持续偏离的失效路径分析",
                    "迪士尼站跨年夜清分系统反向阻抗系数未动态调整，导致11号线车次匹配失序的失效预测",
                    "虹桥枢纽清分路径特征库未融合新线参数引发的既有线通勤路径迁移失真预测",
                    "新生入学时，导致同济大学站清分系统失效的原因有哪些"
                ]
                
                for btn, text in zip(example_questions, question_texts):
                    btn.click(
                        lambda t=text: t,
                        outputs=question,
                        api_name=False
                    ).then(
                        generate_answer,
                        inputs=[question, chat_history, temperature, top_k, top_p],
                        outputs=[question, chatbot, chat_history]
                    )

    # 自定义CSS样式（新增按钮高度和布局调整）
    demo.css += """
    /* 示例按钮容器 */
    .example-group {
        margin-top: 12px;
        border-top: 1px solid #e0e0e0;
        padding-top: 12px;
    }

    /* 示例标题样式 */
    .example-title {
        font-size: 0.9em;
        color: #666;
        margin-bottom: 8px;
        font-weight: 500;
    }

    /* 示例按钮统一样式 */
    button.example-btn {
        min-height: 42px !important;  /* 调整按钮高度 */
        padding: 8px 12px !important;
        white-space: normal !important;  /* 允许文字换行 */
        line-height: 1.4 !important;
        transition: all 0.2s ease;
    }

    /* 悬停效果增强 */
    button.example-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.12);
    }

    /* 移动端适配 */
    @media (max-width: 768px) {
        button.example-btn {
            font-size: 0.9em;
            padding: 6px 10px !important;
        }
    }
    """

    return demo


if __name__ == "__main__":
    # 在界面加载时检查
    if not check_service():
        raise RuntimeError("Ollama服务连接失败")
    # 初始化预测系统
    system = PredictionSystem()
    
    # 启动界面
    demo = create_visualization(system)
    demo.launch(
        server_port=7860,
        server_name="localhost",
        show_error=True
    )