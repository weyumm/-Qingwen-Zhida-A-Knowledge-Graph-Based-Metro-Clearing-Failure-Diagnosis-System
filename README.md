# “清问智答”——基于知识图谱的地铁清分失效诊断系统
![我们的slogan](https://github.com/user-attachments/assets/4de8519e-25d0-4751-b5fd-70a2892906ed)
# "QingwenZhida"——A Knowledge Graph Based Metro Clearing Failure Diagnosis System
<a href="https://github.com/weyumm" target="_blank"><img  align=center src="https://img.shields.io/badge/项目介绍-Qfzd-%2316ff47?style=flat"/></a>    <a href="[https://space.bilibili.com/347006675](https://www.bilibili.com/video/BV1GVdwYMENa/?vd_source=17829e412fbf48cecf092ac260acc65b)" target="_blank"><img  align=center src="https://img.shields.io/badge/演示视频-Qfzd-%2324f9a6?style=flat"/></a>    <a href="https://gitee.com/weyumm" target="_blank"><img  align=center src="https://img.shields.io/badge/gitee-代码仓库-%2324eff9?style=flat"/></a>    <a href="https://gitlab.com/weyumm" target="_blank"><img  align=center src="https://img.shields.io/badge/gitlab-模型备份-%233ab7f2?style=flat"/></a>    <a href="https://modelscope.cn/profile/weiyumm" target="_blank"><img  align=center src="https://img.shields.io/badge/modelscope-魔塔社区-%23a73af2?style=flat"/></a>    <a href="https://weyumm.github.io/" target="_blank"><img  align=center src="https://img.shields.io/badge/Blog-技术博客-%23f283f0?style=flat"/></a>

全国大学生交通科技大赛作品（“清问智答”小组）

# 设计思路
## 1，双轨并行的推理检验架构
![创新点之图](https://github.com/user-attachments/assets/c7ce0fb8-8f28-4ded-a04e-bb8707a0d494)

## 2，面向垂直领域的L2大模型
- L0通识大模型 是指可以在多个领域和任务上通用的大模型。它们利用大算力、使用海量的开放数据与具有巨量参数的深度学习算法，在大规模无标注数据上进行训练，以寻找特征并发现规律，进而形成可“举一反三”的强大泛化能力，可在不进行微调或少量微调的情况下完成多场景任务，相当于AI完成了“通识教育”

- L1行业大模型 是指那些针对特定行业或领域的大模型。它们通常使用行业相关的数据进行预训练或微调，以提高在该领域的性能和准确度，相当于AI成为“行业专家”
  
- **L2垂直大模型** 是指那些针对特定任务或场景的大模型。它们通常使用任务相关的数据进行预训练或微调，以提高在该任务上的性能和效果，如该作品“清问智答”
![L0L1L2](https://github.com/user-attachments/assets/986a7b24-def7-4604-b5dd-956b3d463ce2)

# 视频演示
【最新一代模型】以下为演示视频

https://github.com/user-attachments/assets/f1773b8b-fb7e-4023-8c1a-37b20a1a6ff2


# 旧版本线上应用
【旧模型应用】可扫码试用

![二维码](https://github.com/user-attachments/assets/7beacdaa-ae4d-4169-9eb9-6572995bd528)
![a249a81b272302a3e454fe6867abda8](https://github.com/user-attachments/assets/74c9d939-e5b9-4e17-be0b-146bd9dd12db)
# 成果梗概
## 1，知识图谱
![image](https://github.com/user-attachments/assets/56691d97-0163-483d-827b-ca0a67343d94)

## 2，模型训练AUC、ROC曲线
在训练轮次达173次后，测试集数据交叉熵损失持续大于最低计算损失，早停机制触发停止训练；模型在训练集和验证集上都表现出了良好的收敛性和稳定性，最终训练损失和验证损失分别收敛于0.129、0.123；验证AUC在训练中期就一直保持在0.986，训练效果接近理想水平，表明模型具有较强的特征学习能力和泛化能力，能够有效地对数据进行拟合和预测，整体训练效果较为理想。
![image](https://github.com/user-attachments/assets/6df7f2c4-b692-44f8-ad75-a7257b9c5cb8)

## 3，上海地铁案例研究
根据三元组数据中龙阳路具有换乘站、大站车车站、早高峰大客流等特点，滴水湖、11号线、10号线、东方体育中心等也都有类似特点，故作为预测组；6号线、7号线、华宁路、五莲路、国帆路等均不具有相关特征，故作为参照组。分别检验它们与失效致因之间的可能性大小：
![image](https://github.com/user-attachments/assets/23cd55d3-c723-4e66-ae39-6240a37f1408)
![image](https://github.com/user-attachments/assets/4f0cff35-c994-403e-a648-1dadf280918c)
# 大模型特色
## 1，LoRA技术
![image](https://github.com/user-attachments/assets/3347aa51-ffba-4632-9046-35db6d8adf18)
![image](https://github.com/user-attachments/assets/6777d2b1-d78e-4c82-851c-d9d516edb290)

## 2，GraphRAG检索增强
![image](https://github.com/user-attachments/assets/1989addc-a655-4549-b704-376e4f288fc2)
![image](https://github.com/user-attachments/assets/80ad6974-d8f8-410c-a384-e6c1c1a4001b)
## 3，Prompt交互工程
- Prompt由人类设计的，以帮助模型更好地理解特定任务或领域。
- 对于大语言模型，一个好的 Prompt 决定了其能力的上限与下限，且会塑造其输出文本的内容、风格和整体质量。
- 我们基于乔哈里视窗的概念，并且结合地铁清分工作人员的专有名词与惯用语，设计了Prompt模版，旨在准确地将信息与任务传达给大模型。
![image](https://github.com/user-attachments/assets/67afdec5-9d8e-4407-85cd-6851c0e63212)
![image](https://github.com/user-attachments/assets/88ff693f-4a99-4a4c-aa65-392f6e37383b)

# 神经网络特色
## 1，基于文献驱动的数据集构建
使用VOSviewer工具，从237篇中文文献、262篇英文文献中，精炼核心关键词，并且根据其词频、热度、关联度等，形成【节点-边-节点】的网络结构，作为原始知识库，进行关键词消歧后，构造为数据集的一部分。
![image](https://github.com/user-attachments/assets/630c144d-7916-4050-8bb5-9fd92d546932)
## 2，图卷积神经网络(R-GCN)
为了对城市轨道交通网络客流分布计算失效知识图谱进行推理，并开展链路预测任务。我们构建了构建关系图卷积神经网络（R-GCN）模型。

- 从知识图谱中，随机抽取90%的“实体-关系”上海地铁10号线的数据作为正样本，共计166条。
- 在编码器-解码器框架下，输入为知识图谱中由实体和关系构成的三元组。
- 其中，编码器模块是一个R-GCN模型，用于实现交叉熵的计算。
![image](https://github.com/user-attachments/assets/c3a44c17-4619-40f7-b0df-31300065a801)
## 3，知识推理与知识类比的拓展方案
结果表明：

- 轨道交通10号线与轨道交通1号线、2号线、8号线、9号线及11号线都具有日均客流量大的特征。
- 当其他几条线路与留乘现象相关时，模型也能通过知识类比推理出10号线与留乘现象相关的可能性较高。
- 即10号线进站客流易出现留乘行为。相反，6号线与留乘现象相关的可能性较低。
![image](https://github.com/user-attachments/assets/2ca092aa-3c5e-4d21-a0fb-be0cbb4b3926)

# 双轨并行
借助两种人工智能学派的不同观点，分别以人工赋予智能（神经网络）、自主学习智能（大语言模型）的双轨并行架构，赋予【清问智答】既准确又灵活的输出结果。为赋能轨道交通领域智能化展现出一类创新的思路。
![image](https://github.com/user-attachments/assets/f17d7de7-4588-425a-9067-5f2f5106adc4)
![16ca65aaa243d9009808c707edc6439](https://github.com/user-attachments/assets/48da59bc-1b59-419d-a109-c86ab1a514ef)




