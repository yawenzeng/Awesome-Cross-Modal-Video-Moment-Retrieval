# Awesome-Cross-Modal-Video-Moment-Retrieval
持续更新该方向的重要论文，并做一句话简评。

* Temporally Language Grounding，text-to-clip retrieval，query-based moment retrieval等等都是本领域的工作。旨在使用一段文本检索视频中的对应片段/时刻。

* 部分文章涉及多个“创新点”，暂时只整理了最突出的贡献，完整论述请对应原文。

* 有 :o: 的论文是个人认为比较突出的必看论文。

* 部分重要论文已经整理在博客中：https://nakaizura.blog.csdn.net/article/details/107316185

# 2017:2
1[ICCV 2017] [CTRL] TALL Temporal Activity Localization via Language Query:o:   
2[ICCV 2017] [MCN] Localizing Moments in Video with Natural Language:o:  

[ICCV 2017] [CTRL] TALL Temporal Activity Localization via Language Query  
* 动机：开山之作，点出该问题需要解决的两大困难1作为跨模态任务，如何得到适当的文本和视频表示特征，以允许跨模态匹配操作和完成语言查询。2理论上可以生成无限粒度的视频片段，然后逐一比较。但时间消耗过大，那么如何能够从有限粒度的滑动窗口做到准确的具体帧定位。
* 方法：特征抽取--使用全局的sentence2vec和C3D。模态交互--加+乘+拼+FC。多任务：对齐分数+回归。

[ICCV 2017] [MCN] Localizing Moments in Video with Natural Language  
* 方法：特征抽取--使用全局的LSTM和VGG。模态交互--无。多任务：模态间的均方差（而不是预测对齐分数），同时使用RGB和 optical flow光流。

#年度关键词：两条路。


# 2018:5
3[MM 2018] Cross-modal Moment Localization in Videos:o:  
4[SIGIR 2018] Attentive Moment Retrieval in Videos:o:  
5[EMNLP 2018] Localizing Moments in Video with Temporal Language  
6[EMNLP 2018] Temporally Grounding Natural Sentence in Video  
7[IJCAI 2018] Multi-modal Circulant Fusion for Video-to-Language and Backward  

[MM 2018] Cross-modal Moment Localization in Videos  
* 动机：强调时间语态问题，即句子中的“先”“前”往往被忽视，需要更细致的理解。
* 方法：尝试结合上下文，用文本给视频加attention。

[SIGIR 2018] Attentive Moment Retrieval in Videos  
* 方法：尝试结合上下文，用视频给文本加attention。以上两篇的特征处理由于要加attention就开始局部化。

[EMNLP 2018] Localizing Moments in Video with Temporal Language  
* 动机：时序上下文是很重要的，不然无法理解句子中一些词之间的关系。
* 方法：延续MCN的思路走算匹配度无模态交互，且同时使用RGB和 optical flow光流。特征抽取--LSTM，视频表示上同时融合了多种上下文。

[EMNLP 2018] Temporally Grounding Natural Sentence in Video  
* 动机：捕捉视频和句子之间不断发展的细粒度的交互。
* 方法：从每帧/逐词。使用两组lstm处理每个时刻文本和视频的交互。

[IJCAI 2018] Multi-modal Circulant Fusion for Video-to-Language and Backward   
* 动机：增强视觉和文本表达的融合
* 方法：同时使用vector和matrix的融合方式，其中circulant matrix的操作方式是每一行平移一个元素以探索不同模态向量的所有可能交互。

#年度关键词：上下文。


# 2019: 16
8[AAAI 2019] Localizing Natural Language in Videos  
9[AAAI 2019] Multilevel language and vision integration for text-to-clip retrieval  
10[AAAI 2019] Semantic Proposal for Activity Localization in Videos via Sentence Query  
11[AAAI 2019] To Find Where You Talk Temporal Sentence Localization in Video with Attention Based Location Regression  
12[AAAI 2019] Read, Watch, and Move Reinforcement Learning for Temporally Grounding Natural Language Descriptions in Videos:o:  
13[CVPR 2019] MAN_ Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment:o:  
14[CVPR Workshop 2019] Tripping through time Efficient Localization of Activities in Videos  
15[CVPR 2019] Language-Driven Temporal Activity Localization A Semantic Matching Reinforcement Learning Model:o:  
16[CVPR 2019] Weakly Supervised Video Moment Retrieval From Text Queries:o:  
17[ICMR 2019] Cross-Modal Video Moment Retrieval with Spatial and Language-Temporal Attention:o:  
18[MM 2019] Exploiting Temporal Relationships in Video Moment Localization with Natural Language  
19[NIPS 2019] Semantic Conditioned Dynamic Modulation for Temporal Sentence Grounding in Videos  
20[SIGIR 2019] Cross-modal interaction networks for query-based moment retrieval in videos  
21[WACV2019] MAC: Mining Activity Concepts for Language-Based Temporal Localization  
22[ArXiv 2019] Temporal Localization of Moments in Video Collections with Natural Language  
23[ArXiv 2019] wMAN Weakly-supervised Moment Alignment Network for Text-based Video Segment Retrieval  

[AAAI 2019] Localizing Natural Language in Videos  
* 动机：以前的工作忽略模态间细粒度的交互和上下文信息。
* 方法：交叉门控的递归网络来匹配自然句子和视频序列（类似cross-attention），以完成细粒度交互。

[AAAI 2019] Multilevel language and vision integration for text-to-clip retrieval  
* 动机：global embedding缺少细粒度，且不同模态之间独立嵌入缺乏交互。
* 方法：先融语义再用R-C3D生成候选时刻以加快效率。同时用LSTM对查询语句和视频片段之间的细粒度相似性进行时序融合。

[AAAI 2019] Semantic Proposal for Activity Localization in Videos via Sentence Query  
* 动机：滑动窗口数据量多，效率低。
* 方法：将句子中的语义信息集成到动作候选生成过程中以得到更好的时刻候选集。
* 其他：该任务不同于时序动作定位的地方 （1）某一个动作是各种行动的组合，可能持续很长时间。 （2）句子查询不限于预定义的类列表。 （3）视频通常包含多个不同的活动实例。

[AAAI 2019] To Find Where You Talk Temporal Sentence Localization in Video with Attention Based Location Regression  
* 动机：1视频的时间结构和全局上下文没有充分探索2句子处理太粗3候选切分太耗时。
* 方法：特征抽取--全局+局部，即用C3D+BiLSTM处理视频，用Glove+BiLSTM处理文本。然后co-attention进行对齐。并且不做切分，直接预测定位点。

[AAAI 2019] Read, Watch, and Move Reinforcement Learning for Temporally Grounding Natural Language Descriptions in Videos  
* 动机：候选太耗时。
* 方法：首次将强化学习引入到这个领域，处理成顺序决策问题。

[CVPR 2019] MAN_ Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment  
* 动机：语义失调和结构失调（候选集之间的处理是独立的）。
* 方法：特征抽取--LSTM和I3D。使用迭代图学习候选时刻之间的关系。

[CVPR Workshop 2019] Tripping through time Efficient Localization of Activities in Videos  
* 动机：1文本和视频的联合表示以local细粒化。2候选效率低且不符合人类的自然定位方法。
* 方法：使用门控注意力建模细粒度的文本和视觉表示+强化学习定位。

[CVPR 2019] Language-Driven Temporal Activity Localization A Semantic Matching Reinforcement Learning Model  
* 动机：直接匹配句子与视频内容使强化学习表现不佳，更需要更多的关注句子和图片中目标的细粒度来增强语义。
* 方法：在强化学习框架下引入语义特征。

[CVPR 2019] Weakly Supervised Video Moment Retrieval From Text Queries  
* 动机：标注句子边界太耗时，且不可扩展在实践中
* 方法：弱监督--在训练时不需要对文本描述进行时间边界注释，而只使用视频级别的文本描述。技术上主要使用文本和视频在时序上的注意力对齐。

[ICMR 2019] Cross-Modal Video Moment Retrieval with Spatial and Language-Temporal Attention  
* 动机：没有关注空间目标信息和文本的语义对齐，且需要结合视频的时空信息。
* 方法：特征抽取--全局+局部，即全局还是用C3D和BiLSTM。然后全局特征分别给局部（词和目标）加attention突出重点。

[MM 2019] Exploiting Temporal Relationships in Video Moment Localization with Natural Language  
* 动机：之前对文本的处理是Global，而文本中事件之间的时间依赖和推理没有得到充分考虑。
* 方法：树注意网络将文本分为主事件，上下文时间，时间信号。

[NIPS 2019] Semantic Conditioned Dynamic Modulation for Temporal Sentence Grounding in Videos  
* 动机：现有的方法主要做语义匹配和对齐，而忽略了句子信息在视频的时间关联和组合中起着重要作用。
* 方法：提出语义条件动态调制(SCDM)机制，它依靠句子语义来调节时间卷积运算，从而更好地关联和组合句子相关的视频内容。

[SIGIR 2019] Cross-modal interaction networks for query-based moment retrieval in videos  
* 动机：对句子处理的细粒度不够（语境长依赖）。
* 方法：对切分的词构建语义树/图抽特征。

[WACV2019] MAC: Mining Activity Concepts for Language-Based Temporal Localization  
* 动机：联合嵌入的子空间忽略了有关视频和查询中动作的丰富语义线索
* 方法：从视频和语言模态中挖掘动作概念，其他部分同TALL。

[ArXiv 2019] Temporal Localization of Moments in Video Collections with Natural Language  
* 动机：欧几里德居来来衡量文本视频距离不好（MCN的对比路线），无法大规模应用。
* 方法：使用平方差距离更精准的对齐，并应用视频重排序方法以应对大规模视频库。

[ArXiv 2019] wMAN Weakly-supervised Moment Alignment Network for Text-based Video Segment Retrieval  
* 动机：自动推断无监督的视觉和语言表示之间的潜在对应关系/对齐。
* 方法：词条件视觉图以捕捉交互，并应用多实例学习完成无监督下的细粒度对齐。


#年度关键词：文本和视频的细粒化+强化学习+候选框关系/生成+弱监督。


# 2020:22
24[AAAI 2020] Temporally Grounding Language Queries in Videos by Contextual Boundary-aware Prediction  
25[AAAI 2020] Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video  
26[AAAI 2020] Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language:o:  
27[AAAI 2020] Weakly-Supervised Video Moment Retrieval via Semantic Completion Network  
28[TIP 2020] Moment Retrieval via Cross-Modal Interaction Networks With Query Reconstruction:o:  
29[MM 2020] Adversarial Video Moment Retrieval by Jointly Modeling Ranking and Localization:o:  
30[MM 2020] STRONG: Spatio-Temporal Reinforcement Learning for Cross-Modal Video Moment Localization  
31[MM 2020] Dual Path Interaction Network for video Moment Localization  
32[MM 2020] Regularized Two-Branch Proposal Networks for Weakly-Supervised Moment Retrieval in Videos  
33[ECCV 2020] VLANet: Video-Language Alignment Network for Weakly-Supervised Video Moment Retrieval  
34[WACV2020] Proposal-free Temporal Moment Localization of a Natural-Language Query in Video using Guided Attention  
35[ECCV 2020] TVR A Large-Scale Dataset for Video-Subtitle Moment Retrieval  
36[ArXiv 2020] Video Moment Localization using Object Evidence and Reverse Captioning  
37[ArXiv 2020] Graph Neural Network for Video-Query based Video Moment Retrieval  
38[ArXiv 2020] Text-based Localization of Moments in a Video Corpus  
39[ArXiv 2020] Generating Adjacency Matrix for Video-Query based Video Moment Retrieval  
40[BMVC 2020] Uncovering Hidden Challenges in Query-Based Video Moment Retrieval  
41[ArXiv 2020] Video Moment Retrieval via Natural Language Queries  
42[ArXiv 2020] Frame-wise Cross-modal Match for Video Moment Retrieval  
43[MM 2020] Fine-grained Iterative Attention Network for Temporal Language Localization in Videos  
44[MM 2020] Jointly Cross- and Self-Modal Graph Attention Network for Query-Based Moment Localization  
45[CVPR 2020] Dense Regression Network for Video Grounding  

[AAAI 2020] Temporally Grounding Language Queries in Videos by Contextual Boundary-aware Prediction  
* 动机：当模型无法定位最佳时刻时，添加的偏移回归可能会失败。
* 方法：语言集成的“感知”周围的预测，即每个时间步长都会预测时间锚和边界。

[AAAI 2020] Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video  
* 动机：现有的方法大多存在效率低下、缺乏可解释性和偏离人类感知机制（从粗到细）问题。
* 方法：基于树结构政策的渐进强化学习(TSP-PRL)框架，通过迭代细化过程来顺序地调节时间边界。

[AAAI 2020] Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language  
* 动机：以往模型一般单独考虑时间，忽视了时间的依赖性。（不能在当前时刻对其他时刻进行预测）。
* 方法：时间从一维变为二维，即变成二维矩阵，预测每一个位置的score。

[AAAI 2020] Weakly-Supervised Video Moment Retrieval via Semantic Completion Network  
* 动机：弱监督的注意力权重通常只对很明显的区域有反应，即候选太相似。
* 方法：候选生成--视觉语义词和文本对齐得到分数，并反馈给候选生成的打分。

[TIP 2020] Moment Retrieval via Cross-Modal Interaction Networks With Query Reconstruction  
* sigir19的扩充，增加查询重建任务，整体上联合自然语言查询的句法依赖、视频上下文中的长程语义依赖和足够的跨模态交互一起建模。

[MM 2020] Adversarial Video Moment Retrieval by Jointly Modeling Ranking and Localization  
* 动机：强化学习定位不稳定，检索任务候选框效率低。
* 方法：用对抗学习联合建模，即用强化学习定位来生成候选，进而在生成候选集上进行检索。

[MM 2020] STRONG: Spatio-Temporal Reinforcement Learning for Cross-Modal Video Moment Localization  
* 动机：空间上的场景信息很重要+候选框效率低。
* 方法：除了时序上的强化学习，其在空间上也加入强化学习，并以弱监督追踪的方式聚焦场景，去除冗余。

[MM 2020] Dual Path Interaction Network for video Moment Localization  
* 动机：很少有研究将候选框层级的对齐信息（算语义匹配分数的技术路线）和帧层级的边界信息（预测边界起止的路线）结合在一起，并考虑它们之间的互补性。
* 方法：双路径交互网络(DPIN)，分别做两大任务，同时两者之间通过信息交互相互增强。

[MM 2020] Regularized Two-Branch Proposal Networks for Weakly-Supervised Moment Retrieval in Videos  
* 动机：大多数现有的弱监督方法都采用基于多实例的框架来做样本间对抗，但忽略了相似内容之间的样本间对抗。
* 方法：同时考虑样本间和样本内的对抗，其中会采用语言感知得到两个分支的视觉表示（被增强的和被印制的），这两者将构成很强的正负样本对。

[ECCV 2020] VLANet: Video-Language Alignment Network for Weakly-Supervised Video Moment Retrieval  
* 动机：粗略的查询表示和单向注意机制导致模糊的注意映射。
* 方法：修剪候选集+更细粒度的多注意力机制来获得更好的注意力增加定位性能。

[WACV2020] Proposal-free Temporal Moment Localization of a Natural-Language Query in Video using Guided Attention  
* 动机：不依赖候选的端对端框架，文本注释主观性。
* 方法：使用attention的动态过滤器（利用文本对视频的过滤），软标签分布损失解决文本注释的不确定性。

[ECCV 2020] TVR A Large-Scale Dataset for Video-Subtitle Moment Retrieval  
* 动机：视频+字幕才能进一步的理解。
* 方法：提出一个新数据集，并设计一个类似多Transformer的模型完成不同模态的理解和融合。

[ArXiv 2020] Video Moment Localization using Object Evidence and Reverse Captioning  
* 动机：仅仅有语义概念是不够的，还需要对象和字幕的补充。
* 方法：在MAC（视频，视觉语义概念，文本，文本语义概念）的方法上再增加对象和字幕的特征。

[ArXiv 2020] Graph Neural Network for Video-Query based Video Moment Retrieval  
* 动机：帧间的特征相似度与视频间的特征相似度之间存在不一致，这会影响特征融合。
* 方法：沿时间维度对视频构多个图，然后再做多图特征融合。

[ArXiv 2020] Text-based Localization of Moments in a Video Corpus  
* 动机：text-video可能会有一对多+搜索全库是必要的
* 方法：层次时刻对齐网络能更好的编码跨模态特征以理解细微的差异，然后做模态内的三元组和模态间的三元组以应对搜索所有的时刻。

[ArXiv 2020] Generating Adjacency Matrix for Video-Query based Video Moment Retrieval  
* 动机：现有利用图技术的方法得到的邻接矩阵是固定的。
* 方法：使用图抽取模态内和模态间的特征关系。

[BMVC 2020] Uncovering Hidden Challenges in Query-Based Video Moment Retrieval  
* 讨论性文章：虽然现有技术已经取得了很好的提高，这篇文章探究了存在的两大问题，1数据集存在偏差，2评价指标不一定可靠。

[ArXiv 2020] Video Moment Retrieval via Natural Language Queries  
* 动机：起止时间标记有噪音，需要结合上下文的更好表示。
* 方法：多头自我注意+交叉注意力，以捕获视频/查询交互和远程查询依存关系。多任务学习中加入moment segmentation任务。

[ArXiv 2020] Frame-wise Cross-modal Match for Video Moment Retrieval  
* 动机：以往的方法不能很好地捕获查询和视频帧之间的跨模态交互。
* 方法：利用注意力对齐，做帧间的预测和边界预测两个任务。

[MM 2020] Fine-grained Iterative Attention Network for Temporal Language Localization in Videos
* 动机：现有工作只集中在从视频到查询的单向交互上。
* 方法：针对句子和视频的cross-attention做双向相互。

[MM 2020] Jointly Cross- and Self-Modal Graph Attention Network for Query-Based Moment Localization
* 动机：视频大部分都无关，只有小部分视频相关，所以应该更加重视两者的融合。
* 方法：提出模态间和模态内两种交互模式，以帧和词构图通过消息传递来增强表示。

[CVPR 2020] Dense Regression Network for Video Grounding
* 动机：以往训练的模型都是在正负例样本不平衡的情况下，所以作者尝试利用帧之间的距离作为更密集的监督信号。
* 方法：预测起始结束帧，然后认为只要在ground truth中的片段都认为是正例。

#年度关键词：任务扩展（如对全库搜索moment，增加字幕，标定鲁棒性等等），继续细化和融合，候选框关系。
#水文趋势开始增加...


# 2021:1
46[CVPR 2021] Multi-Modal Relational Graph for Cross-Modal Video Moment Retrieval:o:  

[CVPR 2021] Multi-Modal Relational Graph for Cross-Modal Video Moment Retrieval
* 动机：基于候选的方法会导致候选之间非常相似难以区别，那么直接理解内容（局部目标）的细微变化会是更好的选择。
* 方法：提出多模态关系图模型来捕捉视频中的局部目标之间的时空关系，并且在图预训练的框架下优化特征表达。

----------
# 稍作分类
* 两条路：处理成预测任务[1]；处理成匹配任务[2]。大多数文章都follow了[1]，也有少数沿着[2]的路线的文章如[5][22]，也有融合这两条路线的[33]。  

跨模态视频时刻相关任务主要需要集中解决的问题有两点：一.跨模态 二.高效检索/定位  

对于一.  

* 模态特征的细化：模态间的交互和融合[3-9，37，39，42]，局部对齐[17-19]，模态表征细粒度[20，21，28]，局部目标间的关系[46]
* 结合其他外部信息，如字幕[35，36]

对于二.  
检索方法主要涉及到候选集的问题  
* 常规方法会提前切分好，或者修剪候选集[33]
* 生成候选：预测一些可能的边界[10]
* 探讨候选之间的关系：使用图或者矩阵的形式[13，26]
* 正负例样本不平衡问题[45]

定位方法可分直接预测和强化学习  
* 直接预测起止点：[11，31，34，36，38]，对边界预测的多任务增强如边界感知[24]，moment segmentation[41]。
* 强化学习：[12，14，15，25，29]，主要涉及一些语义概念融合，如何使强化学习更高效。

也有融合定位和检索的方法[30]  

除了一和二问题的，其他话题  
* 标注text-video对太耗时，无监督方法[16，23，27，32]，主要会使用注意力对齐，正负样本对抗，多示例学习等等技术。
* 标注鲁棒性[34，40，41]，解决方案主要有比较分布和标注预测。
* 搜索全库视频时刻[35，38]，要求对模态内和模态外都有更好的理解和区分能力。
