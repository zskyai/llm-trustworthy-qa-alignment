# 项目设计

## 1. 问题背景

专业知识问答不同于开放聊天。真实业务更关注回答是否有依据、引用是否正确、信息不足时是否能保持保守。

## 2. 核心失败模式

- 答案流畅但证据不足
- 引用段落和回答结论不一致
- 无答案场景强行回答
- 过度扩展，超出证据范围

## 3. 后训练思路

SFT 用于学习基本回答格式和任务规范；DPO 用于学习偏好行为，即在候选答案中更偏向有证据、可追溯、保守的回答。

## 4. 偏好对设计

每条偏好样本包含 prompt、chosen、rejected。chosen 更符合证据一致性和业务可信性，rejected 体现常见错误模式。

## 5. 评测指标

- Answer correctness
- Citation precision
- Citation recall
- Unsupported claim rate
- Refusal accuracy
- Helpfulness under evidence constraints
