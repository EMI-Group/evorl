# 开发待办

- [x] 清理算法构建环境时重复传入的 `episode_length`。
  RL、EC、ERL、PBT 和 contrib 算法统一使用 `create_env()` 从环境配置的 `max_episode_steps` 获取默认上限；评估器自身的 `max_episode_steps` 参数保留。
- [x] PRNGKey 改为新 API（`jax.random.key` / typed keys）。
  已迁移构造函数、批量 key 处理及 host callback，并验证 typed key checkpoint；不支持旧 checkpoint 格式迁移。
- [x] 检查 `check_vma=False` 相关使用地方的合理性。
  已全部采用默认 VMA 检查，并修复循环与分支类型、gather 输出类型和梯度尺度。
  `pmean_loss_and_grad` 对全局平均 loss 求导，aux 保持本地；双 CPU 数值测试验证 loss、梯度和优先级数据。
- [x] 绕过 JAX 0.12.0 下连续动作训练的 `erf` VMA 报错。
  仅 JAX 0.12.0 使用局部兼容实现：普通区间调用 `log_ndtr`，负尾部直接对渐近展开式求导，避免 float32 梯度相消；其他版本保留原 TFP 路径。保留默认 VMA 检查，JAX 自身的问题仍需上游修复。
- [x] 补齐普通 RL/EC 的多卡 `step/evaluate` 执行。
  子类的 `enable_shmap` 类方法直接定义混合 State 布局，以 `in_specs=None` 声明静态 `self`；构建时先应用 `shard_map`，再按需应用 JIT。两设备测试覆盖 PPO、SAC、OpenES 的初始化、训练和评估。
- [x] 调整多目标奖励 wrapper 的命名。
  Brax 使用 `BraxMultiObjectiveRewardWrapper`；Playground 使用继承它的 `MujocoMultiObjectiveRewardWrapper`，复用相同的 metrics 取值和奖励转换逻辑。
- [ ] 实现并验证多智能体训练算法。
  `multi_agents/` 下的 IPPO、MAPPO、MADDPG、MATD3 文件目前均为空文件；环境 API 和 wrapper 测试不代表算法训练已经实现。
- [ ] 验证 NSGA-II 的完整优化训练。
  已验证通用 evaluator、向量目标聚合及分布式计数；当前测试环境未安装 EvoX，尚未运行完整优化训练。
