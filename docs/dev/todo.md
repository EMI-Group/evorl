# 开发待办

- [ ] 拆开 PopEpisodicTD3 与 CEMRL 的工作流继承关系。
  独立 TD3 种群不再依赖 CEMRL 模板、跳过父类构造函数或保留无用的 `ec_opt_state`、`num_learning_offspring`。
- [ ] 拆开 CEMRL 与 ERL 的 actor 参数导入策略。
  清理 `cemrl_replace_td3_actor_params` 对 ERL 实现的别名复用，各自明确 target actor 和网络辅助状态的恢复规则。

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
- [x] 实现并验证多智能体训练算法。
  IPPO、MAPPO、MADDPG、MATD3 已实现；测试覆盖损失、完整训练、nested vmap、双设备和 JaxMARL／Jumanji 环境。IPPO／MAPPO 使用 ENVPOOL autoreset。
- [ ] 对齐 A2C／IMPALA 的多设备有效样本权重。
  当前 actor／critic／entropy loss 仍先取每卡均值再同步；有效样本数不同时与全局样本均值不同，空 mask 还可能产生 NaN。IMPALA 的 KL 指标已单独修复。
- [ ] 调整 PPO／ParamPPO 的观测统计更新时机。
  `normalize_obs=True` 时，采样与训练可能使用不同的统计量；应在整批 PPO 更新完成后再更新统计量，并验证行为策略概率和 value 的一致性。
- [ ] 修复宿主环境的独立兼容问题。
  EnvPool 会将不支持的 `record_ori_obs` 参数传给原生构造函数；PBT-SAC 的 Gymnasium 私有 replay 预热在 Manual `META_POP` 内触发 callback 输出 VMA 丢失。训练前 reset 已补齐，这两项尚未修复。
- [x] 移除核心 EC 算法对 EvoX 的依赖并验证 NSGA-II 优化流程。
  CMA-ES、SepCMA-ES 和 NSGA-II 直接实现 `EvoOptimizer`；旧 EvoX 算法、适配器与配置已删除。测试覆盖原生更新、ZDT1 优化、nested vmap、单/多设备 workflow，以及真实 Brax Ant 多目标奖励的两代优化。
- [x] 修复 CMA-ES 正权重配置约束与 EC 训练返回值。
  按论文附录 C 保留种群相关对数权重，限制 `1 <= num_elites <= pop_size // 2`；白化改用矩阵—向量乘法。ES 与 NSGA-II 的 `learn()` 返回最终状态，测试同时覆盖已完成训练的空循环。
- [x] 优化 `non_dominated_sort` 的大种群性能。
  纯 JAX 实现分块排序、双目标 sweep 和 ENS-SS/BS；`auto` 在 CPU 上按目标数选择 sweep 或 ENS-BS，其他后端使用分块算法，避免保留完整支配矩阵。入口要求至少两个目标，旧实现注释保留用于对照。
