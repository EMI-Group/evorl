# 开发待办

- [ ] 拆开 PopEpisodicTD3 与 CEMRL 的工作流继承关系（暂缓）。
  独立 TD3 种群不再依赖 CEMRL 模板、跳过父类构造函数或保留无用的 `ec_opt_state`、`num_learning_offspring`。
- [ ] 拆开 CEMRL 与 ERL 的 actor 参数导入策略（暂缓）。
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
- [x] 对齐 A2C／IMPALA 的多设备有效样本权重。
  actor／critic／entropy loss 按全局有效样本数加权，空 mask 贡献零；IMPALA 的 rho／KL 指标使用相同口径。双设备回归比较合并批次的 loss、梯度和指标。
- [x] 调整 PPO／ParamPPO 的观测统计更新时机。
  A2C、IMPALA、PPO、ParamPPO 均在整批训练完成后更新观测统计。回归比较冻结采样时统计量的参考更新，覆盖 JIT 与非 JIT。
- [x] 修复宿主环境的参数传递与 callback VMA 问题。
  EnvPool 适配器接收 `record_ori_obs` 并提供 `info.ori_obs`；宿主 callback 输出显式标记当前 Manual 轴。PBT-SAC 的共享／私有 replay 均覆盖预热、训练和评估，保留默认 VMA 检查。
- [x] 支持 EnvPool 的 NORMAL autoreset。
  同一有序宿主回调中先推进整批环境，再只 reset 刚结束的子环境，按返回的 `env_id` 回填新观测；保留终止 reward／flags／`ori_obs`，正确重置步数和回报统计。覆盖并行、nested vmap、字典观测及共享／私有 replay 训练。FAST 不支持，创建环境前直接报错。
- [x] 统一 replay／archive 预填充的 episode 计数口径。
  旧 off-policy、TD7 与 MA／MO 的预填充均只计采样步数，不计 `sampled_episodes`；后者从正式训练开始累计，PCN 初始化 archive 同样处理。PGMORL 的 PPO warmup 属于实际训练，保留计数；ERL／CEMRL 的原有约定保留。
- [x] 提前检查 PopTD3 的预热参数。
  构建环境前要求 `0 <= random_timesteps <= learning_start_timesteps`，避免第二阶段计算负的 rollout 长度。保留零预热和两阶段长度相等的配置，并覆盖 JIT／非 JIT 初始化。
- [x] 对齐 PPO／MOPPO 的 critic loss 系数。
  MOPPO 移除均方误差上额外的 `0.5`，沿用普通 PPO 的外层 critic 权重；保留目标维平均与 value clipping。
- [x] 严格校验 PPO 类 rollout 的 minibatch 划分。
  PPO、ParamPPO 要求 rollout 样本数能被 `minibatch_size` 整除；contrib TD3-onpolicy 要求能被 `minibatch_size * actor_update_interval` 整除。在创建环境前抛出 `ValueError`，移除拼错字段的警告分支，避免丢弃余数。
- [ ] 核查 MJX 3.16.0／Playground 0.2.0 的多设备 VMA 报错。
  用户明确暂缓处理。Cartpole／Cheetah 的 line search 和 Pendulum 的 physics step 存在循环或分支类型不匹配；普通 PPO 也能触发。保留默认检查，当前不宣称这些组合可多设备训练。
- [x] 移除核心 EC 算法对 EvoX 的依赖并验证 NSGA-II 优化流程。
  CMA-ES、SepCMA-ES 和 NSGA-II 直接实现 `EvoOptimizer`；旧 EvoX 算法、适配器与配置已删除。测试覆盖原生更新、ZDT1 优化、nested vmap、单/多设备 workflow，以及真实 Brax Ant 多目标奖励的两代优化。
- [x] 修复 CMA-ES 正权重配置约束与 EC 训练返回值。
  按论文附录 C 保留种群相关对数权重，限制 `1 <= num_elites <= pop_size // 2`；白化改用矩阵—向量乘法。ES 与 NSGA-II 的 `learn()` 返回最终状态，测试同时覆盖已完成训练的空循环。
- [x] 优化 `non_dominated_sort` 的大种群性能。
  纯 JAX 实现分块排序、双目标 sweep 和 ENS-SS/BS；`auto` 在 CPU 上按目标数选择 sweep 或 ENS-BS，其他后端使用分块算法，避免保留完整支配矩阵。入口要求至少两个目标，旧实现注释保留用于对照。
