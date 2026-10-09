# 开发待办

- [x] PRNGKey 改为新 API（`jax.random.key` / typed keys）。
  已迁移构造函数、批量 key 处理及 host callback，并验证 typed key checkpoint；不支持旧 checkpoint 格式迁移。
- [ ] 检查 `check_vma=False` 相关使用地方的合理性。
  逐项核对 `shard_map` 的输入/输出分片声明、跨设备归约及梯度行为，确认哪些调用可以启用检查。
- [x] 补齐普通 RL/EC 的多卡 `step/evaluate` 执行。
  子类的 `enable_shmap` 类方法直接定义混合 State 布局，以 `in_specs=None` 声明静态 `self`；构建时先应用 `shard_map`，再按需应用 JIT。两设备测试覆盖 PPO、SAC、OpenES 的初始化、训练和评估。
