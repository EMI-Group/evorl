# 开发待办

- [x] PRNGKey 改为新 API（`jax.random.key` / typed keys）。
  已迁移构造函数、批量 key 处理及 host callback，并验证 typed key checkpoint；不支持旧 checkpoint 格式迁移。
- [ ] 检查 `check_vma=False` 相关使用地方的合理性。
  逐项核对 `shard_map` 的输入/输出分片声明、跨设备归约及梯度行为，确认哪些调用可以启用检查。
- [ ] 补齐普通 RL/EC 的 DP `step/evaluate` 执行封装。
  `enable_shmap` 当前仍为空方法；应在 `shard_map` 的 Manual 函数体内执行本地计算及显式归约，并验证复制模型与分片 key/环境/replay buffer 的混合状态。
