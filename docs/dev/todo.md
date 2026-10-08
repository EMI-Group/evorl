# 开发待办

- [ ] PRNGKey 改为新 API（`jax.random.key` / typed keys）。
  同步检查随机工具、环境包装器和演化算子中对 key 的 `uint32[..., 2]` 形状假设，以及 checkpoint 兼容性。
- [ ] 检查 `check_vma=False` 相关使用地方的合理性。
  逐项核对 `shard_map` 的输入/输出分片声明、跨设备归约及梯度行为，确认哪些调用可以启用检查。
