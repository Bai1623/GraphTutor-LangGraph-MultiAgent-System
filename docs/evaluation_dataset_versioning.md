# Golden Dataset 版本与元数据约定

所有 `eval/golden/*.yaml` 数据集必须包含顶层 `metadata`。元数据用于说明数据集的身份、来源和研究目的，并由加载器在评测开始前严格校验。

```yaml
metadata:
  dataset_name: supervisor_routing
  version: v1.0.0
  source: Manually curated Chinese Gaokao tutoring queries.
  updated_at: 2026-09-05
  research_goal: Measure Supervisor intent and subject routing accuracy.
```

字段约定：

- `dataset_name`：稳定的数据集名称，不随单次运行改变。
- `version`：三段式语义化版本 `vMAJOR.MINOR.PATCH`。新增 case 通常递增 `MINOR`，修正标签或删除 case 递增 `MAJOR`，文字说明修订递增 `PATCH`。
- `source`：数据来源和构造方式，不能只写“golden data”。
- `updated_at`：最后一次修改日期，使用 `YYYY-MM-DD`。
- `research_goal`：该数据集要验证的工程或科研问题，而不是泛泛描述功能。

评测 JSON 和 Markdown 报告会复制完整 `metadata`，并额外记录报告生成时间和当前 Golden 文件路径。这样一次实验可以同时追溯到数据集版本、代码版本和运行配置。

修改数据集时应遵循：

1. 先判断修改属于新增样本、标签修正还是描述修订。
2. 更新 `version` 和 `updated_at`。
3. 在提交说明或变更日志中说明修改原因。
4. 运行 `pytest tests/test_eval_harness.py tests/test_compression_harness.py`，确认所有套件仍能被加载。
