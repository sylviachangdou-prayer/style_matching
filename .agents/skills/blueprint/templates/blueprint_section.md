<!--
  把下面这段抄到 BLUEPRINT.md 的 "## 一、系统架构图" 节(没有就新建)。
  路径 docs/architecture/ 相对于 BLUEPRINT.md 所在目录。
  下钻情况:如果还有子图,在本段后面加一句 "下钻详图见 [docs/architecture/<id>/](docs/architecture/<id>/)"。
-->

## 一、系统架构图

![架构图](docs/architecture/diagram.png)

架构图源文件都在 [`docs/architecture/`](docs/architecture/):

- [`schema.yaml`](docs/architecture/schema.yaml) —— **真值**,改架构改这个
- [`diagram.svg`](docs/architecture/diagram.svg) —— 确定性 renderer 生成的可编辑维护图
- [`diagram.render.png`](docs/architecture/diagram.render.png) —— render 轨道的位图候选
- [`diagram.generated.png`](docs/architecture/diagram.generated.png) —— image-gen 轨道的展示候选(能力可用时)
- [`diagram.png`](docs/architecture/diagram.png) —— 两个候选并排审阅后选定的人类展示图
- [`diagram.html`](docs/architecture/diagram.html) —— 可选预览页
- [`diagram.mmd`](docs/architecture/diagram.mmd) —— Mermaid fallback,粘到 [mermaid.live](https://mermaid.live) 可看
- [`render.py`](docs/architecture/render.py) —— 项目本地 renderer(如存在)

这些 view 必须与 schema 保持一致。改架构先改 schema,然后同步 Mermaid fallback,并行重跑 render/image-gen 轨道,最后重新选择 `diagram.png`。

---
