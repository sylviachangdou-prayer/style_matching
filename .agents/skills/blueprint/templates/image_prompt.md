# 文生图 Prompt 模板(architecture presentation diagram)

> **定位**:这是与 deterministic render 并行的 generate / presentation track。两条轨道共用 `schema.yaml` 和 composition contract；render 负责可维护真值，image-gen 负责更强的整体空间构图和展示效果。
>
> **怎么用**:照 schema.yaml 的内容填占位符 `<...>`,把下面整段交给当前环境可用的 image-generation 工具。候选图保存为 `diagram.generated.png`,不能替代 schema、SVG 或 renderer。
>
> **如果多次输出仍然崩坏**(框数错、文字乱、箭头乱指),保留已经完成的 render track,并把 generate track 记为 failed/skipped；不要把失败藏起来,也不要让它阻塞架构真值交付。
>
> 中间 `[`...`]` 标的是要根据 schema 写出来的循环段落,不是 prompt 的一部分。

---

```text
A clean modern architecture diagram, flat infographic style, white background, 3:2 aspect ratio, pixel-aligned, no shadows, no 3D, no gradients.

Title (centered, top, bold dark blue): "<meta.title>"
Subtitle (centered, just below title, smaller gray): "<meta.subtitle>"

Layout: <若 direction == top_to_bottom 写 "vertically stacked rows",若 left_to_right 写 "horizontally stacked columns"> of rounded-rectangle cards. Each card has 2px solid colored border with a very light pastel fill of the same hue. All connecting arrows are STRICTLY ORTHOGONAL — only 90° bends, NO diagonal lines.

──────────── COMPOSITION CONTRACT ────────────

[ 对 composition.paths 每项写一行: ]
• <role> path "<id>": <node.label> → <node.label> → ...

[ 对 composition.lanes 每项写一行: ]
• Keep lane "<id>" on the <left / center / right>, with these cards visually aligned across layers: <node.label>, <node.label>, ...

The primary path must be the strongest visual reading order. Context and secondary paths stay peripheral and lighter. Feedback routes use an outer lane when possible. Preserve these semantic-to-spatial relationships even if exact pixel positions change.

──────────── NODES ────────────

[ 对每个 node 写一段,先按 composition.lanes,再按 layer + align 分组,每节点格式: ]

• <category 颜色名> <shape: card / cylinder / file-icon> at <layer-align position>:
   line 1 (bold): "<label>"
   line 2 (regular): "<signature>"  [若有]
   line 3 (small, after a thin divider): "<note>"  [若有]
   [若 style == dashed,加一句: "Border is DASHED, indicating evaluation/optional path."]
   [若 expand,加一句: "Mark with a small subtle indicator (e.g., layered shadow) suggesting drill-down detail exists."]

──────────── ARROWS ────────────

[ 对每个 edge 写一行,kind 决定描述: ]

• <from.label> → <to.label>:
    - data_flow:    "<color from from-node category> arrow, <direction>, label '<edge.label>' above the line"
    - config_input: "<color> arrow, no label"
    - orchestration: "gray arrow, <direction>"
    - calibration: "DASHED <color> arrow going <direction>, label '<edge.label>' beside the line"

──────────── STYLE ────────────

Typography: <若有中文 → "Chinese and English text crisp and legible">; card titles bold; function signatures monospace; subtexts regular weight. No watermark.
```

---

## 跑模型的额外提示

- 第一次跑出来对照下面 checklist:
  - [ ] 节点数对得上 schema.yaml 的 `len(nodes)`
  - [ ] 中英混排没乱码 / 没错位
  - [ ] 箭头方向跟 `edges` 一致(特别注意 calibration 那种反向回喂)
  - [ ] primary/context/feedback 的阅读顺序与 `composition` 一致
  - [ ] lane 内节点跨层对齐,context/side service 没挤进主路径
  - [ ] 没出现斜线(模型经常无视约束)
  - [ ] 配色跟 category 大致对得上(模型很难精确匹配 hex,达到"是粉色就行"即可)
- 一次跑 4-8 张,挑最好那张存为 `diagram.generated.png`
- 把 `diagram.generated.png` 与 `diagram.render.png` 并排检查；语义对齐后,更适合报告/幻灯片的候选可以复制为 `diagram.png`
- 如果工具支持参考图,可以把上一版生成候选或 approved presentation reference 作为风格/构图参考；节点和边仍以 schema 为准
