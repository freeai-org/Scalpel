# GitHub Pages 与项目文档

`docs/` 既保存项目说明，也作为 Highway 项目页的 GitHub Pages 发布目录。

## Highway 项目页

本地预览：

```bash
cd /home/alex/soulgard/soulgard-vl
python -m http.server 8000 --directory highway/Scalpel/docs
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

GitHub Pages 部署：

1. 将修改推送到 GitHub 的 `main` 分支。
2. 打开仓库 `Settings → Pages`。
3. 在 `Build and deployment` 中选择 `Deploy from a branch`。
4. Branch 选择 `main`，目录选择 `/docs`，点击 `Save`。
5. 页面将发布到 `https://soulgard.github.io/soulgard-vl/`。

页面是纯 HTML/CSS/JavaScript，不需要 npm 构建。当前页面说明的是最终 logits
CE+KL 恢复协议；`assets/` 中的方法图和 Loss 图来自 Highway 的实验结果。
逐轮图表与表格数据内嵌在 `app.js` 中，因此在项目页、用户页或本地静态服务器下都能工作。

## Contents

| Path | Purpose |
|---|---|
| `index.html` | Highway / Scalpel 项目页结构与内容 |
| `styles.css` | 响应式视觉设计、动画与可访问性适配 |
| `app.js` | 真实实验数据、交互图表、逐轮层映射和页面交互 |
| `assets/` | 方法图、Loss 图和站点图标 |
| `rl_notes/` | Reinforcement-learning and post-training notes for the RL experiments. |
