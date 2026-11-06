# dsh-agent

基于 `deepseek-harness-sdk` 开发的持久化 Web 对话 Agent 项目脚手架。

## 📖 博客

项目原理与开发记录见博客：
🔗 [https://www.chengpei.top/archives/deepseek-harness-ai-agent](https://www.chengpei.top/archives/deepseek-harness-ai-agent)

## 📦 项目结构

```
dsh-agent/
├── server.py        # FastAPI + SSE 持久 Web 对话服务
├── web/
│   └── index.html   # 前端界面
├── workspace/       # 运行时工作区（gitignore）
├── dsh-home/        # 运行时用户会话/存储（gitignore）
└── .env             # 配置环境变量（gitignore）
```

## 🚀 快速开始

1. **配置环境变量**

   ```bash
   cp .env.example .env
   # 编辑 .env，填入真实的 DEEPSEEK_API_KEY
   ```

2. **启动服务**

   ```bash
   python server.py
   ```

   访问 http://localhost:8000 即可。