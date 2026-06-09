/**
 * 对话导出 PDF —— 通过浏览器打印功能生成 PDF
 *
 * 采用"打印友好 HTML → 新窗口 → window.print()"方案，因为：
 * 1. 浏览器原生支持中文渲染，不依赖字体文件
 * 2. 零额外依赖（不需要 html2canvas / jspdf 的字体嵌入）
 * 3. 用户可在打印对话框中选择"另存为 PDF"
 */

import type { Message } from "@/components/chat-area"

/**
 * 将对话消息导出为 PDF（通过浏览器打印）。
 *
 * @param messages 要导出的消息列表
 */
export function exportChatAsPdf(messages: Message[]) {
  if (messages.length === 0) return

  const now = new Date().toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  })

  // 生成打印友好的 HTML
  const html = buildPrintHtml(messages, now)

  // 在新窗口中打开并打印
  const printWindow = window.open("", "_blank", "width=800,height=600")
  if (!printWindow) {
    alert("弹窗被浏览器拦截，请允许此网站的弹窗后重试。")
    return
  }

  printWindow.document.write(html)
  printWindow.document.close()

  // 等字体加载完成后触发打印
  printWindow.onload = () => {
    setTimeout(() => printWindow.print(), 300)
  }
}

function buildPrintHtml(messages: Message[], exportTime: string): string {
  const messageHtml = messages
    .map((msg) => {
      const role = msg.role === "user" ? "学生" : "AI 助手"
      const roleColor = msg.role === "user" ? "#3D5A40" : "#5C3D2E"
      // 转义 HTML 特殊字符
      const content = escapeHtml(msg.content)
      // 简单处理换行
      const formatted = content.replace(/\n/g, "<br>")

      return `
        <div class="message">
          <div class="role" style="color:${roleColor}">${role}</div>
          <div class="content">${formatted}</div>
        </div>
      `
    })
    .join("\n")

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>对话记录 - GraphTutor</title>
  <style>
    @page {
      size: A4;
      margin: 20mm;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif;
      font-size: 13px;
      line-height: 1.8;
      color: #2D2D2D;
      max-width: 700px;
      margin: 0 auto;
      padding: 20px;
    }
    .header {
      text-align: center;
      border-bottom: 2px solid #3D5A40;
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    .header h1 {
      font-size: 20px;
      color: #3D5A40;
      margin-bottom: 6px;
    }
    .header .time {
      font-size: 12px;
      color: #888;
    }
    .message {
      margin-bottom: 18px;
      padding: 12px 16px;
      border-radius: 8px;
      background: #f9f9f9;
      page-break-inside: avoid;
    }
    .message .role {
      font-weight: bold;
      font-size: 12px;
      margin-bottom: 4px;
    }
    .message .content {
      white-space: pre-wrap;
      word-break: break-word;
    }
    .footer {
      text-align: center;
      font-size: 11px;
      color: #aaa;
      border-top: 1px solid #e0e0e0;
      padding-top: 12px;
      margin-top: 24px;
    }
    @media print {
      body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>GraphTutor - 高考辅导 AI 助手</h1>
    <div class="time">导出时间：${exportTime} | 共 ${messages.length} 条消息</div>
  </div>
  ${messageHtml}
  <div class="footer">
    GraphTutor 对话记录 · https://gitee.com/git_bai/work
  </div>
</body>
</html>`
}

function escapeHtml(text: string): string {
  const map: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }
  return text.replace(/[&<>"']/g, (c) => map[c] || c)
}
