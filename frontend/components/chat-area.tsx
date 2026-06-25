"use client"

import { useState, useRef, useEffect } from "react"
import { Send, Bot, User, FileUp, FileText, X, SlidersHorizontal, Mic, Download, ThumbsUp, ThumbsDown, Loader2 } from "lucide-react"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { exportChatAsPdf } from "@/lib/export-pdf"

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
}

interface ChatAreaProps {
  messages: Message[]
  onSendMessage: (content: string) => void | Promise<void>
  onUploadDocuments?: (files: File[], question: string) => Promise<boolean>
  isLoading?: boolean
}

export function ChatArea({ messages, onSendMessage, onUploadDocuments, isLoading }: ChatAreaProps) {
  const [showExportTip, setShowExportTip] = useState(false)
  const [input, setInput] = useState("")
  const [pendingFiles, setPendingFiles] = useState<File[]>([])
  const [isUploadingDocument, setIsUploadingDocument] = useState(false)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const question = input.trim()
    if (!question || isLoading || isUploadingDocument) return

    if (pendingFiles.length && onUploadDocuments) {
      setIsUploadingDocument(true)
      try {
        const sent = await onUploadDocuments(pendingFiles, question)
        if (sent) {
          setPendingFiles([])
          setInput("")
        }
      } finally {
        setIsUploadingDocument(false)
      }
    } else {
      await onSendMessage(question)
      setInput("")
    }
  }

  const handleDocumentChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    e.target.value = ""
    if (!files.length || !onUploadDocuments || isLoading || isUploadingDocument) return
    setPendingFiles((current) => {
      const existing = new Set(current.map((file) => `${file.name}:${file.size}:${file.lastModified}`))
      return [
        ...current,
        ...files.filter((file) => !existing.has(`${file.name}:${file.size}:${file.lastModified}`)),
      ]
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-background">
      {/* Messages Area — native scroll container with constrained height */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto px-8 min-h-0">
        <div className="max-w-3xl mx-auto py-6 flex flex-col gap-6">
          {/* Export button — only show when there are messages */}
          {messages.length > 0 && (
            <div className="flex justify-end">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => {
                  exportChatAsPdf(messages)
                  setShowExportTip(false)
                }}
                onMouseEnter={() => setShowExportTip(true)}
                onMouseLeave={() => setShowExportTip(false)}
                className="h-8 gap-1.5 text-xs text-muted-foreground hover:text-[#3D5A40] hover:bg-[#3D5A40]/5 rounded-lg transition-colors relative"
              >
                <Download className="h-3.5 w-3.5" />
                导出对话
                {showExportTip && (
                  <span className="absolute -bottom-7 left-1/2 -translate-x-1/2 text-[11px] text-muted-foreground whitespace-nowrap bg-white px-2 py-0.5 rounded border shadow-sm">
                    通过浏览器打印 → 另存为 PDF
                  </span>
                )}
              </Button>
            </div>
          )}
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-[60vh] text-center">
              {/* Phoenix Icon for Empty State */}
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#3D5A40] to-[#5A7A5E] mb-4">
                <svg width="36" height="36" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path 
                    d="M16 28C16 28 12 24 12 18C12 14 14 10 16 8" 
                    stroke="#FFCC99" 
                    strokeWidth="2" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path 
                    d="M16 28C16 28 20 24 20 18C20 14 18 10 16 8" 
                    stroke="#FFCC99" 
                    strokeWidth="2" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path 
                    d="M16 12C16 12 10 10 6 12C4 13 3 15 4 17C5 19 8 18 10 16C12 14 14 13 16 14" 
                    stroke="white" 
                    strokeWidth="1.8" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path 
                    d="M16 12C16 12 22 10 26 12C28 13 29 15 28 17C27 19 24 18 22 16C20 14 18 13 16 14" 
                    stroke="white" 
                    strokeWidth="1.8" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path 
                    d="M16 8C16 8 14 5 16 3C18 5 16 8 16 8Z" 
                    fill="#FFCC99"
                    stroke="#FFCC99"
                    strokeWidth="1"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <circle cx="16" cy="14" r="1.5" fill="#FFCC99" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-[#3D5A40] mb-2">高考辅导 AI 助手</h2>
              <p className="text-muted-foreground max-w-md leading-relaxed">
                我是你的高考学习伙伴，可以帮助你解答学科问题、制定学习计划、提供情绪支持。有什么想问的吗？
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))
          )}
          {/* 思考中动画 — 等待 SSE 连接建立时显示（空助手气泡创建后会自动显示气泡内动画） */}
          {isLoading && messages[messages.length - 1]?.role !== "assistant" && (
            <div className="flex items-start gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#3D5A40]/10 text-[#3D5A40] flex-shrink-0">
                <Bot className="h-4 w-4" />
              </div>
              <div className="bg-white border border-[#C8D6C9] rounded-2xl rounded-tl-sm px-5 py-4">
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-2.5 h-2.5 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-2.5 h-2.5 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                  <span className="text-xs text-muted-foreground">正在思考...</span>
                </div>
              </div>
            </div>
          )}
          {/* Scroll anchor */}
          <div ref={messagesEndRef} className="h-4 shrink-0" />
        </div>
      </div>

      {/* Input Area - Gemini Style with new palette */}
      <div className="bg-background px-8 py-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="bg-[#F5F3E8] rounded-3xl overflow-hidden border border-[#E8E5D8]">
            {/* Text Area at Top */}
            <div className="px-4 pt-4 pb-2">
              {pendingFiles.length > 0 && (
                <div className="mb-3 flex flex-wrap gap-2">
                  {pendingFiles.map((file, index) => (
                    <div
                      key={`${file.name}:${file.size}:${file.lastModified}`}
                      className="flex h-8 max-w-full items-center gap-2 rounded-md border border-[#D8D4C5] bg-white px-2 text-xs text-foreground"
                    >
                      <FileText className="h-3.5 w-3.5 shrink-0 text-[#3D5A40]" />
                      <span className="max-w-48 truncate">{file.name}</span>
                      <button
                        type="button"
                        onClick={() => setPendingFiles((files) => files.filter((_, i) => i !== index))}
                        className="text-muted-foreground hover:text-foreground"
                        title={`移除 ${file.name}`}
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={pendingFiles.length ? "针对附件输入你的问题..." : "输入你的问题..."}
                rows={2}
                className={cn(
                  "w-full resize-none bg-transparent",
                  "text-sm text-foreground placeholder:text-muted-foreground",
                  "focus:outline-none",
                  "min-h-[60px] max-h-[200px]"
                )}
              />
            </div>
            
            {/* Toolbar at Bottom */}
            <div className="flex items-center px-3 pb-3 gap-1">
              {/* Left Side: Plus and Tools */}
              <Button
                type="button"
                variant="ghost"
                size="icon"
                disabled={isLoading || isUploadingDocument}
                onClick={() => fileInputRef.current?.click()}
                className="h-9 w-9 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50"
                title="上传 PDF、Word 或试卷图片"
              >
                {isUploadingDocument ? <Loader2 className="h-5 w-5 animate-spin" /> : <FileUp className="h-5 w-5" />}
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,image/png,image/jpeg,image/webp"
                multiple
                className="hidden"
                onChange={handleDocumentChange}
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-9 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50 gap-1.5 px-3"
              >
                <SlidersHorizontal className="h-4 w-4" />
                <span className="text-sm">工具</span>
              </Button>
              
              {/* Flexible Spacer */}
              <div className="flex-1" />
              
              {/* Right Side: Mic, Send */}
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-9 w-9 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50"
              >
                <Mic className="h-5 w-5" />
              </Button>
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || isLoading || isUploadingDocument}
                className={cn(
                  "h-9 w-9 rounded-full",
                  "bg-[#3D5A40] hover:bg-[#4A6B4D] text-white",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}

// Tailwind-styled overrides for react-markdown elements
const markdownComponents: Components = {
  h1: ({ children }) => <h1 className="text-lg font-bold mt-4 mb-2 text-[#3D5A40]">{children}</h1>,
  h2: ({ children }) => <h2 className="text-base font-bold mt-3 mb-1.5 text-[#3D5A40]">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold mt-2 mb-1 text-[#3D5A40]">{children}</h3>,
  p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="list-disc pl-5 mb-2 space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 space-y-1">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-[#3D5A40]">{children}</strong>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-3 border-[#7A9E7E] pl-3 my-2 text-muted-foreground italic">
      {children}
    </blockquote>
  ),
  code: ({ className, children }) => {
    // Fenced code blocks get a className like "language-xxx" from remark
    const isBlock = className?.startsWith("language-")
    if (isBlock) {
      return (
        <code className="block bg-[#F5F3E8] rounded-lg p-3 my-2 text-xs font-mono overflow-x-auto whitespace-pre">
          {children}
        </code>
      )
    }
    // Inline code
    return (
      <code className="bg-[#F5F3E8] rounded px-1.5 py-0.5 text-xs font-mono text-[#5C3D2E]">
        {children}
      </code>
    )
  },
  pre: ({ children }) => <pre className="my-2">{children}</pre>,
  table: ({ children }) => (
    <div className="overflow-x-auto my-2">
      <table className="min-w-full text-xs border-collapse">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-[#C8D6C9] bg-[#F5F3E8] px-2 py-1 text-left font-semibold">{children}</th>
  ),
  td: ({ children }) => (
    <td className="border border-[#C8D6C9] px-2 py-1">{children}</td>
  ),
  hr: () => <hr className="border-[#C8D6C9] my-3" />,
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user"
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null)

  const handleFeedback = (type: "up" | "down") => {
    if (feedback) return // 已经投过票
    setFeedback(type)
    // 异步发送到后端（非阻塞）
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
    fetch(`${apiBase}/feedback`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message_id: message.id, rating: type, query_preview: message.content.slice(0, 200) }),
    }).catch(() => {}) // 静默失败，不打扰用户
  }

  return (
    <div className={cn("flex items-start gap-3", isUser && "flex-row-reverse")}>
      <div className={cn(
        "flex h-8 w-8 items-center justify-center rounded-full flex-shrink-0",
        isUser ? "bg-[#3D5A40] text-white" : "bg-[#3D5A40]/10 text-[#3D5A40]"
      )}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className="flex flex-col gap-1" style={{ maxWidth: "80%" }}>
        <div className={cn(
          "rounded-2xl px-4 py-3 text-sm leading-relaxed overflow-hidden",
          isUser
            ? "bg-[#3D5A40] text-white rounded-tr-sm"
            : "bg-white border border-[#C8D6C9] text-[#2D2D2D] rounded-tl-sm"
        )}>
          {isUser ? (
            <div className="whitespace-pre-wrap">{message.content}</div>
          ) : message.content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {message.content}
            </ReactMarkdown>
          ) : (
            <div className="flex items-center gap-1.5 py-1">
              <span className="w-2 h-2 bg-[#3D5A40]/50 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
              <span className="w-2 h-2 bg-[#3D5A40]/50 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
              <span className="w-2 h-2 bg-[#3D5A40]/50 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              <span className="text-xs text-muted-foreground ml-1">正在思考...</span>
            </div>
          )}
        </div>
        {/* 反馈按钮 — 仅AI回复且内容完整时显示 */}
        {!isUser && message.content && (
          <div className="flex items-center gap-1 px-1">
            {feedback ? (
              <span className="text-[11px] text-muted-foreground">
                {feedback === "up" ? "👍 感谢反馈" : "👎 感谢反馈"}
              </span>
            ) : (
              <>
                <button
                  onClick={() => handleFeedback("up")}
                  className="p-0.5 rounded hover:bg-[#3D5A40]/10 text-muted-foreground hover:text-[#3D5A40] transition-colors"
                  title="回答有帮助"
                >
                  <ThumbsUp className="h-3 w-3" />
                </button>
                <button
                  onClick={() => handleFeedback("down")}
                  className="p-0.5 rounded hover:bg-[#D97B6C]/10 text-muted-foreground hover:text-[#D97B6C] transition-colors"
                  title="回答没有帮助"
                >
                  <ThumbsDown className="h-3 w-3" />
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
