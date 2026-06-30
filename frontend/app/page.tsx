"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { LeftSidebar } from "@/components/left-sidebar"
import { RightPanel, NodeEvent, LogEntry } from "@/components/right-panel"
import { ChatArea, Message } from "@/components/chat-area"
import { PlanReview } from "@/components/plan-review"
import { LoginScreen } from "@/components/login-screen"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
const CHAT_STORAGE_KEY = "gaokao_tutor_conversations"

interface ChatSession {
  id: string
  title: string
  threadId: string | null
  messages: Message[]
  updatedAt: string
}

const initialChatHistory: ChatSession[] = []

type SSEEvent =
  | { type: "thread_id"; thread_id: string }
  | { type: "interrupt"; draft: string; thread_id?: string }
  | { type: "token"; content: string }
  | { type: "text"; content: string; node?: string }
  | { type: "done" }
  | { type: "error"; message: string }
  | {
      type: "node_event"
      node: string
      status: "start" | "end"
      duration_ms?: number | null
      error?: string | null
    }
  | {
      type: "usage"
      node: string
      input_tokens?: number
      output_tokens?: number
      total_tokens?: number
    }

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function timestamp(): string {
  return new Date().toLocaleTimeString("en-GB", { hour12: false })
}

export default function Home() {
  const [authState, setAuthState] = useState<"checking" | "authenticated" | "anonymous">("checking")
  const [chatHistory, setChatHistory] = useState(initialChatHistory)
  const [selectedChatId, setSelectedChatId] = useState<string | undefined>()
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [logs, setLogs] = useState<LogEntry[]>([
    { type: "info", message: "[INFO] System initialized.", ts: timestamp() },
  ])
  const [nodeEvents, setNodeEvents] = useState<NodeEvent[]>([])
  const [tokenUsage, setTokenUsage] = useState({ input: 0, output: 0, total: 0 })

  // HIL state
  const [isInterrupted, setIsInterrupted] = useState(false)
  const [interruptDraft, setInterruptDraft] = useState("")
  const [isResuming, setIsResuming] = useState(false)
  const [chatStorageLoaded, setChatStorageLoaded] = useState(false)
  const threadIdRef = useRef<string | null>(null)
  const userIdRef = useRef<string | null>(null)
  const activeChatIdRef = useRef<string | null>(null)
  const assistantMessageIdRef = useRef<string>("")

  useEffect(() => {
    fetch(`${API_BASE_URL}/auth/me`, { credentials: "include" })
      .then((response) => setAuthState(response.ok ? "authenticated" : "anonymous"))
      .catch(() => setAuthState("anonymous"))
  }, [])

  useEffect(() => {
    try {
      const stored = localStorage.getItem(CHAT_STORAGE_KEY)
      if (stored) {
        const sessions = JSON.parse(stored) as ChatSession[]
        if (Array.isArray(sessions)) {
          setChatHistory(sessions)
        }
      }
    } catch {
      localStorage.removeItem(CHAT_STORAGE_KEY)
    } finally {
      setChatStorageLoaded(true)
    }
  }, [])

  useEffect(() => {
    if (!chatStorageLoaded) return
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chatHistory))
  }, [chatHistory, chatStorageLoaded])

  useEffect(() => {
    const chatId = activeChatIdRef.current
    if (!chatId) return
    setChatHistory((prev) =>
      prev.map((chat) =>
        chat.id === chatId
          ? { ...chat, messages, updatedAt: new Date().toISOString() }
          : chat
      )
    )
  }, [messages])

  const handleNewChat = useCallback(() => {
    setSelectedChatId(undefined)
    activeChatIdRef.current = null
    setMessages([])
    setNodeEvents([])
    setLogs([{ type: "info", message: "[INFO] New chat session started.", ts: timestamp() }])
    setTokenUsage({ input: 0, output: 0, total: 0 })
    setIsInterrupted(false)
    setInterruptDraft("")
    threadIdRef.current = null
  }, [])

  const handleSelectChat = useCallback((id: string) => {
    const chat = chatHistory.find((item) => item.id === id)
    if (!chat) return

    setSelectedChatId(id)
    activeChatIdRef.current = id
    setMessages(chat.messages)
    setNodeEvents([])
    setIsInterrupted(false)
    setInterruptDraft("")
    threadIdRef.current = chat.threadId
  }, [chatHistory])

  /** Process a single SSE data payload — shared between /stream and /resume */
  const processSSEEvent = useCallback((data: SSEEvent) => {
    const asstId = assistantMessageIdRef.current

    if (data.type === "thread_id") {
      threadIdRef.current = data.thread_id
      const chatId = activeChatIdRef.current
      if (chatId) {
        setChatHistory((prev) =>
          prev.map((chat) =>
            chat.id === chatId
              ? { ...chat, threadId: data.thread_id, updatedAt: new Date().toISOString() }
              : chat
          )
        )
      }
      setLogs((prev) => [
        ...prev,
        { type: "info", message: `[INFO] Thread: ${data.thread_id.slice(0, 8)}...`, ts: timestamp() },
      ])
      return
    }

    if (data.type === "interrupt") {
      setInterruptDraft(data.draft)
      setIsInterrupted(true)
      if (data.thread_id) threadIdRef.current = data.thread_id
      setLogs((prev) => [
        ...prev,
        { type: "warning", message: "[HIL] Graph interrupted — awaiting user plan review", ts: timestamp() },
      ])
      return
    }

    if (data.type === "token") {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === asstId
            ? { ...msg, content: msg.content + data.content }
            : msg
        )
      )
      return
    }

    if (data.type === "text") {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === asstId ? { ...msg, content: data.content } : msg
        )
      )
      return
    }

    if (data.type === "done") {
      return
    }

    if (data.type === "error") {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] Server: ${data.message}`, ts: timestamp() },
      ])
      return
    }

    if (data.type === "node_event") {
      const node: string = data.node
      const status: "start" | "end" = data.status
      const now = timestamp()

      setNodeEvents((prev) => {
        if (status === "start") {
          return [...prev, { node, status: "running", ts: now }]
        }
        return prev.map((e) =>
          e.node === node && e.status === "running"
            ? { ...e, status: "done", endTs: now, durationMs: data.duration_ms ?? undefined }
            : e
        )
      })

      const label = status === "start" ? "Entering" : "Leaving"
      setLogs((prev) => [
        ...prev,
        { type: "info", message: `[INFO] ${label} node: ${node}`, ts: now },
      ])

      if (status === "end" && data.duration_ms != null) {
        setLogs((prev) => [
          ...prev,
          { type: "perf", message: `[PERF] Node "${node}" completed in ${data.duration_ms}ms`, ts: now },
        ])
      }

      if (status === "end" && data.error) {
        setLogs((prev) => [
          ...prev,
          { type: "error", message: `[ERROR] Node "${node}": ${data.error}`, ts: now },
        ])
      }
      return
    }

    if (data.type === "usage") {
      const now = timestamp()
      setTokenUsage((prev) => ({
        input: prev.input + (data.input_tokens ?? 0),
        output: prev.output + (data.output_tokens ?? 0),
        total: prev.total + (data.total_tokens ?? 0),
      }))
      setLogs((prev) => [
        ...prev,
        { type: "usage", message: `[USAGE] ${data.node}: ${data.input_tokens} in / ${data.output_tokens} out`, ts: now },
      ])
    }
  }, [])

  /** Read an SSE response body and dispatch events via processSSEEvent */
  const consumeSSEStream = useCallback(async (body: ReadableStream<Uint8Array>) => {
    const reader = body.getReader()
    const decoder = new TextDecoder()
    let buffer = ""

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split("\n\n")
      buffer = parts.pop() || ""

      for (const part of parts) {
        if (part.startsWith("data: ")) {
          try {
            const data = JSON.parse(part.slice(6))
            processSSEEvent(data)
          } catch {
            // Ignore partial or malformed JSON chunks
          }
        }
      }
    }
  }, [processSSEEvent])

  /** Fetch helper with shared HTTP error handling. Returns response body or null on handled error. */
  const fetchWithErrorHandling = useCallback(async (url: string, init: RequestInit): Promise<ReadableStream<Uint8Array> | null> => {
    const response = await fetch(url, { ...init, credentials: "include" })

    if (response.status === 429) {
      setMessages((prev) => [
        ...prev,
        { id: (Date.now() + 1).toString(), role: "assistant", content: "⚠️ 服务繁忙，请稍后重试。" },
      ])
      setLogs((prev) => [
        ...prev,
        { type: "warning", message: "[WARN] 429 Too Many Requests", ts: timestamp() },
      ])
      return null
    }

    if (response.status === 401) {
      setAuthState("anonymous")
      return null
    }

    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    if (!response.body) throw new Error("No response body")

    return response.body
  }, [])

  const handleSendMessage = useCallback(async (content: string, displayContent = content) => {
    let chatId = activeChatIdRef.current
    if (!chatId) {
      chatId = crypto.randomUUID()
      const newChat: ChatSession = {
        id: chatId,
        title: displayContent.slice(0, 30) + (displayContent.length > 30 ? "..." : ""),
        threadId: null,
        messages: [],
        updatedAt: new Date().toISOString(),
      }
      activeChatIdRef.current = chatId
      setSelectedChatId(chatId)
      setChatHistory((prev) => [newChat, ...prev])
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: displayContent,
    }

    setMessages((prev) => [...prev, userMessage])
    setNodeEvents([])
    setTokenUsage({ input: 0, output: 0, total: 0 })
    setIsInterrupted(false)
    setInterruptDraft("")
    setLogs((prev) => [
      ...prev,
      { type: "info" as const, message: `[INFO] User query: ${displayContent.slice(0, 60)}`, ts: timestamp() },
    ])

    setIsLoading(true)

    try {
      if (!userIdRef.current && typeof window !== "undefined") {
        const storedUserId = localStorage.getItem("gaokao_tutor_user_id")
        const userId = storedUserId || crypto.randomUUID()
        if (!storedUserId) localStorage.setItem("gaokao_tutor_user_id", userId)
        userIdRef.current = userId
      }
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: content,
          thread_id: threadIdRef.current,
          user_id: userIdRef.current,
        }),
      })

      if (!body) return

      // Create an empty assistant message placeholder
      const assistantMessageId = (Date.now() + 1).toString()
      assistantMessageIdRef.current = assistantMessageId
      setMessages((prev) => [
        ...prev,
        { id: assistantMessageId, role: "assistant", content: "" },
      ])

      await consumeSSEStream(body)

      setLogs((prev) => [
        ...prev,
        { type: "info", message: "[INFO] Stream complete.", ts: timestamp() },
      ])
    } catch (error: unknown) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] ${errorMessage(error)}`, ts: timestamp() },
      ])
    } finally {
      setIsLoading(false)
    }
  }, [fetchWithErrorHandling, consumeSSEStream])

  const handleUploadExamDocuments = useCallback(async (files: File[], question: string) => {
    const filenames = files.map((file) => file.name).join(", ")
    setLogs((prev) => [
      ...prev,
      { type: "info", message: `[INFO] Document upload: ${filenames}`, ts: timestamp() },
    ])

    const formData = new FormData()
    files.forEach((file) => formData.append("files", file))
    formData.append("question", question)

    try {
      const response = await fetch(`${API_BASE_URL}/documents/parse`, {
        method: "POST",
        credentials: "include",
        body: formData,
      })

      if (!response.ok) {
        if (response.status === 401) {
          setAuthState("anonymous")
          return false
        }
        let detail = `${response.status} ${response.statusText}`
        try {
          const payload = await response.json()
          detail = payload.detail || detail
        } catch {
          // Keep HTTP status text when the server does not return JSON.
        }
        throw new Error(detail)
      }

      const data = await response.json()
      setLogs((prev) => [
        ...prev,
        {
          type: "info",
          message: `[INFO] Parsed ${data.questions.length} questions via ${data.parser}${data.segmenter_used ? " + question_segmenter" : ""}.`,
          ts: timestamp(),
        },
      ])
      const attachmentLabel = files.length === 1
        ? `附件：${files[0].name}`
        : `附件：${files.length} 个文件（${filenames}）`
      await handleSendMessage(data.query, `${question}\n\n${attachmentLabel}`)
      return true
    } catch (error: unknown) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] Document parsing failed: ${errorMessage(error)}`, ts: timestamp() },
      ])
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString(), role: "assistant", content: `文档解析失败：${errorMessage(error)}` },
      ])
      return false
    }
  }, [handleSendMessage])

  const handleResume = useCallback(async (editedPlan: string) => {
    const threadId = threadIdRef.current
    if (!threadId) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] No thread_id — cannot resume", ts: timestamp() },
      ])
      return
    }

    setIsResuming(true)
    setLogs((prev) => [
      ...prev,
      { type: "info", message: "[INFO] Resuming graph with edited plan...", ts: timestamp() },
    ])

    try {
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: threadId, edited_plan: editedPlan }),
      })

      if (!body) return

      setIsInterrupted(false)
      setInterruptDraft("")

      await consumeSSEStream(body)

      setLogs((prev) => [
        ...prev,
        { type: "info", message: "[INFO] Resume stream complete.", ts: timestamp() },
      ])
    } catch (error: unknown) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] Resume failed: ${errorMessage(error)}`, ts: timestamp() },
      ])
    } finally {
      setIsResuming(false)
      setIsLoading(false)
    }
  }, [fetchWithErrorHandling, consumeSSEStream])

  const handleFeedback = useCallback(async (feedback: string) => {
    const threadId = threadIdRef.current
    if (!threadId) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] No thread_id — cannot send feedback", ts: timestamp() },
      ])
      return
    }

    setIsResuming(true)
    setLogs((prev) => [
      ...prev,
      { type: "info", message: `[INFO] Sending feedback: ${feedback.slice(0, 40)}...`, ts: timestamp() },
    ])

    try {
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: threadId, feedback }),
      })

      if (!body) return

      // Hide PlanReview while system processes feedback
      setIsInterrupted(false)
      setInterruptDraft("")

      // Create new assistant message placeholder for the revised plan streaming
      const newAsstId = (Date.now() + 1).toString()
      assistantMessageIdRef.current = newAsstId
      setMessages((prev) => [
        ...prev,
        { id: newAsstId, role: "assistant", content: "" },
      ])

      await consumeSSEStream(body)

      setLogs((prev) => [
        ...prev,
        { type: "info", message: "[INFO] Feedback revision complete.", ts: timestamp() },
      ])
    } catch (error: unknown) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] Feedback failed: ${errorMessage(error)}`, ts: timestamp() },
      ])
    } finally {
      setIsResuming(false)
      setIsLoading(false)
    }
  }, [fetchWithErrorHandling, consumeSSEStream])

  const handleLogin = useCallback(async (username: string, password: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      })
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}))
        return payload.detail || "登录失败，请检查账号和密码。"
      }
      setAuthState("authenticated")
      return null
    } catch {
      return "无法连接后端服务。"
    }
  }, [])

  const handleLogout = useCallback(async () => {
    await fetch(`${API_BASE_URL}/auth/logout`, {
      method: "POST",
      credentials: "include",
    }).catch(() => undefined)
    setAuthState("anonymous")
  }, [])

  if (authState === "checking") {
    return <div className="flex h-screen items-center justify-center text-sm text-muted-foreground">正在验证登录状态...</div>
  }

  if (authState === "anonymous") {
    return <LoginScreen onLogin={handleLogin} />
  }

  return (
    <div className="flex h-screen overflow-hidden">
        <LeftSidebar
        chatHistory={chatHistory}
        onNewChat={handleNewChat}
          onSelectChat={handleSelectChat}
          onLogout={handleLogout}
        selectedChatId={selectedChatId}
      />
      <div className="flex-1 flex flex-col h-full">
        <ChatArea
          messages={messages}
          onSendMessage={handleSendMessage}
          onUploadDocuments={handleUploadExamDocuments}
          isLoading={isLoading && !isInterrupted}
        />
        {isInterrupted && (
          <PlanReview
            draft={interruptDraft}
            onConfirm={handleResume}
            onFeedback={handleFeedback}
            isSubmitting={isResuming}
          />
        )}
      </div>
      <RightPanel
        logs={logs}
        nodeEvents={nodeEvents}
        tokenUsage={tokenUsage}
        isInterrupted={isInterrupted}
      />
    </div>
  )
}
