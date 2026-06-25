"use client"

import { useState } from "react"
import { BookOpen, Loader2, LockKeyhole, UserRound } from "lucide-react"
import { Button } from "@/components/ui/button"

interface LoginScreenProps {
  onLogin: (username: string, password: string) => Promise<string | null>
}

export function LoginScreen({ onLogin }: LoginScreenProps) {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!username.trim() || !password) return
    setIsSubmitting(true)
    setError("")
    try {
      const message = await onLogin(username.trim(), password)
      if (message) setError(message)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-[#FFFDF0] px-5">
      <section className="w-full max-w-sm">
        <div className="mb-8 flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-md bg-[#3D5A40] text-white">
            <BookOpen className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-[#2D2D2D]">高考辅导 AI 助手</h1>
            <p className="text-sm text-muted-foreground">登录后进入学习空间</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="border-t border-[#D8D4C5] pt-6">
          <label className="mb-2 block text-sm font-medium text-foreground" htmlFor="username">
            账号
          </label>
          <div className="mb-4 flex h-11 items-center border border-[#D8D4C5] bg-white px-3 focus-within:border-[#3D5A40]">
            <UserRound className="mr-2 h-4 w-4 text-muted-foreground" />
            <input
              id="username"
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              className="h-full min-w-0 flex-1 bg-transparent text-sm outline-none"
              placeholder="请输入账号"
            />
          </div>

          <label className="mb-2 block text-sm font-medium text-foreground" htmlFor="password">
            密码
          </label>
          <div className="flex h-11 items-center border border-[#D8D4C5] bg-white px-3 focus-within:border-[#3D5A40]">
            <LockKeyhole className="mr-2 h-4 w-4 text-muted-foreground" />
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="h-full min-w-0 flex-1 bg-transparent text-sm outline-none"
              placeholder="请输入密码"
            />
          </div>

          <div className="mt-2 min-h-5 text-sm text-[#B94A3C]" role="alert">
            {error}
          </div>

          <Button
            type="submit"
            disabled={isSubmitting || !username.trim() || !password}
            className="mt-3 h-11 w-full rounded-md bg-[#3D5A40] text-white hover:bg-[#4A6B4D]"
          >
            {isSubmitting && <Loader2 className="h-4 w-4 animate-spin" />}
            登录
          </Button>
        </form>
      </section>
    </main>
  )
}
