"use client"

import * as React from "react"

type ToastProps = React.HTMLAttributes<HTMLDivElement> & {
  open?: boolean
  onOpenChange?: (open: boolean) => void
}

type ToastActionElement = React.ReactElement

export type { ToastActionElement, ToastProps }
