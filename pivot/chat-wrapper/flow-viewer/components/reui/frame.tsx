import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

const frameVariants = cva(
  "reui-frame relative w-full overflow-hidden border bg-background text-foreground transition-colors",
  {
    variants: {
      variant: {
        default: "border-border bg-card",
        ghost: "border-transparent bg-transparent",
      },
      spacing: {
        sm: "p-3 gap-2",
        default: "p-4 gap-3",
        lg: "p-6 gap-4",
      },
      stacked: {
        true: "divide-y divide-border",
        false: "",
      },
      dense: {
        true: "p-0",
        false: "",
      },
    },
    defaultVariants: {
      variant: "default",
      spacing: "default",
      stacked: false,
      dense: false,
    },
  }
)

export interface FrameProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof frameVariants> {}

export const Frame = React.forwardRef<HTMLDivElement, FrameProps>(
  ({ className = "", variant, spacing, stacked, dense, ...props }, ref) => (
    <div
      ref={ref}
      className={`${frameVariants({ variant, spacing, stacked, dense })} ${className}`}
      style={{ borderRadius: "var(--frame-radius, 12px)" }}
      {...props}
    />
  )
)
Frame.displayName = "Frame"

export const FramePanel = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className = "", ...props }, ref) => (
  <div
    ref={ref}
    className={`reui-frame-panel flex flex-col gap-3 rounded-[calc(var(--frame-radius,12px)-2px)] p-4 ${className}`}
    {...props}
  />
))
FramePanel.displayName = "FramePanel"

export const FrameHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className = "", ...props }, ref) => (
  <div
    ref={ref}
    className={`reui-frame-header flex items-start justify-between gap-4 border-b pb-3 ${className}`}
    {...props}
  />
))
FrameHeader.displayName = "FrameHeader"

export const FrameTitle = React.forwardRef<
  HTMLHeadingElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className = "", ...props }, ref) => (
  <h3
    ref={ref}
    className={`reui-frame-title text-sm font-semibold tracking-tight text-foreground flex items-center gap-2 ${className}`}
    {...props}
  />
))
FrameTitle.displayName = "FrameTitle"

export const FrameDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className = "", ...props }, ref) => (
  <p
    ref={ref}
    className={`reui-frame-description text-xs text-muted-foreground ${className}`}
    {...props}
  />
))
FrameDescription.displayName = "FrameDescription"

export const FrameFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className = "", ...props }, ref) => (
  <div
    ref={ref}
    className={`reui-frame-footer border-t pt-3 text-xs text-muted-foreground flex items-center justify-between ${className}`}
    {...props}
  />
))
FrameFooter.displayName = "FrameFooter"
