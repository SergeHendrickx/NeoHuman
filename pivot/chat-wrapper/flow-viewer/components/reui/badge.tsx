import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

export const badgeVariants = cva(
  "reui-badge inline-flex items-center gap-1.5 font-mono font-medium leading-none whitespace-nowrap select-none transition-all",
  {
    variants: {
      variant: {
        default: "bg-muted text-foreground border border-border",
        outline: "bg-transparent text-muted-foreground border border-border",
        "primary-light": "bg-amber-500/10 text-amber-500 border border-amber-500/30",
        "success-light": "bg-emerald-500/10 text-emerald-500 border border-emerald-500/30",
        "info-light": "bg-blue-500/10 text-blue-500 border border-blue-500/30",
        "warning-light": "bg-amber-500/10 text-amber-500 border border-amber-500/30",
        "destructive-light": "bg-red-500/10 text-red-500 border border-red-500/30",
        "purple-light": "bg-purple-500/10 text-purple-500 border border-purple-500/30",
        "cyan-light": "bg-cyan-500/10 text-cyan-500 border border-cyan-500/30",
      },
      size: {
        default: "text-[11px] px-2 py-0.5",
        xs: "text-[9.5px] px-1.5 py-0.5",
        sm: "text-[10.5px] px-2 py-0.5",
        lg: "text-xs px-2.5 py-1",
      },
      radius: {
        default: "rounded-sm",
        full: "rounded-full px-2.5",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
      radius: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className = "", variant, size, radius, ...props }, ref) => (
    <span
      ref={ref}
      data-variant={variant}
      data-size={size}
      data-radius={radius}
      className={`${badgeVariants({ variant, size, radius })} ${className}`}
      {...props}
    />
  )
)
Badge.displayName = "Badge"
