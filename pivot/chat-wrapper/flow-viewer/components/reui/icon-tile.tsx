import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

export const iconTileVariants = cva(
  "reui-icon-tile inline-flex items-center justify-center shrink-0 transition-all",
  {
    variants: {
      variant: {
        outline: "bg-transparent border border-border text-muted-foreground",
        elevated: "bg-muted border border-border text-foreground shadow-sm",
        soft: "bg-amber-500/10 border border-amber-500/30 text-amber-500",
        solid: "bg-primary text-primary-foreground",
        frame: "bg-card border border-border text-foreground shadow-inner",
      },
      size: {
        xs: "h-6 w-6 [&_svg]:h-3.5 [&_svg]:w-3.5",
        sm: "h-8 w-8 [&_svg]:h-4 [&_svg]:w-4",
        default: "h-10 w-10 [&_svg]:h-4.5 [&_svg]:w-4.5",
        lg: "h-12 w-12 [&_svg]:h-5.5 [&_svg]:w-5.5",
        xl: "h-14 w-14 [&_svg]:h-6.5 [&_svg]:w-6.5",
      },
      radius: {
        default: "rounded-md",
        full: "rounded-full",
      },
    },
    defaultVariants: {
      variant: "outline",
      size: "default",
      radius: "default",
    },
  }
)

export interface IconTileProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof iconTileVariants> {}

export const IconTile = React.forwardRef<HTMLSpanElement, IconTileProps>(
  ({ className = "", variant, size, radius, children, ...props }, ref) => (
    <span
      ref={ref}
      data-slot="icon-tile"
      data-variant={variant}
      data-size={size}
      data-radius={radius}
      className={`${iconTileVariants({ variant, size, radius })} ${className}`}
      {...props}
    >
      {children}
    </span>
  )
)
IconTile.displayName = "IconTile"
