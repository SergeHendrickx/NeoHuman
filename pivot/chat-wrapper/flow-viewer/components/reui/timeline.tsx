import * as React from "react"

export interface TimelineProps extends React.HTMLAttributes<HTMLDivElement> {
  defaultValue?: number
  value?: number
  orientation?: "horizontal" | "vertical"
}

export const Timeline = React.forwardRef<HTMLDivElement, TimelineProps>(
  ({ defaultValue = 1, value, orientation = "vertical", className = "", children, ...props }, ref) => (
    <div
      ref={ref}
      data-orientation={orientation}
      className={`reui-timeline flex flex-col gap-0 ${className}`}
      {...props}
    >
      {children}
    </div>
  )
)
Timeline.displayName = "Timeline"

export interface TimelineItemProps extends React.HTMLAttributes<HTMLDivElement> {
  step: number
  active?: boolean
  completed?: boolean
}

export const TimelineItem = React.forwardRef<HTMLDivElement, TimelineItemProps>(
  ({ step, active, completed, className = "", children, ...props }, ref) => (
    <div
      ref={ref}
      data-step={step}
      className={`reui-timeline-item grid grid-cols-[24px_1fr] gap-3 pb-5 ${
        active ? "active" : completed ? "completed" : ""
      } ${className}`}
      {...props}
    >
      {children}
    </div>
  )
)
TimelineItem.displayName = "TimelineItem"

export const TimelineIndicator = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className = "", ...props }, ref) => (
  <div
    ref={ref}
    className={`reui-timeline-indicator h-3.5 w-3.5 rounded-full border-2 border-border bg-card transition-transform ${className}`}
    {...props}
  />
))
TimelineIndicator.displayName = "TimelineIndicator"

export const TimelineSeparator = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className = "", ...props }, ref) => (
  <div
    ref={ref}
    className={`reui-timeline-separator w-0.5 flex-1 bg-border my-1 ${className}`}
    {...props}
  />
))
TimelineSeparator.displayName = "TimelineSeparator"

export const TimelineHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className = "", ...props }, ref) => (
  <div
    ref={ref}
    className={`reui-timeline-header flex items-baseline justify-between gap-2 ${className}`}
    {...props}
  />
))
TimelineHeader.displayName = "TimelineHeader"

export const TimelineTitle = React.forwardRef<
  HTMLSpanElement,
  React.HTMLAttributes<HTMLSpanElement>
>(({ className = "", ...props }, ref) => (
  <span
    ref={ref}
    className={`reui-timeline-title text-xs font-semibold text-foreground flex items-center gap-2 ${className}`}
    {...props}
  />
))
TimelineTitle.displayName = "TimelineTitle"

export const TimelineDate = React.forwardRef<
  HTMLSpanElement,
  React.HTMLAttributes<HTMLSpanElement>
>(({ className = "", ...props }, ref) => (
  <span
    ref={ref}
    className={`reui-timeline-date text-[10.5px] font-mono text-muted-foreground ${className}`}
    {...props}
  />
))
TimelineDate.displayName = "TimelineDate"

export const TimelineContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className = "", ...props }, ref) => (
  <div
    ref={ref}
    className={`reui-timeline-content text-xs text-muted-foreground leading-relaxed ${className}`}
    {...props}
  />
))
TimelineContent.displayName = "TimelineContent"
