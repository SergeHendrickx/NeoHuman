import * as React from "react"

export interface CodeBlockProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string
  language?: string
  code: string
  copyable?: boolean
}

export const CodeBlock = React.forwardRef<HTMLDivElement, CodeBlockProps>(
  ({ title, language = "json", code, copyable = true, className = "", ...props }, ref) => {
    const [copied, setCopied] = React.useState(false)

    const handleCopy = () => {
      navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }

    return (
      <div
        ref={ref}
        className={`reui-code-block overflow-hidden rounded-md border border-border bg-[#06090e] font-mono text-xs ${className}`}
        {...props}
      >
        {(title || copyable) && (
          <div className="reui-code-header flex items-center justify-between border-b border-border bg-muted/60 px-3 py-1.5 text-[11px] text-muted-foreground">
            <span className="reui-code-title font-medium text-foreground">{title || language}</span>
            {copyable && (
              <button
                type="button"
                onClick={handleCopy}
                className="reui-copy-btn rounded border border-border px-2 py-0.5 text-[10px] text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                {copied ? "Copied!" : "Copy"}
              </button>
            )}
          </div>
        )}
        <pre className="reui-code-body max-h-72 overflow-x-auto p-3 text-[11px] leading-relaxed text-[#c9d1d9] whitespace-pre-wrap break-all">
          {code}
        </pre>
      </div>
    )
  }
)
CodeBlock.displayName = "CodeBlock"
