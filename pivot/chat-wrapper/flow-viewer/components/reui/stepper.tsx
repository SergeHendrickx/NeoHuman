import * as React from "react"

interface StepperContextType {
  currentStep: number
  setCurrentStep: (step: number) => void
  orientation?: "horizontal" | "vertical"
}

const StepperContext = React.createContext<StepperContextType | undefined>(undefined)

export function useStepper() {
  const context = React.useContext(StepperContext)
  if (!context) {
    throw new Error("useStepper must be used within a Stepper")
  }
  return context
}

export interface StepperProps extends React.HTMLAttributes<HTMLDivElement> {
  defaultValue?: number
  value?: number
  onValueChange?: (value: number) => void
  orientation?: "horizontal" | "vertical"
}

export const Stepper = React.forwardRef<HTMLDivElement, StepperProps>(
  ({ defaultValue = 1, value, onValueChange, orientation = "horizontal", children, className = "", ...props }, ref) => {
    const [step, setStep] = React.useState<number>(defaultValue)
    const isControlled = value !== undefined
    const currentStep = isControlled ? value : step

    const handleStepChange = React.useCallback(
      (nextStep: number) => {
        if (!isControlled) {
          setStep(nextStep)
        }
        onValueChange?.(nextStep)
      },
      [isControlled, onValueChange]
    )

    return (
      <StepperContext.Provider value={{ currentStep, setCurrentStep: handleStepChange, orientation }}>
        <div
          ref={ref}
          className={`reui-stepper flex ${orientation === "vertical" ? "flex-col" : "w-full"} gap-3 ${className}`}
          {...props}
        >
          {children}
        </div>
      </StepperContext.Provider>
    )
  }
)
Stepper.displayName = "Stepper"

export const StepperNav = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className = "", ...props }, ref) => (
  <div
    ref={ref}
    className={`reui-stepper-nav flex items-center gap-2 overflow-x-auto pb-1 ${className}`}
    {...props}
  />
))
StepperNav.displayName = "StepperNav"

export interface StepperItemProps extends React.HTMLAttributes<HTMLDivElement> {
  step: number
  completed?: boolean
  disabled?: boolean
  loading?: boolean
}

export const StepperItem = React.forwardRef<HTMLDivElement, StepperItemProps>(
  ({ step, completed, disabled, loading, className = "", children, ...props }, ref) => {
    const { currentStep, setCurrentStep } = useStepper()
    const isCompleted = completed ?? currentStep > step
    const isActive = currentStep === step

    return (
      <div
        ref={ref}
        data-step={step}
        data-state={isActive ? "active" : isCompleted ? "completed" : "inactive"}
        className={`reui-stepper-item flex items-center gap-2 rounded-md p-2 transition-colors cursor-pointer ${
          isActive ? "bg-muted font-medium" : "hover:bg-muted/50 text-muted-foreground"
        } ${disabled ? "opacity-50 pointer-events-none" : ""} ${className}`}
        onClick={() => !disabled && setCurrentStep(step)}
        {...props}
      >
        {children}
      </div>
    )
  }
)
StepperItem.displayName = "StepperItem"

export const StepperIndicator = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className = "", children, ...props }, ref) => (
  <div
    ref={ref}
    className={`reui-stepper-indicator flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-xs font-semibold ${className}`}
    {...props}
  >
    {children}
  </div>
))
StepperIndicator.displayName = "StepperIndicator"

export const StepperSeparator = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className = "", ...props }, ref) => (
  <div
    ref={ref}
    className={`reui-stepper-separator h-px w-4 bg-border shrink-0 ${className}`}
    {...props}
  />
))
StepperSeparator.displayName = "StepperSeparator"

export const StepperTitle = React.forwardRef<
  HTMLSpanElement,
  React.HTMLAttributes<HTMLSpanElement>
>(({ className = "", ...props }, ref) => (
  <span
    ref={ref}
    className={`reui-stepper-title text-xs font-medium text-foreground whitespace-nowrap ${className}`}
    {...props}
  />
))
StepperTitle.displayName = "StepperTitle"

export const StepperDescription = React.forwardRef<
  HTMLSpanElement,
  React.HTMLAttributes<HTMLSpanElement>
>(({ className = "", ...props }, ref) => (
  <span
    ref={ref}
    className={`reui-stepper-desc text-[10px] text-muted-foreground font-mono whitespace-nowrap ${className}`}
    {...props}
  />
))
StepperDescription.displayName = "StepperDescription"
