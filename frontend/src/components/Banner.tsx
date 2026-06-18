import type { ReactNode } from 'react'

type BannerVariant = 'info' | 'warning' | 'error'

interface BannerProps {
  variant: BannerVariant
  children: ReactNode
  action?: ReactNode
}

const VARIANT_CLASSES: Record<BannerVariant, string> = {
  info: 'border-indigo-500/40 bg-indigo-500/10 text-indigo-200',
  warning: 'border-amber-500/40 bg-amber-500/10 text-amber-200',
  error: 'border-rose-500/40 bg-rose-500/10 text-rose-200',
}

export function Banner({ variant, children, action }: BannerProps) {
  return (
    <div
      role={variant === 'error' ? 'alert' : 'status'}
      className={`flex items-start justify-between gap-3 rounded-lg border px-4 py-3 text-sm ${VARIANT_CLASSES[variant]}`}
    >
      <span className="leading-relaxed">{children}</span>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  )
}
