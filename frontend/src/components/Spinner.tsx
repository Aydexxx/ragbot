interface SpinnerProps {
  size?: 'sm' | 'md'
  className?: string
}

const SIZE_CLASSES: Record<NonNullable<SpinnerProps['size']>, string> = {
  sm: 'h-3.5 w-3.5 border-2',
  md: 'h-5 w-5 border-2',
}

export function Spinner({ size = 'md', className = '' }: SpinnerProps) {
  return (
    <span
      role="status"
      aria-label="Loading"
      className={`inline-block animate-spin rounded-full border-slate-600 border-t-indigo-400 ${SIZE_CLASSES[size]} ${className}`}
    />
  )
}
