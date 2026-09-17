/*
 * The presentational pieces every page shares.
 *
 * These exist because the stylesheet expects certain classes to be
 * added at certain times — `.reveal.in` once a card is on screen, a
 * width on `.bar i` once it is — and doing that inline in each page
 * would spread the same four effects across seven files.
 *
 * Every animation here checks `prefers-reduced-motion` and settles
 * straight to its final state when it is set.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react'

export const reduceMotion = window.matchMedia(
  '(prefers-reduced-motion: reduce)'
).matches

/* ---- toast ---------------------------------------------------- */

const ToastContext = createContext(() => {})

/** `const toast = useToast()` — call it with a line of text. */
export function useToast() {
  return useContext(ToastContext)
}

export function ToastHost({ children }) {
  const [message, setMessage] = useState('')
  const timer = useRef()

  const toast = useCallback((text) => {
    setMessage(text)
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setMessage(''), 2400)
  }, [])

  useEffect(() => () => clearTimeout(timer.current), [])

  return (
    <ToastContext.Provider value={toast}>
      {children}
      <div
        className={'toast' + (message ? ' show' : '')}
        role="status"
        aria-live="polite"
      >
        {message}
      </div>
    </ToastContext.Provider>
  )
}

/* ---- reveal on scroll ----------------------------------------- */

/**
 * A card that fades in when it scrolls into view.
 *
 * A page that is not the active one never renders, so by the time this
 * mounts it is either on screen already or one scroll away — the
 * observer covers both.
 */
export function Card({ className = '', children, ...rest }) {
  const ref = useRef(null)
  const [shown, setShown] = useState(reduceMotion)

  useEffect(() => {
    if (reduceMotion) return

    const node = ref.current
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return
          setShown(true)
          observer.unobserve(entry.target)
        })
      },
      { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
    )

    observer.observe(node)

    return () => observer.disconnect()
  }, [])

  return (
    <div
      ref={ref}
      className={`card reveal ${shown ? 'in ' : ''}${className}`.trim()}
      {...rest}
    >
      {children}
    </div>
  )
}

/* ---- bars and counters ---------------------------------------- */

/**
 * A progress bar that grows to `value` (0..100) once it is mounted.
 *
 * The width starts at 0 and is set on the next frame, because that is
 * what makes the CSS transition run rather than snap.
 */
export function Bar({ value, delay = 180 }) {
  const [width, setWidth] = useState(reduceMotion ? `${value}%` : '0%')

  useEffect(() => {
    if (reduceMotion) {
      setWidth(`${value}%`)
      return
    }

    const timer = setTimeout(() => setWidth(`${value}%`), delay)

    return () => clearTimeout(timer)
  }, [value, delay])

  return (
    <div className="bar">
      <i style={{ width }} />
    </div>
  )
}

/** A headline number that counts up to `value` on first paint. */
export function Metric({ value, suffix = '', className = '' }) {
  const [shown, setShown] = useState(reduceMotion ? value : 0)

  useEffect(() => {
    if (reduceMotion) {
      setShown(value)
      return
    }

    const duration = 1100
    const start = performance.now()
    let frame

    const tick = (now) => {
      const p = Math.min(1, (now - start) / duration)
      const eased = 1 - Math.pow(1 - p, 3)

      setShown(Math.round(value * eased))

      if (p < 1) frame = requestAnimationFrame(tick)
    }

    frame = requestAnimationFrame(tick)

    return () => cancelAnimationFrame(frame)
  }, [value])

  return (
    <div className={`metric ${className}`.trim()}>
      {shown.toLocaleString()}
      {shown === value ? suffix : ''}
    </div>
  )
}

/* ---- button ripple -------------------------------------------- */

/**
 * The ripple on `.btn`, attached once at the document level.
 *
 * It is deliberately not a component: every button in the app should
 * get it, including the ones inside dangerously-rendered markup, and a
 * single delegated listener is cheaper than a wrapper per button.
 */
export function useRipple() {
  useEffect(() => {
    if (reduceMotion) return

    const onDown = (event) => {
      const button = event.target.closest('.btn')

      if (!button) return

      const box = button.getBoundingClientRect()
      const size = Math.max(box.width, box.height)
      const ripple = document.createElement('span')

      ripple.className = 'ripple'
      ripple.style.width = ripple.style.height = `${size}px`
      ripple.style.left = `${event.clientX - box.left - size / 2}px`
      ripple.style.top = `${event.clientY - box.top - size / 2}px`
      button.appendChild(ripple)

      setTimeout(() => ripple.remove(), 620)
    }

    document.addEventListener('pointerdown', onDown)

    return () => document.removeEventListener('pointerdown', onDown)
  }, [])
}
