import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Brush WASM Demo',
  description: 'Minimal Next.js demo using Brush WASM for 3D Gaussian Splatting',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html>
      <body style={{
        margin: 0,
        padding: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: '#0a0a0a',
        overflow: 'hidden'
      }}>
        {children}
      </body>
    </html>
  )
}
