import './globals.css'

export const metadata = {
  title: 'Agentic AI | Investment Analyst',
  description: 'Multi-Agent Investment Research Platform',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
