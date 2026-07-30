import './globals.css'

export const metadata = {
  title: 'Haki AI',
  description: 'Kenyan Legal Document Generator',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}