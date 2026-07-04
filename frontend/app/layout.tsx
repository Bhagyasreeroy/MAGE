import type { Metadata } from 'next';
import { Cormorant_Garamond, Manrope } from 'next/font/google';
import './globals.css';

const manrope = Manrope({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
});

const cormorant = Cormorant_Garamond({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-serif',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'MAGE — Multi-Agent Goal-conditioned EDA',
  description:
    'MAGE ingests heterogeneous data, conditions EDA workflows on your analytical goal, ' +
    'and returns RAG-grounded, explainable recommendations tailored to your expertise.',
  keywords: ['EDA', 'multi-agent AI', 'data analysis', 'RAG', 'LangChain', 'FastAPI'],
  openGraph: {
    title: 'MAGE — Multi-Agent Goal-conditioned EDA',
    description: 'AI-powered exploratory data analysis with multi-agent orchestration.',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${manrope.variable} ${cormorant.variable}`}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
