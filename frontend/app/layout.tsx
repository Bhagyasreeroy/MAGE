import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
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
    <html lang="en" className={inter.variable}>
      <body className="antialiased font-sans">{children}</body>
    </html>
  );
}
