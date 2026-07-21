'use client';

import Link from 'next/link';
import { useState } from 'react';

const CheckIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
  </svg>
);

const SparkleIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
  </svg>
);

const ArrowRightIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
  </svg>
);

const DocumentIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
  </svg>
);

const SendIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
  </svg>
);

const MOCK_RESULT = {
  goal: 'Identify the top factors driving customer churn in Q4 2025',
  expertise_level: 'intermediate',
  summary:
    'The analysis identified three primary drivers of customer churn: contract type (month-to-month contracts churn at 42%), lack of tech support enrollment, and high monthly charges (>$70). Customers with fiber optic internet service showed 2.3x higher churn rates compared to DSL users.',
  steps: [
    { agent: 'IngestionAgent', status: 'success', output: 'Loaded customer_churn.csv — 7,043 rows × 21 columns' },
    { agent: 'MiningAgent', status: 'success', output: 'Computed statistical profiles, correlation matrix, and churn segmentation' },
    { agent: 'VisualizationAgent', status: 'success', output: 'Generated 3 Vega-Lite chart specifications' },
    { agent: 'RecommendationAgent', status: 'success', output: 'Produced 4 RAG-grounded recommendations' },
  ],
  recommendations: [
    'Focus retention efforts on month-to-month contract customers by offering incentive-based annual plans.',
    'Investigate fiber optic service quality — churn rate is 2.3x higher than DSL, suggesting potential service issues.',
    'Implement an early-warning scoring model using MonthlyCharges, TechSupport, and Contract features.',
    'Consider bundling TechSupport with high-tier plans to reduce churn risk.',
  ],
  rag_sources: [
    'churn_analysis_handbook.pdf',
    'retention_best_practices.md',
    'telecom_industry_report_2025.pdf',
  ],
};

// Initial mockup chat history
const INITIAL_CHAT = [
  {
    id: 1,
    role: 'user',
    content: 'Can you show me the breakdown of churn by contract type specifically for senior citizens?',
  },
  {
    id: 2,
    role: 'assistant',
    content: 'Certainly! I\'ve queried the dataset for senior citizens. Among this demographic, month-to-month contracts still have the highest churn rate (48%), which is slightly higher than the general population (42%). One year and two-year contracts for seniors have negligible churn (<2%). Would you like me to generate a visualization for this?',
  }
];

export default function AnalysisResultPage() {
  const result = MOCK_RESULT;
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState(INITIAL_CHAT);

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    // Add user message
    const newMessage = {
      id: Date.now(),
      role: 'user',
      content: chatInput.trim(),
    };
    
    setChatHistory([...chatHistory, newMessage]);
    setChatInput('');

    // Mock AI response
    setTimeout(() => {
      setChatHistory(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: "I'm looking into that now. The agents are mining the dataset for new patterns based on your query...",
      }]);
    }, 1000);
  };

  return (
    <div className="max-w-4xl mx-auto pb-32">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between mb-10 animate-fade-in">
        <div>
          <div className="flex items-center gap-4 mb-3">
            <h1 className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy">
              Workspace
            </h1>
            <span className="bg-sage-light/40 text-sage border border-sage/30 text-xs font-bold px-3 py-1.5 rounded-full flex items-center gap-1.5 uppercase tracking-wider">
              <CheckIcon /> Initial Analysis Complete
            </span>
          </div>
          <p className="text-navy/60 font-light text-lg">Goal: {result.goal}</p>
        </div>
      </div>

      {/* ── Initial Report (Treated as first AI response) ──────────── */}
      <div className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8 mb-8 animate-slide-up delay-100 shadow-sm shadow-navy/5">
        <div className="flex items-center gap-4 mb-6 pb-6 border-b border-dusty-rose/15">
          <div className="w-12 h-12 bg-navy text-cream rounded-2xl flex items-center justify-center">
            <SparkleIcon />
          </div>
          <div>
            <h2 className="font-[family-name:var(--font-serif)] font-bold text-navy text-xl">Initial Report Generated</h2>
            <p className="text-xs text-navy/40 uppercase tracking-widest font-medium mt-0.5">Automated Pipeline</p>
          </div>
        </div>

        {/* Summary */}
        <div className="mb-8">
          <h3 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">Executive Summary</h3>
          <p className="text-navy/70 leading-relaxed font-light">{result.summary}</p>
        </div>

        {/* Pipeline */}
        <div className="mb-8">
          <h3 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">Agent Actions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {result.steps.map((step, idx) => (
              <div key={idx} className="bg-cream/50 border border-dusty-rose/15 rounded-2xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <p className="font-bold text-sm text-navy">{step.agent}</p>
                  <CheckIcon />
                </div>
                <p className="text-xs text-navy/50 leading-relaxed font-light">{step.output}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Recommendations */}
        <div className="mb-8">
          <h3 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">Recommendations</h3>
          <ul className="space-y-3">
            {result.recommendations.map((rec, idx) => (
              <li key={idx} className="flex gap-3 text-navy/70 font-light">
                <span className="text-navy-muted bg-cream-dark w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5">{idx + 1}</span>
                {rec}
              </li>
            ))}
          </ul>
        </div>

        {/* Sources */}
        <div>
          <h3 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">Grounded In</h3>
          <div className="flex flex-wrap gap-2">
            {result.rag_sources.map((src, idx) => (
              <span key={idx} className="bg-cream-dark/60 border border-dusty-rose/20 rounded-xl px-3 py-1.5 text-xs text-navy/60 font-light flex items-center gap-1.5 cursor-default">
                <DocumentIcon /> {src}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── Continuous Chat History ────────────────────────────────── */}
      <div className="space-y-6 animate-slide-up delay-200">
        {chatHistory.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-[2rem] p-6 ${
              msg.role === 'user' 
                ? 'bg-navy text-cream shadow-xl shadow-navy/10 rounded-tr-none' 
                : 'bg-warm-white/80 border border-dusty-rose/20 text-navy rounded-tl-none shadow-sm shadow-navy/5'
            }`}>
              {msg.role === 'assistant' && (
                <div className="flex items-center gap-2 mb-3 text-navy-muted">
                  <SparkleIcon />
                  <span className="text-xs font-bold uppercase tracking-widest">MAGE</span>
                </div>
              )}
              <p className="font-light leading-relaxed text-sm md:text-base">
                {msg.content}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* ── Sticky Chat Input ──────────────────────────────────────── */}
      <div className="fixed bottom-0 left-64 right-0 p-8 bg-gradient-to-t from-cream via-cream to-transparent pointer-events-none z-30">
        <div className="max-w-4xl mx-auto pointer-events-auto">
          <form onSubmit={handleSendMessage} className="relative group">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Ask a follow-up question or request a new visualization..."
              className="w-full bg-warm-white/90 backdrop-blur-md border border-dusty-rose/30 rounded-full pl-6 pr-16 py-5 text-navy placeholder:text-navy/40 shadow-xl shadow-navy/5 focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all"
            />
            <button
              type="submit"
              disabled={!chatInput.trim()}
              className="absolute right-3 top-3 bottom-3 w-12 bg-navy text-cream rounded-full flex items-center justify-center hover:bg-navy-light disabled:opacity-50 disabled:hover:bg-navy transition-all"
            >
              <SendIcon />
            </button>
          </form>
          <p className="text-center text-[10px] text-navy/40 font-medium uppercase tracking-widest mt-4">
            MAGE can process new queries and control agents dynamically
          </p>
        </div>
      </div>
    </div>
  );
}
