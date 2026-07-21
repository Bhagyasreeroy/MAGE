'use client';

import { useState } from 'react';
import { useAuth } from '../../lib/auth-context';

const CheckIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
  </svg>
);

export default function SettingsPage() {
  const { user, logout } = useAuth();
  // Dashboard layout only renders this page once `user` is loaded, so this is safe.
  const [name, setName] = useState(user!.full_name);
  const [email, setEmail] = useState(user!.email);
  const [openaiKey, setOpenaiKey] = useState('');
  const [defaultExpertise, setDefaultExpertise] = useState('intermediate');
  const [saved, setSaved] = useState(false);

  function handleSave(e: React.FormEvent) {
    e.preventDefault();
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-10 animate-fade-in">
        <h1 className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy mb-2">
          Settings
        </h1>
        <p className="text-navy/50 font-light text-lg">
          Manage your profile and preferences.
        </p>
      </div>

      <form onSubmit={handleSave} className="space-y-8 animate-slide-up delay-100">
        {/* ── Profile ──────────────────────────────────────────────── */}
        <div className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8">
          <h2 className="font-[family-name:var(--font-serif)] text-xl font-bold text-navy mb-6">
            Profile
          </h2>

          <div className="flex items-center gap-6 mb-8">
            <div className="w-20 h-20 bg-gradient-to-br from-lavender to-dusty-rose rounded-2xl flex items-center justify-center text-3xl font-bold text-navy shadow-inner shadow-white/20">
              {name.charAt(0)}
            </div>
            <div>
              <p className="font-bold text-lg text-navy">{name}</p>
              <p className="text-sm text-navy/50 font-light">{email}</p>
            </div>
          </div>

          <div className="space-y-5">
            <div>
              <label htmlFor="settings-name" className="block text-xs font-bold text-navy/60 uppercase tracking-widest mb-3">
                Name
              </label>
              <input
                id="settings-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full bg-cream/50 border border-dusty-rose/20 rounded-2xl px-5 py-4 text-navy text-sm focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all"
              />
            </div>
            <div>
              <label htmlFor="settings-email" className="block text-xs font-bold text-navy/60 uppercase tracking-widest mb-3">
                Email
              </label>
              <input
                id="settings-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-cream/50 border border-dusty-rose/20 rounded-2xl px-5 py-4 text-navy text-sm focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all"
              />
            </div>
          </div>
        </div>

        {/* ── API Keys ─────────────────────────────────────────────── */}
        <div className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8">
          <h2 className="font-[family-name:var(--font-serif)] text-xl font-bold text-navy mb-2">
            API Keys
          </h2>
          <p className="text-sm text-navy/50 font-light mb-6">
            These are used for LLM-powered analysis and embeddings.
          </p>

          <div>
            <label htmlFor="openai-key" className="block text-xs font-bold text-navy/60 uppercase tracking-widest mb-3">
              OpenAI API Key
            </label>
            <input
              id="openai-key"
              type="password"
              value={openaiKey}
              onChange={(e) => setOpenaiKey(e.target.value)}
              placeholder="sk-..."
              className="w-full bg-cream/50 border border-dusty-rose/20 rounded-2xl px-5 py-4 text-navy placeholder:text-navy/20 text-sm focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all font-mono"
            />
          </div>
        </div>

        {/* ── Preferences ──────────────────────────────────────────── */}
        <div className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8">
          <h2 className="font-[family-name:var(--font-serif)] text-xl font-bold text-navy mb-6">
            Preferences
          </h2>

          <div>
            <label htmlFor="default-expertise" className="block text-xs font-bold text-navy/60 uppercase tracking-widest mb-3">
              Default Expertise Level
            </label>
            <select
              id="default-expertise"
              value={defaultExpertise}
              onChange={(e) => setDefaultExpertise(e.target.value)}
              className="w-full bg-cream/50 border border-dusty-rose/20 rounded-2xl px-5 py-4 text-navy text-sm focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all appearance-none cursor-pointer"
            >
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="expert">Expert</option>
            </select>
          </div>
        </div>

        {/* ── Save ─────────────────────────────────────────────────── */}
        <button
          type="submit"
          className="w-full bg-navy text-cream font-medium py-4 rounded-[2rem] hover:bg-navy-light transition-all hover:-translate-y-1 shadow-xl shadow-navy/15 flex items-center justify-center gap-3 text-base"
        >
          {saved ? (
            <><CheckIcon /> Saved Successfully</>
          ) : (
            'Save Changes'
          )}
        </button>

        {/* ── Danger Zone ──────────────────────────────────────────── */}
        <div className="border border-dusty-rose/30 rounded-[2rem] p-8 mt-6">
          <h3 className="text-sm font-bold text-dusty-rose uppercase tracking-widest mb-3">Danger Zone</h3>
          <p className="text-sm text-navy/50 font-light mb-6">
            Permanently delete your account and all associated data.
          </p>
          <button
            type="button"
            className="text-sm font-medium text-dusty-rose border border-dusty-rose/40 px-6 py-3 rounded-2xl hover:bg-dusty-rose/10 transition-colors"
          >
            Delete Account
          </button>
        </div>
      </form>
    </div>
  );
}
