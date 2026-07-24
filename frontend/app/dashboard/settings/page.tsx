'use client';

import { useState } from 'react';
import { changePassword, deleteAccount, updateProfile } from '../../lib/api';
import { useAuth } from '../../lib/auth-context';

const CheckIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
  </svg>
);

type ExpertiseLevel = 'beginner' | 'intermediate' | 'expert';

export default function SettingsPage() {
  const { user, logout } = useAuth();
  // Dashboard layout only renders this page once `user` is loaded, so this is safe.
  const [name, setName] = useState(user!.full_name);
  const [email, setEmail] = useState(user!.email);
  const [defaultExpertise, setDefaultExpertise] = useState<ExpertiseLevel>(
    (user!.default_expertise_level as ExpertiseLevel) || 'intermediate',
  );

  const [profileSaving, setProfileSaving] = useState(false);
  const [profileSaved, setProfileSaved] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);

  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [passwordSaving, setPasswordSaving] = useState(false);
  const [passwordSaved, setPasswordSaved] = useState(false);
  const [passwordError, setPasswordError] = useState<string | null>(null);

  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  async function handleSaveProfile(e: React.FormEvent) {
    e.preventDefault();
    setProfileSaving(true);
    setProfileError(null);
    setProfileSaved(false);
    try {
      const updated = await updateProfile({
        full_name: name,
        email,
        default_expertise_level: defaultExpertise,
      });
      setName(updated.full_name);
      setEmail(updated.email);
      setProfileSaved(true);
      setTimeout(() => setProfileSaved(false), 2000);
    } catch (err) {
      setProfileError(err instanceof Error ? err.message : 'Failed to save changes');
    } finally {
      setProfileSaving(false);
    }
  }

  async function handleChangePassword(e: React.FormEvent) {
    e.preventDefault();
    setPasswordError(null);
    setPasswordSaved(false);

    if (newPassword !== confirmPassword) {
      setPasswordError('New passwords do not match.');
      return;
    }

    setPasswordSaving(true);
    try {
      await changePassword(currentPassword, newPassword);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setPasswordSaved(true);
      setTimeout(() => setPasswordSaved(false), 2000);
    } catch (err) {
      setPasswordError(err instanceof Error ? err.message : 'Failed to change password');
    } finally {
      setPasswordSaving(false);
    }
  }

  async function handleDeleteAccount() {
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteAccount();
      logout();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete account');
      setDeleting(false);
    }
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

      <div className="space-y-8 animate-slide-up delay-100">
        {/* ── Profile ──────────────────────────────────────────────── */}
        <form onSubmit={handleSaveProfile} className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8">
          <h2 className="font-[family-name:var(--font-serif)] text-xl font-bold text-navy mb-6">
            Profile
          </h2>

          <div className="flex items-center gap-6 mb-8">
            <div className="w-20 h-20 bg-gradient-to-br from-lavender to-dusty-rose rounded-2xl flex items-center justify-center text-3xl font-bold text-navy shadow-inner shadow-white/20">
              {(name || email).charAt(0).toUpperCase()}
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
                minLength={2}
                required
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
                required
                className="w-full bg-cream/50 border border-dusty-rose/20 rounded-2xl px-5 py-4 text-navy text-sm focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all"
              />
            </div>
            <div>
              <label htmlFor="default-expertise" className="block text-xs font-bold text-navy/60 uppercase tracking-widest mb-3">
                Default Expertise Level
              </label>
              <select
                id="default-expertise"
                value={defaultExpertise}
                onChange={(e) => setDefaultExpertise(e.target.value as ExpertiseLevel)}
                className="w-full bg-cream/50 border border-dusty-rose/20 rounded-2xl px-5 py-4 text-navy text-sm focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all appearance-none cursor-pointer"
              >
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="expert">Expert</option>
              </select>
              <p className="text-xs text-navy/40 mt-2 font-light">
                Pre-selected whenever you start a new analysis.
              </p>
            </div>
          </div>

          {profileError && (
            <div className="mt-5 bg-dusty-rose/10 border border-dusty-rose/30 rounded-2xl p-4 text-dusty-rose text-sm">
              {profileError}
            </div>
          )}

          <button
            type="submit"
            disabled={profileSaving}
            className="w-full mt-6 bg-navy text-cream font-medium py-4 rounded-[2rem] hover:bg-navy-light transition-all hover:-translate-y-1 shadow-xl shadow-navy/15 flex items-center justify-center gap-3 text-base disabled:opacity-60"
          >
            {profileSaved ? (
              <><CheckIcon /> Saved Successfully</>
            ) : profileSaving ? (
              'Saving…'
            ) : (
              'Save Changes'
            )}
          </button>
        </form>

        {/* ── Change Password ──────────────────────────────────────── */}
        <form onSubmit={handleChangePassword} className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8">
          <h2 className="font-[family-name:var(--font-serif)] text-xl font-bold text-navy mb-6">
            Change Password
          </h2>

          <div className="space-y-5">
            <div>
              <label htmlFor="current-password" className="block text-xs font-bold text-navy/60 uppercase tracking-widest mb-3">
                Current Password
              </label>
              <input
                id="current-password"
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                required
                className="w-full bg-cream/50 border border-dusty-rose/20 rounded-2xl px-5 py-4 text-navy text-sm focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all"
              />
            </div>
            <div>
              <label htmlFor="new-password" className="block text-xs font-bold text-navy/60 uppercase tracking-widest mb-3">
                New Password
              </label>
              <input
                id="new-password"
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                minLength={8}
                required
                className="w-full bg-cream/50 border border-dusty-rose/20 rounded-2xl px-5 py-4 text-navy text-sm focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all"
              />
            </div>
            <div>
              <label htmlFor="confirm-password" className="block text-xs font-bold text-navy/60 uppercase tracking-widest mb-3">
                Confirm New Password
              </label>
              <input
                id="confirm-password"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                minLength={8}
                required
                className="w-full bg-cream/50 border border-dusty-rose/20 rounded-2xl px-5 py-4 text-navy text-sm focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all"
              />
            </div>
          </div>

          {passwordError && (
            <div className="mt-5 bg-dusty-rose/10 border border-dusty-rose/30 rounded-2xl p-4 text-dusty-rose text-sm">
              {passwordError}
            </div>
          )}

          <button
            type="submit"
            disabled={passwordSaving}
            className="w-full mt-6 bg-navy text-cream font-medium py-4 rounded-[2rem] hover:bg-navy-light transition-all hover:-translate-y-1 shadow-xl shadow-navy/15 flex items-center justify-center gap-3 text-base disabled:opacity-60"
          >
            {passwordSaved ? (
              <><CheckIcon /> Password Updated</>
            ) : passwordSaving ? (
              'Updating…'
            ) : (
              'Update Password'
            )}
          </button>
        </form>

        {/* ── Danger Zone ──────────────────────────────────────────── */}
        <div className="border border-dusty-rose/30 rounded-[2rem] p-8">
          <h3 className="text-sm font-bold text-dusty-rose uppercase tracking-widest mb-3">Danger Zone</h3>
          <p className="text-sm text-navy/50 font-light mb-6">
            Permanently delete your account and all associated datasets and analysis history. This cannot be undone.
          </p>

          {deleteError && (
            <div className="mb-4 bg-dusty-rose/10 border border-dusty-rose/30 rounded-2xl p-4 text-dusty-rose text-sm">
              {deleteError}
            </div>
          )}

          {!confirmingDelete ? (
            <button
              type="button"
              onClick={() => setConfirmingDelete(true)}
              className="text-sm font-medium text-dusty-rose border border-dusty-rose/40 px-6 py-3 rounded-2xl hover:bg-dusty-rose/10 transition-colors"
            >
              Delete Account
            </button>
          ) : (
            <div className="flex items-center gap-3">
              <p className="text-sm text-navy font-medium">Are you sure?</p>
              <button
                type="button"
                onClick={handleDeleteAccount}
                disabled={deleting}
                className="text-sm font-bold text-cream bg-dusty-rose px-6 py-3 rounded-2xl hover:bg-red-500 transition-colors disabled:opacity-60"
              >
                {deleting ? 'Deleting…' : 'Yes, delete everything'}
              </button>
              <button
                type="button"
                onClick={() => setConfirmingDelete(false)}
                disabled={deleting}
                className="text-sm font-medium text-navy/50 px-6 py-3 rounded-2xl hover:bg-cream-dark/50 transition-colors"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
