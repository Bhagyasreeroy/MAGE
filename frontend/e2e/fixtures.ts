import { expect, type Page } from '@playwright/test';

/** A fresh account per test, so runs never collide over saved datasets. */
export function uniqueEmail(): string {
  return `e2e-${Date.now()}-${Math.floor(Math.random() * 1e4)}@example.com`;
}

/**
 * Register and land on an authenticated page.
 *
 * The redirect to /dashboard fires before the auth context has written the
 * token, so waiting on the URL alone is a race: navigating immediately after
 * it lands you back on /signin with no session. Waiting for the token itself
 * is the real "signed in" signal.
 */
export async function signUp(page: Page, email: string): Promise<void> {
  await page.goto('/signup');
  await page.fill('#name', 'E2E Test');
  await page.fill('#email', email);
  await page.fill('#password', 'Passw0rd!');
  await page.click('button[type="submit"]');

  await page.waitForFunction(
    () => Boolean(window.localStorage.getItem('mage_access_token')),
    null,
    { timeout: 30_000 },
  );
  await expect(page).toHaveURL(/dashboard/, { timeout: 30_000 });
}

/** Navigate within the dashboard and assert we did not get bounced to sign-in. */
export async function gotoAuthed(page: Page, path: string): Promise<void> {
  await page.goto(path);
  await expect(page).toHaveURL(new RegExp(path.replace(/\//g, '\\/')));
}
