import { test as setup } from '@playwright/test';

import { signUp, uniqueEmail } from './fixtures';

/**
 * Authenticate once for the whole run and save the session.
 *
 * Registering per test hits the backend's 20/minute auth rate limit once a
 * suite grows or is re-run quickly, and the resulting failure looks exactly
 * like a broken page: the app bounces to /signin with no visible reason. One
 * account, reused, keeps auth out of what these tests are actually measuring.
 */
setup('create an authenticated session', async ({ page }) => {
  await signUp(page, uniqueEmail());
  await page.context().storageState({ path: 'e2e/.auth/user.json' });
});
