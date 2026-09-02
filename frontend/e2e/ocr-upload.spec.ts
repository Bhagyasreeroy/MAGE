import { expect, test, type Page } from '@playwright/test';
import path from 'node:path';

import { gotoAuthed } from './fixtures';

/**
 * OCR upload, from the user's side.
 *
 * These drive the real stack — a real image goes to the real OCR.space API and
 * comes back as a real dataset — because the whole point of the feature is
 * that chain, and a mocked version would prove nothing about it.
 */

const IMAGE = path.join(__dirname, 'assets', 'table.png');
const PDF = path.join(__dirname, 'assets', 'table.pdf');
const CSV = path.join(__dirname, 'assets', 'plain.csv');
const LONG_PDF = path.join(__dirname, 'assets', 'five-pages.pdf');


/**
 * OCR.space throttles free API keys whenever their shared service is busy,
 * answering 503 with "E571: Free OCR API overloaded currently". That is a
 * third-party outage, not a defect here, and letting it fail the suite trains
 * everyone to ignore red. So these tests wait for either the result or the
 * app's error message, and skip — with the real reason — when it is the
 * throttle. A PRO key is never throttled.
 */
async function expectDatasetOrSkip(page: Page, pattern: RegExp): Promise<void> {
  const result = page.getByText(pattern);
  const throttled = page.getByText(/overloaded|throttled|E571/i);

  await expect(result.or(throttled).first()).toBeVisible({ timeout: 90_000 });

  if (await throttled.count()) {
    test.skip(true, `OCR.space is throttling free keys: ${await throttled.first().textContent()}`);
  }
  await expect(result.first()).toBeVisible();
}

test.describe('Datasets page', () => {
  test('an image becomes a dataset', async ({ page }) => {
    await gotoAuthed(page, '/dashboard/datasets');

    await page.locator('input[type="file"]').setInputFiles(IMAGE);

    // The extracted table lands as a normal dataset row, named after the image.
    await expectDatasetOrSkip(page, /table_ocr\.csv/i);
    // 4 data rows and 4 columns came out of the picture.
    await expect(page.getByText(/4\s*(rows|×|x)/i).first()).toBeVisible();
  });

  test('a PDF becomes a dataset', async ({ page }) => {
    await gotoAuthed(page, '/dashboard/datasets');
    await page.locator('input[type="file"]').setInputFiles(PDF);
    await expectDatasetOrSkip(page, /table_ocr\.csv/i);
  });

  test('a CSV still uploads through the normal ingest path', async ({ page }) => {
    await gotoAuthed(page, '/dashboard/datasets');
    await page.locator('input[type="file"]').setInputFiles(CSV);
    // Named as uploaded — no "_ocr" suffix, because OCR was never involved.
    await expect(page.getByText(/plain\.csv/i)).toBeVisible({ timeout: 60_000 });
    await expect(page.getByText(/plain_ocr\.csv/i)).toHaveCount(0);
  });

  test('the file picker offers images and PDF', async ({ page }) => {
    await gotoAuthed(page, '/dashboard/datasets');
    const accept = await page.locator('input[type="file"]').getAttribute('accept');
    expect(accept).toContain('.png');
    expect(accept).toContain('.pdf');
    expect(accept).toContain('.csv');
  });
});

test.describe('Limits are explained before an upload is wasted', () => {
  test('a PDF over the 3-page limit is refused with a clear reason', async ({ page }) => {
    await gotoAuthed(page, '/dashboard/datasets');

    // Refused in the browser, so this never reaches OCR.space and never
    // spends a request from the daily allowance.
    let ocrCalled = false;
    page.on('request', (r) => {
      if (r.url().includes('/ocr/')) ocrCalled = true;
    });

    await page.locator('input[type="file"]').setInputFiles(LONG_PDF);

    await expect(page.getByText(/3 pages/i).first()).toBeVisible({ timeout: 30_000 });
    expect(ocrCalled).toBe(false);
    await expect(page.getByText(/five-pages_ocr\.csv/i)).toHaveCount(0);
  });
});

test.describe('New Analysis page', () => {
  test('an image attaches as the run dataset and enables Run', async ({ page }) => {
    await gotoAuthed(page, '/dashboard/analysis/new');

    await page.fill('#goal-input', 'Which product generates the most revenue');
    await page.locator('input[type="file"]#dataset-file').setInputFiles(IMAGE);

    // The image is OCR'd immediately and shown as an attached dataset, so the
    // user can see what was extracted before committing to a run.
    await expectDatasetOrSkip(page, /table_ocr\.csv|extracted/i);

    // Attach-and-wait: the pipeline must not have started on its own.
    await expect(page).toHaveURL(/analysis\/new/);
    const run = page.locator('button[type="submit"]');
    await expect(run).toBeEnabled();
  });

  test('the picker offers images and PDF here too', async ({ page }) => {
    await gotoAuthed(page, '/dashboard/analysis/new');
    const accept = await page.locator('input[type="file"]#dataset-file').getAttribute('accept');
    expect(accept).toContain('.png');
    expect(accept).toContain('.pdf');
  });
});
