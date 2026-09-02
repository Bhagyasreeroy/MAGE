'use client';

/**
 * lib/upload-routing.ts
 * ─────────────────────
 * One upload control, two destinations.
 *
 * A CSV, Excel or Parquet file already *is* structured data and goes to
 * /analysis/ingest, which parses it properly. An image or PDF is a *picture*
 * of data and goes through OCR first. The user should not have to know which
 * is which — they pick a file, and this decides where it goes.
 *
 * Sizing is the other thing handled here. OCR.space's free plan hard-rejects
 * anything over 1.5 MB, and a phone photo is routinely 3-5 MB, so images are
 * downscaled in the browser before upload. That uses canvas rather than an
 * image library, keeping this dependency-free like the voice WAV encoder.
 */

import { convertImageToDataset, ingestDataset, type OCRToDatasetResult } from './api';

/** Pictures of data — these go through OCR. */
export const OCR_EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff', 'gif', 'pdf'];

/** Structured data — these go straight to the ingestion pipeline. */
export const DATASET_EXTENSIONS = ['csv', 'tsv', 'json', 'parquet', 'xlsx', 'xls'];

/** The `accept` attribute for every dataset file input in the app. */
export const UPLOAD_ACCEPT = [...DATASET_EXTENSIONS, ...OCR_EXTENSIONS]
  .map((ext) => `.${ext}`)
  .join(',');

/**
 * OCR.space documents a 1 MB limit for free keys while the live API's own
 * rejection says 1.5 MB. Targeting the smaller, documented number keeps
 * uploads under both, so a file cannot land in the grey zone between them and
 * fail depending on which limit is being enforced that day.
 */
const OCR_TARGET_BYTES = 1_000_000;

/** Below this, shrinking further destroys the text OCR needs to read. */
const MIN_DIMENSION_SCALE = 0.3;

export function extensionOf(filename: string): string {
  return filename.includes('.') ? filename.split('.').pop()!.toLowerCase() : '';
}

export function isOcrFile(file: File): boolean {
  return OCR_EXTENSIONS.includes(extensionOf(file.name));
}

/** OCR.space's free and standard PRO plans both stop at three PDF pages. */
export const MAX_PDF_PAGES = 3;

/**
 * Count the pages in a PDF, or return null when it cannot be determined.
 *
 * Deliberately a best-effort read of the raw bytes rather than a PDF library:
 * the only question here is "is this obviously over the limit", and pulling in
 * a parser to answer it would cost more than the check is worth. PDFs that
 * store their page tree in compressed object streams will not match either
 * pattern — hence null, which is treated as "let the API decide" rather than
 * blocking a file that might be fine.
 */
export async function countPdfPages(file: File): Promise<number | null> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  // latin1 keeps every byte a single character, so byte offsets in the
  // structure survive the conversion to string intact.
  let text = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    text += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }

  // "/Type /Page" but not "/Type /Pages", which is the tree node, not a page.
  const pageObjects = text.match(/\/Type\s*\/Page[^s]/g);
  if (pageObjects && pageObjects.length > 0) return pageObjects.length;

  // Fall back to the page tree's own count; take the largest, since nested
  // trees each carry one and the root holds the total.
  const counts = [...text.matchAll(/\/Count\s+(\d+)/g)].map((m) => Number(m[1]));
  if (counts.length > 0) return Math.max(...counts);

  return null;
}

export function isPdf(file: File): boolean {
  return extensionOf(file.name) === 'pdf';
}

/**
 * Shrink an oversized image until it fits, preserving aspect ratio.
 *
 * Re-encodes as JPEG: OCR reads text, not colour fidelity, and JPEG at 0.85
 * is dramatically smaller than PNG for a photograph. Returns the original
 * untouched when it already fits, so a small screenshot is never re-encoded
 * and never loses sharpness it did not need to.
 */
export async function downscaleImage(file: File): Promise<File> {
  if (file.size <= OCR_TARGET_BYTES) return file;

  const bitmap = await loadImage(file);
  let scale = Math.sqrt(OCR_TARGET_BYTES / file.size);

  // Compression is not linear in pixel count, so the first guess can still
  // land over the limit. Step down until it fits or we hit the floor, past
  // which the text would be too small for OCR to resolve anyway.
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const blob = await drawToJpeg(bitmap, scale);
    if (blob && blob.size <= OCR_TARGET_BYTES) {
      return new File([blob], replaceExtension(file.name, 'jpg'), { type: 'image/jpeg' });
    }
    scale *= 0.75;
    if (scale < MIN_DIMENSION_SCALE) break;
  }

  throw new Error(
    'This image is too large to process even after resizing. Try cropping it to just the table, or use a lower-resolution photo.',
  );
}

function replaceExtension(filename: string, extension: string): string {
  const base = filename.includes('.') ? filename.slice(0, filename.lastIndexOf('.')) : filename;
  return `${base}.${extension}`;
}

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      // Revoked only after decoding: releasing it earlier can abort the load
      // in some browsers.
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('That image could not be read. It may be corrupt or in an unsupported format.'));
    };
    img.src = url;
  });
}

function drawToJpeg(img: HTMLImageElement, scale: number): Promise<Blob | null> {
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('This browser cannot resize the image.');
  // White ground: a transparent PNG would otherwise flatten to black in JPEG
  // and hide the very text being extracted.
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.85));
}

export interface UploadedDataset {
  datasetId: string;
  filename: string;
  rowCount: number;
  columnCount: number;
  warnings: string[];
  /** True when the file was a picture and went through OCR. */
  viaOcr: boolean;
  /** A short excerpt of what OCR read, so the user can sanity-check it. */
  extractedPreview?: string;
}

/**
 * Upload any supported file and get back a dataset, whichever route it took.
 *
 * PDFs are not downscaled — re-rendering a PDF client-side is not something
 * canvas can do safely — so an oversized one is reported plainly instead.
 */
export async function uploadAnyFile(file: File): Promise<UploadedDataset> {
  if (!isOcrFile(file)) {
    const result = await ingestDataset(file);
    return {
      datasetId: result.dataset_id ?? '',
      filename: file.name,
      rowCount: result.row_count,
      columnCount: result.column_count,
      warnings: result.warnings ?? [],
      viaOcr: false,
    };
  }

  let toSend = file;
  if (isPdf(file)) {
    const pages = await countPdfPages(file);
    if (pages !== null && pages > MAX_PDF_PAGES) {
      throw new Error(
        `This PDF has ${pages} pages. The OCR service reads at most ${MAX_PDF_PAGES} pages, so only part of it would come through. Split it, or export just the pages with the table on them.`,
      );
    }
    if (file.size > OCR_TARGET_BYTES) {
      throw new Error(
        `This PDF is ${(file.size / 1_048_576).toFixed(1)} MB, over the 1 MB OCR limit. Try exporting fewer pages, or a lower-quality scan.`,
      );
    }
  } else {
    toSend = await downscaleImage(file);
  }

  const ocr: OCRToDatasetResult = await convertImageToDataset(toSend);
  return {
    datasetId: ocr.dataset_id,
    filename: ocr.filename,
    rowCount: ocr.row_count,
    columnCount: ocr.column_count,
    warnings: [],
    viaOcr: true,
    extractedPreview: ocr.full_text_preview,
  };
}
