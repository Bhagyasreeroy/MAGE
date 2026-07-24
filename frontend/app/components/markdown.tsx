'use client';

/**
 * components/markdown.tsx
 * ───────────────────────
 * A tiny, dependency-free Markdown renderer.
 *
 * The RAG layer returns methodology text as Markdown (headings, bold, inline
 * code, bullet / numbered lists, the odd table row). Rendering it as a raw
 * string shows literal `##`, `**` and backticks, so this component turns the
 * common constructs into styled JSX. It intentionally supports only what the
 * knowledge base actually emits — it is not a full CommonMark parser.
 */

import { Fragment, type ReactNode } from 'react';

// ── Inline formatting: **bold**, *italic*, `code` ────────────────────────────
function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  // Match `code`, **bold**, or *italic* (code first so its contents are literal).
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let i = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) {
      nodes.push(text.slice(last, match.index));
    }
    const token = match[0];
    const key = `${keyPrefix}-${i++}`;
    if (token.startsWith('`')) {
      nodes.push(
        <code key={key} className="bg-cream-dark/70 rounded px-1.5 py-0.5 text-[0.85em] font-mono">
          {token.slice(1, -1)}
        </code>,
      );
    } else if (token.startsWith('**')) {
      nodes.push(<strong key={key} className="font-semibold">{token.slice(2, -2)}</strong>);
    } else {
      nodes.push(<em key={key}>{token.slice(1, -1)}</em>);
    }
    last = match.index + token.length;
  }
  if (last < text.length) {
    nodes.push(text.slice(last));
  }
  return nodes;
}

// A row of `| cell | cell |` — captures the inner content, not the separator row.
const TABLE_ROW_RE = /^\s*\|(.+)\|\s*$/;
// The `|---|---|` (or `:---:`) row directly under a table header.
const TABLE_SEPARATOR_RE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

function splitTableRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

export function Markdown({ text, className = '' }: { text: string; className?: string }) {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const blocks: ReactNode[] = [];

  let paragraph: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let code: string[] | null = null;
  let table: string[][] | null = null;
  let key = 0;

  const flushTable = () => {
    if (table && table.length) {
      const [header, ...rows] = table;
      blocks.push(
        <div key={`tbl-${key++}`} className="overflow-x-auto my-2">
          <table className="text-xs border-collapse w-full">
            <thead>
              <tr>
                {header.map((cell, i) => (
                  <th key={i} className="text-left font-semibold text-navy border-b border-dusty-rose/30 px-2 py-1.5 whitespace-nowrap">
                    {renderInline(cell, `th-${key}-${i}`)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, r) => (
                <tr key={r} className="border-b border-dusty-rose/10 last:border-0">
                  {row.map((cell, c) => (
                    <td key={c} className="px-2 py-1.5 text-navy/70 align-top">
                      {renderInline(cell, `td-${key}-${r}-${c}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      table = null;
    }
  };

  const flushParagraph = () => {
    if (paragraph.length) {
      const content = paragraph.join(' ');
      blocks.push(
        <p key={`p-${key++}`} className="leading-relaxed mb-2 last:mb-0">
          {renderInline(content, `p-${key}`)}
        </p>,
      );
      paragraph = [];
    }
  };

  const flushList = () => {
    if (list && list.items.length) {
      const items = list.items.map((it, idx) => (
        <li key={idx} className="mb-1">{renderInline(it, `li-${key}-${idx}`)}</li>
      ));
      blocks.push(
        list.ordered ? (
          <ol key={`ol-${key++}`} className="list-decimal ml-5 mb-2 space-y-0.5">{items}</ol>
        ) : (
          <ul key={`ul-${key++}`} className="list-disc ml-5 mb-2 space-y-0.5">{items}</ul>
        ),
      );
      list = null;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const line = raw.trimEnd();

    // Table: a `| ... |` row followed by a `|---|---|` separator starts a
    // table; subsequent `| ... |` rows are its body until a non-table line.
    if (table !== null && TABLE_ROW_RE.test(line)) {
      table.push(splitTableRow(line));
      continue;
    }
    if (
      table === null &&
      TABLE_ROW_RE.test(line) &&
      i + 1 < lines.length &&
      TABLE_SEPARATOR_RE.test(lines[i + 1])
    ) {
      flushParagraph();
      flushList();
      table = [splitTableRow(line)];
      i += 1; // skip the separator row
      continue;
    }
    if (table !== null) {
      flushTable();
    }

    // Fenced code block toggling
    if (line.trim().startsWith('```')) {
      if (code === null) {
        flushParagraph();
        flushList();
        code = [];
      } else {
        blocks.push(
          <pre key={`code-${key++}`} className="bg-navy/90 text-cream rounded-xl p-3 my-2 overflow-x-auto text-xs font-mono">
            <code>{code.join('\n')}</code>
          </pre>,
        );
        code = null;
      }
      continue;
    }
    if (code !== null) {
      code.push(raw);
      continue;
    }

    // Blank line → end current block
    if (line.trim() === '') {
      flushParagraph();
      flushList();
      continue;
    }

    // Horizontal rule
    if (/^ {0,3}(-{3,}|\*{3,}|_{3,})$/.test(line)) {
      flushParagraph();
      flushList();
      blocks.push(<hr key={`hr-${key++}`} className="border-dusty-rose/20 my-3" />);
      continue;
    }

    // Heading
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      const sizes = ['text-lg', 'text-base', 'text-sm', 'text-sm', 'text-sm', 'text-sm'];
      blocks.push(
        <p key={`h-${key++}`} className={`font-bold text-navy mt-3 mb-1 first:mt-0 ${sizes[level - 1]}`}>
          {renderInline(heading[2], `h-${key}`)}
        </p>,
      );
      continue;
    }

    // List item (unordered - / * , or ordered 1. )
    const ul = /^\s*[-*]\s+(.*)$/.exec(line);
    const ol = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (ul || ol) {
      flushParagraph();
      const ordered = Boolean(ol);
      const item = (ul ? ul[1] : ol![1]);
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      list.items.push(item);
      continue;
    }

    // Otherwise: paragraph text
    flushList();
    paragraph.push(line.trim());
  }

  // Flush anything left open
  flushParagraph();
  flushList();
  flushTable();
  if (code !== null && code.length) {
    blocks.push(
      <pre key={`code-${key++}`} className="bg-navy/90 text-cream rounded-xl p-3 my-2 overflow-x-auto text-xs font-mono">
        <code>{code.join('\n')}</code>
      </pre>,
    );
  }

  return <div className={className}>{blocks.map((b, i) => <Fragment key={i}>{b}</Fragment>)}</div>;
}
