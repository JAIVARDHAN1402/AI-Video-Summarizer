"use client";

import ReactMarkdown from "react-markdown";

/**
 * The model writes markdown — fenced code, bold, lists — in both the
 * explanation bodies and the Q&A answers. Rendering it as plain text showed
 * literal asterisks and backticks, and turned code into an unreadable run of
 * lines. Elements are styled here rather than with a typography plugin so they
 * match the surrounding design tokens in both themes.
 */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="prose-body">
      <ReactMarkdown
        components={{
          p: ({ children }) => <p className="my-2">{children}</p>,
          ul: ({ children }) => <ul className="my-2 ml-5 list-disc space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="my-2 ml-5 list-decimal space-y-1">{children}</ol>,
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
          strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer" className="text-accent hover:underline">
              {children}
            </a>
          ),
          h1: ({ children }) => <h3 className="mt-3 mb-1 font-semibold text-ink">{children}</h3>,
          h2: ({ children }) => <h3 className="mt-3 mb-1 font-semibold text-ink">{children}</h3>,
          h3: ({ children }) => <h3 className="mt-3 mb-1 font-semibold text-ink">{children}</h3>,
          blockquote: ({ children }) => (
            <blockquote className="my-2 border-l-2 border-border-strong pl-3 text-ink-muted">
              {children}
            </blockquote>
          ),
          code: ({ className, children }) => {
            // react-markdown gives fenced blocks a language class and inline
            // code none; only the fenced form should become a dark block.
            const fenced = /language-/.test(className || "");
            if (!fenced) {
              return (
                <code className="rounded bg-surface-active px-1 py-0.5 font-mono text-[0.85em] text-accent-bright">
                  {children}
                </code>
              );
            }
            return (
              <code className="block overflow-x-auto rounded border border-transparent bg-code-bg p-3 font-mono text-[0.82rem] leading-relaxed text-code-ink">
                {children}
              </code>
            );
          },
          pre: ({ children }) => <pre className="my-2 overflow-x-auto">{children}</pre>,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
