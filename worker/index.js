// Single Worker entry point for the unified Workers+static-assets model
// (what "wrangler deploy" / wrangler.jsonc's `assets` config uses now --
// Cloudflare Pages' old file-based `functions/api/*.js` routing convention,
// which static_site/functions/api/rewrite.js was originally written for,
// does not carry over to this model). This Worker explicitly routes
// /api/rewrite itself and delegates every other request to the ASSETS
// binding (the static_site/ directory).
//
// Contract for /api/rewrite: POST {prompt: string} -> {rewritten: string}
// on success, {rewritten: null} once every model in the chain has failed
// (the client treats this as the graceful-degradation signal, not an
// error -- a network hiccup and "quota's actually gone today" both resolve
// to the same branch). No ML/embedding logic here at all -- that all
// happens client-side via transformers.js; this is a pure proxy to Groq
// for the query-rewrite step, same SYSTEM_PROMPT/MODEL_CHAIN/
// reasoning_effort handling as query_rewrite.py.

const MODEL_CHAIN = [
  "openai/gpt-oss-20b",
  "openai/gpt-oss-120b",
  "qwen/qwen3.8-27b",
  "openai/gpt-oss-safeguard-20b",
];

const SYSTEM_PROMPT = `A user is describing, in their own words, the kind of algorithmic \
problem or solution mechanism they want to find. Rewrite their description as a single \
paragraph (2-4 sentences) in the exact register used by a corpus of solution-mechanism \
abstracts, so it can be embedded and compared against them directly.

Rules (identical to the corpus's own generation rules, since this text must land in the \
same region of embedding space):
- Use ONLY bare computer-science vocabulary: "graph", "node", "directed edge", "interval", \
  "array", "sequence", "tree", "set", "state", "connected component", "prefix sum", "sliding \
  window", "monotonic stack", and so on. If the user's own words use a domain-flavor noun \
  (e.g. "course", "student", "stone", "building"), translate it to the underlying CS object \
  rather than repeating it.
- Describe the underlying goal/invariant and how the technique(s) achieve it, the way a \
  solution's own mechanism would be described -- not a restatement or summary of the user's \
  request.
- If the user's description is vague or high-level (e.g. "graph problems" or "something \
  with dynamic programming"), write the most representative, concrete mechanism paragraph \
  for that request rather than staying abstract or hedging.
- If the user names a specific technique (e.g. "two pointers", "topological sort", "binary \
  search on the answer"), center the paragraph on that technique's actual mechanism.
- Output ONLY the rewritten paragraph. No preamble, no quotes, no markdown, no commentary.
`;

async function callGroq(apiKey, model, prompt, useReasoningEffort) {
  const body = {
    model,
    max_tokens: 600,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: prompt },
    ],
  };
  if (useReasoningEffort) body.reasoning_effort = "low";

  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errText = await res.text();
    const err = new Error(`Groq ${res.status}: ${errText}`);
    err.status = res.status;
    err.body = errText;
    throw err;
  }
  const data = await res.json();
  return (data.choices?.[0]?.message?.content || "").trim();
}

async function handleRewrite(request, env) {
  let prompt;
  try {
    const body = await request.json();
    prompt = (body.prompt || "").trim();
  } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }
  if (!prompt) {
    return Response.json({ error: "empty prompt" }, { status: 400 });
  }

  const apiKey = env.GROQ_API_KEY;
  if (!apiKey) {
    return Response.json({ error: "server misconfigured: no API key" }, { status: 500 });
  }

  for (const model of MODEL_CHAIN) {
    try {
      let text;
      try {
        text = await callGroq(apiKey, model, prompt, true);
      } catch (e) {
        // non-reasoning models (qwen, safeguard) reject reasoning_effort outright
        if (e.body && e.body.includes("reasoning_effort")) {
          text = await callGroq(apiKey, model, prompt, false);
        } else {
          throw e;
        }
      }
      if (text) {
        return Response.json({ rewritten: text });
      }
    } catch (e) {
      continue; // try next model in the chain
    }
  }

  return Response.json({ rewritten: null });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.pathname === "/api/rewrite" && request.method === "POST") {
      return handleRewrite(request, env);
    }
    return env.ASSETS.fetch(request);
  },
};
