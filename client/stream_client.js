#!/usr/bin/env node
// Generate an image via mfluxible, rendering step progress and previews inline in
// the terminal as they stream in, then save the final image to disk.
//
// Uses the iTerm2 inline-image protocol, supported by iTerm2, WezTerm, and some
// other modern terminals. In an unsupported terminal the escape codes are just
// ignored (or print as stray characters) -- the text progress lines and the
// saved file still work either way. Not tmux-aware: iTerm2's protocol needs
// extra passthrough wrapping inside tmux, which this script doesn't do.
//
// No npm dependencies -- just Node built-ins.
//
// Uses node:http/https directly rather than fetch(): generation can take a long
// time (the very first request after a server restart also pays the one-time
// weight-quantization cost), and undici's fetch() imposes a default ~5-minute
// body/header inactivity timeout that isn't reachable/disableable via a plain
// require("node:undici") in every Node build. http.request has no such timeout
// unless one is explicitly set, which is what we want for a stream this bursty.
//
// Renders using the protocol's chunked MultipartFile variant -- the same one
// iTerm2's own `imgcat` reference tool uses by default -- rather than one giant
// File=...:<base64> sequence. iTerm2's own source caps how much data it'll
// accumulate for a single OSC sequence at 1,048,576 bytes (VT100XtermParser.m),
// truncating (not just dropping) whatever comes after, which can corrupt what
// renders afterward too. Detailed/photographic diffusion output compresses
// poorly and unpredictably as PNG, so a full-resolution image can realistically
// cross that limit. Splitting the base64 payload into many small FilePart
// sequences (no single one over ~200 bytes) means no image, at any size, can
// ever hit that cap -- so unlike an earlier version of this script, there's no
// need to downscale or recompress anything (and no `sips` shell-out either):
// the terminal renders the exact bytes that get saved.

const fs = require("node:fs");
const http = require("node:http");
const https = require("node:https");
const { URL } = require("node:url");
const { parseArgs } = require("node:util");

// imgcat (iTerm2's own reference tool) uses 200-byte chunks, but its own
// comment says that's specifically "to help it get through tmux" -- we're not
// tmux-wrapping at all (see the header comment above), so we don't need chunks
// that small. A ~1.5MB image at 200 bytes/chunk means ~10,000 separate escape
// sequences in one burst; using a much larger chunk size (still safely under
// iTerm2's real 1,048,576-byte single-sequence cap, with margin) cuts that by
// ~2500x, which is worth trying against intermittent-failure reports that
// aren't explained by anything in the data itself (verified byte-for-byte
// correct in both a failing and a succeeding capture).
const CHUNK_SIZE = 500_000;

function showImage(pngBytes, width = "auto") {
  // width="auto" (with height defaulting to auto too) renders at the image's
  // native pixel size divided by the display's backing scale factor -- the
  // same sizing `imgcat` uses by default. A fixed cell-count width instead
  // scales with the terminal's font/cell size, unrelated to the image's
  // actual dimensions, and renders inconsistently with `imgcat`'s output.
  const b64 = pngBytes.toString("base64");
  const parts = [`\x1b]1337;MultipartFile=inline=1;size=${pngBytes.length};width=${width};preserveAspectRatio=1\x07`];
  for (let i = 0; i < b64.length; i += CHUNK_SIZE) {
    parts.push(`\x1b]1337;FilePart=${b64.slice(i, i + CHUNK_SIZE)}\x07`);
  }
  parts.push("\x1b]1337;FileEnd\x07\n");
  // A large stdout.write() isn't guaranteed to finish flushing before the
  // process exits right after -- Node's docs call TTY/file writes on POSIX
  // "synchronous", but that only means the call is dispatched via a blocking
  // path, not that completion is otherwise observed by the caller; if stdout
  // is ever a pipe (Node's own docs call pipes genuinely asynchronous on
  // POSIX), an unawaited write can race process exit and get truncated. Wait
  // for the write's own completion callback rather than trust either case.
  return new Promise((resolve, reject) => {
    process.stdout.write(parts.join(""), (err) => (err ? reject(err) : resolve()));
  });
}

function postSSE(url, body, onEvent) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const transport = parsed.protocol === "https:" ? https : http;
    const payload = JSON.stringify(body);

    const req = transport.request(
      {
        hostname: parsed.hostname,
        port: parsed.port,
        path: parsed.pathname + parsed.search,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
        },
      },
      (res) => {
        if (res.statusCode !== 200) {
          // Read the body before giving up rather than res.resume()-ing it away: the
          // server rejects --guidance/--negative-prompt on a model that can't act on
          // them, and its message says which, where the status line says nothing.
          let errBody = "";
          res.setEncoding("utf8");
          res.on("data", (chunk) => (errBody += chunk));
          res.on("end", () => {
            let detail = errBody;
            try {
              detail = JSON.parse(errBody).message || errBody;
            } catch {}
            reject(new Error(`request failed: ${res.statusCode} ${detail || res.statusMessage}`));
          });
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";
        // onEvent may be async (it awaits showImage's write completion) --
        // chain calls through `pending` so events are handled strictly in
        // order and "end" doesn't resolve until the last one has actually
        // finished, not just been kicked off.
        let pending = Promise.resolve();
        res.on("data", (chunk) => {
          buffer += decoder.decode(chunk, { stream: true });
          let idx;
          while ((idx = buffer.indexOf("\n\n")) !== -1) {
            const rawEvent = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);
            for (const line of rawEvent.split("\n")) {
              if (!line.startsWith("data: ")) continue;
              const event = JSON.parse(line.slice("data: ".length));
              pending = pending.then(() => onEvent(event));
            }
          }
        });
        res.on("end", () => {
          pending.then(resolve, reject);
        });
        res.on("error", reject);
      },
    );

    req.on("error", reject);
    req.write(payload);
    req.end();
  });
}

async function main() {
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    allowPositionals: true,
    options: {
      url: { type: "string", default: "http://127.0.0.1:8420/v1/images/generations" },
      width: { type: "string", default: "1024" },
      height: { type: "string", default: "1024" },
      // No default for steps/guidance: left unset they're the server's decision, and
      // it knows which model it loaded (a step count tuned for Z-Image-Turbo is four
      // times too small for FLUX.1-dev). Sending null asks for that model's own
      // default rather than overriding it with this script's guess.
      steps: { type: "string" },
      seed: { type: "string" },
      guidance: { type: "string" },
      "negative-prompt": { type: "string" },
      "preview-every": { type: "string", default: "0" },
      out: { type: "string", default: "output.png" },
    },
  });

  const prompt = positionals[0];
  if (!prompt) {
    console.error("usage: stream_client.js <prompt> [--url URL] [--width N] [--height N] [--steps N] [--seed N] [--guidance F] [--negative-prompt TEXT] [--preview-every N] [--out FILE]");
    process.exit(1);
  }

  const body = {
    prompt,
    width: Number(values.width),
    height: Number(values.height),
    steps: values.steps !== undefined ? Number(values.steps) : null,
    seed: values.seed !== undefined ? Number(values.seed) : null,
    guidance: values.guidance !== undefined ? Number(values.guidance) : null,
    negative_prompt: values["negative-prompt"] ?? null,
    preview_every: Number(values["preview-every"]),
    stream: true,
  };

  await postSSE(values.url, body, (event) => handleEvent(event, values.out));
}

async function handleEvent(event, outPath) {
  switch (event.type) {
    case "start":
      console.log(`seed=${event.seed} steps=${event.total_steps}`);
      break;
    case "thinking":
      console.log(
        `  step ${event.step}/${event.total_steps} (${event.step_ms}ms, ${event.elapsed_ms}ms total)`,
      );
      if (event.preview) {
        await showImage(Buffer.from(event.preview, "base64"));
      }
      break;
    case "image": {
      const pngBytes = Buffer.from(event.data, "base64");
      fs.writeFileSync(outPath, pngBytes);
      console.log(`saved ${outPath} (seed=${event.seed}, ${event.generation_time.toFixed(1)}s)`);
      await showImage(pngBytes);
      break;
    }
    case "error":
      console.error(`error: ${event.message}`);
      process.exit(1);
      break;
  }
}

main().catch((err) => {
  // Message only, no stack: a refused request (guidance on a model that ignores it,
  // server not running) is ordinary CLI input feedback, not a crash to debug -- and
  // it matches what stream_client.py prints for the same cases.
  console.error(`error: ${err.message || err}`);
  process.exit(1);
});
