# hx

A small coding agent for the terminal, kind of a mini Claude Code. I built it to understand what actually goes on
around the model: the agent loop, tools, permissions, sandboxing, context management, and so on.

It's mostly Python, with two small C++ programs for code search and running shell commands. It runs local models
through Ollama (I've been using `qwen3:14b`), or anything with an OpenAI-compatible API.

## What's in it

- Agent loop with tools: read, search, edit, bash
- Permission prompts, a macOS sandbox for shell commands, and undo via git snapshots
- Context handling: `AGENTS.md`, memory, compaction, sessions you can resume
- Hooks, skills, subagents, and a basic MCP client
- A small eval suite to check whether changes actually help

## Running it

```bash
ollama pull qwen3:14b
uv sync
./scripts/build_cpp.sh   # optional, there are Python fallbacks
uv run hx                # /help for commands, -v to see everything the model gets
uv run pytest -q
```

## About the evals

There are 10 tasks, and I run each one twice, so the numbers are noisy (roughly ±10%). The full setup passes about 80%,
and without the "go verify your changes" check it's about 70%. That difference is basically one task, though, so take
it as a hint rather than proof.

The model matters a lot too. Turning on qwen3's thinking mode got about the same 80% with no extra checks, but each
task took around 8× longer. A bigger hosted model would very likely do better still (I haven't tested that). It's a
tradeoff: a small local model is fast, free and private, and the harness has to make up for what it gets wrong.

## Things I learned along the way

Some ideas from people who've thought about this much more than me, and where they show up in hx:

- **An agent is "an LLM, a loop, and enough tokens"** ([Thorsten Ball](https://ampcode.com/how-to-build-an-agent)).
  The loop really is small. Most of the code is everything around it.
- **The context window is the scarcest resource.** Models get less reliable as the input grows
  ([Chroma, "Context Rot"](https://www.trychroma.com/research/context-rot)). So hx caps tool output, clears old tool
  results first, and only summarizes when that's not enough
  ([Anthropic on context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).
- **Keep the prompt prefix stable.** Manus calls KV-cache hit rate "the single most important metric for a
  production-stage AI agent" ([Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)).
  I saw it locally too: re-reading a 6K-token prompt took 14 s without the cache and 0.04 s with it.
- **Leave errors in the context.** "Erasing failure removes evidence" (same Manus post). Tool errors go back to the
  model with a hint on how to fix the call, instead of crashing the turn.
- **Agents declare victory too early**
  ([Anthropic, long-running harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)).
  That's exactly what I saw with a small model, and it's why hx asks it to run something before it's allowed to finish.
- **Write the plan back into the context.** A todo list the model keeps rewriting keeps the goal near the end of the
  context, where it pays the most attention. Manus calls this recitation.
- **Subagents should go off and read, not write in parallel**
  ([Cognition, "Don't Build Multi-Agents"](https://cognition.com/blog/dont-build-multi-agents)). In hx a subagent gets a
  fresh context, runs on its own, and only its final report comes back.
- **Interrupts are normal.** Pressing Ctrl-C in the middle of a tool call still has to leave a valid conversation, so
  every tool call gets a result and the model is told you stopped it.

### Where's the Ralph loop I keep hearing about?

Not in hx, and that's on purpose. [Ralph](https://ghuntley.com/ralph/) (Geoffrey Huntley) isn't a harness feature,
it's a bash loop that runs a whole agent again and again: `while :; do cat PROMPT.md | claude-code ; done`. Each run
starts with a fresh context, picks one item from a plan file, and the tests keep it honest. So it sits *outside* a
harness, and hx can be the agent inside it:

```bash
while :; do cat PROMPT.md | hx -p - --mode auto-edit; done
```

I haven't tried this properly yet. With a 14B model I'd guess "one item per loop" and good tests matter even more.

## Note

I built this with Claude as a pair programmer. It was a learning project for me, to see what goes into a coding agent
harness by building one.
