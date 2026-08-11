# BookRec

Self-hosted book recommender that reads your existing Calibre library and shows one recommendation at a time.

- Reads Calibre `metadata.db` read-only.
- Embeds books locally with `all-MiniLM-L6-v2`.
- Learns from your 👍 / 👎 / skip / "more like this" feedback.
- Optionally generates reasons via any Ollama-compatible endpoint.

## Run

```bash
docker compose up -d
```

Open `http://your-host:8484`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://192.168.1.104:11434` | Ollama API base URL. Empty disables LLM reasons. |
| `OLLAMA_MODEL` | `gemma3:4b` | Model name for reason generation. |

## Volumes

| Host path | Container path | Purpose |
|---|---|---|
| `/mnt/user/Books` | `/calibre` | Calibre library (read-only) |
| `/mnt/user/appdata/bookrec` | `/config` | State, feedback, generated index |
