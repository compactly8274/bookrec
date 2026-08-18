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
| `OLLAMA_URL` | *(empty)* | Ollama API base URL. Empty disables LLM reasons. |
| `OLLAMA_MODEL` | `gemma3:4b` | Model name for reason generation. |

## Volumes

| Host path | Container path | Purpose |
|---|---|---|
| `/mnt/user/Books` | `/calibre` | Calibre library (read-only) |
| `/mnt/user/appdata/bookrec` | `/config` | State, feedback, generated index |

## Permissions

The container runs as an unprivileged user (`UID 99`, `GID 1000`). The `/config`
volume must be writable by that user, otherwise the app cannot create
`bookrec.db` and will fail to start. On Unraid this usually works out of the box
(`nobody:users`), but if you see permission errors, fix the ownership of the
host directory:

```bash
chown -R 99:1000 /mnt/user/appdata/bookrec
```

## First boot

The first start downloads the embedding model (~90 MB) and indexes your entire
Calibre library, which can take several minutes on a large collection. The
healthcheck has an extended `start_period` to allow for this; the app is ready
once `http://your-host:8484/api/stats` returns `200`.
