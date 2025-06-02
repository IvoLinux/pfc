# PFC CIC‑IDS 2017 – RAPIDS GPU Notebook

# Prerequisites

| Requirement                  | Version              | Quick check                                                         |
| ---------------------------- | -------------------- | ------------------------------------------------------------------- |
| **Docker Engine**            | ≥ 25.0               | `docker ‑‑version`                                                  |
| **Docker Compose V2**        | ships with Docker 25 | `docker compose version`                                            |
| **NVIDIA driver**            |  ≥ 535 (CUDA 12.x)   | `nvidia‑smi`                                                        |
| **NVIDIA Container Toolkit** | any recent           | run `docker run --rm --gpus all nvidia/cuda:12.0.1-base nvidia-smi` |

> **WSL 2 / Windows users** – install the [CUDA‑enabled driver for WSL](https://developer.nvidia.com/cuda/wsl) and reboot.

---

# Quick start (recommended)

```bash
# 1. build (first time or after Dockerfile changes)
$ docker compose build

# 2. run Jupyter Lab (detached)
$ docker compose up -d

# 3. open the browser
→ http://localhost:8888

# 4. stop when finished
$ docker compose stop
```

## First‑time Kaggle setup

1. Create an API token at [https://www.kaggle.com/settings](https://www.kaggle.com/settings) (button *Create New Token*).
2. Save the downloaded `kaggle.json` **in the project root**.
3. Keep permissions strict:

   ```bash
   chmod 600 kaggle.json
   ```

`start.sh` copies the token to the expected path inside the container on startup.

---

# Useful commands

| Action                                                      | Command                                                         |
| ----------------------------------------------------------- | --------------------------------------------------------------- |
| Attach a shell to the running container                     | `docker compose exec rapids bash`                               |
| View live logs                                              | `docker compose logs -f`                                        |
| Clean everything (container + networks + anonymous volumes) | `docker compose down --volumes`                                 |
| Rebuild after dependency changes                            | `docker compose build && docker compose up -d --force-recreate` |

---