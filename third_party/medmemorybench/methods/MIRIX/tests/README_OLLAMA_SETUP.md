# Ollama Setup Guide

Use this guide to install Ollama locally and make it available to Mirix for topic extraction or other local LLM workflows.

## 1. Install Ollama

### macOS
- Download the `.pkg` installer from https://ollama.com/download and run it, **or**
- Use Homebrew:
  ```bash
  brew install ollama
  brew services start ollama  # optional; keeps the daemon running in the background
  ```
- Launch the “Ollama” app once (or run `ollama serve`) so the daemon is active.

### Linux
- Run the official install script:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- Start the daemon:
  ```bash
  ollama serve
  ```
- For persistent/background usage, create a systemd service that runs `ollama serve`.

### Windows
- Install Windows Subsystem for Linux (WSL2) if you haven’t already.
- Inside WSL2 (Ubuntu or similar), follow the Linux instructions above.
- Expose the service to Windows by adding this to PowerShell (adjust the distribution/IP as needed):
  ```powershell
  $wslIp = wsl hostname -I | ForEach-Object { $_.Split(' ')[0] }
  netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=11434 connectaddress=$wslIp connectport=11434
  ```
- Keep the WSL2 session running; the Windows host can now reach `http://127.0.0.1:11434`.

## 2. Pull the Model You Need

Example for Gemma 3 1B:
```bash
ollama pull gemma3:1b
```

If you plan to reference the model via `local_model_for_retrieval`, match the tag exactly (e.g., `gemma3:1b`).

## 3. Configure Mirix Environment

Set the base URL so Mirix can reach the Ollama daemon. Either export it in the shell before running Mirix or add it to the project’s `.env` file:
```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
```
or in `.env`:
```
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Restart the Mirix server/process after changing the environment so `model_settings` picks up the new value.

## 4. Verify

Run a quick health check:
```bash
ollama list
curl -s http://127.0.0.1:11434/api/version
```

To test Mirix’s topic extraction path directly, send a POST to `/api/chat` using the sample payload in `payload.json` or by calling the `/memory/retrieve/conversation` endpoint with `"local_model_for_retrieval": "<model-name>"`.
