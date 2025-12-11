# CLI Council

A multi-LLM collaboration system inspired by [Karpathy's LLM Council](https://github.com/karpathy/llm-council). It orchestrates multiple CLI-based LLM tools to answer questions through a three-stage deliberation process.

## How It Works

```
    Question
        |
        v
+-------+-------+-------+
|       |       |       |
v       v       v       v
Codex  Gemini  Claude   ...     <- Stage 1: First Opinions
|       |       |       |
+-------+-------+-------+
        |
        v
  Anonymous Peer Review          <- Stage 2: Each model reviews & ranks all
        |
        v
  Chairman (Gemini)              <- Stage 3: Synthesize final answer
        |
        v
   Final Answer
```

1. **Stage 1 (First Opinions)**: All LLMs answer the question in parallel
2. **Stage 2 (Peer Review)**: Each LLM anonymously reviews and ranks all responses (including its own)
3. **Stage 3 (Final Response)**: Chairman synthesizes the best answer based on all inputs

## Features

- **Multi-model collaboration**: Leverages diverse perspectives from different LLMs
- **Anonymous peer review**: Models evaluate responses without knowing authorship
- **Consensus ranking**: Aggregates rankings to identify best responses
- **Sandbox isolation**: CLI calls run in temporary directories to prevent context pollution
- **Telegram bot integration**: Query the council via Telegram

## Prerequisites

The following CLI tools must be installed and available in PATH:

- `codex` - [OpenAI Codex CLI](https://github.com/openai/codex)
- `gemini` - [Google Gemini CLI](https://github.com/google-gemini/gemini-cli)
- `claude` - [Anthropic Claude Code CLI](https://github.com/anthropics/claude-code)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cli-council.git
cd cli-council

# Install dependencies with uv
uv sync
```

## Usage

### CLI Mode

```bash
# Basic usage
./cli_council.py "What is the best programming language for beginners?"

# Quiet mode (only show final answer)
./cli_council.py -q "Your question"

# Skip review stage (faster but may reduce quality)
./cli_council.py --skip-review "Your question"

# Interactive mode
./cli_council.py
```

### Telegram Bot Mode

1. Create a bot via [@BotFather](https://t.me/BotFather) and get the token

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your TELEGRAM_BOT_TOKEN and ALLOWED_USER_IDS
   ```

3. Run the bot:
   ```bash
   uv run python telegram_bot.py
   ```

**Bot Commands:**
- `/start` - Welcome message
- `/quiet` - Switch to quiet mode (final answer only)
- `/verbose` - Switch to verbose mode (show all stages)
- `/status` - Show current mode
- Direct message - Ask a question

## Deployment

For production deployment on a VM:

```bash
# Using systemd (recommended)
sudo nano /etc/systemd/system/cli-council.service
```

Example service file:
```ini
[Unit]
Description=CLI Council Telegram Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/cli-council
ExecStart=/path/to/cli-council/.venv/bin/python telegram_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable cli-council
sudo systemctl start cli-council

# View logs
journalctl -u cli-council -f
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `ALLOWED_USER_IDS` | Comma-separated list of allowed Telegram user IDs (empty = allow all) |
| `TELEGRAM_PROXY` | (Optional) Proxy URL for Telegram API |

## License

MIT
