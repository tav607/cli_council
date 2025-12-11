# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLI Council is a Python CLI tool that implements Karpathy's three-stage LLM collaboration mechanism (https://github.com/karpathy/llm-council). It orchestrates multiple LLM CLI tools to answer questions through collective deliberation:

1. **Stage 1 (First Opinions)**: Parallel queries to all configured LLMs (Codex, Gemini, Claude Code)
2. **Stage 2 (Review)**: Each LLM anonymously reviews and ranks all responses (including its own)
3. **Stage 3 (Final Response)**: A Chairman (Gemini) synthesizes all responses and reviews into a final answer

## Running the Tool

```bash
# Basic usage
./cli_council.py "your question here"

# Quiet mode (only show final answer)
./cli_council.py -q "your question"

# Skip review stage (faster but potentially lower quality)
./cli_council.py --skip-review "your question"

# Interactive mode (prompts for question)
./cli_council.py
```

## Architecture

The entire implementation is in `cli_council.py`. Key components:

- **CLIS dict**: Configuration for LLM CLI tools (Codex, Gemini, Claude Code) with their command-line invocations
- **CHAIRMAN_CMD**: Gemini CLI used as the final arbitrator
- **CliResult/ReviewResult**: Dataclasses for structured output handling
- **Sandbox Isolation**: Each CLI call runs in a temporary directory (`tempfile.TemporaryDirectory`) to prevent context pollution from project files
- **create_label_mapping()**: Maps model names to anonymous labels (Response A/B/C) for blind peer review
- **parse_ranking_from_text()**: Regex-based extraction of rankings from LLM review responses
- **calculate_aggregate_rankings()**: Computes consensus ranking based on average position across reviews

## Telegram Bot

The project includes a Telegram bot interface (`telegram_bot.py`).

### Setup
```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your TELEGRAM_BOT_TOKEN and ALLOWED_USER_IDS

# Run the bot
uv run python telegram_bot.py
```

### Bot Commands
- `/start` - Welcome message
- `/quiet` - Switch to quiet mode (final answer only)
- `/verbose` - Switch to verbose mode (all stages)
- `/status` - Show current mode
- Direct text message - Ask a question

### Features
- HTML formatting for bold/italic/code in responses
- Configurable HTTP timeouts for long-running queries
- User whitelist access control
- Concurrent request prevention per user

## Dependencies

**CLI tools** (must be in PATH):
- `codex` (OpenAI Codex CLI)
- `gemini` (Google Gemini CLI)
- `claude` (Anthropic Claude Code CLI)

**Python packages** (managed via `pyproject.toml`):
- `python-telegram-bot` - Telegram bot framework
- `python-dotenv` - Environment variable management
