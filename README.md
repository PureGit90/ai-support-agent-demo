# AI Support QA Analyzer

A Streamlit demo app that analyzes failed AI customer support conversations, identifies failure patterns, and generates targeted prompt and workflow fixes — moving resolution rates toward 80%.

## Features

- **Upload / Paste tab** — Paste any support conversation or pick from 3 realistic ecommerce failure examples (wrong return policy, hallucinated tracking, exchange refused)
- **Analysis tab** — Claude Haiku analyzes the conversation and outputs: Failure Type, Root Cause, Severity, Prompt Fix, Workflow Fix, Estimated Impact
- **Playbook tab** — A growing record of shipped fixes, color-coded by failure type
- **Metrics tab** — 8-week resolution rate trend (62% to 74%), escalation rate decline, failure type breakdown pie chart, fix velocity bar chart

## Local Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

The app runs without an API key (demo mode shows placeholder analysis). Set `ANTHROPIC_API_KEY` to enable live Claude analysis.

## Deploy to Streamlit Cloud

1. Push this directory to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set `ANTHROPIC_API_KEY` as a secret in the Streamlit Cloud app settings (Settings > Secrets):
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Deploy. The app will be live at `https://<your-app>.streamlit.app`.

## Model

Uses `claude-haiku-4-5-20251001` for fast, cost-efficient analysis. Each analysis call consumes ~800-1200 input tokens + ~400 output tokens.

## File Structure

```
app.py            — Main Streamlit application
requirements.txt  — Python dependencies
README.md         — This file
```
