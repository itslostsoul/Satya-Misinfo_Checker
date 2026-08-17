# Satya — AI Forward-Checker (PS-S03) starter

End-to-end skeleton so the team can work in parallel from minute one.
It already runs — every pipeline is a stub that returns `unverifiable`
until you fill it in.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # then paste your BotFather token in
python bot.py
```

Forward a photo or a text message to your bot on Telegram — you'll get
a (stub) verdict card back immediately.

## Where to plug in (matches the 5-way team split)

| File | Owner | Job |
|---|---|---|
| [bot.py](bot.py) | Person 1 — Orchestrator | Telegram wiring, 60s budget, dispatch |
| [pipelines/image_forensics.py](pipelines/image_forensics.py) | Person 2 | AI-gen/manipulation detection |
| [pipelines/reverse_context.py](pipelines/reverse_context.py) | Person 3 | Reverse image search + date/context check |
| [pipelines/text_claim.py](pipelines/text_claim.py) | Person 4 | Claim extraction + fact-checker matching + calibration |
| [card.py](card.py) | Person 5 | Verdict merge, card copy, regional-language translation |
| [pipelines/voice.py](pipelines/voice.py) | stretch | Synthetic-voice detection |
| [schema.py](schema.py) | shared contract | Don't change without telling everyone |

Every pipeline function returns a `PipelineResult` (see `schema.py`).
As long as you keep that shape, you can rewrite the internals however
you want without breaking anyone else's code.

## Before the demo

- Run the bot against the 8-item judging set repeatedly, not just once.
- Check every reply lands under 60s — `card.py`'s footer prints actual latency.
- Fill in `blind_spots` honestly in `card.py` — it's part of the deliverables.
