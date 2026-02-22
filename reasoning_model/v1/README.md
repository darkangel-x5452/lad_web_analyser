# 🧠 LLM Thinking-Mode Reasoning Engine

> Mimics how OpenAI o1, DeepSeek-R1, and Gemini Thinking analyse data and make decisions.  
> **100% free — runs locally — no credit card — no API keys required.**

---

## How Thinking Mode Works in LLMs

Traditional LLMs predict the next token **directly** from input. Modern "reasoning" models add a hidden **scratchpad phase** before generating the final answer.

```
Input ──► [UNDERSTAND] ──► [ANALYZE] ──► [REASON] ──► [CHALLENGE] ──► [CONCLUDE]
               │               │             │              │               │
          Decompose       Extract key    Build logic    Devil's        Final answer
          the problem     facts/data     chains         advocate       + confidence
```

### Key Techniques Used

| Technique | What it does |
|---|---|
| **Chain-of-Thought (CoT)** | "Let me think step by step..." — generates intermediate reasoning |
| **Tree-of-Thought (ToT)** | Explores multiple reasoning branches in parallel |
| **Self-Reflection** | Model critiques its own intermediate answers |
| **Evidence Grounding** | Every claim tied back to provided data |
| **Confidence Calibration** | Honest about uncertainty (avoids overconfidence) |

### Training Methods (how models learn to reason)

- **RLHF** — Reinforcement Learning from Human Feedback
- **Process Reward Models (PRM)** — reward each reasoning *step* (not just the final answer)
- **Outcome Reward Models (ORM)** — reward the final answer quality
- **GRPO / PPO** — optimization algorithms to improve the reasoning policy

---

## 5-Stage Pipeline

```
🔍 STAGE 1 · UNDERSTAND
   → Decompose the question into sub-problems
   → Identify what type of reasoning is needed

📊 STAGE 2 · ANALYZE  
   → Extract facts & evidence from provided context
   → Categorize evidence (which side does it support?)

🔗 STAGE 3 · REASON
   → Build 2-3 logical inference chains
   → Assign strength to each argument

🔥 STAGE 4 · CHALLENGE
   → Devil's advocate: find flaws in reasoning
   → Test counter-arguments
   → Update confidence

✅ STAGE 5 · CONCLUDE
   → Final answer with confidence score (X/10)
   → Key evidence summary
   → Caveats and unknowns
```

---

## Setup (5 minutes)

### Step 1: Install Ollama (free, local, no sign-up)
```bash
# macOS
brew install ollama
# or download from https://ollama.com

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: download from https://ollama.com/download
```

### Step 2: Pull a free model
```bash
ollama pull llama3.2       # Recommended — fast, smart (2GB)
# OR
ollama pull mistral        # Alternative (4GB)
# OR  
ollama pull phi3.5         # Lightweight (2GB)
# OR
ollama pull gemma2:9b      # Google's model (5GB)
```

### Step 3: Install Python deps
```bash
pip install rich ollama
```

### Step 4: Run!
```bash
# Sports demo — "Who will win the game?"
python thinking_engine.py --demo sports

# Business demo — "Q3 or Q4 product launch?"
python thinking_engine.py --demo business

# Medical demo — "What's the diagnosis?"
python thinking_engine.py --demo medical

# Interactive mode — your own question + data
python thinking_engine.py --interactive

# Pass question & context directly
python thinking_engine.py --question "Who will win?" --context "Team A stats..."

# Use a specific model
python thinking_engine.py --demo sports --model mistral
```

---

## Alternative Free Backend: HuggingFace

If you can't install Ollama, use HuggingFace's free Inference API:

1. Create free account at https://huggingface.co (no credit card)
2. Get free API token at https://huggingface.co/settings/tokens
3. Set environment variable:
   ```bash
   export HF_TOKEN=hf_your_token_here
   python thinking_engine.py --demo sports
   ```

---

## Example Output

```
╔══════════════════════════════════════════════════════════════════════╗
║  🧠 THINKING ENGINE ACTIVATED                                        ║
║  QUESTION: Who will win the game?                                    ║
╚══════════════════════════════════════════════════════════════════════╝

──────────────── STAGE 1/5  🔍  UNDERSTAND ────────────────
The question asks which of the two NBA teams — the Los Angeles Lakers
or the Golden State Warriors — is more likely to win this specific game.

Sub-questions to resolve:
  1. What is each team's current form and momentum?
  2. How significant is the home court advantage?
  3. Are any key players injured or limited?
  4. What does the head-to-head record show?
  ...

──────────────── STAGE 2/5  📊  ANALYZE ────────────────
📌 KEY FACTS:
  • Warriors record: 38-18 vs Lakers 34-21
  • Home record for Warriors: 22-5 (exceptional)
  • LeBron James at 85% fitness (knee soreness)
  • Warriors 3-point %: 41.8% (league-best)
  • Last 5 H2H: Warriors won 4
  ...

──────────────── STAGE 5/5  ✅  CONCLUDE ────────────────

## 🎯 FINAL ANSWER
Golden State Warriors will most likely win this game.

## 📊 CONFIDENCE SCORE
7.5/10 — Strong evidence favours Warriors, but LeBron + AD can always
close gaps.

## 🏆 KEY EVIDENCE
  1. Warriors 22-5 home record — elite home court advantage
  2. Stephen Curry on 5-game 35+ PPG streak
  3. LeBron at 85% fitness limits Lakers' ceiling
  ...
```

---

## File Structure

```
thinking_engine.py   ← Main application (single file, self-contained)
setup.sh             ← One-click setup script (macOS/Linux)
README.md            ← This file
```

---

## Customising

The `ThinkingEngine.STAGES` list defines each reasoning stage. You can:
- Add new stages (e.g., a `RESEARCH` stage that queries a database)
- Change the system prompts to specialise for a domain
- Adjust `temperature` per stage (lower = more deterministic for CONCLUDE)
- Add a `VERIFY` stage that cross-checks the conclusion against the data

```python
# Run programmatically
from thinking_engine import ThinkingEngine

engine = ThinkingEngine(model="llama3.2")
result = engine.analyze(
    question="Who will win the match?",
    context="Team A: ... Team B: ..."
)

print(result.final_answer)
print(f"Confidence: {result.confidence * 100:.0f}%")
```
