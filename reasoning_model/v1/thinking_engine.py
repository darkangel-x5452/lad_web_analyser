"""
╔══════════════════════════════════════════════════════════════════════╗
║          🧠  LLM THINKING-MODE REASONING ENGINE  🧠                  ║
║   Mimics o1 / DeepSeek-R1 / Gemini Thinking chain-of-thought style  ║
║   Backend: Ollama (FREE · LOCAL · NO CREDIT CARD)                    ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
  1. Install Ollama  →  https://ollama.com  (free, no sign-up)
  2. Pull a model    →  ollama pull llama3.2   (or mistral, phi3, gemma2, etc.)
  3. pip install ollama rich
  4. python thinking_engine.py

WHAT IT DOES:
  Implements a 5-stage reasoning pipeline:
    STAGE 1 · UNDERSTAND  → Decompose the question into clear sub-problems
    STAGE 2 · ANALYZE     → Extract facts & evidence from provided context
    STAGE 3 · REASON      → Build logical chains, weigh options
    STAGE 4 · CHALLENGE   → Devil's advocate — find flaws in reasoning
    STAGE 5 · CONCLUDE    → Final answer with confidence score & explanation
"""

import json
import time
import textwrap
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

# ── Try to import rich for pretty output, fall back to plain text ──────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich.rule import Rule
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, title=""): print(f"\n{'─'*60} {title} {'─'*60}\n")
    console = Console()
    print("💡 Tip: pip install rich  →  for beautiful terminal output")

# ── Try Ollama ─────────────────────────────────────────────────────────────────
try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ── Try HuggingFace as fallback ────────────────────────────────────────────────
try:
    import urllib.request, urllib.parse
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  HOW "THINKING MODE" WORKS — EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════

THINKING_MODE_EXPLAINER = """
╔══════════════════════════════════════════════════════════════════════╗
║              HOW THINKING MODE WORKS IN LLMs                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Traditional LLMs predict the next token directly from input.        ║
║  Thinking-Mode models (o1, DeepSeek-R1, Gemini Thinking) add a       ║
║  hidden "scratchpad" reasoning phase BEFORE generating the answer.   ║
║                                                                       ║
║  THE PIPELINE:                                                        ║
║                                                                       ║
║   Input ──► [UNDERSTAND] ──► [ANALYZE] ──► [REASON]                  ║
║                  │               │            │                       ║
║          Decompose into    Extract key    Build logical               ║
║          sub-problems      facts/data     inference chains            ║
║                                                                       ║
║            ──► [CHALLENGE] ──► [CONCLUDE] ──► Output                 ║
║                     │               │                                 ║
║             Find flaws in    Final answer +                           ║
║             own reasoning    confidence score                         ║
║                                                                       ║
║  KEY TECHNIQUES:                                                      ║
║  • Chain-of-Thought (CoT): "Let me think step by step..."            ║
║  • Tree-of-Thought (ToT): Explore multiple reasoning branches        ║
║  • Self-Reflection: Model critiques its own intermediate answers     ║
║  • Evidence Grounding: Every claim tied back to provided data        ║
║  • Confidence Calibration: Honest about uncertainty                  ║
║                                                                       ║
║  TRAINING METHODS:                                                    ║
║  • Reinforcement Learning from Human Feedback (RLHF)                 ║
║  • Process Reward Models (PRMs): reward each reasoning STEP          ║
║  • Outcome Reward Models (ORMs): reward the final answer             ║
║  • GRPO / PPO: optimization algorithms to improve reasoning          ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThinkingStage:
    """Represents one stage in the reasoning pipeline."""
    name: str
    emoji: str
    color: str
    system_prompt: str
    output: str = ""
    duration: float = 0.0

@dataclass
class ReasoningResult:
    """Final output of the thinking engine."""
    question: str
    context: str
    stages: list = field(default_factory=list)
    final_answer: str = ""
    confidence: float = 0.0
    key_evidence: list = field(default_factory=list)
    total_time: float = 0.0
    model_used: str = ""


# ══════════════════════════════════════════════════════════════════════════════
#  LLM BACKEND — OLLAMA (Primary) + HuggingFace (Fallback)
# ══════════════════════════════════════════════════════════════════════════════

class LLMBackend:
    """Unified interface for different free LLM backends."""

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.backend = self._detect_backend()

    def _detect_backend(self) -> str:
        if OLLAMA_AVAILABLE:
            try:
                models = _ollama.list()
                available = [m.model for m in models.models]
                if not available:
                    console.print("[yellow]⚠ Ollama installed but no models pulled.[/yellow]"
                                  if RICH_AVAILABLE else
                                  "⚠ Ollama installed but no models pulled.")
                    console.print("[cyan]Run: ollama pull llama3.2[/cyan]"
                                  if RICH_AVAILABLE else
                                  "Run: ollama pull llama3.2")
                    sys.exit(1)

                # Auto-select a model if preferred not available
                preferred = ["llama3.2", "llama3.1", "llama3", "mistral",
                             "phi3", "phi3.5", "gemma2", "qwen2.5"]
                for pref in preferred:
                    if any(pref in m for m in available):
                        for m in available:
                            if pref in m:
                                self.model = m
                                break
                        break
                else:
                    self.model = available[0]

                if RICH_AVAILABLE:
                    console.print(f"[green]✓ Using Ollama model: {self.model}[/green]")
                else:
                    print(f"✓ Using Ollama model: {self.model}")
                return "ollama"
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"[yellow]⚠ Ollama not running: {e}[/yellow]")
                else:
                    print(f"⚠ Ollama not running: {e}")

        # Fallback: HuggingFace Inference API (free tier)
        if HF_AVAILABLE:
            hf_token = os.environ.get("HF_TOKEN", "")
            if hf_token:
                if RICH_AVAILABLE:
                    console.print("[green]✓ Using HuggingFace Inference API[/green]")
                else:
                    print("✓ Using HuggingFace Inference API")
                return "huggingface"

        console.print(
            "[red]✗ No LLM backend found!\n\n"
            "OPTION 1 (recommended): Install Ollama\n"
            "  → https://ollama.com (free, runs locally)\n"
            "  → ollama pull llama3.2\n\n"
            "OPTION 2: Set HuggingFace token\n"
            "  → export HF_TOKEN=hf_your_token_here\n"
            "  → Get free token at: https://huggingface.co/settings/tokens[/red]"
            if RICH_AVAILABLE else
            "✗ No LLM backend found!\n"
            "Install Ollama from https://ollama.com and run: ollama pull llama3.2"
        )
        sys.exit(1)

    def generate(self, system: str, user: str, temperature: float = 0.7) -> str:
        """Generate text using the available backend."""
        if self.backend == "ollama":
            return self._ollama_generate(system, user, temperature)
        elif self.backend == "huggingface":
            return self._hf_generate(system, user)
        return ""

    def _ollama_generate(self, system: str, user: str, temperature: float) -> str:
        try:
            response = _ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ],
                options={"temperature": temperature}
            )
            return response.message.content
        except Exception as e:
            return f"[Error generating response: {e}]"

    def _hf_generate(self, system: str, user: str) -> str:
        """HuggingFace Inference API fallback."""
        import urllib.request, json
        hf_token = os.environ.get("HF_TOKEN", "")
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        prompt = f"[INST] {system}\n\n{user} [/INST]"
        data = json.dumps({
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1024, "temperature": 0.7}
        }).encode()
        req = urllib.request.Request(
            f"https://api-inference.huggingface.co/models/{model_id}",
            data=data,
            headers={"Authorization": f"Bearer {hf_token}",
                     "Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "").replace(prompt, "").strip()
        except Exception as e:
            return f"[HuggingFace API error: {e}]"
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  THINKING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class ThinkingEngine:
    """
    5-stage thinking pipeline that mimics how reasoning models like o1,
    DeepSeek-R1, and Gemini Thinking analyse data and reach conclusions.
    """

    STAGES = [
        ThinkingStage(
            name="UNDERSTAND",
            emoji="🔍",
            color="cyan",
            system_prompt="""You are an expert analyst in the first stage of a multi-step reasoning pipeline.

Your ONLY job right now: UNDERSTAND the question deeply.

Do the following:
1. Restate the core question in your own words
2. Identify EXACTLY what information/answer is being requested
3. List all sub-questions that must be answered to reach a conclusion
4. Identify the type of reasoning required (statistical, comparative, causal, predictive, etc.)
5. Note any ambiguities or assumptions that need to be made

Be systematic. Use numbered lists. Do NOT answer the question yet."""
        ),
        ThinkingStage(
            name="ANALYZE",
            emoji="📊",
            color="blue",
            system_prompt="""You are a data analyst in stage 2 of a reasoning pipeline.
You have already understood the question. Now ANALYZE the provided context/data.

Do the following:
1. Extract every relevant fact, statistic, and data point from the context
2. Categorize each piece of evidence (supports which side/option?)
3. Identify the STRONGEST pieces of evidence
4. Identify any missing data that would be helpful
5. Note any contradictions or inconsistencies in the data
6. Quantify where possible (percentages, ratios, rankings)

Format as:
  📌 KEY FACTS: (bullet list of extracted facts)
  ⚖️ EVIDENCE FOR [option A]: ...
  ⚖️ EVIDENCE FOR [option B]: ...
  ❓ MISSING INFO: ..."""
        ),
        ThinkingStage(
            name="REASON",
            emoji="🔗",
            color="yellow",
            system_prompt="""You are a logical reasoner in stage 3 of a reasoning pipeline.
You have the facts. Now BUILD your reasoning chains.

Do the following:
1. Form 2-3 distinct logical arguments (inference chains)
2. For each argument: state premise → logic → conclusion
3. Assign a weight/strength to each argument (weak/moderate/strong)
4. Apply probabilistic thinking where appropriate
5. Identify which argument is most compelling and WHY
6. Build toward a preliminary conclusion

Format each chain as:
  ARGUMENT [N]: [Name]
  Premise: ...
  Logic: ...
  Conclusion: ...
  Strength: [weak/moderate/strong] because..."""
        ),
        ThinkingStage(
            name="CHALLENGE",
            emoji="🔥",
            color="red",
            system_prompt="""You are a devil's advocate in stage 4 of a reasoning pipeline.
Your job: CHALLENGE the preliminary reasoning ruthlessly.

Do the following:
1. List the top 3 weaknesses in the current reasoning
2. Identify what assumptions might be WRONG
3. Present the strongest counter-argument
4. Consider edge cases and alternative outcomes
5. Ask: "What would have to be true for the opposite conclusion to be correct?"
6. Assess how robust the preliminary conclusion is after this challenge

Be critical but fair. After identifying weaknesses, update the confidence level."""
        ),
        ThinkingStage(
            name="CONCLUDE",
            emoji="✅",
            color="green",
            system_prompt="""You are the final decision-maker in stage 5 of a reasoning pipeline.
Synthesize ALL previous stages and deliver the FINAL ANSWER.

Structure your response EXACTLY like this:

## 🎯 FINAL ANSWER
[Clear, direct answer to the original question]

## 📊 CONFIDENCE SCORE
[X/10] — [Brief reason for this confidence level]

## 🏆 KEY EVIDENCE SUMMARY
(Top 3 pieces of evidence that drove this conclusion, numbered)

## ⚡ REASONING PATH
(2-3 sentence summary of the logical chain that led here)

## ⚠️ CAVEATS & UNCERTAINTIES
(What could change this answer? What unknowns remain?)

## 💡 ADDITIONAL INSIGHTS
(1-2 extra observations or implications the asker might find useful)

Be definitive. Give a real answer. Don't hedge excessively."""
        ),
    ]

    def __init__(self, model: str = "llama3.2"):
        self.llm = LLMBackend(model)

    def _print_stage_header(self, stage: ThinkingStage, stage_num: int, total: int):
        if RICH_AVAILABLE:
            title = f"STAGE {stage_num}/{total}  {stage.emoji}  {stage.name}"
            console.print()
            console.rule(f"[bold {stage.color}]{title}[/bold {stage.color}]")
        else:
            print(f"\n{'='*70}")
            print(f"  STAGE {stage_num}/{total}  {stage.emoji}  {stage.name}")
            print(f"{'='*70}")

    def _print_stage_output(self, stage: ThinkingStage):
        if RICH_AVAILABLE:
            panel = Panel(
                Markdown(stage.output),
                border_style=stage.color,
                padding=(1, 2)
            )
            console.print(panel)
            console.print(f"[dim]  ⏱ {stage.duration:.1f}s[/dim]")
        else:
            print(stage.output)
            print(f"\n  ⏱ {stage.duration:.1f}s")

    def _build_user_prompt(self, question: str, context: str,
                           previous_stages: list, current_stage_name: str) -> str:
        prompt = f"ORIGINAL QUESTION:\n{question}\n\n"
        prompt += f"CONTEXT / DATA PROVIDED:\n{context}\n\n"

        if previous_stages:
            prompt += "═══ PREVIOUS REASONING STAGES ═══\n\n"
            for s in previous_stages:
                prompt += f"[{s.emoji} {s.name}]\n{s.output}\n\n"
            prompt += "═══ END OF PREVIOUS STAGES ═══\n\n"

        prompt += f"Now perform the {current_stage_name} stage of reasoning."
        return prompt

    def analyze(self, question: str, context: str) -> ReasoningResult:
        """Run the full 5-stage thinking pipeline."""
        result = ReasoningResult(
            question=question,
            context=context,
            model_used=self.llm.model
        )
        start_total = time.time()
        completed_stages = []

        if RICH_AVAILABLE:
            console.print()
            console.print(Panel(
                f"[bold white]QUESTION:[/bold white]\n{question}\n\n"
                f"[bold white]CONTEXT:[/bold white]\n{textwrap.shorten(context, 300, placeholder='...')}",
                title="[bold cyan]🧠 THINKING ENGINE ACTIVATED[/bold cyan]",
                border_style="cyan"
            ))
        else:
            print(f"\n{'*'*70}")
            print("🧠 THINKING ENGINE ACTIVATED")
            print(f"QUESTION: {question}")
            print(f"{'*'*70}")

        for i, stage in enumerate(self.STAGES, 1):
            self._print_stage_header(stage, i, len(self.STAGES))

            user_prompt = self._build_user_prompt(
                question, context, completed_stages, stage.name
            )

            t0 = time.time()

            if RICH_AVAILABLE:
                with console.status(
                    f"[{stage.color}]{stage.emoji} Thinking... ({stage.name})[/{stage.color}]",
                    spinner="dots"
                ):
                    stage.output = self.llm.generate(
                        system=stage.system_prompt,
                        user=user_prompt,
                        temperature=0.6 if stage.name == "CONCLUDE" else 0.7
                    )
            else:
                print(f"{stage.emoji} Thinking... ({stage.name})")
                stage.output = self.llm.generate(
                    system=stage.system_prompt,
                    user=user_prompt
                )

            stage.duration = time.time() - t0
            self._print_stage_output(stage)

            completed_stages.append(stage)
            result.stages.append(stage)

        result.total_time = time.time() - start_total
        result.final_answer = result.stages[-1].output if result.stages else ""

        # ── Parse confidence score from final answer ──────────────────────────
        import re
        match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', result.final_answer)
        if match:
            result.confidence = float(match.group(1)) / 10.0

        self._print_summary(result)
        return result

    def _print_summary(self, result: ReasoningResult):
        if RICH_AVAILABLE:
            table = Table(title="📈 REASONING PIPELINE SUMMARY", show_header=True,
                          header_style="bold magenta")
            table.add_column("Stage", style="cyan", width=12)
            table.add_column("Time", justify="right", style="yellow")
            table.add_column("Output Length", justify="right")

            for s in result.stages:
                table.add_row(
                    f"{s.emoji} {s.name}",
                    f"{s.duration:.1f}s",
                    f"{len(s.output)} chars"
                )

            table.add_row(
                "[bold]TOTAL[/bold]", f"[bold]{result.total_time:.1f}s[/bold]",
                f"[bold]{sum(len(s.output) for s in result.stages)} chars[/bold]"
            )
            console.print()
            console.print(table)

            conf_bar = "█" * int(result.confidence * 10) + "░" * (10 - int(result.confidence * 10))
            console.print(f"\n[bold]Confidence:[/bold] [{conf_bar}] "
                         f"{result.confidence*100:.0f}%   "
                         f"[dim]Model: {result.model_used}[/dim]")
        else:
            print(f"\n{'─'*50}")
            print(f"Total time: {result.total_time:.1f}s | Model: {result.model_used}")
            if result.confidence:
                print(f"Confidence: {result.confidence*100:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE CLI
# ══════════════════════════════════════════════════════════════════════════════

def interactive_mode(engine: ThinkingEngine):
    """Run in interactive mode, accepting custom questions and context."""
    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold]Enter your question and context data.[/bold]\n"
            "The thinking engine will reason through it in 5 stages.\n\n"
            "Type [cyan]quit[/cyan] or [cyan]exit[/cyan] to stop.",
            title="[bold green]INTERACTIVE MODE[/bold green]",
            border_style="green"
        ))
    else:
        print("\n=== INTERACTIVE MODE ===")
        print("Enter your question and context. Type 'quit' to exit.\n")

    while True:
        print()
        question = input("❓ Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("📋 Paste your context/data (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        context = "\n".join(lines).strip()

        if not context:
            context = "No additional context provided. Reason based on general knowledge."

        engine.analyze(question, context)


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO EXAMPLES
# ══════════════════════════════════════════════════════════════════════════════

DEMO_CASES = {
    "sports": {
        "question": "Who will win the game?",
        "context": """
MATCH: Los Angeles Lakers vs Golden State Warriors
DATE: Regular Season Game, Week 18

TEAM STATS (Current Season):
  Lakers:
    - Win/Loss Record: 34-21 (5th in Western Conference)
    - Points Per Game: 118.4 (7th in NBA)
    - Points Allowed Per Game: 115.2 (12th in NBA)
    - Last 10 Games: 6-4
    - Home/Away: This is an AWAY game for Lakers
    - Key Player - LeBron James: 28.4 PPG, 7.8 RPG, 7.2 APG, playing at 85% fitness (knee soreness)
    - Key Player - Anthony Davis: 25.1 PPG, 12.6 RPG — Full fitness
    - 3-point %: 36.2%

  Warriors:
    - Win/Loss Record: 38-18 (2nd in Western Conference)
    - Points Per Game: 122.1 (3rd in NBA)
    - Points Allowed Per Game: 111.4 (4th in NBA)
    - Last 10 Games: 8-2
    - Home/Away: HOME game — Warriors home record this season: 22-5
    - Key Player - Stephen Curry: 31.2 PPG — Full fitness, on a 5-game 35+ PPG streak
    - Key Player - Klay Thompson: 18.6 PPG — Full fitness
    - 3-point %: 41.8% (League-best)

HEAD-TO-HEAD (Last 5 meetings):
  Warriors won 4 of last 5 matchups
  Average margin of victory: Warriors by 8.2 points

BETTING ODDS: Warriors -6.5 (heavy favourites)
VENUE: Chase Center, San Francisco (Warriors home court)
WEATHER: N/A (indoor arena)
"""
    },
    "business": {
        "question": "Should the company launch Product X in Q3 or wait until Q4?",
        "context": """
COMPANY: TechStartup Inc. — B2B SaaS platform
PRODUCT X: AI-powered project management tool

Q3 LAUNCH SCENARIO:
  - Development: 90% complete, 2 critical bugs outstanding
  - Market window: 3 main competitors launching in Q4 (first-mover advantage possible)
  - Sales pipeline: 45 prospects already interested, 12 contracts contingent on Q3 delivery
  - Engineering team capacity: 60% (other projects ongoing)
  - Marketing budget available: $180,000

Q4 LAUNCH SCENARIO:
  - Development: Would be 100% complete, all bugs fixed
  - Market: Competitors likely to have established presence
  - Sales pipeline: Prospects may choose competitor solutions
  - Engineering team capacity: 95% (other projects wrap up Oct)
  - Marketing budget available: $250,000 (new fiscal year)
  - Holiday season (Nov-Dec) typically slower for B2B sales

FINANCIALS:
  - Each contract worth avg $48,000 ARR
  - 12 contingent contracts = potential $576,000 ARR at risk if delayed
  - Cost of fixing bugs now (crunch): estimated $35,000
  - Cost of a failed/buggy launch: estimated $200,000 (support, churn, reputation)
  - Company runway: 14 months

RISK ASSESSMENT:
  - Q3 launch risk score: 7/10 (medium-high)
  - Q4 launch risk score: 4/10 (medium-low)
  - Customer satisfaction for buggy launch historically: 2.8/5
"""
    },
    "medical": {
        "question": "Based on the patient data, what is the most likely diagnosis and recommended next steps?",
        "context": """
PATIENT: 47-year-old male, office worker
CHIEF COMPLAINT: Fatigue, increased thirst, frequent urination (past 6 weeks)

VITALS:
  - Blood Pressure: 138/88 mmHg (elevated)
  - BMI: 31.2 (obese class I)
  - Resting Heart Rate: 82 bpm

LAB RESULTS:
  - Fasting Blood Glucose: 7.4 mmol/L (reference: <5.6 normal, 5.6-6.9 prediabetes, ≥7.0 diabetes)
  - HbA1c: 7.1% (reference: <5.7% normal, 5.7-6.4% prediabetes, ≥6.5% diabetes)
  - Cholesterol Total: 5.9 mmol/L (borderline high)
  - LDL: 3.8 mmol/L (elevated)
  - HDL: 0.9 mmol/L (low)
  - Triglycerides: 2.4 mmol/L (elevated)
  - eGFR: 72 mL/min (slightly reduced)
  - Urine microalbumin: 42 mg/g (mildly elevated, reference: <30)

FAMILY HISTORY:
  - Father: Type 2 diabetes diagnosed at age 52
  - Mother: Hypertension, dyslipidemia

LIFESTYLE:
  - Diet: High processed food intake, low vegetable consumption
  - Exercise: Sedentary (< 30 min/week)
  - Sleep: ~5.5 hours/night
  - Alcohol: 15-18 units/week

NOTE: This is for educational/analytical purposes. Always consult a licensed physician.
"""
    }
}


def run_demo(engine: ThinkingEngine, demo_key: str = "sports"):
    """Run a built-in demo case."""
    demo = DEMO_CASES.get(demo_key, DEMO_CASES["sports"])
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold yellow]DEMO: {demo_key.upper()}[/bold yellow]\n\n"
            f"Running pre-loaded example to show the thinking pipeline in action.",
            border_style="yellow"
        ))
    else:
        print(f"\n=== DEMO: {demo_key.upper()} ===")

    engine.analyze(demo["question"], demo["context"])


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Print explainer
    if RICH_AVAILABLE:
        console.print(Panel(
            Markdown(THINKING_MODE_EXPLAINER.strip()),
            title="[bold magenta]🧠 HOW LLM THINKING MODE WORKS[/bold magenta]",
            border_style="magenta",
            padding=(1, 2)
        ))
    else:
        print(THINKING_MODE_EXPLAINER)

    # Parse args
    import argparse
    parser = argparse.ArgumentParser(
        description="LLM Thinking-Mode Reasoning Engine"
    )
    parser.add_argument("--demo", choices=["sports", "business", "medical"],
                        default=None, help="Run a built-in demo")
    parser.add_argument("--model", default="llama3.2",
                        help="Ollama model name (default: llama3.2)")
    parser.add_argument("--question", default=None,
                        help="Question to analyze (use with --context)")
    parser.add_argument("--context", default=None,
                        help="Context/data file path or string")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    args = parser.parse_args()

    # Initialize engine
    engine = ThinkingEngine(model=args.model)

    if args.demo:
        run_demo(engine, args.demo)

    elif args.question:
        context = args.context or "No additional context provided."
        # If context looks like a file path
        if args.context and os.path.isfile(args.context):
            with open(args.context) as f:
                context = f.read()
        engine.analyze(args.question, context)

    elif args.interactive:
        interactive_mode(engine)

    else:
        # Default: show menu
        if RICH_AVAILABLE:
            console.print(Panel(
                "[bold]Choose how to run the thinking engine:[/bold]\n\n"
                "1. [cyan]python thinking_engine.py --demo sports[/cyan]\n"
                "   → Who will win the Lakers vs Warriors game?\n\n"
                "2. [cyan]python thinking_engine.py --demo business[/cyan]\n"
                "   → Launch product Q3 or Q4?\n\n"
                "3. [cyan]python thinking_engine.py --demo medical[/cyan]\n"
                "   → Patient diagnosis from lab results\n\n"
                "4. [cyan]python thinking_engine.py --interactive[/cyan]\n"
                "   → Enter your own question + data\n\n"
                "5. [cyan]python thinking_engine.py --question 'Q?' --context 'data...'[/cyan]\n"
                "   → Pass question & context directly",
                title="[bold green]🚀 QUICK START[/bold green]",
                border_style="green"
            ))
        else:
            print("\nQUICK START:")
            print("  python thinking_engine.py --demo sports")
            print("  python thinking_engine.py --demo business")
            print("  python thinking_engine.py --interactive")

        # Run sports demo by default
        print("\nRunning default sports demo...\n")
        run_demo(engine, "sports")


if __name__ == "__main__":
    main()
