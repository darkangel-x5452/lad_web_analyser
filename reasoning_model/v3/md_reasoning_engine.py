"""
╔══════════════════════════════════════════════════════════════════════╗
║         🧠  MARKDOWN → LLM REASONING ENGINE                          ║
║   Loads multiple .md files, formats them as structured context,      ║
║   and feeds them to a local LLM to reason a winner/decision.         ║
║                                                                       ║
║   FREE STACK:                                                         ║
║     LLM Backend  →  Ollama  (ollama.com — no API key needed)        ║
║     Models       →  llama3.2 / mistral / phi3.5 / gemma2            ║
║     UI           →  rich    (beautiful terminal output)              ║
╚══════════════════════════════════════════════════════════════════════╝

QUICK START:
    1. Install Ollama  →  https://ollama.com
    2. ollama pull llama3.2
    3. pip install ollama rich
    4. python md_reasoning_engine.py

    Or with custom files:
    python md_reasoning_engine.py --files team_a.md team_b.md stats.md
    python md_reasoning_engine.py --dir ./my_data_folder
    python md_reasoning_engine.py --question "Who will win?" --dir ./data
"""

import os
import sys
import glob
import time
import argparse
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Rich (pretty terminal output) ─────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.columns import Columns
    from rich.text import Text
    from rich.rule import Rule
    from rich.progress import track
    from rich import box
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    class _Console:
        def print(self, *a, **k): print(*[str(x) for x in a])
        def rule(self, t=""): print(f"\n{'─'*60} {t}\n")
    console = _Console()
    print("💡 pip install rich  — for beautiful output")

# ── Ollama ─────────────────────────────────────────────────────────────────────
try:
    import ollama as _ollama
    OLLAMA = True
except ImportError:
    OLLAMA = False


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarkdownFile:
    """Represents one loaded markdown dataset."""
    path: str
    filename: str
    content: str
    size_chars: int
    section_count: int   # number of ## headers = data sections

@dataclass
class ReasoningStage:
    name: str
    emoji: str
    color: str
    prompt: str
    output: str = ""
    duration: float = 0.0

@dataclass
class ReasoningReport:
    question: str
    files: list
    stages: list
    final_answer: str = ""
    predicted_winner: str = ""
    confidence: float = 0.0
    total_time: float = 0.0
    model: str = ""


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — MARKDOWN FILE LOADER
# ══════════════════════════════════════════════════════════════════════════════

class MarkdownLoader:
    """
    Loads, validates, and prepares markdown files for LLM ingestion.
    
    KEY DESIGN DECISIONS:
    - Each file is labelled with its filename so the LLM knows what dataset
      each block of data came from (important for attribution in reasoning)
    - Files are separated by clear visual dividers the LLM can parse
    - A table of contents is prepended so the LLM knows what data is available
      before reading it all (improves attention allocation)
    - Token estimation is shown so user knows context window usage
    """

    def load_files(self, paths: list[str]) -> list[MarkdownFile]:
        """Load a list of file paths into MarkdownFile objects."""
        loaded = []
        for path in paths:
            p = Path(path)
            if not p.exists():
                _warn(f"File not found, skipping: {path}")
                continue
            if p.suffix.lower() not in (".md", ".markdown", ".txt"):
                _warn(f"Not a markdown file, skipping: {path}")
                continue
            content = p.read_text(encoding="utf-8")
            sections = content.count("\n## ") + content.count("\n# ")
            loaded.append(MarkdownFile(
                path=str(p),
                filename=p.name,
                content=content,
                size_chars=len(content),
                section_count=max(1, sections)
            ))
        return loaded

    def load_directory(self, directory: str,
                       pattern: str = "*.md") -> list[MarkdownFile]:
        """Load all markdown files from a directory."""
        paths = sorted(glob.glob(os.path.join(directory, pattern)))
        if not paths:
            paths = sorted(glob.glob(os.path.join(directory, "*.markdown")))
        return self.load_files(paths)

    def build_prompt_context(self, files: list[MarkdownFile]) -> str:
        """
        Assemble all markdown files into a single, well-structured context
        block for the LLM.

        FORMAT STRATEGY:
        ┌─────────────────────────────────────┐
        │  TABLE OF CONTENTS (what's included) │
        │  FILE 1: [filename]                  │
        │    <content>                         │
        │  FILE 2: [filename]                  │
        │    <content>                         │
        │  ...                                 │
        │  SUMMARY FOOTER                      │
        └─────────────────────────────────────┘

        Why this structure:
        - TOC lets the model "know what it knows" before reading details
        - Clear file separators prevent data from different sources bleeding together
        - Footer reinforces the dataset scope before the question is asked
        """
        lines = []

        # ── Table of Contents ──────────────────────────────────────────────
        lines.append("=" * 70)
        lines.append("📁 DATASET INDEX — FILES PROVIDED FOR ANALYSIS")
        lines.append("=" * 70)
        for i, f in enumerate(files, 1):
            est_tokens = f.size_chars // 4  # rough token estimate
            lines.append(
                f"  [{i}] {f.filename:<35} "
                f"({f.size_chars:,} chars / ~{est_tokens:,} tokens / "
                f"{f.section_count} sections)"
            )
        lines.append("")
        lines.append(f"  TOTAL: {sum(f.size_chars for f in files):,} chars "
                     f"/ ~{sum(f.size_chars for f in files)//4:,} tokens")
        lines.append("=" * 70)
        lines.append("")

        # ── File Contents ──────────────────────────────────────────────────
        for i, f in enumerate(files, 1):
            lines.append("")
            lines.append("┌" + "─" * 68 + "┐")
            lines.append(f"│  📄 FILE [{i}/{len(files)}]: {f.filename:<52}│")
            lines.append("└" + "─" * 68 + "┘")
            lines.append("")
            lines.append(f.content.strip())
            lines.append("")
            lines.append("─" * 70)

        # ── Footer ────────────────────────────────────────────────────────
        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF DATASET — All files above constitute your evidence base.")
        lines.append("Reason ONLY from the data provided in these files.")
        lines.append("=" * 70)

        return "\n".join(lines)

    def print_summary(self, files: list[MarkdownFile]):
        """Print a rich table summarising loaded files."""
        if not RICH:
            for f in files:
                print(f"  ✓ {f.filename} ({f.size_chars} chars)")
            return

        table = Table(
            title="📁 Loaded Markdown Datasets",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("File", style="bold white")
        table.add_column("Size", justify="right", style="yellow")
        table.add_column("~Tokens", justify="right", style="green")
        table.add_column("Sections", justify="right", style="blue")
        table.add_column("Preview", style="dim")

        total_chars = 0
        for i, f in enumerate(files, 1):
            preview = f.content.strip().split("\n")[0][:45].replace("#", "").strip()
            table.add_row(
                str(i),
                f.filename,
                f"{f.size_chars:,}",
                f"~{f.size_chars//4:,}",
                str(f.section_count),
                preview + "..."
            )
            total_chars += f.size_chars

        table.add_section()
        table.add_row(
            "", "[bold]TOTAL[/bold]",
            f"[bold]{total_chars:,}[/bold]",
            f"[bold]~{total_chars//4:,}[/bold]",
            "", ""
        )
        console.print(table)
        console.print()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — LLM BACKEND
# ══════════════════════════════════════════════════════════════════════════════

class OllamaBackend:
    """
    Interface to Ollama (free, local LLM runner).
    
    Ollama runs models like LLaMA 3, Mistral, Phi-3, Gemma 2 locally.
    No API key. No credit card. Data stays on your machine.
    
    Get it at: https://ollama.com
    Then: ollama pull llama3.2
    """

    def __init__(self, preferred_model: str = "llama3.2"):
        if not OLLAMA:
            _die(
                "ollama package not installed.\n"
                "Run: pip install ollama\n"
                "And: install Ollama from https://ollama.com"
            )
        self.model = self._pick_model(preferred_model)

    def _pick_model(self, preferred: str) -> str:
        try:
            available_models = _ollama.list().models
            if not available_models:
                _die(
                    "Ollama has no models pulled.\n"
                    "Run: ollama pull llama3.2\n"
                    "Other options: ollama pull mistral | phi3.5 | gemma2"
                )
            names = [m.model for m in available_models]

            # Try preferred first, then ranked fallbacks
            priority = [preferred, "llama3.2", "llama3.1", "llama3",
                        "mistral", "phi3.5", "phi3", "gemma2", "qwen2.5"]
            for p in priority:
                for n in names:
                    if p in n:
                        if RICH:
                            console.print(f"[green]✓ LLM: [bold]{n}[/bold] (Ollama — local, free)[/green]")
                        else:
                            print(f"✓ LLM: {n}")
                        return n

            # Any model works
            if RICH:
                console.print(f"[green]✓ LLM: [bold]{names[0]}[/bold][/green]")
            else:
                print(f"✓ LLM: {names[0]}")
            return names[0]

        except Exception as e:
            _die(
                f"Cannot connect to Ollama: {e}\n\n"
                "Make sure Ollama is running:\n"
                "  macOS/Windows: Open the Ollama app\n"
                "  Linux: ollama serve\n"
                "Then: ollama pull llama3.2"
            )

    def chat(self, system: str, user: str,
             temperature: float = 0.65) -> str:
        """Send a system+user message, return the assistant reply."""
        try:
            resp = _ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ],
                options={"temperature": temperature}
            )
            return resp.message.content.strip()
        except Exception as e:
            return f"[LLM Error: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — 5-STAGE REASONING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class MDReasoningEngine:
    """
    Feeds markdown files through a 5-stage chain-of-thought reasoning pipeline
    to produce an explainable winner prediction.

    Each stage receives:
      - The full dataset context (all markdown files)
      - The original question  
      - All previous stages' outputs (growing reasoning chain)
    
    This mirrors how o1/DeepSeek-R1 builds a hidden scratchpad before answering.
    """

    STAGE_DEFINITIONS = [
        {
            "name": "UNDERSTAND",
            "emoji": "🔍",
            "color": "cyan",
            "prompt": """You are in STAGE 1 of a 5-stage reasoning pipeline.

Your task: UNDERSTAND the question and map the available data.

1. Restate what the question is asking for (in your own words)
2. List every dataset/file provided and what type of information each contains
3. Identify which datasets are most relevant to answering the question
4. List the key sub-questions you must answer (e.g. "Which team has better form?")
5. Note any data gaps or limitations

DO NOT answer the main question yet. Only map and understand."""
        },
        {
            "name": "ANALYZE",
            "emoji": "📊",
            "color": "blue",
            "prompt": """You are in STAGE 2 of a 5-stage reasoning pipeline.

Your task: EXTRACT and ORGANIZE all relevant facts from every markdown file.

For EACH dataset file:
  - Pull out every statistic, metric, or data point relevant to the question
  - Note which team/side each fact favours
  - Highlight the most significant data points

Format your output as:

### From [filename]:
  ✅ FAVOURS [Team/Option A]: [fact]
  ✅ FAVOURS [Team/Option B]: [fact]
  ⚠️ NEUTRAL / CONTEXT: [fact]

End with a tally:
  EVIDENCE SCORE: [Team A] = X points | [Team B] = Y points"""
        },
        {
            "name": "REASON",
            "emoji": "🔗",
            "color": "yellow",
            "prompt": """You are in STAGE 3 of a 5-stage reasoning pipeline.

Your task: BUILD logical inference chains from the extracted evidence.

Create 3 distinct arguments, using this format for each:

ARGUMENT [N]: [Title]
  📌 Key Evidence: [2-3 specific data points from the files]
  💭 Logic: [if X and Y, then Z...]
  🎯 Conclusion: [what this argument concludes]
  💪 Strength: [STRONG / MODERATE / WEAK] — because [reason]

Then combine them:
PRELIMINARY VERDICT: [Who wins and why, based on arguments so far]
PRELIMINARY CONFIDENCE: [X/10]"""
        },
        {
            "name": "CHALLENGE",
            "emoji": "🔥",
            "color": "red",
            "prompt": """You are in STAGE 4 of a 5-stage reasoning pipeline.
You are now the devil's advocate. CHALLENGE the preliminary verdict.

1. What are the top 3 weaknesses in the current reasoning?
2. What data point most strongly contradicts the preliminary verdict?
3. Under what conditions could the OTHER side win?
4. What assumptions in the reasoning might be wrong?
5. How does this challenge change your confidence level?

Be tough but fair. After challenging:
REVISED VERDICT: [still the same or changed?]
REVISED CONFIDENCE: [X/10] [higher/lower/same] because..."""
        },
        {
            "name": "CONCLUDE",
            "emoji": "✅",
            "color": "green",
            "prompt": """You are in STAGE 5 (FINAL) of a 5-stage reasoning pipeline.

Synthesise everything and deliver the FINAL VERDICT as a single valid JSON object.

Output ONLY raw JSON — no markdown, no code fences, no explanation outside the JSON.

The JSON must follow this exact schema:

{
  "predicted_winner": "string — team or option name",
  "loser": "string — the other team or option",
  "confidence_score": number between 1 and 10,
  "confidence_percent": number between 0 and 100,
  "confidence_explanation": "string — one sentence explaining confidence level",
  "top_reasons": [
    {
      "rank": 1,
      "reason": "string — most important reason",
      "supporting_data": "string — specific stat or fact from the files"
    },
    {
      "rank": 2,
      "reason": "string",
      "supporting_data": "string"
    },
    {
      "rank": 3,
      "reason": "string",
      "supporting_data": "string"
    },
    {
      "rank": 4,
      "reason": "string",
      "supporting_data": "string"
    },
    {
      "rank": 5,
      "reason": "string",
      "supporting_data": "string"
    }
  ],
  "strongest_counter_argument": "string — best reason the other side could still win",
  "score_prediction": {
    "winner_score": "string or number",
    "loser_score": "string or number",
    "margin": "string — e.g. by 8 points",
    "reasoning": "string — brief explanation"
  },
  "what_could_change_result": [
    "string — factor 1 that could flip the prediction",
    "string — factor 2 that could flip the prediction"
  ],
  "key_insight": "string — one non-obvious observation most people would miss",
  "evidence_summary": {
    "files_analysed": number,
    "evidence_points_for_winner": number,
    "evidence_points_for_loser": number,
    "decisive_factor": "string — the single most decisive piece of evidence"
  }
}

Be definitive. Every field must be populated. Output ONLY the JSON object, nothing else."""
        }
    ]

    def __init__(self, model: str = "llama3.2"):
        self.llm = OllamaBackend(preferred_model=model)

    def run(self, question: str, files: list[MarkdownFile],
            context: str) -> ReasoningReport:
        """Execute the full 5-stage pipeline."""

        report = ReasoningReport(
            question=question,
            files=files,
            stages=[],
            model=self.llm.model
        )
        t_start = time.time()
        history = []   # accumulates previous stage outputs

        _header(f"🧠 REASONING PIPELINE — {len(files)} MARKDOWN FILES")

        for i, defn in enumerate(self.STAGE_DEFINITIONS, 1):
            stage = ReasoningStage(
                name=defn["name"],
                emoji=defn["emoji"],
                color=defn["color"],
                prompt=defn["prompt"]
            )

            _stage_banner(i, len(self.STAGE_DEFINITIONS), stage)

            # ── Build the user message ─────────────────────────────────────
            # Contains: system prompt for this stage + context + history + question
            user_msg = _build_user_message(
                question=question,
                context=context,
                history=history,
                stage_name=stage.name
            )

            # ── Call LLM ──────────────────────────────────────────────────
            t0 = time.time()
            if RICH:
                with console.status(
                    f"[{stage.color}]{stage.emoji} {stage.name} — thinking...[/{stage.color}]",
                    spinner="dots12"
                ):
                    stage.output = self.llm.chat(
                        system=stage.prompt,
                        user=user_msg,
                        temperature=0.55 if stage.name == "CONCLUDE" else 0.68
                    )
            else:
                print(f"{stage.emoji} {stage.name} — thinking...")
                stage.output = self.llm.chat(
                    system=stage.prompt,
                    user=user_msg
                )

            stage.duration = time.time() - t0

            # ── Print output ───────────────────────────────────────────────
            _print_stage_output(stage)

            history.append(stage)
            report.stages.append(stage)

        # ── Finalize ───────────────────────────────────────────────────────
        report.total_time = time.time() - t_start
        report.final_answer = report.stages[-1].output

        # Extract winner and confidence from final stage output
        report.predicted_winner, report.confidence = _parse_final_answer(
            report.final_answer
        )

        _print_pipeline_summary(report)
        return report


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _build_user_message(question: str, context: str,
                        history: list[ReasoningStage],
                        stage_name: str) -> str:
    """Build the full user message passed to the LLM at each stage."""
    parts = []

    # ── Context (all markdown files) ──────────────────────────────────────
    parts.append("═" * 70)
    parts.append("📁 DATASET CONTEXT (all markdown files)")
    parts.append("═" * 70)
    parts.append(context)
    parts.append("")

    # ── Question ──────────────────────────────────────────────────────────
    parts.append("═" * 70)
    parts.append(f"❓ QUESTION TO ANSWER: {question}")
    parts.append("═" * 70)
    parts.append("")

    # ── Previous stage outputs ────────────────────────────────────────────
    if history:
        parts.append("═" * 70)
        parts.append("📋 PREVIOUS REASONING STAGES (your work so far)")
        parts.append("═" * 70)
        for s in history:
            parts.append(f"\n[{s.emoji} STAGE: {s.name}]")
            parts.append(s.output)
            parts.append("─" * 50)
        parts.append("")

    parts.append(f"Now perform the {stage_name} stage.")
    return "\n".join(parts)


def _parse_json_conclusion(text: str) -> dict:
    """
    Extract and parse the JSON object produced by the CONCLUDE stage.

    The LLM is instructed to return raw JSON only, but it sometimes wraps
    output in markdown code fences (```json ... ```) or adds stray text.
    This function strips all that away and returns a clean Python dict.
    Falls back to a minimal dict if JSON is unparseable.
    """
    import re, json

    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    # Find the outermost {...} block
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Partial rescue: try to at least get winner + confidence via regex
        winner = ""
        conf   = 0.0
        wm = re.search(r'"predicted_winner"\s*:\s*"([^"]+)"', text)
        if wm:
            winner = wm.group(1)
        cm = re.search(r'"confidence_score"\s*:\s*(\d+(?:\.\d+)?)', text)
        if cm:
            conf = float(cm.group(1))
        return {
            "predicted_winner": winner,
            "confidence_score": conf,
            "confidence_percent": conf * 10,
            "_parse_error": "JSON was malformed; fields may be incomplete",
            "_raw_output": text
        }


def _parse_final_answer(text: str) -> tuple[str, float]:
    """
    Backwards-compat wrapper used by _print_pipeline_summary.
    Delegates to _parse_json_conclusion and extracts the two fields it needs.
    """
    data = _parse_json_conclusion(text)
    winner     = data.get("predicted_winner", "")
    conf_score = data.get("confidence_score", 0)
    confidence = float(conf_score) / 10.0 if conf_score else 0.0
    return winner, confidence


def _stage_banner(num: int, total: int, stage: ReasoningStage):
    label = f"STAGE {num}/{total}  {stage.emoji}  {stage.name}"
    if RICH:
        console.print()
        console.rule(f"[bold {stage.color}]{label}[/bold {stage.color}]")
    else:
        print(f"\n{'='*70}\n  {label}\n{'='*70}")


def _print_stage_output(stage: ReasoningStage):
    if RICH:
        console.print(Panel(
            Markdown(stage.output),
            border_style=stage.color,
            padding=(1, 2)
        ))
        console.print(f"[dim]  ⏱ {stage.duration:.1f}s — {len(stage.output):,} chars[/dim]")
    else:
        print(stage.output)
        print(f"\n  ⏱ {stage.duration:.1f}s")


def _header(title: str):
    if RICH:
        console.print()
        console.print(Panel(
            f"[bold white]{title}[/bold white]",
            style="bold magenta",
            padding=(0, 4)
        ))
    else:
        print(f"\n{'*'*70}\n  {title}\n{'*'*70}")


def _print_pipeline_summary(report: ReasoningReport):
    if RICH:
        console.print()
        console.rule("[bold magenta]📊 PIPELINE COMPLETE[/bold magenta]")

        table = Table(box=box.SIMPLE_HEAD, show_header=True,
                      header_style="bold cyan")
        table.add_column("Stage", style="cyan")
        table.add_column("Time", justify="right", style="yellow")
        table.add_column("Output", justify="right", style="dim")

        for s in report.stages:
            table.add_row(
                f"{s.emoji} {s.name}",
                f"{s.duration:.1f}s",
                f"{len(s.output):,} chars"
            )
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold green]{report.total_time:.1f}s[/bold green]",
            f"[bold]{sum(len(s.output) for s in report.stages):,}[/bold]"
        )
        console.print(table)

        # Winner box
        if report.predicted_winner:
            conf_pct = int(report.confidence * 100)
            bar = "█" * int(report.confidence * 10) + "░" * (10 - int(report.confidence * 10))
            console.print(Panel(
                f"[bold yellow]🏆 {report.predicted_winner}[/bold yellow]\n"
                f"[green]{bar}[/green] [bold]{conf_pct}%[/bold] confidence\n\n"
                f"[dim]Model: {report.model} | "
                f"Files: {len(report.files)} | "
                f"Total time: {report.total_time:.1f}s[/dim]",
                title="[bold]FINAL PREDICTION[/bold]",
                border_style="yellow",
                padding=(1, 4)
            ))
    else:
        print(f"\n{'─'*60}")
        print(f"TOTAL TIME: {report.total_time:.1f}s | MODEL: {report.model}")
        if report.predicted_winner:
            print(f"WINNER: {report.predicted_winner} ({report.confidence*100:.0f}% confidence)")


def _warn(msg: str):
    if RICH:
        console.print(f"[yellow]⚠ {msg}[/yellow]")
    else:
        print(f"⚠ {msg}")


def _die(msg: str):
    if RICH:
        console.print(Panel(f"[red]{msg}[/red]", title="[bold red]✗ ERROR[/bold red]"))
    else:
        print(f"✗ ERROR: {msg}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_QUESTION = (
    "Based on all the provided data, who will win the game "
    "between the Los Angeles Lakers and the Golden State Warriors? "
    "Provide a detailed, evidence-based prediction."
)

def main():
    parser = argparse.ArgumentParser(
        description="Feed markdown files into an LLM reasoning pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--files", nargs="+", metavar="FILE.md",
        help="Specific markdown files to load (space-separated)"
    )
    parser.add_argument(
        "--dir", metavar="DIRECTORY", default="./data",
        help="Directory to load all .md files from (default: ./data)"
    )
    parser.add_argument(
        "--question", metavar="QUESTION", default=DEFAULT_QUESTION,
        help="The question to answer from the data"
    )
    parser.add_argument(
        "--model", default="llama3.2",
        help="Ollama model name (default: llama3.2)"
    )
    parser.add_argument(
        "--max-files", type=int, default=10,
        help="Maximum number of files to load (default: 10)"
    )
    args = parser.parse_args()

    # ── Print header ───────────────────────────────────────────────────────
    if RICH:
        console.print(Panel(
            "[bold cyan]Markdown → LLM Reasoning Engine[/bold cyan]\n"
            "[dim]Loads .md files → formats as structured context "
            "→ 5-stage chain-of-thought → winner prediction[/dim]",
            border_style="cyan",
            padding=(1, 4)
        ))
    else:
        print("\n=== MARKDOWN → LLM REASONING ENGINE ===\n")

    # ── Load files ─────────────────────────────────────────────────────────
    loader = MarkdownLoader()

    if args.files:
        files = loader.load_files(args.files)
    else:
        if RICH:
            console.print(f"[dim]Loading .md files from: {args.dir}[/dim]")
        files = loader.load_directory(args.dir)

    if not files:
        _die(
            f"No markdown files found.\n\n"
            f"Tried directory: {args.dir}\n\n"
            f"Usage examples:\n"
            f"  python {sys.argv[0]} --dir ./data\n"
            f"  python {sys.argv[0]} --files team_a.md team_b.md stats.md\n\n"
            f"The ./data/ folder should contain .md files with your dataset."
        )

    files = files[:args.max_files]

    if RICH:
        console.print()
    loader.print_summary(files)

    # ── Show the question ──────────────────────────────────────────────────
    if RICH:
        console.print(Panel(
            f"[bold]{args.question}[/bold]",
            title="[bold yellow]❓ QUESTION[/bold yellow]",
            border_style="yellow"
        ))
    else:
        print(f"QUESTION: {args.question}\n")

    # ── Build combined context ─────────────────────────────────────────────
    context = loader.build_prompt_context(files)

    # Show token estimate warning if large
    total_tokens = sum(f.size_chars for f in files) // 4
    if total_tokens > 8000:
        _warn(
            f"Large context (~{total_tokens:,} tokens). "
            f"Make sure your model supports this context window. "
            f"llama3.2 supports 128k tokens."
        )

    # ── Run reasoning pipeline ─────────────────────────────────────────────
    engine = MDReasoningEngine(model=args.model)
    report = engine.run(
        question=args.question,
        files=files,
        context=context
    )

    # ── Save JSON conclusion ───────────────────────────────────────────────
    json_path = "conclusion.json"
    conclusion, full_json_path = _save_json_report(report, json_path)
    _print_json_conclusion(conclusion)

    if RICH:
        console.print(Panel(
            f"[bold green]📄 conclusion.json[/bold green]                   ← structured conclusion only\n"
            f"[bold cyan]📄 conclusion_full_pipeline.json[/bold cyan]  ← full pipeline (all 5 stages)\n"
            f"[bold yellow]📄 reasoning_report.md[/bold yellow]              ← human-readable markdown",
            title="[bold]💾 FILES SAVED[/bold]",
            border_style="dim"
        ))
    else:
        print(f"\nFiles saved:")
        print(f"  {json_path}           — conclusion JSON")
        print(f"  {full_json_path}  — full pipeline JSON")
        print(f"  reasoning_report.md   — markdown report")

    # ── Save markdown report ───────────────────────────────────────────────
    _save_report(report, "reasoning_report.md")


def _save_json_report(report: ReasoningReport, path: str) -> dict:
    """
    Parse the CONCLUDE stage output as JSON and save two files:
      1. <path>  — the structured conclusion JSON (clean, minimal)
      2. A full pipeline JSON alongside it with all stage outputs

    Returns the parsed conclusion dict so it can be printed.
    """
    import json, re
    from datetime import datetime

    # ── Parse the conclusion JSON from the LLM output ─────────────────────
    conclude_text = report.stages[-1].output if report.stages else ""
    conclusion = _parse_json_conclusion(conclude_text)

    # ── Build the full pipeline record ────────────────────────────────────
    pipeline_record = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "question": report.question,
            "model": report.model,
            "total_time_seconds": round(report.total_time, 2),
            "files_analysed": [
                {
                    "filename": f.filename,
                    "size_chars": f.size_chars,
                    "sections": f.section_count
                }
                for f in report.files
            ]
        },
        "conclusion": conclusion,
        "reasoning_stages": [
            {
                "stage": i + 1,
                "name": s.name,
                "emoji": s.emoji,
                "duration_seconds": round(s.duration, 2),
                "output_length_chars": len(s.output),
                "output": s.output
            }
            for i, s in enumerate(report.stages)
        ]
    }

    # ── Save conclusion-only JSON ──────────────────────────────────────────
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conclusion, f, indent=2, ensure_ascii=False)

    # ── Save full pipeline JSON ────────────────────────────────────────────
    full_path = path.replace(".json", "_full_pipeline.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_record, f, indent=2, ensure_ascii=False)

    return conclusion, full_path


def _print_json_conclusion(conclusion: dict):
    """Pretty-print the JSON conclusion in the terminal."""
    import json
    if RICH:
        from rich.syntax import Syntax
        json_str = json.dumps(conclusion, indent=2, ensure_ascii=False)
        console.print()
        console.print(Panel(
            Syntax(json_str, "json", theme="monokai", line_numbers=False),
            title="[bold green]✅ CONCLUSION JSON OUTPUT[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
    else:
        print("\n=== CONCLUSION JSON ===")
        print(json.dumps(conclusion, indent=2, ensure_ascii=False))


def _save_report(report: ReasoningReport, path: str):
    """Save full reasoning report to a markdown file."""
    lines = [
        f"# 🧠 Reasoning Report\n",
        f"**Question:** {report.question}\n",
        f"**Model:** {report.model}",
        f"**Files Analysed:** {len(report.files)}",
        f"**Total Time:** {report.total_time:.1f}s",
        f"**Predicted Winner:** {report.predicted_winner}",
        f"**Confidence:** {report.confidence*100:.0f}%\n",
        "---\n",
        "## 📁 Files Loaded\n"
    ]
    for f in report.files:
        lines.append(f"- `{f.filename}` ({f.size_chars:,} chars)")

    lines.append("\n---\n")

    for i, stage in enumerate(report.stages, 1):
        lines.append(f"\n## Stage {i}: {stage.emoji} {stage.name} ⏱ {stage.duration:.1f}s\n")
        lines.append(stage.output)
        lines.append("\n---")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()