"""
Analyze haki_ai.rag_engine logs to answer: is the correction retry
actually rescuing good answers, or just delaying an inevitable reject?

Usage:
    python3 analyze_grounding_logs.py /path/to/your/log/file.log

    # or pipe logs in directly:
    tail -n 5000 app.log | python3 analyze_grounding_logs.py -

What it reports:
1. RESCUE RATE: of all answers that failed grounding on the first try,
   what fraction passed after the one-shot correction retry vs. still
   failed and fell back to UNAVAILABLE_MESSAGE.
2. ISSUE BREAKDOWN: which specific check (missing citation, bad case
   citation, bad section number, bad penalty claim) is actually firing
   most often — this tells you where to focus tuning effort, or whether
   a semantic checker like MiniCheck would target a real gap.
3. RAW COUNTS for total attempts logged, so you know the sample size
   before trusting the ratio (don't draw conclusions from <20-30 events).
"""

import sys
import re
from collections import Counter

FOUND_LINE = "GROUNDING_ISSUES_FOUND (attempt 1, retrying)"
PERSIST_LINE = "GROUNDING_ISSUES_PERSIST (attempt 2, giving up)"

# Matches the human-readable issue bullets logged inside issues=[...]
ISSUE_PATTERNS = {
    "missing_citation": re.compile(r"No valid, in-range document or section"),
    "bad_case_citation": re.compile(r"doesn't correspond to.*real case examples"),
    "bad_section_number": re.compile(r"could not be verified in the provided legal text"),
    "bad_penalty_claim": re.compile(r"could not be found\s+verbatim in the provided legal text"),
}


def analyze(lines):
    found = 0
    persisted = 0
    issue_counts = Counter()

    for line in lines:
        if FOUND_LINE in line:
            found += 1
            for label, pattern in ISSUE_PATTERNS.items():
                if pattern.search(line):
                    issue_counts[label] += 1
        elif PERSIST_LINE in line:
            persisted += 1

    rescued = found - persisted
    return found, persisted, rescued, issue_counts


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_grounding_logs.py <logfile|->")
        sys.exit(1)

    source = sys.argv[1]
    lines = sys.stdin if source == "-" else open(source, "r", errors="replace")

    found, persisted, rescued, issue_counts = analyze(lines)

    print("=" * 60)
    print("GROUNDING RETRY ANALYSIS")
    print("=" * 60)
    print(f"Total first-attempt failures (FOUND):        {found}")
    print(f"  -> Rescued by correction retry:             {rescued}")
    print(f"  -> Still failed after retry (PERSIST):      {persisted}")

    if found == 0:
        print("\nNo GROUNDING_ISSUES_FOUND lines detected in this log slice.")
        print("Either the pipeline hasn't hit a failure yet, or these logs")
        print("aren't from rag_engine's logger.")
        return

    rescue_rate = rescued / found * 100
    print(f"\nRescue rate: {rescue_rate:.1f}%  "
          f"({rescued}/{found} first-attempt failures were fixed by retry)")

    if found < 20:
        print("\nNote: sample size is small (<20 events) — treat this ratio")
        print("as a rough signal, not a reliable number yet.")

    print("\n" + "-" * 60)
    print("WHICH CHECK IS FIRING MOST (on first-attempt failures):")
    print("-" * 60)
    if not issue_counts:
        print("Could not parse issue types from these log lines — check that")
        print("your log format includes the full message text, not just level/timestamp.")
    else:
        for label, count in issue_counts.most_common():
            pct = count / found * 100
            print(f"  {label:22s} {count:5d}  ({pct:5.1f}% of first-attempt failures)")

    print("\nInterpretation:")
    print("- High rescue rate + one issue type dominating -> that check may be")
    print("  over-triggering on formatting/chunking noise; consider loosening it")
    print("  or normalizing before comparing, rather than adding MiniCheck yet.")
    print("- Low rescue rate (retry rarely helps) -> the model is genuinely")
    print("  fabricating, not just getting unlucky with chunk boundaries; a")
    print("  semantic checker like MiniCheck is more likely to be worth the cost.")
    print("- Very few FOUND events overall -> hallucination may already be rare")
    print("  enough that MiniCheck's overhead isn't justified yet either way.")


if __name__ == "__main__":
    main()
