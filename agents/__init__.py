"""
agents/ — Multi-agent reasoning system for the NOVA Reasoning System.

Each agent is an instance of Qwen3-72B with a distinct system prompt.
The system prompt is the ONLY thing that differentiates the agents.

Agents:
  generator.py           — Main reasoning agent (12-step free exploration).
  adversarial_critic.py  — Finds the weakest point in the current proof sketch.
  devils_advocate.py     — Argues the hypothesis is unnecessary or already known.
  synthesizer.py         — Manages the round, tags objections, checks termination.
  constraint_relaxer.py  — Proposes weakened/negated/replaced assumption variants.
  analogy_agent.py       — Finds cross-domain structural parallels via corpus retrieval.

CRITICAL: No agent system prompt contains any reference to the target result.
The only steering is the problem prompt provided by the user at runtime.
"""
