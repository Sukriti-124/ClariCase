# Unit economics and pricing

What it costs to route a complaint, what the market charges for comparable
work, and what the gap between those two numbers means for how this product
could be sold. Compiled November 2026.

Follows [`market_analysis.md`](market_analysis.md), which sized the category
and declined to put a figure on the triage step. This closes that gap from the
cost side, using the project's own measurements rather than vendor estimates.

**On the sources.** Compute figures are measured in this repository and
reproducible. Competitor pricing comes from vendor-adjacent comparison blogs
that cite one another; those figures are directional and should be checked
against a sales quote before being relied on.

---

## 1. What it costs us to route one complaint

Measured on the served model (Improved Model 1, `docs/metrics/improved_model_1/model.joblib`),
single core, warm process:

| | |
|---|---|
| Inference time per complaint | **4.0 ms** |
| Throughput | 253 complaints/sec/core |
| Model load | 13.6 s, once per process |

At a commodity cloud rate of roughly $0.05 per vCPU-hour, one core handles
about 911,000 complaints an hour. That puts marginal compute at **under $0.10
per million complaints routed**.

For scale: every narrative complaint filed with the CFPB across US financial
services in a year — about 1M — could be routed for roughly five cents of CPU.

The alternative architectures the team evaluated are more expensive but still
negligible: Jev costs about $0.00006 per complaint at 1,418 tokens each, or
roughly $60 per million. That is a thousand times the linear model's cost and
still immaterial next to anything below.

**Marginal cost is not the constraint on this product.**

## 2. What the market charges

| | Rate |
|---|---|
| Human-handled contact | $13.50 (NA enterprise $13–18) |
| Zendesk AI agent, per automated resolution | ~$1.50 committed, ~$2.00 pay-as-you-go |
| Zendesk Copilot, per agent | $50/month, billed annually |
| Self-service contact | $1.84 |
| ClariCase marginal compute | **$0.0000001** |

Zendesk moved to outcome pricing: AI agents bill per *automated resolution* —
a request closed without human escalation — rather than per seat. Since May
2026 only "Verified Resolutions" are billed, and since January 2026 overages
bill automatically with no cap.

## 3. The gap, and what it means

Between what routing costs to run and what the market charges for automation
there are roughly **seven orders of magnitude**. Two conclusions follow.

**Pricing here is entirely value-based.** Cost-plus is meaningless when cost is
effectively zero. The question is not what the inference costs but what the
displaced work is worth, which puts the ceiling near the human baseline and
the floor wherever a competitor sets it.

**Zendesk prices an outcome; we currently produce a classification.** A
resolution is a billable event because something finished. A routing decision
is an input to work someone else still has to do, which is why routing shows
up as a platform feature rather than a line item. This is the same conclusion
`market_analysis.md` reached from the distribution side, arrived at
independently from pricing.

The project's own roadmap — route, guide, resolve, escalate — is the move from
the unbillable half of that line to the billable half. The resolution layer is
not an enhancement to the router; commercially, it is the part that can carry
a price.

## 4. One structural advantage worth noting

Zendesk bills only **Verified Resolutions**, having introduced verification
tiers in May 2026. The industry is converging on the idea that an automated
outcome should be billed only when it can be shown to be correct.

ClariCase's confidence threshold is that principle applied one step earlier:
the system declines to act below 94.8% precision and escalates instead. A
product that already refuses to guess has the mechanism an outcome-priced
contract needs, because the thing being charged for is the thing it was
willing to stand behind. Competitors reporting accuracy alone have no
equivalent — accuracy is computed over whatever the system chose to decide.

## 5. What this does not establish

- No per-customer volume analysis, so no revenue estimate per account.
- Compute cost excludes hosting, storage, monitoring and support, which
  dominate at this scale and are not measured here.
- Competitor rates are third-party reported and vary between sources
  ($1.20–$2.00 per resolution).
- The resolution layer is not built, so its cost per complaint is unknown —
  an LLM drafting a resolution is far more expensive than a linear classifier,
  and that is the number that will actually matter.

---

## Sources

- [Intercom — AI customer service agent pricing comparison](https://www.intercom.com/learning-center/ai-customer-service-agent-pricing-comparison)
- [eesel.ai — understanding Zendesk AI pay-per-resolution pricing](https://www.eesel.ai/blog/understanding-zendesk-ai-pricing-a-complete-pay-per-resolution-guide)
- [eesel.ai — Zendesk AI dynamic pricing plan](https://www.eesel.ai/blog/zendesk-ai-dynamic-pricing-plan)
- [Robylon — Zendesk AI pricing per resolution](https://www.robylon.ai/blog/zendesk-ai-pricing-per-resolution)
- [eesel.ai — cost per ticket benchmarks](https://www.eesel.ai/blog/cost-per-ticket-benchmarks)
