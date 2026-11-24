# Deployment Cost Analysis

**ClariCase: Automated consumer-complaint routing and solution**


---

## Purpose and scope

This document establishes the cost of operating ClariCase across three
deployment profiles: a minimum viable deployment, a single-tenant production
system within one mid-size US bank, and a multi-tenant software service. It
concludes with the commercial considerations that sit outside infrastructure.

It is the third in a series. [`market_analysis.md`](market_analysis.md) sized
the category and identified distribution as the principal barrier.
[`unit_economics.md`](unit_economics.md) priced marginal compute and expressly
excluded hosting. This document addresses hosting, scale and the surrounding
business case.

**Basis of figures.** Throughput, model size and prompt dimensions are
measured within this repository and are reproducible. Bank complaint volumes
are drawn from the CFPB Consumer Complaint Database API for calendar 2025.
Cloud and compliance rates are list prices taken from vendor documentation and
third-party comparison sources; they vary by region and provider and should be
confirmed against a quotation. Token counts for the agentic layer are
estimated from the prompt builders rather than metered from live traffic.

---

## Executive summary

| Measure | Value |
|---|---|
| Monthly cost, minimum viable deployment | **$25–30** |
| Capacity of that deployment | **21.9 million complaints per day** |
| Annual cost at full US written-complaint volume | **Under $1,000** |
| Marginal infrastructure cost per additional tenant | **Approximately zero** |
| Largest operating cost | LLM inference, ~10,000× the classifier it supports |
| Principal barrier to revenue | SOC 2 certification: **$15–25k, 6–15 months** |

**Infrastructure cost is not a constraint on this product at any volume it is
likely to encounter.** A single CPU core exceeds the entire United States'
annual written-complaint volume by four orders of magnitude, and the cost of
keeping that core available is approximately $25 per month.

Three conclusions follow:

1. The deployment decision turns on **idle pricing**, not compute capacity,
   because the serving tier is unoccupied 99.99% of the time.
2. The commercially correct model is **multi-tenant software**, because the
   infrastructure floor does not grow with customers.
3. The constraint on revenue is **certification and procurement**, not
   engineering or cost. The product has been operational since session 07 and
   could not be purchased by a regulated institution today.

---

## 1. Deployment options and recommendation

### 1.1 System under consideration

Phase 1 (`platform/`) constitutes the deployable unit:

```
ingress → router (FastAPI) → svm-api (FastAPI, 66 MB model in memory) → PostgreSQL 16
```

The complete product additionally comprises the Streamlit intake application,
object storage for model artefacts, and the agentic issue classifier, which
calls GPT-OSS 120B via Groq.

| Component | Resource profile |
|---|---|
| `router` | 0.25 vCPU, 0.5 GB — stateless proxy |
| `svm-api` | 1 vCPU, 2 GB — 66 MB model resident, 13.6 s load time |
| `postgres` | Smallest burstable tier |
| Object storage | ~90 MB of artefacts |
| Agentic layer | External API; no infrastructure |

### 1.2 The determining constraint

The serving tier is idle for substantially all of its operating life. At a
measured 4.0 ms per complaint:

| Annual volume | CPU-seconds consumed | Duty cycle |
|---|---:|---:|
| 50,000 (pilot) | 200 | 0.0006% |
| 500,000 (large bank) | 2,000 | 0.0063% |
| 1,000,000 (all US written complaints) | 4,000 | **0.0127%** |

Every written complaint filed with the CFPB across US financial services in a
year represents approximately **1.1 vCPU-hours** of computation.

The operative question is therefore not how much compute to procure, but the
least expensive means of keeping a 66 MB model warm and reachable. The
expenditure purchases availability, not processing. Two consequences follow,
both of which invert conventional sizing guidance:

**Idle rate governs total cost.** At a duty cycle of 0.0127%, substantially
every billed second is idle.

**Scale-to-zero is unavailable.** Cloud Run and Container Apps can both
reduce to zero instances, which would render compute effectively free.
However, the model requires **13.6 seconds to load**, and a cold start of that
duration is unacceptable on an interactive intake form. A minimum of one warm
replica is mandatory, and this single requirement exceeds the cost of every
other infrastructure decision combined.

### 1.3 Comparison of providers

| | AWS | Azure | GCP |
|---|---|---|---|
| Container service | ECS Fargate | Container Apps | Cloud Run |
| vCPU-hour, active | $0.04048 | $0.0864 | $0.036–0.065 |
| vCPU-hour, **idle** | Billed as active | **$0.0108** | Billed as active |
| Scale-to-zero | Not available | Available, unusable here | Available, unusable here |
| Managed PostgreSQL, smallest tier | db.t4g.micro, ~$12 | B1ms, ~$12–15 | db-f1-micro, ~$11 |
| Ingress | ALB, ~$16–22 | Included | Included |

### 1.4 Recommendation

**Azure Container Apps.**

The recommendation is counterintuitive: Azure carries the highest active
compute rate of the three, more than double AWS. It is nonetheless the correct
choice, because it is the only provider of the three billing a distinct idle
rate, approximately eight times lower than its active rate. On a workload
idle 99.99% of the time, the idle rate is the only rate of consequence.

**AWS is an acceptable alternative** where existing expertise or credits
apply. The approximate $50 monthly difference does not justify a migration or
a learning curve.

**GCP offers no distinguishing advantage** for this workload and sits between
the two on cost.

---

## 2. Minimum viable deployment: cost and capacity

### 2.1 Fixed monthly cost

| Line item | AWS | Azure | GCP |
|---|---:|---:|---:|
| Containers (1.25 vCPU, 2.5 GB, continuously warm) | ~$45 | **~$10** | ~$33–59 |
| Managed PostgreSQL | ~$12 | ~$12–15 | ~$11 |
| Ingress / load balancer | ~$16–22 | Included | Included |
| Object storage | <$1 | <$1 | <$1 |
| **Monthly total** | **~$75–80** | **~$25–30** | **~$45–70** |

### 2.2 Capacity

| Measure | Value |
|---|---:|
| Complaints per second (one core, measured) | 253 |
| Complaints per day | **21,900,000** |
| Time to process the full US annual written-complaint volume | **66 minutes** |

The minimum deployment is not an introductory tier to be outgrown. A single
core exceeds the United States' entire annual written-complaint volume by four
orders of magnitude. PostgreSQL connection limits, network throughput and
memory will constrain the system long before the classifier does.

### 2.3 Variable cost: the agentic layer

The agentic issue classifier issues a classification call and, where
confidence is insufficient, a verification call. Derived from the prompt
builders — eight few-shot examples of up to 1,200 characters, the candidate
issue list for the predicted team, and the complaint text — each complaint
consumes approximately 2,900 input tokens, rising to approximately 4,100 where
verification is invoked.

At Groq's published rate for `gpt-oss-120b` ($0.15 per million input tokens,
$0.60 per million output):

| Annual volume | Agentic layer | Classifier compute |
|---|---:|---:|
| 50,000 | $22 – $31 | $0.003 |
| 500,000 | $224 – $314 | $0.028 |
| 1,000,000 | $447 – $627 | $0.056 |

The LLM layer costs approximately **10,000 times the classifier it supports**
and is the only line item that scales with complaint volume. All remaining
costs constitute a fixed floor.

### 2.4 Total cost of operation

| | Pilot (50k) | Large bank (500k) | All US (1M) |
|---|---:|---:|---:|
| Infrastructure | ~$340 | ~$340 | ~$340 |
| Agentic layer | $22–31 | $224–314 | $447–627 |
| Classifier compute | $0.003 | $0.03 | $0.06 |
| **Annual total** | **~$370** | **~$600** | **~$900–970** |

---

## 3. Production deployment

### 3.1 Observed complaint volumes

Complaints filed with the CFPB during calendar 2025, retrieved from the
Consumer Complaint Database API:

| Institution | 2025 complaints |
|---|---:|
| Citibank | 19,792 |
| Wells Fargo | 19,468 |
| Bank of America | 19,168 |
| U.S. Bancorp | 5,586 |
| Truist | 5,135 |
| PNC | 3,198 |
| Citizens Financial | 2,017 |
| Fifth Third | 1,697 |
| Comerica | 1,245 |
| M&T Bank | 1,142 |
| Regions Financial | 1,113 |
| KeyCorp | 804 |

A mid-size US regional bank files between **800 and 2,000** CFPB complaints
annually. The four largest institutions file approximately 19,000.

**These figures represent complaints escalated to the federal regulator
only.** A bank's internal complaint volume — matters resolved in branch, by
telephone or through its own portal — is a multiple of this figure, and that
multiple is not publicly disclosed. The volumes above therefore constitute a
floor rather than the addressable volume, and the sizing below is expressed as
a range.

### 3.2 Option A: single-tenant deployment

| | Mid-size bank |
|---|---|
| Internal complaint volume | 10,000 – 200,000 annually |
| Infrastructure | ~$340 per year |
| Agentic layer | $5 – $125 per year |
| **Total operating cost** | **Under $500 per year** |

Operating cost is immaterial against a single support salary. The expenditure
a bank incurs is not in running the software but in integration, security
review and accountability for misrouted complaints.

Regulated institutions will generally request this model, as it keeps data
within their own estate. It nonetheless carries the least favourable
economics: each deployment is bespoke, and revenue does not scale against a
cost base that was already negligible.

### 3.3 Option B: multi-tenant software service

Because the serving tier is idle 99.99% of the time, **a single warm replica
serves several hundred tenants**. The infrastructure floor does not grow with
the customer base.

| Tenants | Infrastructure | Agentic layer (50k complaints each) | Annual total |
|---:|---:|---:|---:|
| 1 | $340 | $22–31 | ~$370 |
| 10 | $340 | $220–310 | ~$600 |
| 50 | ~$700 | $1,100–1,550 | ~$2,000 |
| 200 | ~$1,400 | $4,400–6,200 | ~$7,000 |

The marginal infrastructure cost of each additional tenant is approximately
zero. Only the LLM layer scales, and it scales with complaint volume rather
than tenant count.

**Gross margin** at 50 tenants of 50,000 complaints each, at 61.5% routing
coverage:

| Price per routed complaint | Annual revenue | Annual cost | Gross margin |
|---:|---:|---:|---:|
| $0.05 | $77,000 | ~$2,000 | **97%** |
| $0.10 | $154,000 | ~$2,000 | **99%** |
| $0.25 | $384,000 | ~$2,000 | **99%** |

At five cents per complaint — one two-hundred-seventieth of a human-handled
contact — gross margin remains above 97%.

### 3.4 Recommendation

**Build as multi-tenant software; offer single-tenant deployment as an
enterprise tier at a substantial premium.**

The cost structure is inherently multi-tenant. Single-tenant deployment
converts a near-zero marginal cost into a bespoke engineering engagement, and
the enterprise tier should be priced to recover that engagement rather than to
match the standard rate.

---

## 4. Commercial considerations beyond infrastructure

Infrastructure is the least significant cost this business carries. Each item
below exceeds the entire hosting expenditure.

### 4.1 Certification is the critical path to revenue

**SOC 2 Type II: $15,000–25,000 and 6–15 months.** First-year cost for a team
under 15 people is approximately $15,000; for 15–30 people, $20,000–22,000.
The observation window alone is typically six months. Bank vendor-risk
functions treat SOC 2 Type II as an entry requirement. Type I is accepted as a
bridge at seed stage where paired with a public commitment to Type II.

This timeline, rather than engineering, determines time to first revenue. The
product has been operational since session 07; a regulated institution could
not purchase it today at any price.

Of the competitors surveyed in `market_analysis.md`, only Ada publishes SOC 2
Type II alongside GDPR and HIPAA; Forethought's published compliance footprint
is narrower. Certification is therefore a point of differentiation in this
market, not solely a cost of entry.

### 4.2 Pricing position

Cost-plus pricing is inapplicable where cost is effectively zero. Pricing is
value-based and bounded by the work displaced.

| Reference point | Rate |
|---|---|
| Human-handled contact | $13.50 |
| Zendesk AI, per automated resolution | $1.50–2.00 |
| Self-service contact | $1.84 |
| **ClariCase, cost per routed complaint** | **~$0.0000015** |

The structural difficulty is that **Zendesk prices a resolution, whereas
ClariCase currently produces a classification**. A routing decision is an
input to work that someone else must still perform, which is why routing
appears as a platform feature rather than a billable line item. The product
roadmap — route, guide, resolve, escalate — represents the transition from the
unbillable portion of that distinction to the billable portion, and is the
most commercially significant item on it.

### 4.3 Risks

**Distribution.** Zendesk and Salesforce own the ticketing systems in which
complaints already reside, and Zendesk acquired Forethought in March 2026.
Selling a routing layer into that environment is selling a feature. This
barrier was identified in `market_analysis.md` and remains unchanged.

**Closed data source.** The CFPB has ceased publishing complaint narratives.
The corpus is historical and cannot grow from that source, so a deployed model
will in time encounter text drawn from a different distribution than that on
which it was trained.

**Label quality bounds accuracy.** 568,050 complaints with identical text were
filed under conflicting categories. A proportion of residual error is
attributable to the data rather than the model, and no model resolves it.

**Unmeasured cost of the resolution layer.** An LLM drafting a resolution is
materially more expensive per complaint than a classifier. The agentic layer
already costs 10,000 times the classifier; a resolution agent will cost more
again. This figure will determine software gross margin and has not yet been
measured.

### 4.4 Hardware implications of the transformer models

Improved Model 3 (FLAN-T5 with LoRA) and the Ettin encoder are not currently
served. Were either to replace the linear model, inference would require GPU:

| Instance | Monthly cost, one instance continuously available |
|---|---:|
| AWS `g4dn.xlarge` (T4) | ~$380 |
| Azure `NC4as_T4_v3` | ~$390 |
| GCP `g2-standard-4` (L4) | ~$520 |

This represents **10–15 times the entire current stack**, for models that at
most recent measurement did not exceed the linear model on the served metric.
The CPU-only architecture is not a limitation requiring remediation; it is the
reason the product is close to free to operate, and should be preserved.

---

## 5. Limitations

- **Personnel cost exceeds every figure in this document.** One engineer costs
  more than a century of the infrastructure described.
- Egress, backup, multi-availability-zone and disaster recovery are excluded.
  Multi-AZ approximately doubles managed-PostgreSQL cost on AWS.
- Streamlit Cloud and Supabase are costed at free tier only.
- Token counts in section 2.3 are estimated from prompt structure rather than
  metered. This is the dominant variable cost and the figure most warranting
  replacement by measurement.
- Cloud rates are list prices and vary by region and source. Committed-use
  discounts reduce compute by 30–70%, relevant only once the fixed floor
  dominates.
- Internal bank complaint volumes are not publicly disclosed; section 3 is
  expressed as a range accordingly.

---

## Sources

| Reference | |
|---|---|
| AWS Fargate pricing | https://aws.amazon.com/fargate//pricing |
| Amazon ECS pricing | https://aws.amazon.com/ecs/pricing/ |
| Container service price comparison | https://dev.to/wimadev/comparing-prices-aws-fargate-vs-azure-container-apps-vs-google-cloud-run-3p4e |
| PostgreSQL hosting comparison | https://www.bytebase.com/blog/postgres-hosting-options-pricing-comparison/ |
| Azure Flexible Server pricing | https://www.bytebase.com/dbcost/azure-flexible-server-pricing/ |
| GroqCloud model pricing | https://www.morphllm.com/groq-pricing |
| SOC 2 audit cost | https://drata.com/grc-central/soc-2/how-much-does-a-soc-2-audit-cost |
| SOC 2 cost, 2026 | https://www.scrut.io/soc-2/cost-of-soc-2-audit |
| SOC 2 certification cost | https://sprinto.com/soc-2/certification-cost/ |
| Bank complaint volumes | CFPB Consumer Complaint Database API, calendar 2025 |
