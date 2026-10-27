# MahoRAGa — Monetization & Business-Model Strategy

_Working document. Goal: develop income streams and assess multiple business models (subscriptions, microtransactions, etc.) for the MahoRAGa / Project Aletheia RAG system._

---

## 1. What is being sold (product reality check)

MahoRAGa is an **autonomous, citation-grounded, multi-agent RAG system for AI research literature**. Concretely, the monetizable assets are:

- **The running application** — a Gradio UI + FastAPI backend that answers research questions with cited, hallucination-audited answers over a corpus of papers.
- **The ingestion pipeline** — PDF → structure-aware chunks → three-layer memory (Neo4j graph + FAISS dense + BM25 lexical), plus an autonomous arXiv watcher.
- **The orchestration IP** — the 6-node LangGraph pipeline (router → retriever → evaluator → external-search → AttentionRAG pruner → generator+critic), the 100-point evidence rubric, and RRF fusion.
- **Deployment flexibility** — runs fully local (Ollama/qwen) for privacy, or against OpenAI/Anthropic for quality. This "bring-your-own-LLM / self-host" angle is a genuine differentiator vs. closed SaaS.

### Who would pay (candidate segments)
| Segment | Pain it solves | Willingness to pay |
|---|---|---|
| Individual researchers / PhD students | Literature review, staying current, grounded answers | Low–moderate ($) |
| Research labs / university groups | Shared lab knowledge base, reproducible citations | Moderate ($$) |
| AI/ML startups & R&D teams | Private corpus Q&A without sending data to OpenAI | High ($$$) |
| Enterprises (pharma, legal, finance R&D) | On-prem, auditable, zero-leakage document intelligence | Highest ($$$$) |
| Developers | Want the orchestration engine as a library/API | Moderate ($$) |

**Key strategic fact:** the self-hostable + local-LLM property means the highest-value buyers (privacy-sensitive enterprises) are reachable *because* the data never has to leave their infrastructure. That should anchor the pricing strategy.

---

## 2. Business Model A — Subscriptions (SaaS tiers)

**Fit: STRONG (recommended primary model).** RAG-over-your-papers is a recurring-use workflow; usage recurs monthly, so recurring revenue matches value delivered.

### Proposed tiers

| Tier | Price (indicative) | Target | Limits / features |
|---|---|---|---|
| **Free / Community** | $0 | Individuals, trial, OSS goodwill | Self-host only OR hosted with hard caps: ~50 queries/mo, 1 corpus, ≤25 PDFs, offline/local-LLM answers, community support. No external-API fallback. |
| **Pro** | $15–25 / user / mo | Serious individual researchers | Hosted, ~1,500 queries/mo, 5 corpora, ≤500 PDFs, cloud-LLM answers (OpenAI/Anthropic), external academic-API fallback (Crossref/OpenAlex/S2), citation export, priority queue. |
| **Team / Lab** | $40–60 / user / mo (min 3 seats) | Research groups | Shared corpora, role-based access, higher caps (~5k queries/user), audit log (the `query_audits` table already exists), Slack/Notion export, usage dashboard. |
| **Enterprise / Self-Managed** | Custom ($20k–150k / yr) | Privacy-sensitive orgs | On-prem or VPC deploy, SSO/SAML, no data egress, dedicated support + SLA, custom ingestion connectors, the whole thing behind their firewall with their own Ollama/vLLM. |

### Why tiering works for this product
- The **evidence rubric + external-search fallback** is a natural paywall line: free users get internal-corpus-only answers; paid users unlock the async Crossref/OpenAlex/Semantic Scholar fallback that fires when internal evidence scores < 80.
- **Local vs. cloud LLM** is a natural quality axis: free = local Ollama (free inference, lower quality); paid = cloud models (you pass through or mark up the token cost).
- Caps map directly to **real marginal cost** (embeddings, LLM tokens, storage), so tiers are cost-defensible, not arbitrary.

### Unit-economics caveat (must model before pricing)
Cloud-LLM answers cost real money per query (generator + critic = ≥2 LLM calls, plus optional regeneration = up to 3; plus external API calls). A Pro query with regeneration can be 3–4 LLM round-trips. **Price the paid tiers off measured per-query cost × expected monthly volume, with margin** — the `pipeline_cost` field already tracked per query is the exact telemetry needed to do this. Local-LLM (Ollama) answers are ~free compute, which is why the Free tier should default to local.

### Risks
- Churn if the corpus is small/static (one-time literature review, then cancel). Mitigation: the **arXiv watcher** creates recurring value (keeps the corpus fresh daily → ongoing reason to stay subscribed).
- Hosting cost of idle paid seats. Mitigation: usage-based component (see Model C) layered on top.

---

_Further models (microtransactions / usage-based, open-core, API/usage metering, marketplace, services) are assessed in the sections below, appended in later passes._

---

## 3. Business Model B — Usage-based / Microtransactions (credits)

**Fit: STRONG as a LAYER on top of subscriptions, WEAK as the sole model.**

The natural "unit" in this app is a **query** (and a secondary unit, an **ingested document**), because both carry real marginal cost (LLM tokens, embeddings, storage). That makes a credit/metering model clean to implement and honest to price.

### Mechanics
- **Credit packs** (prepaid microtransactions): e.g. 500 credits = $5, 3,000 = $25. 1 credit ≈ 1 internal-corpus query; a cloud-LLM query with critic + regeneration costs more credits; an external-API-fallback query (Crossref/OpenAlex/S2) costs more again; ingesting a PDF costs credits (embeddings + extraction).
- **Metered overage** on subscription tiers: included monthly quota, then $X per 1k extra queries — the cleanest way to stop idle-seat loss and capture heavy users.
- **Priced off telemetry you already have:** `pipeline_cost` (per-query LLM spend) + node counts in `query_audits` give exact cost per query, so credit cost = measured cost × margin. No guessing.

### Where microtransactions specifically fit
- **Pay-per-deep-report:** a premium "exhaustive literature review" action that deliberately runs more external-search iterations + a larger corpus sweep → charge a flat micro-fee ($0.50–$2) per report. This is the strongest microtransaction candidate: discrete, high-perceived-value, bounded cost.
- **Pay-per-corpus-refresh:** trigger the arXiv watcher to harvest + ingest N new papers on demand → per-run fee.
- **Export unlocks:** one-off fee to export a cited report to PDF/DOCX/BibTeX (low cost to you, clear value).

### Pros / Cons
- ✅ Aligns price with cost and value; no "wasted subscription" feeling; great for infrequent users.
- ✅ Converts the real variable cost (LLM tokens) into a revenue line instead of a margin leak.
- ❌ Credit UX creates purchase friction and "metering anxiety" — users ration usage, which suppresses engagement. **Do not make it the only model.**
- ❌ Hard to forecast revenue vs. subscriptions.

**Recommendation:** hybrid — subscription for the base (predictable), metered overage + a couple of discrete microtransactions (deep report, bulk ingest, export) for upside. This is the standard modern SaaS pattern (seat + usage).

---

## 4. Business Model C — Open-Core / Open-Source

**Fit: STRONG and strategically important — this is likely already an OSS-flavored project (MIT-licensed skill lineage, local-LLM support).**

### Structure
- **Open core (free, self-host):** the pipeline, three-layer memory, basic UI — MIT/Apache. Drives adoption, trust, contributor funnel, and is the on-ramp for privacy-sensitive buyers who must audit the code.
- **Commercial layer (paid):** the things enterprises need that individuals don't —
  - SSO/SAML, RBAC, multi-tenant isolation
  - Audit/compliance exports, data-retention controls
  - Managed hosting + autoscaling
  - Priority connectors (SharePoint, Confluence, S3, Zotero, internal wikis)
  - Support + SLA + indemnification
  - Advanced retrieval (true GDS PageRank, cross-encoder rerank at scale)

### Monetization paths within open-core
1. **Open-core + hosted SaaS** (sell the convenience of not self-hosting) → ties back to Model A.
2. **Open-core + enterprise license** (sell the proprietary enterprise modules) → highest ACV.
3. **"Managed open source"** (they self-host, you operate/support it under contract).

### Pros / Cons
- ✅ Adoption flywheel + credibility; the local-LLM/self-host story is only believable if the core is open.
- ✅ Lets the highest-value (privacy) segment buy at all — they can inspect the code.
- ❌ Must draw the open/paid line carefully; give away too much → no upsell, too little → no adoption.
- ❌ Requires real support/ops investment to monetize enterprise.

**Recommendation:** adopt open-core as the overarching GTM frame, with Model A (subscriptions/hosted) and the enterprise license as the two revenue engines on top of it.

---

_Further models (API/usage metering for developers, marketplace, services/consulting) and a final comparison + recommendation are appended in later passes._

---

## 5. Business Model D — Developer API / Usage Metering

**Fit: MODERATE–STRONG (a second product, lower effort to expose).**

The FastAPI backend already exposes `/api/v1/query`, `/ingest`, `/corpus`, `/graph`, `/metrics`. Packaging that as a **metered developer API** turns the orchestration engine itself into a product for people who want RAG-as-a-service in their own apps.

### Mechanics
- API keys + per-request metering (priced per query / per 1k tokens / per ingested doc), billed monthly or against prepaid credits (ties to Model B).
- Free dev tier (low rate limit) → paid tiers by throughput + rate limit + SLA.
- Sell the *differentiators* as API features: citation-grounded output, the evidence rubric score in the response, hallucination-audit verdict, RRF-fused retrieval.

### Pros / Cons
- ✅ Reuses existing endpoints; developers integrate once and generate recurring usage.
- ✅ Natural fit with usage metering already justified by `pipeline_cost`.
- ❌ Competes with raw LLM APIs + generic RAG frameworks (LangChain, LlamaIndex) — must win on "grounded + audited + batteries-included," not on being another wrapper.
- ❌ Needs hardening the report already flagged (rate limiting, auth, tenant isolation) before it is safe to expose publicly.

**Recommendation:** viable as a *second* revenue line once the core SaaS proves demand; do not lead with it.

---

## 6. Business Model E — Marketplace / Curated Content

**Fit: WEAK–MODERATE (optional, later-stage, some legal risk).**

- **Curated corpora / "knowledge packs":** pre-ingested, maintained topic bundles (e.g. "RAG & retrieval 2023–2026," "LLM alignment," "diffusion models") users can subscribe to or buy once. The arXiv watcher makes maintaining these cheap.
- **Template/prompt/connector marketplace:** third parties publish ingestion connectors or domain prompt packs; platform takes a cut.

### Cons / cautions
- ⚠️ **Copyright:** redistributing ingested paper content (not just links/abstracts) is a licensing minefield. Safer to sell *curation + the maintained index/metadata* and let the user supply/fetch the PDFs, or restrict to open-access (arXiv, OA journals).
- ❌ Marketplace needs a supply side (network effects) — not viable until there's a user base.

**Recommendation:** park until there is scale; if pursued, limit to open-access content + curation value to avoid copyright exposure.

---

## 7. Business Model F — Services, Consulting & Academic Licensing

**Fit: STRONG for early revenue / bootstrapping (high touch, high margin, immediate).**

- **Deployment & integration services:** paid setup of on-prem/VPC installs, custom ingestion connectors to a client's document store, Neo4j/vLLM tuning. Natural companion to the enterprise self-managed tier.
- **Custom pipeline work:** bespoke nodes (domain extractors, custom rubrics) billed as projects.
- **Academic / institutional site license:** a university library or department pays a flat annual fee for unlimited internal seats (universities strongly prefer a fixed site license over per-seat/metered). Good fit given the research-literature focus.
- **Support & SLA contracts:** recurring, attaches to open-core/self-host users.

### Pros / Cons
- ✅ Immediate cash, validates enterprise demand, funds product dev before SaaS scales.
- ✅ Academic site licenses fit the exact buyer the product is designed for.
- ❌ Doesn't scale linearly (people-bound); keep it as a funnel into product revenue, not the end state.

**Recommendation:** use services + academic site licenses as the **bootstrap revenue** that funds the subscription/open-core build-out, then let product revenue overtake it.

---

_Final comparison matrix, phased rollout recommendation, and required product/engineering prerequisites are appended in the final pass._

---

## 8. Comparison matrix

| Model | Revenue predictability | Margin | Effort to launch | Scalability | Fit for THIS product | Verdict |
|---|---|---|---|---|---|---|
| A. Subscriptions (tiers) | High | Med–High | Medium | High | ★★★★★ | **Primary** |
| B. Usage / microtransactions | Low–Med | High | Low–Med | High | ★★★★☆ | **Layer on A** |
| C. Open-core / OSS | Med (via A/F) | Med | Medium | High | ★★★★★ | **GTM frame** |
| D. Developer API metering | Med | High | Med (needs hardening) | High | ★★★☆☆ | Phase 2 |
| E. Marketplace / content | Low | Med | High | Med | ★★☆☆☆ | Later / optional |
| F. Services + academic license | High (contract) | High | Low | Low | ★★★★☆ | **Bootstrap** |

## 9. Recommended strategy — phased

**Phase 0 (now → validate): bootstrap with services + academic site licenses (F).**
High-touch deploys and university/lab site licenses bring immediate cash and prove enterprise/academic demand, funding the build-out. Keep the core open (C) to drive adoption and trust.

**Phase 1 (core product): open-core (C) + subscription SaaS (A), with usage overage + a few microtransactions (B).**
- Free/local tier for adoption; Pro/Team hosted tiers for revenue.
- Metered overage + discrete microtransactions (deep report, bulk ingest, export) for upside.
- Enterprise self-managed tier (on-prem, SSO, no egress) as the high-ACV engine — this is where the self-host/local-LLM differentiator pays off.

**Phase 2 (expand): developer API (D)** once SaaS demand is proven and the backend is hardened; **marketplace (E)** only at scale and only for open-access content.

### The pricing North Star
Everything is anchored to **measured per-query cost** (`pipeline_cost` + `query_audits` node counts). Local-LLM answers ≈ free → anchor the free tier; cloud-LLM + external-API answers cost real money → paywall + meter them. This keeps every tier/credit cost-defensible.

## 10. Engineering prerequisites before charging money

From the earlier code review (`docs/code-issues-report.md`), these must exist before monetizing:

1. **Metering/billing plumbing** — `pipeline_cost` is tracked per query but not aggregated per tenant/API-key; need per-account usage accounting + a billing integration (Stripe).
2. **Auth & multi-tenancy** — API keys, per-tenant isolation, SSO/SAML for enterprise; the backend currently has none (CORS was wildcard, now pinned — Issue 22 fixed).
3. **Rate limiting & quotas** — enforce tier caps; none today.
4. **Cost controls** — cap LLM spend per request/tenant; the generator can fire up to 3 LLM calls (Issue 8 re-audit) — must be bounded per plan.
5. **Reliability** — the graph-layer crash (Issue 37) and others are fixed; keep the suite green (currently 53 passing) as the quality gate buyers will expect.
6. **Data/privacy posture** — document the zero-egress self-host story formally; it is the core enterprise selling point.

## 11. Bottom line

- **Lead model:** subscription tiers (A) inside an **open-core** frame (C), with **usage-based overage + microtransactions** (B) layered for upside.
- **Bootstrap it** with services + academic site licenses (F) for immediate revenue.
- **Expand later** into a developer API (D); treat a content marketplace (E) as optional and copyright-constrained.
- **Differentiator to price around:** self-hostable, local-LLM-capable, citation-grounded + hallucination-audited RAG → sell privacy + trust to the highest-value (enterprise/academic) buyers, and convenience (hosted) to everyone else.
- **Prerequisite:** build metering/auth/rate-limiting/cost-caps first; all pricing is anchored to the per-query cost telemetry the app already records.



