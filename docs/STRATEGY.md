# CI strategy — business, competition, positioning, roadmap, risks

Covers Steps 7, 8 and 10 and Parts VIII–XI.

- **Market facts** come from a web research pass dated 2026-10-15. Each carries its
  confidence label; "[R]" means reported in the press but not confirmed by the company.
- **Traction:** none, and none is implied.
- **Unit-economics figures** are assumptions with their inputs shown.

---

## 1. What the evidence changes about the company

The founding brief imagines CI as an individually evolving intelligence. The first
evidence (RESEARCH.md §4, simulation only) says something sharper and less romantic.
**The learning that pays is collective.** A value model fit across many people's outcome
feedback beat a tuned memory heuristic by about 15pp. Each person's own deviation added
1–3pp, and only when they differ a lot and give dense feedback.

Three strategic consequences follow.

1. **A single-user product cannot reach the main effect.** "Nishanth 2.0" alone is a
   sovereign memory with good defaults. The learned layer needs a population.
2. **The population can be served without pooling anyone's context.** The shared model
   learns from 44-dimensional, content-free feature statistics: which *kinds* of item
   helped which *kinds* of task. Federated EM with secure aggregation and differential
   privacy is the honest path to "collective learning without a data grab". That is a
   claim incumbents with ad businesses cannot easily make.
3. **Distribution has to come through other products, not just our own app.** The
   fastest population is developers' users, reached through an MCP server and an SDK. A
   consumer subscription alone is too slow.

---

## 2. Positioning (Part XI, items 1–4)

**One sentence.**
CI is the user-owned context layer that decides what each AI model should know about you
for each task, learns which context actually helps from outcomes across many people
without pooling their data, and works with every major model.

**30 seconds.**
Every assistant now has memory, and every one keeps it inside its own walls. People use
three assistants on average. None of them can carry what they know about you to the
others, and none can show you what they sent or prove they deleted it. CI holds your
context on your side, with consent per source and a receipt for every call. It learns,
across its whole user base, which context improves answers for which kinds of task. Then
it sends each model the smallest set that does the job.

**Two-minute founder pitch.**
- *Problem.* Personal context is the scarcest input to AI, and it is locked in silos.
  - ChatGPT, Claude and Gemini all ship free memory, each fused to its own data:
    Gmail and Photos at Google, connectors at OpenAI.
  - Importing between them means pasting a prompt or uploading a ZIP. No portability
    standard has been adopted.
  - Meanwhile the average US AI user juggles 3.0 general assistants [Menlo 2026].
- *What CI does.*
  - It is a sovereign memory: encrypted per record, consent per source and provider,
    deletion that destroys the key, a receipt showing exactly what reached which model.
  - In front of it sits a learned policy that selects context for each request and
    model.
- *Why now.* MCP is the neutral rail (Linux Foundation AAIF, about 0.5B SDK downloads a
  month), so CI can plug into any assistant.
- *Our first evidence.* It is simulation, and we say so. Learned selection beat a tuned
  strong-memory heuristic by 15 points, and the per-person part is small. So the
  advantage comes from learning across users, done federated, on content-free features.
- *The ask.* Funding to run the real-model headroom test (weeks, under $100 in tokens),
  then a 20-person study. Each step has a pre-registered kill rule.
- *Why it could matter even if models get much better.* Data minimization and user
  ownership do not get cheaper as models improve.

**Technical abstract.**
> We formulate personal context provision for black-box foundation models as budgeted,
> privacy-priced subset selection by a learned policy external to the model. A
> hierarchical Bayesian linear value model over content-free item×task features is fit by
> EM across a population, with each user's deviation constrained to a learned low-rank
> subspace. In two pre-registered simulation studies with planted ground truth (12
> conditions; 200 test users each), the population model exceeded a strong temporal
> memory heuristic tuned on noiseless ground truth by 15–17pp of response quality,
> capturing 72% of oracle headroom, and was non-inferior when relevance was similarity
> alone. Per-user adaptation added 1.7–2.8pp only under strong individual structure and
> dense feedback, cost 0.8pp without it, and lost 1.6pp under sparse feedback. Two method
> defects found by the first study — budget-blind ranking and Thompson-sampling
> exploration cost — are characterized. We propose a headroom-times-capture decomposition
> and a real-model headroom test as the next falsification step.

---

## 3. Business model (Part VIII)

### 3.1 Three businesses, in the order the evidence supports

| business | what is sold | to whom | why this order |
|---|---|---|---|
| **CI Developer Platform** (first) | MCP server + SDK: sovereign memory, consent, receipts, learned selection as a service; usage-priced | developers of AI apps and agents who need memory but not a memory team | fastest route to the population the learned layer needs; Mem0's numbers (186M API calls/quarter by Q3'25; exclusive memory provider for AWS's Agent SDK) show the demand is real and the competition serious |
| **Personal CI** (second) | subscription: your context across every assistant, with receipts and deletion | multi-assistant power users, researchers, professionals | the brand and the trust story; shows the receipts work; feeds the population with consent |
| **Enterprise CI** (third) | governed context layer across many models: egress policy, audit, retention, regional keys | regulated teams adopting several model vendors | highest prices, longest sales cycles; needs the security review and track record first |

### 3.2 Pricing and unit economics (assumptions, inputs shown)

**Token prices, list, USD per million tokens, from official pages checked 2026-10-15:**

| tier | models (input / output) | cache reads |
|---|---|---|
| mid-tier | Claude Sonnet 5.5, GPT-6.1 Sol, Gemini 4 Argon (promotional until 11-30): all $2 / $10 | 0.05–0.1× of input |
| small | Claude Haiku 5.5 and GPT-6 Luna: $0.10 / $0.50 | — |

Personal CI, bundled model usage, assuming one active user makes 300 requests a month:

| item | assumption | cost per month |
|---|---|---|
| personal context sent | 1,200 tokens/request at $2/M | $0.72 |
| rest of the prompt | 1,800 tokens/request; stable system prefix cached at 0.05–0.1× | ≈ $0.60 |
| output | 800 tokens/request at $10/M | $2.40 |
| storage, compute, egress | local-first: mostly on the user's device | ≈ $0.20 |
| **total** | | **≈ $3.9** |

| plan | price | gross margin |
|---|---|---|
| bundled | $20/month | ≈ 80% before support and payment fees |
| bring-your-own-key | $8/month | ≈ 95% |

The sensitivity that matters is heavy users. At 3,000 requests a month the bundled cost is
about $39, so the bundled plan needs fair-use limits or tiering.

**Developer platform.**
- **Pricing:** per active end-user per month plus metered selection calls, positioned
  against Mem0's ($19/$249/enterprise), Zep's ($125/$375) and Supermemory's
  ($19/$100/$399) public tiers.
- **Margin case:** CI never pays for the model call there. Selection is about 1 ms of CPU
  and storage is per user, so infrastructure margin should exceed 85% at scale. The real
  cost line is support and security.

**Market size.** Built bottom-up because no credible top-down figure exists for this
category:

- US adults using AI: 64% of about 260M ≈ 166M.
- Users paying for at least one AI product: 55% of users [Menlo, Jul 2026].
- Multi-homing is the norm: 3.0 assistants per user.
- If 2–5% of paying US multi-assistant users would pay $8–20/month for portable, sovereign
  context, that is ≈ $175M–$1.1B ARR in the US consumer segment alone. The arithmetic is
  166M × 0.55 × 0.02–0.05 × $96–240/yr.
- That is an assumption with a wide band, not a forecast.
- The developer and enterprise markets are larger and harder to size. The memory-layer
  financings below are the only priced signal.

**Retention and switching costs.**
- *For incumbents:* accumulated memory is a switching cost, which is precisely what
  providers have no incentive to make portable.
- *For CI:* the store is exportable by design, so CI's own lock-in has to come from
  quality and trust, not captivity. That is deliberate and should be said out loud.

**Regulatory obligations.**
- GDPR applies unchanged: inferred profiles are personal data, and the EDPB has flagged
  erasure in backups as a weak spot. CI's crypto-shredding addresses the key, but backups
  predating a deletion need KEK rotation.
- EU AI Act Article 50 transparency has applied since 2026-08-02.
- California CPPA and Colorado SB 26-189 ADMT duties start 2027-01-01, and only if CI's
  output drives "significant decisions".
- Companion-chatbot laws in NY and CA SB 243 do not apply unless CI presents as a companion.

---

## 4. Competitive strategy (Part IX)

### 4.1 Landscape (as of 2026-10-15)

| player | what they have | overlap with CI | what they will not do |
|---|---|---|---|
| OpenAI (ChatGPT memory, "Dreaming" rebuild June 2026, Agents SDK memory) | free memory from chats, files and connectors; sources behind answers | high, inside ChatGPT | make memory portable to rivals; no consumer memory API found |
| Anthropic (Claude memory free since Mar 2026; import tool; memory tool and Managed Agents memory stores in the API) | editable memory, sensitive categories off by default, import from rivals | high, inside Claude | export to rivals beyond user-driven copy |
| Google (Personal Intelligence: Gmail, Photos, YouTube, Search; history import) | the deepest first-party data | high, inside Gemini | neutrality; Meta and Google also use AI chats for ads [high] |
| Apple (Siri AI, personal context on device, Gemini-based models) | on-device personal context | medium | cross-provider portability |
| Mem0 ($24M; OpenMemory MCP; "memory passport" pitch; AWS Agent SDK exclusive) | developer memory layer, MCP | **very high**: the closest competitor | (unknown: they could add learned selection) |
| Engram ($98M at ~$600M [R], "learned memory layer", 100× fewer tokens claimed) | enterprise learned memory | **very high** on "learned" | consumer sovereignty, so far |
| Letta, Zep/Graphiti, Supermemory, Cognee, LangMem | agent memory frameworks; temporal graphs (Zep) | high for developers | — |
| AWS AgentCore Memory (GA), Microsoft Foundry memory (beta) | hyperscaler memory primitives | medium-high for enterprise | neutrality across clouds |
| OpenRouter (Stripe agreed to acquire, ~$7.5B [R]), OpenAI Decisions API | routing | low: routing is commoditized | — |

### 4.2 Commoditized and defensible

| commoditizing: do not build a company on these | possibly defensible: must be earned |
|---|---|
| storing memories; vector retrieval; MCP plumbing; model routing; import/export scripts | outcome-learned selection that measurably beats B-class memory, shown on public benchmarks (the "learned memory" claim is crowded and funded, so CI needs published numbers) |
| | a privacy-preserving *population* prior: the more users, the better selection for each, without pooling content (a data network effect that incumbents' ad models make hard to copy credibly) |
| | trust infrastructure: receipts, lineage-aware consent, verifiable deletion, an audited red-team suite |
| | standards position: contributing the consent, receipt and memory interchange format to AAIF/DTI, where no format has been adopted |

Not depending on any provider's goodwill is built in: CI only needs public APIs, or local
models. The design needs no special access.

### 4.3 The honest competitive risk

Mem0 or Engram could add learned selection, and a provider could ship "export your memory
anywhere". CI's durable edges are therefore measured quality (a published benchmark
lead), the federated population prior, and trust. If G1 shows no quality headroom, CI
competes on trust and cost alone, against well-funded memory startups. That is a worse
business and should be valued as one.

---

## 5. Investor materials (Part XI, items 5–10)

### 5.1 Ten-slide deck (outline)

1. **Title.** CI: your context, every model, nothing more than needed.
2. **Problem.**
   - Memory is everywhere and portable nowhere.
   - The average user runs 3.0 assistants.
   - Imports are copy-paste.
   - No one shows you what was sent.
3. **Insight.**
   - Closed models only take tokens, so the personal layer has to live *outside* the
     model: a context compiler.
   - Portability exists only in text: adapters break across model updates; text
     summaries transfer.
4. **Product.** Sovereign memory with receipts, consent per source × provider, and
   deletion that destroys the key, demoed live from this repo.
5. **The learned layer and its evidence.**
   - G0/G0b in simulation: +15pp over a tuned strong-memory baseline, from population
     learning; per-person adaptation +1–3pp.
   - Shown with the failures and the fixes.
6. **Why collective, and why private.** Federated EM on content-free 44-dimensional
   features: a network effect without a data grab.
7. **Market.** Bottom-up consumer band; the developer platform via MCP; competitors named
   (Mem0, Engram, native memory).
8. **Business model.** Developer platform first, Personal CI second, Enterprise third;
   unit economics from current token prices.
9. **Plan with kill rules.**
   - G1a headroom (weeks): stop if frontier headroom < 3pp.
   - G1b capture on public benchmarks; G2 with 20 people; G3 capability.
10. **Team and ask.**
    - Founder's track record of falsification (the alpha-crowding G0 and the edge-search
      protocol).
    - Hires (§5.3); round and milestones (§5.4).

### 5.2 Whitepaper outline

1. Problem and threat model.
2. The interface constraint and its consequences.
3. PCIM.
4. The constrained objective.
5. Meta-learning ≡ learned prior covariance.
6. Temporal belief state.
7. Lineage-aware consent and egress.
8. The G0 series, with deviations.
9. The Headroom × Capture decomposition and G1 design.
10. Federated EM with DP: protocol and accounting.
11. Limitations.
12. Related work [references 1–83].

### 5.3 Research hiring plan (first 6 hires, in order)

| # | role | why |
|---|---|---|
| 1 | ML research engineer: bandits, Bayesian methods, evaluation | runs G1, owns the observatory |
| 2 | security engineer: applied cryptography, LLM red-teaming | the trust claims are only as good as this hire |
| 3 | full-stack product engineer | the Next.js app, connectors, MCP server |
| 4 | privacy and federated-learning researcher | federated EM and DP accounting |
| 5 | learning scientist / HCI researcher | G3 design, growth modes, dependency measurement |
| 6 | developer relations and partnerships | the platform route to a population |

### 5.4 Fundraising strategy

- **Do not raise on simulation alone.**
- **Pre-seed** (about $1–2M, 12 months): G1a + G1b, the MCP server, a 20-person G2.
- **Milestone for seed:** a published capture result on two public benchmarks, plus G2
  data.
- **Investors to target:** technical funds with an AI-infrastructure thesis and the
  patience for a falsification-first plan. Strategic money from a model provider would
  compromise the neutrality story, so take it late or not at all.

### 5.5 Demonstration plan ("Nishanth 2.0")

1. `pcim init` and grants.
2. Ingest notes with an injected instruction; show it stored as data, flagged, and
   withheld.
3. Ask the same question to two providers; show both receipts and the identical packet.
4. Correct a fact; show supersession and the "as of" history.
5. Delete; show the key destroyed and the audit entry.
6. Show the G0b report.

What the demo must not do is imply the learning works on the person yet. It does not, at
n = 1, and the dashboard's metrics page says so.

---

## 6. Execution roadmap (Part X)

| horizon | deliverables | staffing | compute / spend | success | failure |
|---|---|---|---|---|---|
| **30 days** | G1a generator and harness; G1a on 4 models; real embedder; randomized-inclusion logging; MCP server wrapper over the existing API | founder + 1 contractor | ≤ $100 tokens; laptop | headroom measured with CIs on ≥ 2 providers | headroom < 3pp on frontier models → pivot pitch to cost + privacy |
| **90 days** | G1b on LongMemEval/PersonaMem/CUPID; Next.js app; 3 connectors (notes, calendar, repo); security review of sovereignty code; G2 with n = 1 then 20 | 3 people | $1–3k tokens; small cloud | D > B with CI > 0 on 2 benchmarks; zero critical security findings | capture ≈ 0 → learned layer shelved; ship sovereign memory only |
| **12 months** | federated EM with DP; developer platform GA; published benchmark and paper; enterprise pilot | 6 people | $20–60k compute and tokens | ≥ 50 developer apps, ≥ 10k consenting users in the population prior; privacy-utility curve published | federated prior no better than designed prior → the network effect does not exist |
| **5 years** | neural-linear and summary-based portable representations; open-weight adapter path; contributed interchange standard; G3 capability results | 30–60 people | $1–5M/yr | an adopted interchange format; a replicated capability gain | — (research vision; speculative) |

Near-term rows are engineering. The 12-month federated-prior row is applied research with
a real chance of a negative result. The 5-year row is speculative.

---

## 7. Ten greatest risks (Step 8)

| # | risk | mitigation | abandonment criterion |
|---|---|---|---|
| 1 | No real headroom: frontier models already use full context well | G1a before anything else; sell cost and privacy | frontier headroom < 3pp → drop quality claims |
| 2 | The learned layer adds nothing over B-class memory on real outcomes | G1b on public benchmarks, pre-registered | capture ≤ 0 on 2 benchmarks |
| 3 | The population effect doesn't survive federation and DP | measure the privacy-utility curve early | federated prior ≤ designed prior at usable ε |
| 4 | Providers ship real portability, or "remember everywhere" | position on receipts, deletion, neutrality; contribute the standard | a provider-neutral standard adopted without CI's involvement *and* CI has no quality lead |
| 5 | Mem0, Engram or a hyperscaler wins developers first | MCP-first distribution; published benchmark lead; open core | after 12 months, < 50 developer apps |
| 6 | Feedback optimization learns sycophancy or manipulation [73–75] | verified outcomes dominate the reward; growth constraints; per-cohort audits | measured sycophancy increase vs A in G2 |
| 7 | A memory poisoning or exfiltration incident | write gating; egress guard; no side-effecting tools without confirmation; external red team | any critical exploit in production → halt until fixed |
| 8 | Dependency and deskilling | growth modes; unassisted probes; G3 | G3 shows harm vs control |
| 9 | Regulatory: GDPR erasure across backups; ADMT if used for significant decisions | crypto-shredding plus KEK rotation; scope exclusions; DPIA | a regulator finds erasure non-compliant and it is not fixable |
| 10 | Unit economics: heavy users, bundled tokens | BYO-key tier; fair-use limits; caching-aware context ordering | cost per active user > 40% of price at scale |

---

## 8. Step 10

The single most important next engineering action is **G1a** (RESEARCH.md §5.2, §9).
It needs provider API keys (none are configured on this machine) and your approval to
spend roughly $10–60.
