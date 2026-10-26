# CI roadmap

The ordering follows evidence. Each item says what motivates it and what result would show
it worked. Nothing here is a claim that it will work.

## Immediate: next two to four weeks

1. **Make context selection relevance-selective.**
   - **Motivation:** E1 measured context precision of 3.3%, and the budget never binds. CI sends 57% more input tokens than naive history and ties it on cost per success.
   - **Plan:**
     1. Write a development dataset: a different person and different tasks from E1.
     2. Calibrate a relevance threshold, or a minimum-value cut, in the designed prior, and shorten the renderer's per-item labels, on that dataset only.
     3. Register E2 on a fresh test set, written before calibration and ideally by someone else.
   - **Done when:** E2 shows precision of at least 0.3 with recall of at least 0.9, and median input tokens no higher than naive history, at no loss of success.
2. **Grade use, not mention.**
   - **Motivation:** E1's keyword rules misread refusals as injection-following, and mentions as leaks.
   - **Plan:** E2 grades with a written rubric applied blind, by a grader that is not one of the systems compared. Keyword rules stay as a secondary check.
3. **A second vendor, live.**
   - **Motivation:** every live run so far was Anthropic-only, so "cross-model" has not yet been "cross-vendor", and verification independence is only "same vendor, different model."
   - **Plan:** set `OPENAI_API_KEY` and `PCIM_OPENAI_MODEL`, then rerun `scripts/milestone1_live.py` with OpenAI as Provider B and as the verifier.
4. **Test the Sonnet finding.**
   - **Motivation:** with the full history, Sonnet used unconsented private data in its answers; Haiku did not. That was exploratory.
   - **Plan:** register it in E2 as a confirmatory test across at least two models. The hypothesis: filtering matters more as models get stronger.
5. **Browser test the dashboard.** Approve, deny, revoke vendor and release must call the right endpoints. The server enforces everything; the UI is untested.

## Medium term: one to three months

1. **Intent engine.** Turn the first message into an objective, constraints, success criteria and open questions, shown to the person to confirm before saving. That gives a structured goal for selection and routing.
2. **Routing on measured track records.** The registry already counts, per verifier, assessments later borne out or contradicted by a test or the person. Route on them only once each model has enough settled claims, with intervals shown.
3. **An OS-level sandbox for `run_tests`:** a container or `sandbox-exec`, with no network by default. This is the largest remaining execution risk.
4. **Key custody.** Move the master key to the macOS Keychain, or a KMS for hosted use.
5. **A TypeScript SDK** generated from the OpenAPI schema FastAPI already serves: `createProject`, `recordContext`, `retrieveContext`, `proposePlan`, `requestApproval`, `executeAuthorizedTool`, `cancelTask`, `listActivity`, `deleteMemory`. Integration tests against a live local server.
6. **Billing.** Not started on purpose. These decisions have to be written down before any code:
   - how student eligibility is verified;
   - senior verification without identity documents;
   - for under-15 users: parental consent where required, restricted integrations, no purchases;
   - per-plan caps on model spend, enforced by `TurnLimits` plus a monthly meter, so free tiers cannot run up unbounded cost;
   - Stripe in test mode first, with webhook verification and entitlement checks server-side.

## Long term: research and architecture

1. **Population learning, done privately.**
   - **Motivation:** G0b found the learned value is collective. A population prior beat a tuned heuristic by about 15 percentage points in simulation; per-person adaptation added 1–3.
   - **Question:** can the prior be learned across people without pooling their content, using federated updates on content-free features?
   - **Kill rule:** no gain over the designed prior on real usage data.
2. **Multi-tenant hosting:**
   - Postgres with row-level security;
   - per-tenant keys;
   - real authentication;
   - per-tenant workers, since the session lock is per process today;
   - the isolation tests in `tests/test_isolation.py` rerun against the hosted stack.
3. **Agent-to-agent exchange beyond verification.** The `AgentMessage` gateway supports one request type, `verify_claim`. Delegated subtasks (analysis, security review) need the same discipline:
   - scoped refs;
   - stored messages;
   - typed replies;
   - no transcript sharing;
   - derived-content lineage.

   They also need a budget per delegation.
4. **Does CI make people better at their work?** The question in the original brief. It needs a study with people, real tasks, blinded outcome measures and a control condition (`docs/RESEARCH.md`, G2–G3). Nothing in this repository answers it yet.
