# Monetization Gate

Synapse must remain no-payment pilot software until the product quality gate is proven with real users.

## Do Not Build Yet

- Stripe checkout
- Subscriptions
- License enforcement
- Paid trial gates
- Seat billing
- Payment analytics
- Sales automation

## Required Before Billing

- CI green on every push.
- Windows build verified with `docs/release_checklist.md`.
- At least 3 real users complete onboarding and monitor sessions without developer help.
- Reports are trusted enough that users do not see embarrassing false labels as dominant outcomes.
- Privacy notice, local data controls, and delete/export commands are understood by pilot users.
- One unpaid team pilot produces useful feedback and a clear value story.

## Allowed Before Billing

- Free pilot summaries.
- Manual feedback interviews.
- Documentation for future pricing hypotheses.
- Manual invoices only after a human-approved pilot agreement, not inside the product.

## Decision Rule

If the work improves reliability, privacy, onboarding, report clarity, or pilot learning, it belongs before monetization.

If the work exists primarily to collect money or restrict access, it waits.
