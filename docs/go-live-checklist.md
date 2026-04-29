# Go Live Checklist

- Paper P&L > 0 with MDD < X% over ≥ 20 sessions

- Commission & slippage modeled; sensitivity tested

- Fail-closed kill-switch (network/API failure, price staleness, drawdown limit)

- Alerting to phone/Slack; dry-run of recovery runbook

- Read-only IAM for metrics; secrets not in repo; roll-back plan
