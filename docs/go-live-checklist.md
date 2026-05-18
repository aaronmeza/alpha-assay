# Go Live Checklist

- Paper P&L > 0 with MDD < X% over ≥ 20 sessions

- Commission & slippage modeled; sensitivity tested

- Fail-closed kill-switch (network/API failure, price staleness, drawdown limit)

- Alerting to phone/Slack; dry-run of recovery runbook

- Read-only IAM for metrics; secrets not in repo; roll-back plan

- Verify front-month metadata key matches IBKR's current front-month: `docker exec alphaassay-redis redis-cli get alpha_assay:front_month:es.cme` must return an `YYYYMMDD` matching the contract IBKR currently treats as front. Cross-check against the `alpha_assay_front_month_expiry{symbol="ES",exchange="CME"}` Prometheus gauge. If they disagree, the pre-open re-qualify partially failed: restart `ibkr-feed` to force resolution from scratch before going live.
