# Monitoring Notes

## Dashboard Template

Use `grafana-dashboard.json` as a starting point for:
- HAProxy queue depth
- HAProxy active backend sessions
- Backend active servers
- PostgreSQL container restarts

## Alert Metadata

Set these Helm values for better on-call routing:
- `monitoring.alerts.owner`
- `monitoring.alerts.runbookBaseUrl`

Example:

```yaml
monitoring:
  alerts:
    enabled: true
    owner: "data-platform"
    runbookBaseUrl: "https://internal-wiki/runbooks/pg-deeplake"
```
