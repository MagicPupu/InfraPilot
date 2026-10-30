# Naming conventions

## Container names
Every container managed by InfraPilot must start with the prefix `ip-`, followed by lowercase letters, digits and hyphens. Example: `ip-web`, `ip-cache-01`.

## Replica names
Additional replicas of a service are named `<service>-r<N>`, starting at `r2`. The original container keeps the service name.

## Forbidden names
Names without the `ip-` prefix are rejected, so unmanaged containers can never be addressed by mistake.
