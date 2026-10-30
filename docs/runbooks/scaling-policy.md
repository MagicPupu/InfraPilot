# Scaling policy

## Replica limits
A service may have between 0 and 5 replicas. Requests for more than 5 replicas are refused.

## Scaling up
Scaling up creates new replicas from the image and limits of the first container of the service.

## Scaling to zero
Scaling a service to zero removes all of its containers. It is treated as a destructive operation and requires explicit human approval.
