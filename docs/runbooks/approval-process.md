# Approval process

## What needs approval
Any change to the sandbox needs a human approval: creating containers, scaling services and deleting containers. Read-only requests such as listing containers do not.

## Destructive operations
Deleting a container and scaling a service to zero are destructive. The approval prompt shows an explicit warning.

## What is approved
An approval applies to one exact plan, identified by its hash. If the plan changes, the approval is void and a new one is required.
