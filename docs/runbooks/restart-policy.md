# Restart policy

## Default behavior
Managed containers do not restart automatically. A container that exits stays stopped so that the failure can be inspected.

## Recovering a failed container
Delete the failed container, then create it again with the same name and limits. Both steps require approval.

## Crash loops
If a container fails twice in a row, do not recreate it a third time. Open an incident with the platform team.
