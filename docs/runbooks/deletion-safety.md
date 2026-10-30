# Deletion safety

## Managed containers only
InfraPilot only deletes containers labeled `infrapilot.managed=true`. Unlabeled containers are never touched, even when the name matches.

## Before deleting
List the managed containers and confirm the exact name. Deletion is immediate and data in the container is lost.

## After deleting
The agent lists containers again to verify that the deleted container is gone.
