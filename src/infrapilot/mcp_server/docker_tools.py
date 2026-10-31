"""Sandbox tools over the Docker SDK.

Safety invariant: every container created here carries ``infrapilot.managed=true``
and no tool ever reads, modifies or deletes a container without that label.
"""

from typing import Any

import docker
from docker.errors import DockerException, NotFound
from docker.models.containers import Container

MANAGED_LABEL = "infrapilot.managed"
SERVICE_LABEL = "infrapilot.service"

Result = dict[str, Any]


def _err(message: str) -> Result:
    return {"ok": False, "error": message}


class DockerTools:
    def __init__(self, client: docker.DockerClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _managed(self, service: str | None = None) -> list[Container]:
        labels = [f"{MANAGED_LABEL}=true"]
        if service:
            labels.append(f"{SERVICE_LABEL}={service}")
        containers: list[Container] = self.client.containers.list(
            all=True, filters={"label": labels}
        )
        return containers

    def list_containers(self) -> Result:
        try:
            items = [
                {
                    "name": c.name,
                    "image": c.attrs["Config"]["Image"],
                    "status": c.status,
                    "service": c.labels.get(SERVICE_LABEL),
                }
                for c in self._managed()
            ]
            return {"ok": True, "containers": items}
        except DockerException as exc:
            return _err(f"docker unavailable: {exc}")

    def create_container(self, image: str, name: str, mem_mb: int, cpus: float) -> Result:
        try:
            self.client.containers.get(name)
            return _err(f"a container named '{name}' already exists")
        except NotFound:
            pass
        except DockerException as exc:
            return _err(f"docker unavailable: {exc}")
        return self._run(image, name, name, mem_mb, cpus)

    def _run(self, image: str, name: str, service: str, mem_mb: int, cpus: float) -> Result:
        try:
            c = self.client.containers.run(
                image,
                name=name,
                detach=True,
                labels={MANAGED_LABEL: "true", SERVICE_LABEL: service},
                mem_limit=f"{mem_mb}m",
                nano_cpus=int(cpus * 1_000_000_000),
            )
            return {"ok": True, "name": c.name, "status": c.status}
        except DockerException as exc:
            return _err(str(exc))

    def scale_service(self, name: str, replicas: int) -> Result:
        try:
            members = sorted(self._managed(service=name), key=lambda c: c.name or "")
            if not members:
                return _err(f"no managed service named '{name}'")
            template = members[0]
            image = template.attrs["Config"]["Image"]
            host = template.attrs["HostConfig"]
            mem_mb = max(16, int(host.get("Memory") or 256 * 1024 * 1024) // (1024 * 1024))
            cpus = (host.get("NanoCpus") or 500_000_000) / 1_000_000_000
            current = len(members)
            for victim in members[replicas:] if replicas < current else []:
                victim.remove(force=True)
            taken = {c.name for c in members}
            n = 2
            for _ in range(max(0, replicas - current)):
                while f"{name}-r{n}" in taken:
                    n += 1
                created = self._run(image, f"{name}-r{n}", name, mem_mb, cpus)
                if not created["ok"]:
                    return created
                taken.add(f"{name}-r{n}")
            return {"ok": True, "service": name, "replicas": replicas}
        except DockerException as exc:
            return _err(str(exc))

    def restart_container(self, name: str) -> Result:
        try:
            c = self.client.containers.get(name)
        except NotFound:
            return _err(f"no container named '{name}'")
        except DockerException as exc:
            return _err(f"docker unavailable: {exc}")
        if c.labels.get(MANAGED_LABEL) != "true":
            return _err(f"container '{name}' is not managed by InfraPilot; refusing")
        try:
            c.restart(timeout=10)
            c.reload()
            return {"ok": True, "restarted": name, "status": c.status}
        except DockerException as exc:
            return _err(str(exc))

    def delete_container(self, name: str) -> Result:
        try:
            c = self.client.containers.get(name)
        except NotFound:
            return _err(f"no container named '{name}'")
        except DockerException as exc:
            return _err(f"docker unavailable: {exc}")
        if c.labels.get(MANAGED_LABEL) != "true":
            return _err(f"container '{name}' is not managed by InfraPilot; refusing")
        try:
            c.remove(force=True)
            return {"ok": True, "deleted": name}
        except DockerException as exc:
            return _err(str(exc))
