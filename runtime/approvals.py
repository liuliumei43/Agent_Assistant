from __future__ import annotations

import asyncio
import secrets


class ApprovalBroker:
    def __init__(self, timeout: float = 300) -> None:
        self.timeout = timeout
        self.pending: dict[str, asyncio.Future[bool]] = {}

    def create(self) -> str:
        approval_id = secrets.token_urlsafe(32)
        self.pending[approval_id] = asyncio.get_running_loop().create_future()
        return approval_id

    def resolve(self, approval_id: str, approved: bool) -> None:
        future = self.pending.get(approval_id)
        if future is None or future.done():
            raise ValueError("Approval is expired, already answered, or unknown")
        future.set_result(approved)

    async def wait(self, approval_id: str) -> bool:
        try:
            return await asyncio.wait_for(self.pending[approval_id], self.timeout)
        except TimeoutError:
            return False
        finally:
            self.discard(approval_id)

    def discard(self, approval_id: str) -> None:
        future = self.pending.pop(approval_id, None)
        if future is not None and not future.done():
            future.cancel()
