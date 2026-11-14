"""Deterministic runtime demo. No API key, network, or real workspace edits."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from Agent_Assistant.config import LlmConfig
from Agent_Assistant.llm_clients.client import ModelClient
from Agent_Assistant.runtime.agent import AgentRunner
from Agent_Assistant.runtime.approvals import ApprovalBroker


class DemoClient(ModelClient):
    def __init__(self) -> None:
        self.config = LlmConfig(
            "unused", "https://example.invalid", "scripted-demo", "chat_completions"
        )
        self.turn = 0

    async def chat(self, messages, tools):
        self.turn += 1
        if self.turn == 1:
            message = {
                "role": "assistant",
                "content": "读取示例配置。",
                "tool_calls": [
                    {
                        "id": "read1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"settings.py"}'},
                    }
                ],
            }
        elif self.turn == 2:
            original = json.loads(messages[-1]["content"])
            args = {
                "path": "settings.py",
                "old_text": "RETRIES = 1",
                "new_text": "RETRIES = 3",
                "expected_sha256": original["sha256"],
            }
            message = {
                "role": "assistant",
                "content": "准备最小修改。",
                "tool_calls": [
                    {
                        "id": "patch1",
                        "type": "function",
                        "function": {"name": "apply_patch", "arguments": json.dumps(args)},
                    }
                ],
            }
        elif self.turn == 3:
            message = {"role": "assistant", "content": "已把 RETRIES 从 1 改为 3。"}
        else:
            assert any("RETRIES 从 1 改为 3" in str(m.get("content", "")) for m in messages)
            message = {
                "role": "assistant",
                "content": "上一轮把 RETRIES 从 1 改为 3，会话历史已恢复。",
            }
        return {
            "choices": [
                {
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ]
        }

    async def chat_stream(self, messages, tools):
        data = await self.chat(messages, tools)
        yield {"type": "text_delta", "text": data["choices"][0]["message"]["content"]}
        yield {"type": "final", "data": data}

    async def aclose(self):
        pass


async def main() -> None:
    print("离线脚本模型演示：验证真实工具、审批、会话；不代表真实 LLM 的任务成功率。")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workspace = root / "project"
        workspace.mkdir()
        file = workspace / "settings.py"
        file.write_text("RETRIES = 1\n")
        broker = ApprovalBroker()
        client = DemoClient()
        runner = AgentRunner(client=client, state_dir=root / "state", approvals=broker)
        async for event in runner.stream_run(
            "把 RETRIES 改为 3", workspace, 5, False, session_id="demo"
        ):
            if event.get("type") == "permission.requested":
                print(event["diff"])
                assert file.read_text() == "RETRIES = 1\n"
                print("演示审批器：只批准这个临时文件的已展示 diff。")
                broker.resolve(event["approval_id"], True)
            elif event.get("type") == "text.delta":
                print(event["text"])
            elif "result" in event:
                assert event["result"]["status"] == "success", event["result"]
        assert file.read_text() == "RETRIES = 3\n"
        restarted = AgentRunner(client=client, state_dir=root / "state")
        result = await restarted.run("上一轮改了什么？", workspace, 2, False, session_id="demo")
        assert result.status == "success", result
        print(result.result)
        print("PASS：diff 审批、精确修改、文件验证、会话恢复。")


if __name__ == "__main__":
    asyncio.run(main())
