# Copyright Sierra

import json
import time
from litellm import completion
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME


class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        # --- Timing instrumentation (TVCache workload characterization) ---
        # Buckets (seconds): agent_llm = agent completion() calls; user_llm = time inside the
        # simulated user's reset/step (its LLM calls); tool = env.step() for tool actions;
        # reward_calc = env.calculate_reward() (DB reload + ground-truth replay, runs on the
        # final step); other = total - the rest (DB load in reset, bookkeeping).
        t_solve0 = time.perf_counter()
        timings = {"agent_llm": 0.0, "user_llm": 0.0, "tool": 0.0, "reward_calc": 0.0}
        tool_calls: List[Dict[str, Any]] = []
        llm_calls: List[Dict[str, Any]] = []
        state = {"in_reward": False, "step": -1}

        def _timed_user(fn):
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    dt = time.perf_counter() - t0
                    if not state["in_reward"]:
                        timings["user_llm"] += dt
                        llm_calls.append({"who": "user", "step": state["step"], "seconds": dt})
            return wrapper

        orig_calculate_reward = env.calculate_reward

        def _timed_reward(*args, **kwargs):
            state["in_reward"] = True
            t0 = time.perf_counter()
            try:
                return orig_calculate_reward(*args, **kwargs)
            finally:
                timings["reward_calc"] += time.perf_counter() - t0
                state["in_reward"] = False

        env.user.reset = _timed_user(env.user.reset)
        env.user.step = _timed_user(env.user.step)
        env.calculate_reward = _timed_reward
        user_turn = 0
        # --- end timing setup ---

        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        for step_idx in range(max_num_steps):
            state["step"] = step_idx
            t0 = time.perf_counter()
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
            dt = time.perf_counter() - t0
            timings["agent_llm"] += dt
            usage = getattr(res, "usage", None)
            llm_calls.append({
                "who": "agent", "step": step_idx, "seconds": dt,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            })
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params["response_cost"] or 0
            action = message_to_action(next_message)
            if action.name == RESPOND_ACTION_NAME:
                env_response = env.step(action)
                user_turn += 1
            else:
                reward_before = timings["reward_calc"]
                t0 = time.perf_counter()
                env_response = env.step(action)
                # A terminate tool (retail: transfer_to_human_agents) triggers reward_calc inside
                # step(); subtract it so tool time is only the tool itself.
                dt = time.perf_counter() - t0 - (timings["reward_calc"] - reward_before)
                timings["tool"] += dt
                tool_calls.append({
                    "tool": action.name, "args": action.kwargs, "seconds": dt,
                    "turn_index": step_idx, "user_turn": user_turn,
                    "error": str(env_response.observation).startswith(("Error:", "Unknown action")),
                })
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
        timings["total"] = time.perf_counter() - t_solve0
        timings["other"] = timings["total"] - sum(
            timings[k] for k in ("agent_llm", "user_llm", "tool", "reward_calc")
        )
        timings["num_tool_calls"] = len(tool_calls)
        timings["num_agent_steps"] = sum(1 for c in llm_calls if c["who"] == "agent")
        info["timings"] = timings
        info["tool_calls"] = tool_calls
        info["llm_calls"] = llm_calls
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
