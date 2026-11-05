"""Tests for injected tool latency. Run with: python -m unittest discover tests"""

import json
import math
import random
import statistics
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from tau_bench.envs import get_env
from tau_bench.envs.airline.tools import ALL_TOOLS as AIRLINE_TOOLS
from tau_bench.envs.latency import (
    PROFILES,
    LogNormalDelay,
    ToolLatency,
    make_tool_latency,
)
from tau_bench.envs.retail.data import load_data as load_retail_data
from tau_bench.envs.retail.tools import ALL_TOOLS as RETAIL_TOOLS
from tau_bench.run import run
from tau_bench.types import Action, RunConfig

SERVICE_API = PROFILES["service_api"]


def fake_response(message):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        _hidden_params={"response_cost": 0.0},
    )


class FakeMessage:
    def __init__(self, content=None, tool=None, **arguments):
        self.content = content
        self.tool = tool
        self.arguments = arguments

    def model_dump(self):
        tool_calls = None
        if self.tool is not None:
            tool_calls = [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": self.tool,
                        "arguments": json.dumps(self.arguments),
                    },
                }
            ]
        return {"role": "assistant", "content": self.content, "tool_calls": tool_calls}


def fake_user_completion(messages, **kwargs):
    """The simulated user opens the conversation, then ends it on its next turn."""
    opening = messages[-1]["content"] == "Hi! How can I help you today?"
    return fake_response(FakeMessage(content="Hi" if opening else "###STOP###"))


def make_env(env_name="retail", tool_latency=None, task_index=0):
    with mock.patch("tau_bench.envs.user.completion", fake_user_completion):
        env = get_env(
            env_name,
            user_strategy="llm",
            user_model="fake",
            user_provider="openai",
            task_split="test",
            task_index=task_index,
            tool_latency=tool_latency,
        )
    env.user.step = lambda content: "###STOP###"
    return env


def tool_name(tool):
    return tool.get_info()["function"]["name"]


class ProfileTest(unittest.TestCase):
    def test_profile_covers_exactly_the_tools_of_both_domains(self):
        tools = {tool_name(t) for t in [*RETAIL_TOOLS, *AIRLINE_TOOLS]}
        self.assertEqual(tools, set(SERVICE_API))

    def test_only_local_tools_have_no_delay(self):
        local = {name for name, delay in SERVICE_API.items() if delay is None}
        self.assertEqual(local, {"think", "calculate"})

    def test_tools_that_mutate_the_database_are_writes(self):
        writes = {
            name for name, delay in SERVICE_API.items() if delay == LogNormalDelay(0.3, 0.8)
        }
        # Name-based, so a mutating tool can't be missed by a new naming pattern
        # without also failing test_profile_covers_exactly_the_tools_of_both_domains.
        expected = {
            name
            for name in SERVICE_API
            if name.startswith(("cancel_", "modify_", "exchange_", "return_", "book_", "update_", "send_"))
        } | {"transfer_to_human_agents"}
        self.assertEqual(writes, expected)

    def test_env_refuses_a_profile_missing_one_of_its_tools(self):
        delays = {k: v for k, v in SERVICE_API.items() if k != "get_order_details"}
        with self.assertRaisesRegex(ValueError, "get_order_details"):
            make_env(tool_latency=ToolLatency(delays, 1.0, "s"))


class EstimateProfileTest(unittest.TestCase):
    def test_estimates_cover_exactly_the_tools_of_both_domains(self):
        tools = {tool_name(t) for t in [*RETAIL_TOOLS, *AIRLINE_TOOLS]}
        for name in ("tau_fast", "tau_central", "tau_slow"):
            self.assertEqual(tools, set(PROFILES[name]), name)

    def test_estimates_have_the_intended_medians(self):
        expected = {
            "tau_fast": {"get_order_details": 5, "cancel_pending_order": 215},
            "tau_central": {"get_order_details": 60, "search_onestop_flight": 1600},
            "tau_slow": {"get_order_details": 150, "search_direct_flight": 3000},
        }
        for profile, tools in expected.items():
            for tool, median_ms in tools.items():
                delay = PROFILES[profile][tool]
                median = math.sqrt(delay.p5_s * delay.p95_s) * 1000
                self.assertAlmostEqual(median, median_ms, places=6)

    def test_estimates_use_the_intended_spread(self):
        delay = PROFILES["tau_central"]["get_order_details"]
        rng = random.Random(0)
        samples = sorted(delay.sample(rng) for _ in range(50_000))
        self.assertAlmostEqual(samples[47_500] / samples[25_000], 2.28, delta=0.05)

    def test_think_and_calculate_stay_free(self):
        for name in ("tau_fast", "tau_central", "tau_slow"):
            self.assertIsNone(PROFILES[name]["think"])
            self.assertIsNone(PROFILES[name]["calculate"])


class SamplingTest(unittest.TestCase):
    def test_samples_match_the_configured_percentiles(self):
        rng = random.Random(0)
        delay = LogNormalDelay(p5_s=0.1, p95_s=0.3)
        cuts = statistics.quantiles([delay.sample(rng) for _ in range(50_000)], n=20)
        self.assertAlmostEqual(cuts[0], 0.1, delta=0.003)
        self.assertAlmostEqual(cuts[-1], 0.3, delta=0.009)

    def _samples(self, session_seed, scale=1.0):
        latency = ToolLatency(SERVICE_API, scale, session_seed)
        with mock.patch("tau_bench.envs.latency.time.sleep"):
            return [latency.apply("get_order_details") for _ in range(5)]

    def test_same_session_seed_gives_same_delays(self):
        self.assertEqual(self._samples("10:0:3"), self._samples("10:0:3"))

    def test_different_sessions_get_different_delays(self):
        self.assertNotEqual(self._samples("10:0:3"), self._samples("10:0:4"))
        self.assertNotEqual(self._samples("10:0:3"), self._samples("10:1:3"))

    def test_scale_multiplies_the_delay(self):
        for base, scaled in zip(self._samples("s"), self._samples("s", scale=2.0)):
            self.assertAlmostEqual(scaled, base * 2)

    def test_apply_sleeps_for_the_returned_delay(self):
        latency = ToolLatency(SERVICE_API, 1.0, "s")
        with mock.patch("tau_bench.envs.latency.time.sleep") as sleep:
            seconds = latency.apply("get_order_details")
        sleep.assert_called_once_with(seconds)

    def test_no_latency_profile_builds_nothing(self):
        self.assertIsNone(make_tool_latency("none", 1.0, "s"))


class EnvStepTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("tau_bench.envs.latency.time.sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def _env(self, env_name="retail"):
        return make_env(env_name, make_tool_latency("service_api", 1.0, "s"))

    def test_off_by_default(self):
        env = make_env()
        response = env.step(Action(name="get_order_details", kwargs={"order_id": "#W0"}))
        self.sleep.assert_not_called()
        self.assertIsNone(response.info.injected_latency)

    def test_lookup_reports_its_delay(self):
        env = self._env()
        response = env.step(Action(name="get_order_details", kwargs={"order_id": "#W0"}))
        self.sleep.assert_called_once_with(response.info.injected_latency)

    def test_airline_tools_are_delayed(self):
        env = self._env("airline")
        response = env.step(Action(name="list_all_airports", kwargs={}))
        self.sleep.assert_called_once_with(response.info.injected_latency)

    def test_local_tool_does_not_sleep(self):
        env = self._env()
        response = env.step(Action(name="think", kwargs={"thought": "hmm"}))
        self.sleep.assert_not_called()
        self.assertEqual(response.info.injected_latency, 0.0)

    def test_tool_error_still_pays_the_delay(self):
        env = self._env()
        response = env.step(Action(name="get_order_details", kwargs={"order_id": "nope"}))
        self.assertTrue(response.observation.startswith("Error"))
        self.sleep.assert_called_once()

    def test_unknown_action_and_user_reply_do_not_sleep(self):
        env = self._env()
        env.step(Action(name="not_a_tool", kwargs={}))
        env.step(Action(name="respond", kwargs={"content": "hello"}))
        self.sleep.assert_not_called()

    def test_reward_replay_does_not_sleep(self):
        env = self._env()
        tool_actions = [
            a for a in env.task.actions if a.name in env.tools_map
        ]
        self.assertGreater(len(tool_actions), 0)
        env.calculate_reward()
        self.sleep.assert_not_called()
        self.assertFalse(env.replaying_ground_truth)

    def test_terminate_tool_reports_its_own_delay_not_the_replays(self):
        env = self._env()
        response = env.step(
            Action(name="transfer_to_human_agents", kwargs={"summary": "help"})
        )
        self.assertTrue(response.done)
        # One sleep for the transfer itself, none for the replay it triggers.
        self.sleep.assert_called_once_with(response.info.injected_latency)
        self.assertIsNotNone(response.info.reward_info)


class FullRunTest(unittest.TestCase):
    """Drives tau_bench.run.run with real sleeps and only the LLMs faked."""

    @classmethod
    def setUpClass(cls):
        orders = load_retail_data()["orders"]
        cls.pending_order = next(
            order_id for order_id, order in orders.items() if order["status"] == "pending"
        )

    def _agent_completion(self, messages, **kwargs):
        script = [
            FakeMessage(tool="think", thought="look it up"),
            FakeMessage(tool="get_order_details", order_id=self.pending_order),
            FakeMessage(
                tool="cancel_pending_order",
                order_id=self.pending_order,
                reason="no longer needed",
            ),
            FakeMessage(content="All done"),
        ]
        step = sum(1 for m in messages if m["role"] == "assistant")
        return fake_response(script[step])

    def _run(self, **config):
        with tempfile.TemporaryDirectory() as log_dir, mock.patch(
            "tau_bench.envs.user.completion", fake_user_completion
        ), mock.patch(
            "tau_bench.agents.tool_calling_agent.completion", self._agent_completion
        ):
            return run(
                RunConfig(
                    model_provider="openai",
                    user_model_provider="openai",
                    model="fake",
                    user_model="fake",
                    log_dir=log_dir,
                    task_ids=[0, 1],
                    max_concurrency=2,
                    **config,
                )
            )

    def test_timers_capture_the_injected_delay(self):
        results = self._run(tool_latency_profile="service_api", tool_latency_scale=0.2)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertNotIn("error", result.info)
            calls = {c["tool"]: c for c in result.info["tool_calls"]}
            self.assertEqual(
                set(calls), {"think", "get_order_details", "cancel_pending_order"}
            )
            self.assertEqual(calls["think"]["injected_seconds"], 0.0)
            for call in result.info["tool_calls"]:
                self.assertGreaterEqual(call["seconds"], call["injected_seconds"])
            self.assertGreater(calls["get_order_details"]["injected_seconds"], 0.0)
            self.assertGreater(
                calls["cancel_pending_order"]["injected_seconds"],
                calls["get_order_details"]["injected_seconds"],
            )
            timings = result.info["timings"]
            self.assertAlmostEqual(
                timings["tool_injected"],
                sum(c["injected_seconds"] for c in result.info["tool_calls"]),
            )
            self.assertGreaterEqual(timings["tool"], timings["tool_injected"])
            agent_calls = [c for c in result.info["llm_calls"] if c["who"] == "agent"]
            self.assertEqual([c["user_turn"] for c in agent_calls], [0, 0, 0, 0])
            self.assertIn("cached_tokens", agent_calls[0])

    def _injected(self, results):
        return {
            r.task_id: [c["injected_seconds"] for c in r.info["tool_calls"]]
            for r in results
        }

    def test_delays_reproduce_across_runs_and_concurrency(self):
        first = self._run(tool_latency_profile="service_api", tool_latency_scale=0.05)
        second = self._run(tool_latency_profile="service_api", tool_latency_scale=0.05)
        self.assertEqual(self._injected(first), self._injected(second))

    def test_seed_changes_the_delays(self):
        first = self._run(tool_latency_profile="service_api", tool_latency_scale=0.05)
        second = self._run(
            tool_latency_profile="service_api", tool_latency_scale=0.05, seed=11
        )
        self.assertNotEqual(self._injected(first), self._injected(second))

    def test_each_task_and_trial_gets_its_own_delays(self):
        results = self._run(
            tool_latency_profile="service_api", tool_latency_scale=0.05, num_trials=2
        )
        delays = {
            tuple(c["injected_seconds"] for c in r.info["tool_calls"]) for r in results
        }
        self.assertEqual(len(results), 4)
        self.assertEqual(len(delays), 4)

    def test_default_run_injects_nothing(self):
        for result in self._run():
            self.assertEqual(result.info["timings"]["tool_injected"], 0.0)

    def test_invalid_profile_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "latency profile"):
            self._run(tool_latency_profile="bogus")

    def test_negative_scale_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "scale"):
            self._run(tool_latency_profile="service_api", tool_latency_scale=-1.0)


if __name__ == "__main__":
    unittest.main()
