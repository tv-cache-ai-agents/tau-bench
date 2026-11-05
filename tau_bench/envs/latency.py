"""Injected tool latency, so tau-bench's in-memory tools take realistic time.

Out of the box every tool is a dict lookup that returns in microseconds, which leaves
a tool-value cache nothing to save. A profile assigns each tool a delay distribution,
and Env.step sleeps for a sample before running the tool.

The profile values are modelling assumptions, not measurements. Reported cache savings
scale with them, so sweep --tool-latency-scale rather than quoting a single value.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

# 95th percentile of the standard normal distribution.
_Z_95 = 1.6449


@dataclass(frozen=True)
class LogNormalDelay:
    """A long-tailed delay, specified by its 5th and 95th percentiles in seconds."""

    p5_s: float
    p95_s: float

    def sample(self, rng: random.Random) -> float:
        mu = math.log(self.p5_s * self.p95_s) / 2
        sigma = math.log(self.p95_s / self.p5_s) / (2 * _Z_95)
        return rng.lognormvariate(mu, sigma)


# None means the tool runs locally and gets no delay.
_LOCAL = None
_LOOKUP = LogNormalDelay(p5_s=0.100, p95_s=0.300)
_WRITE = LogNormalDelay(p5_s=0.300, p95_s=0.800)

# Treats tools as calls to internal service APIs. Covers both the retail and airline
# domains; Env refuses to start if one of its tools is missing from the profile.
_SERVICE_API: Dict[str, Optional[LogNormalDelay]] = {
    # Both domains
    "calculate": _LOCAL,
    "think": _LOCAL,
    "get_user_details": _LOOKUP,
    "transfer_to_human_agents": _WRITE,
    # Retail
    "find_user_id_by_email": _LOOKUP,
    "find_user_id_by_name_zip": _LOOKUP,
    "get_order_details": _LOOKUP,
    "get_product_details": _LOOKUP,
    "list_all_product_types": _LOOKUP,
    "cancel_pending_order": _WRITE,
    "exchange_delivered_order_items": _WRITE,
    "modify_pending_order_address": _WRITE,
    "modify_pending_order_items": _WRITE,
    "modify_pending_order_payment": _WRITE,
    "modify_user_address": _WRITE,
    "return_delivered_order_items": _WRITE,
    # Airline
    "get_reservation_details": _LOOKUP,
    "list_all_airports": _LOOKUP,
    "search_direct_flight": _LOOKUP,
    "search_onestop_flight": _LOOKUP,
    "book_reservation": _WRITE,
    "cancel_reservation": _WRITE,
    "send_certificate": _WRITE,
    "update_reservation_baggages": _WRITE,
    "update_reservation_flights": _WRITE,
    "update_reservation_passengers": _WRITE,
}


# The tau_* profiles are bottom-up estimates. Replaying saved trajectories shows tools touch
# only a few rows, so a call's latency is set by the network and external systems around it:
# a remote read (about 57 ms in the TVCache paper's SkyRL-SQL workload), a commit, a payment
# provider, or a flight-search system. Values below are median ms per call for a same-region
# backend, the central estimate, and a vendor-API-style backend. All are estimates.
_ESTIMATE_GROUPS = {
    "key_lookup": (
        (5, 60, 150),
        (
            "get_order_details",
            "get_user_details",
            "get_product_details",
            "get_reservation_details",
        ),
    ),
    "indexed_lookup": (
        (8, 65, 200),
        (
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
            "list_all_product_types",
            "list_all_airports",
        ),
    ),
    "database_write": (
        (15, 75, 250),
        (
            "modify_pending_order_address",
            "modify_user_address",
            "exchange_delivered_order_items",
            "update_reservation_passengers",
            "send_certificate",
            "transfer_to_human_agents",
        ),
    ),
    "payment_write": (
        (215, 370, 800),
        (
            "cancel_pending_order",
            "return_delivered_order_items",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "book_reservation",
            "cancel_reservation",
            "update_reservation_flights",
            "update_reservation_baggages",
        ),
    ),
    "direct_flight_search": ((400, 800, 3000), ("search_direct_flight",)),
    "onestop_flight_search": ((800, 1600, 8000), ("search_onestop_flight",)),
}
# Call-to-call spread for the estimates: p95 is about 2.3x the median. This is a judgement.
_ESTIMATE_SIGMA = 0.5


def _estimate_profile(column: int) -> Dict[str, Optional[LogNormalDelay]]:
    spread = math.exp(_Z_95 * _ESTIMATE_SIGMA)
    profile: Dict[str, Optional[LogNormalDelay]] = {"think": _LOCAL, "calculate": _LOCAL}
    for medians_ms, tools in _ESTIMATE_GROUPS.values():
        median_s = medians_ms[column] / 1000
        delay = LogNormalDelay(p5_s=median_s / spread, p95_s=median_s * spread)
        profile.update({tool: delay for tool in tools})
    return profile


NO_LATENCY_PROFILE = "none"
PROFILES: Dict[str, Dict[str, Optional[LogNormalDelay]]] = {
    "service_api": _SERVICE_API,
    "tau_fast": _estimate_profile(0),
    "tau_central": _estimate_profile(1),
    "tau_slow": _estimate_profile(2),
}
PROFILE_NAMES = [NO_LATENCY_PROFILE, *PROFILES]


class ToolLatency:
    """Samples and applies tool delays for one session.

    Each session owns its random stream, so a run's delays do not depend on how
    concurrent sessions happen to interleave.
    """

    def __init__(
        self,
        delays: Dict[str, Optional[LogNormalDelay]],
        scale: float,
        session_seed: str,
    ) -> None:
        self.delays = delays
        self.scale = scale
        self.rng = random.Random(session_seed)

    def check_covers(self, tool_names: Iterable[str]) -> None:
        missing = sorted(set(tool_names) - set(self.delays))
        if missing:
            raise ValueError(f"Latency profile has no entry for tools: {missing}")

    def apply(self, tool_name: str) -> float:
        """Sleeps for this tool's delay and returns the delay in seconds."""
        delay = self.delays[tool_name]
        if delay is None:
            return 0.0
        seconds = delay.sample(self.rng) * self.scale
        time.sleep(seconds)
        return seconds


def make_tool_latency(
    profile: str, scale: float, session_seed: str
) -> Optional[ToolLatency]:
    if profile == NO_LATENCY_PROFILE:
        return None
    return ToolLatency(PROFILES[profile], scale, session_seed)
