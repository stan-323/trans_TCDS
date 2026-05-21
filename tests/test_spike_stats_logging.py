from types import SimpleNamespace

from onpolicy.algorithms.r_mappo.r_mappo import collect_actor_spike_stats


class FakeBase:
    def get_spike_stats(self):
        return {
            "spike_rate": 0.25,
            "threshold_mean": 1.2,
            "time_steps": 4,
        }


class FakePolicy:
    actor = SimpleNamespace(base=FakeBase())


def test_collect_actor_spike_stats_prefixes_numeric_stats():
    stats = collect_actor_spike_stats(FakePolicy())

    assert stats == {
        "spike_stats/spike_rate": 0.25,
        "spike_stats/threshold_mean": 1.2,
        "spike_stats/time_steps": 4.0,
    }
