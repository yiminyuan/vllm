# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aux hidden-state transport across pipeline stages.

EAGLE3-style drafters (eagle3, dflash, dspark) consume auxiliary hidden states
tapped from target layers. Under pipeline parallelism those layers can sit on an
earlier stage than the drafter, which runs on the last rank, so the taps have to
ride the intermediate-tensor handoff. Slot numbering is derived independently on
every stage from the layer split, so these tests pin that the derivations agree
-- a stage disagreeing by one silently feeds the drafter the wrong tap.
"""

import pytest
import torch

from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.sequence import IntermediateTensors


class _Config:
    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers


class _Model(EagleModelMixin):
    supports_aux_hidden_states_over_pp = True

    def __init__(self, num_hidden_layers: int, aux_layers: tuple[int, ...]):
        self.config = _Config(num_hidden_layers)
        self.aux_hidden_state_layers = aux_layers


@pytest.mark.parametrize(
    "start,end,aux_ids,is_first,expected",
    [
        # First stage also taps its own start layer, as the runner does.
        (0, 4, (0, 2, 4), True, (0, 2, 4)),
        (0, 4, (0, 2, 4), False, (2, 4)),
        # A later stage only claims taps whose layer it owns.
        (4, 8, (0, 2, 4), False, ()),
        (4, 8, (5, 8), False, (5, 8)),
        (4, 8, (), False, ()),
    ],
)
def test_local_aux_tap_ids(start, end, aux_ids, is_first, expected):
    assert EagleModelMixin.local_aux_tap_ids(start, end, aux_ids, is_first) == expected


def test_every_tap_is_claimed_by_exactly_one_stage():
    """The per-stage split must partition the taps, with none lost or doubled."""
    num_layers, pp = 43, 2
    aux = (0, 21, 42)
    model = _Model(num_layers, aux)

    from vllm.distributed.utils import get_pp_indices

    claimed: list[int] = []
    for rank in range(pp):
        start, end = get_pp_indices(num_layers, rank, pp)
        claimed.extend(
            EagleModelMixin.local_aux_tap_ids(start, end, aux, rank == 0)
        )
    assert sorted(claimed) == sorted(aux)
    assert len(claimed) == len(set(claimed))
    # And the totals line up with the slot arithmetic.
    assert sum(model._num_local_taps_on_rank(r, pp) for r in range(pp)) == len(aux)


def test_slot_numbering_is_contiguous_and_agreed():
    """Each rank's base is the count of taps produced by all earlier ranks."""
    num_layers, pp = 43, 2
    model = _Model(num_layers, (0, 21, 42))

    bases = [model._aux_slot_base(r, pp) for r in range(pp)]
    assert bases[0] == 0
    for r in range(1, pp):
        assert bases[r] == bases[r - 1] + model._num_local_taps_on_rank(r - 1, pp)


def test_pack_local_aux_keys_by_global_slot():
    model = _Model(43, (0, 21, 42))
    model._aux_slot_base_cached = 2
    taps = [torch.zeros(1), torch.ones(1)]

    packed = model.pack_local_aux_for_last(taps)

    assert sorted(packed) == ["aux_hidden_states_2", "aux_hidden_states_3"]
    assert torch.equal(packed["aux_hidden_states_2"], taps[0])
    assert torch.equal(packed["aux_hidden_states_3"], taps[1])
    # Nothing to say when this stage owns no taps.
    assert model.pack_local_aux_for_last([]) == {}


def test_recv_remote_aux_returns_producer_order():
    model = _Model(43, (0, 21, 42))
    model._aux_upstream_total_cached = 2
    a, b = torch.zeros(1), torch.ones(1)
    it = IntermediateTensors(
        {"aux_hidden_states_0": a, "aux_hidden_states_1": b, "hidden_states": a}
    )

    got = model.recv_remote_aux_from_producers(it)

    assert len(got) == 2
    assert torch.equal(got[0], a)
    assert torch.equal(got[1], b)


def test_recv_remote_aux_raises_rather_than_zero_filling():
    """A missing slot must fail loudly; zeros would only cost acceptance."""
    model = _Model(43, (0, 21, 42))
    model._aux_upstream_total_cached = 2
    it = IntermediateTensors({"aux_hidden_states_0": torch.zeros(1)})

    with pytest.raises(RuntimeError, match="aux_hidden_states_1 missing"):
        model.recv_remote_aux_from_producers(it)


def test_no_upstream_taps_needs_no_intermediate_tensors():
    """At PP=1, or when every tap is local, the receive side is a no-op."""
    model = _Model(43, (0, 21, 42))
    model._aux_upstream_total_cached = 0
    assert model.recv_remote_aux_from_producers(None) == []
