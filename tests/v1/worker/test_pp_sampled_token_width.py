# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The PP sampled-token broadcast must agree on width between the two sides.

`PPHandler.receive` allocates `[num_reqs, max_sample_len + draft_token_width]`
unconditionally, while the sampler hands `broadcast` a single column whenever
there were no draft tokens to verify -- which is the case throughout warmup.
Broadcasting the narrower tensor makes the ranks disagree on the element count,
and the collective then never completes: the last rank sails on while the other
blocks in the next device sync with its GPU spinning.

Without speculation `max_sample_len` is 1 and the two widths coincide, so this
only bites with MTP enabled, which is why it is worth pinning here.
"""

import pytest
import torch


def _payload_width(max_sample_len: int, draft_token_width: int) -> int:
    return max_sample_len + draft_token_width


def _sender_payload(
    sampled_token_ids: torch.Tensor,
    max_sample_len: int,
    draft_token_width: int,
    draft_token_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Mirrors PPHandler.broadcast's payload construction."""
    payload = sampled_token_ids.new_zeros(
        sampled_token_ids.shape[0], max_sample_len + draft_token_width
    )
    num_cols = min(sampled_token_ids.shape[1], max_sample_len)
    payload[:, :num_cols].copy_(sampled_token_ids[:, :num_cols])
    if draft_token_width:
        assert draft_token_ids is not None
        payload[:, max_sample_len:].copy_(draft_token_ids)
    return payload


@pytest.mark.parametrize("num_speculative_steps", [0, 1, 3, 7])
@pytest.mark.parametrize("sampler_cols", [1, None])
def test_sender_width_matches_receiver_allocation(num_speculative_steps, sampler_cols):
    """A one-column sampler output must still broadcast at the agreed width."""
    num_reqs = 4
    max_sample_len = num_speculative_steps + 1
    cols = sampler_cols if sampler_cols is not None else max_sample_len
    sampled = torch.arange(num_reqs * cols, dtype=torch.int64).view(num_reqs, cols)

    payload = _sender_payload(sampled, max_sample_len, 0, None)

    # This is exactly what receive() allocates.
    assert payload.shape == (num_reqs, _payload_width(max_sample_len, 0))
    # The columns the sampler did provide survive unchanged.
    assert torch.equal(payload[:, :cols], sampled[:, :cols])


@pytest.mark.parametrize("num_speculative_steps", [1, 7])
def test_draft_tokens_ride_the_same_payload(num_speculative_steps):
    num_reqs = 3
    max_sample_len = num_speculative_steps + 1
    draft_width = num_speculative_steps
    sampled = torch.full((num_reqs, 1), 5, dtype=torch.int64)
    drafts = torch.arange(num_reqs * draft_width, dtype=torch.int64).view(
        num_reqs, draft_width
    )

    payload = _sender_payload(sampled, max_sample_len, draft_width, drafts)

    assert payload.shape == (num_reqs, _payload_width(max_sample_len, draft_width))
    # Receiver splits at max_sample_len; both halves must come back intact.
    assert torch.equal(payload[:, :1], sampled)
    assert torch.equal(payload[:, max_sample_len:], drafts)


def test_narrow_sampler_output_is_not_replicated_across_columns():
    """Padding must be zeros, not a broadcast of the single sampled column.

    `Tensor.copy_` broadcasts a [n, 1] source across a [n, k] destination, which
    would silently fill every speculative slot with the bonus token. num_sampled
    bounds what is read, but replicating real token ids into unused slots makes
    any later off-by-one read plausible-looking garbage instead of an obvious 0.
    """
    num_reqs, max_sample_len = 2, 8
    sampled = torch.full((num_reqs, 1), 7, dtype=torch.int64)

    payload = _sender_payload(sampled, max_sample_len, 0, None)

    assert torch.equal(payload[:, :1], sampled)
    assert torch.equal(payload[:, 1:], torch.zeros(num_reqs, max_sample_len - 1,
                                                   dtype=torch.int64))


def test_stale_slot_rows_are_filtered_before_restore():
    """Only still-valid rows may be scattered back into request state.

    A pending PP entry is consumed pp_size steps after it is received; a request
    can finish in that window and its state index be handed to a new request.
    Restoring through the unfiltered mapping would write the finished request's
    drafts into whoever owns the index now.
    """
    import numpy as np

    idx_mapping_np = np.array([3, 1, 2], dtype=np.int32)
    exclude_mask = np.array([False, True, False])  # row 1 finished
    drafts = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int64)

    valid_rows = np.flatnonzero(~exclude_mask)
    update_indices = np.stack((valid_rows, idx_mapping_np[valid_rows]))
    draft_rows = torch.from_numpy(update_indices[0]).to(torch.int64)
    draft_idx_mapping = torch.from_numpy(update_indices[1]).to(torch.int64)
    selected = drafts.index_select(0, draft_rows)

    # Row 1 (state index 1) must not be written at all.
    assert draft_idx_mapping.tolist() == [3, 2]
    assert selected.tolist() == [[10, 11], [30, 31]]
