import pytest
import imaging.bracket_hdr_deghost as mod

CASES = [
    ("local_align", (1,)),
    ("ghost_mask", (1,)),
    ("ev_vignette_correct", (1,)),
    ("decode_barrier", (1,)),
    ("rollback_all", (1,)),
    ("tone_map_out", (1,)),
    ("load_bracket", (1,)),
    ("align", (1,)),
    ("merge", (1,)),
    ("compose", (1,)),
    ("validate_ghost", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
