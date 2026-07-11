import importlib.util
import sys
from pathlib import Path

ONBOARD = Path(__file__).resolve().parents[1] / "test_onboard.py"
SPEC = importlib.util.spec_from_file_location("onboard_module", ONBOARD)
onboard_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = onboard_module
SPEC.loader.exec_module(onboard_module)


def test_expression_arm_key_matches_current_step_letter():
    mad_step = onboard_module.EXPRESSION_STEPS[-1]
    neutral_step = onboard_module.EXPRESSION_STEPS[0]
    assert onboard_module._expression_arm_key(mad_step, ord("m"))
    assert onboard_module._expression_arm_key(mad_step, ord("M"))
    assert not onboard_module._expression_arm_key(mad_step, ord("n"))
    assert onboard_module._expression_arm_key(neutral_step, ord("n"))


def test_expression_confirm_key_is_space_only():
    assert onboard_module._expression_confirm_key(ord(" "))
    assert not onboard_module._expression_confirm_key(ord("m"))
    assert not onboard_module._expression_confirm_key(ord("M"))
