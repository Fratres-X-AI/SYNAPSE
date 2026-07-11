from synapse_launcher import SCRIPTS, SCRIPT_MODULES, build_parser


def test_every_script_has_frozen_handler():
    for script_name in SCRIPTS.values():
        assert script_name in SCRIPT_MODULES


def test_frozen_handlers_are_importable():
    import importlib

    for module_name in SCRIPT_MODULES.values():
        module = importlib.import_module(module_name)
        assert hasattr(module, "main")


def test_home_is_a_launch_command():
    parser = build_parser()
    command_action = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert "home" in command_action.choices
    assert "home" in SCRIPTS
