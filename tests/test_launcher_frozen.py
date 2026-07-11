from synapse_launcher import SCRIPTS, SCRIPT_MODULES, build_parser


def test_every_script_has_frozen_handler():
    for script_name in SCRIPTS.values():
        assert script_name in SCRIPT_MODULES


def test_frozen_handlers_are_importable():
    import importlib

    for module_name in SCRIPT_MODULES.values():
        module = importlib.import_module(module_name)
        assert hasattr(module, "main")

    for module_name in (
        "src.ui.dialogs",
        "src.ui.theme",
        "utils.onboarding_progress",
        "utils.product",
        "utils.user_profiles",
    ):
        importlib.import_module(module_name)


def test_home_is_a_launch_command():
    parser = build_parser()
    command_action = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert "home" in command_action.choices
    assert "home" in SCRIPTS


def test_needs_cli_console_for_help_and_utilities():
    from synapse_launcher import _needs_cli_console

    assert not _needs_cli_console([])
    assert not _needs_cli_console(["home"])
    assert not _needs_cli_console(["--tray"])
    assert _needs_cli_console(["--help"])
    assert _needs_cli_console(["data"])
    assert _needs_cli_console(["settings"])
