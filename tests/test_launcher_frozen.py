from synapse_launcher import SCRIPTS, SCRIPT_MODULES


def test_every_script_has_frozen_handler():
    for script_name in SCRIPTS.values():
        assert script_name in SCRIPT_MODULES


def test_frozen_handlers_are_importable():
    import importlib

    for module_name in SCRIPT_MODULES.values():
        module = importlib.import_module(module_name)
        assert hasattr(module, "main")
