from utils.product import MODULE_LABELS, PRODUCT_NAME, VERSION


def test_product_metadata():
    assert PRODUCT_NAME == "Synapse"
    assert VERSION
    assert "test_monitor" in MODULE_LABELS
    assert MODULE_LABELS["test_monitor"] == "Monitor"
