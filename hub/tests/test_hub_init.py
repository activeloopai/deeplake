import hub
import hub.config


def test_local_mode():
    hub.local_mode()
    assert hub.config.HUB_REST_ENDPOINT == "http://localhost:5000"


def test_dev_mode():
    hub.dev_mode()
    assert hub.config.HUB_REST_ENDPOINT == "https://app.dev.activeloop.ai"


def test_load(caplog):
    obj = hub.load("./data/new/test")
    assert "Deprecated Warning" in caplog.text
    obj = hub.load("./data/test/test_dataset2")
    assert isinstance(obj, hub.Dataset) == True


if __name__ == "__main__":
    test_local_mode()
    test_dev_mode()
    # test_load()
