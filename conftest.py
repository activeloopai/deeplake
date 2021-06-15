import pytest

from humbug.consent import HumbugConsent

from hub_v1.report import hub_reporter


@pytest.fixture(scope="session", autouse=True)
def reporting_off():
    hub_reporter.consent = HumbugConsent(False)
