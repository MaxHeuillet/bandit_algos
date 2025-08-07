# utils/ssl_tools.py
import contextlib
import warnings
import requests
from urllib3.exceptions import InsecureRequestWarning

_old_merge_env = requests.Session.merge_environment_settings


@contextlib.contextmanager
def no_ssl_verification():
    """
    Temporarily disable TLS/SSL certificate verification for *all* requests
    issued inside this context manager.

    Example
    -------
    >>> from utils.ssl_tools import no_ssl_verification
    >>> with no_ssl_verification():
    ...     requests.get("https://self‑signed.badssl.com/")
    """
    opened_adapters = set()

    def _patched(self, url, proxies, stream, verify, cert):
        opened_adapters.add(self.get_adapter(url))
        cfg = _old_merge_env(self, url, proxies, stream, verify, cert)
        cfg["verify"] = False
        return cfg

    requests.Session.merge_environment_settings = _patched

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = _old_merge_env
        for ad in opened_adapters:
            with contextlib.suppress(Exception):
                ad.close()