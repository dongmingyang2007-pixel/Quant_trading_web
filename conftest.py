import os

import django
from django.conf import settings


# Ensure Django settings are configured before importing app modules in tests.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quant_trading_site.settings")
# Use an isolated in-memory database for tests to avoid clobbering dev data.
settings.DATABASES["default"]["NAME"] = ":memory:"
settings.DATABASES["default"]["TEST"] = {"NAME": ":memory:"}
settings.ALLOWED_HOSTS = ["testserver", "localhost", "127.0.0.1"]
django.setup()

# Create database schema in memory for tests.
from django.core.management import call_command  # noqa: E402

call_command("migrate", run_syncdb=True, verbosity=0)
