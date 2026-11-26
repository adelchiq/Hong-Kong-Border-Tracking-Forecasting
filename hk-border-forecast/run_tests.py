"""Run tests without pytest: python run_tests.py"""
import tests.test_pipeline as t
for name in [n for n in dir(t) if n.startswith("test_")]:
    getattr(t, name)(); print("PASS", name)
