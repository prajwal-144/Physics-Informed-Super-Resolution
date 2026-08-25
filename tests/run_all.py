"""Run every test that does not need torch. Runnable from any working directory:
       python sie_pipeline/tests/run_all.py

The data-dependent tests locate Model_A relative to this file, so no particular
cwd is required. Override with the MODELA_ROOT environment variable if Model_A
lives outside the repository (MODELA_ROOT must be the directory *containing*
Model_A, not Model_A itself).
"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ["test_lens_models.py", "test_sources.py", "test_jacobian.py",
         "test_raytrace.py", "test_theta_e_init.py"]
def main():
    fails = []
    for t in TESTS:
        print("\n" + "=" * 78); print(t); print("=" * 78)
        r = subprocess.run([sys.executable, os.path.join(HERE, t)])
        if r.returncode != 0:
            fails.append(t)
    print("\n" + "=" * 78)
    print("ALL PASSED" if not fails else "FAILED: " + ", ".join(fails))
    print("test_backend_parity.py needs torch and is not run here.")
    return 1 if fails else 0
if __name__ == "__main__":
    raise SystemExit(main())

