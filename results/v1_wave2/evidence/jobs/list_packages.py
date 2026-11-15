"""Print the interpreter version and every installed distribution, sorted."""
import importlib.metadata as m
import sys

print("# python", sys.version.split()[0], sys.executable)
for d in sorted(m.distributions(), key=lambda d: d.metadata["Name"].lower()):
    print("%s==%s" % (d.metadata["Name"], d.version))
