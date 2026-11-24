"""Write a JSON manifest (file name -> sha256, size) of a download cache."""
import hashlib
import json
import os
import sys
import time

d, out = sys.argv[1], sys.argv[2]
files = {}
for name in sorted(os.listdir(d)):
    p = os.path.join(d, name)
    if os.path.isfile(p) and name != os.path.basename(out):
        with open(p, "rb") as fh:
            files[name] = {"sha256": hashlib.sha256(fh.read()).hexdigest(), "bytes": os.path.getsize(p)}
with open(out, "w") as fh:
    json.dump({"created": time.strftime("%Y-%m-%d"), "files": files}, fh, indent=1)
