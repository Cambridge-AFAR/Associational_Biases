import os
import sys

from utils.data import Data

if __name__ == "__main__":
    exceptions = []
    for p in os.listdir("prompts"):
        if os.path.splitext(p)[1] != ".yaml":
            continue
        path = os.path.join("prompts", p)
        try:
            f = Data(path)
        except Exception as e:
            print(f"#####  Exception raised at {p} #####")
            print(e)
            exceptions.append(e)

    if len(exceptions) > 0:
        sys.exit(1)
