import json

from . import scaffold_info


def main() -> None:
    info = scaffold_info()
    if not info["gloo_available"]:
        raise RuntimeError("the CPU correctness path requires PyTorch Gloo support")
    print(json.dumps(info, sort_keys=True))


if __name__ == "__main__":
    main()
