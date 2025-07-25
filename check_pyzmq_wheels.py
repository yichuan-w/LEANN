#!/usr/bin/env python3
"""Check which pyzmq versions have manylinux2014 wheels available."""

import json
import urllib.request


def check_pyzmq_wheels():
    url = "https://pypi.org/pypi/pyzmq/json"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())

    versions_with_manylinux2014 = {}

    for version, releases in data["releases"].items():
        manylinux_wheels = []
        for release in releases:
            filename = release["filename"]
            if "manylinux2014" in filename or "manylinux_2_17" in filename:
                if "cp310" in filename:  # Python 3.10
                    manylinux_wheels.append(filename)

        if manylinux_wheels:
            versions_with_manylinux2014[version] = manylinux_wheels

    # Sort versions
    from packaging.version import parse

    sorted_versions = sorted(
        versions_with_manylinux2014.keys(), key=parse, reverse=True
    )

    print("PyZMQ versions with manylinux2014 wheels for Python 3.10:")
    for version in sorted_versions[:10]:  # Show top 10
        print(f"  {version}")
        for wheel in versions_with_manylinux2014[version]:
            print(f"    - {wheel}")


if __name__ == "__main__":
    check_pyzmq_wheels()
