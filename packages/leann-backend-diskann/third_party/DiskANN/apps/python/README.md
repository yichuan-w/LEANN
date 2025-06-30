<!-- Copyright (c) Microsoft Corporation. All rights reserved.
   Licensed under the MIT license. -->

# Integration Tests
The following tests use Python to prepare, run, verify, and tear down the rest api services.

We do make use of the built-in `unittest` library, but that's only to take advantage of test reporting purposes.

These are decidedly **not** _unit_ tests. These are end to end integration tests.

## Caveats
This has only been tested or built for Linux, though we have written platform agnostic Python for the smoke test 
(i.e. using `os.path.join`, etc)

It has been tested on Python 3.9 and 3.10, but should work on Python 3.6+.

## How to Run

First, build the DiskANN RestAPI code; see $REPOSITORY_ROOT/workflows/rest_api.md for detailed instructions.

```bash
cd tests/python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export DISKANN_BUILD_DIR=/path/to/your/diskann/build
python -m unittest
```

## Smoke Test Failed, Now What?
The smoke test written takes advantage of temporary directories that are only valid during the 
lifetime of the test. The contents of these directories include:
- Randomized vectors (first in tsv, then bin form) used to build the PQFlashIndex
- The PQFlashIndex files

It is useful to keep these around. By setting some environment variables, you can control whether an ephemeral,
temporary directory is used (and deleted on test completion), or left as an exercise for the developer to 
clean up.  

The valid environment variables are:
- `DISKANN_REST_TEST_WORKING_DIR` (example: `$USER/DiskANNRestTest`)
  - If this is specified, it **must exist** and **must be writeable**. Any existing files will be clobbered.
- `DISKANN_REST_SERVER` (example: `http://127.0.0.1:10067`)
  - Note that if this is set, no data will be generated, nor will a server be started; it is presumed you have done 
    all the work in creating and starting the rest server prior to running the test and just submits requests against it.
