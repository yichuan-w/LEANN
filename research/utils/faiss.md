sudo apt-get install libgflags-dev

uv pip install swig

sudo apt-get update && sudo apt-get install libzmq3-dev libmsgpack-dev pkg-config

/home/ubuntu/Power-RAG/.venv/bin/cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_GPU=OFF -DCMAKE_BUILD_TYPE=Debug . 

make -C build -j faiss && make -C build -j swigfaiss && uv pip install -e build/faiss/python



## Some outdated info (may not needeed)

&& cp ./build/faiss/python/_swigfaiss.so /home/andy/Power-RAG/.venv/lib/python3.10/site-packages/faiss-1.10.0-py3.10.egg/faiss/


export LD_PRELOAD="/lib/x86_64-linux-gnu/libiomp5.so /lib/x86_64-linux-gnu/libmkl_core.so /lib/x86_64-linux-gnu/libmkl_intel_lp64.so /lib/x86_64-linux-gnu/libmkl_intel_thread.so"

set -x LD_PRELOAD "/lib/x86_64-linux-gnu/libiomp5.so /lib/x86_64-linux-gnu/libmkl_core.so /lib/x86_64-linux-gnu/libmkl_intel_lp64.so /lib/x86_64-linux-gnu/libmkl_intel_thread.so"