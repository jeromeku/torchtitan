#uv pip install torch --index-url https://download.pytorch.org/whl/nightly/cu129
#uv pip install -r requirements-dev.txt
USE_CUDA=1 uv pip install -e thirdparty/ao --no-build-isolation -v 2>&1 | tee _ao.install.log
