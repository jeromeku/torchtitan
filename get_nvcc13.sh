CUDA_HOME=$(cat << 'EOF' | python
import nvidia.cu13
cuda_home = nvidia.cu13.__path__[0]
print(cuda_home)
EOF
)
CUDA_COMPILER=${CUDA_HOME}/bin/nvcc
echo $CUDA_HOME
echo $CUDA_COMPILER
$CUDA_COMPILER --version