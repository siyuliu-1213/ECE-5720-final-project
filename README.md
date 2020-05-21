# ECE-5720-final-project
The code repository for Cornell ECE 5720 final project.

<h4>1. KMP_seq.c</h4>
<p>The sequential implementation for KMP algorithm.</p>
<p>Compile : gcc -o KMP_seq KMP_seq.c</p>
<p>Run     : ./KMP_cuda</p>

<h4>2. KMP_m.c</h4>
<p>The MPI strategy parrallelism implementation for KMP algorithm.</p>
<p>Compile : mpicc KMP_m.c -o KMP_m</p>
<p>Run     : mpirun -np 64 ./KMP_m --mca opal_warn_on_missing_libcuda 0</p>

<h4>3. KMP_cuda.cu</h4>
<p>The CUDA strategy parrallelism implementation for KMP algorithm.</p>
<p>Compile : /usr/local/cuda-10.1/bin/nvcc -arch=compute_52 -o KMP_cuda KMP_cuda.cu
<p>Run     : ./KMP_cuda</p>
