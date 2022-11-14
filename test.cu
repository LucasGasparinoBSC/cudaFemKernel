#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define NELEM 1000000
#define NNODES 64
#define NGAUSS 64

extern "C" {

    __global__ void kernel0(int* listNodes, float* Ngp, float* phi, float* Rmass) {

        // Associate indexes to grid
        int iElem = blockIdx.x;
        int iNode = threadIdx.x;

        // Zero Rmass
        Rmass[listNodes[iElem*NNODES + iNode]] = 0.0f;

        // Create shared memory for phi and Re
        __shared__ float s_phi[NNODES];
        __shared__ float s_Re[NNODES];
        s_phi[iNode] = phi[listNodes[iElem*NNODES + iNode]];
        s_Re[iNode] = 0.0f;

        // Wait for all threads to finish
        __syncthreads();

        // Loop over gauss points
        for (int iGauss = 0; iGauss < NGAUSS; iGauss++) {
            // Compute dot(Ngp,s_phi)
            float aux = 0.0f;
            for (int jNode = 0; jNode < NNODES; jNode++) {
                aux += Ngp[iGauss*NNODES + jNode] * s_phi[jNode];
            }
            // Compute nodal residual
            s_Re[iNode] += aux * Ngp[iGauss*NNODES + iNode];
        }

        // Wait for all threads to finish
        __syncthreads();

        // Add the residual to the global residual
        atomicAdd(&Rmass[listNodes[iElem * NNODES + iNode]], s_Re[iNode]);
    }

    __global__ void kernel1(int* listNodes, float* Ngp, float* phi, float* Rmass) {

        // Associate indexes to grid
        int iElem = blockIdx.x;
        int iNode = threadIdx.x;
        int iGauss = blockIdx.y*blockDim.y + threadIdx.y;

        // Zero Rmass
        Rmass[listNodes[iElem*NNODES + iNode]] = 0.0f;

        // Create shared memory for phi Re
        __shared__ float s_phi[NNODES];
        __shared__ float s_aux[NGAUSS];
        __shared__ float s_Re[NNODES];
        s_phi[iNode] = phi[listNodes[iElem*NNODES + iNode]];
        s_aux[iGauss] = 0.0f;
        s_Re[iNode] = 0.0f;

        // Wait for all threads to finish
        __syncthreads();

        // Compute dot(Ngp,s_phi)
        s_aux[iGauss] += Ngp[iGauss*NNODES + iNode] * s_phi[iNode];

        // Wait for all threads to finish
        __syncthreads();

        // Compute nodal residual
        s_Re[iNode] += s_aux[iGauss] * Ngp[iGauss*NNODES + iNode];

        // Wait for all threads to finish
        __syncthreads();

        // Add the residual to the global residual
        atomicAdd(&Rmass[listNodes[iElem * NNODES + iNode]], s_Re[iNode]);
    }

}

int main() {

    // Geneerate dummy host data
    int* listNodes = new int[NELEM*NNODES];
    float* phi = new float[NELEM*NNODES];
    float* Rmass = new float[NELEM*NNODES];
    for (int i = 0; i < NELEM*NNODES; i++) {
        listNodes[i] = i;
        phi[i] = 1.0f;
        Rmass[i] = 0.0f;
    }

    // Generate device data
    int* d_listNodes;
    float* d_phi;
    float* d_Rmass;
    cudaMalloc((void**)&d_listNodes, NELEM*NNODES*sizeof(int));
    cudaMalloc((void**)&d_phi, NELEM*NNODES*sizeof(float));
    cudaMalloc((void**)&d_Rmass, NELEM*NNODES*sizeof(float));

    // Copy data to device
    cudaMemcpy(d_listNodes, listNodes, NELEM*NNODES*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi, phi, NELEM*NNODES*sizeof(float), cudaMemcpyHostToDevice);

    // Create dummy Ngp
    float* Ngp = new float[NGAUSS*NNODES];
    for (int i = 0; i < NGAUSS*NNODES; i++) {
        Ngp[i] = 1.0f;
    }
    float* d_Ngp;
    cudaMalloc((void**)&d_Ngp, NGAUSS*NNODES*sizeof(float));
    cudaMemcpy(d_Ngp, Ngp, NGAUSS*NNODES*sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    kernel0<<<NELEM,NNODES>>>(d_listNodes, d_Ngp, d_phi, d_Rmass);

    // Copy data back to host
    cudaMemcpy(Rmass, d_Rmass, NELEM*NNODES*sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < NNODES; i++) {
        std::cout << Rmass[i] << std::endl;
    }

    // Configure kernel1 grid
    dim3 block(NNODES,8,1);
    dim3 grid(NELEM,8,1);

    // Launch kernel1
    kernel1<<<grid,block>>>(d_listNodes, d_Ngp, d_phi, d_Rmass);

    // Copy data back to host
    cudaMemcpy(Rmass, d_Rmass, NELEM*NNODES*sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "kernel1" << std::endl;
    for (int i = 0; i < NNODES; i++) {
        std::cout << Rmass[i] << std::endl;
    }


    // End
    return 0;
}