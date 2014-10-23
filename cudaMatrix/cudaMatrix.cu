#include	<stdio.h>
#include	<malloc.h>
#include	<stdlib.h>
#include	<cutil_inline.h>

// �s��̃T�C�Y�A�u���b�N�T�C�Y
#define	MATRIX_SIZE	4096
#define	BLOCK_SIZE	128

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC);

int
main(int argc, char** argv)
{
	// �ϐ��錾
	unsigned int matrixSize = sizeof(unsigned int) * MATRIX_SIZE * MATRIX_SIZE;
	int* hMatrixA;
	int* hMatrixB;
	int* hMatrixC;
	hMatrixA = (int*) malloc(matrixSize);
	hMatrixB = (int*) malloc(matrixSize);

	// �����l�ݒ�
	unsigned int	col_idx, row_idx;
	for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++) {
		for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++) {
			hMatrixA[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024 * 1024);
			hMatrixB[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024 * 1024);
		}
	}

	// �f�o�C�X���̕ϐ�
	int* dMatrixA;
	int* dMatrixB;
	int* dMatrixC;

	// �f�o�C�X�������̊m��
	cutilSafeCall(cudaMalloc((void**) &dMatrixA, matrixSize));
	cutilSafeCall(cudaMemcpy(dMatrixA, hMatrixA, matrixSize, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMalloc((void**) &dMatrixB, matrixSize));
	cutilSafeCall(cudaMemcpy(dMatrixB, hMatrixB, matrixSize, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMalloc((void**) &dMatrixC, matrixSize));

	// �u���b�N�T�C�Y�ƃO���b�h�T�C�Y�̐ݒ�
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(MATRIX_SIZE / BLOCK_SIZE, MATRIX_SIZE / BLOCK_SIZE);

	// �^�C�}�[�ϐ��̐錾�A����J�n
	printf("Matrix calculation start in the GPU!\n");
	printf("Matrix size\t:\t%d * %d\n", MATRIX_SIZE, MATRIX_SIZE);
	printf("BlockSize\t:\t%d\nGridSize\t:\t%d\n", BLOCK_SIZE, grid);
	float millseconds = 0.0f;
	float sum = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < 10; i++){
		cudaEventRecord(start, 0);

		// �J�[�l���̋N��
		matrixMul <<<grid, block>>>(dMatrixA, dMatrixB, dMatrixC);
		cudaThreadSynchronize();

		// ����I���A���ʕ\��
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&millseconds, start, stop);
		printf("Time required\t:\t%f millseconds\n", millseconds);
		sum += millseconds;
	}
//	printf("Matrix size\t:\t%d * %d\n", MATRIX_SIZE, MATRIX_SIZE);
//	printf("BlockSize\t:\t%d\nGridSize\t:\t%d\n", BLOCK_SIZE, MATRIX_SIZE / BLOCK_SIZE);
	printf("Time average\t:\t%f millseconds\n", sum /10);

	// ���ʂ̗̈�̊m�ۂƁA�f�o�C�X������̃������]��
	hMatrixC = (int*) malloc(matrixSize);
	cutilSafeCall(cudaMemcpy(hMatrixC, dMatrixC, matrixSize, cudaMemcpyDeviceToHost));

	// �������J��
	free(hMatrixA);
	free(hMatrixB);
	free(hMatrixC);
	cutilSafeCall(cudaFree(dMatrixA));
	cutilSafeCall(cudaFree(dMatrixB));
	cutilSafeCall(cudaFree(dMatrixC));

	cudaThreadExit();
}

// �s��v�Z������J�[�l���֐�
__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC)
{
	unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int scan_idx;
	int	target = 0;

	for (scan_idx = 0; scan_idx < MATRIX_SIZE; scan_idx++) {
		// �ΏۂƂȂ镔�������������̂𑫂��Ă���
		target += inMatrixA[col_idx * MATRIX_SIZE + scan_idx] * inMatrixB[scan_idx * MATRIX_SIZE + row_idx];
		__syncthreads();	// �X���b�h����
	}
	inMatrixC[col_idx * MATRIX_SIZE + row_idx] = target;
}
