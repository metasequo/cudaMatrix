#include	<stdio.h>
#include	<malloc.h>
#include	<stdlib.h>
#include	<cutil_inline.h>

// 行列のサイズ、ブロックサイズ
#define	MATRIX_SIZE	4096
#define	BLOCK_SIZE	128

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC);

int
main(int argc, char** argv)
{
	// 変数宣言
	unsigned int matrixSize = sizeof(unsigned int) * MATRIX_SIZE * MATRIX_SIZE;
	int* hMatrixA;
	int* hMatrixB;
	int* hMatrixC;
	hMatrixA = (int*) malloc(matrixSize);
	hMatrixB = (int*) malloc(matrixSize);

	// 初期値設定
	unsigned int	col_idx, row_idx;
	for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++) {
		for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++) {
			hMatrixA[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024 * 1024);
			hMatrixB[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024 * 1024);
		}
	}

	// デバイス側の変数
	int* dMatrixA;
	int* dMatrixB;
	int* dMatrixC;

	// デバイスメモリの確保
	cutilSafeCall(cudaMalloc((void**) &dMatrixA, matrixSize));
	cutilSafeCall(cudaMemcpy(dMatrixA, hMatrixA, matrixSize, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMalloc((void**) &dMatrixB, matrixSize));
	cutilSafeCall(cudaMemcpy(dMatrixB, hMatrixB, matrixSize, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMalloc((void**) &dMatrixC, matrixSize));

	// ブロックサイズとグリッドサイズの設定
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(MATRIX_SIZE / BLOCK_SIZE, MATRIX_SIZE / BLOCK_SIZE);

	// タイマー変数の宣言、測定開始
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

		// カーネルの起動
		matrixMul <<<grid, block>>>(dMatrixA, dMatrixB, dMatrixC);
		cudaThreadSynchronize();

		// 測定終了、結果表示
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&millseconds, start, stop);
		printf("Time required\t:\t%f millseconds\n", millseconds);
		sum += millseconds;
	}
//	printf("Matrix size\t:\t%d * %d\n", MATRIX_SIZE, MATRIX_SIZE);
//	printf("BlockSize\t:\t%d\nGridSize\t:\t%d\n", BLOCK_SIZE, MATRIX_SIZE / BLOCK_SIZE);
	printf("Time average\t:\t%f millseconds\n", sum /10);

	// 結果の領域の確保と、デバイス側からのメモリ転送
	hMatrixC = (int*) malloc(matrixSize);
	cutilSafeCall(cudaMemcpy(hMatrixC, dMatrixC, matrixSize, cudaMemcpyDeviceToHost));

	// メモリ開放
	free(hMatrixA);
	free(hMatrixB);
	free(hMatrixC);
	cutilSafeCall(cudaFree(dMatrixA));
	cutilSafeCall(cudaFree(dMatrixB));
	cutilSafeCall(cudaFree(dMatrixC));

	cudaThreadExit();
}

// 行列計算をするカーネル関数
__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC)
{
	unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int scan_idx;
	int	target = 0;

	for (scan_idx = 0; scan_idx < MATRIX_SIZE; scan_idx++) {
		// 対象となる部分をかけたものを足していく
		target += inMatrixA[col_idx * MATRIX_SIZE + scan_idx] * inMatrixB[scan_idx * MATRIX_SIZE + row_idx];
		__syncthreads();	// スレッド同期
	}
	inMatrixC[col_idx * MATRIX_SIZE + row_idx] = target;
}
