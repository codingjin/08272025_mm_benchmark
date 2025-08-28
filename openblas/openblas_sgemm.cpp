#include <cblas.h>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <string>
#include <random>
#include <cmath>
#include <unordered_map>
#include <cassert>
#include <unordered_set>
#include <cstdlib>
#include <cstring>

const std::unordered_set<std::string> validcpu = {"r9", "i7", "rt"};
const std::unordered_map<std::string, int> maxthreadnum{
    {"r9", 32}, 
    {"i7", 20},
    {"rt", 128}
};

const int WARMUP = 100;
const int RUNS = 1000;
int MAXTHREADNUM;
const float ERR = 0.001;
double FLOPs;

void matmul(const float * const A, const float * const B, float * const C, const int M, const int K, const int N);
void compare(const float * const A, const float * const B, const int M, const int N);
void initInputs(float * const A, float * const B, const int M, const int K, const int N);
void printResults(const std::string& name, const std::vector<double>& results, const double FLOPs);
double openblas(const float *A, const float *B, float *C, const int M, const int K, const int N);


int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr << "Invalid Usage!\n";
        std::cout << "Usage: ./openblas_sgemm M K N CPU\n";
        std::cout << "M N K are positive Integer for problem sizes, CPU could be [r9, i7, rt]\n";
        exit(1);
    }

    int M = std::stoi(argv[1]);
    int K = std::stoi(argv[2]);
    int N = std::stoi(argv[3]);
    std::string CPU = argv[4];
    assert(M > 0 && K > 0 && N > 0 && validcpu.count(CPU));
    MAXTHREADNUM = maxthreadnum.at(CPU);
    FLOPs = 2.0 * M * K * N;

    std::cout << "M=" << M << " K=" << K << " N=" << N << " CPU=" << CPU << " MaxThreadnum=" << MAXTHREADNUM << "\n";
    float *A = (float*)malloc(sizeof(float) * M * K);
    float *B = (float*)malloc(sizeof(float) * K * N);
    initInputs(A, B, M, K, N);
    float *C = (float*)malloc(sizeof(float) * M * N);
    std::memset(C, 0, sizeof(float) * M * N);
    float *D = (float*)malloc(sizeof(float) * M * N);
    std::memset(D, 0, sizeof(float) * M * N);

    /*
    // check correctness first
    matmul(A, B, D, M, K, N);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1,      // Alpha
        A, K,   // A and strides between rows
        B, N,   // B and strides between rows
        0,      // Beta
        C, N    // C and strides between rows
    );
    compare(C, D, M, N);
    */

    for (int threadnum = 1; threadnum <= MAXTHREADNUM; threadnum <<= 1){
        std::vector<double> timings;
        openblas_set_num_threads(threadnum);
        for (int i = 0; i < WARMUP; ++i) cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
        for (int i = 0; i < RUNS; ++i) timings.push_back(openblas(A, B, C, M, K, N));
        std::sort(timings.begin(), timings.end());
        printResults("ThreadNum=" + std::to_string(threadnum), timings, FLOPs);
    }

    // check correctness last
    //compare(C, D, M, N);
    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}

double openblas(const float *A, const float *B, float *C, const int M, const int K, const int N)
{
    auto start = std::chrono::high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

void initInputs(float * const A, float * const B, const int M, const int K, const int N)
{
    std::mt19937 engine{137};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) A[i] = dist(engine);
    for (int i = 0; i < K * N; ++i) B[i] = dist(engine);
}

void matmul(const float * const A, const float * const B, float * const C, const int M, const int K, const int N)
{
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

void compare(const float * const A, const float * const B, const int M, const int N)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (std::abs(A[i * N + j] - B[i * N + j]) > ERR) {
                std::cerr << "Correctness check failed!\n" << "M=" << M << " N=" << N << std::endl;
                std::cerr << "i=" << i << ", j=" << j << " A=" << A[i*M+j] << ", B=" << B[i*M+j] << std::endl;
                exit(1);
            }
        }
    }
}

void printResults(const std::string& name, const std::vector<double>& results, const double FLOPs)
{
    double median = results[results.size()/2];
    std::cout << name << "\tPerformance=" << std::lround(FLOPs/1.0e9/median) << " GFLOPs\n\n";
    /*
    double total = std::accumulate(results.begin(), results.end(), 0.0);
    double avg = total/results.size();
    double median = results[results.size()/2];
    double min = results[0];
    double dev = 0.0;

    for (const auto re : results)
        dev += (re - avg) * (re - avg);
    dev /= results.size();

    std::cout << "=== " << name << " ===\n";
    std::cout << "Took " << total << " seconds for " << RUNS << " runs.\t" << WARMUP << " warmups.\n";
    std::cout << avg << " Avg.\t(" << FLOPs/1.0e9/avg << " GFLOPS)\n";
    std::cout << median << " Med.\t(" << FLOPs/1.0e9/median << " GFLOPS)\n";
    std::cout << min << " Max.\t(" << FLOPs/1.0e9/min << " GFLOPS)\n";
    std::cout << dev << " Dev.\t(" << dev << ")\n\n";
    */
}



