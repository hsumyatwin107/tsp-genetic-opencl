#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <limits>
#include <chrono>
#include <OpenCL/opencl.h>

#define NUM_CITIES 10
#define POP_SIZE 10
#define GENERATIONS 10
#define MUTATION_RATE 0.1f

std::vector<std::vector<float>> cities(NUM_CITIES, std::vector<float>(2));
std::vector<std::vector<float>> distanceMatrix(NUM_CITIES, std::vector<float>(NUM_CITIES));

float euclideanDistance(int i, int j) {
    float dx = cities[i][0] - cities[j][0];
    float dy = cities[i][1] - cities[j][1];
    return sqrt(dx * dx + dy * dy);
}

void generateCities() {
    srand(time(0));
    for (int i = 0; i < NUM_CITIES; ++i) {
        cities[i][0] = static_cast<float>(rand() % 1000);
        cities[i][1] = static_cast<float>(rand() % 1000);
    }
}

void generateDistanceMatrixCPU() {
    for (int i = 0; i < NUM_CITIES; ++i)
        for (int j = 0; j < NUM_CITIES; ++j)
            distanceMatrix[i][j] = euclideanDistance(i, j);
}

const char* kernelSource = R"(
__kernel void computeFitness(__global const float* distMatrix, __global const int* routes, __global float* fitnesses, const int numCities) {
    int gid = get_global_id(0);
    float total = 0.0f;
    for (int i = 0; i < numCities - 1; i++) {
        int from = routes[gid * numCities + i];
        int to = routes[gid * numCities + i + 1];
        total += distMatrix[from * numCities + to];
    }
    int last = routes[gid * numCities + numCities - 1];
    int first = routes[gid * numCities];
    total += distMatrix[last * numCities + first];
    fitnesses[gid] = total;
}
)";

std::vector<std::vector<int>> generateInitialPopulation() {
    std::vector<std::vector<int>> population;
    std::vector<int> baseRoute(NUM_CITIES);
    for (int i = 0; i < NUM_CITIES; ++i) baseRoute[i] = i;

    for (int i = 0; i < POP_SIZE; ++i) {
        static std::mt19937 gen(static_cast<unsigned>(std::time(nullptr)));
        std::shuffle(baseRoute.begin(), baseRoute.end(), gen);
        population.push_back(baseRoute);
    }
    return population;
}

std::vector<float> runOpenCL(const std::vector<std::vector<int>>& population) {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, nullptr);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    kernel = clCreateKernel(program, "computeFitness", &err);

    std::vector<int> flatRoutes;
    for (const auto& route : population)
        flatRoutes.insert(flatRoutes.end(), route.begin(), route.end());

    std::vector<float> flatDist;
    for (const auto& row : distanceMatrix)
        flatDist.insert(flatDist.end(), row.begin(), row.end());

    size_t routeSize = flatRoutes.size() * sizeof(int);
    size_t distSize = flatDist.size() * sizeof(float);
    std::vector<float> fitnesses(POP_SIZE);

    cl_mem distBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, distSize, flatDist.data(), &err);
    cl_mem routeBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, routeSize, flatRoutes.data(), &err);
    cl_mem fitnessBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, POP_SIZE * sizeof(float), nullptr, &err);

    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &distBuf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &routeBuf);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &fitnessBuf);
    int numCities = NUM_CITIES;
    err |= clSetKernelArg(kernel, 3, sizeof(int), &numCities);

    size_t globalWorkSize = POP_SIZE;
    cl_event kernelEvent;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, &kernelEvent);

    clWaitForEvents(1, &kernelEvent);

    cl_ulong startTime, endTime;
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, nullptr);
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, nullptr);

    double executionTimeMs = (endTime - startTime) * 1e-6;
    std::cout << "Kernel Execution Time: " << executionTimeMs << " ms\n";

    err = clEnqueueReadBuffer(queue, fitnessBuf, CL_TRUE, 0, POP_SIZE * sizeof(float), fitnesses.data(), 0, nullptr, nullptr);

    clReleaseMemObject(distBuf);
    clReleaseMemObject(routeBuf);
    clReleaseMemObject(fitnessBuf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseEvent(kernelEvent);

    return fitnesses;
}

std::vector<int> tournamentSelection(const std::vector<std::vector<int>>& population, const std::vector<float>& fitnesses) {
    int best = rand() % POP_SIZE;
    for (int i = 0; i < 2; ++i) {
        int competitor = rand() % POP_SIZE;
        if (fitnesses[competitor] < fitnesses[best])
            best = competitor;
    }
    return population[best];
}

std::vector<int> crossover(const std::vector<int>& parent1, const std::vector<int>& parent2) {
    int start = rand() % NUM_CITIES;
    int end = start + rand() % (NUM_CITIES - start);
    std::vector<int> child(NUM_CITIES, -1);
    for (int i = start; i < end; ++i)
        child[i] = parent1[i];

    int p2_index = 0;
    for (int i = 0; i < NUM_CITIES; ++i) {
        if (std::find(child.begin(), child.end(), parent2[p2_index]) == child.end()) {
            for (int j = 0; j < NUM_CITIES; ++j) {
                if (child[j] == -1) {
                    child[j] = parent2[p2_index];
                    break;
                }
            }
        }
        ++p2_index;
    }
    return child;
}

void mutate(std::vector<int>& route) {
    if ((float)rand() / RAND_MAX < MUTATION_RATE) {
        int i = rand() % NUM_CITIES;
        int j = rand() % NUM_CITIES;
        std::swap(route[i], route[j]);
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now(); // Start timing

    generateCities();

    auto startCPU = std::chrono::high_resolution_clock::now();
    generateDistanceMatrixCPU();
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Distance Matrix Time: " << std::chrono::duration<double>(endCPU - startCPU).count() << " seconds\n";

    std::vector<std::vector<int>> population = generateInitialPopulation();
    std::vector<float> fitnesses;
    float bestDistance = std::numeric_limits<float>::max();
    std::vector<int> bestRoute;

    for (int gen = 0; gen < GENERATIONS; ++gen) {
        std::cout << "\nGeneration " << gen + 1 << ":\n";

        fitnesses = runOpenCL(population);

        for (int i = 0; i < std::min(10, POP_SIZE); ++i) {
            std::cout << "Route " << i + 1 << ": ";
            for (int city : population[i]) {
                std::cout << city << " ";
            }
            std::cout << " | Distance: " << fitnesses[i] << "\n";
        }

        int bestIdx = std::min_element(fitnesses.begin(), fitnesses.end()) - fitnesses.begin();
        if (fitnesses[bestIdx] < bestDistance) {
            bestDistance = fitnesses[bestIdx];
            bestRoute = population[bestIdx];
        }

        std::vector<std::vector<int>> newPopulation;
        for (int i = 0; i < POP_SIZE; ++i) {
            auto parent1 = tournamentSelection(population, fitnesses);
            auto parent2 = tournamentSelection(population, fitnesses);
            auto child = crossover(parent1, parent2);
            mutate(child);
            newPopulation.push_back(child);
        }
        population = newPopulation;
    }

    std::cout << "\nBest Route Found:\n";
    for (int city : bestRoute) std::cout << city << " ";
    std::cout << "\nMinimum Distance: " << bestDistance << "\n";

    auto end = std::chrono::high_resolution_clock::now(); // End timing
    std::cout << "Total Execution Time: " << std::chrono::duration<double>(end - start).count() << " seconds\n";

    return 0;
}