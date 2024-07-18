#include <torch/script.h>
#include <tuple>
#include <map>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include <iostream>
#include "../xxhash.h"
#include <hnswlib.h>

// Network

#define HASH_SIZE 128
typedef std::bitset<HASH_SIZE> MYHASH;

class NetworkHash {
    private:
    int BATCH_SIZE;
    torch::jit::script::Module module;
    float* data;
    bool* memout;
    int* index;
    int cnt;

    public:
    NetworkHash(int BATCH_SIZE, char* module_name) {
        this->BATCH_SIZE = BATCH_SIZE;
        this->module = torch::jit::load(module_name);
        this->module.to(at::kCPU);
        this->module.eval();
        this->data = new float[BATCH_SIZE * BLOCK_SIZE];
        this->memout = new bool[BATCH_SIZE * HASH_SIZE];
        this->index = new int[BATCH_SIZE];
        this->cnt = 0;
    }
    ~NetworkHash() {
        delete[] this->data;
        delete[] this->memout;
        delete[] this->index;
    }
    bool push(char* ptr, int label);
    std::vector<std::pair<MYHASH, int>> request();
};

bool NetworkHash::push(char* ptr, int label) {
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        data[cnt * BLOCK_SIZE + i] = ((int)(unsigned char)(ptr[i]) - 128) / 128.0;
    }
    index[cnt++] = label;

    if (cnt == BATCH_SIZE) return true;
    else return false;
}

std::vector<std::pair<MYHASH, int>> NetworkHash::request() {
    if (cnt == 0) return std::vector<std::pair<MYHASH, int>>();

    std::vector<std::pair<MYHASH, int>> ret(cnt);

    std::vector<torch::jit::IValue> inputs;
    torch::Tensor t = torch::from_blob(data, {cnt, BLOCK_SIZE}).to(torch::kCPU);
    inputs.push_back(t);

    torch::Tensor output = module.forward(inputs).toTensor().cpu();

    torch::Tensor comp = output.ge(0.0);
    memcpy(memout, comp.cpu().data_ptr<bool>(), cnt * HASH_SIZE);

    bool* ptr = this->memout;

    for (int i = 0; i < cnt; ++i) {
        for (int j = 0; j < HASH_SIZE; ++j) {
            if (ptr[HASH_SIZE * i + j]) ret[i].first.flip(j);
        }
        ret[i].second = index[i];
    }

    cnt = 0;
    return ret;
}

// ANN

class ANN {
    private:
        int ANN_SEARCH_CNT, LINEAR_SIZE, NUM_THREAD, THRESHOLD;
        std::vector<MYHASH> linear;
        std::unordered_map<MYHASH, std::vector<int>> hashtable;
        hnswlib::L2Space space;
        hnswlib::HierarchicalNSW<float>* index;

        int lastDist; // Added to analyze the Hamming distances

    public:
    ANN(int ANN_SEARCH_CNT, int LINEAR_SIZE, int NUM_THREAD, int THRESHOLD, int MAX_ELEMENTS) 
        : space(HASH_SIZE), index(new hnswlib::HierarchicalNSW<float>(&space, MAX_ELEMENTS)) {
        this->ANN_SEARCH_CNT = ANN_SEARCH_CNT; // The number of candidates extract from ANN class
        this->LINEAR_SIZE = LINEAR_SIZE; // Size of linear buffer
        this->NUM_THREAD = NUM_THREAD;
        this->THRESHOLD = THRESHOLD;
        this->lastDist = -1; // Distance of the last request
    }
    ~ANN() {
        delete index;
    }
    int request(MYHASH h);
    void insert(MYHASH h, int label);
    int getLastDist() {
        return this->lastDist;
    }
};

int ANN::request(MYHASH h) {
    float query[HASH_SIZE];
    for (int i = 0; i < HASH_SIZE; ++i) {
        query[i] = h[i] ? 1.0 : 0.0;
    }

    auto result = index->searchKnn(query, this->ANN_SEARCH_CNT);

    int best_index = -1;
    float best_distance = std::numeric_limits<float>::max();

    while (!result.empty()) {
        auto res = result.top();
        result.pop();

        if (res.first < best_distance) {
            best_distance = res.first;
            best_index = res.second;
        }
    }

    if (best_distance <= THRESHOLD) {
        return best_index;
    } else {
        return -1;
    }
}

void ANN::insert(MYHASH h, int label) {
    float data[HASH_SIZE];
    for (int i = 0; i < HASH_SIZE; ++i) {
        data[i] = h[i] ? 1.0 : 0.0;
    }

    if (index->cur_element_count >= index->max_elements_) {
        std::cerr << "Index is full, cannot add more elements." << std::endl;
        return;
    }

    index->addPoint(data, label);
    hashtable[h].push_back(label);

    if (linear.size() == LINEAR_SIZE) {
        linear.clear();
    }
}
