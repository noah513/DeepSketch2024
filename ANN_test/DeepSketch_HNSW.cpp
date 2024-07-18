#include <iostream>
#include <vector>
#include <set>
#include <bitset>
#include <map>
#include <cmath>
#include <algorithm>
#include "../compress.h"
#include "ann_HNSW.h"
#include "../lz4.h"
#include "../xxhash.h"
extern "C" {
    #include "../xdelta3/xdelta3.h"
}
#define INF 987654321
using namespace std;

typedef pair<int, int> ii;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "usage: ./ann_inf [input_file] [script_module] [threshold] [output_hash_dist]\n";
        exit(0);
    }
    int threshold = atoi(argv[3]);
    int max_elements = 25000; // 최대 요소 수를 충분히 크게 설정

    DATA_IO f(argv[1]);
    f.read_file();

    int dist_histogram[HASH_SIZE];
    int comp_size_for_dist[HASH_SIZE];
    memset(dist_histogram, 0, sizeof(int)*HASH_SIZE);
    memset(comp_size_for_dist, 0, sizeof(int)*HASH_SIZE);

    map<XXH64_hash_t, int> dedup;
    list<ii> dedup_lazy_recipe;
    NetworkHash network(256, argv[2]);

    ANN ann(20, 128, 16, threshold, max_elements);
    
    unsigned long long total = 0;
    f.time_check_start();
    for (int i = 0; i < f.N; ++i) {
        XXH64_hash_t h = XXH64(f.trace[i], BLOCK_SIZE, 0);

        if (dedup.count(h)) { // deduplication
            dedup_lazy_recipe.push_back({i, dedup[h]});
            continue;
        }

        dedup[h] = i;

        if (network.push(f.trace[i], i)) {
            vector<pair<MYHASH, int>> myhash = network.request(); // hash 생성 (sketch) 
            for (int j = 0; j < myhash.size(); ++j) {
                RECIPE r;

                MYHASH& h = myhash[j].first;
                int index = myhash[j].second;

                int comp_self = LZ4_compress_default(f.trace[index], compressed, BLOCK_SIZE, 2 * BLOCK_SIZE);
                int dcomp_ann = INF, dcomp_ann_ref;
                dcomp_ann_ref = ann.request(h);

                if (dcomp_ann_ref != -1) {
                    dcomp_ann = xdelta3_compress(f.trace[index], BLOCK_SIZE, f.trace[dcomp_ann_ref], BLOCK_SIZE, delta_compressed, 1);
                }

                set_offset(r, total);

                if (min(comp_self, BLOCK_SIZE) > dcomp_ann) { // delta compress
                    set_size(r, (unsigned long)(dcomp_ann - 1));
                    set_ref(r, dcomp_ann_ref);
                    set_flag(r, 0b11);
                    f.write_file(delta_compressed, dcomp_ann);
                    dist_histogram[ann.getLastDist()]++;
                    comp_size_for_dist[ann.getLastDist()] += dcomp_ann;
                    total += dcomp_ann;
                }
                else {
                    if (comp_self < BLOCK_SIZE) { // self compress
                        set_size(r, (unsigned long)(comp_self - 1));
                        set_flag(r, 0b01);
                        f.write_file(compressed, comp_self);
                        total += comp_self;
                    }
                    else { // no compress
                        set_flag(r, 0b00);
                        f.write_file(f.trace[index], BLOCK_SIZE);
                        total += BLOCK_SIZE;
                    }
                }
#ifdef PRINT_HASH
                cout << index << ' ' << h << '\n';
#endif

                ann.insert(h, index);

                while (!dedup_lazy_recipe.empty() && dedup_lazy_recipe.begin()->first < index) {
                    RECIPE rr;
                    set_ref(rr, dedup_lazy_recipe.begin()->second);
                    set_flag(rr, 0b10);
                    f.recipe_insert(rr);
                    dedup_lazy_recipe.pop_front();
                }
                f.recipe_insert(r);
            }
        }
    }
    // LAST REQUEST
    {
        vector<pair<MYHASH, int>> myhash = network.request();
        for (int j = 0; j < myhash.size(); ++j) {
            RECIPE r;

            MYHASH& h = myhash[j].first;
            int index = myhash[j].second;

            int comp_self = LZ4_compress_default(f.trace[index], compressed, BLOCK_SIZE, 2 * BLOCK_SIZE);
            int dcomp_ann = INF, dcomp_ann_ref;
            dcomp_ann_ref = ann.request(h);

            if (dcomp_ann_ref != -1) {
                dcomp_ann = xdelta3_compress(f.trace[index], BLOCK_SIZE, f.trace[dcomp_ann_ref], BLOCK_SIZE, delta_compressed, 1);
            }

            set_offset(r, total);

            if (min(comp_self, BLOCK_SIZE) > dcomp_ann) { // delta compress
                set_size(r, (unsigned long)(dcomp_ann - 1));
                set_ref(r, dcomp_ann_ref);
                set_flag(r, 0b11);
                f.write_file(delta_compressed, dcomp_ann);
                dist_histogram[ann.getLastDist()]++;
                comp_size_for_dist[ann.getLastDist()] += dcomp_ann;
                total += dcomp_ann;
            }
            else {
                if (comp_self < BLOCK_SIZE) { // self compress
                    set_size(r, (unsigned long)(comp_self - 1));
                    set_flag(r, 0b01);
                    f.write_file(compressed, comp_self);
                    total += comp_self;
                }
                else { // no compress
                    set_flag(r, 0b00);
                    f.write_file(f.trace[index], BLOCK_SIZE);
                    total += BLOCK_SIZE;
                }
            }
#ifdef PRINT_HASH
                cout << index << ' ' << h << '\n';
#endif

            ann.insert(h, index);

            while (!dedup_lazy_recipe.empty() && dedup_lazy_recipe.begin()->first < index) {
                RECIPE rr;
                set_ref(rr, dedup_lazy_recipe.begin()->second);
                set_flag(rr, 0b10);
                f.recipe_insert(rr);
                dedup_lazy_recipe.pop_front();
            }
            f.recipe_insert(r);
        }
    }
    f.recipe_write();
    cout << "Total time: " << f.time_check_end() << "us\n";

    printf("ANN %s with model %s\n", argv[1], argv[2]);
    printf("Final size: %llu (%.2lf%%)\n", total, (double)total * 100 / f.N / BLOCK_SIZE);

    if (argc >= 5) {
        // Save the hash distance histogram
        std::ofstream ofile(argv[4]);
        for (int i = 0; i < HASH_SIZE; ++i) {
            ofile << i << "\t" << dist_histogram[i] << "\t" << comp_size_for_dist[i] << "\n";
        }
        ofile.close();
        printf("Saved the hash distance histogram in %s\n", argv[4]);
    }
}
