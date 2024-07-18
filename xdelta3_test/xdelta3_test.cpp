#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include "../xxhash.h"
#include "../xdelta3/xdelta3.h"
#define BLOCK_SIZE 4096

using namespace std;

char* generate_random_block() {
    char* block = new char[BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        block[i] = rand() % 256;
    }
    return block;
}

char* modify_block(const char* original, int start, int length) {
    char* modified = new char[BLOCK_SIZE];
    memcpy(modified, original, BLOCK_SIZE);
    for (int i = start; i < start + length && i < BLOCK_SIZE; ++i) {
        modified[i] = rand() % 256;
    }
    return modified;
}

char* scramble_block(const char* original, int chunk_size) {
    char* scrambled = new char[BLOCK_SIZE];
    memcpy(scrambled, original, BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; i += chunk_size) {
        int swap_index = rand() % (BLOCK_SIZE / chunk_size) * chunk_size;
        for (int j = 0; j < chunk_size && i + j < BLOCK_SIZE && swap_index + j < BLOCK_SIZE; ++j) {
            swap(scrambled[i + j], scrambled[swap_index + j]);
        }
    }
    return scrambled;
}

int do_xdelta3(const char* src, const char* tgt) {
    char* compressed = new char[2 * BLOCK_SIZE];
    int result = xdelta3_compress(const_cast<char*>(src), BLOCK_SIZE, const_cast<char*>(tgt), BLOCK_SIZE, compressed, 1);
    delete[] compressed;
    return result;
}

int main() {
    srand(time(NULL));

    char* reference_block = generate_random_block();
    char* half_modified_block = modify_block(reference_block, 0, BLOCK_SIZE / 2);
    char* slightly_modified_block = modify_block(reference_block, 0, 16);
    char* scrambled_block_16 = scramble_block(reference_block, 16);
    char* scrambled_block_8 = scramble_block(reference_block, 8);
    char* half_swapped_block = scramble_block(reference_block, BLOCK_SIZE / 2);

    cout << "Reference block vs. Half modified block: " << do_xdelta3(reference_block, half_modified_block) << " bytes" << endl;
    cout << "Reference block vs. Slightly modified block: " << do_xdelta3(reference_block, slightly_modified_block) << " bytes" << endl;
    cout << "Reference block vs. Scrambled block (16 bytes): " << do_xdelta3(reference_block, scrambled_block_16) << " bytes" << endl;
    cout << "Reference block vs. Scrambled block (8 bytes): " << do_xdelta3(reference_block, scrambled_block_8) << " bytes" << endl;
    cout << "Reference block vs. Half swapped block: " << do_xdelta3(reference_block, half_swapped_block) << " bytes" << endl;

    delete[] reference_block;
    delete[] half_modified_block;
    delete[] slightly_modified_block;
    delete[] scrambled_block_16;
    delete[] scrambled_block_8;
    delete[] half_swapped_block;

    return 0;
}
