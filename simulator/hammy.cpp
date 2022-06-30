#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <array>

const unsigned EPOCH_LENGTH { 1'000'000 };
const unsigned EPOCHS_TARGET_COUNT { 1'000 };
using position_t = int;
using epoch_t = std::array<position_t, EPOCH_LENGTH>;
using simulation_t = std::array<epoch_t, EPOCHS_TARGET_COUNT>;
const position_t TARGET_POSITION { 1'000 };

class RandomBitSource {
	int64_t bits = rand();
	int64_t bitMask = RAND_MAX;
	static_assert((RAND_MAX ^ (RAND_MAX - 1)) == 1, "No support for RAND_MAX != 2^(n-1)");
public:
	auto getBit() {
		if (!bitMask) bits = rand(), bitMask = RAND_MAX;
		bitMask >>= 1;
		bits >>= 1;
		return static_cast<bool>(bits & 1);
	}
};

int main(int argc, char **argv) {
	std::cout << "XXXs";
	simulation_t simulation {};
	RandomBitSource src;
	for (auto& epoch : simulation) {
		position_t position = 0;
		do {
			for (auto& point : epoch) {
				const int b {src.getBit()};
				position += b - ((~b) & 1);
				point = position;
			}			
		}  while (epoch.back() != TARGET_POSITION);
		std::cout << "Finished iteration" << std::endl;
	}
	return 0;
}
