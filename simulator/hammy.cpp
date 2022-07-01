#include <cassert>
#include <iostream>
#include <stdexcept>
#include <time.h>
#include <array>

using std::cout;
using std::endl;

const unsigned EPOCH_LENGTH { 1000'00 };
const unsigned EPOCHS_TARGET_COUNT { 1 };
using position_t = int;
using epoch_t = std::array<position_t, EPOCH_LENGTH>;
using simulation_t = std::array<epoch_t, EPOCHS_TARGET_COUNT>;
const position_t TARGET_POSITION { 1000 };

class RandomBitSource {
	int64_t m_bits {rand()};
	int64_t m_bitMask {RAND_MAX};
	static_assert((RAND_MAX ^ (RAND_MAX - 1)) == 1,
			"No support for RAND_MAX != 2^(n-1)");
public:
	auto getBit() {
		if (!m_bitMask)
			m_bits = rand(), m_bitMask = RAND_MAX;
		bool ret = m_bits & 1;
		m_bitMask >>= 1;
		m_bits >>= 1;
		return ret;
	}
};

class TimeMeasurer {
	timespec m_beginRealTime;
	timespec m_beginCpuTime;
public:
	TimeMeasurer() {
		clock_gettime(CLOCK_MONOTONIC, &m_beginRealTime);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &m_beginCpuTime);
	}

	auto getTime(bool cpuTime) const {
		const timespec& beginTime {cpuTime ? m_beginCpuTime : m_beginRealTime};
		timespec endTime;
		clock_gettime(cpuTime ? CLOCK_PROCESS_CPUTIME_ID : CLOCK_MONOTONIC, &endTime);		
		long seconds = endTime.tv_sec - beginTime.tv_sec;
		long nanoseconds = endTime.tv_nsec - beginTime.tv_nsec;
		return seconds + nanoseconds * 1e-9;
	}
};

int main(int argc, char **argv) {
	TimeMeasurer const timeMeasurer;
	simulation_t simulation;
	RandomBitSource src;
	size_t numIterations { };
	for (auto &epoch : simulation) {
		do {
			position_t position = 0;
			for (auto &point : epoch) {
				numIterations++;
				const int b { src.getBit() };
				position += b - ((~b) & 1);
				point = position;
			}
		} while (epoch.back() != TARGET_POSITION);
		cout << "Finished iteration: " << numIterations << endl;
	}
	cout << "Wall time: " << timeMeasurer.getTime(false) << endl;
	cout << "CPU time: " << timeMeasurer.getTime(true) << endl;
	return 0;
}
