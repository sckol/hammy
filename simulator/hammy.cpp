#include <random>
#include <cassert>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <time.h>
#include <array>

using std::cout;
using std::endl;
using std::string;
using std::stringstream;
namespace fs = std::filesystem;

const string TAG = "walk";
const unsigned EXPERIMENT_NUMBER = 1;
const unsigned HYPOTHESIS_NUMBER = 1;
const unsigned IMPLEMENTATION_NUMBER = 1;

const unsigned EPOCH_LENGTH { 100'000 };
const unsigned EPOCHS_TARGET_COUNT { 10 };
const unsigned EPOCHS_IN_CHUNK { 1'000 };
const char SEPARATOR { ',' };
const unsigned THREAD_ID_LENGTH { 6 };

using position_t = int;
using epoch_t = std::array<position_t, EPOCH_LENGTH + 1>;
using epochs_chunk_t = std::array<epoch_t, EPOCHS_IN_CHUNK>;
const position_t TARGET_POSITION { 500 };

const string STATS_FILE_NAME = "stats.csv";

class RandomBitSource {
	static const uint64_t GENERATOR_MAX_VALUE { std::mt19937_64::max() };
	std::mt19937_64 gen;
	uint64_t m_bits;
	uint64_t m_bitMask { GENERATOR_MAX_VALUE };
	static_assert((GENERATOR_MAX_VALUE ^ (GENERATOR_MAX_VALUE - 1)) == 1,
			"No support for GENERATOR_MAX_VALUE != 2^(n-1)");
public:
	RandomBitSource() :
			gen(time(0)), m_bits(gen()) {
	}

	auto getBit() {
		if (!m_bitMask)
			m_bits = gen(), m_bitMask = GENERATOR_MAX_VALUE;
		bool ret = m_bits & 1;
		m_bitMask >>= 1;
		m_bits >>= 1;
		return ret;
	}
};

class TimeMeasurer {
	const timespec m_beginRealTime;
	const timespec m_beginCpuTime;
	const string m_programTimestamp;

	auto buildTimespec(clockid_t clock_type) {
		timespec t;
		clock_gettime(clock_type, &t);
		return t;
	}

	auto buildProgramTimestamp() {
		char res[15];
		timespec beginUnixTime;
		clock_gettime(CLOCK_REALTIME, &beginUnixTime);
		auto parsedTime { gmtime(&beginUnixTime.tv_sec) };
		snprintf(res, 15, "%4d%02d%02d%02d%02d%02d", parsedTime->tm_year + 1900, parsedTime->tm_mon + 1,
				parsedTime->tm_mday, parsedTime->tm_hour, parsedTime->tm_min, parsedTime->tm_sec);
		return string(res);
	}

public:
	TimeMeasurer() :
			m_beginRealTime(buildTimespec(CLOCK_MONOTONIC)), m_beginCpuTime(buildTimespec(CLOCK_PROCESS_CPUTIME_ID)), m_programTimestamp(
					buildProgramTimestamp()) {
	}

	auto getTime(bool cpuTime) const {
		const timespec &beginTime { cpuTime ? m_beginCpuTime : m_beginRealTime };
		timespec endTime;
		clock_gettime(cpuTime ? CLOCK_PROCESS_CPUTIME_ID : CLOCK_MONOTONIC, &endTime);
		long seconds = endTime.tv_sec - beginTime.tv_sec;
		long nanoseconds = endTime.tv_nsec - beginTime.tv_nsec;
		return seconds + nanoseconds * 1e-9;
	}

	auto getProgramTimestamp() const {
		return m_programTimestamp;
	}
};

struct VersionTag {
	const string m_tag;
	const unsigned m_experimentNumber, m_hypothesisNumber, m_implementationNumber;
	const string m_programTimestamp;
	string getFilename(string threadId, unsigned chunkNumber, string extension) const {
		stringstream ss;
		ss << m_tag << "_" << m_experimentNumber << "_" << m_hypothesisNumber << "-" << m_implementationNumber
				<< "_" + m_programTimestamp << "_" << threadId << "_" << chunkNumber << extension;
		return ss.str();
	}
	VersionTag(string tag, unsigned experimentNumber, unsigned hypothesisNumber, unsigned implementationNumber, string programTimestamp) :
			m_tag(tag), m_experimentNumber(experimentNumber), m_hypothesisNumber(hypothesisNumber), m_implementationNumber(
					implementationNumber), m_programTimestamp(programTimestamp) {
	}
};

class Simulator {
	const TimeMeasurer m_timeMeasurer;
	epochs_chunk_t m_chunk;
	RandomBitSource m_src;
	const VersionTag m_versionTag;
	const unsigned m_epochsTargetCount;
	long long unsigned m_numIterations { };
	unsigned m_numEpochs { };
	unsigned m_numChunks { };
	const fs::path m_resultsDir;

	inline void do_epoch(epoch_t &epoch) {
		do {
			position_t position = 0;
			epoch[0] = position;
			for (unsigned i = 1; i < sizeof(epoch); i++) {
				const int b { m_src.getBit() };
				position += b - ((~b) & 1);
				epoch[i] = position;
				m_numIterations++;
			}
		} while (epoch.back() != TARGET_POSITION);
	}

	void writeChunk(string threadId) const {
		std::ofstream outFile;
		outFile.open(m_resultsDir / m_versionTag.getFilename(generateThreadId(), m_numChunks, ".csv"));
		outFile << "epoch" << SEPARATOR << "time" << SEPARATOR << "position" << "\n";
		for (unsigned i = 0; (i < EPOCHS_IN_CHUNK) && (i < m_numEpochs % EPOCHS_IN_CHUNK); i++) {
			const unsigned epoch = 1 + i + (m_numChunks - 1) * EPOCHS_IN_CHUNK;
			for (unsigned t = 0; t < m_chunk[i].size(); t++) {
				outFile << epoch << SEPARATOR << t << SEPARATOR << m_chunk[i][t] << "\n";
			}
		}
		outFile.close();
	}

	static string generateThreadId() {
		static const char alphanum[] = "0123456789abcdefghijklmnopqrstuvwxyz";
		string res;
		res.reserve(THREAD_ID_LENGTH);
		for (unsigned i = 0; i < THREAD_ID_LENGTH; i++) {
			res += alphanum[rand() % (sizeof(alphanum) - 1)];
		}
		return res;
	}

public:
	Simulator(VersionTag versionTag, unsigned epochTargetCount, fs::path resultsDir) :
			m_versionTag(versionTag), m_epochsTargetCount(epochTargetCount), m_resultsDir(resultsDir) {
		if (!fs::is_directory(resultsDir) || !fs::is_directory(resultsDir)) {
			fs::create_directory(resultsDir); // create src folder
		}

	}

	void run() {
		string threadId = generateThreadId();
		while (true) {
			m_numChunks++;
			for (auto &epoch : m_chunk) {
				if (m_numEpochs >= m_epochsTargetCount) {
					break;
				}
				m_numEpochs++;
				do_epoch(epoch);
				cout << m_numEpochs << " " << m_epochsTargetCount << endl;
			}
			writeChunk(threadId);
			if (m_numEpochs >= m_epochsTargetCount)
				break;
		}
	}

	auto getNumChunks() const {
		return m_numChunks;
	}

	auto getNumEpochs() const {
		return m_numEpochs;
	}

	auto getNumIterations() const {
		return m_numIterations;
	}
};

int main(int argc, char **argv) {
	srand(time(NULL));
	const unsigned epochsTargetCount { argc >= 2 ? std::stoi(argv[1]) : EPOCHS_TARGET_COUNT };
	const fs::path resultsDir = { argc >= 3 ? argv[2] : "out" };
	const TimeMeasurer timeMeasurer;
	RandomBitSource src;
	const VersionTag versionTag { TAG, EXPERIMENT_NUMBER, HYPOTHESIS_NUMBER, IMPLEMENTATION_NUMBER, timeMeasurer.getProgramTimestamp() };
	Simulator sim { versionTag, epochsTargetCount, resultsDir };
	std::fstream statsFile;
	statsFile.open(resultsDir / STATS_FILE_NAME, std::ios_base::app);
	if (!fs::exists(resultsDir / STATS_FILE_NAME)) {
		statsFile << "tag" << SEPARATOR << "experiment_number" << SEPARATOR;
		statsFile << "hypothesis_number" << SEPARATOR << "implementation_number" << SEPARATOR;
		statsFile << "wall_time" << SEPARATOR << "cpu_time" << SEPARATOR;
		statsFile << "epochs" << SEPARATOR << "chunks" << SEPARATOR << "iterations" << SEPARATOR;
		statsFile << "avg_epoch_length" << SEPARATOR << "threads" << "\n";
	}
	sim.run();
	statsFile << versionTag.m_tag << SEPARATOR << versionTag.m_experimentNumber << SEPARATOR;
	statsFile << versionTag.m_hypothesisNumber << SEPARATOR << versionTag.m_implementationNumber << SEPARATOR;
	statsFile << timeMeasurer.getTime(false) << SEPARATOR << timeMeasurer.getTime(true) << SEPARATOR;
	statsFile << sim.getNumEpochs() << SEPARATOR << sim.getNumChunks() << SEPARATOR << sim.getNumIterations()
			<< SEPARATOR;
	statsFile << EPOCH_LENGTH << SEPARATOR << 1 << endl;
	statsFile.close();
	return 0;
}
