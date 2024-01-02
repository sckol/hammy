#include <random>
#include <cassert>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <time.h>
#include <unistd.h>
#include <array>
#include <csignal>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <stdio.h>

using std::cout;
using std::endl;
using std::string;
using std::stringstream;
namespace fs = std::filesystem;

const string TAG = "lagrangian";
const unsigned EXPERIMENT_NUMBER = 1;
const unsigned HYPOTHESIS_NUMBER = 1;
const unsigned IMPLEMENTATION_NUMBER = 1;

const unsigned EPOCH_LENGTH { 6'000 };
const unsigned EPOCHS_IN_CHUNK { 10'000'000 };

const unsigned DEFAULT_EPOCHS_TARGET_COUNT { 10'000'000 };
const unsigned DEFAULT_THREADS_COUNT { 4 };
const string DEFAULT_RESULTS_DIR { "out" };

const char SEPARATOR { ',' };

using position_t = int;
const position_t TARGET_POSITIONS[] {0, 6, 12, 60, 120};
const unsigned CHECKPOINTS[] = {1000, 1001, 1005, 1010, 1050, 1100, 2000, 2001, 2005, 2010, 2050, 2100, 3000, 3001, 3005, 3010, 3050, 3100, 4000, 4001, 4005, 4010, 4050, 4100, 5000, 5001, 5005, 5010, 5050, 5100};
const unsigned MAIN_CHECKPOINTS_COUNT = 5;
const unsigned ADDITIONAL_CHECKPOINTS_COUNT = 5;
const unsigned CHECKPOINTS_LEN {sizeof(CHECKPOINTS) / sizeof(*CHECKPOINTS)};
static_assert(CHECKPOINTS_LEN == MAIN_CHECKPOINTS_COUNT * (ADDITIONAL_CHECKPOINTS_COUNT + 1), "CHECKPOINT_LEN does not math MAIN_CHECKPOINTS_COUNT and ADDITIONAL_CHECKPOINTS_COUNT");

using epoch_t = std::array<position_t, CHECKPOINTS_LEN + 1>;
using epochs_chunk_t = std::array<epoch_t, EPOCHS_IN_CHUNK>;

const string STATS_FILE_NAME = "stats.csv";

volatile sig_atomic_t WAS_INTERRUPT;

class RandomBitSource {
	static const uint64_t GENERATOR_MAX_VALUE { std::mt19937_64::max() };
	std::mt19937_64 m_gen;
	uint64_t m_bits;
	uint64_t m_bitMask { GENERATOR_MAX_VALUE };
	static_assert((GENERATOR_MAX_VALUE ^ (GENERATOR_MAX_VALUE - 1)) == 1,
			"No support for GENERATOR_MAX_VALUE != 2^(n-1)");
public:
	void seed(unsigned threadNumber) {
		m_gen = std::mt19937_64(std::random_device { }());
		m_bits = m_gen();
	}

	auto getBit() {
		if (!m_bitMask)
			m_bits = m_gen(), m_bitMask = GENERATOR_MAX_VALUE;
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
	string getFilename(unsigned threadNumber, unsigned chunkNumber, string extension) const {
		stringstream ss;
		ss << m_tag << "_" << m_experimentNumber << "_" << m_hypothesisNumber << "-" << m_implementationNumber
				<< "_" + m_programTimestamp << "_" << threadNumber << "_" << chunkNumber << extension;
		return ss.str();
	}
	VersionTag() = delete;
	VersionTag(string tag, unsigned experimentNumber, unsigned hypothesisNumber, unsigned implementationNumber,
			string programTimestamp) :
			m_tag(tag), m_experimentNumber(experimentNumber), m_hypothesisNumber(hypothesisNumber), m_implementationNumber(
					implementationNumber), m_programTimestamp(programTimestamp) {
	}
};

struct StatisticsRecord {
	long long unsigned m_numIterations { };
	unsigned m_numEpochs { };
	unsigned m_numChunks { };
	double m_cpuTime { };

	StatisticsRecord operator+(const StatisticsRecord &first) const {
		StatisticsRecord res = first;
		res.m_numIterations += m_numIterations;
		res.m_numEpochs += m_numEpochs;
		res.m_numChunks += m_numChunks;
		res.m_cpuTime += m_cpuTime;
		return res;
	}
};

class StatisticsWriter {
	const unsigned m_threadsCount;
	const TimeMeasurer m_timeMeasurer;
	const VersionTag m_versionTag;
	const fs::path m_resultsDir;
	StatisticsRecord *m_threadRecords;
	
	StatisticsRecord aggregate() const {
		StatisticsRecord res;
		for (unsigned i = 0; i < m_threadsCount; ++i) {
			res = res + m_threadRecords[i];
		}
		return res;
	}
	
public:
	StatisticsWriter() = delete;
	StatisticsWriter(unsigned threadsCount, TimeMeasurer timeMeasurer, VersionTag versionTag, fs::path resultsDir) :
			m_threadsCount(threadsCount), m_timeMeasurer(timeMeasurer), m_versionTag(versionTag), m_resultsDir(resultsDir), m_threadRecords(
					static_cast<StatisticsRecord*>(mmap(NULL, threadsCount * sizeof(StatisticsRecord),
					PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0))) {
	}

	void report(unsigned threadNumber, StatisticsRecord record) const {
		if (threadNumber > m_threadsCount) {
			perror("Thread number is greated then threads count");
			abort();
		}
		m_threadRecords[threadNumber - 1] = record;
	}

	void write() const {
		std::fstream statsFile;
		statsFile.open(m_resultsDir / STATS_FILE_NAME, std::ios_base::app);
		if (!fs::exists(m_resultsDir / STATS_FILE_NAME)) {
			statsFile << "tag" << SEPARATOR << "experiment_number" << SEPARATOR;
			statsFile << "hypothesis_number" << SEPARATOR << "implementation_number" << SEPARATOR;
			statsFile << "start_datetime" << SEPARATOR;
			statsFile << "wall_time" << SEPARATOR << "cpu_time" << SEPARATOR;
			statsFile << "epochs" << SEPARATOR << "chunks" << SEPARATOR << "iterations" << SEPARATOR;
			statsFile << "avg_epoch_length" << SEPARATOR << "threads" << "\n";
		}
		const StatisticsRecord res = aggregate();
		statsFile << m_versionTag.m_tag << SEPARATOR << m_versionTag.m_experimentNumber << SEPARATOR;
		statsFile << m_versionTag.m_hypothesisNumber << SEPARATOR << m_versionTag.m_implementationNumber << SEPARATOR;
		statsFile << m_timeMeasurer.getProgramTimestamp() << SEPARATOR;
		statsFile << m_timeMeasurer.getTime(false) << SEPARATOR << res.m_cpuTime << SEPARATOR;
		statsFile << res.m_numEpochs << SEPARATOR << res.m_numChunks << SEPARATOR << res.m_numIterations << SEPARATOR;
		statsFile << EPOCH_LENGTH << SEPARATOR << m_threadsCount << endl;
		statsFile.close();
	}
};

class Simulator {
	epochs_chunk_t m_chunk;
	StatisticsRecord m_statRecord;
	const VersionTag m_versionTag;
	const unsigned m_epochsTargetCount;
	const fs::path m_resultsDir;
	RandomBitSource m_src;
	const StatisticsWriter m_statWriter;
	const TimeMeasurer m_timeMeasurer;

	inline void do_epoch(epoch_t &epoch) {
		bool successful_epoch = 0;		
		position_t position ;
		do {							
			position = 0;
			unsigned checkpoint_index = 0;
			unsigned checkpoint = CHECKPOINTS[0];		
			for (unsigned i = 1; i <= EPOCH_LENGTH; ++i) {
				const int b { m_src.getBit() };
				position += b - ((~b) & 1);
				if (i == checkpoint) {
					epoch[checkpoint_index + 1] = position;
					checkpoint_index++;
					checkpoint = CHECKPOINTS[checkpoint_index];
				}
				++(m_statRecord.m_numIterations);
			}			
			for (auto target_position : TARGET_POSITIONS) {
				if (position == target_position || position == -target_position) {
					successful_epoch = 1;
					break;
				}
			}
		} while (!successful_epoch);
		epoch[0] = position;
	}

	void writeChunk(unsigned threadNumber) const {
		std::ofstream outFile;
		outFile.open(m_resultsDir / m_versionTag.getFilename(threadNumber, m_statRecord.m_numChunks, ".csv"));
		outFile << "epoch" << SEPARATOR << "target_position";
		for (unsigned i = 1; i <= MAIN_CHECKPOINTS_COUNT; i++) {
			for(unsigned j = 0; j <= ADDITIONAL_CHECKPOINTS_COUNT; j++) {
				outFile << SEPARATOR << "checkpoint" << i << j << "_position";
			}
		}
		outFile << "\n";
		for (unsigned i = 0; (i < EPOCHS_IN_CHUNK) && (i <= (m_statRecord.m_numEpochs - 1) % EPOCHS_IN_CHUNK); ++i) {
			const unsigned epoch = 1 + i + (m_statRecord.m_numChunks - 1) * EPOCHS_IN_CHUNK;
			outFile << epoch;
			for (unsigned t = 0; t < m_chunk[i].size(); ++t) {
				outFile << SEPARATOR << m_chunk[i][t];
			}
			outFile << "\n";
		}
		outFile.close();
	}
public:
	Simulator() = delete;
	Simulator(VersionTag versionTag, unsigned epochTargetCount, fs::path resultsDir, RandomBitSource src,
			StatisticsWriter statWriter, TimeMeasurer timeMeasurer) :
			m_versionTag(versionTag), m_epochsTargetCount(epochTargetCount), m_resultsDir(resultsDir), m_src(src), m_statWriter(
					statWriter), m_timeMeasurer(timeMeasurer) {
		if (!fs::is_directory(resultsDir) || !fs::is_directory(resultsDir)) {
			fs::create_directory(resultsDir);
		}

	}

	void run(unsigned threadNumber) {		
		m_src.seed(threadNumber);
		while (true) {
			++(m_statRecord.m_numChunks);
			for (auto &epoch : m_chunk) {
				if (WAS_INTERRUPT || (m_statRecord.m_numEpochs >= m_epochsTargetCount)) {
					break;
				}
				++(m_statRecord.m_numEpochs);
				do_epoch(epoch);				
				cout << m_statRecord.m_numEpochs << " " << m_epochsTargetCount << endl;				
			}
			cout << "Writing chunk from thread " << threadNumber << endl;
			writeChunk(threadNumber);
			if (WAS_INTERRUPT || (m_statRecord.m_numEpochs >= m_epochsTargetCount)) {
				break;
			}
		}
		m_statRecord.m_cpuTime = m_timeMeasurer.getTime(true);
		m_statWriter.report(threadNumber, m_statRecord);
	}

	auto getStatisticsRecord() const {
		return m_statRecord;
	}
};

void make_interrupt(int s) {
	cout << "Caught signal " << s << endl;
	WAS_INTERRUPT = 1;
}

int main(int argc, char **argv) {
	const unsigned epochsTargetCount { argc >= 2 ? std::stoi(argv[1]) : DEFAULT_EPOCHS_TARGET_COUNT };
	const unsigned threadsCount { argc >= 3 ? std::stoi(argv[2]) : DEFAULT_THREADS_COUNT };
	const fs::path resultsDir = { argc >= 4 ? argv[3] : DEFAULT_RESULTS_DIR };
	const TimeMeasurer timeMeasurer;
	const VersionTag versionTag { TAG, EXPERIMENT_NUMBER, HYPOTHESIS_NUMBER, IMPLEMENTATION_NUMBER,
				timeMeasurer.getProgramTimestamp() };
	const StatisticsWriter statWriter { threadsCount, timeMeasurer, versionTag, resultsDir };
	RandomBitSource src;	
	Simulator sim { versionTag, epochsTargetCount, resultsDir, src, statWriter, timeMeasurer };
	std::signal(SIGINT, make_interrupt);
	std::vector<pid_t> pids(threadsCount);
    for (unsigned threadNumber = 1; threadNumber <= pids.size(); ++threadNumber) {
		if ((pids[threadNumber - 1] = fork()) < 0) {
			perror("Cannot fork");
			abort();
		} else if (pids[threadNumber - 1] == 0) {
			sim.run(threadNumber);
			exit(0);
		}
	}
	int status;
	while (wait(&status) > 0);
	statWriter.write();
	return 0;
}
