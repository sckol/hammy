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
#include <arrow/io/api.h>
#include "parquet/stream_writer.h"
#include <cmath>
#include <limits>

using std::cout;
using std::endl;
using std::string;
using std::stringstream;
namespace fs = std::filesystem;

const string TAG = "lagrangian";
const unsigned EXPERIMENT_NUMBER = 2;
const unsigned HYPOTHESIS_NUMBER = 1;
const unsigned IMPLEMENTATION_NUMBER = 2;

const unsigned EPOCH_LENGTH{6'000};
const unsigned EPOCHS_IN_CHUNK{10'000'000};

const unsigned DEFAULT_EPOCHS_TARGET_COUNT{10'000'000};
const unsigned DEFAULT_THREADS_COUNT{4};
const string DEFAULT_RESULTS_DIR{"out"};

const char SEPARATOR{','};
const uint32_t MAX_UINT32{std::numeric_limits<uint32_t>::max()};

using position_t = int;
const position_t TARGET_POSITIONS[]{0, 24};
const unsigned short CHECKPOINT_LENGTH{100};
static_assert(EPOCH_LENGTH % CHECKPOINT_LENGTH == 0, "EPOCH_LENGTH must be divided by CHECKPOINT_LEN");

uint32_t POTENTIAL_THRESHOLDS[2 * EPOCH_LENGTH + 1];

using epoch_t = std::array<position_t, EPOCH_LENGTH / CHECKPOINT_LENGTH>;
using epochs_chunk_t = std::array<epoch_t, EPOCHS_IN_CHUNK>;

const string STATS_FILE_NAME = "stats.csv";

volatile sig_atomic_t WAS_INTERRUPT;

class RandomBitSource
{
	static const uint64_t GENERATOR_MAX_VALUE{std::mt19937_64::max()};
	std::mt19937_64 m_gen;
	uint64_t m_bits;
	uint64_t m_bitMask{GENERATOR_MAX_VALUE};
	static_assert((((GENERATOR_MAX_VALUE >> 1) + 1) % 8 == 0),
				  "No support for GENERATOR_MAX_VALUE not divided by 16");

public:
	void seed(unsigned threadNumber)
	{
		m_gen = std::mt19937_64(std::random_device{}());
		m_bits = m_gen();
	}

	auto get4Bits()
	{
		if (!(m_bitMask & ~1))
			m_bits = m_gen(), m_bitMask = GENERATOR_MAX_VALUE;
		int ret = m_bits & 15;
		m_bitMask >>= 4;
		m_bits >>= 4;
		return ret;
	}
	auto get32Bits()
	{
		if (!(m_bitMask & ~1))
			m_bits = m_gen(), m_bitMask = GENERATOR_MAX_VALUE;
		uint32_t ret = m_bits & MAX_UINT32;
		m_bitMask >>= 32;
		m_bits >>= 32;
		return ret;
	}
};

class TimeMeasurer
{
	const timespec m_beginRealTime;
	const timespec m_beginCpuTime;
	const string m_programTimestamp;

	auto buildTimespec(clockid_t clock_type)
	{
		timespec t;
		clock_gettime(clock_type, &t);
		return t;
	}

	auto buildProgramTimestamp()
	{
		char res[15];
		timespec beginUnixTime;
		clock_gettime(CLOCK_REALTIME, &beginUnixTime);
		auto parsedTime = gmtime(&beginUnixTime.tv_sec);
		strftime(res, sizeof(res), "%Y%m%d%H%M%S", parsedTime);
		return std::string(res);
	}

public:
	TimeMeasurer() : m_beginRealTime(buildTimespec(CLOCK_MONOTONIC)),
					 m_beginCpuTime(buildTimespec(CLOCK_PROCESS_CPUTIME_ID)),
					 m_programTimestamp(
						 buildProgramTimestamp())
	{
	}

	auto getTime(bool cpuTime) const
	{
		const timespec &beginTime{cpuTime ? m_beginCpuTime : m_beginRealTime};
		timespec endTime;
		clock_gettime(cpuTime ? CLOCK_PROCESS_CPUTIME_ID : CLOCK_MONOTONIC, &endTime);
		long seconds = endTime.tv_sec - beginTime.tv_sec;
		long nanoseconds = endTime.tv_nsec - beginTime.tv_nsec;
		return seconds + nanoseconds * 1e-9;
	}

	auto getProgramTimestamp() const
	{
		return m_programTimestamp;
	}
};

struct VersionTag
{
	const string m_tag;
	const unsigned m_experimentNumber, m_hypothesisNumber, m_implementationNumber;
	const string m_programTimestamp;
	string getFilename(unsigned threadNumber, unsigned chunkNumber, string extension) const
	{
		stringstream ss;
		ss << m_tag << "_" << m_experimentNumber << "_" << m_hypothesisNumber << "-" << m_implementationNumber
		   << "_" + m_programTimestamp << "_" << threadNumber << "_" << chunkNumber << extension;
		return ss.str();
	}
	VersionTag() = delete;
	VersionTag(string tag, unsigned experimentNumber, unsigned hypothesisNumber, unsigned implementationNumber,
			   string programTimestamp) : m_tag(tag), m_experimentNumber(experimentNumber), m_hypothesisNumber(hypothesisNumber), m_implementationNumber(implementationNumber), m_programTimestamp(programTimestamp)
	{
	}
};

struct StatisticsRecord
{
	long long unsigned m_numIterations{};
	unsigned m_numEpochs{};
	unsigned m_numChunks{};
	double m_cpuTime{};

	StatisticsRecord operator+(const StatisticsRecord &first) const
	{
		StatisticsRecord res = first;
		res.m_numIterations += m_numIterations;
		res.m_numEpochs += m_numEpochs;
		res.m_numChunks += m_numChunks;
		res.m_cpuTime += m_cpuTime;
		return res;
	}
};

class StatisticsWriter
{
	const unsigned m_threadsCount;
	const TimeMeasurer m_timeMeasurer;
	const VersionTag m_versionTag;
	const fs::path m_resultsDir;
	StatisticsRecord *m_threadRecords;

	StatisticsRecord aggregate() const
	{
		StatisticsRecord res;
		for (unsigned i = 0; i < m_threadsCount; ++i)
		{
			res = res + m_threadRecords[i];
		}
		return res;
	}

public:
	StatisticsWriter() = delete;
	StatisticsWriter(unsigned threadsCount, TimeMeasurer timeMeasurer,
					 VersionTag versionTag, fs::path resultsDir) : m_threadsCount(threadsCount),
																   m_timeMeasurer(timeMeasurer),
																   m_versionTag(versionTag),
																   m_resultsDir(resultsDir),
																   m_threadRecords(
																	   static_cast<StatisticsRecord *>(mmap(NULL, threadsCount * sizeof(StatisticsRecord),
																											PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0)))
	{
	}

	void report(unsigned threadNumber, StatisticsRecord record) const
	{
		if (threadNumber > m_threadsCount)
		{
			perror("Thread number is greated then threads count");
			abort();
		}
		m_threadRecords[threadNumber - 1] = record;
	}

	void write() const
	{
		std::fstream statsFile;
		const bool fileExists = fs::exists(m_resultsDir / STATS_FILE_NAME);
		statsFile.open(m_resultsDir / STATS_FILE_NAME, std::ios_base::app);
		if (!fileExists)
		{
			statsFile << "tag" << SEPARATOR << "experiment_number" << SEPARATOR;
			statsFile << "hypothesis_number" << SEPARATOR << "implementation_number" << SEPARATOR;
			statsFile << "start_datetime" << SEPARATOR;
			statsFile << "wall_time" << SEPARATOR << "cpu_time" << SEPARATOR;
			statsFile << "epochs" << SEPARATOR << "chunks" << SEPARATOR << "iterations" << SEPARATOR;
			statsFile << "avg_epoch_length" << SEPARATOR << "threads"
					  << "\n";
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

class Simulator
{
	epochs_chunk_t *m_chunk;
	StatisticsRecord m_statRecord;
	const VersionTag m_versionTag;
	const unsigned m_epochsTargetCount;
	const fs::path m_resultsDir;
	RandomBitSource m_src_kinetic;
	RandomBitSource m_src_potential;
	const StatisticsWriter m_statWriter;
	const TimeMeasurer m_timeMeasurer;

	inline void do_epoch(epoch_t &epoch)
	{
		bool successful_epoch = 0;
		position_t position;
		do
		{
			position = 0;			
			bool potential_kill = 0;
			for (unsigned i = 1; i <= EPOCH_LENGTH; ++i)
			{
				++(m_statRecord.m_numIterations);
				auto death = m_src_potential.get32Bits();
				// cout << death << " " << POTENTIAL_THRESHOLDS[EPOCH_LENGTH + position] << endl; 
				if (death > POTENTIAL_THRESHOLDS[EPOCH_LENGTH + position]) {
					potential_kill = 1;										
					break;
				}
				const int b{m_src_kinetic.get4Bits()};								
				if (b & 14) {
					position += (b & 1) - ((~b) & 1);
				}
				if (i % CHECKPOINT_LENGTH == 0 && i != EPOCH_LENGTH && i != 0)
				{
					epoch[i / CHECKPOINT_LENGTH] = position;										
				}				
			}
			if (potential_kill) {
				continue;
			}
			for (auto target_position : TARGET_POSITIONS)
			{
				if (position == target_position)
				{
					successful_epoch = 1;
					break;
				}
			}
		} while (!successful_epoch);
		epoch[0] = position;
	}

	void writeChunkCSV(unsigned threadNumber) const
	{
		std::ofstream outFile;
		outFile.open(m_resultsDir / m_versionTag.getFilename(threadNumber, m_statRecord.m_numChunks, ".csv"));
		outFile << "epoch" << SEPARATOR << "target_position" << SEPARATOR << "checkpoint" << SEPARATOR << "position";		
		outFile << "\n";
		for (unsigned i = 0; (i < EPOCHS_IN_CHUNK) && (i <= (m_statRecord.m_numEpochs - 1) % EPOCHS_IN_CHUNK); ++i)
		{
			const unsigned epoch = 1 + i + (m_statRecord.m_numChunks - 1) * EPOCHS_IN_CHUNK;			
			for (unsigned t = 1; t < (*m_chunk)[i].size(); ++t)
			{
				outFile << epoch;
				outFile << SEPARATOR << (*m_chunk)[i][0];
				outFile << SEPARATOR << t * CHECKPOINT_LENGTH;
				outFile << SEPARATOR << (*m_chunk)[i][t];
				outFile << "\n";
			}			
		}
		outFile.close();
	}

	void writeChunkParquet(unsigned threadNumber, const std::string &compressionMethod) const
	{
		std::map<std::string, parquet::Compression::type> compressionMap = {
			{"snappy", parquet::Compression::SNAPPY},
			{"gzip", parquet::Compression::GZIP},
			{"lz4", parquet::Compression::LZ4},
			{"zstd", parquet::Compression::ZSTD},
			{"uncompressed", parquet::Compression::UNCOMPRESSED}};

		std::shared_ptr<arrow::io::FileOutputStream> outFile;
		auto fileName = m_versionTag.getFilename(threadNumber, m_statRecord.m_numChunks, "." + compressionMethod + ".parquet");
		PARQUET_ASSIGN_OR_THROW(
			outFile, arrow::io::FileOutputStream::Open(m_resultsDir / fileName));

		parquet::schema::NodeVector fields;
		fields.push_back(parquet::schema::PrimitiveNode::Make("epoch", parquet::Repetition::REQUIRED, parquet::Type::INT32, parquet::ConvertedType::UINT_32));
		fields.push_back(parquet::schema::PrimitiveNode::Make("target_position", parquet::Repetition::REQUIRED, parquet::Type::INT32, parquet::ConvertedType::INT_16));
		fields.push_back(parquet::schema::PrimitiveNode::Make("checkpoint", parquet::Repetition::REQUIRED, parquet::Type::INT32, parquet::ConvertedType::UINT_16));
		fields.push_back(parquet::schema::PrimitiveNode::Make("position", parquet::Repetition::REQUIRED, parquet::Type::INT32, parquet::ConvertedType::INT_32));		
		auto schema = std::static_pointer_cast<parquet::schema::GroupNode>(parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));
		auto props = parquet::WriterProperties::Builder().compression(compressionMap[compressionMethod])->build();
		auto file_writer = parquet::ParquetFileWriter::Open(outFile, schema, props);
		parquet::StreamWriter os{std::move(file_writer)};

		for (unsigned i = 0; (i < EPOCHS_IN_CHUNK) && (i <= (m_statRecord.m_numEpochs - 1) % EPOCHS_IN_CHUNK); ++i)
		{
			const unsigned epoch = 1 + i + (m_statRecord.m_numChunks - 1) * EPOCHS_IN_CHUNK;			
			for (unsigned t = 1; t < (*m_chunk)[i].size(); ++t)
			{
				os << epoch;
				os << (short)((*m_chunk)[i][0]);
				os << (unsigned short) (t * CHECKPOINT_LENGTH);
				os << (*m_chunk)[i][t];
				os << parquet::EndRow;
			}					
		}
	}

public:
	Simulator() = delete;
	Simulator(VersionTag versionTag, unsigned epochTargetCount, fs::path resultsDir, RandomBitSource src_kinetic, RandomBitSource src_potential,
			  StatisticsWriter statWriter, TimeMeasurer timeMeasurer) : m_versionTag(versionTag), m_epochsTargetCount(epochTargetCount), m_resultsDir(resultsDir), m_src_kinetic(src_kinetic), m_src_potential(src_potential), m_statWriter(statWriter), m_timeMeasurer(timeMeasurer)
	{
		if (!fs::is_directory(resultsDir) || !fs::is_directory(resultsDir))
		{
			fs::create_directory(resultsDir);
		}
	}

	void run(unsigned threadNumber)
	{
		m_chunk = static_cast<epochs_chunk_t *>(calloc(1, sizeof(epochs_chunk_t)));
		if (!m_chunk)
		{
			throw std::bad_alloc();
		}
		m_src_kinetic.seed(threadNumber);
		m_src_potential.seed(threadNumber);
		while (true)
		{
			++(m_statRecord.m_numChunks);
			for (auto &epoch : *m_chunk)
			{
				if (WAS_INTERRUPT || (m_statRecord.m_numEpochs >= m_epochsTargetCount))
				{
					break;
				}
				++(m_statRecord.m_numEpochs);
				do_epoch(epoch);
				if (m_statRecord.m_numEpochs % 50'000 == 1) {
					cout << m_statRecord.m_numEpochs << " " << m_epochsTargetCount << endl;
				}
			}
			cout << "Writing chunk from thread " << threadNumber << endl;
			//writeChunkCSV(threadNumber);			
			writeChunkParquet(threadNumber, "gzip");			
			if (WAS_INTERRUPT || (m_statRecord.m_numEpochs >= m_epochsTargetCount))
			{
				break;
			}
		}
		m_statRecord.m_cpuTime = m_timeMeasurer.getTime(true);
		m_statWriter.report(threadNumber, m_statRecord);
	}

	auto getStatisticsRecord() const
	{
		return m_statRecord;
	}
};

void make_interrupt(int s)
{
	cout << "Caught signal " << s << endl;
	WAS_INTERRUPT = 1;
}

void init_thresholds() {
	double GROUND_LEVEL = -0.16152380912;
	double MAX_VALUE = pow(2., -32);	
	for (int x = -EPOCH_LENGTH; x <= static_cast<int>(EPOCH_LENGTH); x++) {		
		unsigned i = x + EPOCH_LENGTH;
		double energy = -8. / 7. / 1000. * 4. / 3. * x + GROUND_LEVEL;		
		uint64_t threshold = exp(energy / 1000) / MAX_VALUE;
		POTENTIAL_THRESHOLDS[i] = (threshold <= MAX_UINT32) ? round(threshold) : MAX_UINT32;
	}
}

int main(int argc, char **argv)
{
	const unsigned epochsTargetCount{argc >= 2 ? std::stoi(argv[1]) : DEFAULT_EPOCHS_TARGET_COUNT};
	const unsigned threadsCount{argc >= 3 ? std::stoi(argv[2]) : DEFAULT_THREADS_COUNT};
	const fs::path resultsDir = {argc >= 4 ? argv[3] : DEFAULT_RESULTS_DIR};	
	const TimeMeasurer timeMeasurer;
	const VersionTag versionTag{TAG, EXPERIMENT_NUMBER, HYPOTHESIS_NUMBER, IMPLEMENTATION_NUMBER,
								timeMeasurer.getProgramTimestamp()};
	const StatisticsWriter statWriter{threadsCount, timeMeasurer, versionTag, resultsDir};
	RandomBitSource kinetic_src;
	RandomBitSource potential_src;
	Simulator sim{versionTag, epochsTargetCount, resultsDir, kinetic_src, potential_src, statWriter, timeMeasurer};
	init_thresholds();
	std::signal(SIGINT, make_interrupt);
	std::vector<pid_t> pids(threadsCount);
	for (unsigned threadNumber = 1; threadNumber <= pids.size(); ++threadNumber)
	{
		if ((pids[threadNumber - 1] = fork()) < 0)
		{
			perror("Cannot fork");
			abort();
		}
		else if (pids[threadNumber - 1] == 0)
		{
			sim.run(threadNumber);
			exit(0);
		}
	}
	int status;
	while (wait(&status) > 0)
		;
	statWriter.write();
	return 0;
}
