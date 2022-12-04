{% macro prepare() %}
{% set sql %}
CREATE VIEW IF NOT EXISTS hammy.parsed_files AS
  WITH splitFileName AS (SELECT *, splitByNonAlpha(_file) AS fileArr FROM file('*.csv', 'CSVWithNames',
  'epoch UInt32, time UInt32, position Int32') WHERE _file != 'stats.csv'),
  parseFileName AS (SELECT CAST(fileArr[1] AS Enum('walk' = 1)) AS tag, 
  CAST(fileArr[2] AS UInt8) AS experimentNumber, CAST(fileArr[3] AS UInt8) AS hypothesisNumber, 
  CAST(fileArr[4] AS UInt8) AS implementationNumber,   
  parseDateTimeBestEffort(fileArr[5], 'UTC') programTimestamp, 
  CAST(fileArr[6] AS UInt8) AS threadNumber,
  CAST(fileArr[7] AS UInt8) AS chunkNumber, epoch, time, position FROM splitFileName)
  SELECT * FROM parseFileName       
{% endset %}
{% do run_query(sql) %}
{% do log("hammy.parsed_files view created", info=True) %}
{% endmacro %}