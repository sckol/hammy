insert into function s3(
  's3:///hammy/lagrangian/1/3/proc/count-targets.csv',
  'CSVWithNames'
) 
select target_position, count(*) cnt
from s3('s3:///hammy/lagrangian/1/1/raw/lagrangian_*.gzip.parquet', Parquet)
group by target_position