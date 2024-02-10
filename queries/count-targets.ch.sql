insert into function s3(
  's3:///hammy/lagrangian/1/3/proc/count-targets.csv',
  'CSVWithNames'
) 
select target_position, count(*) cnt
from file('../build/out/lagrangian_1_3-*.gzip.parquet', Parquet)
group by target_position