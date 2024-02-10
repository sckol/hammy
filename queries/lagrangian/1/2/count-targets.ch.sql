insert into function s3(
  's3:///hammy/lagrangian/1/2/proc/count-targets.csv',
  'CSVWithNames'
) 
select target_position, count(*) cnt
from file('../build/out/lagrangian_1_2-*.gzip.parquet', Parquet)
group by target_position