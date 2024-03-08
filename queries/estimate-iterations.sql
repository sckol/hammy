insert into function s3(
  's3:///hammy/lagrangian/2/1/proc/estimate-iterations.csv',
  'CSVWithNames'
) 
with src as (
  select *
  from s3('s3:///hammy/lagrangian/2/1/raw/lagrangian_2_1-2*.gzip.parquet', Parquet)),
cnts as (
  select target_position, checkpoint, position, count(*) as cnt,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, checkpoint, position),
traj as (
  select target_position, checkpoint, position, cnt, rn
  from cnts where rn <= 2),
diff as (
  select target_position, checkpoint, cnt, rn, 
    cnt - lagInFrame(cnt) over (partition by target_position, checkpoint order by cnt) df from traj),
totals as (select target_position, count(*) / 59 total from src group by target_position),    
stat as (
  select target_position, max(total) total, sum(cnt) cnt, min(df) min_df, avg(df) avg_df, max(df) max_df from diff 
  join totals using(target_position)
  where rn = 1 group by target_position)
select * from stat