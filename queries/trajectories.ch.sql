insert into function s3(
  's3:///hammy/lagrangian/2/1/proc/trajectories.csv',
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
  select false is_theoretical, target_position, checkpoint, position,
  if(target_position = 0, -4, 0) * (checkpoint / 1000) + 2 / 3 * checkpoint / 1000 * checkpoint / 1000 theoretical_position, 
  position = toInt32(round(theoretical_position)) is_correct
  from cnts where rn = 1),
fll as (select * from traj union all
  select false, target_position, 0, 0, 0, true from traj group by target_position union all
  select false, target_position, 6000, target_position, target_position, true from traj group by target_position),
theor as (select * from fll union all select * replace(true as is_theoretical, theoretical_position as position) from fll)
select * from theor