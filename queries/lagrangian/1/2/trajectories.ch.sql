insert into function s3(
  's3:///hammy/lagrangian/1/2/proc/trajectories.csv',
  'CSVWithNames'
) 
with src as (
  select *
  from file('../build/out/lagrangian_*.gzip.parquet', Parquet)),
cnts as (
  select target_position,
    1000 checkpoint, 
    checkpoint10_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    1 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    1001 checkpoint, 
    checkpoint11_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    1005 checkpoint, 
    checkpoint12_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    1010 checkpoint, 
    checkpoint13_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    1050 checkpoint, 
    checkpoint14_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    2000 checkpoint, 
    checkpoint20_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    1 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    2001 checkpoint, 
    checkpoint21_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    2005 checkpoint, 
    checkpoint22_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    2010 checkpoint, 
    checkpoint23_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    2050 checkpoint, 
    checkpoint24_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    3000 checkpoint, 
    checkpoint30_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    1 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    3001 checkpoint, 
    checkpoint31_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    3005 checkpoint, 
    checkpoint32_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    3010 checkpoint, 
    checkpoint33_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    3050 checkpoint, 
    checkpoint34_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    4000 checkpoint, 
    checkpoint40_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    1 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    4001 checkpoint, 
    checkpoint41_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    4005 checkpoint, 
    checkpoint42_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    4010 checkpoint, 
    checkpoint43_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    4050 checkpoint, 
    checkpoint44_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    5000 checkpoint, 
    checkpoint50_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    1 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    5001 checkpoint, 
    checkpoint51_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    5005 checkpoint, 
    checkpoint52_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    5010 checkpoint, 
    checkpoint53_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select target_position,
    5050 checkpoint, 
    checkpoint54_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) is_correct,
    0 is_main_checkpoint,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position),
traj as (
  select target_position, checkpoint, is_main_checkpoint, position, is_correct
  from cnts where rn = 1),
fll as (select * from traj union all
  select target_position, 0, 1, 0, true from traj group by target_position union all
  select target_position, 6000, 1, target_position, true from traj group by target_position)
select * from fll