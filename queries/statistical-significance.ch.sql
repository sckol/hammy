insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/statistical-significance.csv',
  'CSVWithNames'
) 
with src as (
  select *, if(target_position >= 0, 1, -1) sgn 
  from s3('s3:///hammy/lagrangian/1/1/raw/lagrangian_*.gzip.parquet', Parquet)),
cnts as (
  select abs(target_position) target_position,
    1000 checkpoint, 
    sgn * checkpoint10_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    1001 checkpoint, 
    sgn * checkpoint11_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    1005 checkpoint, 
    sgn * checkpoint12_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    1010 checkpoint, 
    sgn * checkpoint13_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    1050 checkpoint, 
    sgn * checkpoint14_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    2000 checkpoint, 
    sgn * checkpoint20_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    2001 checkpoint, 
    sgn * checkpoint21_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    2005 checkpoint, 
    sgn * checkpoint22_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    2010 checkpoint, 
    sgn * checkpoint23_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    2050 checkpoint, 
    sgn * checkpoint24_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    3000 checkpoint, 
    sgn * checkpoint30_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    3001 checkpoint, 
    sgn * checkpoint31_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    3005 checkpoint, 
    sgn * checkpoint32_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    3010 checkpoint, 
    sgn * checkpoint33_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    3050 checkpoint, 
    sgn * checkpoint34_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    4000 checkpoint, 
    sgn * checkpoint40_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    4001 checkpoint, 
    sgn * checkpoint41_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    4005 checkpoint, 
    sgn * checkpoint42_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    4010 checkpoint, 
    sgn * checkpoint43_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    4050 checkpoint, 
    sgn * checkpoint44_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    5000 checkpoint, 
    sgn * checkpoint50_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    5001 checkpoint, 
    sgn * checkpoint51_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    5005 checkpoint, 
    sgn * checkpoint52_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    5010 checkpoint, 
    sgn * checkpoint53_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position union all 
  select abs(target_position) target_position,
    5050 checkpoint, 
    sgn * checkpoint54_position position, count(*) cnt,
    position = round(target_position / 6000 * checkpoint) as is_correct,
    row_number() over (partition by target_position, checkpoint order by cnt desc) rn
  from src group by target_position, position),
jin as (
  select target_position, checkpoint, 
    x.position position1, x.cnt cnt1, x.is_correct is_correct1, 
    (cnt1 - cnt2) / sqrt(cnt1 + cnt2) > 1.64 as is_significant,
    y.position position2, y.cnt cnt2, y.is_correct is_correct2 
  from cnts x 
  left join (select * from cnts where rn = 2) y using (target_position, checkpoint)
  where rn = 1 settings join_use_nulls=1)
select * from jin