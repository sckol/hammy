insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-accumulated.csv',
  'CSVWithNames'
) 
with uni as (select sorted,
  cast(sum(change1) as Float) / 100 as exactness_ratio1,
  cast(sum(change2) as Float) / 100 as exactness_ratio2,
  cast(sum(change3) as Float) / 100 as exactness_ratio3,
  cast(sum(change4) as Float) / 100 as exactness_ratio4,
  cast(sum(change5) as Float) / 100 as exactness_ratio5
  from s3('s3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/*/single-ensemble.*csv', CSVWithNames)
  group by sorted),
cum as (select sorted,
  sum(exactness_ratio1) over (order by sorted) exactness_ratio1,
  sum(exactness_ratio2) over (order by sorted) exactness_ratio2,
  sum(exactness_ratio3) over (order by sorted) exactness_ratio3,
  sum(exactness_ratio4) over (order by sorted) exactness_ratio4,
  sum(exactness_ratio5) over (order by sorted) exactness_ratio5
   from uni),
discretize as (select cast(sorted / 100000 as Int32) * 100000 steps,
  min(exactness_ratio1) exactness_ratio1,
  min(exactness_ratio2) exactness_ratio2,
  min(exactness_ratio3) exactness_ratio3,
  min(exactness_ratio4) exactness_ratio4,
  min(exactness_ratio5) exactness_ratio5
  from cum group by steps),
x_axis as (select arrayJoin(range(0, (select max(steps) from discretize) + 1, 10000)) as steps),
fill_zeros as (select steps,
  last_value(exactness_ratio1) ignore nulls over (order by steps) exactness_ratio1,
  last_value(exactness_ratio2) ignore nulls over (order by steps) exactness_ratio2,
  last_value(exactness_ratio3) ignore nulls over (order by steps) exactness_ratio3,
  last_value(exactness_ratio4) ignore nulls over (order by steps) exactness_ratio4,
  last_value(exactness_ratio5) ignore nulls over (order by steps) exactness_ratio5
  from x_axis left join discretize using(steps) settings join_use_nulls=1)
select * from fill_zeros