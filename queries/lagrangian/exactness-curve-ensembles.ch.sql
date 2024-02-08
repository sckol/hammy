create view generate_exactess_curve as with  
srt as (select
  checkpoint10_position,
  checkpoint20_position,
  checkpoint30_position,
  checkpoint40_position,
  checkpoint50_position,
  row_number() over(order by rand()) sorted
  from s3('s3:///hammy/lagrangian/1/1/raw/*.gzip.parquet', Parquet)
  where target_position = 60),
counts as (select sorted,
  countIf(checkpoint10_position = 1 * 10) over(order by sorted) position1_exact, 
  countIf(checkpoint10_position = 1 * 10 - 2) over(order by sorted) position1_left,
  countIf(checkpoint10_position = 1 * 10 + 2) over(order by sorted) position1_right,
  countIf(checkpoint20_position = 2 * 10) over(order by sorted) position2_exact, 
  countIf(checkpoint20_position = 2 * 10 - 2) over(order by sorted) position2_left,
  countIf(checkpoint20_position = 2 * 10 + 2) over(order by sorted) position2_right,
  countIf(checkpoint30_position = 3 * 10) over(order by sorted) position3_exact, 
  countIf(checkpoint30_position = 3 * 10 - 2) over(order by sorted) position3_left,
  countIf(checkpoint30_position = 3 * 10 + 2) over(order by sorted) position3_right,
  countIf(checkpoint40_position = 4 * 10) over(order by sorted) position4_exact, 
  countIf(checkpoint40_position = 4 * 10 - 2) over(order by sorted) position4_left,
  countIf(checkpoint40_position = 4 * 10 + 2) over(order by sorted) position4_right,
  countIf(checkpoint50_position = 5 * 10) over(order by sorted) position5_exact, 
  countIf(checkpoint50_position = 5 * 10 - 2) over(order by sorted) position5_left,
  countIf(checkpoint50_position = 5 * 10 + 2) over(order by sorted) position5_right
  from srt),
is_top_exact as (select sorted,
  position1_exact > position1_left and position1_exact > position1_right correct1,
  position2_exact > position2_left and position2_exact > position2_right correct2,
  position3_exact > position3_left and position3_exact > position3_right correct3,
  position4_exact > position4_left and position4_exact > position4_right correct4,
  position5_exact > position5_left and position5_exact > position5_right correct5
  from counts),
fix_changes as (select sorted,
  correct1 - lagInFrame(correct1) over (order by sorted) as change1,
  correct2 - lagInFrame(correct2) over (order by sorted) as change2,
  correct3 - lagInFrame(correct3) over (order by sorted) as change3,
  correct4 - lagInFrame(correct4) over (order by sorted) as change4,
  correct5 - lagInFrame(correct5) over (order by sorted) as change5
  from is_top_exact),
leave_changes as (select * 
  from fix_changes
  where
    change1 != 0 or change2 != 0 or change3 != 0 or change4 != 0 or change5 != 0)
select * from leave_changes;


insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 1 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 2 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 3 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 4 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 5 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 6 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 7 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 8 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 9 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 10 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 11 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 12 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 13 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 14 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 15 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 16 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;
insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve-ensembles/5/single-ensemble.csv',
  'CSVWithNames'
) select 17 ensemble, * from generate_exactess_curve settings s3_create_new_file_on_insert=1;