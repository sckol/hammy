insert into function s3(
  's3:///hammy/lagrangian/1/1/proc/exactness-curve.csv',
  'CSVWithNames'
) with  
sorted as (select checkpoint10_position,
  row_number() over(order by rand()) sorted1,
  row_number() over(order by rand()) sorted2,
  row_number() over(order by rand()) sorted3,
  row_number() over(order by rand()) sorted4,
  row_number() over(order by rand()) sorted5,
  row_number() over(order by rand()) sorted6,
  row_number() over(order by rand()) sorted7,
  row_number() over(order by rand()) sorted8,
  row_number() over(order by rand()) sorted9,
  row_number() over(order by rand()) sorted10
  from s3('s3:///hammy/lagrangian/1/1/raw/*.gzip.parquet', Parquet)  --file('hammy/lagrangian_*.snappy.parquet', Parquet) 
  where target_position = 60),

counts as (select *,
  countIf(checkpoint10_position = 10) over(order by sorted1) position1_exact, 
  countIf(checkpoint10_position = 8) over(order by sorted1) position1_left,
  countIf(checkpoint10_position = 12) over(order by sorted1) position1_right,
  countIf(checkpoint10_position = 10) over(order by sorted1) position2_exact, 
  countIf(checkpoint10_position = 8) over(order by sorted1) position2_left,
  countIf(checkpoint10_position = 12) over(order by sorted1) position2_right,
  countIf(checkpoint10_position = 10) over(order by sorted1) position3_exact, 
  countIf(checkpoint10_position = 8) over(order by sorted1) position3_left,
  countIf(checkpoint10_position = 12) over(order by sorted1) position3_right,
  countIf(checkpoint10_position = 10) over(order by sorted1) position4_exact, 
  countIf(checkpoint10_position = 8) over(order by sorted1) position4_left,
  countIf(checkpoint10_position = 12) over(order by sorted1) position4_right,
  countIf(checkpoint10_position = 10) over(order by sorted1) position5_exact, 
  countIf(checkpoint10_position = 8) over(order by sorted1) position5_left,
  countIf(checkpoint10_position = 12) over(order by sorted1) position5_right,
  countIf(checkpoint10_position = 10) over(order by sorted1) position6_exact, 
  countIf(checkpoint10_position = 8) over(order by sorted1) position6_left,
  countIf(checkpoint10_position = 12) over(order by sorted1) position6_right,
  countIf(checkpoint10_position = 10) over(order by sorted1) position7_exact, 
  countIf(checkpoint10_position = 8) over(order by sorted1) position7_left,
  countIf(checkpoint10_position = 12) over(order by sorted1) position7_right,
  countIf(checkpoint10_position = 10) over(order by sorted1) position8_exact, 
  countIf(checkpoint10_position = 8) over(order by sorted1) position8_left,
  countIf(checkpoint10_position = 12) over(order by sorted1) position8_right,
  countIf(checkpoint10_position = 10) over(order by sorted1) position9_exact, 
  countIf(checkpoint10_position = 8) over(order by sorted1) position9_left,
  countIf(checkpoint10_position = 12) over(order by sorted1) position9_right,
  countIf(checkpoint10_position = 10) over(order by sorted1) position10_exact, 
  countIf(checkpoint10_position = 8) over(order by sorted1) position10_left,
  countIf(checkpoint10_position = 12) over(order by sorted1) position10_right
  from sorted),
is_top_exact as (select
  sorted1,
  position1_exact > position1_left and position1_exact > position1_right correct1,
  sorted2,
  position2_exact > position2_left and position2_exact > position2_right correct2,
  sorted3,
  position3_exact > position3_left and position3_exact > position3_right correct3,
  sorted4,
  position4_exact > position4_left and position4_exact > position4_right correct4,
  sorted5,
  position5_exact > position5_left and position5_exact > position5_right correct5,
  sorted6,
  position6_exact > position6_left and position6_exact > position6_right correct6,
  sorted7,
  position7_exact > position7_left and position7_exact > position7_right correct7,
  sorted8,
  position8_exact > position8_left and position8_exact > position8_right correct8,
  sorted9,
  position9_exact > position9_left and position9_exact > position9_right correct9,
  sorted10,
  position10_exact > position10_left and position10_exact > position10_right correct10
  from counts),
fix_changes as (select
  sorted1,
  correct1 - lagInFrame(correct1) over (order by sorted1) as change1,
  sorted2,
  correct2 - lagInFrame(correct2) over (order by sorted2) as change2,
  sorted3,
  correct3 - lagInFrame(correct3) over (order by sorted3) as change3,
  sorted4,
  correct4 - lagInFrame(correct4) over (order by sorted4) as change4,
  sorted5,
  correct5 - lagInFrame(correct5) over (order by sorted5) as change5,
  sorted6,
  correct6 - lagInFrame(correct6) over (order by sorted6) as change6,
  sorted7,
  correct7 - lagInFrame(correct7) over (order by sorted7) as change7,
  sorted8,
  correct8 - lagInFrame(correct8) over (order by sorted8) as change8,
  sorted9,
  correct9 - lagInFrame(correct9) over (order by sorted9) as change9,
  sorted10,
  correct10 - lagInFrame(correct10) over (order by sorted10) as change10
  from is_top_exact),
leave_changes as (select *
  from fix_changes
  where
    change1 or change2 or change3 or change4 or change5 or change6 or change7 or change8 or change9 or change10),
union_changes as (select sorted1 sorted, change1 change
    from leave_changes where change1 != 0 union all select sorted2 sorted, change2 change
    from leave_changes where change2 != 0 union all select sorted3 sorted, change3 change
    from leave_changes where change3 != 0 union all select sorted4 sorted, change4 change
    from leave_changes where change4 != 0 union all select sorted5 sorted, change5 change
    from leave_changes where change5 != 0 union all select sorted6 sorted, change6 change
    from leave_changes where change6 != 0 union all select sorted7 sorted, change7 change
    from leave_changes where change7 != 0 union all select sorted8 sorted, change8 change
    from leave_changes where change8 != 0 union all select sorted9 sorted, change9 change
    from leave_changes where change9 != 0 union all select sorted10 sorted, change10 change
    from leave_changes where change10 != 0),
accumulate_changes as (select sorted, 
  cast(sum(change) over (order by sorted) as Float32) / 10 as exactness_ratio
  from union_changes)
select * from accumulate_changes
settings s3_create_new_file_on_insert=1