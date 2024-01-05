$sorted = (select checkpoint10_position,
  row_number() over(order by Random(epoch)) sorted1
  from `hammy`.`/lagrangian/1/1/raw/*.gzip.parquet`
  with
   (
      format=parquet, 
      schema=(
         epoch Int32, target_position Int32, checkpoint10_position Int32, checkpoint11_position Int32, checkpoint12_position Int32, checkpoint13_position Int32, checkpoint14_position Int32, checkpoint15_position Int32, checkpoint20_position Int32, checkpoint21_position Int32, checkpoint22_position Int32, checkpoint23_position Int32, checkpoint24_position Int32, checkpoint25_position Int32, checkpoint30_position Int32, checkpoint31_position Int32, checkpoint32_position Int32, checkpoint33_position Int32, checkpoint34_position Int32, checkpoint35_position Int32, checkpoint40_position Int32, checkpoint41_position Int32, checkpoint42_position Int32, checkpoint43_position Int32, checkpoint44_position Int32, checkpoint45_position Int32, checkpoint50_position Int32, checkpoint51_position Int32, checkpoint52_position Int32, checkpoint53_position Int32, checkpoint54_position Int32, checkpoint55_position Int32
      )
   )
  where target_position = 60
  );

$counts = (select x.*,
  countIf(checkpoint10_position = 10) over(order by sorted1) position1_exact, 
  countIf(checkpoint10_position = 8) over(order by sorted1) position1_left,
  countIf(checkpoint10_position = 12) over(order by sorted1) position1_right
  from $sorted x);
$is_top_exact = (select
  sorted1,
  cast(position1_exact > position1_left and position1_exact > position1_right AS Int32) correct1
  from $counts);
$fix_changes = (select
  sorted1,
  correct1 - coalesce(lag(correct1), 0) over (order by sorted1) change1
  from $is_top_exact);
$leave_changes = (select *
  from $fix_changes
  where
    change1 !=0 );
$union_changes = (select sorted1 sorted, change1 change
    from $leave_changes where change1 != 0);
$accumulate_changes = (select sorted, 
  cast(sum(change) over (order by sorted) as Float) / 1 as exactness_ratio 
  from $union_changes);
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/0/` 
  with (format='csv_with_names')
  select 0 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/1/` 
  with (format='csv_with_names')
  select 1 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/2/` 
  with (format='csv_with_names')
  select 2 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/3/` 
  with (format='csv_with_names')
  select 3 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/4/` 
  with (format='csv_with_names')
  select 4 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/5/` 
  with (format='csv_with_names')
  select 5 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/6/` 
  with (format='csv_with_names')
  select 6 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/7/` 
  with (format='csv_with_names')
  select 7 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/8/` 
  with (format='csv_with_names')
  select 8 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/9/` 
  with (format='csv_with_names')
  select 9 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/10/` 
  with (format='csv_with_names')
  select 10 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/11/` 
  with (format='csv_with_names')
  select 11 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/12/` 
  with (format='csv_with_names')
  select 12 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/13/` 
  with (format='csv_with_names')
  select 13 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/14/` 
  with (format='csv_with_names')
  select 14 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/15/` 
  with (format='csv_with_names')
  select 15 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/16/` 
  with (format='csv_with_names')
  select 16 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/17/` 
  with (format='csv_with_names')
  select 17 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/18/` 
  with (format='csv_with_names')
  select 18 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/19/` 
  with (format='csv_with_names')
  select 19 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/20/` 
  with (format='csv_with_names')
  select 20 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/21/` 
  with (format='csv_with_names')
  select 21 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/22/` 
  with (format='csv_with_names')
  select 22 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/23/` 
  with (format='csv_with_names')
  select 23 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/24/` 
  with (format='csv_with_names')
  select 24 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/25/` 
  with (format='csv_with_names')
  select 25 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/26/` 
  with (format='csv_with_names')
  select 26 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/27/` 
  with (format='csv_with_names')
  select 27 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/28/` 
  with (format='csv_with_names')
  select 28 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/29/` 
  with (format='csv_with_names')
  select 29 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/30/` 
  with (format='csv_with_names')
  select 30 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/31/` 
  with (format='csv_with_names')
  select 31 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/32/` 
  with (format='csv_with_names')
  select 32 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/33/` 
  with (format='csv_with_names')
  select 33 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/34/` 
  with (format='csv_with_names')
  select 34 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/35/` 
  with (format='csv_with_names')
  select 35 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/36/` 
  with (format='csv_with_names')
  select 36 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/37/` 
  with (format='csv_with_names')
  select 37 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/38/` 
  with (format='csv_with_names')
  select 38 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/39/` 
  with (format='csv_with_names')
  select 39 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/40/` 
  with (format='csv_with_names')
  select 40 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/41/` 
  with (format='csv_with_names')
  select 41 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/42/` 
  with (format='csv_with_names')
  select 42 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/43/` 
  with (format='csv_with_names')
  select 43 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/44/` 
  with (format='csv_with_names')
  select 44 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/45/` 
  with (format='csv_with_names')
  select 45 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/46/` 
  with (format='csv_with_names')
  select 46 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/47/` 
  with (format='csv_with_names')
  select 47 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/48/` 
  with (format='csv_with_names')
  select 48 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/49/` 
  with (format='csv_with_names')
  select 49 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/50/` 
  with (format='csv_with_names')
  select 50 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/51/` 
  with (format='csv_with_names')
  select 51 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/52/` 
  with (format='csv_with_names')
  select 52 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/53/` 
  with (format='csv_with_names')
  select 53 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/54/` 
  with (format='csv_with_names')
  select 54 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/55/` 
  with (format='csv_with_names')
  select 55 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/56/` 
  with (format='csv_with_names')
  select 56 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/57/` 
  with (format='csv_with_names')
  select 57 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/58/` 
  with (format='csv_with_names')
  select 58 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/59/` 
  with (format='csv_with_names')
  select 59 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/60/` 
  with (format='csv_with_names')
  select 60 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/61/` 
  with (format='csv_with_names')
  select 61 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/62/` 
  with (format='csv_with_names')
  select 62 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/63/` 
  with (format='csv_with_names')
  select 63 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/64/` 
  with (format='csv_with_names')
  select 64 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/65/` 
  with (format='csv_with_names')
  select 65 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/66/` 
  with (format='csv_with_names')
  select 66 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/67/` 
  with (format='csv_with_names')
  select 67 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/68/` 
  with (format='csv_with_names')
  select 68 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/69/` 
  with (format='csv_with_names')
  select 69 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/70/` 
  with (format='csv_with_names')
  select 70 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/71/` 
  with (format='csv_with_names')
  select 71 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/72/` 
  with (format='csv_with_names')
  select 72 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/73/` 
  with (format='csv_with_names')
  select 73 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/74/` 
  with (format='csv_with_names')
  select 74 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/75/` 
  with (format='csv_with_names')
  select 75 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/76/` 
  with (format='csv_with_names')
  select 76 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/77/` 
  with (format='csv_with_names')
  select 77 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/78/` 
  with (format='csv_with_names')
  select 78 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/79/` 
  with (format='csv_with_names')
  select 79 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/80/` 
  with (format='csv_with_names')
  select 80 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/81/` 
  with (format='csv_with_names')
  select 81 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/82/` 
  with (format='csv_with_names')
  select 82 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/83/` 
  with (format='csv_with_names')
  select 83 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/84/` 
  with (format='csv_with_names')
  select 84 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/85/` 
  with (format='csv_with_names')
  select 85 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/86/` 
  with (format='csv_with_names')
  select 86 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/87/` 
  with (format='csv_with_names')
  select 87 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/88/` 
  with (format='csv_with_names')
  select 88 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/89/` 
  with (format='csv_with_names')
  select 89 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/90/` 
  with (format='csv_with_names')
  select 90 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/91/` 
  with (format='csv_with_names')
  select 91 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/92/` 
  with (format='csv_with_names')
  select 92 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/93/` 
  with (format='csv_with_names')
  select 93 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/94/` 
  with (format='csv_with_names')
  select 94 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/95/` 
  with (format='csv_with_names')
  select 95 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/96/` 
  with (format='csv_with_names')
  select 96 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/97/` 
  with (format='csv_with_names')
  select 97 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/98/` 
  with (format='csv_with_names')
  select 98 as m, x.* from $accumulate_changes as x order by sorted;
insert into `hammy`.`/lagrangian/1/1/proc/exactness-curve/99/` 
  with (format='csv_with_names')
  select 99 as m, x.* from $accumulate_changes as x order by sorted;