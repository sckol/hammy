select avg(checkpoint10_position) avg10, stddevSamp(checkpoint10_position) sd10, avg(checkpoint20_position) avg20, stddevSamp(checkpoint20_position) sd20, avg(checkpoint30_position) avg30, stddevSamp(checkpoint30_position) sd30, avg(checkpoint40_position) avg40, stddevSamp(checkpoint40_position) sd40, avg(checkpoint50_position) avg50, stddevSamp(checkpoint50_position) sd50
from `hammy`.`lagrangian/1/1/*.snappy.parquet`
with
   (
      format=parquet, 
      schema=(
         epoch Int32, target_position Int32, checkpoint10_position Int32, checkpoint11_position Int32, checkpoint12_position Int32, checkpoint13_position Int32, checkpoint14_position Int32, checkpoint15_position Int32, checkpoint20_position Int32, checkpoint21_position Int32, checkpoint22_position Int32, checkpoint23_position Int32, checkpoint24_position Int32, checkpoint25_position Int32, checkpoint30_position Int32, checkpoint31_position Int32, checkpoint32_position Int32, checkpoint33_position Int32, checkpoint34_position Int32, checkpoint35_position Int32, checkpoint40_position Int32, checkpoint41_position Int32, checkpoint42_position Int32, checkpoint43_position Int32, checkpoint44_position Int32, checkpoint45_position Int32, checkpoint50_position Int32, checkpoint51_position Int32, checkpoint52_position Int32, checkpoint53_position Int32, checkpoint54_position Int32, checkpoint55_position Int32
      )
   )
where target_position = 60
